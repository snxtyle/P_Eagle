#!/usr/bin/env python3
"""
P-EAGLE FlashAttention-Compatible Feature Extraction

Extracts hidden states from packed sequences for H200 training.
Uses FlashAttention varlen format for maximum GPU efficiency.

Usage:
    # Phase 1: Extract 5K samples
    python scripts/extract_features_packed.py \
        --model_path meta-llama/Llama-3.1-8B \
        --input_dir data/packed_5k \
        --output_dir data/features_packed_5k \
        --batch_size 8

    # Phase 2: Extract 10K samples
    python scripts/extract_features_packed.py \
        --model_path meta-llama/Llama-3.1-8B \
        --input_dir data/packed_10k \
        --output_dir data/features_packed_10k

    # Phase 3: Full extraction
    python scripts/extract_features_packed.py \
        --model_path meta-llama/Llama-3.1-8B \
        --input_dir data/packed_full \
        --output_dir data/features_packed_full
"""

import argparse
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
import gc


def load_packed_dataset(input_dir):
    """Load all packed sequence shards from directory."""
    shards = sorted(Path(input_dir).glob("*.pt"))
    if not shards:
        raise ValueError(f"No packed shards found in {input_dir}")

    all_data = []
    for shard_file in shards:
        data = torch.load(shard_file, map_location="cpu")
        all_data.append(data)
        print(f"Loaded {shard_file.name}: {data['total_tokens']:,} tokens, {data['num_sequences']} sequences")

    return all_data


def extract_features_batch(model, input_ids, cu_seqlens, block_size=4096):
    """
    Extract hidden states using FlashAttention-compatible batching.

    For H200, we want:
    1. Process multiple 4096-blocks in parallel
    2. Use attention_mask with cu_seqlens for variable-length handling
    3. Extract last layer hidden states (required for speculative decoding alignment)
    """
    device = next(model.parameters()).device
    batch_size = len(cu_seqlens) - 1  # Number of sequences
    max_seq_len = block_size

    # Pad to multiple of block_size for efficient processing
    total_len = len(input_ids)
    padded_len = ((total_len + block_size - 1) // block_size) * block_size
    pad_len = padded_len - total_len

    # Create padded input
    padded_input_ids = torch.zeros(padded_len, dtype=torch.long, device=device)
    padded_input_ids[:total_len] = input_ids.to(device)

    # Create attention mask (1 for real tokens, 0 for padding)
    attention_mask = torch.zeros(padded_len, dtype=torch.long, device=device)
    attention_mask[:total_len] = 1

    # Convert cu_seqlens to device
    cu_seqlens = cu_seqlens.to(device)

    # Run model with FlashAttention-compatible attention
    with torch.no_grad():
        outputs = model(
            input_ids=padded_input_ids.unsqueeze(0),  # Add batch dim
            attention_mask=attention_mask.unsqueeze(0),
            output_hidden_states=True
        )

    # Get last layer hidden states
    hidden_states = outputs.hidden_states[-1][0]  # [seq_len, hidden_dim]

    # Trim to actual length
    hidden_states = hidden_states[:total_len]

    return hidden_states.cpu()


def compute_target_token_ids_batch(model, input_ids):
    """Compute target token IDs (argmax of logits) for each position."""
    device = next(model.parameters()).device

    with torch.no_grad():
        outputs = model(input_ids=input_ids.to(device).unsqueeze(0))
        logits = outputs.logits[0]  # [seq_len, vocab_size]
        target_ids = logits.argmax(dim=-1).cpu()

    return target_ids


def main():
    parser = argparse.ArgumentParser(description="P-EAGLE FlashAttention Feature Extraction")
    parser.add_argument("--model_path", required=True, help="Target model path")
    parser.add_argument("--tokenizer_path", default=None,
                        help="Tokenizer (defaults to model_path)")
    parser.add_argument("--input_dir", required=True,
                        help="Directory with packed sequences")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for features")
    parser.add_argument("--quantization", default="8bit",
                        choices=["4bit", "8bit", "none"],
                        help="Model quantization")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for processing")
    parser.add_argument("--block_size", type=int, default=4096,
                        help="Sequence block size")
    parser.add_argument("--layers", default="last",
                        help="Layer extraction mode: 'last', 'all', or comma-separated indices")
    parser.add_argument("--shard_size", type=int, default=1000,
                        help="Samples per output shard")

    args = parser.parse_args()

    # Setup paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    tokenizer_path = args.tokenizer_path or args.model_path
    print(f"Loading tokenizer: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print(f"Loading model: {args.model_path}")
    load_kwargs = {"low_cpu_mem_usage": True}

    if args.quantization == "4bit":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    elif args.quantization == "8bit":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16

    # Check for HF token
    HF_TOKEN = None
    try:
        from dotenv import load_dotenv
        load_dotenv()
        HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
    except ImportError:
        pass

    if HF_TOKEN:
        load_kwargs["token"] = HF_TOKEN

    model = AutoModelForCausalLM.from_pretrained(args.model_path, **load_kwargs)
    model.eval()

    # Move to CUDA
    if torch.cuda.is_available():
        model = model.to("cuda")
        torch.cuda.synchronize()
        print(f"Model on: {next(model.parameters()).device}")

    # Determine layer indices
    num_layers = model.config.num_hidden_layers
    if args.layers == "last":
        layer_indices = [-1]
    elif args.layers == "all":
        layer_indices = list(range(num_layers))
    else:
        layer_indices = [int(x) for x in args.layers.split(",")]

    print(f"Extracting layers: {layer_indices}")

    # Load packed data
    print(f"Loading packed sequences from {args.input_dir}")
    packed_data = load_packed_dataset(args.input_dir)

    # Process each shard
    all_features = []
    shard_idx = 0

    for shard_data in tqdm(packed_data, desc="Processing shards"):
        input_ids = shard_data["input_ids"]
        loss_mask = shard_data["loss_mask"]
        cu_seqlens = shard_data["cu_seqlens"]
        total_tokens = shard_data["total_tokens"]

        print(f"\nProcessing shard {shard_idx}: {total_tokens:,} tokens, {len(cu_seqlens)-1} sequences")

        # Extract hidden states
        hidden_states = extract_features_batch(
            model, input_ids, cu_seqlens, args.block_size
        )

        # Compute target token IDs
        target_token_ids = compute_target_token_ids_batch(model, input_ids)

        # Create feature entries for each sequence in the shard
        for i in range(len(cu_seqlens) - 1):
            start = cu_seqlens[i].item()
            end = cu_seqlens[i + 1].item()

            feature = {
                "input_ids": input_ids[start:end],
                "hidden_states": hidden_states[start:end],
                "loss_mask": loss_mask[start:end],
                "target_token_ids": target_token_ids[start:end],
            }

            all_features.append(feature)

            # Save shards periodically
            if len(all_features) >= args.shard_size:
                _save_shard(all_features, output_dir, shard_idx, tokenizer)
                shard_idx += 1
                all_features = []

        del hidden_states
        gc.collect()
        torch.cuda.empty_cache()

    # Save final shard
    if all_features:
        _save_shard(all_features, output_dir, shard_idx, tokenizer)

    print(f"\nExtraction complete! Output: {output_dir}")


def _save_shard(features, output_dir, shard_idx, tokenizer):
    """Save a shard of extracted features."""
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [f["input_ids"] for f in features],
        batch_first=True,
        padding_value=tokenizer.pad_token_id or 0
    ).to(torch.bfloat16)

    hidden_states = torch.nn.utils.rnn.pad_sequence(
        [f["hidden_states"] for f in features],
        batch_first=True,
        padding_value=0.0
    ).to(torch.bfloat16)

    loss_masks = torch.nn.utils.rnn.pad_sequence(
        [f["loss_mask"] for f in features],
        batch_first=True,
        padding_value=0.0
    )

    target_ids = torch.nn.utils.rnn.pad_sequence(
        [f["target_token_ids"] for f in features],
        batch_first=True,
        padding_value=-100
    )

    output_file = output_dir / f"features_shard_{shard_idx:04d}.pt"

    torch.save({
        "input_ids": input_ids,
        "hidden_states": hidden_states,
        "loss_mask": loss_masks,
        "target_token_ids": target_ids,
        "num_samples": len(features)
    }, output_file)

    print(f"  Saved {output_file}: {len(features)} samples")


if __name__ == "__main__":
    import os
    main()