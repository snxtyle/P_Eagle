#!/usr/bin/env python3
"""
P-EAGLE Hidden State Extraction

Loads pre-tokenized sequences from packed_features_5k/ and extracts
hidden states from Gemma-3.

IMPORTANT: This script processes data that was already tokenized with
apply_chat_template() in fast_sequence_packing.py. Do NOT re-apply
chat template - the data is already formatted correctly!

Usage:
    python scripts/extract_features_packed.py \
        --input data/packed_features_5k \
        --output data/features_packed_5k \
        --model google/gemma-3-4b-it \
        --shard_size 500
"""

import argparse
import json
import os
import torch
import gc
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
from dataclasses import dataclass

import torch.nn.functional as F


def load_packed_dataset(input_dir: str) -> List[Dict]:
    """Load all packed sequence shards."""
    shards = sorted(Path(input_dir).glob("packed_shard_*.pt"))
    if not shards:
        raise ValueError(f"No packed shards found in {input_dir}")

    all_data = []
    for shard_file in shards:
        data = torch.load(shard_file, map_location="cpu")
        all_data.append(data)
        print(f"Loaded {shard_file.name}: {data['total_tokens']:,} tokens, {data['num_sequences']} sequences")

    return all_data


def extract_hidden_states_single(
    model,
    input_ids: torch.Tensor,
    device: str = 'cuda'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract hidden states for a single sequence.
    Memory-efficient: disables KV cache and clears GPU memory.

    input_ids: [seq_len] token IDs
    Returns: (hidden_states [seq_len, hidden_dim], target_ids [seq_len])
    """
    input_ids = input_ids.unsqueeze(0).to(device)  # [1, seq_len]
    seq_len = input_ids.shape[1]

    # Disable KV cache to prevent memory accumulation
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        outputs = model(
            input_ids=input_ids,
            output_hidden_states=True,
            use_cache=False  # Disable KV cache to save memory
        )

    # Get last hidden states [1, seq_len, hidden_dim] -> [seq_len, hidden_dim]
    hidden_states = outputs.hidden_states[-1][0].float().cpu()

    # Create target tokens (next token prediction) - shift by 1
    target_ids = input_ids[0].cpu().roll(-1)
    target_ids[-1] = 0  # Padding at end

    # Clear GPU cache after each sequence to prevent OOM
    torch.cuda.empty_cache()

    return hidden_states, target_ids


def create_loss_mask(input_ids: torch.Tensor, tokenizer) -> torch.Tensor:
    """
    Create loss mask - marks where to calculate loss.

    Masks: BOS, EOS, PAD, control tokens
    """
    loss_mask = torch.ones_like(input_ids, dtype=torch.float32)

    # Special tokens to mask
    mask_ids = set()
    mask_ids.add(tokenizer.bos_token_id)
    mask_ids.add(tokenizer.eos_token_id)
    mask_ids.add(tokenizer.pad_token_id)
    if tokenizer.unk_token_id:
        mask_ids.add(tokenizer.unk_token_id)

    # Gemma-3 control tokens
    control_tokens = ['<start_of_turn>', '<end_of_turn>']
    for tok in control_tokens:
        ids = tokenizer.encode(tok, add_special_tokens=False)
        mask_ids.update(ids)

    for i, tid in enumerate(input_ids):
        if tid.item() in mask_ids:
            loss_mask[i] = 0.0

    return loss_mask


def process_packed_sequences(
    model,
    tokenizer,
    packed_data: List[Dict],
    output_dir: Path,
    shard_size: int = 500,
    max_seq_len: int = 4096
):
    """
    Process packed sequences and extract hidden states.

    Each packed sequence has:
    - input_ids: [max_seq_len] token IDs (padded)
    - loss_mask: [max_seq_len] 0/1 mask
    - cu_seqlens: cumulative sequence lengths within the packed block
    """
    all_features = []
    total_samples = 0
    shard_idx = 0

    for shard_data in tqdm(packed_data, desc="Processing shards"):
        input_ids = shard_data["input_ids"]  # [N, max_seq_len]
        loss_mask = shard_data["loss_mask"]   # [N, max_seq_len]
        cu_seqlens_list = shard_data["cu_seqlens"]  # list of tensors

        N = input_ids.shape[0]

        for seq_idx in tqdm(range(N), desc=f"Shard {shard_idx}"):
            ids = input_ids[seq_idx].long()
            mask = loss_mask[seq_idx]

            # Skip all-padding sequences
            if (ids == 0).all():
                continue

            # Extract hidden states
            try:
                hidden, targets = extract_hidden_states_single(model, ids)

                # Use the loss_mask from packed data (already correct!)
                # No need to re-create it

                # Get cu_seqlens for this sequence (conversation boundaries)
                seq_cu_seqlens = cu_seqlens_list[seq_idx]

                all_features.append({
                    "input_ids": ids,
                    "hidden_states": hidden,
                    "target_token_ids": targets,
                    "loss_mask": mask,
                    "cu_seqlens": seq_cu_seqlens,  # Preserve conversation tracking
                })

                total_samples += 1

                # Save shard more frequently to prevent data loss
                if len(all_features) >= shard_size:
                    save_shard(all_features, output_dir / f"features_shard_{shard_idx:04d}.pt")
                    print(f"  Saved shard {shard_idx}: {len(all_features)} samples")
                    all_features = []
                    shard_idx += 1
                    # Clear GPU cache after saving each shard
                    torch.cuda.empty_cache()

                # Check memory usage every 100 sequences
                if total_samples % 100 == 0:
                    mem_allocated = torch.cuda.memory_allocated() / 1e9
                    mem_reserved = torch.cuda.memory_reserved() / 1e9
                    print(f"  Progress: {total_samples} samples, GPU memory: {mem_allocated:.1f}GB / {mem_reserved:.1f}GB")

            except Exception as e:
                print(f"Error processing sequence {seq_idx}: {e}")
                # Try to recover by clearing cache
                torch.cuda.empty_cache()
                continue

        del input_ids, loss_mask
        gc.collect()

    # Save remaining
    if all_features:
        save_shard(all_features, output_dir / f"features_shard_{shard_idx:04d}.pt")
        print(f"  Saved final shard {shard_idx}: {len(all_features)} samples")

    return total_samples


def save_shard(features: List[Dict], path: Path):
    """Save a shard of extracted features."""
    # input_ids should be int64 (token IDs), NOT bfloat16!
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [f["input_ids"] for f in features],
        batch_first=True,
        padding_value=0
    ).to(torch.int64)  # FIX: Keep as int64!

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

    # Save cu_seqlens for each sequence
    cu_seqlens_list = [f["cu_seqlens"] for f in features]

    torch.save({
        "input_ids": input_ids,
        "hidden_states": hidden_states,
        "loss_mask": loss_masks,
        "target_token_ids": target_ids,
        "cu_seqlens": cu_seqlens_list,  # Preserve conversation tracking
        "num_samples": len(features)
    }, path)


def main():
    parser = argparse.ArgumentParser(description='P-EAGLE Hidden State Extraction')
    parser.add_argument('--input', type=str, required=True,
                        help='Input directory with packed sequences (packed_features_5k/)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for features')
    parser.add_argument('--model', type=str, default='google/gemma-3-4b-it',
                        help='Gemma-3 model path/name')
    parser.add_argument('--shard_size', type=int, default=500,
                        help='Samples per output shard')
    parser.add_argument('--max_seq_len', type=int, default=4096,
                        help='Max sequence length')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for inference (currently only 1 supported)')

    args = parser.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM

    print("=" * 60)
    print("P-EAGLE Hidden State Extraction")
    print("Loading from pre-tokenized packed sequences")
    print("=" * 60)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizer (for loss mask creation if needed)
    print(f"\n[1/4] Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print(f"  Vocab size: {tokenizer.vocab_size}")

    # Load model
    print(f"\n[2/4] Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map='cuda',
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()
    print(f"  Model loaded on GPU")

    # Load packed data
    print(f"\n[3/4] Loading packed sequences: {args.input}")
    packed_data = load_packed_dataset(args.input)

    # Process
    print(f"\n[4/4] Extracting hidden states...")
    total_samples = process_packed_sequences(
        model, tokenizer, packed_data, output_dir,
        shard_size=args.shard_size, max_seq_len=args.max_seq_len
    )

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\n{'=' * 60}")
    print("COMPLETE!")
    print(f"{'=' * 60}")
    print(f"Total samples processed: {total_samples}")
    print(f"Output directory: {args.output}")

    # Verify output
    if output_dir.exists():
        shards = list(output_dir.glob("features_shard_*.pt"))
        print(f"Generated {len(shards)} shard files")


if __name__ == '__main__':
    main()