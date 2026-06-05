#!/usr/bin/env python3
"""
FAST P-EAGLE Feature Extraction

Simple, fast extraction without packing or complex processing:
1. Tokenize each conversation
2. Pad to fixed max_length
3. Single forward pass
4. Save hidden states, target_token_ids, loss_mask, lm_head

This is much faster than the full extraction with sliding windows.
"""

import argparse
import json
import torch
from pathlib import Path
from tqdm import tqdm
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer


def find_lm_head(model):
    """Find lm_head at any valid location."""
    locations = [
        ("model.lm_head", lambda m: m.lm_head if hasattr(m, 'lm_head') and m.lm_head is not None else None),
        ("model.model.lm_head", lambda m: m.model.lm_head if hasattr(m, 'model') and hasattr(m.model, 'lm_head') and m.model.lm_head is not None else None),
        ("model.model.model.lm_head", lambda m: m.model.model.lm_head if hasattr(m.model, 'model') and hasattr(m.model.model, 'lm_head') and m.model.model.lm_head is not None else None),
        ("model.language_model.lm_head", lambda m: m.language_model.lm_head if hasattr(m, 'language_model') and hasattr(m.language_model, 'lm_head') and m.language_model.lm_head is not None else None),
        # Gemma-3 multimodal structure
        ("model.model.language_model.lm_head", lambda m: m.model.language_model.lm_head if hasattr(m, 'model') and hasattr(m.model, 'language_model') and hasattr(m.model.language_model, 'lm_head') and m.model.language_model.lm_head is not None else None),
    ]

    for name, check_func in locations:
        lm = check_func(model)
        if lm is not None:
            print(f"  Found lm_head at: {name}")
            return lm
    return None


def find_norm_layer(model):
    """Find the final norm layer."""
    locations = [
        ("model.model.language_model.norm", lambda m: m.model.language_model.norm if hasattr(m, 'model') and hasattr(m.model, 'language_model') and hasattr(m.model.language_model, 'norm') else None),
        ("model.model.model.norm", lambda m: m.model.model.norm if hasattr(m.model, 'model') and hasattr(m.model.model, 'norm') else None),
        ("model.model.norm", lambda m: m.model.norm if hasattr(m.model, 'norm') else None),
        ("model.norm", lambda m: m.norm if hasattr(m, 'norm') else None),
    ]

    for name, check_func in locations:
        norm = check_func(model)
        if norm is not None:
            return norm
    return None


def conversation_to_text(messages):
    """Convert OpenAI messages to text."""
    parts = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        # Handle tool_calls
        if not content and msg.get("tool_calls"):
            tool_parts = []
            for tc in msg.get("tool_calls", []):
                func = tc.get("function", {})
                tool_parts.append(f"{func.get('name', '')}({func.get('arguments', '')})")
            content = "[TOOL_CALLS] " + "; ".join(tool_parts)

        if content:
            parts.append(f"{role}: {content}")

    return "\n\n".join(parts)


def extract_features_for_sample(model, tokenizer, messages, max_length, device):
    """Extract features for a single conversation."""
    # Convert to text
    text = conversation_to_text(messages)

    # Tokenize
    encoding = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        # Get hidden states from last layer
        hidden_states = outputs.hidden_states[-1][0]  # [seq_len, hidden_dim]

        # Apply final norm
        norm_layer = find_norm_layer(model)
        if norm_layer is not None:
            hidden_states = norm_layer(hidden_states)

        # Get target token IDs (argmax of logits)
        target_token_ids = outputs.logits.argmax(dim=-1)[0]  # [seq_len]

        # Create loss mask (1 for assistant tokens, 0 for system/user)
        # We'll mark padding as 0, and assume assistant content is trainable
        loss_mask = attention_mask[0].clone().float()

        # Mark padding as 0
        loss_mask[input_ids[0] == tokenizer.pad_token_id] = 0

    return {
        "input_ids": input_ids[0].cpu(),
        "hidden_states": hidden_states.cpu(),
        "target_token_ids": target_token_ids.cpu(),
        "loss_mask": loss_mask.cpu(),
        "attention_mask": attention_mask[0].cpu(),
    }


def main():
    parser = argparse.ArgumentParser(description="Fast P-EAGLE Feature Extraction")
    parser.add_argument("--model_path", default="google/gemma-3-4b-it", help="Target model path")
    parser.add_argument("--input_data", required=True, help="Input JSONL file")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--max_length", type=int, default=4096, help="Max sequence length")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (currently only 1)")
    parser.add_argument("--shard_size", type=int, default=500, help="Samples per shard file")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to process")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Find and print lm_head info
    print("Checking lm_head locations...")
    lm_head = find_lm_head(model)
    if lm_head is not None:
        try:
            weight_shape = lm_head.weight.shape
            print(f"  lm_head shape: {weight_shape}")
            print(f"  lm_head will be saved to feature files")
        except AttributeError:
            print(f"  lm_head found at {type(lm_head).__name__} (no separate weight, likely tied embeddings)")
            print(f"  Skipping lm_head save (not needed for feature extraction)")
    else:
        print("  WARNING: Could not find lm_head!")

    # Find norm layer info
    norm_layer = find_norm_layer(model)
    if norm_layer is not None:
        print("  Found final norm layer")

    # Load input data
    print(f"\nLoading input data: {args.input_data}")
    samples = []
    with open(args.input_data, 'r') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    if args.max_samples:
        samples = samples[:args.max_samples]

    print(f"Loaded {len(samples)} samples")
    print(f"Processing with max_length={args.max_length}")

    # Process in shards
    all_features = []
    shard_idx = 0
    sample_idx = 0

    pbar = tqdm(samples, desc="Extracting features")

    for i, sample in enumerate(pbar):
        messages = sample.get("messages", [])

        if not messages:
            continue

        try:
            features = extract_features_for_sample(model, tokenizer, messages, args.max_length, model.device if hasattr(model, 'device') else device)
            all_features.append(features)

            pbar.set_postfix({"shard": shard_idx, "in_shard": len(all_features)})

            # Save shard when full
            if len(all_features) >= args.shard_size:
                save_shard(output_dir, all_features, shard_idx, lm_head, model)
                print(f"\n  Saved shard {shard_idx} with {len(all_features)} samples")
                all_features = []
                shard_idx += 1

        except Exception as e:
            print(f"\n  Error processing sample {i}: {e}")
            continue

    # Save remaining samples
    if all_features:
        save_shard(output_dir, all_features, shard_idx, lm_head, model)
        print(f"\n  Saved final shard {shard_idx} with {len(all_features)} samples")

    print(f"\n✅ Feature extraction complete!")
    print(f"   Output: {output_dir}")
    print(f"   Shards: {shard_idx + 1}")


def save_shard(output_dir, features, shard_idx, lm_head, model):
    """Save a shard of features."""
    # Get lm_head state dict
    lm_head_state = None
    if lm_head is not None:
        lm_head_state = {
            "weight": lm_head.weight.detach().cpu(),
        }
        if hasattr(lm_head, 'bias') and lm_head.bias is not None:
            lm_head_state["bias"] = lm_head.bias.detach().cpu()

    # Stack all features
    num_samples = len(features)

    # Get max sequence length from first feature
    seq_len = features[0]["hidden_states"].shape[0]
    hidden_dim = features[0]["hidden_states"].shape[1]

    # Allocate arrays
    input_ids = torch.full((num_samples, seq_len), 0, dtype=torch.long)
    hidden_states = torch.zeros((num_samples, seq_len, hidden_dim), dtype=torch.bfloat16)
    target_token_ids = torch.zeros((num_samples, seq_len), dtype=torch.long)
    loss_mask = torch.zeros((num_samples, seq_len), dtype=torch.float)
    attention_mask = torch.zeros((num_samples, seq_len), dtype=torch.long)

    for i, feat in enumerate(features):
        input_ids[i] = feat["input_ids"]
        hidden_states[i] = feat["hidden_states"]
        target_token_ids[i] = feat["target_token_ids"]
        loss_mask[i] = feat["loss_mask"]
        attention_mask[i] = feat["attention_mask"]

    save_dict = {
        "input_ids": input_ids,
        "hidden_states": hidden_states,
        "target_token_ids": target_token_ids,
        "loss_mask": loss_mask,
        "attention_mask": attention_mask,
        "num_samples": num_samples,
        "lm_head": lm_head_state,
        "vocab_size": lm_head.weight.shape[0] if lm_head else None,
        "hidden_size": hidden_dim,
    }

    output_file = output_dir / f"features_shard_{shard_idx:04d}.pt"
    torch.save(save_dict, output_file)
    print(f"  Saved: {output_file.name}")


if __name__ == "__main__":
    main()