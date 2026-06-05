#!/usr/bin/env python3
"""
FAST Feature Extraction from Packed Data

Reads already-packed sequences and extracts hidden_states + target_token_ids.
Much faster than extracting from raw JSONL because:
- No tokenization needed (already done)
- No chunking/overlap needed (already packed to fixed length)
- Just forward pass + save

Also saves lm_head (the critical fix for Gemma-3 models).
"""

import argparse
import torch
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoModelForCausalLM


def find_lm_head(model):
    """Find lm_head at any valid location."""
    locations = [
        ("model.lm_head", lambda m: m.lm_head if hasattr(m, 'lm_head') and m.lm_head is not None else None),
        ("model.model.lm_head", lambda m: m.model.lm_head if hasattr(m, 'model') and hasattr(m.model, 'lm_head') and m.model.lm_head is not None else None),
        ("model.model.model.lm_head", lambda m: m.model.model.lm_head if hasattr(m.model, 'model') and hasattr(m.model.model, 'lm_head') and m.model.model.lm_head is not None else None),
        ("model.language_model.lm_head", lambda m: m.language_model.lm_head if hasattr(m, 'language_model') and hasattr(m.language_model, 'lm_head') and m.language_model.lm_head is not None else None),
        ("model.model.language_model.lm_head", lambda m: m.model.language_model.lm_head if hasattr(m, 'model') and hasattr(m.model, 'language_model') and hasattr(m.model.language_model, 'lm_head') and m.model.language_model.lm_head is not None else None),
    ]
    for name, check_func in locations:
        lm = check_func(model)
        if lm is not None:
            print(f"  Found lm_head at: {name}")
            return lm
    return None


def get_lm_head_weight(model, lm_head):
    """Get weight tensor from lm_head, handling different model structures."""
    # Try direct weight access first
    if hasattr(lm_head, 'weight') and lm_head.weight is not None:
        return lm_head.weight

    # Try state_dict approach
    for key, param in model.state_dict().items():
        if 'lm_head.weight' in key:
            print(f"  Found weight via state_dict at: {key}")
            return param

    # Try embedding table in different locations
    for key, param in model.state_dict().items():
        if 'embed_tokens' in key and 'weight' in key:
            print(f"  Found embedding at: {key}")
            return param

    raise ValueError("Could not find lm_head weight!")


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


def main():
    parser = argparse.ArgumentParser(description="Extract features from packed data")
    parser.add_argument("--model_path", default="google/gemma-3-4b-it")
    parser.add_argument("--packed_dir", required=True, help="Directory with packed .pt files")
    parser.add_argument("--output_dir", required=True, help="Output directory for features")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for forward pass")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    print(f"Loading model: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True,
    )
    model.eval()
    print("Model loaded")

    # Find lm_head
    print("Finding lm_head...")
    lm_head = find_lm_head(model)
    if lm_head is None:
        raise ValueError("Could not find lm_head in model!")

    # Get lm_head weight using helper that handles different model structures
    lm_head_weight = get_lm_head_weight(model, lm_head)

    lm_head_state = {
        "weight": lm_head_weight.detach().cpu(),
    }
    if hasattr(lm_head, 'bias') and lm_head.bias is not None:
        lm_head_state["bias"] = lm_head.bias.detach().cpu()

    print(f"  lm_head shape: {lm_head_weight.shape}")
    print(f"  Will save lm_head with features")

    # Find norm layer
    norm_layer = find_norm_layer(model)
    if norm_layer is not None:
        print("  Found final norm layer")

    # Get packed files
    packed_dir = Path(args.packed_dir)
    packed_files = sorted(packed_dir.glob("*.pt"))
    print(f"\nFound {len(packed_files)} packed files")

    total_samples = 0
    for packed_file in packed_files:
        print(f"\nProcessing: {packed_file.name}")

        # Load packed data
        packed = torch.load(packed_file, map_location="cpu", weights_only=False)

        input_ids = packed["input_ids"]  # [num_sequences, seq_len]
        loss_mask = packed["loss_mask"]  # [num_sequences, seq_len]
        attention_mask = packed.get("attention_mask", torch.ones_like(input_ids))

        num_sequences = input_ids.shape[0]
        seq_len = input_ids.shape[1]

        print(f"  Sequences: {num_sequences}, Seq len: {seq_len}")

        # Process in batches
        all_hidden = []
        all_target_ids = []

        batch_size = args.batch_size
        num_batches = (num_sequences + batch_size - 1) // batch_size

        with torch.no_grad():
            for i in tqdm(range(num_batches), desc="Forward pass"):
                start = i * batch_size
                end = min(start + batch_size, num_sequences)

                batch_input_ids = input_ids[start:end].to(model.device)
                batch_attention_mask = attention_mask[start:end].to(model.device)

                outputs = model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    output_hidden_states=True,
                )

                # Get hidden states from last layer
                hidden = outputs.hidden_states[-1]  # [batch, seq_len, hidden_dim]

                # Apply final norm
                if norm_layer is not None:
                    hidden = norm_layer(hidden)

                # Get target token IDs
                target_ids = outputs.logits.argmax(dim=-1)  # [batch, seq_len]

                all_hidden.append(hidden.cpu())
                all_target_ids.append(target_ids.cpu())

                del outputs
                torch.cuda.empty_cache()

        # Concatenate all batches
        all_hidden = torch.cat(all_hidden, dim=0)  # [num_sequences, seq_len, hidden_dim]
        all_target_ids = torch.cat(all_target_ids, dim=0)  # [num_sequences, seq_len]

        # Save as feature file
        output_name = f"features_{packed_file.stem}.pt"
        output_path = output_dir / output_name

        save_dict = {
            "input_ids": input_ids,
            "hidden_states": all_hidden,
            "target_token_ids": all_target_ids,
            "loss_mask": loss_mask,
            "attention_mask": attention_mask,
            "num_samples": num_sequences,
            "lm_head": lm_head_state,
            "vocab_size": lm_head_weight.shape[0],
            "hidden_size": all_hidden.shape[-1],
        }

        torch.save(save_dict, output_path)
        print(f"  Saved: {output_name} ({num_sequences} samples)")
        total_samples += num_sequences

    print(f"\n{'='*50}")
    print(f"Extraction complete!")
    print(f"  Total samples: {total_samples}")
    print(f"  Output: {output_dir}")
    print(f"  lm_head: {lm_head_weight.shape} (saved with features)")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()