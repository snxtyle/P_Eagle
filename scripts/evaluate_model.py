#!/usr/bin/env python3
"""
Evaluate P-EAGLE Model on Validation Set

Evaluates the trained model and shows:
- Loss and accuracy metrics
- Sample predictions
- Token-level accuracy breakdown
"""

import argparse
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import sys
import os
from pathlib import Path
import random

# Add parent dir to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scripts.train_packed import PackedDataset, PEagleDrafterPacked, PEaglePackedConfig, compute_packed_loss, MTPHead


def load_model(checkpoint_path, config, device):
    """Load the trained model from checkpoint."""
    model = PEagleDrafterPacked(config).to(device)

    # Add target_lm_head
    model.target_lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=True, dtype=torch.bfloat16).to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    # Handle DDP state dict prefixes
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    return model


def evaluate_model(model, dataset, device, config, num_samples=50, use_wandb=False):
    """Evaluate the model on validation samples."""
    model.eval()

    total_ce = 0.0
    total_correct = 0
    total_predictions = 0
    total_mse = 0.0

    all_predictions = []

    ce_criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)

    with torch.no_grad():
        num_val_samples = min(dataset.__len_val__(), num_samples)

        for i in range(num_val_samples):
            batch = dataset.get_validation_item(i)
            input_ids = batch["input_ids"].unsqueeze(0).to(device).long()
            hidden_states = batch["hidden_states"].unsqueeze(0).to(device)
            target_ids = batch["target_token_ids"].unsqueeze(0).to(device)

            # Convert to model dtype
            model_dtype = next(model.parameters()).dtype
            hidden_states = hidden_states.to(model_dtype)

            # Forward pass
            raw_predictions = model(hidden_states, return_all_predictions=True)

            # Convert to logits
            predictions = []
            for pred in raw_predictions:
                if pred is not None:
                    pred = pred.to(model.target_lm_head.weight.dtype)
                    logits = model.target_lm_head(pred)
                    predictions.append(logits)
                else:
                    predictions.append(None)

            # Compute metrics
            loss, metrics = compute_packed_loss(
                predictions, target_ids, config,
                return_metrics=True
            )

            total_ce += metrics.get('ce', 0.0) * config.speculation_depth
            total_correct += metrics.get('acc', 0.0) / 100.0 * config.speculation_depth
            total_mse += metrics.get('mse', 0.0) * config.speculation_depth
            total_predictions += config.speculation_depth

            # Store sample predictions
            if i < 5:  # Store first 5 for display
                sample = {
                    'input_ids': input_ids[0].cpu().tolist()[:20],
                    'target_ids': target_ids[0].cpu().tolist(),
                    'predictions': [p[0].cpu() if p is not None else None for p in predictions],
                    'head_idx': 0,  # Track which head this prediction is from
                }
                all_predictions.append(sample)

    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Samples evaluated: {num_val_samples}")
    print(f"Average CE Loss: {total_ce / num_val_samples:.4f}")
    print(f"Average Accuracy: {100.0 * total_correct / total_predictions:.2f}%")
    print(f"Average MSE: {total_mse / num_val_samples:.4f}")
    print("="*60)

    # Show sample predictions
    print("\nSAMPLE PREDICTIONS (first 5 samples):")
    print("-"*60)

    for i, sample in enumerate(all_predictions):
        print(f"\nSample {i+1}:")

        # Get top predictions for each MTP head
        for head_idx, pred in enumerate(sample['predictions']):
            if pred is not None:
                # Get top 5 predicted tokens
                # pred shape: [1, vocab_size]
                top_probs, top_indices = torch.topk(pred.squeeze(0), k=min(5, pred.size(-1)))  # squeeze batch dim
                top_probs = torch.softmax(top_probs.float(), dim=-1)

                # Get target token
                target_pos = head_idx + 1  # T + k + 1
                if target_pos < len(sample['target_ids']):
                    target_token = sample['target_ids'][target_pos]
                    pred_token = top_indices[0].item()

                    top5_list = [(int(top_indices[j].item()), f'{top_probs[j].item():.3f}') for j in range(min(5, len(top_indices)))]
                    print(f"  Head {head_idx}: Target={target_token}, Predicted={pred_token} "
                          f"(prob={top_probs[0].item():.4f}), "
                          f"Top-5: {top5_list}")

        print()

    return {
        'ce_loss': total_ce / num_val_samples,
        'accuracy': 100.0 * total_correct / total_predictions,
        'mse': total_mse / num_val_samples,
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate P-EAGLE model')
    parser.add_argument('--checkpoint', type=str, default='outputs/gemma3_p1/final_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--feature_dir', type=str, default='data/gemma3_features/features',
                        help='Path to feature directory')
    parser.add_argument('--hidden_size', type=int, default=2560,
                        help='Hidden size')
    parser.add_argument('--vocab_size', type=int, default=262208,
                        help='Vocab size')
    parser.add_argument('--speculation_depth', type=int, default=1,
                        help='Speculation depth')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of validation samples to evaluate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    # Set seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    print(f"\nLoading dataset from {args.feature_dir}")
    dataset = PackedDataset(args.feature_dir, val_split=0.05, seed=args.seed)
    print(f"Validation samples: {dataset.__len_val__()}")

    # Create config
    config = PEaglePackedConfig(argparse.Namespace(
        hidden_size=args.hidden_size,
        vocab_size=args.vocab_size,
        speculation_depth=args.speculation_depth,
        block_size=4096,
        lr=0,
        epochs=0,
        batch_size=1,
        gradient_accumulation=1,
        warmup_steps=0,
        max_steps=0,
    ))

    # Load model
    print(f"\nLoading model from {args.checkpoint}")
    model = load_model(args.checkpoint, config, device)

    # Evaluate
    results = evaluate_model(
        model, dataset, device, config,
        num_samples=args.num_samples
    )

    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()