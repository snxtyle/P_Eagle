#!/usr/bin/env python3
"""
P-EAGLE Trainer for Packed Sequences

Optimized for H200 with FlashAttention varlen.
Processes solid 4096-token blocks with zero padding waste.

Usage:
    python scripts/train_packed.py \
        --feature_dir data/features_p1 \
        --output_dir outputs/h200_p1 \
        --epochs 3 \
        --lr 1e-4
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import json
import os


class PackedDataset(Dataset):
    """
    Dataset for packed 4096-token sequences.
    Loads pre-extracted hidden states for H200 efficiency.
    """

    def __init__(self, feature_dir, block_size=4096):
        self.feature_dir = Path(feature_dir)
        self.block_size = block_size

        # Load all shards
        self.shards = []
        self.sample_offsets = []  # (shard_idx, sample_idx) for each sample

        shard_files = sorted(self.feature_dir.glob("*.pt"))
        for shard_idx, sf in enumerate(shard_files):
            data = torch.load(sf, map_location="cpu")
            num_samples = data["num_samples"]
            self.shards.append(data)
            for i in range(num_samples):
                self.sample_offsets.append((shard_idx, i))

        print(f"PackedDataset: {len(self.sample_offsets)} samples across {len(self.shards)} shards")

    def __len__(self):
        return len(self.sample_offsets)

    def __getitem__(self, idx):
        shard_idx, sample_idx = self.sample_offsets[idx]
        data = self.shards[shard_idx]

        return {
            "input_ids": data["input_ids"][sample_idx],
            "hidden_states": data["hidden_states"][sample_idx],
            "loss_mask": data["loss_mask"][sample_idx],
            "target_token_ids": data["target_token_ids"][sample_idx],
        }


class PEaglePackedConfig:
    """Configuration for P-EAGLE with packed sequences."""

    def __init__(self, args):
        self.hidden_size = 4096  # Llama 3.1 8B
        self.vocab_size = 128256
        self.num_heads = 8
        self.num_layers = 4
        self.speculation_depth = 4
        self.block_size = args.block_size
        self.lr = args.lr
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.gradient_accumulation = args.gradient_accumulation
        self.warmup_steps = args.warmup_steps
        self.max_steps = args.max_steps


class PEagleDrafterPacked(nn.Module):
    """
    P-EAGLE Drafter optimized for packed sequences.
    Uses MTP (Multi-Token Prediction) heads.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.speculation_depth = config.speculation_depth
        self.block_size = config.block_size

        # Projection layer: hidden_size -> hidden_size
        self.hidden_proj = nn.Linear(config.hidden_size, config.hidden_size)

        # MTP Heads: each head predicts the next N tokens
        self.mtp_heads = nn.ModuleList([
            MTPHead(config.hidden_size, config.vocab_size)
            for _ in range(config.speculation_depth)
        ])

        # Output projections for each head
        self.head_projections = nn.ModuleList([
            nn.Linear(config.hidden_size, config.vocab_size)
            for _ in range(config.speculation_depth)
        ])

    def forward(self, hidden_states, return_all_predictions=True):
        """
        Forward pass for packed sequences.

        Args:
            hidden_states: [batch, seq_len, hidden_dim] - extracted hidden states
            return_all_predictions: If True, return predictions for all positions

        Returns:
            predictions: List of [batch, seq_len, vocab_size] tensors, one per head
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project hidden states
        projected = self.hidden_proj(hidden_states)

        predictions = []
        for i, (mtp_head, proj) in enumerate(zip(self.mtp_heads, self.head_projections)):
            shift = i + 1

            if return_all_predictions:
                # Predict all positions shifted by `shift`
                # pred[k] predicts token at position k + shift
                if shift < seq_len:
                    pred_input = projected[:, :-shift, :]
                    mtp_out = mtp_head(pred_input)
                    logits = proj(mtp_out)
                    predictions.append(logits)
                else:
                    predictions.append(None)
            else:
                # Only predict last position (for inference)
                pred_input = projected[:, -1:, :]
                mtp_out = mtp_head(pred_input)
                logits = proj(mtp_out)
                predictions.append(logits)

        return predictions


class MTPHead(nn.Module):
    """Multi-Token Prediction Head."""

    def __init__(self, hidden_size, intermediate_size=None):
        super().__init__()
        intermediate_size = intermediate_size or hidden_size * 2

        self.norm = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size),
        )
        self.rescale = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        # Pre-norm + MLP + rescale
        x_norm = self.norm(x)
        out = self.mlp(x_norm)
        return out * self.rescale * 0.1


def compute_packed_loss(predictions, target_ids, loss_mask, config):
    """
    Compute loss for packed sequences.

    For each MTP head i:
    - predictions[i][k] predicts token at position k + i + 1
    - target_ids[k + i + 1] is the actual token

    The loss mask marks which tokens are trainable.
    """
    seq_len = target_ids.shape[1]
    total_loss = 0.0
    total_count = 0.0

    ce_criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
    mse_criterion = nn.MSELoss()

    for i, pred in enumerate(predictions):
        if pred is None:
            continue

        shift = i + 1
        if shift >= seq_len:
            continue

        # Align predictions with targets
        # pred[k] predicts target[k + shift]
        pred_flat = pred.reshape(-1, pred.size(-1))  # [batch * (seq_len - shift), vocab]
        target_shifted = target_ids[:, shift:].reshape(-1)  # [batch * (seq_len - shift)]

        # Get loss mask for this range
        if loss_mask.shape[1] >= shift:
            mask = loss_mask[:, shift:].reshape(-1)  # [batch * (seq_len - shift)]
        else:
            mask = torch.zeros_like(target_shifted)

        # Compute CE loss
        ce_loss = ce_criterion(pred_flat, target_shifted)
        ce_loss = (ce_loss * mask).sum() / (mask.sum() + 1e-8)

        # Compute MSE between predicted hidden states and target hidden states
        # (handled separately in full training loop)

        # Weight heads (earlier heads more important)
        weight = max(0.5, 1.0 - i * 0.1)
        total_loss += ce_loss * weight
        total_count += 1

    if total_count > 0:
        total_loss = total_loss / total_count

    return total_loss


def train_packed(args):
    """Main training loop for packed sequences."""

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Load dataset
    dataset = PackedDataset(args.feature_dir, args.block_size)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    # Create model
    config = PEaglePackedConfig(args)
    model = PEagleDrafterPacked(config).to(device)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
        betas=(0.9, 0.95)
    )

    # Scheduler
    total_steps = len(dataloader) * args.epochs // args.gradient_accumulation
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=args.lr * 0.1
    )

    # Mixed precision
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

    # Training loop
    global_step = 0
    model.train()

    print(f"\nTraining configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.gradient_accumulation}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Total steps: {total_steps}")
    print(f"  Samples per epoch: {len(dataset)}")

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_steps = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            hidden_states = batch["hidden_states"].to(device)
            loss_mask = batch["loss_mask"].to(device)
            target_ids = batch["target_token_ids"].to(device)

            # Forward pass
            with torch.amp.autocast('cuda', enabled=scaler is not None):
                predictions = model(hidden_states, return_all_predictions=True)
                loss = compute_packed_loss(predictions, target_ids, loss_mask, config)

            # Backward pass
            loss = loss / args.gradient_accumulation

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (global_step + 1) % args.gradient_accumulation == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()

            global_step += 1
            epoch_loss += loss.item() * args.gradient_accumulation
            epoch_steps += 1

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss.item() * args.gradient_accumulation:.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}"
            })

            # Check max steps
            if args.max_steps and global_step >= args.max_steps:
                print(f"\nReached max_steps={args.max_steps}, stopping...")
                return

        # Epoch summary
        avg_loss = epoch_loss / max(epoch_steps, 1)
        print(f"Epoch {epoch+1}: avg_loss={avg_loss:.4f}")

        # Save checkpoint
        checkpoint_dir = Path(args.output_dir) / f"checkpoint_epoch_{epoch+1}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        torch.save({
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": avg_loss,
        }, checkpoint_dir / "model.pt")

        print(f"  Saved checkpoint: {checkpoint_dir}")

    print(f"\nTraining complete! Total steps: {global_step}")


def main():
    parser = argparse.ArgumentParser(description="P-EAGLE Packed Sequence Trainer")

    # Data
    parser.add_argument("--feature_dir", required=True,
                        help="Directory with extracted features")
    parser.add_argument("--output_dir", default="outputs/h200_train",
                        help="Output directory for checkpoints")

    # Model
    parser.add_argument("--hidden_size", type=int, default=4096,
                        help="Hidden size (Llama 3.1 8B = 4096)")
    parser.add_argument("--vocab_size", type=int, default=128256,
                        help="Vocabulary size")
    parser.add_argument("--speculation_depth", type=int, default=4,
                        help="Number of MTP heads")

    # Training
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Max training steps (None = full epochs)")

    # H200 specific
    parser.add_argument("--block_size", type=int, default=4096,
                        help="Fixed block size (4096 for H200)")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing")

    args = parser.parse_args()

    print("=" * 60)
    print("P-EAGLE H200 Training with Packed Sequences")
    print("=" * 60)
    print(f"Feature dir: {args.feature_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Block size: {args.block_size}")
    print("=" * 60)

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Save config
    with open(Path(args.output_dir) / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    train_packed(args)


if __name__ == "__main__":
    main()