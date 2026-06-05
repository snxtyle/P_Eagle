#!/usr/bin/env python3
"""
P-EAGLE Trainer for Packed Sequences - FIXED VERSION

Key fixes:
1. Initialize dim_projection as identity and FREEZE it for self-speculative mode
2. Ensure MTP heads receive proper gradients
3. Use pre-extracted hidden states directly (not EagleDrafterModel forward pass)
4. Remove the broken use_eagle3 mode that caused the mismatch

Usage:
    python scripts/train_packed_fixed.py \
        --feature_dir data/gemma3_features/features \
        --output_dir outputs/gemma3_p1_fixed \
        --epochs 15 \
        --lr 1e-4

"""

import argparse
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from pathlib import Path
from tqdm import tqdm
import json
import os
import time
import logging
import sys

# Weights & Biases
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed.")

# Setup file logging
def setup_file_logging(output_dir):
    """Setup logging to both file and stdout."""
    log_file = Path(output_dir) / "training.log"

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return log_file


class PackedDataset(Dataset):
    """Dataset for packed 4096-token sequences."""

    def __init__(self, feature_dir, block_size=4096, val_split=0.1, seed=42):
        self.feature_dir = Path(feature_dir)
        self.block_size = block_size

        self.shards = []
        self.sample_offsets = []

        shard_files = sorted(self.feature_dir.glob("*.pt"))
        for shard_idx, sf in enumerate(shard_files):
            data = torch.load(sf, map_location="cpu")
            num_samples = data["num_samples"]
            self.shards.append(data)
            for i in range(num_samples):
                self.sample_offsets.append((shard_idx, i))

        print(f"PackedDataset: {len(self.sample_offsets)} samples across {len(self.shards)} shards")

        if val_split > 0:
            import random
            random.seed(seed)
            indices = list(range(len(self.sample_offsets)))
            random.shuffle(indices)

            val_size = int(len(indices) * val_split)
            self.train_indices = indices[val_size:]
            self.val_indices = indices[:val_size]

            print(f"  Train: {len(self.train_indices)} samples, Val: {len(self.val_indices)} samples")
        else:
            self.train_indices = list(range(len(self.sample_offsets)))
            self.val_indices = []

    def __len__(self):
        return len(self.train_indices)

    def get_validation_set(self):
        return [self.sample_offsets[i] for i in self.val_indices]

    def __getitem__(self, idx):
        actual_idx = self.train_indices[idx]
        shard_idx, sample_idx = self.sample_offsets[actual_idx]
        data = self.shards[shard_idx]

        return {
            "input_ids": data["input_ids"][sample_idx],
            "hidden_states": data["hidden_states"][sample_idx],
            "target_token_ids": data["target_token_ids"][sample_idx],
            "loss_mask": data["loss_mask"][sample_idx],
        }

    def get_validation_item(self, val_idx):
        actual_idx = self.val_indices[val_idx]
        shard_idx, sample_idx = self.sample_offsets[actual_idx]
        data = self.shards[shard_idx]

        return {
            "input_ids": data["input_ids"][sample_idx],
            "hidden_states": data["hidden_states"][sample_idx],
            "target_token_ids": data["target_token_ids"][sample_idx],
            "loss_mask": data["loss_mask"][sample_idx],
        }

    def __len_val__(self):
        return len(self.val_indices)


class MTPHeadFixed(nn.Module):
    """
    Fixed Multi-Token Prediction Head.

    Key changes:
    - Initialize with larger weights for faster learning
    - Lower dropout (0.05 instead of 0.1)
    - Proper rescale initialization
    - Uses BFloat16 dtype
    """

    def __init__(self, hidden_size, intermediate_size=None, dropout=0.05, dtype=torch.bfloat16):
        super().__init__()
        intermediate_size = intermediate_size or hidden_size
        self.dtype = dtype

        self.norm = nn.LayerNorm(hidden_size, dtype=dtype)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size, dtype=dtype),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, hidden_size, dtype=dtype),
        )
        # Initialize rescale to larger value for stronger signal
        self.rescale = nn.Parameter(torch.ones(hidden_size, dtype=dtype) * 5.0)

    def forward(self, x):
        # Ensure input is in correct dtype
        if x.dtype != self.dtype:
            x = x.to(self.dtype)
        x_norm = self.norm(x)
        out = self.mlp(x_norm)
        # Effective scale: rescale * 0.1 = 5.0 * 0.1 = 0.5
        # This gives moderate initial scaling
        return out * self.rescale * 0.1


class PEagleDrafterFixed(nn.Module):
    """
    Fixed P-EAGLE Drafter.

    Key fixes:
    1. dim_projection is IDENTITY and FROZEN for self-speculative mode
    2. MTP heads have proper initialization
    3. Uses pre-extracted hidden states directly (no base_model forward pass)
    """

    def __init__(self, hidden_size, vocab_size, speculation_depth=4, dropout=0.05):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.speculation_depth = speculation_depth

        # FIX 1: dim_projection is IDENTITY and FROZEN
        # For self-speculative (base_hidden_dim == target_hidden_dim),
        # identity is the correct projection
        self.dim_projection = nn.Identity()
        # Freeze it - no gradients needed
        for param in self.dim_projection.parameters():
            param.requires_grad = False
        print("  dim_projection: IDENTITY (frozen)")

        # MTP Heads with proper initialization (use bfloat16)
        self.mtp_heads = nn.ModuleList([
            MTPHeadFixed(hidden_size, hidden_size, dropout=dropout, dtype=torch.bfloat16)
            for _ in range(speculation_depth)
        ])
        print(f"  MTP heads: {speculation_depth} heads with dropout={dropout}")

        # target_lm_head: projects hidden states to vocabulary
        # Initialize from base model lm_head weights
        lm_head_weights_path = Path(__file__).parent.parent / 'lm_head_weights.pt'
        if lm_head_weights_path.exists():
            lm_head_state = torch.load(lm_head_weights_path, map_location='cpu', weights_only=True)
            self.target_lm_head = nn.Linear(hidden_size, vocab_size, bias=False, dtype=torch.bfloat16)
            with torch.no_grad():
                self.target_lm_head.weight.copy_(lm_head_state['weight'].to(torch.bfloat16))
            print(f"  target_lm_head: loaded from base model (frozen)")
            # Freeze target_lm_head - we only train MTP heads
            for param in self.target_lm_head.parameters():
                param.requires_grad = False
        else:
            self.target_lm_head = nn.Linear(hidden_size, vocab_size, bias=False, dtype=torch.bfloat16)
            print(f"  target_lm_head: random init (frozen)")
            for param in self.target_lm_head.parameters():
                param.requires_grad = False

    def forward(self, hidden_states, return_all_predictions=True):
        """
        Forward pass.

        Args:
            hidden_states: [batch, seq_len, hidden_dim] - pre-extracted hidden states from target model

        Returns:
            predictions: List of [batch, seq_len, vocab_size] tensors, one per head
        """
        batch_size, seq_len, _ = hidden_states.shape

        # dim_projection is identity, so this is just hidden_states
        projected = self.dim_projection(hidden_states)

        # MTP heads predict tokens at future positions
        predictions = []
        for i, mtp_head in enumerate(self.mtp_heads):
            shift = i + 1  # Head k predicts token at position T + k + 1

            if shift < seq_len:
                # Use ALL valid positions (positions 0 to seq_len-shift-1)
                pred_input = projected[:, :-shift, :]  # [batch, seq-shift, hidden]

                # MTP head processes each position
                mtp_hidden = mtp_head(pred_input)  # [batch, seq-shift, hidden]

                # Project to vocabulary
                logits = self.target_lm_head(mtp_hidden)  # [batch, seq-shift, vocab]
                predictions.append(logits)
            else:
                predictions.append(None)

        return predictions


def compute_loss(predictions, target_ids, config, loss_mask=None, return_metrics=False):
    """
    Compute loss for packed sequences.

    Each MTP head predicts tokens at shift positions ahead.
    """
    seq_len = target_ids.shape[1]
    device = target_ids.device
    batch_size = target_ids.shape[0]

    ce_criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)

    total_loss = torch.tensor(0.0, device=device)
    total_count = 0

    total_ce = 0.0
    total_correct = 0
    total_valid = 0

    for i, pred in enumerate(predictions):
        if pred is None:
            continue

        shift = i + 1
        if shift >= seq_len:
            continue

        # pred: [batch, seq-shift, vocab]
        # target: positions shift to end predict tokens shift+1 to end
        targets = target_ids[:, shift:].contiguous()  # [batch, seq-shift]

        # Apply loss mask if provided
        if loss_mask is not None:
            mask = loss_mask[:, shift:].float()  # [batch, seq-shift]
        else:
            mask = torch.ones_like(targets, dtype=torch.float32)

        # Flatten for loss computation
        pred_flat = pred.float().reshape(-1, pred.size(-1))  # [batch*(seq-shift), vocab]
        targets_flat = targets.reshape(-1)  # [batch*(seq-shift)]
        mask_flat = mask.reshape(-1)  # [batch*(seq-shift)]

        # Compute loss
        ce_loss = ce_criterion(pred_flat, targets_flat)  # [batch*(seq-shift)]
        masked_loss = ce_loss * mask_flat
        valid_count = mask_flat.sum().item()

        if valid_count > 0:
            head_loss = masked_loss.sum() / valid_count
        else:
            head_loss = torch.tensor(0.0, device=device)

        # Weighted loss (earlier heads are more important)
        weight = max(0.5, 1.0 - i * 0.1)
        total_loss = total_loss + head_loss * weight
        total_count += 1

        if return_metrics:
            total_ce += masked_loss.sum().item()
            total_valid += valid_count

            # Accuracy
            preds_flat = pred_flat.argmax(dim=-1)
            correct = (preds_flat == targets_flat) & mask_flat.bool()
            total_correct += correct.sum().item()

    if total_count > 0:
        total_loss = total_loss / total_count

    metrics = {}
    if return_metrics:
        metrics['ce'] = total_ce / total_valid if total_valid > 0 else 0.0
        metrics['acc'] = 100.0 * total_correct / total_valid if total_valid > 0 else 0.0

    return total_loss, metrics


def validate(model, dataset, device, config, max_samples=100):
    """Validation pass with metrics."""
    model.eval()
    total_ce = 0.0
    total_correct = 0
    total_predictions = 0
    num_samples = 0

    with torch.no_grad():
        for i in range(min(dataset.__len_val__(), max_samples)):
            batch = dataset.get_validation_item(i)
            hidden_states = batch["hidden_states"].unsqueeze(0).to(device, dtype=torch.bfloat16)
            target_ids = batch["target_token_ids"].unsqueeze(0).to(device)
            loss_mask = batch["loss_mask"].unsqueeze(0).to(device)

            # Forward pass
            predictions = model(hidden_states)

            # Compute loss
            _, metrics = compute_loss(
                predictions, target_ids, config,
                loss_mask=loss_mask,
                return_metrics=True
            )

            # Accumulate metrics
            num_predictions = sum(1 for p in predictions if p is not None)
            total_predictions += num_predictions
            total_ce += metrics.get('ce', 0.0) * num_predictions
            total_correct += metrics.get('acc', 0.0) / 100.0 * num_predictions
            num_samples += 1

    model.train()

    return {
        'val_loss': total_ce / total_predictions if total_predictions > 0 else 0.0,
        'val_acc': 100.0 * total_correct / total_predictions if total_predictions > 0 else 0.0,
        'num_samples': num_samples,
    }


def train_fixed(args):
    """Main training loop for fixed P-EAGLE."""

    local_rank = int(os.getenv("LOCAL_RANK", 0))
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    is_main_process = (rank == 0)

    if world_size > 1:
        from datetime import timedelta
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(seconds=3600),
            init_method="env://",
        )

    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if available_gpus > 0:
        local_rank = local_rank % available_gpus

    if torch.cuda.is_available() and available_gpus > 0:
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    if is_main_process:
        print(f"Training on: {device} (rank {rank}/{world_size})")

    # Load dataset
    dataset = PackedDataset(args.feature_dir, args.block_size, val_split=args.val_split)

    # Create DataLoader
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

    # Create model
    model = PEagleDrafterFixed(
        hidden_size=args.hidden_size,
        vocab_size=args.vocab_size,
        speculation_depth=args.speculation_depth,
        dropout=args.dropout
    ).to(device)

    # Wrap with DDP if multi-GPU
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
        model._set_static_graph()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if is_main_process:
        print(f"\nModel configuration:")
        print(f"  Hidden size: {args.hidden_size}")
        print(f"  Vocab size: {args.vocab_size}")
        print(f"  Speculation depth: {args.speculation_depth}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")

        # Setup logging
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        log_file = setup_file_logging(output_dir)
        logging.info(f"Fixed training - dim_projection is identity (frozen)")
        logging.info(f"Training config: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")

        # Initialize wandb
        if args.wandb and WANDB_AVAILABLE:
            wandb_init_kwargs = {
                "project": args.wandb_project or "p-eagle",
                "entity": args.wandb_entity,
                "name": args.wandb_name,
                "config": {
                    "hidden_size": args.hidden_size,
                    "vocab_size": args.vocab_size,
                    "speculation_depth": args.speculation_depth,
                    "batch_size": args.batch_size,
                    "learning_rate": args.lr,
                    "epochs": args.epochs,
                    "dropout": args.dropout,
                }
            }
            wandb.init(**wandb_init_kwargs)
            print(f"Weights & Biases: {wandb.run.url if hasattr(wandb, 'run') else 'N/A'}")
        elif args.wandb and not WANDB_AVAILABLE:
            print("Warning: --wandb specified but wandb not installed")

    # Optimizer - only optimize MTP heads (dim_projection and target_lm_head are frozen)
    optimizer = optim.AdamW(
        model.module.mtp_heads.parameters() if isinstance(model, DDP) else model.mtp_heads.parameters(),
        lr=args.lr,
        weight_decay=0.1
    )

    total_steps = len(dataset) * args.epochs // args.batch_size
    warmup_steps = args.warmup_steps

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return 1.0

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    model.train()
    global_step = 0

    if is_main_process:
        print(f"\nTraining configuration:")
        print(f"  Epochs: {args.epochs}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Learning rate: {args.lr}")
        print(f"  Training ONLY MTP heads (dim_projection=identity, target_lm_head=frozen)")

    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(args.epochs):
        if world_size > 1:
            sampler.set_epoch(epoch)

        epoch_loss = 0.0
        epoch_steps = 0
        total_tokens = 0
        total_ce = 0.0
        total_acc = 0.0
        num_updates = 0

        if is_main_process:
            pbar = tqdm(total=len(dataloader), desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch in dataloader:
            hidden_states = batch["hidden_states"].to(device, dtype=torch.bfloat16)
            target_ids = batch["target_token_ids"].to(device)
            loss_mask = batch["loss_mask"].to(device)

            # Forward pass
            predictions = model(hidden_states)

            # Compute loss
            loss, metrics = compute_loss(
                predictions, target_ids, None,
                loss_mask=loss_mask,
                return_metrics=True
            )

            # Scale loss for gradient accumulation
            loss = loss / args.gradient_accumulation

            # Backward
            loss.backward()

            # Accumulate metrics (scale back up)
            epoch_loss += loss.item() * args.gradient_accumulation
            total_ce += metrics.get('ce', 0.0)
            total_acc += metrics.get('acc', 0.0)
            epoch_steps += 1
            num_updates += 1
            total_tokens += hidden_states.size(0) * hidden_states.size(1)

            # Gradient accumulation: only optimize every N steps
            if (epoch_steps) % args.gradient_accumulation == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

                if is_main_process:
                    pbar.set_postfix({
                        'loss': f'{loss.item() * args.gradient_accumulation:.4f}',
                        'ce': f'{metrics.get("ce", 0):.4f}',
                        'acc': f'{metrics.get("acc", 0):.1f}%',
                        'lr': f'{scheduler.get_last_lr()[0]:.2e}',
                    })
                    pbar.update(world_size)

        if is_main_process:
            pbar.close()

            # Epoch summary
            elapsed = time.time() - epoch_start_time if epoch > 0 else 0
            epoch_avg_loss = epoch_loss / max(epoch_steps, 1)
            epoch_avg_ce = total_ce / max(num_updates, 1)
            epoch_avg_acc = total_acc / max(num_updates, 1)

            print(f"\nEpoch {epoch+1}: loss={epoch_avg_loss:.4f}, ce={epoch_avg_ce:.4f}, acc={epoch_avg_acc:.1f}%")

            # Validation
            if (epoch + 1) % args.val_every == 0:
                val_results = validate(model, dataset, device, None)
                val_loss = val_results['val_loss']
                val_acc = val_results['val_acc']

                print(f"  Validation: loss={val_loss:.4f}, acc={val_acc:.1f}%")
                logging.info(f"Epoch {epoch+1} validation: loss={val_loss:.4f}, acc={val_acc:.1f}%")

                # Log to wandb
                if args.wandb and WANDB_AVAILABLE:
                    wandb.log({
                        'epoch': epoch + 1,
                        'train/loss': epoch_avg_loss,
                        'train/ce': epoch_avg_ce,
                        'train/accuracy': epoch_avg_acc,
                        'val/loss': val_loss,
                        'val/accuracy': val_acc,
                        'lr': scheduler.get_last_lr()[0],
                    }, step=global_step)

                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = epoch + 1

                    # Get state dict
                    if isinstance(model, DDP):
                        state_dict = model.module.state_dict()
                        net = model.module
                    else:
                        state_dict = model.state_dict()
                        net = model

                    # Handle dim_projection - it might not exist if it's identity/frozen
                    if 'dim_projection' in state_dict:
                        dim_proj_state = state_dict['dim_projection']
                    else:
                        # Get from model directly (it might be identity or frozen)
                        dim_proj_state = net.dim_projection.state_dict() if hasattr(net, 'dim_projection') else None

                    # Handle mtp_heads - get directly from model since state_dict keys might vary
                    mtp_heads_state = []
                    if hasattr(net, 'mtp_heads'):
                        for i, head in enumerate(net.mtp_heads):
                            head_state = {}
                            for key, value in head.state_dict().items():
                                head_state[key] = value
                            mtp_heads_state.append(head_state)
                    elif 'mtp_heads.0' in state_dict:
                        mtp_heads_state = [state_dict[f'mtp_heads.{i}'] for i in range(args.speculation_depth)]

                    # Handle target_lm_head - get from model if not in state_dict (might be frozen)
                    if 'target_lm_head' in state_dict:
                        target_lm_head_state = state_dict['target_lm_head']
                    elif hasattr(net, 'target_lm_head'):
                        target_lm_head_state = net.target_lm_head.state_dict()
                    else:
                        target_lm_head_state = None

                    # Save checkpoint
                    ckpt = {
                        'dim_projection': dim_proj_state,
                        'mtp_heads': mtp_heads_state,
                        'target_lm_head': target_lm_head_state,
                        'config': {
                            'hidden_size': args.hidden_size,
                            'vocab_size': args.vocab_size,
                            'speculation_depth': args.speculation_depth,
                        }
                    }
                    torch.save(ckpt, Path(args.output_dir) / 'best_model_fixed.pt')
                    print(f"  Saved best model (val_acc={val_acc:.1f}%)")

            # Save checkpoint
            if (epoch + 1) % args.save_every == 0:
                if isinstance(model, DDP):
                    state_dict = model.module.state_dict()
                    net = model.module
                else:
                    state_dict = model.state_dict()
                    net = model

                # Handle dim_projection - it might not exist if it's identity/frozen
                if 'dim_projection' in state_dict:
                    dim_proj_state = state_dict['dim_projection']
                else:
                    # Get from model directly (it might be identity or frozen)
                    dim_proj_state = net.dim_projection.state_dict() if hasattr(net, 'dim_projection') else None

                # Handle mtp_heads - get directly from model since state_dict keys might vary
                mtp_heads_state = []
                if hasattr(net, 'mtp_heads'):
                    for i, head in enumerate(net.mtp_heads):
                        head_state = {}
                        for key, value in head.state_dict().items():
                            head_state[key] = value
                        mtp_heads_state.append(head_state)
                elif 'mtp_heads.0' in state_dict:
                    mtp_heads_state = [state_dict[f'mtp_heads.{i}'] for i in range(args.speculation_depth)]

                # Handle target_lm_head - get from model if not in state_dict (might be frozen)
                if 'target_lm_head' in state_dict:
                    target_lm_head_state = state_dict['target_lm_head']
                elif hasattr(net, 'target_lm_head'):
                    target_lm_head_state = net.target_lm_head.state_dict()
                else:
                    target_lm_head_state = None

                ckpt = {
                    'dim_projection': dim_proj_state,
                    'mtp_heads': mtp_heads_state,
                    'target_lm_head': target_lm_head_state,
                    'config': {
                        'hidden_size': args.hidden_size,
                        'vocab_size': args.vocab_size,
                        'speculation_depth': args.speculation_depth,
                    }
                }
                torch.save(ckpt, Path(args.output_dir) / f'checkpoint_epoch_{epoch+1}.pt')

        epoch_start_time = time.time()

    if is_main_process:
        print(f"\nTraining complete! Best val_acc={best_val_acc:.1f}% at epoch {best_epoch}")
        logging.info(f"Training complete. Best val_acc={best_val_acc:.1f}% at epoch {best_epoch}")

        # Log final metrics to wandb
        if args.wandb and WANDB_AVAILABLE:
            wandb.finish()

    if world_size > 1:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="P-EAGLE Fixed Training")

    parser.add_argument("--feature_dir", type=str, required=True,
                        help="Directory with pre-extracted features")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for checkpoints")
    parser.add_argument("--hidden_size", type=int, default=2560)
    parser.add_argument("--vocab_size", type=int, default=262208)
    parser.add_argument("--speculation_depth", type=int, default=4)
    parser.add_argument("--block_size", type=int, default=4096)

    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)  # Learning rate
    parser.add_argument("--gradient_accumulation", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--dropout", type=float, default=0.05)  # Lower dropout

    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--val_every", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)

    # Weights & Biases arguments
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="p-eagle", help="Wandb project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Wandb entity name")
    parser.add_argument("--wandb_name", type=str, default=None, help="Wandb run name")

    args = parser.parse_args()

    import random
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_fixed(args)


if __name__ == "__main__":
    main()