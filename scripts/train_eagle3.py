#!/usr/bin/env python3
"""
EAGLE-3 Aligned P-EAGLE Trainer

This training script ensures training matches inference by using the SAME model
architecture (EagleDrafterModel) for both training and evaluation.

CRITICAL: This fixes the MAL=0 issue caused by train-inference architecture mismatch.

The problem:
- Original training uses PEagleDrafterPacked which takes pre-extracted hidden states
- Inference uses EagleDrafterModel which generates its own hidden states via base_model
- MTP heads trained on pre-extracted hidden states don't work on drafter-generated hidden states

The fix:
- Use EagleDrafterModel for training (same as inference)
- Pass input_ids and target_hidden to the drafter
- Drafter's base_model generates hidden states with hidden injection
- MTP heads train on drafter-generated hidden states
- Training and inference now use the same architecture

Usage:
    python scripts/train_eagle3.py \
        --feature_dir data/features_p1 \
        --output_dir outputs/h200_p1_eagle3 \
        --base_model model_cache/gemma-3-4b-it \
        --epochs 3 \
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
from torch.distributed.elastic.multiprocessing.errors import record
from pathlib import Path
from tqdm import tqdm
import json
import os
import time
import logging
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from p_eagle.models.peagle_drafter import EagleDrafterModel, EagleMTPHead


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

    def __getitem__(self, idx):
        actual_idx = self.train_indices[idx]
        shard_idx, sample_idx = self.sample_offsets[actual_idx]
        data = self.shards[shard_idx]

        return {
            "input_ids": data["input_ids"][sample_idx],
            "target_hidden": data["hidden_states"][sample_idx],  # Pre-extracted target model hidden states
            "target_token_ids": data["target_token_ids"][sample_idx],
        }

    def get_validation_item(self, val_idx):
        actual_idx = self.val_indices[val_idx]
        shard_idx, sample_idx = self.sample_offsets[actual_idx]
        data = self.shards[shard_idx]

        return {
            "input_ids": data["input_ids"][sample_idx],
            "target_hidden": data["hidden_states"][sample_idx],
            "target_token_ids": data["target_token_ids"][sample_idx],
        }

    def __len_val__(self):
        return len(self.val_indices)


def compute_eagle3_loss(mtp_predictions, target_ids, config, speculation_depth, return_metrics=False, hidden_dim=2560):
    """
    Compute loss for EAGLE-3 aligned training.

    Each MTP head predicts the next token given the LAST hidden state.
    - Head 0 predicts token at T+1
    - Head 1 predicts token at T+2
    - etc.

    IMPORTANT: The mtp_predictions should be in VOCABULARY space (after applying target_lm_head).
    If predictions are in hidden_dim space (e.g., 2560), skip them.
    For Gemma-3-4b-it: hidden_dim=2560, vocab_size=262208.
    """
    batch_size = target_ids.shape[0]
    seq_len = target_ids.shape[1]
    device = target_ids.device

    total_loss = torch.tensor(0.0, device=device)
    total_count = 0

    total_ce = 0.0
    total_correct = 0
    total_valid = 0

    ce_criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)

    for i in range(speculation_depth):
        if i >= len(mtp_predictions) or mtp_predictions[i] is None:
            continue

        pred = mtp_predictions[i]  # Shape: [batch, 1, target_hidden_dim]

        shift = i + 1  # Head k predicts token at position T + k + 1
        if shift >= seq_len:
            continue

        # Check if prediction is in hidden_dim space (not applied target_lm_head)
        # Gemma-3-4b-it: hidden_dim=2560, vocab_size=262208
        # If pred.shape[-1] == hidden_dim (2560), target_lm_head was NOT applied, skip
        # If pred.shape[-1] > hidden_dim (262208), we're in vocab space, compute loss
        if pred.shape[-1] <= hidden_dim:
            # Prediction is in hidden_dim space (target_lm_head not applied), skip
            continue

        pred_squeezed = pred.squeeze(1)  # [batch, vocab]
        targets = target_ids[:, shift].clone()
        targets = torch.clamp(targets, 0, pred_squeezed.size(-1) - 1)

        if return_metrics:
            ce_loss = ce_criterion(pred_squeezed, targets)
            total_ce += ce_loss.sum().item()
            total_valid += batch_size

            preds = pred_squeezed.argmax(dim=-1)
            correct = (preds == targets)
            total_correct += correct.sum().item()

        head_loss = ce_criterion(pred_squeezed, targets).mean()
        weight = max(0.5, 1.0 - i * 0.1)
        total_loss = total_loss + head_loss * weight
        total_count += 1

    if total_count > 0:
        total_loss = total_loss / total_count

    metrics = {}
    if return_metrics:
        metrics['ce'] = total_ce / total_valid if total_valid > 0 else 0.0
        metrics['acc'] = 100.0 * total_correct / total_valid if total_valid > 0 else 0.0

    return total_loss, metrics


def train_eagle3(args):
    """Main training loop using EagleDrafterModel to match inference."""

    # Get distributed training info
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    is_main_process = (rank == 0)

    if world_size > 1:
        dist.init_process_group(backend="nccl")

    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if available_gpus > 0:
        local_rank = local_rank % available_gpus

    if torch.cuda.is_available() and available_gpus > 0:
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    if is_main_process:
        print(f"Training on: {device} (rank {rank}/{world_size}, local_rank {local_rank})")
        print(f"Using EAGLE-3 aligned training (matches inference!)")

    # Load dataset
    dataset = PackedDataset(args.feature_dir, args.block_size, val_split=args.val_split)

    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                              num_workers=4, pin_memory=True)
    else:
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
        sampler = None

    # Get config from base model
    from transformers import AutoConfig
    cache_dir = os.environ.get("HF_HOME") or os.path.join(os.getcwd(), "models_cache")
    base_config = AutoConfig.from_pretrained(args.base_model, cache_dir=cache_dir)

    # Handle Gemma3-style nested config
    if hasattr(base_config, 'text_config') and base_config.text_config is not None:
        target_hidden_dim = base_config.text_config.hidden_size
        vocab_size = base_config.text_config.vocab_size
    else:
        target_hidden_dim = base_config.hidden_size
        vocab_size = base_config.vocab_size

    if is_main_process:
        print(f"\nModel configuration:")
        print(f"  Base model: {args.base_model}")
        print(f"  Target hidden dim: {target_hidden_dim}")
        print(f"  Vocab size: {vocab_size}")
        print(f"  Speculation depth: {args.speculation_depth}")

    # Create EagleDrafterModel (same as used in inference!)
    if is_main_process:
        print(f"\nLoading EagleDrafterModel (EAGLE-3 aligned)...")

    drafter = EagleDrafterModel(
        base_model_name=args.base_model,
        target_hidden_dim=target_hidden_dim,
        speculation_depth=args.speculation_depth,
        use_lora=False,  # No LoRA for training
        device=f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu",
        use_hidden_injection=True,
        injection_mode='concat',
        use_flash_attention=False,  # Disable for training stability
    )

    # Add target_lm_head for computing logits from MTP predictions
    # This is what allows us to train with cross-entropy loss
    # CRITICAL FIX: Use bias=True to match evaluate.py and peagle_drafter.py loading
    drafter.target_lm_head = nn.Linear(target_hidden_dim, vocab_size, bias=True,
                                        dtype=torch.bfloat16).to(device)
    print(f"  Added target_lm_head: {target_hidden_dim} -> {vocab_size} (bias=True)")

    drafter = drafter.to(device)

    # Only train: target_hidden_proj, mtp_heads, target_lm_head
    # Freeze base model
    for name, param in drafter.base_model.named_parameters():
        param.requires_grad = False

    # Unfreeze training components
    if hasattr(drafter, 'target_hidden_proj'):
        for param in drafter.target_hidden_proj.parameters():
            param.requires_grad = True
    for param in drafter.mtp_heads.parameters():
        param.requires_grad = True
    for param in drafter.target_lm_head.parameters():
        param.requires_grad = True

    # Count parameters
    trainable_params = sum(p.numel() for p in drafter.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in drafter.parameters())

    if is_main_process:
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  (Base model is frozen, only training MTP heads and projections)")

    # Setup logging
    if is_main_process:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        log_file = setup_file_logging(output_dir)
        logging.info(f"Logging to {log_file}")
        logging.info(f"EAGLE-3 aligned training: base_model={args.base_model}")

    # Optimizer - only for trainable parameters
    optimizer = optim.AdamW(
        [p for p in drafter.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01
    )

    total_steps = len(dataset) * args.epochs // args.batch_size
    warmup_steps = args.warmup_steps

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return 1.0

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

    # Training loop
    drafter.train()
    global_step = 0

    if is_main_process:
        print(f"\nTraining configuration:")
        print(f"  Epochs: {args.epochs}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Gradient accumulation: {args.gradient_accumulation}")
        print(f"  Learning rate: {args.lr}")

    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        if world_size > 1:
            dist.barrier()

        epoch_loss = 0.0
        epoch_steps = 0
        epoch_start_time = time.time()

        if is_main_process:
            total_combined = len(dataloader) * world_size
            pbar = tqdm(total=total_combined, desc=f"Epoch {epoch+1}/{args.epochs}")
        else:
            pbar = None

        accum_counter = 0

        data_iterator = iter(dataloader)

        while True:
            try:
                batch = next(data_iterator)
            except StopIteration:
                break

            input_ids = batch["input_ids"].to(device)  # [batch, seq_len]
            target_hidden = batch["target_hidden"].to(device)  # [batch, seq_len, target_hidden_dim]
            target_ids = batch["target_token_ids"].to(device)  # [batch, seq_len]

            # Forward pass using EagleDrafterModel (MATCHES INFERENCE!)
            with torch.amp.autocast('cuda', enabled=scaler is not None):
                drafter_outputs = drafter.forward(
                    input_ids=input_ids,
                    target_hidden=target_hidden,
                    is_training=True,  # Use full sequence for training
                )

                mtp_predictions = drafter_outputs["mtp_predictions"]

                # Apply target_lm_head to get logits (same as inference evaluation)
                logits_predictions = []
                for i, pred in enumerate(mtp_predictions):
                    if pred is not None:
                        # pred: [batch, 1, target_hidden_dim]
                        # Apply target_lm_head: [batch, 1, vocab]
                        logits = drafter.target_lm_head(pred)
                        logits_predictions.append(logits)
                    else:
                        logits_predictions.append(None)

                # Compute loss
                loss, metrics = compute_eagle3_loss(
                    logits_predictions, target_ids, None,
                    args.speculation_depth, return_metrics=True
                )

            # Backward pass
            loss = loss / args.gradient_accumulation

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accum_counter += 1

            if accum_counter >= args.gradient_accumulation:
                if scaler is not None:
                    scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    [p for p in drafter.parameters() if p.requires_grad],
                    max_norm=1.0
                )

                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                accum_counter = 0

                epoch_loss += loss.item() * args.gradient_accumulation
                epoch_steps += 1

                if pbar is not None:
                    pbar.set_postfix({
                        'loss': f'{loss.item() * args.gradient_accumulation:.4f}',
                        'acc': f'{metrics.get("acc", 0):.1f}%',
                        'lr': f'{scheduler.get_last_lr()[0]:.2e}',
                    })
                    pbar.update(world_size)

        if pbar is not None:
            pbar.close()

        elapsed = time.time() - epoch_start_time
        epoch_avg_loss = epoch_loss / max(epoch_steps, 1)

        if is_main_process:
            print(f"\nEpoch {epoch+1}/{args.epochs} completed in {elapsed:.1f}s")
            print(f"  Average loss: {epoch_avg_loss:.4f}")
            print(f"  Accuracy: {metrics.get('acc', 0):.1f}%")
            logging.info(f"Epoch {epoch+1}: loss={epoch_avg_loss:.4f}, acc={metrics.get('acc', 0):.1f}%")

            # Save checkpoint
            ckpt = {
                'model_state_dict': drafter.state_dict(),
                'config': vars(args),
                'epoch': epoch,
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step,
            }
            torch.save(ckpt, Path(args.output_dir) / f'checkpoint_epoch_{epoch+1}.pt')
            print(f"  Saved checkpoint: checkpoint_epoch_{epoch+1}.pt")

            # Also save eagle_heads.pt for compatibility with evaluate.py
            # CRITICAL FIX: Save dim_projection (projects base_hidden -> target_hidden)
            # NOT target_hidden_proj (which projects target_hidden -> drafter_hidden for injection)
            eagle_heads = {
                'dim_projection': drafter.dim_projection.state_dict(),  # Fixed: was target_hidden_proj
                'mtp_heads': [head.state_dict() for head in drafter.mtp_heads],
                'target_lm_head': drafter.target_lm_head.state_dict(),
                'target_hidden_dim': target_hidden_dim,
                'vocab_size': vocab_size,
            }
            torch.save(eagle_heads, Path(args.output_dir) / 'eagle_heads.pt')
            print(f"  Saved eagle_heads.pt (dim_projection: {drafter.dim_projection.weight.shape})")

        if world_size > 1:
            dist.barrier()

    if is_main_process:
        final_ckpt = {
            'model_state_dict': drafter.state_dict(),
            'config': vars(args),
            'epoch': args.epochs,
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step,
        }
        torch.save(final_ckpt, Path(args.output_dir) / 'final_model.pt')

        # Save eagle_heads.pt
        # CRITICAL FIX: Save dim_projection (projects base_hidden -> target_hidden)
        # NOT target_hidden_proj (which projects target_hidden -> drafter_hidden for injection)
        eagle_heads = {
            'dim_projection': drafter.dim_projection.state_dict(),  # Fixed: was target_hidden_proj
            'mtp_heads': [head.state_dict() for head in drafter.mtp_heads],
            'target_lm_head': drafter.target_lm_head.state_dict(),
            'target_hidden_dim': target_hidden_dim,
            'vocab_size': vocab_size,
        }
        torch.save(eagle_heads, Path(args.output_dir) / 'eagle_heads.pt')

        print(f"\nTraining complete! Model saved to {args.output_dir}/")
        logging.info(f"Training complete. Final model saved to {args.output_dir}/")

    if world_size > 1:
        dist.destroy_process_group()


@record
def main():
    parser = argparse.ArgumentParser(description="EAGLE-3 Aligned P-EAGLE Training")

    # Data arguments
    parser.add_argument("--feature_dir", type=str, required=True,
                        help="Directory with pre-extracted features")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for checkpoints")
    parser.add_argument("--base_model", type=str, required=True,
                        help="Base model path (same as used for inference)")
    parser.add_argument("--block_size", type=int, default=4096,
                        help="Sequence block size")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Resume from checkpoint")

    # Model arguments
    parser.add_argument("--speculation_depth", type=int, default=4,
                        help="Number of MTP heads (speculation depth)")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size per GPU")
    parser.add_argument("--gradient_accumulation", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=100,
                        help="Warmup steps for LR scheduler")

    # Validation arguments
    parser.add_argument("--val_split", type=float, default=0.05,
                        help="Validation split ratio")

    # Misc arguments
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    import random
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_eagle3(args)


if __name__ == "__main__":
    main()