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

For EAGLE-3 aligned training (matches inference):
    python scripts/train_packed.py \
        --feature_dir data/features_p1 \
        --output_dir outputs/h200_p1_eagle3 \
        --use_eagle3 \
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

# Setup file logging
def setup_file_logging(output_dir):
    """Setup logging to both file and stdout."""
    log_file = Path(output_dir) / "training.log"

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # File handler
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return log_file

# Weights & Biases for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Run `pip install wandb` for experiment tracking.")


class PackedDataset(Dataset):
    """
    Dataset for packed 4096-token sequences.
    Loads pre-extracted hidden states for H200 efficiency.

    Supports train/validation split for overfitting detection.
    """

    def __init__(self, feature_dir, block_size=4096, val_split=0.1, seed=42):
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

        # Split into train and validation
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
        """Return a list of validation samples for evaluation."""
        return [self.sample_offsets[i] for i in self.val_indices]

    def __getitem__(self, idx):
        # Map idx to actual sample offset (for training)
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
        """Get a validation sample by validation index."""
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
        """Return number of validation samples."""
        return len(self.val_indices)


class PEaglePackedConfig:
    """Configuration for P-EAGLE with packed sequences."""

    def __init__(self, args):
        self.hidden_size = args.hidden_size  # Use CLI argument
        self.vocab_size = args.vocab_size
        self.num_heads = max(1, args.hidden_size // 256)  # Scale heads
        self.num_layers = 4
        self.speculation_depth = args.speculation_depth
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
    Uses MTP (Multi-Token Prediction) heads with dropout for regularization.
    """

    def __init__(self, config, dropout=0.1):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.speculation_depth = config.speculation_depth
        self.block_size = config.block_size

        # Projection layer: hidden_size -> hidden_size
        self.dim_projection = nn.Linear(config.hidden_size, config.hidden_size)

        # MTP Heads: each head predicts the next N tokens with dropout
        self.mtp_heads = nn.ModuleList([
            MTPHead(config.hidden_size, config.hidden_size, dropout=dropout)
            for _ in range(config.speculation_depth)
        ])

        # NOTE: head_projections removed - we use target_lm_head from the model instead
        # The target_lm_head is applied AFTER the forward pass to convert hidden states to logits
        # This matches the evaluation flow where drafter.target_lm_head is used

    def forward(self, hidden_states, return_all_predictions=True, use_checkpoint=False, max_chunk_size=256):
        """
        Forward pass for packed sequences.

        Args:
            hidden_states: [batch, seq_len, hidden_dim] - extracted hidden states
            return_all_predictions: If True, return predictions for all positions
            use_checkpoint: Use gradient checkpointing for memory efficiency
            max_chunk_size: Process MTP heads in chunks to limit memory

        Returns:
            predictions: List of [batch, seq_len, vocab_size] tensors, one per head
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project hidden states
        projected = self.dim_projection(hidden_states)

        # CRITICAL FIX: Output HIDDEN STATES (not logits) to match evaluation
        # During evaluation, we use target's lm_head to convert hidden → logits
        # So training should predict HIDDEN STATES, not logits
        predictions = []
        for i, mtp_head in enumerate(self.mtp_heads):
            shift = i + 1

            if shift < seq_len:
                # CRITICAL FIX: Use LAST TOKEN ONLY for ALL heads
                # This matches inference where all MTP heads use hidden[T] to predict tokens at T+1, T+2, T+3, T+4
                pred_input = projected[:, -1:, :]  # Last token only: [batch, 1, hidden]
                mtp_hidden = mtp_head(pred_input)  # Output hidden states, NOT logits
                predictions.append(mtp_hidden)
            else:
                predictions.append(None)

        return predictions


class MTPHead(nn.Module):
    """Multi-Token Prediction Head with dropout for regularization."""

    def __init__(self, hidden_size, intermediate_size=None, dropout=0.1):
        super().__init__()
        intermediate_size = intermediate_size or hidden_size

        self.norm = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Dropout(dropout),  # Added dropout for regularization
            nn.Linear(intermediate_size, hidden_size),
        )
        self.rescale = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        # Pre-norm + MLP + rescale
        x_norm = self.norm(x)
        out = self.mlp(x_norm)
        return out * self.rescale * 0.1


def compute_packed_loss(predictions, target_ids, config, max_chunk_size=512, return_metrics=False, loss_mask=None):
    """
    Compute loss for packed sequences with "Last Token Only" training.

    With "Last Token Only", each MTP head:
    - Takes as input: the LAST hidden state (position T)
    - Predicts: ONE token at position T + k + 1 (where k is the head index)

    This matches inference where all heads use hidden[T] to predict tokens at T+1, T+2, T+3, T+4.

    Loss = cross-entropy between predicted token and target token at position T + k + 1
    """
    seq_len = target_ids.shape[1]
    device = target_ids.device
    batch_size = target_ids.shape[0]
    total_loss = torch.tensor(0.0, device=device)
    total_count = 0

    # Metrics tracking
    total_ce = 0.0
    total_correct = 0
    total_valid = 0

    # MSE tracking - true mean squared error between probability distributions
    total_mse_sum = 0.0
    total_mse_count = 0

    # Debug tracking
    total_p_correct_sum = 0.0
    mse_debug_count = 0

    ce_criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)

    for i, pred in enumerate(predictions):
        if pred is None:
            continue

        shift = i + 1  # Head k predicts token at position T + k + 1
        if shift >= seq_len:
            continue

        vocab_size = pred.size(-1)

        # CRITICAL FIX: With "Last Token Only" training, pred has shape [batch, 1, vocab]
        # Each head predicts ONE token per sample
        pred = pred.float()

        # Squeeze to [batch, vocab] - single prediction per sample
        pred_squeezed = pred.squeeze(1)

        # Target computation for "Last Token Only" training
        #
        # IMPORTANT: target_ids = input_ids.roll(-1) means:
        # - target_ids[0] = input_ids[1]
        # - target_ids[1] = input_ids[2]
        # - target_ids[L-1] = input_ids[0] = BOS (WRAPPED - WRONG!)
        #
        # For "Last Token Only" training with the LAST hidden state (position L-1):
        # - There's NO valid target for position L-1
        # - target_ids[L-1] = BOS (wrong)
        # - target_ids[L-2] = input_ids[L-1] = EOS/PAD (also not ideal)
        #
        # NOTE: This training mode is NOT recommended!
        # Use --use_eagle3 for proper full-sequence training instead.
        #
        # For now, we use position L-2 as a rough approximation.
        target_position = max(0, seq_len - 2)  # Second-to-last position (L-2)

        targets = target_ids[:, target_position].clone()

        # Clip to valid range
        targets = torch.clamp(targets, 0, vocab_size - 1)

        # Apply loss_mask if provided
        # loss_mask indicates which positions are valid for training (1 = train, 0 = ignore)
        if loss_mask is not None:
            # loss_mask shape: [batch, seq_len], get mask at target position
            mask = loss_mask[:, target_position].float()  # [batch,]
        else:
            # No loss_mask provided - assume all positions are valid
            mask = torch.ones(batch_size, device=device)

        if return_metrics:
            # CE and accuracy (only on valid positions)
            ce_loss = ce_criterion(pred_squeezed, targets)  # [batch,]

            # Apply mask - multiply loss by mask (masked positions contribute 0)
            masked_ce_loss = ce_loss * mask
            total_ce += masked_ce_loss.sum().item()
            total_valid += mask.sum().item()  # Count only valid positions

            # Accuracy (only on valid positions)
            preds = pred_squeezed.argmax(dim=-1)  # [batch,]
            correct = (preds == targets) & (mask.bool())
            total_correct += correct.sum().item()

            # TRUE MSE: mean((p_target - p_pred)^2) per position (only on valid positions)
            probs = torch.softmax(pred_squeezed, dim=-1)  # [batch, vocab]
            p_correct = probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # [batch,]
            mse_per_token = 2.0 * (1.0 - p_correct)  # ranges 0 to 2

            masked_mse = mse_per_token * mask
            total_mse_sum += masked_mse.sum().item()
            total_mse_count += mask.sum().item()

            masked_p_correct = p_correct * mask
            total_p_correct_sum += masked_p_correct.sum().item()
            mse_debug_count += mask.sum().item()

        # Compute head loss (average across valid positions only)
        ce_loss = ce_criterion(pred_squeezed, targets)  # [batch,]
        masked_loss = ce_loss * mask
        valid_count = mask.sum().item()

        if valid_count > 0:
            head_loss = masked_loss.sum() / valid_count
        else:
            head_loss = torch.tensor(0.0, device=device)

        weight = max(0.5, 1.0 - i * 0.1)
        total_loss = total_loss + head_loss * weight
        total_count += 1

    if total_count > 0:
        total_loss = total_loss / total_count

    # Compute final metrics
    metrics = {}
    if return_metrics:
        metrics['ce'] = total_ce / total_valid if total_valid > 0 else 0.0
        metrics['acc'] = 100.0 * total_correct / total_valid if total_valid > 0 else 0.0

        # TRUE MSE: average MSE per token (ranges 0 to 2)
        metrics['mse'] = total_mse_sum / total_mse_count if total_mse_count > 0 else 0.0
        metrics['avg_p_correct'] = total_p_correct_sum / mse_debug_count if mse_debug_count > 0 else 0.0

    return total_loss, metrics


def compute_full_sequence_loss(predictions, target_ids, config, return_metrics=False, loss_mask=None):
    """
    Compute loss for full-sequence training with EagleDrafterModel.

    For each head:
    - predictions[head] has shape [batch, seq-1, vocab] (one prediction per position)
    - target_ids has shape [batch, seq_len]
    - Each position t predicts target_ids[t]

    The last position (L-1) is excluded because target_ids[L-1] = input_ids[0] = BOS (wrong!)
    """
    batch_size, seq_len_minus_1, vocab_size = predictions[0].shape
    seq_len = target_ids.shape[1]
    device = target_ids.device

    # target_ids[:, :-1] gives targets for positions 0 to L-2
    # This aligns with predictions which have seq-1 positions (0 to L-2)
    targets = target_ids[:, :-1].contiguous()  # [batch, seq-1]

    # Apply loss_mask if provided (exclude last position)
    if loss_mask is not None:
        mask = loss_mask[:, :-1].float()  # [batch, seq-1]
    else:
        mask = torch.ones(batch_size, seq_len_minus_1, device=device)

    ce_criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)

    total_loss = torch.tensor(0.0, device=device)
    total_count = 0

    total_ce = 0.0
    total_correct = 0
    total_valid = 0
    total_mse_sum = 0.0
    total_mse_count = 0

    # Stochastic sequence training: sample a subset of positions to reduce memory
    # Use 512 positions out of 4095 to fit in memory (~8x reduction)
    num_positions_to_sample = min(512, seq_len_minus_1)
    position_indices = torch.randperm(seq_len_minus_1, device=device)[:num_positions_to_sample]

    for i, pred in enumerate(predictions):
        if pred is None:
            continue

        # Sample positions: pred[:, position_indices, :] selects only sampled positions
        pred_sampled = pred[:, position_indices, :].contiguous()  # [batch, num_sampled, vocab]
        targets_sampled = targets[:, position_indices].contiguous()  # [batch, num_sampled]
        mask_sampled = mask[:, position_indices].contiguous()  # [batch, num_sampled]

        # Flatten for loss computation
        # pred_sampled: [batch, num_sampled, vocab] -> [batch * num_sampled, vocab]
        pred_flat = pred_sampled.float().reshape(-1, vocab_size)  # [1024, vocab]
        targets_flat = targets_sampled.reshape(-1)  # [1024]
        mask_flat = mask_sampled.reshape(-1)  # [1024]

        # Compute loss
        ce_loss = ce_criterion(pred_flat, targets_flat)  # [batch * num_sampled]
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

            # MSE
            probs = torch.softmax(pred_flat, dim=-1)
            p_correct = probs.gather(1, targets_flat.unsqueeze(1)).squeeze(1)
            mse_per_token = 2.0 * (1.0 - p_correct)
            masked_mse = mse_per_token * mask_flat
            total_mse_sum += masked_mse.sum().item()
            total_mse_count += valid_count

    if total_count > 0:
        total_loss = total_loss / total_count

    metrics = {}
    if return_metrics:
        metrics['ce'] = total_ce / total_valid if total_valid > 0 else 0.0
        metrics['acc'] = 100.0 * total_correct / total_valid if total_valid > 0 else 0.0
        metrics['mse'] = total_mse_sum / total_mse_count if total_mse_count > 0 else 0.0
        metrics['avg_p_correct'] = 0.0  # Not computed for full sequence

    return total_loss, metrics


def validate(model, dataset, device, config, args=None, max_chunk_size=512):
    """Validation pass with metrics."""
    model.eval()
    total_ce = 0.0
    total_correct = 0
    total_mse = 0.0
    total_p_correct = 0.0
    total_predictions = 0
    num_val_samples = 0

    use_eagle3 = args is not None and getattr(args, 'use_eagle3', False)

    max_val_samples = 100  # Limit validation samples for speed
    with torch.no_grad():
        for i in range(min(dataset.__len_val__(), max_val_samples)):
            batch = dataset.get_validation_item(i)
            input_ids = batch["input_ids"].unsqueeze(0).to(device).long()
            hidden_states = batch["hidden_states"].unsqueeze(0).to(device)
            target_ids = batch["target_token_ids"].unsqueeze(0).to(device)
            loss_mask = batch["loss_mask"].unsqueeze(0).to(device)  # BUG FIX: Use loss_mask

            # Convert hidden_states to match model's dtype (BFloat16)
            model_dtype = next(model.parameters()).dtype
            hidden_states = hidden_states.to(model_dtype)

            # Forward pass
            if use_eagle3:
                # EAGLE-3 aligned validation
                drafter_outputs = model.forward(
                    input_ids=input_ids,
                    target_hidden=hidden_states,
                    is_training=True
                )
                mtp_predictions = drafter_outputs["mtp_predictions"]

                # Apply target_lm_head to get logits
                predictions = []
                for pred in mtp_predictions:
                    if pred is not None:
                        # For EAGLE-3 training, use ONLY the LAST token (matches inference)
                        logits = model.module.target_lm_head(pred[:, -1:, :])
                        predictions.append(logits)
                    else:
                        predictions.append(None)
            else:
                # Original validation - Apply target_lm_head to get logits
                raw_predictions = model(hidden_states, return_all_predictions=True)
                # Get the actual model (handle DDP wrapper)
                ddp_model = model.module if hasattr(model, 'module') else model

                predictions = []
                for i, pred in enumerate(raw_predictions):
                    if pred is not None:
                        # Convert hidden states to logits using target_lm_head
                        pred = pred.to(ddp_model.target_lm_head.weight.dtype)
                        logits = ddp_model.target_lm_head(pred)
                        predictions.append(logits)
                    else:
                        predictions.append(None)

            # Compute loss
            _, metrics = compute_packed_loss(
                predictions, target_ids, config,
                max_chunk_size=max_chunk_size,
                return_metrics=True,
                loss_mask=loss_mask  # BUG FIX: Apply loss_mask during validation
            )

            # Accumulate metrics
            # Each head makes 1 prediction per sample, so num_predictions = num_heads = speculation_depth
            num_predictions = config.speculation_depth
            total_predictions += num_predictions
            total_ce += metrics.get('ce', 0.0) * num_predictions
            total_correct += metrics.get('acc', 0.0) / 100.0 * num_predictions
            total_mse += metrics.get('mse', 0.0) * num_predictions
            total_p_correct += metrics.get('avg_p_correct', 0.0) * num_predictions
            num_val_samples += 1

    model.train()  # Set back to training mode

    return {
        'val_loss': total_ce / total_predictions if total_predictions > 0 else 0.0,
        'val_acc': 100.0 * total_correct / total_predictions if total_predictions > 0 else 0.0,
        'val_mse': total_mse / total_predictions if total_predictions > 0 else 0.0,
        'val_p_correct': total_p_correct / total_predictions if total_predictions > 0 else 0.0,
        'num_samples': num_val_samples,
    }


def train_packed(args):
    """Main training loop for packed sequences."""

    # Get distributed training info
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    is_main_process = (rank == 0)

    # Initialize distributed process group (required for DDP)
    # Use "nccl" backend for GPU-to-GPU communication (much faster than gloo)
    if world_size > 1:
        # Set timeout to 1 hour to avoid premature NCCL timeouts during long operations
        # This prevents training from crashing when one GPU takes longer (e.g., during validation/checkpointing)
        from datetime import timedelta
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(seconds=3600),
            init_method="env://",
        )

    # Check available GPUs
    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    # Ensure local_rank is valid
    if available_gpus > 0:
        local_rank = local_rank % available_gpus

    # Setup device
    if torch.cuda.is_available() and available_gpus > 0:
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    if is_main_process:
        print(f"Training on: {device} (rank {rank}/{world_size}, local_rank {local_rank})")
        if torch.cuda.is_available():
            print(f"Available GPUs: {torch.cuda.device_count()}")

    # Load dataset (only on each process independently, or broadcast)
    dataset = PackedDataset(args.feature_dir, args.block_size, val_split=args.val_split)

    # Use DistributedSampler only for multi-GPU training
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed if hasattr(args, 'seed') else 0,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
        )
    else:
        # Single GPU mode - use regular DataLoader with shuffle
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        sampler = None

    # Create model
    config = PEaglePackedConfig(args)

    if is_main_process:
        print(f"\nModel configuration:")
        print(f"  Hidden size: {config.hidden_size}")
        print(f"  Vocab size: {config.vocab_size}")
        print(f"  Speculation depth: {config.speculation_depth}")

    # CRITICAL FIX: Use EagleDrafterModel for training to match inference
    # This fixes the train-inference architecture mismatch that causes MAL=0
    if args.use_eagle3:
        if is_main_process:
            print(f"\n=== EAGLE-3 ALIGNED TRAINING (MATCHES INFERENCE) ===")
            print(f"  Base model: {args.base_model}")
            print(f"  This trains MTP heads on DRAFTER-generated hidden states")
            print(f"  (not pre-extracted target model hidden states)")

        # Import EagleDrafterModel
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from p_eagle.models.peagle_drafter import EagleDrafterModel

        # Create EagleDrafterModel (same as used in inference!)
        model = EagleDrafterModel(
            base_model_name=args.base_model,
            target_hidden_dim=config.hidden_size,
            speculation_depth=config.speculation_depth,
            use_lora=False,  # No LoRA for training
            device=device,
            use_hidden_injection=True,
            injection_mode='concat',
            use_flash_attention=False,
        )

        # Add target_lm_head for converting MTP predictions to logits
        # CRITICAL FIX: Initialize from base model's lm_head weights (which are pre-trained)
        # NOTE: Base model's lm_head has NO bias, so target_lm_head should also have bias=False
        lm_head_weights_path = Path(__file__).parent.parent / 'lm_head_weights.pt'
        if lm_head_weights_path.exists():
            lm_head_state = torch.load(lm_head_weights_path, map_location='cpu', weights_only=True)
            model.target_lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, dtype=torch.bfloat16).to(device)
            with torch.no_grad():
                model.target_lm_head.weight.copy_(lm_head_state['weight'].to(torch.bfloat16))
                # NOTE: Base lm_head has no bias, so we don't copy bias
            if is_main_process:
                print(f"  Added target_lm_head: {config.hidden_size} -> {config.vocab_size} (initialized from base model, bias=False)")
        else:
            model.target_lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, dtype=torch.bfloat16).to(device)
            if is_main_process:
                print(f"  Added target_lm_head: {config.hidden_size} -> {config.vocab_size} (random init, bias=False)")

        # Freeze base model - only train MTP heads and projections
        for name, param in model.base_model.named_parameters():
            param.requires_grad = False
        if hasattr(model, 'target_hidden_proj'):
            for param in model.target_hidden_proj.parameters():
                param.requires_grad = True
        for param in model.mtp_heads.parameters():
            param.requires_grad = True
        for param in model.target_lm_head.parameters():
            param.requires_grad = True

        if is_main_process:
            print(f"  Base model is FROZEN (only MTP heads and projections train)")
    else:
        model = PEagleDrafterPacked(config, dropout=args.dropout).to(device)
        # Add target_lm_head to convert MTP hidden outputs to logits for loss computation
        # CRITICAL FIX: Initialize from base model's lm_head weights (which are pre-trained)
        # NOTE: Base model's lm_head has NO bias, so target_lm_head should also have bias=False
        lm_head_weights_path = Path(__file__).parent.parent / 'lm_head_weights.pt'
        if lm_head_weights_path.exists():
            lm_head_state = torch.load(lm_head_weights_path, map_location='cpu', weights_only=True)
            model.target_lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, dtype=torch.bfloat16).to(device)
            with torch.no_grad():
                model.target_lm_head.weight.copy_(lm_head_state['weight'].to(torch.bfloat16))
                # NOTE: Base lm_head has no bias, so we don't copy bias
            if is_main_process:
                print(f"  Added target_lm_head: {config.hidden_size} -> {config.vocab_size} (initialized from base model, bias=False)")
        else:
            model.target_lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, dtype=torch.bfloat16).to(device)
            if is_main_process:
                print(f"  Added target_lm_head: {config.hidden_size} -> {config.vocab_size} (random init, bias=False)")

    # Resume from checkpoint if specified
    if args.resume_from:
        if is_main_process:
            print(f"\nResuming from checkpoint: {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)
        if is_main_process:
            ckpt_config = checkpoint.get('config', {})
            print(f"  Loaded checkpoint config: {ckpt_config}")
            print(f"  Checkpoint loaded successfully!")

    # Only wrap with DDP for multi-GPU training
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
        # Enable static graph mode to avoid "Expected to mark a variable ready only once" errors
        # when target_lm_head participates in multiple backward paths within a single iteration
        model._set_static_graph()

    # Count parameters (only on main process to avoid clutter)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if is_main_process:
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Distributed training across {world_size} GPUs")

    # Setup file logging (only on rank 0)
    if is_main_process:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        log_file = setup_file_logging(output_dir)
        logging.info(f"Logging to {log_file}")
        logging.info(f"Training config: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    total_steps = len(dataset) * args.epochs // args.batch_size
    warmup_steps = args.warmup_steps

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return 1.0

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Mixed precision - disabled for bfloat16
    # H200/H100 have native bfloat16 support, GradScaler has issues with bfloat16
    scaler = None

    # Initialize Weights & Biases (only on rank 0)
    if is_main_process and args.wandb and WANDB_AVAILABLE:
        wandb_init_kwargs = {
            "project": args.wandb_project or "p-eagle-training",
            "entity": args.wandb_entity,
            "name": args.wandb_name,
            "config": {
                "hidden_size": config.hidden_size,
                "vocab_size": config.vocab_size,
                "speculation_depth": config.speculation_depth,
                "batch_size": args.batch_size,
                "gradient_accumulation": args.gradient_accumulation,
                "learning_rate": args.lr,
                "epochs": args.epochs,
                "total_steps": total_steps,
                "world_size": world_size,
                "resume_from": args.resume_from,
            }
        }

        # If resuming from a checkpoint, continue the same wandb run
        if args.resume_from and args.wandb_run_id:
            wandb_init_kwargs["id"] = args.wandb_run_id
            wandb_init_kwargs["resume"] = "must"
            print(f"Resuming wandb run: {args.wandb_run_id}")

        wandb.init(**wandb_init_kwargs)
        print(f"Weights & Biases initialized: {wandb.run.url if hasattr(wandb, 'run') else 'N/A'}")
    elif args.wandb and not WANDB_AVAILABLE and is_main_process:
        print("Warning: --wandb specified but wandb is not installed")

    # Training loop
    global_step = 0
    model.train()

    # Memory optimization
    torch.backends.cudnn.benchmark = True
    if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
        torch.cuda.set_per_process_memory_fraction(0.85)

    if is_main_process:
        print(f"\nTraining configuration:")
        print(f"  Epochs: {args.epochs}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Gradient accumulation: {args.gradient_accumulation}")
        print(f"  Learning rate: {args.lr}")
        print(f"  Total steps: {total_steps}")
        print(f"  Samples per epoch: {len(dataset)}")

    # Initialize best validation loss for model selection
    best_val_loss = float('inf')
    best_val_epoch = 0

    # Validation info
    has_validation = args.val_split > 0 and dataset.__len_val__() > 0
    if is_main_process and has_validation:
        print(f"  Validation: {dataset.__len_val__()} samples, eval every {args.val_every} epochs")

    # Overfitting detection initialization
    prev_val_loss = float('inf')
    epochs_without_improvement = 0
    overfitting_warning_issued = False
    train_loss_history = []
    val_loss_history = []

    if is_main_process and has_validation:
        print(f"  Overfitting detection: patience={args.patience}, min_delta={args.min_delta}")

    for epoch in range(args.epochs):
        # Set epoch for distributed sampler to ensure proper shuffling
        if sampler is not None:
            sampler.set_epoch(epoch)

        # No epoch-start barrier needed - DDP handles all synchronization automatically.
        # All ranks are guaranteed to process the same number of iterations due to
        # DistributedSampler + gradient synchronization.

        epoch_loss = 0.0
        epoch_steps = 0
        epoch_start_time = time.time()
        total_tokens_processed = 0

        # Initialize metrics before the loop
        current_ce = 0.0
        current_mse = 0.0
        current_acc = 0.0
        current_mask_pct = 0.0
        current_loss = 0.0
        current_perplexity = 0.0
        grad_norm = 0.0
        # Accumulation variables for epoch averages
        total_ce = 0.0
        total_mse = 0.0
        total_acc = 0.0
        total_p_correct = 0.0
        num_metric_updates = 0

        # Only rank 0 creates the progress bar
        if is_main_process:
            # Show total samples (combined across all GPUs) in the progress bar
            total_combined = len(dataloader) * world_size
            pbar = tqdm(total=total_combined, desc=f"Epoch {epoch+1}/{args.epochs}")
        else:
            pbar = None

        # Reset accumulation counter for each epoch
        accum_counter = 0

        # Training loop - works on all processes, tqdm only on rank 0
        data_iterator = iter(dataloader)

        while True:
            try:
                batch = next(data_iterator)
            except StopIteration:
                break

            input_ids = batch["input_ids"].to(device, dtype=torch.long)
            hidden_states = batch["hidden_states"].to(device)
            target_ids = batch["target_token_ids"].to(device)
            loss_mask = batch["loss_mask"].to(device)  # BUG FIX: Use loss_mask to ignore padding/special tokens

            # Forward pass
            with torch.amp.autocast('cuda', enabled=scaler is not None):
                if args.use_eagle3:
                    # EAGLE-3 aligned training: use EagleDrafterModel forward
                    # This matches inference - MTP heads train on drafter-generated hidden states
                    drafter_outputs = model.forward(
                        input_ids=input_ids,
                        target_hidden=hidden_states,  # Pre-extracted target model hidden states
                        is_training=True  # Use full sequence for training
                    )
                    mtp_predictions = drafter_outputs["mtp_predictions"]

                    # Apply target_lm_head to get logits (same as inference evaluation)
                    predictions = []
                    for i, pred in enumerate(mtp_predictions):
                        if pred is not None:
                            # pred: [batch, seq, target_hidden_dim]
                            # For FULL SEQUENCE training:
                            # - Use ALL positions (0 to L-2) for loss computation
                            # - Position L-1 excluded because target_ids[L-1] = input_ids[0] = BOS (wrong!)
                            # - This is how proper MTP training works
                            logits = model.module.target_lm_head(pred[:, :-1, :])  # [batch, seq-1, vocab]
                            predictions.append(logits)
                        else:
                            predictions.append(None)

                    # Use full-sequence loss function
                    loss, metrics = compute_full_sequence_loss(
                        predictions, target_ids, config,
                        return_metrics=True, loss_mask=loss_mask
                    )
                else:
                    # Original training: use PEagleDrafterPacked forward
                    # PEagleDrafterPacked returns hidden states, so apply target_lm_head to get logits
                    # FIX: Cast hidden_states to float to match model dtype
                    hidden_states = hidden_states.float()
                    raw_predictions = model(hidden_states, return_all_predictions=True,
                                           use_checkpoint=args.gradient_checkpointing)
                    # Convert hidden states to logits using target_lm_head
                    predictions = []
                    for i, pred in enumerate(raw_predictions):
                        if pred is not None:
                            # pred: [batch, 1, hidden_size] -> [batch, 1, vocab_size]
                            # Cast pred to match target_lm_head dtype
                            ddp_model = model.module if isinstance(model, DDP) else model
                            pred = pred.to(ddp_model.target_lm_head.weight.dtype)
                            logits = ddp_model.target_lm_head(pred)
                            predictions.append(logits)
                        else:
                            predictions.append(None)
                    loss, metrics = compute_packed_loss(predictions, target_ids, config,
                                                       return_metrics=True, loss_mask=loss_mask)

            # Update metrics from forward pass
            current_ce = metrics.get('ce', 0.0)
            current_mse = metrics.get('mse', 0.0)
            current_acc = metrics.get('acc', 0.0)
            avg_p_correct = metrics.get('avg_p_correct', 0.0)
            # Accumulate metrics for epoch average
            total_ce += current_ce
            total_mse += current_mse
            total_acc += current_acc
            total_p_correct += avg_p_correct
            num_metric_updates += 1

            # Backward pass
            loss = loss / args.gradient_accumulation

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Increment accumulation counter
            accum_counter += 1

            # Gradient accumulation - optimizer step when counter reaches target
            if accum_counter >= args.gradient_accumulation:
                # Gradient norm
                grad_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.data.norm(2).item() ** 2
                grad_norm = math.sqrt(grad_norm)

                # Unscale before clipping
                if scaler is not None:
                    scaler.unscale_(optimizer)

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Optimizer step
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                accum_counter = 0

                # Log to wandb every step (per-step metrics, not just per-epoch)
                if is_main_process and args.wandb and WANDB_AVAILABLE:
                    wandb.log({
                        'step': global_step,
                        'train_step/loss': loss.item() * args.gradient_accumulation,
                        'train_step/ce': current_ce,
                        'train_step/mse': current_mse,
                        'train_step/accuracy': current_acc,
                        'train_step/p_correct': avg_p_correct,
                        'train_step/lr': scheduler.get_last_lr()[0],
                    }, step=global_step)

                # Update metrics for progress bar
                epoch_loss += loss.item() * args.gradient_accumulation
                epoch_steps += 1
                total_tokens_processed += input_ids.size(0) * input_ids.size(1)

                # Update progress bar
                if pbar is not None:
                    pbar.set_postfix({
                        'loss': f'{loss.item() * args.gradient_accumulation:.4f}',
                        'ce': f'{current_ce:.4f}',
                        'acc': f'{current_acc:.1f}%',
                        'p(corr)': f'{avg_p_correct:.3f}',
                        'lr': f'{scheduler.get_last_lr()[0]:.2e}',
                    })
                    pbar.update(world_size)  # Update by world_size since each sample is counted per GPU

        if pbar is not None:
            pbar.close()

        # End of epoch
        elapsed = time.time() - epoch_start_time
        epoch_avg_loss = epoch_loss / max(epoch_steps, 1)

        if is_main_process:
            print(f"\nEpoch {epoch+1}/{args.epochs} completed in {elapsed:.1f}s")
            print(f"  Average loss: {epoch_avg_loss:.4f}")
            print(f"  Tokens processed: {total_tokens_processed:,}")
            print(f"  Throughput: {total_tokens_processed / elapsed:.0f} tokens/sec")

            # Log to file (use epoch averages, not last batch values)
            if num_metric_updates > 0:
                epoch_ce = total_ce / num_metric_updates
                epoch_mse = total_mse / num_metric_updates
                epoch_acc = total_acc / num_metric_updates
                epoch_p_corr = total_p_correct / num_metric_updates
                logging.info(f"Epoch {epoch+1}: loss={epoch_avg_loss:.4f}, ce={epoch_ce:.4f}, mse={epoch_mse:.4f}, acc={epoch_acc:.1f}%, p(corr)={epoch_p_corr:.3f}")
            else:
                logging.info(f"Epoch {epoch+1}: loss={epoch_avg_loss:.4f}, no metrics recorded")

            # Log to wandb
            if args.wandb and WANDB_AVAILABLE:
                # Compute epoch averages
                avg_train_ce = total_ce / max(num_metric_updates, 1)
                avg_train_mse = total_mse / max(num_metric_updates, 1)
                avg_train_acc = total_acc / max(num_metric_updates, 1)
                avg_train_p_correct = total_p_correct / max(num_metric_updates, 1)

                wandb_log_dict = {
                    'epoch': epoch + 1,
                    'train/loss': epoch_avg_loss,
                    'train/ce': avg_train_ce,
                    'train/mse': avg_train_mse,
                    'train/accuracy': avg_train_acc,
                    'train/p_correct': avg_train_p_correct,
                    'train/throughput_tokens_per_sec': total_tokens_processed / elapsed,
                    'train/epoch_time_sec': elapsed,
                    'lr': scheduler.get_last_lr()[0] if scheduler else 0,
                }

            # Validation (only on main process to avoid duplicate work)
            val_loss = None
            val_acc = None
            val_mse = None
            val_p_correct = None
            if has_validation and (epoch + 1) % args.val_every == 0:
                if is_main_process:
                    val_results = validate(model, dataset, device, config, args=args)
                    val_loss = val_results['val_loss']
                    val_acc = val_results['val_acc']
                    val_mse = val_results['val_mse']
                    val_p_correct = val_results['val_p_correct']

                    print(f"  Validation: loss={val_loss:.4f}, acc={val_acc:.1f}%, p(corr)={val_p_correct:.3f}")
                    logging.info(f"Epoch {epoch+1} validation: loss={val_loss:.4f}, acc={val_acc:.1f}%, p(corr)={val_p_correct:.3f}")

                    # Log validation metrics to wandb
                    if args.wandb and WANDB_AVAILABLE:
                        wandb_log_dict['val/loss'] = val_loss
                        wandb_log_dict['val/accuracy'] = val_acc
                        wandb_log_dict['val/mse'] = val_mse
                        wandb_log_dict['val/p_correct'] = val_p_correct
                        wandb_log_dict['val/best_val_loss'] = best_val_loss

                # Log to wandb (both train and val metrics) at current global_step
                if args.wandb and WANDB_AVAILABLE:
                    wandb.log(wandb_log_dict, step=global_step)

                val_loss_history.append(val_loss)
                train_loss_history.append(epoch_avg_loss)

                # Check for overfitting
                if val_loss > prev_val_loss + args.min_delta:
                    epochs_without_improvement += 1
                    if not overfitting_warning_issued and epochs_without_improvement >= args.patience:
                        print(f"  WARNING: Overfitting detected! Validation loss has not improved for {epochs_without_improvement} epochs")
                        logging.warning(f"Overfitting detected at epoch {epoch+1}")
                        overfitting_warning_issued = True
                else:
                    epochs_without_improvement = 0

                prev_val_loss = val_loss

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_epoch = epoch + 1
                    # Save checkpoint
                    ckpt = {
                        'model_state_dict': model.state_dict() if not isinstance(model, DDP) else model.module.state_dict(),
                        'config': vars(args),
                        'epoch': epoch,
                        'val_loss': val_loss,
                        'optimizer_state_dict': optimizer.state_dict(),
                    }
                    torch.save(ckpt, Path(args.output_dir) / 'best_model.pt')
                    print(f"  Saved best model (val_loss={val_loss:.4f})")
                    logging.info(f"Saved best model at epoch {epoch+1}")

        # Skip barrier entirely - DDP handles gradient synchronization automatically.
        # The epoch-start barrier at line 692 also removed for consistent behavior.

        # Save checkpoint
        if is_main_process and (epoch + 1) % args.save_every == 0:
            ckpt = {
                'model_state_dict': model.state_dict() if not isinstance(model, DDP) else model.module.state_dict(),
                'config': vars(args),
                'epoch': epoch,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'global_step': global_step,
            }
            ckpt_path = Path(args.output_dir) / f'checkpoint_epoch_{epoch+1}.pt'
            torch.save(ckpt, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")
            logging.info(f"Saved checkpoint at epoch {epoch+1}")

    # Final save
    if is_main_process:
        final_ckpt = {
            'model_state_dict': model.state_dict() if not isinstance(model, DDP) else model.module.state_dict(),
            'config': vars(args),
            'epoch': args.epochs,
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step,
        }
        torch.save(final_ckpt, Path(args.output_dir) / 'final_model.pt')
        print(f"\nTraining complete! Final model saved to {args.output_dir}/final_model.pt")
        print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_val_epoch}")

        # Log final metrics
        logging.info(f"Training complete. Best val_loss={best_val_loss:.4f} at epoch {best_val_epoch}")

        if args.wandb and WANDB_AVAILABLE:
            wandb.finish()

    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()


@record
def main():
    parser = argparse.ArgumentParser(description="P-EAGLE Training for Packed Sequences")

    # Data arguments
    parser.add_argument("--feature_dir", type=str, required=True,
                        help="Directory with pre-extracted features")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for checkpoints")
    parser.add_argument("--block_size", type=int, default=4096,
                        help="Sequence block size")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Resume from checkpoint")

    # Model arguments
    parser.add_argument("--hidden_size", type=int, default=2560,
                        help="Hidden dimension size")
    parser.add_argument("--vocab_size", type=int, default=262208,
                        help="Vocabulary size (Gemma3: 262208)")
    parser.add_argument("--speculation_depth", type=int, default=4,
                        help="Number of MTP heads (speculation depth)")

    # EAGLE-3 aligned training (CRITICAL FIX for MAL)
    parser.add_argument("--use_eagle3", action="store_true", default=True,
                        help="Use EAGLE-3 aligned training (MATCHES INFERENCE). "
                             "This fixes MAL=0 by training MTP heads on drafter-generated "
                             "hidden states instead of pre-extracted target model hidden states.")
    parser.add_argument("--no_eagle3", dest="use_eagle3", action="store_false",
                        help="Disable EAGLE-3 aligned training (NOT recommended).")
    parser.add_argument("--base_model", type=str, default='model_cache/gemma-3-4b-it',
                        help="Base model path for EAGLE-3 aligned training. "
                             "Should be the same model used for inference.")

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
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="Max training steps (-1 for full training)")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Use gradient checkpointing to save memory")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate for MTP heads (helps prevent overfitting)")

    # Validation arguments
    parser.add_argument("--val_split", type=float, default=0.05,
                        help="Validation split ratio")
    parser.add_argument("--val_every", type=int, default=1,
                        help="Validate every N epochs")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience")
    parser.add_argument("--min_delta", type=float, default=0.01,
                        help="Minimum improvement for early stopping")

    # Logging arguments
    parser.add_argument("--wandb", action="store_true",
                        help="Use Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="Wandb project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="Wandb entity name")
    parser.add_argument("--wandb_name", type=str, default=None,
                        help="Wandb run name")
    parser.add_argument("--wandb_run_id", type=str, default=None,
                        help="Wandb run ID for resume")
    parser.add_argument("--save_every", type=int, default=1,
                        help="Save checkpoint every N epochs")

    # Misc arguments
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    # Set random seed
    import random
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_packed(args)


if __name__ == "__main__":
    main()