#!/usr/bin/env python3
"""
Loss Functions for P-EAGLE Training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def masked_mse_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute masked mean squared error loss.

    Args:
        predictions: [batch, seq_len, hidden_dim]
        targets: [batch, seq_len, hidden_dim]
        mask: [batch, seq_len] - 1 for positions to train on, 0 to ignore

    Returns:
        loss: scalar tensor
    """
    mask_expanded = mask.unsqueeze(-1).float()
    squared_error = (predictions - targets) ** 2
    masked_error = squared_error * mask_expanded

    total_error = masked_error.sum()
    total_tokens = mask.sum()

    if total_tokens > 0:
        loss = total_error / (total_tokens * predictions.shape[-1])
    else:
        # CRITICAL: Empty mask means no learning signal
        import warnings
        warnings.warn("CRITICAL: Loss mask is empty! Model is not learning. Check feature extraction.", RuntimeWarning)
        # Return zero loss that preserves gradient graph
        loss = (predictions * 0).sum()

    return loss


def kl_divergence_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0
) -> torch.Tensor:
    """
    KL divergence loss for knowledge distillation.

    Args:
        student_logits: [batch, seq_len, vocab_size]
        teacher_logits: [batch, seq_len, vocab_size]
        temperature: Softmax temperature

    Returns:
        kl_loss: scalar tensor
    """
    student_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

    kl_loss = F.kl_div(
        student_probs,
        teacher_probs,
        reduction="batchmean"
    ) * (temperature ** 2)

    return kl_loss


def hidden_state_token_loss(
    pred_hidden: torch.Tensor,
    target_hidden: torch.Tensor,
    target_lm_head: nn.Module,
    mask: torch.Tensor,
    temperature: float = 1.0,
    ce_weight: float = 1.0,
    mse_weight: float = 0.1,
    target_token_ids: torch.Tensor = None,
    label_smoothing: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Loss that aligns predicted hidden states with target's token distribution.

    Based on EAGLE paper and official implementation, this uses cross-entropy
    on hard token targets rather than KL divergence on soft distributions.
    Cross-entropy directly optimizes for argmax token matching, which is what
    matters for speculative decoding acceptance.

    P-EAGLE uses the target model's lm_head to convert drafter's predicted
    hidden states to tokens during inference. This loss ensures:
    1. Predicted hidden states produce same tokens as target (via target's lm_head)
    2. Hidden states are close in MSE sense (auxiliary)

    Args:
        pred_hidden: [batch, seq_len, hidden_dim] - drafter predicted hidden states
        target_hidden: [batch, seq_len, hidden_dim] - target model hidden states
        target_lm_head: Target model's lm_head (converts hidden -> logits)
        mask: [batch, seq_len] - 1 for valid positions
        temperature: Softmax temperature (default 1.0, no temperature scaling for CE)
        ce_weight: Weight for cross-entropy loss component (default 1.0)
        mse_weight: Weight for MSE loss component (default 0.1)

    Returns:
        ce_loss: Cross-entropy loss for token prediction
        mse_loss: Mean squared error between hidden states
        accuracy: Token prediction accuracy (%)
    """
    # Guard against NaN/Inf in inputs (can come from bad features or model outputs)
    if not torch.isfinite(pred_hidden).all():
        num_bad = (~torch.isfinite(pred_hidden)).sum().item()
        print(f"WARNING: {num_bad} non-finite values in pred_hidden, clamping")
        pred_hidden = torch.nan_to_num(pred_hidden, nan=0.0, posinf=10.0, neginf=-10.0)
    if not torch.isfinite(target_hidden).all():
        num_bad = (~torch.isfinite(target_hidden)).sum().item()
        print(f"WARNING: {num_bad} non-finite values in target_hidden, clamping")
        target_hidden = torch.nan_to_num(target_hidden, nan=0.0, posinf=10.0, neginf=-10.0)

    # FIX: Removed redundant LayerNorm that was masking genuine prediction errors.
    # The lm_head handles scale differences internally - normalizing both predictions
    # and targets independently made their directions appear identical even when
    # the model's predictions weren't genuinely close to targets.
    # Only normalize if there's an extreme scale mismatch (>10x difference in std)
    pred_std = pred_hidden.std(dim=-1, keepdim=True).clamp(min=1e-6)
    target_std = target_hidden.std(dim=-1, keepdim=True).clamp(min=1e-6)
    scale_ratio = (pred_std / target_std).clamp(0.1, 10.0)

    # Only apply scale correction if there's significant scale mismatch
    if scale_ratio.mean() < 0.5 or scale_ratio.mean() > 2.0:
        target_hidden = target_hidden * scale_ratio

    # Get predicted token distributions from drafter hidden states via TARGET's lm_head
    # This matches inference: drafter hidden -> target lm_head -> tokens
    # Cast to lm_head dtype to handle model/shard dtype mismatch
    lm_head_dtype = next(target_lm_head.parameters()).dtype
    pred_logits = target_lm_head(pred_hidden.to(lm_head_dtype))  # [batch, seq_len, vocab_size]

    # CRITICAL FIX: Use precomputed target token IDs (from actual model logits)
    # instead of computing argmax(lm_head(target_hidden)). This ensures token
    # targets are correct even when target_hidden comes from fused middle layers
    # that are incompatible with the last-layer lm_head.
    if target_token_ids is not None:
        target_tokens = target_token_ids  # [batch, seq_len] — precomputed correct targets
    else:
        # Fallback: compute from target hidden states (only valid if hidden states
        # are from the same layer the lm_head was trained on)
        target_logits = target_lm_head(target_hidden.to(lm_head_dtype))
        target_tokens = target_logits.argmax(dim=-1)

    # Flatten for cross-entropy computation
    pred_logits_flat = pred_logits.reshape(-1, pred_logits.size(-1))
    target_tokens_flat = target_tokens.reshape(-1)
    mask_flat = mask.reshape(-1)

    # Compute cross-entropy loss per token (with label smoothing for regularization)
    ce_loss_per_token = F.cross_entropy(
        pred_logits_flat,
        target_tokens_flat,
        reduction='none',
        label_smoothing=label_smoothing
    )  # [batch * seq_len]

    # Apply mask and average
    ce_loss = (ce_loss_per_token * mask_flat).sum() / (mask.sum() + 1e-8)
    ce_loss = ce_loss * ce_weight

    # Auxiliary MSE loss for hidden state similarity (helps convergence)
    mse_loss = masked_mse_loss(pred_hidden, target_hidden, mask)
    mse_loss = mse_loss * mse_weight

    # Compute token accuracy (hard metric for monitoring)
    pred_tokens = pred_logits.argmax(dim=-1)
    correct = (pred_tokens == target_tokens).float() * mask
    accuracy = correct.sum() / (mask.sum() + 1e-8) * 100  # percentage

    return ce_loss, mse_loss, accuracy
