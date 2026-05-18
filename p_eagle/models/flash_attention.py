#!/usr/bin/env python3
"""
Flash Attention Integration for P-EAGLE

Provides optimized attention mechanisms using:
1. Flash Attention 2 (when available) - fastest for training
2. PyTorch SDPA (scaled_dot_product_attention) - good fallback
3. Manual attention - baseline fallback

Handles EAGLE-3's 2x hidden size concatenated inputs and tree attention masks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, Literal
import math


# Try to import Flash Attention
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False


class FlashAttentionConfig:
    """Configuration for Flash Attention behavior."""

    def __init__(
        self,
        use_flash_attn: bool = True,
        use_sdpa: bool = True,
        attention_dropout: float = 0.0,
        softmax_scale: Optional[float] = None,
        causal: bool = True,
        window_size: Tuple[int, int] = (-1, -1),  # -1 means infinite
        alibi_slopes: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ):
        self.use_flash_attn = use_flash_attn and FLASH_ATTN_AVAILABLE
        self.use_sdpa = use_sdpa
        self.attention_dropout = attention_dropout
        self.softmax_scale = softmax_scale
        self.causal = causal
        self.window_size = window_size
        self.alibi_slopes = alibi_slopes
        self.deterministic = deterministic


class PEFlashAttention(nn.Module):
    """
    Flash Attention module for P-EAGLE with EAGLE-3 support.

    Handles both standard attention and EAGLE-3's 2x hidden size concatenated inputs.
    Automatically selects the best available attention implementation.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_key_value_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
        dtype: torch.dtype = torch.float16,
        device: Optional[Union[str, torch.device]] = None,
        # EAGLE-3 specific
        eagle3_input_mult: int = 1,  # 2 for EAGLE-3 first layer, 1 for standard
        output_mult: int = 1,  # 2 for EAGLE-3 first layer output
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads or num_heads
        self.num_key_value_groups = num_heads // self.num_key_value_heads
        self.head_dim = head_dim or (hidden_size // num_heads)
        self.dropout = dropout
        self.eagle3_input_mult = eagle3_input_mult
        self.output_mult = output_mult

        input_size = hidden_size * eagle3_input_mult
        output_size = self.head_dim * num_heads
        output_proj_size = hidden_size * output_mult

        # Q/K/V projections
        self.q_proj = nn.Linear(
            input_size, num_heads * self.head_dim,
            bias=bias, dtype=dtype, device=device
        )
        self.k_proj = nn.Linear(
            input_size, self.num_key_value_heads * self.head_dim,
            bias=bias, dtype=dtype, device=device
        )
        self.v_proj = nn.Linear(
            input_size, self.num_key_value_heads * self.head_dim,
            bias=bias, dtype=dtype, device=device
        )
        self.o_proj = nn.Linear(
            output_size, output_proj_size,
            bias=bias, dtype=dtype, device=device
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights following standard practices."""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        causal: bool = True,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor]]:
        """
        Forward pass with automatic backend selection.

        Args:
            hidden_states: [batch, seq_len, hidden_size * eagle3_input_mult]
            attention_mask: Optional attention mask
            position_ids: Optional position IDs for RoPE
            past_key_value: Optional cached KV for generation
            use_cache: Whether to return KV cache
            output_attentions: Whether to return attention weights (forces manual attn)
            causal: Whether to use causal masking

        Returns:
            (output, past_key_value, attn_weights)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Compute Q/K/V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape for attention: [batch, seq, num_heads, head_dim]
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim)
        key_states = key_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)

        # Apply RoPE if position_ids provided (handled externally for flexibility)

        # Handle KV cache
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=1)
            value_states = torch.cat([past_key_value[1], value_states], dim=1)

        past_key_value_out = (key_states, value_states) if use_cache else None

        # Expand key/value heads for GQA
        if self.num_key_value_groups > 1:
            key_states = self._repeat_kv(key_states, self.num_key_value_groups)
            value_states = self._repeat_kv(value_states, self.num_key_value_groups)

        # Select and execute attention
        if output_attentions:
            # Must use manual attention to get weights
            attn_output, attn_weights = self._manual_attention(
                query_states, key_states, value_states, attention_mask, causal
            )
        elif FLASH_ATTN_AVAILABLE and hidden_states.is_cuda:
            attn_output, attn_weights = self._flash_attention(
                query_states, key_states, value_states, causal
            )
        else:
            attn_output, attn_weights = self._sdpa_attention(
                query_states, key_states, value_states, attention_mask, causal
            )

        # Reshape and project output
        attn_output = attn_output.reshape(batch_size, seq_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output, past_key_value_out, attn_weights

    def _repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Repeat key/value heads for Group Query Attention."""
        batch, seq_len, n_kv_heads, head_dim = x.shape
        if n_rep == 1:
            return x
        return (
            x[:, :, :, None, :]
            .expand(batch, seq_len, n_kv_heads, n_rep, head_dim)
            .reshape(batch, seq_len, n_kv_heads * n_rep, head_dim)
        )

    def _flash_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        causal: bool = True,
    ) -> Tuple[torch.Tensor, None]:
        """
        Flash Attention 2 implementation.

        Args:
            query: [batch, seq, num_heads, head_dim]
            key: [batch, seq, num_heads, head_dim]
            value: [batch, seq, num_heads, head_dim]
            causal: Whether to use causal masking

        Returns:
            (output, None) - attention weights not returned by Flash Attn
        """
        # Flash attention expects [batch, seq, num_heads, head_dim]
        # Ensure contiguous memory layout
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()

        # Cast to fp16/bf16 as required by flash_attn
        orig_dtype = query.dtype
        if orig_dtype not in [torch.float16, torch.bfloat16]:
            query = query.half()
            key = key.half()
            value = value.half()

        try:
            output = flash_attn_func(
                query, key, value,
                dropout_p=self.dropout if self.training else 0.0,
                softmax_scale=self.head_dim ** -0.5,
                causal=causal,
                window_size=(-1, -1),
                alibi_slopes=None,
                deterministic=False,
                return_attn_probs=False,
            )
        except Exception as e:
            # Fallback to SDPA on any flash attn error
            output, _ = self._sdpa_attention(query, key, value, None, causal)

        if output.dtype != orig_dtype:
            output = output.to(orig_dtype)

        return output, None

    def _sdpa_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        causal: bool = True,
    ) -> Tuple[torch.Tensor, None]:
        """
        PyTorch Scaled Dot-Product Attention.

        Uses efficient fused kernels when available (cuDNN, FlashAttention via PyTorch).
        """
        # SDPA expects [batch, num_heads, seq, head_dim]
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Determine if we can use efficient kernels
        # Efficient kernels require no attention mask or causal-only
        use_efficient = causal and attention_mask is None

        if use_efficient:
            attn_output = F.scaled_dot_product_attention(
                query, key, value,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            # Manual attention with mask
            scale = self.head_dim ** -0.5
            scores = torch.matmul(query, key.transpose(-2, -1)) * scale

            if attention_mask is not None:
                # Handle different mask formats
                if attention_mask.dim() == 2:
                    attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
                elif attention_mask.dim() == 3:
                    attention_mask = attention_mask.unsqueeze(1)
                scores = scores.masked_fill(~attention_mask.bool(), float('-inf'))
            elif causal:
                # Create causal mask
                seq_len = query.size(2)
                causal_mask = torch.triu(
                    torch.ones(seq_len, seq_len, device=query.device, dtype=torch.bool),
                    diagonal=1
                )
                scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

            attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
            if self.training and self.dropout > 0:
                attn_weights = F.dropout(attn_weights, p=self.dropout)
            attn_output = torch.matmul(attn_weights, value)

        # Transpose back to [batch, seq, num_heads, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()

        return attn_output, None

    def _manual_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        causal: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Manual attention computation - returns attention weights."""
        # [batch, num_heads, seq, head_dim]
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        scale = self.head_dim ** -0.5
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
            scores = scores.masked_fill(~attention_mask.bool(), float('-inf'))
        elif causal:
            seq_len = query.size(2)
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=query.device, dtype=torch.bool),
                diagonal=1
            )
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        if self.training and self.dropout > 0:
            attn_weights = F.dropout(attn_weights, p=self.dropout)

        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous()

        return attn_output, attn_weights


class TreeFlashAttention(nn.Module):
    """
    Flash Attention optimized for tree-structured speculative decoding.

    Handles the non-causal tree masks efficiently by using block-sparse patterns
    or falling back to SDPA when needed.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_key_value_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
        dtype: torch.dtype = torch.float16,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads or num_heads
        self.head_dim = head_dim or (hidden_size // num_heads)
        self.dropout = dropout

        # Standard projections
        self.q_proj = nn.Linear(
            hidden_size, num_heads * self.head_dim,
            bias=bias, dtype=dtype, device=device
        )
        self.k_proj = nn.Linear(
            hidden_size, self.num_key_value_heads * self.head_dim,
            bias=bias, dtype=dtype, device=device
        )
        self.v_proj = nn.Linear(
            hidden_size, self.num_key_value_heads * self.head_dim,
            bias=bias, dtype=dtype, device=device
        )
        self.o_proj = nn.Linear(
            num_heads * self.head_dim, hidden_size,
            bias=bias, dtype=dtype, device=device
        )

    def forward_tree(
        self,
        hidden_states: torch.Tensor,
        tree_mask: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward with tree attention mask.

        Args:
            hidden_states: [batch, total_seq, hidden]
            tree_mask: [batch, total_seq, total_seq] boolean mask where True=can attend
            position_ids: Optional position IDs

        Returns:
            output: [batch, total_seq, hidden]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Compute Q/K/V
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # Reshape
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Expand for GQA
        if self.num_heads != self.num_key_value_heads:
            n_rep = self.num_heads // self.num_key_value_heads
            key = key.repeat_interleave(n_rep, dim=1)
            value = value.repeat_interleave(n_rep, dim=1)

        # For tree masks, we need manual attention (Flash Attn doesn't support arbitrary masks)
        scale = self.head_dim ** -0.5
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale

        # Apply tree mask
        if tree_mask.dim() == 3:
            tree_mask = tree_mask.unsqueeze(1)  # [batch, 1, seq, seq]

        # Convert boolean mask: True=keep, False=mask_out
        scores = scores.masked_fill(~tree_mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_output = torch.matmul(attn_weights, value)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        output = self.o_proj(attn_output)

        return output

    def forward_causal(
        self,
        hidden_states: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Standard causal attention using Flash Attention when possible."""
        batch_size, seq_len, _ = hidden_states.shape

        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # Reshape for flash: [batch, seq, num_heads, head_dim]
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim)
        key = key.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        value = value.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)

        if past_key_value is not None:
            key = torch.cat([past_key_value[0], key], dim=1)
            value = torch.cat([past_key_value[1], value], dim=1)

        past_key_value_out = (key, value) if use_cache else None

        # Try Flash Attention first
        if FLASH_ATTN_AVAILABLE and hidden_states.is_cuda and query.dtype in [torch.float16, torch.bfloat16]:
            # Expand KV for GQA
            if self.num_heads != self.num_key_value_heads:
                n_rep = self.num_heads // self.num_key_value_heads
                key = key.repeat_interleave(n_rep, dim=2)
                value = value.repeat_interleave(n_rep, dim=2)

            output = flash_attn_func(
                query.contiguous(), key.contiguous(), value.contiguous(),
                causal=True, dropout_p=self.dropout if self.training else 0.0
            )
        else:
            # SDPA fallback
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)

            if self.num_heads != self.num_key_value_heads:
                n_rep = self.num_heads // self.num_key_value_heads
                key = key.repeat_interleave(n_rep, dim=1)
                value = value.repeat_interleave(n_rep, dim=1)

            output = F.scaled_dot_product_attention(
                query, key, value, is_causal=True,
                dropout_p=self.dropout if self.training else 0.0
            )
            output = output.transpose(1, 2)

        output = output.reshape(batch_size, seq_len, -1)
        output = self.o_proj(output)

        return output, past_key_value_out


def create_eagle3_flash_attention(
    base_attn_module: nn.Module,
    hidden_size: int,
    num_heads: int,
    num_key_value_heads: Optional[int] = None,
    head_dim: Optional[int] = None,
    dtype: torch.dtype = torch.float16,
    device: Optional[Union[str, torch.device]] = None,
) -> PEFlashAttention:
    """
    Create a Flash Attention module configured for EAGLE-3 first layer.

    Args:
        base_attn_module: Original attention module (to copy weights from)
        hidden_size: Hidden dimension
        num_heads: Number of attention heads
        num_key_value_heads: Number of key/value heads (for GQA)
        head_dim: Dimension per head
        dtype: Data type
        device: Device

    Returns:
        Configured PEFlashAttention module
    """
    flash_attn = PEFlashAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
        eagle3_input_mult=2,  # EAGLE-3: 2x input
        output_mult=2,  # EAGLE-3: 2x output
    )

    # Copy weights from base attention if shapes match
    if hasattr(base_attn_module, 'q_proj'):
        try:
            # For EAGLE-3, the base module was already modified for 2x input
            with torch.no_grad():
                flash_attn.q_proj.weight.copy_(base_attn_module.q_proj.weight)
                flash_attn.k_proj.weight.copy_(base_attn_module.k_proj.weight)
                flash_attn.v_proj.weight.copy_(base_attn_module.v_proj.weight)
                flash_attn.o_proj.weight.copy_(base_attn_module.o_proj.weight)

                # Copy biases if present
                for dst, src in [
                    (flash_attn.q_proj, base_attn_module.q_proj),
                    (flash_attn.k_proj, base_attn_module.k_proj),
                    (flash_attn.v_proj, base_attn_module.v_proj),
                    (flash_attn.o_proj, base_attn_module.o_proj),
                ]:
                    if src.bias is not None and dst.bias is not None:
                        dst.bias.copy_(src.bias)
        except RuntimeError as e:
            print(f"Warning: Could not copy weights to Flash Attention: {e}")

    return flash_attn


def patch_model_with_flash_attention(
    model: nn.Module,
    use_flash_attn: bool = True,
    target_layers: Optional[list] = None,
) -> nn.Module:
    """
    Patch a model to use Flash Attention.

    Args:
        model: The model to patch
        use_flash_attn: Whether to use Flash Attention
        target_layers: Specific layer indices to patch (None = all)

    Returns:
        Patched model
    """
    if not use_flash_attn:
        return model

    if not FLASH_ATTN_AVAILABLE:
        print("Flash Attention not available. Install with: pip install flash-attn")
        print("Falling back to SDPA (PyTorch scaled_dot_product_attention)")

    # Find model layers
    base_model = model
    if hasattr(model, 'model'):
        base_model = model.model

    if not hasattr(base_model, 'layers'):
        print("Could not find layers to patch")
        return model

    layers = base_model.layers
    layers_to_patch = target_layers if target_layers else range(len(layers))

    for idx in layers_to_patch:
        if idx >= len(layers):
            continue

        layer = layers[idx]
        if not hasattr(layer, 'self_attn'):
            continue

        attn = layer.self_attn

        # Get attention config
        hidden_size = attn.q_proj.in_features
        num_heads = getattr(attn, 'num_heads', None)
        if num_heads is None:
            # Infer from weight shape
            num_heads = attn.q_proj.out_features // (hidden_size // 32)  # rough estimate

        num_key_value_heads = getattr(attn, 'num_key_value_heads', num_heads)
        head_dim = getattr(attn, 'head_dim', hidden_size // num_heads)

        # Create flash attention
        flash_attn = PEFlashAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            dtype=next(attn.parameters()).dtype,
            device=next(attn.parameters()).device,
            eagle3_input_mult=1,
            output_mult=1,
        )

        # Copy weights
        with torch.no_grad():
            flash_attn.q_proj.weight.copy_(attn.q_proj.weight)
            flash_attn.k_proj.weight.copy_(attn.k_proj.weight)
            flash_attn.v_proj.weight.copy_(attn.v_proj.weight)
            flash_attn.o_proj.weight.copy_(attn.o_proj.weight)

        # Replace attention module
        layer.self_attn = flash_attn
        print(f"Patched layer {idx} with Flash Attention")

    return model


def get_attention_backend() -> str:
    """Get the best available attention backend."""
    if FLASH_ATTN_AVAILABLE:
        return "flash_attn"
    elif hasattr(F, 'scaled_dot_product_attention'):
        return "sdpa"
    else:
        return "manual"


# Export availability flag
__all__ = [
    'FLASH_ATTN_AVAILABLE',
    'PEFlashAttention',
    'TreeFlashAttention',
    'FlashAttentionConfig',
    'create_eagle3_flash_attention',
    'patch_model_with_flash_attention',
    'get_attention_backend',
]
