#!/usr/bin/env python3
"""
P-EAGLE Drafter Model with Multi-Token Prediction (MTP) Capability

Architecture follows EAGLE-3: Hidden state injection via concatenation.
First layer accepts 2x hidden size input [embeds; target_hidden] and splits
inside the forward pass, matching the vLLM speculators implementation.
"""

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from typing import Dict, List, Optional

# Import Flash Attention
from .flash_attention import (
    FLASH_ATTN_AVAILABLE,
    PEFlashAttention,
    create_eagle3_flash_attention,
    get_attention_backend
)

# CRITICAL: Monkey-patch Gemma3RotaryEmbedding BEFORE any model is loaded
# This fixes the 'layer_type' parameter issue in Gemma3 models
# The patch must be applied at module import time to catch all instantiations
try:
    import transformers.models.gemma3.modeling_gemma3 as _gemma3_module
    _orig_rope_forward = _gemma3_module.Gemma3RotaryEmbedding.forward

    def _patched_rope_forward(self, x, position_ids, layer_type=None, **kwargs):
        """Patched forward that defaults layer_type to 'full_attention'."""
        if layer_type is None:
            layer_type = 'full_attention'
        return _orig_rope_forward(self, x, position_ids, layer_type=layer_type, **kwargs)

    _gemma3_module.Gemma3RotaryEmbedding.forward = _patched_rope_forward
except (ImportError, AttributeError):
    # Gemma3 not available or already patched
    pass

# Fix SDPA dtype mismatch: attention mask dtype must match query dtype
try:
    import transformers.integrations.sdpa_attention as _sdpa_module
    _orig_sdpa_forward = _sdpa_module.sdpa_attention_forward

    def _patched_sdpa_forward(module, query, key, value, attention_mask, dropout=0.0, scaling=None, is_causal=False, **kwargs):
        """Patched SDPA that casts attention_mask to match query dtype."""
        if attention_mask is not None and attention_mask.dtype != query.dtype:
            attention_mask = attention_mask.to(query.dtype)
        return _orig_sdpa_forward(module, query, key, value, attention_mask, dropout, scaling, is_causal, **kwargs)

    _sdpa_module.sdpa_attention_forward = _patched_sdpa_forward
except (ImportError, AttributeError):
    pass

# Also patch PyTorch's F.scaled_dot_product_attention directly
# (some model modules cache the original function reference)
import torch.nn.functional as _F
_orig_torch_sdpa = _F.scaled_dot_product_attention

def _patched_torch_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kwargs):
    if attn_mask is not None and hasattr(attn_mask, 'dtype') and attn_mask.dtype != query.dtype:
        attn_mask = attn_mask.to(query.dtype)
    return _orig_torch_sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kwargs)

_F.scaled_dot_product_attention = _patched_torch_sdpa


class Eagle3FirstLayer(nn.Module):
    """
    Wrapper for the first decoder layer that handles 2x hidden size input.

    EAGLE-3 architecture: Input is concatenated [embeds; target_hidden]
    along the last dimension. This layer splits them, applies separate
    normalizations, and manages the dimension mismatch for residual connections.

    CRITICAL: The attention output is 2x hidden size (from concatenated input),
    but we need to project it back to hidden size for the residual connection
    and subsequent layers.
    """

    def __init__(self, base_layer: nn.Module, hidden_size: int, norm_class: type, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.base_layer = base_layer
        self.hidden_size = hidden_size

        # Preserve layer_type from original layer (needed for Gemma3 rotary embeddings)
        if hasattr(base_layer, 'layer_type'):
            self.layer_type = base_layer.layer_type

        # Add separate norm for the hidden state portion
        self.embed_norm = norm_class(hidden_size)
        self.hidden_norm = norm_class(hidden_size)

        # Output projection: 2x hidden -> hidden (for residual connection compatibility)
        # This is REQUIRED because attention sees 2x input but outputs must match residual
        self.output_proj = nn.Linear(2 * hidden_size, hidden_size, bias=False, dtype=dtype)

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        """
        Forward with concatenated input handling.

        Args:
            hidden_states: Concatenated [embeds; target_hidden] of shape
                          [batch, seq_len, 2*hidden_size]
            *args, **kwargs: Passed to base layer (position_embeddings, attention_mask, etc.)
        """
        # Split concatenated input: [embeds; target_hidden]
        mid = self.hidden_size
        embeds, hidden = hidden_states[..., :mid], hidden_states[..., mid:]

        # Store residual from hidden portion BEFORE any normalization
        residual = hidden

        # Apply norms to each portion separately
        embeds = self.embed_norm(embeds)
        hidden = self.hidden_norm(hidden)

        # Re-concatenate for attention (2x hidden size)
        hidden_states = torch.cat([embeds, hidden], dim=-1)

        # Call base layer's self_attn with concatenated input
        # The base layer's q/k/v projections have been modified to handle 2x hidden size
        attn_kwargs = {
            "hidden_states": hidden_states,
        }

        # Pass through kwargs that self_attn expects (filter out layer-specific ones)
        for key in ["attention_mask", "position_embeddings", "position_ids", "past_key_value"]:
            if key in kwargs:
                attn_kwargs[key] = kwargs[key]

        attn_output = self.base_layer.self_attn(**attn_kwargs)

        # Handle tuple return from self_attn
        if isinstance(attn_output, tuple):
            attn_output = attn_output[0]

        # CRITICAL: Project attention output from 2x hidden -> hidden for residual
        # attn_output is [batch, seq, 2*hidden] because it came from 2x input
        # but we need [batch, seq, hidden] to match residual
        attn_projected = self.output_proj(attn_output)

        # Residual connection: hidden (residual) + attn_projected (now matching dims)
        hidden = residual + attn_projected

        # MLP portion (standard - uses hidden_size throughout)
        residual = hidden
        hidden = self.base_layer.post_attention_layernorm(hidden)
        hidden = self.base_layer.mlp(hidden)
        hidden = residual + hidden

        return hidden


def detect_lora_targets(model: nn.Module) -> List[str]:
    """
    Auto-detect LoRA target modules based on model architecture.
    Works for Llama, Mistral, Gemma, Qwen, Phi, and other LLMs.
    """
    target_patterns = []
    module_names = [name for name, _ in model.named_modules()]

    # Common attention patterns
    attention_targets = {
        "q_proj", "k_proj", "v_proj", "o_proj",           # Llama, Mistral, Gemma, Qwen
        "query", "key", "value", "dense",                  # BERT, GPT-2
        "c_attn", "c_proj",                                # GPT-2 alternate
        "attn.q", "attn.k", "attn.v", "attn.o",            # Some variants
    }

    # Common MLP/FFN patterns
    mlp_targets = {
        "gate_proj", "up_proj", "down_proj",               # Llama, Mistral, Gemma
        "fc1", "fc2",                                       # BERT, some GPT
        "c_fc", "c_proj",                                   # GPT-2
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",    # Qwen
    }

    all_targets = attention_targets | mlp_targets

    for target in all_targets:
        # Check if any module name ends with this target
        for name in module_names:
            if name.endswith(f".{target}") or name == target:
                target_patterns.append(target)
                break

    # Remove duplicates while preserving order
    seen = set()
    unique_targets = []
    for t in target_patterns:
        if t not in seen:
            seen.add(t)
            unique_targets.append(t)

    if not unique_targets:
        # Fallback: common default for most decoder LLMs
        return ["q_proj", "v_proj"]

    return unique_targets


class EagleMTPHead(nn.Module):
    """
    Multi-Token Prediction Head for P-EAGLE.
    Predicts the hidden state at position t+k given the hidden state at position t.
    """

    def __init__(
        self,
        hidden_dim: int,
        target_hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.target_hidden_dim = target_hidden_dim

        layers = []
        in_dim = hidden_dim

        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim, dtype=dtype),
                nn.LayerNorm(hidden_dim, dtype=dtype),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, target_hidden_dim, dtype=dtype))
        self.mlp = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.mlp(hidden_states)


class EagleDrafterModel(nn.Module):
    """
    P-EAGLE Drafter with Multi-Token Prediction capability.

    Architecture (EAGLE-3 style):
    1. Base LLM with FIRST LAYER MODIFIED to accept 2x hidden size input
       [embeds; target_hidden] concatenation
    2. Linear projection to match target dimension
    3. K parallel MTP heads for predicting h_{t+1}, ..., h_{t+K}
    4. Hidden state injection via concatenation (CRITICAL for alignment)
    """

    def __init__(
        self,
        base_model_name: str,
        target_hidden_dim: int,
        speculation_depth: int = 4,
        use_lora: bool = True,
        lora_rank: int = 64,
        lora_alpha: int = 128,
        lora_dropout: float = 0.05,
        device: str = "cuda",
        use_hidden_injection: bool = True,
        injection_mode: str = "concat",
        quantization: str = None,
        use_flash_attention: bool = True,
        mtp_dropout: float = 0.1,  # Dropout in MTP heads for regularization
    ):
        super().__init__()

        self.speculation_depth = speculation_depth
        self.target_hidden_dim = target_hidden_dim
        self.device = device
        self.use_hidden_injection = use_hidden_injection
        self.injection_mode = injection_mode
        self.use_flash_attention = use_flash_attention and FLASH_ATTN_AVAILABLE

        # Log attention backend
        backend = get_attention_backend() if self.use_flash_attention else "standard"
        print(f"Attention backend: {backend}")
        if use_flash_attention and not FLASH_ATTN_AVAILABLE:
            print("  (Flash Attention not available, install with: pip install flash-attn)")
            print("  (Falling back to SDPA/manual attention)")

        # Use local cache to avoid re-downloading
        import os
        cache_dir = os.environ.get("HF_HOME") or os.path.join(os.getcwd(), "models_cache")
        os.makedirs(cache_dir, exist_ok=True)

        print(f"Loading base drafter model: {base_model_name}")
        print(f"Cache directory: {cache_dir}")
        if quantization:
            print(f"Quantization: {quantization}")

        self.config = AutoConfig.from_pretrained(base_model_name, cache_dir=cache_dir)

        # Handle Gemma3-style nested config (multimodal models have text_config)
        if hasattr(self.config, 'text_config') and self.config.text_config is not None:
            self.text_config = self.config.text_config
        else:
            self.text_config = self.config

        # Setup quantization config if requested
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": {"": device} if torch.cuda.is_available() else "cpu",
            "cache_dir": cache_dir
        }

        if quantization == "8bit":
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.bfloat16
            )
            model_kwargs["torch_dtype"] = torch.float16  # 8-bit requires float16
        elif quantization == "4bit":
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            **model_kwargs
        )

        self.hidden_dim = self.text_config.hidden_size
        print(f"Base model hidden dim: {self.hidden_dim}")
        print(f"Target model hidden dim: {target_hidden_dim}")

        # Detect the actual dtype used by the base model for consistent layer initialization
        # 8-bit quant uses float16, 4-bit/non-quant uses bfloat16
        base_dtype = next(self.base_model.parameters()).dtype
        print(f"Base model dtype: {base_dtype}")

        # Normalize all floating-point buffers to the model's dtype
        # Fixes SDPA attention mask dtype mismatches
        for buf in self.base_model.buffers():
            if buf.dtype.is_floating_point and buf.dtype != base_dtype:
                buf.data = buf.data.to(base_dtype)

        # ===== EAGLE-3: MODIFY FIRST LAYER FOR 2x HIDDEN SIZE INPUT =====
        # This is the CRITICAL fix - first layer must accept concatenated [embeds; target_hidden]
        self._modify_first_layer_for_concat_injection(base_dtype)
        # ================================================================

        if use_lora:
            print(f"Applying LoRA (rank={lora_rank}, alpha={lora_alpha})")

            # Auto-detect target modules for this architecture
            target_modules = detect_lora_targets(self.base_model)
            print(f"Detected LoRA targets: {target_modules}")

            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            self.base_model = get_peft_model(self.base_model, lora_config)
            self.base_model.print_trainable_parameters()

        # Initialize projection and heads with same dtype as base model
        self.dim_projection = nn.Linear(self.hidden_dim, target_hidden_dim, dtype=base_dtype).to(device)

        # Hidden state injection: project target hidden to match drafter hidden dim
        # for concatenation at the first layer input
        self.target_hidden_proj = nn.Linear(
            target_hidden_dim,
            self.hidden_dim,
            dtype=base_dtype
        ).to(device)

        self.mtp_heads = nn.ModuleList([
            EagleMTPHead(target_hidden_dim, target_hidden_dim, num_layers=2, dropout=mtp_dropout, dtype=base_dtype).to(device)
            for _ in range(speculation_depth)
        ])
        print(f"MTP dropout: {mtp_dropout}")

        print(f"Initialized {speculation_depth} parallel MTP heads")
        print(f"EAGLE-3 mode: First layer accepts 2x hidden size input via concatenation")

        # Optionally patch remaining layers with Flash Attention
        if self.use_flash_attention:
            self._patch_remaining_layers_with_flash_attention(dtype)

    def _patch_remaining_layers_with_flash_attention(self, dtype: torch.dtype):
        """Patch remaining transformer layers with Flash Attention for full optimization."""
        base_model = self.base_model
        if hasattr(base_model, 'model'):
            base_model = base_model.model

        if not hasattr(base_model, 'layers'):
            return

        num_patched = 0
        for idx in range(1, len(base_model.layers)):  # Skip first layer (already handled)
            layer = base_model.layers[idx]
            if not hasattr(layer, 'self_attn'):
                continue

            attn_module = layer.self_attn

            # Get attention config
            hidden_size = self.hidden_dim
            num_heads = getattr(self.text_config, 'num_attention_heads', 32)
            num_kv_heads = getattr(self.text_config, 'num_key_value_heads', num_heads)
            head_dim = getattr(self.text_config, 'head_dim', hidden_size // num_heads)
            attention_bias = getattr(self.text_config, 'attention_bias', False)

            # Create Flash Attention module
            flash_attn = PEFlashAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_key_value_heads=num_kv_heads,
                head_dim=head_dim,
                bias=attention_bias,
                dtype=dtype,
                device=self.device,
                eagle3_input_mult=1,  # Standard layers have 1x input
                output_mult=1,
            )

            # Copy weights from original attention
            try:
                with torch.no_grad():
                    flash_attn.q_proj.weight.copy_(attn_module.q_proj.weight)
                    flash_attn.k_proj.weight.copy_(attn_module.k_proj.weight)
                    flash_attn.v_proj.weight.copy_(attn_module.v_proj.weight)
                    flash_attn.o_proj.weight.copy_(attn_module.o_proj.weight)

                    # Copy biases if present
                    for dst, src in [
                        (flash_attn.q_proj, attn_module.q_proj),
                        (flash_attn.k_proj, attn_module.k_proj),
                        (flash_attn.v_proj, attn_module.v_proj),
                        (flash_attn.o_proj, attn_module.o_proj),
                    ]:
                        if src.bias is not None and dst.bias is not None:
                            dst.bias.copy_(src.bias)

                layer.self_attn = flash_attn
                num_patched += 1
            except RuntimeError as e:
                print(f"  Warning: Could not patch layer {idx}: {e}")

        if num_patched > 0:
            print(f"  Patched {num_patched} additional layers with Flash Attention")

    def _modify_first_layer_for_concat_injection(self, dtype: torch.dtype):
        """
        Modify the first transformer layer to accept 2x hidden size input.

        Following EAGLE-3 architecture:
        1. Replace q/k/v projections with 2x hidden_size input
        2. Wrap the layer with Eagle3FirstLayer to handle dimension mismatch
           (2x input -> attention -> project back to hidden_size for residual)
        3. Optionally use Flash Attention for optimized computation
        """
        base_model = self.base_model
        if hasattr(base_model, 'model'):
            base_model = base_model.model

        # Get first decoder layer
        if hasattr(base_model, 'layers'):
            first_layer = base_model.layers[0]
        else:
            print("Warning: Could not find layers attribute, skipping first layer modification")
            return

        # Detect model type for proper norm class (use text_config for multimodal models)
        model_type = getattr(self.text_config, 'model_type', '').lower()

        # Get appropriate RMSNorm class
        if 'llama' in model_type:
            from transformers.models.llama.modeling_llama import LlamaRMSNorm as RMSNormClass
        elif 'qwen' in model_type:
            from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm as RMSNormClass
        elif 'mistral' in model_type:
            from transformers.models.mistral.modeling_mistral import MistralRMSNorm as RMSNormClass
        else:
            # Fallback: try to infer from the layer's existing norm
            if hasattr(first_layer, 'input_layernorm'):
                RMSNormClass = type(first_layer.input_layernorm)
            else:
                print(f"Warning: Unknown model type {model_type}, using default RMSNorm")
                from transformers.models.llama.modeling_llama import LlamaRMSNorm as RMSNormClass

        # Store original layer
        self.original_first_layer = first_layer

        # Get attention config
        hidden_size = self.hidden_dim
        num_heads = getattr(self.text_config, 'num_attention_heads', 32)
        num_kv_heads = getattr(self.text_config, 'num_key_value_heads', num_heads)
        head_dim = getattr(self.text_config, 'head_dim', hidden_size // num_heads)
        attention_bias = getattr(self.text_config, 'attention_bias', False)

        # Get the attention module
        attn_module = first_layer.self_attn

        # Use Flash Attention if enabled
        if self.use_flash_attention:
            print(f"  Replacing attention with Flash Attention (EAGLE-3 compatible)")
            flash_attn = PEFlashAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_key_value_heads=num_kv_heads,
                head_dim=head_dim,
                bias=attention_bias,
                dtype=dtype,
                device=self.device,
                eagle3_input_mult=2,  # EAGLE-3: 2x input
                output_mult=2,  # EAGLE-3: 2x output
            )
            first_layer.self_attn = flash_attn
            self.flash_attention_installed = True
        else:
            # Standard attention: Replace q/k/v projections with 2x hidden size versions
            attn_module.q_proj = nn.Linear(
                2 * hidden_size,
                num_heads * head_dim,
                bias=attention_bias,
                dtype=dtype,
                device=self.device
            )
            attn_module.k_proj = nn.Linear(
                2 * hidden_size,
                num_kv_heads * head_dim,
                bias=attention_bias,
                dtype=dtype,
                device=self.device
            )
            attn_module.v_proj = nn.Linear(
                2 * hidden_size,
                num_kv_heads * head_dim,
                bias=attention_bias,
                dtype=dtype,
                device=self.device
            )
            # CRITICAL: Also modify o_proj to output 2x hidden size
            # This ensures attention output can be projected back to hidden_size
            attn_module.o_proj = nn.Linear(
                num_heads * head_dim,
                2 * hidden_size,  # Output 2x to match concatenated dimensions
                bias=attention_bias,
                dtype=dtype,
                device=self.device
            )
            self.flash_attention_installed = False

        # WRAP the first layer with Eagle3FirstLayer
        # This handles the dimension mismatch: 2x input -> attention -> project to hidden
        wrapped_layer = Eagle3FirstLayer(first_layer, hidden_size, RMSNormClass, dtype=dtype)
        wrapped_layer = wrapped_layer.to(dtype=dtype, device=self.device)

        # Replace the layer in the model
        base_model.layers[0] = wrapped_layer

        # Store reference to modified first layer for forward pass
        self.modified_first_layer = wrapped_layer

        # Log that rotary_emb patch was applied at module import time
        if hasattr(base_model, 'rotary_emb'):
            print(f"  Using Gemma3RotaryEmbedding with patched layer_type='full_attention'")

        attn_type = "Flash Attention" if self.use_flash_attention else "Standard Attention"
        print(f"Modified first layer: 2x hidden input ({2 * hidden_size}) with {attn_type}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        target_hidden: Optional[torch.Tensor] = None,
        output_hidden_states: bool = True,
        is_training: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for P-EAGLE drafter with EAGLE-3 style hidden injection.

        CRITICAL: When target_hidden is provided, it's concatenated with input embeddings
        at the first layer: [embeds; target_hidden] along hidden dim (2x size).
        This matches the vLLM EAGLE-3 architecture and is ESSENTIAL for distribution
        alignment between drafter and target model.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            target_hidden: Target model hidden states [batch, seq_len, target_hidden_dim]
                          If provided, enables EAGLE-3 style concatenation injection.
            output_hidden_states: Whether to return hidden states
            is_training: If True, apply trimming for training. If False, use last hidden state for inference.

        Returns:
            Dictionary containing base_hidden, projected_hidden, mtp_predictions, lm_logits
        """
        # Get input embeddings
        inputs_embeds = self.base_model.get_input_embeddings()(input_ids)

        # ===== EAGLE-3: CONCATENATE TARGET HIDDEN AT INPUT =====
        if target_hidden is not None and self.use_hidden_injection:
            # Project target hidden to drafter's hidden dim
            target_proj = self.target_hidden_proj(target_hidden.to(inputs_embeds.dtype))

            # Concatenate: [embeds; target_proj] along last dimension -> 2x hidden size
            inputs_embeds = torch.cat([inputs_embeds, target_proj], dim=-1)

            # The modified first layer will handle splitting and processing
        elif self.use_hidden_injection:
            # Target hidden required but not provided - pad with zeros
            # This happens during inference initial step
            zeros = torch.zeros_like(inputs_embeds)
            inputs_embeds = torch.cat([inputs_embeds, zeros], dim=-1)
        # ======================================================

        # Custom forward through the model
        # We need to manually handle the first layer due to 2x input size
        base_model = self.base_model
        if hasattr(base_model, 'model'):
            base_model = base_model.model

        # Process through embedding + first layer (with concatenated input)
        hidden_states = inputs_embeds

        # Handle attention mask creation if needed
        if attention_mask is not None:
            batch_size, seq_length = input_ids.shape
            # CORRECT causal mask: -inf for future positions (mask OUT), 0 for present/past
            # HF transformers add this mask to attention scores, so -inf -> zero after softmax
            causal_mask = torch.triu(
                torch.full((seq_length, seq_length), float('-inf'), device=hidden_states.device),
                diagonal=1
            )
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
            causal_mask = causal_mask.expand(batch_size, 1, seq_length, seq_length)
        else:
            causal_mask = None

        # Process first layer with EAGLE-3 handling (wrapped with Eagle3FirstLayer)
        # CRITICAL FIX: Try multiple methods to find the first layer
        # Method 1: Use stored reference if available
        if hasattr(self, 'modified_first_layer') and self.modified_first_layer is not None:
            first_layer = self.modified_first_layer
        # Method 2: Navigate through model structure
        elif hasattr(base_model, 'layers'):
            first_layer = base_model.layers[0]
        elif hasattr(self.base_model, 'model') and hasattr(self.base_model.model, 'layers'):
            first_layer = self.base_model.model.layers[0]
        else:
            raise AttributeError(f"Could not find layers attribute. Model type: {type(self.base_model)}")

        # Compute position embeddings for rotary attention (needed for Gemma3, Qwen2, etc.)
        batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1]
        position_ids = torch.arange(seq_len, dtype=torch.long, device=hidden_states.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Get rotary embeddings from model
        # CRITICAL FIX: Try multiple paths to find rotary_emb for different model architectures
        position_embeddings = None
        rotary_emb = None

        # Try different paths where rotary_emb might be located
        if hasattr(base_model, 'rotary_emb'):
            rotary_emb = base_model.rotary_emb
        elif hasattr(self.base_model, 'model') and hasattr(self.base_model.model, 'rotary_emb'):
            rotary_emb = self.base_model.model.rotary_emb
        elif hasattr(self.base_model, 'model') and hasattr(self.base_model.model, 'model') and hasattr(self.base_model.model.model, 'rotary_emb'):
            rotary_emb = self.base_model.model.model.rotary_emb

        if rotary_emb is not None:
            # The rotary_emb was already patched in _modify_first_layer_for_concat_injection
            # to always use layer_type='global', so no need to set it here
            cos, sin = rotary_emb(hidden_states, position_ids)
            position_embeddings = (cos, sin)

        # Call wrapped first layer with position embeddings
        # The Eagle3FirstLayer wrapper handles the 2x input -> hidden output conversion
        hidden_states = first_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_embeddings=position_embeddings
        )

        # Process remaining layers normally
        # Note: First layer output is now [batch, seq, hidden_size] (projected back from 2x)
        # CRITICAL FIX: Ensure we have the correct layers reference for Gemma3 models
        remaining_layers = None
        if hasattr(base_model, 'layers'):
            remaining_layers = base_model.layers[1:]
        elif hasattr(self.base_model, 'model') and hasattr(self.base_model.model, 'layers'):
            remaining_layers = self.base_model.model.layers[1:]
        elif hasattr(self.base_model, 'model') and hasattr(self.base_model.model, 'model') and hasattr(self.base_model.model.model, 'layers'):
            remaining_layers = self.base_model.model.model.layers[1:]

        if remaining_layers is None:
            raise AttributeError(f"Could not find layers to iterate. Model type: {type(self.base_model)}")

        for layer in remaining_layers:
            # Qwen2 layers expect position_embeddings for rotary attention
            layer_kwargs = {"attention_mask": causal_mask}
            if position_embeddings is not None:
                layer_kwargs["position_embeddings"] = position_embeddings

            layer_output = layer(hidden_states, **layer_kwargs)
            # Handle both tuple and tensor returns
            if isinstance(layer_output, tuple):
                hidden_states = layer_output[0]
            else:
                hidden_states = layer_output

        # Final norm - try multiple paths
        if hasattr(base_model, 'norm'):
            hidden_states = base_model.norm(hidden_states)
        elif hasattr(self.base_model, 'model') and hasattr(self.base_model.model, 'norm'):
            hidden_states = self.base_model.model.norm(hidden_states)

        base_hidden = hidden_states

        # Project to target hidden dimension
        projected_hidden = self.dim_projection(base_hidden)

        mtp_predictions = []
        for k, mtp_head in enumerate(self.mtp_heads):
            if is_training:
                # TRAINING: Trim projected_hidden to match shifted targets: h_t predicts h_{t+k+1}
                trim = k + 1
                pred_input = projected_hidden[:, :-trim] if trim > 0 else projected_hidden
            else:
                # INFERENCE: Use the last hidden state to predict future tokens
                pred_input = projected_hidden[:, -1:]

            pred = mtp_head(pred_input)
            mtp_predictions.append(pred)

        return {
            "base_hidden": base_hidden,
            "projected_hidden": projected_hidden,
            "mtp_predictions": mtp_predictions,
            "lm_logits": None  # Not computing logits in this path
        }

    def _eagle3_first_layer_forward(self, layer, hidden_states, attention_mask):
        """
        Custom forward for EAGLE-3 modified first layer.

        Splits concatenated [embeds; target_hidden], applies norms,
        re-concatenates, and passes through modified attention.
        """
        hidden_size = layer._eagle3_hidden_size

        # Split concatenated input
        embeds = hidden_states[..., :hidden_size]
        hidden = hidden_states[..., hidden_size:]

        # Store residual from hidden portion
        residual = hidden

        # Apply norms
        embeds = layer.input_layernorm(embeds)
        hidden = layer.hidden_norm(hidden)

        # Re-concatenate
        hidden_states = torch.cat([embeds, hidden], dim=-1)

        # Compute position embeddings for rotary attention (Gemma3 requires this)
        batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1]
        position_ids = torch.arange(seq_len, dtype=torch.long, device=hidden_states.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Get rotary embeddings - try multiple locations (different models store them differently)
        position_embeddings = None
        rotary_emb = None

        # Try to find rotary_emb in various locations
        if hasattr(layer.self_attn, 'rotary_emb'):
            rotary_emb = layer.self_attn.rotary_emb
        elif hasattr(self, 'base_model'):
            # Try model-level rotary_emb
            base = self.base_model
            if hasattr(base, 'rotary_emb'):
                rotary_emb = base.rotary_emb
            elif hasattr(base, 'model') and hasattr(base.model, 'rotary_emb'):
                rotary_emb = base.model.rotary_emb

        if rotary_emb is not None:
            # Gemma3RotaryEmbedding needs layer_type context
            # Try to get it from the layer or use default
            layer_type = getattr(layer, 'layer_type', 'global')
            # Temporarily set layer_type on rotary_emb if needed
            original_layer_type = getattr(rotary_emb, 'layer_type', None)
            rotary_emb.layer_type = layer_type
            try:
                cos, sin = rotary_emb(hidden_states, position_ids)
                position_embeddings = (cos, sin)
            finally:
                # Restore original layer_type
                if original_layer_type is not None:
                    rotary_emb.layer_type = original_layer_type
                else:
                    delattr(rotary_emb, 'layer_type')

        # Self attention with expanded projections (already modified to 2x input)
        attn_kwargs = {
            "hidden_states": hidden_states,
            "attention_mask": attention_mask,
        }
        if position_embeddings is not None:
            attn_kwargs["position_embeddings"] = position_embeddings

        attn_output = layer.self_attn(**attn_kwargs)
        if isinstance(attn_output, tuple):
            attn_output = attn_output[0]

        # Residual connection
        hidden = residual + attn_output

        # MLP
        residual = hidden
        hidden = layer.post_attention_layernorm(hidden)
        hidden = layer.mlp(hidden)
        hidden = residual + hidden

        return hidden

    def get_predicted_hidden(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        num_tokens: int = 1
    ) -> torch.Tensor:
        assert 1 <= num_tokens <= self.speculation_depth
        outputs = self.forward(input_ids, attention_mask, is_training=False)
        return outputs["mtp_predictions"][num_tokens - 1]

    def save_checkpoint(self, checkpoint_dir: str, target_lm_head: torch.nn.Module = None):
        """Save model checkpoint."""
        import json
        from pathlib import Path

        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        if isinstance(self.base_model, PeftModel):
            self.base_model.save_pretrained(Path(checkpoint_dir) / "lora_weights")

        checkpoint_data = {
            "dim_projection": self.dim_projection.state_dict(),
            "mtp_heads": [head.state_dict() for head in self.mtp_heads],
            "speculation_depth": self.speculation_depth,
            "hidden_dim": self.hidden_dim,
            "target_hidden_dim": self.target_hidden_dim,
            "use_hidden_injection": self.use_hidden_injection,
            "injection_mode": self.injection_mode
        }

        # Save hidden injection layer if enabled
        if self.use_hidden_injection:
            checkpoint_data["target_hidden_proj"] = self.target_hidden_proj.state_dict()

        # Save target lm_head if provided (for vocab compatibility during inference)
        if target_lm_head is not None:
            checkpoint_data["target_lm_head"] = target_lm_head.state_dict()

        torch.save(checkpoint_data, Path(checkpoint_dir) / "eagle_heads.pt")

        config = {
            "base_model": self.base_model.name_or_path if hasattr(self.base_model, 'name_or_path') else "unknown",
            "speculation_depth": self.speculation_depth,
            "hidden_dim": self.hidden_dim,
            "target_hidden_dim": self.target_hidden_dim,
            "use_hidden_injection": self.use_hidden_injection,
            "injection_mode": self.injection_mode,
            "use_flash_attention": self.use_flash_attention,
            "vocab_size": target_lm_head.weight.shape[0] if target_lm_head is not None else None
        }
        with open(Path(checkpoint_dir) / "config.json", "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load_checkpoint(cls, checkpoint_dir: str, device: str = "cuda"):
        """Load model from checkpoint."""
        import json
        from pathlib import Path

        with open(Path(checkpoint_dir) / "config.json") as f:
            config = json.load(f)

        # Load hidden injection settings from checkpoint (default to False for backward compat)
        use_hidden_injection = config.get("use_hidden_injection", False)
        injection_mode = config.get("injection_mode", "concat")
        use_flash_attention = config.get("use_flash_attention", True)  # Default to True for new models

        # Check if LoRA weights exist
        lora_weights_path = Path(checkpoint_dir) / "lora_weights"
        has_lora = lora_weights_path.exists()

        # Initialize WITHOUT LoRA first (load base model)
        model = cls(
            base_model_name=config["base_model"],
            target_hidden_dim=config["target_hidden_dim"],
            speculation_depth=config["speculation_depth"],
            use_lora=False,  # Start with base model
            device=device,
            use_hidden_injection=use_hidden_injection,
            injection_mode=injection_mode,
            use_flash_attention=use_flash_attention,
        )

        # Load LoRA weights if they exist
        if has_lora:
            print(f"Loading LoRA weights from {lora_weights_path}")
            from peft import PeftModel
            # Wrap base model with LoRA weights
            model.base_model = PeftModel.from_pretrained(
                model.base_model,
                str(lora_weights_path),
                is_trainable=False
            )
            print("LoRA weights loaded successfully")

        checkpoint = torch.load(
            Path(checkpoint_dir) / "eagle_heads.pt",
            map_location=device
        )

        model.dim_projection.load_state_dict(checkpoint["dim_projection"])
        for i, head_state in enumerate(checkpoint["mtp_heads"]):
            model.mtp_heads[i].load_state_dict(head_state)

        # Load hidden injection layer if present
        if use_hidden_injection and "target_hidden_proj" in checkpoint:
            model.target_hidden_proj.load_state_dict(checkpoint["target_hidden_proj"])

        # Load target lm_head if present (for vocab compatibility)
        if "target_lm_head" in checkpoint:
            import torch.nn as nn
            lm_head_state = checkpoint["target_lm_head"]
            vocab_size = lm_head_state["weight"].shape[0]
            hidden_size = lm_head_state["weight"].shape[1]

            model.target_lm_head = nn.Linear(
                hidden_size, vocab_size, bias=False, dtype=torch.bfloat16, device=device
            )
            model.target_lm_head.load_state_dict(lm_head_state)
            model.target_lm_head.eval()
            print(f"Loaded target lm_head from checkpoint: {hidden_size} -> {vocab_size}")

        return model
