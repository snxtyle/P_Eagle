"""P-EAGLE Models Module"""

from .peagle_drafter import EagleDrafterModel, EagleMTPHead
from .flash_attention import (
    FLASH_ATTN_AVAILABLE,
    PEFlashAttention,
    TreeFlashAttention,
    get_attention_backend,
)

__all__ = [
    "EagleDrafterModel",
    "EagleMTPHead",
    "FLASH_ATTN_AVAILABLE",
    "PEFlashAttention",
    "TreeFlashAttention",
    "get_attention_backend",
]
