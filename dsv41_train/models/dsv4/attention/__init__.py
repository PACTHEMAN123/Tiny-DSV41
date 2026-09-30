"""DSV4 CSA2 attention modules, reuse state, and Triton kernels."""

from .module import (
    AttentionSinks,
    CSA2Attention,
    CSA2Mode,
    GroupedLinear,
    RMSNorm,
    RotaryEmbedding,
    ShadowIndexers,
    apply_rope,
)
from .sparse_attention import is_available, sparse_attention
from .sparse_indexer import fused_index_scores

__all__ = [
    "AttentionSinks",
    "CSA2Attention",
    "CSA2Mode",
    "GroupedLinear",
    "RMSNorm",
    "RotaryEmbedding",
    "ShadowIndexers",
    "apply_rope",
    "fused_index_scores",
    "is_available",
    "sparse_attention",
]
