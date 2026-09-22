"""Configuration for the Qwen3-30B-A3B model."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass
class Qwen3MoeConfig:
    """Qwen3-30B-A3B architecture values published by Qwen."""

    architectures: list[str] = field(default_factory=lambda: ["Qwen3MoeForCausalLM"])
    attention_bias: bool = False
    attention_dropout: float = 0.0
    bos_token_id: int = 151643
    decoder_sparse_step: int = 1
    eos_token_id: int = 151645
    head_dim: int = 128
    hidden_act: str = "silu"
    hidden_size: int = 2048
    initializer_range: float = 0.02
    intermediate_size: int = 6144
    max_position_embeddings: int = 40960
    max_window_layers: int = 48
    mlp_only_layers: list[int] = field(default_factory=list)
    model_type: str = "qwen3_moe"
    moe_intermediate_size: int = 768
    norm_topk_prob: bool = True
    num_attention_heads: int = 32
    num_experts: int = 128
    num_experts_per_tok: int = 8
    num_hidden_layers: int = 48
    num_key_value_heads: int = 4
    output_router_logits: bool = False
    rms_norm_eps: float = 1.0e-6
    rope_scaling: dict[str, Any] | None = None
    rope_theta: float = 1_000_000.0
    router_aux_loss_coef: float = 0.001
    sliding_window: int | None = None
    tie_word_embeddings: bool = False
    torch_dtype: str = "bfloat16"
    transformers_version: str = "4.51.0"
    use_cache: bool = True
    use_sliding_window: bool = False
    vocab_size: int = 151936

    def __post_init__(self) -> None:
        self.architectures = list(self.architectures)
        self.mlp_only_layers = list(self.mlp_only_layers)
        self._validate()

    def _validate(self) -> None:
        positive = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "moe_intermediate_size": self.moe_intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "num_experts": self.num_experts,
            "decoder_sparse_step": self.decoder_sparse_step,
            "max_position_embeddings": self.max_position_embeddings,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"{', '.join(invalid)} must be positive")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_key_value_heads must divide num_attention_heads")
        if not 1 <= self.num_experts_per_tok <= self.num_experts:
            raise ValueError("num_experts_per_tok must be in [1, num_experts]")
        if not 0 <= self.max_window_layers <= self.num_hidden_layers:
            raise ValueError("max_window_layers must be in [0, num_hidden_layers]")
        if any(not 0 <= layer < self.num_hidden_layers for layer in self.mlp_only_layers):
            raise ValueError("mlp_only_layers contains an invalid layer index")
        for name in ("bos_token_id", "eos_token_id"):
            token_id = getattr(self, name)
            if not 0 <= token_id < self.vocab_size:
                raise ValueError(f"{name} must be in [0, vocab_size)")
        if self.use_sliding_window and (
            self.sliding_window is None or self.sliding_window <= 0
        ):
            raise ValueError("sliding_window must be positive when use_sliding_window is enabled")

    @classmethod
    def from_json(cls, path: str | Path) -> "Qwen3MoeConfig":
        with Path(path).open(encoding="utf-8") as file:
            values = json.load(file)
        known = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in known})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = ["Qwen3MoeConfig"]
