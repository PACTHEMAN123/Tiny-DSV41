"""Shared query-LoRA configuration for the native learner and vLLM service."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
import re


SUPPORTED_LORA_RANKS = (8, 16, 32, 64, 128)


def parse_lora_layers(value: str) -> tuple[int, ...] | None:
    """Parse ``all`` or distinct, zero-based layer indices."""
    if not isinstance(value, str):
        raise ValueError("LoRA layers must be 'all' or comma-separated layer indices")
    if value.strip() == "all":
        return None
    parts = [part.strip() for part in value.split(",")]
    if not parts or any(re.fullmatch(r"[0-9]+", part) is None for part in parts):
        raise ValueError("LoRA layers must be 'all' or comma-separated nonnegative indices")
    layers = tuple(int(part) for part in parts)
    if len(set(layers)) != len(layers):
        raise ValueError("LoRA layer indices must be distinct")
    return tuple(sorted(layers))


@dataclass(frozen=True)
class DistributedLoRAConfig:
    """Adapt query down-projections; dropout stays zero for RL probability scoring.

    ``layers=None`` selects every backbone layer. Ranks fit the pinned vLLM
    kernels and the bounded adapter transport for the supported full model.
    """

    rank: int = 8
    alpha: float = 8.0
    layers: tuple[int, ...] | None = None

    def __post_init__(self):
        if type(self.rank) is not int or self.rank not in SUPPORTED_LORA_RANKS:
            raise ValueError(f"LoRA rank must be one of {SUPPORTED_LORA_RANKS}")
        if (type(self.alpha) not in (int, float) or not math.isfinite(self.alpha)
                or self.alpha <= 0):
            raise ValueError("LoRA alpha must be finite and positive")
        if self.layers is not None:
            if (not isinstance(self.layers, tuple) or not self.layers
                    or any(type(layer) is not int or layer < 0 for layer in self.layers)
                    or len(set(self.layers)) != len(self.layers)):
                raise ValueError("LoRA layers must be distinct nonnegative indices or None")
            object.__setattr__(self, "layers", tuple(sorted(self.layers)))
        object.__setattr__(self, "alpha", float(self.alpha))

    def selected_layers(self, num_layers: int) -> tuple[int, ...]:
        if type(num_layers) is not int or num_layers < 1:
            raise ValueError("backbone layer count must be a positive integer")
        layers = tuple(range(num_layers)) if self.layers is None else self.layers
        if layers[-1] >= num_layers:
            raise ValueError(f"LoRA layer index is outside the {num_layers}-layer backbone")
        return layers

    def parameter_count(self, hidden: int, q_rank: int, num_layers: int) -> int:
        if any(type(size) is not int or size < 1 for size in (hidden, q_rank)):
            raise ValueError("query projection dimensions must be positive integers")
        return len(self.selected_layers(num_layers)) * self.rank * (hidden + q_rank)

    def to_dict(self) -> dict:
        return {"rank": self.rank, "alpha": self.alpha,
                "layers": list(self.layers) if self.layers is not None else None}

    @classmethod
    def from_dict(cls, value: Mapping) -> DistributedLoRAConfig:
        if not isinstance(value, Mapping) or set(value) != {"rank", "alpha", "layers"}:
            raise ValueError("LoRA configuration requires exactly rank, alpha and layers")
        layers = value["layers"]
        if layers is not None and not isinstance(layers, list):
            raise ValueError("serialized LoRA layers must be a list or null")
        return cls(rank=value["rank"], alpha=value["alpha"],
                   layers=tuple(layers) if layers is not None else None)
