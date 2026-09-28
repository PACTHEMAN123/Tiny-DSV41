"""Adapter hyperparameters and default DSV4 attention targets."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class LoRAConfig:
    """Match exact module paths or dotted suffixes; alpha / rank scales B @ A."""

    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    targets: tuple[str, ...] = (
        "attention.q_a",
        "attention.q_b",
        "attention.kv_proj",
        "attention.o_b",
    )

    def __post_init__(self) -> None:
        if self.rank < 1 or not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive and finite")
        if not 0 <= self.dropout < 1:
            raise ValueError("LoRA dropout must be in [0, 1)")
        if not self.targets or any(not target for target in self.targets):
            raise ValueError("LoRA targets must contain module names")
