"""Explicit activation recomputation policies for native training modules."""

from dataclasses import dataclass

import torch
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class ActivationCheckpointConfig:
    mode: str = "none"
    components: tuple[str, ...] = ("attention", "moe")

    def __post_init__(self) -> None:
        if self.mode not in {"none", "full", "selective"}:
            raise ValueError("checkpoint mode must be none, full, or selective")
        if not self.components or set(self.components) - {"attention", "moe", "hc"}:
            raise ValueError("checkpoint components must be attention, moe, or hc")

    def selects(self, component: str) -> bool:
        return self.mode == "selective" and component in self.components


def recompute(enabled: bool, function, *args):
    if enabled and torch.is_grad_enabled():
        return checkpoint(function, *args, use_reentrant=False, preserve_rng_state=True)
    return function(*args)
