"""The local computation: linear(x, W) + (alpha / rank) * B(A(dropout(x)))."""

import math
from dataclasses import asdict

import torch
from torch import nn
from torch.nn import functional as F

from .config import LoRAConfig


class LoRALinear(nn.Linear):
    """Reuse the base parameters in place, preserving weight and bias names."""

    def __init__(self, base: nn.Linear, config: LoRAConfig) -> None:
        # nn.Linear.__init__ would allocate and initialize a second base weight.
        nn.Module.__init__(self)
        self.in_features, self.out_features = base.in_features, base.out_features
        self.weight = base.weight
        self.bias = base.bias
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)
        self.config = config
        self.scaling = config.alpha / config.rank
        self.lora_a = nn.Parameter(base.weight.new_empty(config.rank, self.in_features))
        self.lora_b = nn.Parameter(base.weight.new_zeros(self.out_features, config.rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer("_unmerged_weight", None, persistent=False)
        self.train(base.training)

    @property
    def merged(self) -> bool:
        return self._unmerged_weight is not None

    def checkpoint_metadata(self) -> dict:
        return asdict(self.config)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        if self.merged:
            # A training state always stores W, A, B, even during merged inference.
            weight = self._unmerged_weight
            destination[prefix + "weight"] = weight if keep_vars else weight.detach()

    def _load_from_state_dict(self, *args, **kwargs):
        self.unmerge()
        super()._load_from_state_dict(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = F.linear(x, self.weight, self.bias)
        if not self.merged:
            update = F.linear(F.linear(self.dropout(x), self.lora_a), self.lora_b)
            result = result + update * self.scaling
        return result

    def train(self, mode: bool = True):
        if mode and self.merged:
            self.unmerge()
        return super().train(mode)

    @torch.no_grad()
    def merge(self) -> None:
        """Fold the adapter into an unsharded inference weight once."""
        if hasattr(self.weight, "to_local"):
            raise ValueError("merge requires an unsharded model")
        if self.training:
            raise ValueError("call eval() before merging adapters")
        if not self.merged:
            # Subtracting a rounded BF16 delta cannot exactly recover the base.
            self._unmerged_weight = self.weight.detach().clone()
            delta = self.lora_b.float() @ self.lora_a.float()
            self.weight.copy_((self.weight.float() + delta * self.scaling).to(self.weight.dtype))

    @torch.no_grad()
    def unmerge(self) -> None:
        """Restore the exact base weight, including for BF16 models."""
        if self.merged:
            self.weight.copy_(self._unmerged_weight)
            self._unmerged_weight = None
