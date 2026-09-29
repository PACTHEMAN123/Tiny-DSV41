"""Native linear adapters, with base checkpoint names preserved."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class LoRAConfig:
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    targets: tuple[str, ...] = (
        "attention.q_a", "attention.q_b", "attention.kv_proj", "attention.o_b",
    )

    def __post_init__(self) -> None:
        if self.rank < 1 or not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive and finite")
        if not 0 <= self.dropout < 1:
            raise ValueError("LoRA dropout must be in [0, 1)")
        if not self.targets or any(not target for target in self.targets):
            raise ValueError("LoRA targets must contain module names")


class LoRALinear(nn.Linear):
    def __init__(self, base: nn.Linear, config: LoRAConfig) -> None:
        # Do not allocate or initialize a second copy of the base weight.
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
        if hasattr(self.weight, "to_local"):
            raise ValueError("merge requires an unsharded model")
        if self.training:
            raise ValueError("call eval() before merging adapters")
        if not self.merged:
            self._unmerged_weight = self.weight.detach().clone()
            delta = self.lora_b.float() @ self.lora_a.float()
            self.weight.copy_((self.weight.float() + delta * self.scaling).to(self.weight.dtype))

    @torch.no_grad()
    def unmerge(self) -> None:
        if self.merged:
            self.weight.copy_(self._unmerged_weight)
            self._unmerged_weight = None


def inject_lora(model: nn.Module, config: LoRAConfig) -> list[str]:
    """Validate all targets before freezing and replacing matched linear modules."""
    if any(isinstance(module, LoRALinear) for module in model.modules()):
        raise ValueError("LoRA is already installed")
    selected: dict[str, nn.Linear] = {}
    modules = dict(model.named_modules())
    for target in config.targets:
        matches = [(name, module) for name, module in modules.items()
                   if name == target or name.endswith("." + target)]
        if not matches:
            raise ValueError(f"LoRA target did not match a module: {target}")
        for name, module in matches:
            if not isinstance(module, nn.Linear):
                raise ValueError(f"LoRA only supports nn.Linear targets: {name}")
            if module.weight.is_meta or hasattr(module.weight, "to_local"):
                raise ValueError("inject LoRA after loading and before FSDP wrapping")
            selected[name] = module
    model.requires_grad_(False)
    for name, module in selected.items():
        parent_name, _, leaf = name.rpartition(".")
        setattr(model.get_submodule(parent_name), leaf, LoRALinear(module, config))
    return list(selected)


def adapter_parameters(model: nn.Module):
    for module in model.modules():
        if isinstance(module, LoRALinear):
            yield module.lora_a
            yield module.lora_b


def _adapter_modules(model: nn.Module) -> dict[str, LoRALinear]:
    modules = {name: module for name, module in model.named_modules()
               if isinstance(module, LoRALinear)}
    if not modules:
        raise ValueError("model has no LoRA adapters")
    return modules


@torch.no_grad()
def save_adapter(model: nn.Module, path: str | Path) -> None:
    """All ranks participate; only rank zero writes the portable adapter file."""
    modules = _adapter_modules(model)
    weights = {}
    for name, module in modules.items():
        for key in ("lora_a", "lora_b"):
            tensor = getattr(module, key).detach()
            if hasattr(tensor, "full_tensor"):
                tensor = tensor.full_tensor()
            weights[f"{name}.{key}"] = tensor.cpu()
    if not dist.is_initialized() or dist.get_rank() == 0:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"format_version": 1,
                    "layer_ids": getattr(getattr(model, "model", None), "layer_ids", None),
                    "modules": {name: asdict(module.config) for name, module in modules.items()},
                    "weights": weights}, path)


def read_adapter(path: str | Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("unsupported adapter format")
    return payload


def validate_adapter_layout(model: nn.Module, payload: dict) -> None:
    layer_ids = getattr(getattr(model, "model", None), "layer_ids", None)
    if payload.get("layer_ids") != layer_ids:
        raise ValueError("adapter layer window does not match the current model")


@torch.no_grad()
def load_adapter_state(model: nn.Module, payload: dict, *, prefix: str = "") -> None:
    """Apply a full adapter or one layer's subset of a previously read adapter."""
    modules = _adapter_modules(model)
    namespace = prefix + "." if prefix else ""
    expected = {namespace + name: asdict(module.config) for name, module in modules.items()}
    metadata = {name: config for name, config in payload["modules"].items()
                if name.startswith(namespace)}
    if metadata != expected:
        raise ValueError("adapter configuration does not match the current model")
    targets = {f"{namespace}{name}.{key}": getattr(module, key)
               for name, module in modules.items() for key in ("lora_a", "lora_b")}
    weights = {name: tensor for name, tensor in payload["weights"].items()
               if name.startswith(namespace)}
    if weights.keys() != targets.keys():
        raise ValueError("adapter tensor names do not match")
    for name, target in targets.items():
        if target.shape != weights[name].shape:
            raise ValueError(f"adapter tensor shape does not match: {name}")
        if hasattr(target, "to_local"):
            raise ValueError("load adapters before FSDP wrapping")
    if any(module.merged for module in modules.values()):
        raise ValueError("unmerge adapters before loading")
    for name, target in targets.items():
        target.copy_(weights[name])


def load_adapter(model: nn.Module, path: str | Path) -> None:
    """Load into matching, unmerged adapters before FSDP wrapping."""
    payload = read_adapter(path)
    validate_adapter_layout(model, payload)
    load_adapter_state(model, payload)


def merge_adapters(model: nn.Module) -> None:
    modules = _adapter_modules(model)
    if any(module.training or hasattr(module.weight, "to_local") for module in modules.values()):
        raise ValueError("merge requires an unsharded model in eval mode")
    for module in modules.values():
        module.merge()


def unmerge_adapters(model: nn.Module) -> None:
    for module in _adapter_modules(model).values():
        module.unmerge()
