"""Native linear adapters with unchanged base checkpoint parameter names."""

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

    def __post_init__(self):
        if type(self.rank) is not int or self.rank < 1:
            raise ValueError("LoRA rank must be a positive integer")
        if not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("LoRA alpha must be positive and finite")
        if not 0 <= self.dropout < 1:
            raise ValueError("LoRA dropout must be in [0, 1)")
        if (not isinstance(self.targets, (tuple, list)) or not self.targets
                or any(not isinstance(name, str) or not name for name in self.targets)):
            raise ValueError("LoRA targets must contain module names")
        object.__setattr__(self, "targets", tuple(self.targets))


def _check_linear(module):
    if not isinstance(module, nn.Linear):
        raise ValueError("LoRA only supports nn.Linear targets")
    if module.weight.is_meta or hasattr(module.weight, "to_local"):
        raise ValueError("inject LoRA after loading and before FSDP wrapping")


class LoRALinear(nn.Linear):
    def __init__(self, base: nn.Linear, config: LoRAConfig):
        _check_linear(base)
        # nn.Linear.__init__ would allocate and initialize another base weight.
        nn.Module.__init__(self)
        self.in_features, self.out_features = base.in_features, base.out_features
        self.weight, self.bias = base.weight, base.bias
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)
        self.config = config
        self.scaling = config.alpha / config.rank
        self.adapters_enabled = True
        self.lora_a = nn.Parameter(base.weight.new_empty(config.rank, self.in_features))
        self.lora_b = nn.Parameter(base.weight.new_zeros(self.out_features, config.rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer("_unmerged_weight", None, persistent=False)
        self.train(base.training)

    @property
    def merged(self):
        return self._unmerged_weight is not None

    def forward(self, x):
        weight = self._unmerged_weight if self.merged and not self.adapters_enabled else self.weight
        result = F.linear(x, weight, self.bias)
        if self.adapters_enabled and not self.merged:
            result = result + F.linear(F.linear(self.dropout(x), self.lora_a), self.lora_b) * self.scaling
        return result

    def train(self, mode=True):
        if mode and self.merged:
            self.unmerge()
        return super().train(mode)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        if self.merged:
            # A resumable state always stores W, A, B, never W + BA, A, B.
            weight = self._unmerged_weight
            destination[prefix + "weight"] = weight if keep_vars else weight.detach()

    def _load_from_state_dict(self, *args, **kwargs):
        self.unmerge()
        super()._load_from_state_dict(*args, **kwargs)

    @torch.no_grad()
    def merge(self):
        if hasattr(self.weight, "to_local") or self.weight.is_meta or self.training:
            raise ValueError("merge requires a materialized, unsharded model in eval mode")
        if not self.merged:
            self._unmerged_weight = self.weight.detach().clone()
            dtype = torch.float32 if self.weight.dtype in (torch.float16, torch.bfloat16) else self.weight.dtype
            delta = self.lora_b.to(dtype) @ self.lora_a.to(dtype)
            self.weight.copy_((self.weight.to(dtype) + delta * self.scaling).to(self.weight.dtype))

    @torch.no_grad()
    def unmerge(self):
        if self.merged:
            self.weight.copy_(self._unmerged_weight)
            self._unmerged_weight = None


def inject_lora(model: nn.Module, config: LoRAConfig) -> list[str]:
    """Match exact paths or dotted suffixes, validate, then freeze and replace."""
    if any(isinstance(module, LoRALinear) for module in model.modules()):
        raise ValueError("LoRA is already installed")
    selected = {}
    modules = dict(model.named_modules())
    for target in config.targets:
        matches = {name: module for name, module in modules.items()
                   if name == target or name.endswith("." + target)}
        if not matches:
            raise ValueError(f"LoRA target did not match a module: {target}")
        for module in matches.values():
            _check_linear(module)
        selected.update(matches)
    model.requires_grad_(False)
    for name, module in selected.items():
        parent, _, leaf = name.rpartition(".")
        setattr(model.get_submodule(parent), leaf, LoRALinear(module, config))
    return list(selected)


def adapter_parameters(model: nn.Module):
    for module in model.modules():
        if isinstance(module, LoRALinear):
            yield module.lora_a
            yield module.lora_b


def _adapter_modules(model):
    modules = {name: module for name, module in model.named_modules() if isinstance(module, LoRALinear)}
    if not modules:
        raise ValueError("model has no LoRA adapters")
    return modules


def _key(name, leaf):
    return ".".join(part for part in (name, leaf) if part)


def _layer_ids(model):
    decoder = getattr(model, "model", None)
    if hasattr(decoder, "layer_ids"):
        return decoder.layer_ids
    if hasattr(decoder, "layers"):
        return [getattr(layer, "layer_id", index) for index, layer in enumerate(decoder.layers)]
    return None


@torch.no_grad()
def save_adapter(model: nn.Module, path: str | Path):
    """All FSDP ranks participate; rank zero writes A/B only, without base weights."""
    modules = _adapter_modules(model)
    writer = not dist.is_initialized() or dist.get_rank() == 0
    weights = {}
    for name, module in modules.items():
        for key in ("lora_a", "lora_b"):
            value = getattr(module, key).detach()
            if hasattr(value, "full_tensor"):
                value = value.full_tensor()
            if writer:
                weights[_key(name, key)] = value.cpu()
    if writer:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"format_version": 1, "layer_ids": _layer_ids(model),
                    "modules": {name: asdict(module.config) for name, module in modules.items()},
                    "weights": weights}, path)


def read_adapter(path: str | Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise ValueError("unsupported adapter format")
    if (set(payload) != {"format_version", "layer_ids", "modules", "weights"}
            or not isinstance(payload["modules"], dict) or not isinstance(payload["weights"], dict)
            or any(not isinstance(name, str) for name in (*payload["modules"], *payload["weights"]))):
        raise ValueError("invalid adapter payload")
    return payload


def validate_adapter_layout(model: nn.Module, payload: dict):
    if payload.get("layer_ids") != _layer_ids(model):
        raise ValueError("adapter layer window does not match the current model")


@torch.no_grad()
def load_adapter_state(model: nn.Module, payload: dict, *, prefix: str = ""):
    """Load a full adapter or one layer's subset before FSDP, validating before copying."""
    modules = {_key(prefix, name): module for name, module in _adapter_modules(model).items()}
    namespace = prefix + "." if prefix else ""
    expected = {name: asdict(module.config) for name, module in modules.items()}
    metadata = {name: value for name, value in payload["modules"].items()
                if name == prefix or name.startswith(namespace)}
    if metadata != expected:
        raise ValueError("adapter configuration does not match the current model")
    targets = {_key(name, key): getattr(module, key)
               for name, module in modules.items() for key in ("lora_a", "lora_b")}
    weights = {name: value for name, value in payload["weights"].items() if name.startswith(namespace)}
    if weights.keys() != targets.keys():
        raise ValueError("adapter tensor names do not match")
    if any(module.merged for module in modules.values()):
        raise ValueError("unmerge adapters before loading")
    for name, target in targets.items():
        value = weights[name]
        if target.is_meta or hasattr(target, "to_local"):
            raise ValueError("load adapters after materialization and before FSDP wrapping")
        if not isinstance(value, torch.Tensor) or value.shape != target.shape:
            raise ValueError(f"adapter tensor shape does not match: {name}")
        if value.layout != torch.strided or not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f"adapter tensor must contain finite floating values: {name}")
    for name, target in targets.items():
        target.copy_(weights[name])


def load_adapter(model: nn.Module, path: str | Path):
    payload = read_adapter(path)
    validate_adapter_layout(model, payload)
    load_adapter_state(model, payload)


def merge_adapters(model: nn.Module):
    modules = _adapter_modules(model)
    if any(module.training or module.weight.is_meta or hasattr(module.weight, "to_local")
           for module in modules.values()):
        raise ValueError("merge requires a materialized, unsharded model in eval mode")
    for module in modules.values():
        module.merge()


def unmerge_adapters(model: nn.Module):
    for module in _adapter_modules(model).values():
        module.unmerge()
