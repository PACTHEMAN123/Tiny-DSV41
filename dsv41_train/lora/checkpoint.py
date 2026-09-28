"""Portable adapter files; full optimizer/RNG resume uses TrainingState."""

from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn

from .adapters import _adapter_modules


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
        torch.save({
            "format_version": 1,
            "layer_ids": getattr(getattr(model, "model", None), "layer_ids", None),
            "modules": {name: asdict(module.config) for name, module in modules.items()},
            "weights": weights,
        }, path)


def read_adapter(path: str | Path) -> dict:
    """Read once on CPU, including when a loader later consumes it layer by layer."""
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
    metadata = {
        name: config for name, config in payload["modules"].items()
        if name.startswith(namespace)
    }
    if metadata != expected:
        raise ValueError("adapter configuration does not match the current model")
    targets = {
        f"{namespace}{name}.{key}": getattr(module, key)
        for name, module in modules.items() for key in ("lora_a", "lora_b")
    }
    weights = {
        name: tensor for name, tensor in payload["weights"].items()
        if name.startswith(namespace)
    }
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


@torch.no_grad()
def save_merged(model: nn.Module, path: str | Path) -> None:
    """Export plain model weights loadable without injecting LoRA; leave the model intact."""
    modules = _adapter_modules(model)
    if model.training or any(module.training for module in modules.values()):
        raise ValueError("call eval() before exporting merged weights")
    if any(hasattr(p, "to_local") or p.is_meta for p in model.parameters()):
        raise ValueError("merged export requires a materialized, unsharded model")
    if any(getattr(module, "checkpoint_parameter_names", lambda: {})()
           for module in model.modules()):
        raise ValueError("merged export requires a model without manual parameter shards")
    state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    for name, module in modules.items():
        namespace = name + "." if name else ""
        if module.merged:
            weight = module.weight
        else:
            delta = module.lora_b.float() @ module.lora_a.float()
            weight = (module.weight.float() + module.scaling * delta).to(module.weight.dtype)
        state[namespace + "weight"] = weight.detach().cpu()
        del state[namespace + "lora_a"], state[namespace + "lora_b"]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
