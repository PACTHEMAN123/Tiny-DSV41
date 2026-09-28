"""Model traversal, injection, parameter selection, and inference merging."""

from collections.abc import Iterator

from torch import nn

from .config import LoRAConfig
from .linear import LoRALinear


def inject_lora(model: nn.Module, config: LoRAConfig) -> list[str]:
    """Validate all targets, freeze the model, then replace matched linears."""
    if any(isinstance(module, LoRALinear) for module in model.modules()):
        raise ValueError("LoRA is already installed")
    selected: dict[str, nn.Linear] = {}
    modules = dict(model.named_modules())
    for target in config.targets:
        matches = [
            (name, module) for name, module in modules.items()
            if name == target or name.endswith("." + target)
        ]
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


def adapter_parameters(model: nn.Module) -> Iterator[nn.Parameter]:
    """Yield only adapter A/B parameters, suitable for a PyTorch optimizer."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            yield module.lora_a
            yield module.lora_b


def _adapter_modules(model: nn.Module) -> dict[str, LoRALinear]:
    modules = {
        name: module for name, module in model.named_modules()
        if isinstance(module, LoRALinear)
    }
    if not modules:
        raise ValueError("model has no LoRA adapters")
    return modules


def merge_adapters(model: nn.Module) -> None:
    """Merge all adapters after validating the whole model is ready."""
    modules = _adapter_modules(model)
    if any(module.training or hasattr(module.weight, "to_local") for module in modules.values()):
        raise ValueError("merge requires an unsharded model in eval mode")
    for module in modules.values():
        module.merge()


def unmerge_adapters(model: nn.Module) -> None:
    for module in _adapter_modules(model).values():
        module.unmerge()
