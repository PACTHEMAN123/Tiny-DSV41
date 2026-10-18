"""Memory-saving SGD hooks for the native DSV4 FSDP gradient contract."""

from __future__ import annotations

import math

import torch
from torch.distributed.tensor import DTensor

from .models.dsv4.moe import RoutedExperts


def sgd_parameter_groups(model, fp32_master_parameters=()):
    """Select exact trainable BF16 DTensor names; the default stays ungrouped."""
    names = list(fp32_master_parameters)
    parameters = dict(model.named_parameters())
    trainable = [p for p in parameters.values() if p.requires_grad]
    if not names:
        return trainable
    if len(set(names)) != len(names):
        raise ValueError("FP32 master parameter names must be unique")
    for name in names:
        parameter = parameters.get(name)
        if parameter is None or not parameter.requires_grad:
            raise ValueError(f"FP32 master parameter must name a trainable parameter: {name}")
        if not isinstance(parameter, DTensor) or parameter.dtype != torch.bfloat16:
            raise ValueError(f"FP32 master parameter must be a BF16 DTensor: {name}")
    selected = {id(parameters[name]) for name in names}
    ordinary = [p for p in trainable if id(p) not in selected]
    return ([{"params": ordinary}] if ordinary else []) + [
        {"params": [p for p in trainable if id(p) in selected], "fp32_master": True}]


class SGDInBackward:
    """Update reduced DTensor gradients; ordinary parameters stay with SGD.

    Finite checks are local and asynchronous. The caller must check
    ``local_finite`` across ranks after backward, before stepping ordinary
    parameters or saving. A failed step is not atomic: reload a checkpoint.
    """

    def __init__(self):
        self.handles = []
        self.parameter_ids = set()
        self._finite = {}

    def reset(self):
        self._finite.clear()

    def local_finite(self, device):
        finite = torch.ones((), device=device, dtype=torch.bool)
        for value in self._finite.values():
            finite = finite & value.to(device)
        return finite

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    @torch.no_grad()
    def _step(self, parameter, optimizer, group_index, divisor):
        if parameter.grad is None:
            return
        gradient, weight = parameter.grad.to_local(), parameter.to_local()
        finite = torch.isfinite(gradient).all()
        group = optimizer.param_groups[group_index]
        alpha = -group["lr"] / divisor
        if group.get("fp32_master", False):
            # Look up the current optimizer state: checkpoint loading replaces it.
            master = optimizer.state[parameter]["master_weight"]
            if (isinstance(master, DTensor) or master.dtype != torch.float32
                    or master.shape != weight.shape or master.device != weight.device):
                raise ValueError("FP32 master must retain the local parameter shape/device and FP32 dtype")
            master.add_(gradient, alpha=alpha)
            weight.copy_(master)
            finite = finite & torch.isfinite(master).all()
        else:
            weight.add_(gradient, alpha=alpha)
        finite = finite & torch.isfinite(weight).all()
        self._finite[weight.device] = self._finite.get(weight.device, finite) & finite
        parameter.grad = None


def install_sgd_in_backward(model, optimizer, *, gradient_accumulation_steps=1):
    """Install after FSDP wrapping, with one backward per optimizer step.

    The native optimizer still owns every trainable parameter, including those
    updated here, so its groups and checkpoint schema remain complete. DSV4
    expert reductions need their true EP divisor in addition to FSDP averaging.
    Groups with fp32_master=True eagerly store a local FP32 master in optimizer
    state; install hooks before restoring that state from a checkpoint.
    No loss scaling, clipping, momentum, decay, or hook collectives are applied.
    """
    if type(optimizer) is not torch.optim.SGD:
        raise ValueError("optimizer-in-backward requires native SGD")
    if gradient_accumulation_steps != 1:
        raise ValueError("optimizer-in-backward requires gradient-accumulation-steps=1")
    owned = []
    for group in optimizer.param_groups:
        if type(group.get("fp32_master", False)) is not bool:
            raise ValueError("fp32_master must be a boolean optimizer group option")
        if any(group.get(key, 0) for key in (
            "momentum", "dampening", "weight_decay", "nesterov", "maximize",
            "differentiable", "foreach", "fused",
        )):
            raise ValueError("optimizer-in-backward requires plain SGD without momentum or decay")
        if isinstance(group["lr"], torch.Tensor) or not math.isfinite(group["lr"]) or group["lr"] < 0:
            raise ValueError("optimizer-in-backward requires a finite nonnegative scalar learning rate")
        if group.get("fp32_master", False):
            for parameter in group["params"]:
                if not isinstance(parameter, DTensor) or parameter.dtype != torch.bfloat16:
                    raise ValueError("FP32 master groups require BF16 DTensor parameters")
                if optimizer.state.get(parameter):
                    raise ValueError("install FP32 master hooks before restoring optimizer state")
        owned.extend(id(parameter) for parameter in group["params"])
    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if len(set(owned)) != len(owned) or set(owned) != expected:
        raise ValueError("SGD must own every trainable parameter exactly once")
    divisors = {
        id(parameter): module.global_num_experts // module.num_experts
        for module in model.modules() if isinstance(module, RoutedExperts)
        for parameter in module.parameters()
    }
    state = SGDInBackward()
    for group_index, group in enumerate(optimizer.param_groups):
        for parameter in group["params"]:
            if isinstance(parameter, DTensor):
                if group.get("fp32_master", False):
                    optimizer.state[parameter]["master_weight"] = parameter.to_local().detach().to(
                        dtype=torch.float32, copy=True)
                state.parameter_ids.add(id(parameter))
                state.handles.append(parameter.register_post_accumulate_grad_hook(
                    lambda value, index=group_index, divisor=divisors.get(id(parameter), 1):
                    state._step(value, optimizer, index, divisor)
                ))
    return state
