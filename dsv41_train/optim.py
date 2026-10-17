"""Memory-saving SGD hooks for the native DSV4 FSDP gradient contract."""

from __future__ import annotations

import math

import torch
from torch.distributed.tensor import DTensor

from .models.dsv4.moe import RoutedExperts


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
        weight.add_(gradient, alpha=-optimizer.param_groups[group_index]["lr"] / divisor)
        finite = finite & torch.isfinite(weight).all()
        self._finite[weight.device] = self._finite.get(weight.device, finite) & finite
        parameter.grad = None


def install_sgd_in_backward(model, optimizer, *, gradient_accumulation_steps=1):
    """Install after FSDP wrapping, with one backward per optimizer step.

    The native optimizer still owns every trainable parameter, including those
    updated here, so its groups and checkpoint schema remain complete. DSV4
    expert reductions need their true EP divisor in addition to FSDP averaging.
    No loss scaling, clipping, momentum, decay, or hook collectives are applied.
    """
    if type(optimizer) is not torch.optim.SGD:
        raise ValueError("optimizer-in-backward requires native SGD")
    if gradient_accumulation_steps != 1:
        raise ValueError("optimizer-in-backward requires gradient-accumulation-steps=1")
    owned = []
    for group in optimizer.param_groups:
        if any(group.get(key, 0) for key in (
            "momentum", "dampening", "weight_decay", "nesterov", "maximize",
            "differentiable", "foreach", "fused",
        )):
            raise ValueError("optimizer-in-backward requires plain SGD without momentum or decay")
        if isinstance(group["lr"], torch.Tensor) or not math.isfinite(group["lr"]) or group["lr"] < 0:
            raise ValueError("optimizer-in-backward requires a finite nonnegative scalar learning rate")
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
                state.parameter_ids.add(id(parameter))
                state.handles.append(parameter.register_post_accumulate_grad_hook(
                    lambda value, index=group_index, divisor=divisors.get(id(parameter), 1):
                    state._step(value, optimizer, index, divisor)
                ))
    return state
