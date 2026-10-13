"""Optimizer construction without changing per-domain update algorithms."""

from dataclasses import dataclass

import torch

from .layout import parameter_layouts, local_tensor


@dataclass(frozen=True)
class OptimizerPolicy:
    name: str = "sgd"
    learning_rate: float = 1e-4
    foreach_dense: bool = False

    def __post_init__(self):
        if self.name not in {"sgd", "adamw"} or self.learning_rate <= 0:
            raise ValueError("optimizer requires sgd/adamw and a positive learning rate")


def build_optimizer(model, policy):
    layouts = parameter_layouts(model)
    sparse_ids = {id(module.weight) for module in model.modules()
                  if getattr(module, "sparse_gradients", False)}
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        sparse = id(parameter) in sparse_ids
        if policy.name == "adamw" and sparse:
            raise ValueError("AdamW does not support sparse Engram; use SGD or GPU dense gradients")
        if policy.name != "sgd" and layouts[name].role == "engram" and local_tensor(parameter).device.type == "cpu":
            raise ValueError("CPU Engram supports only sparse SGD")
        groups.setdefault((layouts[name].role, sparse), []).append(parameter)
    parameters = [{"params": values, "foreach": policy.foreach_dense and not sparse}
                  for (_, sparse), values in groups.items()]
    optimizer = _PlacementSGD if policy.name == "sgd" else torch.optim.AdamW
    result = optimizer(parameters, lr=policy.learning_rate)
    tables = [module for module in model.modules() if callable(getattr(module, "synchronize", None))
              and hasattr(module, "sparse_gradients")]
    if tables:
        def synchronize_tables(optimizer, args, kwargs):
            for table in tables:
                table.synchronize()
                cache = getattr(table.store, "cache", None)
                if cache is not None:
                    cache.before_step(table.weight)
        result.register_step_pre_hook(synchronize_tables)
        def invalidate_updated_rows(optimizer, args, kwargs):
            groups_by_parameter = {id(p): group for group in optimizer.param_groups for p in group["params"]}
            for table in tables:
                cache = getattr(table.store, "cache", None)
                if cache is None:
                    continue
                group = groups_by_parameter.get(id(table.weight))
                if group is not None and not group.get("momentum") and not group.get("weight_decay"):
                    cache.after_step(table.weight)
                else:
                    cache.clear()
        if any(getattr(table.store, "cache", None) is not None for table in tables):
            result.register_step_post_hook(invalidate_updated_rows)
    return result


def _coalesce_sparse_gradients(optimizer):
    # One SGD update per row avoids repeated BF16 rounding for duplicate IDs.
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if parameter.grad is not None and parameter.grad.is_sparse:
                gradient = parameter.grad
                if gradient.dtype in (torch.float16, torch.bfloat16):
                    parameter.grad = gradient.float().coalesce().to(gradient.dtype)
                else:
                    parameter.grad = gradient.coalesce()


class _PlacementSGD(torch.optim.SGD):
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        _coalesce_sparse_gradients(self)
        handled = []
        try:
            for group in self.param_groups:
                for parameter in group["params"]:
                    if parameter.grad is None:
                        continue
                    if (parameter.is_cuda or parameter.grad.is_sparse
                            or parameter.dtype not in (torch.float16, torch.bfloat16)):
                        continue
                    weight = local_tensor(parameter)
                    if weight.device.type != "cpu" or weight.dtype not in (torch.float16, torch.bfloat16):
                        continue
                    if group["momentum"] or group["weight_decay"] or group["maximize"]:
                        raise ValueError("low-precision CPU SGD requires zero momentum/decay and minimization")
                    gradient = local_tensor(parameter.grad)
                    # Dense CUDA SGD uses FP32 opmath; the CPU dense kernel casts
                    # alpha early. Sparse CPU/CUDA kernels already agree.
                    updated = weight.float().add_(gradient.float(), alpha=-group["lr"])
                    weight.copy_(updated)
                    handled.append((parameter, parameter.grad))
                    parameter.grad = None
            super().step()
        finally:
            for parameter, gradient in handled:
                parameter.grad = gradient
        return loss
