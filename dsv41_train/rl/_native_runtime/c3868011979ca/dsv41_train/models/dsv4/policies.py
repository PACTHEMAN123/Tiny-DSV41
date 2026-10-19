"""FSDP placement and gradient normalization for independent CP/EP meshes."""

from dataclasses import dataclass

import torch
import torch.distributed as dist

from ...layout import parameter_layouts, local_tensor


@dataclass(frozen=True)
class FSDPPolicy:
    cpu_offload_layers: tuple[int, ...] = ()
    cpu_offload_targets: tuple[str, ...] = ("dense", "experts")
    reshard_after_forward: bool = True
    pin_memory: bool = True

    def __post_init__(self):
        if any(layer < 0 for layer in self.cpu_offload_layers):
            raise ValueError("offload layer IDs must be non-negative")
        if not self.cpu_offload_targets or set(self.cpu_offload_targets) - {"dense", "experts"}:
            raise ValueError("FSDP offload targets must be dense and/or experts")

    def offloads(self, layer_id, role):
        return layer_id in self.cpu_offload_layers and role in self.cpu_offload_targets


class GradientPolicy:
    def __init__(self, model, meshes, *, fsdp=True):
        self.model, self.meshes, self.fsdp = model, meshes, fsdp
        self.normalized = False
        self.layouts = parameter_layouts(model)

    def begin_step(self):
        self.normalized = False

    @torch.no_grad()
    def normalize(self):
        if self.normalized:
            raise RuntimeError("gradients were already normalized for this optimizer step")
        ep = self.meshes.ep.size() if self.meshes.ep is not None else 1
        world = dist.get_world_size()
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            role = self.layouts[name].role
            if role == "engram":
                continue
            if self.fsdp:
                if role == "experts" and parameter.grad is not None:
                    local_tensor(parameter.grad).div_(ep)
            else:
                group = self.meshes.expert_fsdp.get_group() if role == "experts" and ep > 1 else None
                present = torch.tensor(int(parameter.grad is not None), device=parameter.device)
                dist.all_reduce(present, group=group)
                if not present.item():
                    continue
                if parameter.grad is None:
                    parameter.grad = torch.zeros_like(parameter)
                dist.all_reduce(parameter.grad, group=group)
                parameter.grad.div_(world)
        self.normalized = True


def data_parallel_rank(rank, cp_size):
    return rank // cp_size
