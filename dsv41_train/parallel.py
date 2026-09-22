"""Model-independent device meshes and context-parallel collectives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh


class _AllGather(torch.autograd.Function):
    """Autograd all-gather whose backward stays inside the given subgroup."""

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, dim: int, group, size: int) -> torch.Tensor:
        ctx.dim = dim
        ctx.group = group
        ctx.size = size
        parts = [torch.empty_like(tensor) for _ in range(size)]
        dist.all_gather(parts, tensor.contiguous(), group=group)
        return torch.cat(parts, dim=dim)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_output = grad_output.movedim(ctx.dim, 0).contiguous()
        exchanged = torch.empty_like(grad_output)
        dist.all_to_all_single(exchanged, grad_output, group=ctx.group)
        local_grad = torch.stack(exchanged.chunk(ctx.size, dim=0)).sum(dim=0)
        return local_grad.movedim(0, ctx.dim).contiguous(), None, None, None


@dataclass(frozen=True)
class ParallelMeshes:
    """The four mesh views needed by dense weights, contexts, and experts."""

    fsdp: DeviceMesh
    cp: DeviceMesh | None
    ep: DeviceMesh | None
    expert_fsdp: DeviceMesh | None
    _dense: DeviceMesh | None = None
    _sparse: DeviceMesh | None = None

    @classmethod
    def build(
        cls,
        *,
        cp: int = 1,
        ep: int = 1,
        device_type: str = "cuda",
    ) -> ParallelMeshes:
        from torch.distributed.device_mesh import init_device_mesh

        if not dist.is_initialized():
            raise RuntimeError("initialize torch.distributed before building parallel meshes")

        world_size = dist.get_world_size()
        if cp < 1 or world_size % cp:
            raise ValueError(f"cp ({cp}) must divide world size ({world_size})")
        if ep < 1 or world_size % ep:
            raise ValueError(f"ep ({ep}) must divide world size ({world_size})")

        dense = cp_mesh = None
        if cp > 1:
            dense = init_device_mesh(
                device_type,
                (world_size // cp, cp),
                mesh_dim_names=("fsdp", "cp"),
            )
            fsdp_mesh = dense["fsdp"]
            cp_mesh = dense["cp"]
        else:
            fsdp_mesh = init_device_mesh(
                device_type,
                (world_size,),
                mesh_dim_names=("fsdp",),
            )

        sparse = ep_mesh = expert_fsdp_mesh = None
        if ep > 1:
            sparse = init_device_mesh(
                device_type,
                (world_size // ep, ep),
                mesh_dim_names=("expert_fsdp", "ep"),
            )
            ep_mesh = sparse["ep"]
            expert_fsdp_mesh = sparse["expert_fsdp"]

        return cls(fsdp_mesh, cp_mesh, ep_mesh, expert_fsdp_mesh, dense, sparse)


class ContextParallel:
    """Shard tokens and gather the KV context required by attention."""

    def __init__(self, mesh: DeviceMesh | None = None) -> None:
        self.mesh = mesh
        self.size = 1 if mesh is None else mesh.size()
        self.rank = 0 if mesh is None else mesh.get_local_rank()
        self.group = None if mesh is None else mesh.get_group()

    @property
    def enabled(self) -> bool:
        return self.size > 1

    def shard(self, tensor: torch.Tensor, dim: int = 1) -> torch.Tensor:
        if not self.enabled:
            return tensor
        if tensor.shape[dim] % self.size:
            raise ValueError(
                f"tensor dimension {dim} ({tensor.shape[dim]}) must divide CP size ({self.size})"
            )
        return tensor.chunk(self.size, dim=dim)[self.rank].contiguous()

    def gather(self, tensor: torch.Tensor, dim: int = 1) -> torch.Tensor:
        if not self.enabled:
            return tensor
        tensor = tensor.contiguous()
        if tensor.requires_grad:
            return _AllGather.apply(tensor, dim, self.group, self.size)
        parts = [torch.empty_like(tensor) for _ in range(self.size)]
        dist.all_gather(parts, tensor, group=self.group)
        return torch.cat(parts, dim=dim)

    def sum(self, tensor: torch.Tensor, *, autograd: bool = False) -> torch.Tensor:
        if not self.enabled:
            return tensor
        if autograd:
            return dist_nn.all_reduce(tensor, op=dist.ReduceOp.SUM, group=self.group)
        result = tensor.clone()
        dist.all_reduce(result, op=dist.ReduceOp.SUM, group=self.group)
        return result

    def mean_loss(self, local_sum: torch.Tensor, local_count: torch.Tensor) -> torch.Tensor:
        """Return the global CP mean while keeping FSDP's gradient averaging correct."""

        if not self.enabled:
            return local_sum / local_count.clamp_min(1)

        total = self.sum(local_sum.detach())
        count = self.sum(local_count.detach()).clamp_min(1)
        backward_loss = local_sum * (self.size / count)
        value = total / count
        return backward_loss + (value - backward_loss.detach())


__all__ = [
    "ContextParallel",
    "ParallelMeshes",
]
