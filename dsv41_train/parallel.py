"""Model-independent device meshes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch.distributed as dist

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh


@dataclass(frozen=True)
class ParallelMeshes:
    """Mesh views for dense weights, contexts, experts, and Engram rows."""

    fsdp: DeviceMesh
    cp: DeviceMesh | None
    ep: DeviceMesh | None
    expert_fsdp: DeviceMesh | None
    _dense: DeviceMesh | None = None
    _sparse: DeviceMesh | None = None
    engram: DeviceMesh | None = None
    engram_fsdp: DeviceMesh | None = None
    _engram_sparse: DeviceMesh | None = None

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


__all__ = ["ParallelMeshes"]
