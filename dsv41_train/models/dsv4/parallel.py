"""DSV4 parallel topology and FSDP wrapping."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch.distributed as dist
from torch import nn

from ...dispatch import AllToAllTokenDispatcher, TokenDispatcher
from ...parallel import ContextParallel, ParallelMeshes
from ...runtime import local_tensor
from .model import (
    DecoderLayer,
    DeepSeekV41ForCausalLM,
    DeepSeekV41Model,
    RowShardedEmbedding,
)
from .moe import RoutedExperts


def build_parallelism(
    cp: int = 1,
    ep: int = 1,
    device_type: str = "cuda",
) -> tuple[ParallelMeshes, ContextParallel, TokenDispatcher]:
    if cp > 1 and cp != ep:
        raise ValueError("DSV4 context parallelism requires cp-size == ep-size")
    meshes = ParallelMeshes.build(cp=cp, ep=ep, device_type=device_type)
    from torch.distributed.device_mesh import init_device_mesh

    if cp > 1:
        dense_fsdp = init_device_mesh(
            device_type,
            (dist.get_world_size(),),
            mesh_dim_names=("fsdp",),
        )
        meshes = replace(meshes, fsdp=dense_fsdp)
    engram_sparse = init_device_mesh(
        device_type,
        (dist.get_world_size(), 1),
        mesh_dim_names=("engram", "engram_fsdp"),
    )
    meshes = replace(
        meshes,
        engram=engram_sparse["engram"],
        engram_fsdp=engram_sparse["engram_fsdp"],
        _engram_sparse=engram_sparse,
    )
    dispatcher = (
        AllToAllTokenDispatcher(meshes.ep) if meshes.ep is not None else TokenDispatcher()
    )
    return meshes, ContextParallel(meshes.cp), dispatcher


@dataclass
class ParallelParameters:
    """Parameters with different gradient semantics on the shared CP/EP mesh."""

    dense: list[nn.Parameter]
    experts: list[nn.Parameter]
    engram: list[nn.Parameter]

    @classmethod
    def collect(cls, model: nn.Module) -> ParallelParameters:
        experts = [
            parameter
            for module in model.modules()
            if isinstance(module, RoutedExperts)
            and module.num_experts < module.global_num_experts
            for parameter in module.parameters()
        ]
        engram = [
            module.weight
            for module in model.modules()
            if isinstance(module, RowShardedEmbedding) and module.size > 1
        ]
        partitioned_ids = {id(parameter) for parameter in (*experts, *engram)}
        dense = [
            parameter
            for parameter in model.parameters()
            if id(parameter) not in partitioned_ids
        ]
        return cls(dense, experts, engram)

    def synchronize(
        self,
        cp: ContextParallel,
        *,
        dense_fully_sharded: bool = False,
    ) -> None:
        """Normalize CP gradients not already reduced by dense FSDP."""

        if not cp.enabled:
            return
        if not dense_fully_sharded:
            for parameter in self.dense:
                if parameter.grad is not None:
                    dist.all_reduce(local_tensor(parameter.grad), group=cp.group)
        scaled = self.experts if dense_fully_sharded else [*self.dense, *self.experts]
        for parameter in scaled:
            if parameter.grad is not None:
                local_tensor(parameter.grad).div_(cp.size)


def _fully_shard():
    try:
        from torch.distributed.fsdp import fully_shard
    except ImportError:
        from torch.distributed._composable.fsdp import fully_shard
    return fully_shard


def apply_fsdp2_layer(
    layer: DecoderLayer,
    meshes: ParallelMeshes,
    *,
    reshard_after_forward: bool = True,
) -> None:
    fully_shard = _fully_shard()
    if meshes.ep is not None:
        assert meshes.expert_fsdp is not None
        layer.moe.routed.set_gradient_group(meshes.expert_fsdp.get_group())
        fully_shard(
            layer.moe.routed,
            mesh=meshes.expert_fsdp,
            reshard_after_forward=reshard_after_forward,
        )
    for fp32_module in (layer.attention_hc, layer.moe_hc, layer.attention.sinks):
        fully_shard(
            fp32_module,
            mesh=meshes.fsdp,
            reshard_after_forward=reshard_after_forward,
        )
    fully_shard(layer, mesh=meshes.fsdp, reshard_after_forward=reshard_after_forward)


def apply_fsdp2_root(
    model: DeepSeekV41ForCausalLM | DeepSeekV41Model,
    meshes: ParallelMeshes,
    *,
    reshard_after_forward: bool = True,
) -> None:
    fully_shard = _fully_shard()

    decoder = model.model if isinstance(model, DeepSeekV41ForCausalLM) else model
    engram_fsdp = (
        meshes.engram_fsdp if meshes.engram_fsdp is not None else meshes.expert_fsdp
    )
    ignored_params = set()
    for table in decoder.engram_tables.values():
        if table.sparse_gradients:
            ignored_params.add(table.weight)
        elif engram_fsdp is not None:
            fully_shard(
                table,
                mesh=engram_fsdp,
                reshard_after_forward=reshard_after_forward,
            )
    root_options = {"mesh": meshes.fsdp}
    if ignored_params:
        root_options["ignored_params"] = ignored_params
    fully_shard(model, **root_options)


def apply_fsdp2(
    model: DeepSeekV41ForCausalLM | DeepSeekV41Model,
    meshes: ParallelMeshes,
    *,
    reshard_after_forward: bool = True,
) -> None:
    decoder = model.model if isinstance(model, DeepSeekV41ForCausalLM) else model
    for layer in decoder.layers:
        apply_fsdp2_layer(
            layer,
            meshes,
            reshard_after_forward=reshard_after_forward,
        )
    apply_fsdp2_root(
        model,
        meshes,
        reshard_after_forward=reshard_after_forward,
    )


__all__ = [
    "ParallelParameters",
    "apply_fsdp2",
    "apply_fsdp2_layer",
    "apply_fsdp2_root",
    "build_parallelism",
]
