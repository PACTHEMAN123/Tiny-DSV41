"""DSV4 parallel topology and FSDP wrapping."""

from dataclasses import replace

import torch.distributed as dist

from ...dispatch import AllToAllTokenDispatcher, TokenDispatcher
from ...parallel import ContextParallel, ParallelMeshes
from .model import DecoderLayer, DeepSeekV41ForCausalLM, DeepSeekV41Model
from .policies import FSDPPolicy


def build_parallelism(
    cp: int = 1,
    ep: int = 1,
    device_type: str = "cuda",
) -> tuple[ParallelMeshes, ContextParallel, TokenDispatcher]:
    meshes = ParallelMeshes.build(cp=cp, ep=ep, device_type=device_type)
    from torch.distributed.device_mesh import init_device_mesh

    if cp > 1:
        meshes = replace(meshes, fsdp=init_device_mesh(
            device_type, (dist.get_world_size(),), mesh_dim_names=("fsdp",),
        ))
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
    policy: FSDPPolicy | None = None,
) -> None:
    if hasattr(layer, "_dsv41_sharded"):
        raise ValueError("decoder layer is already FSDP wrapped")
    policy = policy or FSDPPolicy(reshard_after_forward=reshard_after_forward)
    fully_shard = _fully_shard()
    expert_mesh = meshes.expert_fsdp if meshes.ep is not None else meshes.fsdp
    layer.moe.routed.set_gradient_group(expert_mesh.get_group())
    fully_shard(layer.moe.routed, mesh=expert_mesh,
                **_options(policy, layer.layer_id, "experts"))
    for fp32_module in (layer.attention_hc, layer.moe_hc, layer.attention.sinks):
        fully_shard(
            fp32_module,
            mesh=meshes.fsdp,
            **_options(policy, layer.layer_id, "dense"),
        )
    fully_shard(layer, mesh=meshes.fsdp, **_options(policy, layer.layer_id, "dense"))
    layer._dsv41_sharded = True


def _options(policy, layer_id, role):
    options = {"reshard_after_forward": policy.reshard_after_forward}
    if policy.offloads(layer_id, role):
        from torch.distributed.fsdp import CPUOffloadPolicy
        options["offload_policy"] = CPUOffloadPolicy(pin_memory=policy.pin_memory)
    return options


def shard_embedding_and_head(model, meshes, *, reshard_after_forward=True):
    fully_shard = _fully_shard()
    decoder = model.model if isinstance(model, DeepSeekV41ForCausalLM) else model
    for module in (decoder.embedding, getattr(model, "lm_head", None)):
        if module is not None and not getattr(module, "_dsv41_sharded", False):
            fully_shard(module, mesh=meshes.fsdp, reshard_after_forward=reshard_after_forward)
            module._dsv41_sharded = True


def apply_fsdp2_root(
    model: DeepSeekV41ForCausalLM | DeepSeekV41Model,
    meshes: ParallelMeshes,
    *,
    reshard_after_forward: bool = True,
) -> None:
    if getattr(model, "_dsv41_sharded", False):
        raise ValueError("model root is already FSDP wrapped")
    fully_shard = _fully_shard()

    decoder = model.model if isinstance(model, DeepSeekV41ForCausalLM) else model
    shard_embedding_and_head(model, meshes, reshard_after_forward=reshard_after_forward)
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
    root_options = {"mesh": meshes.fsdp, "reshard_after_forward": reshard_after_forward}
    if ignored_params:
        root_options["ignored_params"] = ignored_params
    fully_shard(model, **root_options)
    model._dsv41_sharded = True
    # Avoid implicit cross-layer prefetch racing collectives on overlapping CP/EP groups.
    from torch.distributed.fsdp import FSDPModule
    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.set_modules_to_backward_prefetch([module])


def apply_fsdp2(
    model: DeepSeekV41ForCausalLM | DeepSeekV41Model,
    meshes: ParallelMeshes,
    *,
    reshard_after_forward: bool = True,
    policy: FSDPPolicy | None = None,
) -> None:
    policy = policy or FSDPPolicy(reshard_after_forward=reshard_after_forward)
    decoder = model.model if isinstance(model, DeepSeekV41ForCausalLM) else model
    for layer in decoder.layers:
        apply_fsdp2_layer(
            layer,
            meshes,
            reshard_after_forward=reshard_after_forward,
            policy=policy,
        )
    apply_fsdp2_root(
        model,
        meshes,
        reshard_after_forward=policy.reshard_after_forward,
    )


__all__ = [
    "apply_fsdp2",
    "apply_fsdp2_layer",
    "apply_fsdp2_root",
    "build_parallelism",
]
