"""DSV4 parallel topology and FSDP wrapping."""

from ...dispatch import AllToAllTokenDispatcher, TokenDispatcher
from ...parallel import ContextParallel, ParallelMeshes
from .model import DeepSeekV41ForCausalLM, DeepSeekV41Model


def build_parallelism(
    cp: int = 1,
    ep: int = 1,
    device_type: str = "cuda",
) -> tuple[ParallelMeshes, ContextParallel, TokenDispatcher]:
    meshes = ParallelMeshes.build(cp=cp, ep=ep, device_type=device_type)
    dispatcher = (
        AllToAllTokenDispatcher(meshes.ep) if meshes.ep is not None else TokenDispatcher()
    )
    return meshes, ContextParallel(meshes.cp), dispatcher


def apply_fsdp2(
    model: DeepSeekV41ForCausalLM | DeepSeekV41Model,
    meshes: ParallelMeshes,
    *,
    reshard_after_forward: bool = True,
) -> None:
    try:
        from torch.distributed.fsdp import fully_shard
    except ImportError:
        from torch.distributed._composable.fsdp import fully_shard

    decoder = model.model if isinstance(model, DeepSeekV41ForCausalLM) else model
    if meshes.ep is not None:
        assert meshes.expert_fsdp is not None
        for table in decoder.engram_tables.values():
            fully_shard(
                table,
                mesh=meshes.expert_fsdp,
                reshard_after_forward=reshard_after_forward,
            )
    for layer in decoder.layers:
        if meshes.ep is not None:
            assert meshes.expert_fsdp is not None
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
    fully_shard(model, mesh=meshes.fsdp)


__all__ = ["apply_fsdp2", "build_parallelism"]
