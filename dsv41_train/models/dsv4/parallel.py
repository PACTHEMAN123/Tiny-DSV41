"""FSDP wrapping for the DSV4 decoder and routed experts."""

from ...parallel import ParallelMeshes
from .model import DeepSeekV41ForCausalLM, DeepSeekV41Model


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
    for layer in decoder.layers:
        if meshes.ep is not None:
            assert meshes.expert_fsdp is not None
            fully_shard(
                layer.moe.routed,
                mesh=meshes.expert_fsdp,
                reshard_after_forward=reshard_after_forward,
            )
        fully_shard(layer, mesh=meshes.fsdp, reshard_after_forward=reshard_after_forward)
    fully_shard(model, mesh=meshes.fsdp)
