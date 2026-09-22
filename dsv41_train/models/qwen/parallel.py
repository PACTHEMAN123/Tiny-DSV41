"""Qwen expert ownership, FSDP wrapping, and gradient reductions."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn

from ...dispatch import AllToAllTokenDispatcher, TokenDispatcher
from ...parallel import ContextParallel, ParallelMeshes
from ...runtime import local_tensor
from .model import Qwen3MoeExperts, Qwen3MoeForCausalLM


def build_parallelism(
    cp: int = 1, ep: int = 1, device_type: str = "cuda",
) -> tuple[ParallelMeshes, ContextParallel, TokenDispatcher]:
    if cp != ep:
        raise ValueError("the overlapping CP/EP topology requires cp-size == ep-size")
    meshes = ParallelMeshes.build(cp=cp, ep=ep, device_type=device_type)
    dispatcher = AllToAllTokenDispatcher(meshes.ep) if meshes.ep is not None else TokenDispatcher()
    return meshes, ContextParallel(meshes.cp), dispatcher


def apply_fsdp2(model: Qwen3MoeForCausalLM, meshes: ParallelMeshes) -> None:
    try:
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    except ImportError:
        from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard

    policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32, output_dtype=torch.bfloat16,
    )
    options = {"mesh": meshes.fsdp, "mp_policy": policy, "reshard_after_forward": True}
    fully_shard(model.model.embed_tokens, **options)
    for layer in model.model.layers:
        experts = getattr(layer.mlp, "experts", None)
        if experts is not None and meshes.expert_fsdp is not None:
            fully_shard(experts, **{**options, "mesh": meshes.expert_fsdp})
        fully_shard(layer, **options)
    fully_shard(model.lm_head, **options)
    fully_shard(model, mesh=meshes.fsdp, mp_policy=policy)


@dataclass
class ParallelParameters:
    """Dense parameters are CP replicas; partitioned experts are not."""

    dense: list[nn.Parameter]
    experts: list[nn.Parameter]

    @classmethod
    def collect(cls, model: nn.Module) -> ParallelParameters:
        experts = [
            parameter
            for module in model.modules()
            if isinstance(module, Qwen3MoeExperts)
            and module.num_experts < module.global_num_experts
            for parameter in module.parameters()
        ]
        expert_ids = {id(parameter) for parameter in experts}
        dense = [parameter for parameter in model.parameters() if id(parameter) not in expert_ids]
        return cls(dense, experts)

    def numel(self, ep_size: int) -> int:
        return sum(p.numel() for p in self.dense) + ep_size * sum(p.numel() for p in self.experts)

    def synchronize(self, cp: ContextParallel) -> None:
        if not cp.enabled:
            return
        for parameters, replicated in ((self.dense, True), (self.experts, False)):
            for parameter in parameters:
                if parameter.grad is None:
                    continue
                gradient = local_tensor(parameter.grad)
                if replicated:
                    dist.all_reduce(gradient, group=cp.group)
                # CP mean_loss scales local losses by CP size; EP experts need
                # the same correction as dense replicas, without another sum.
                gradient.div_(cp.size)

    def clip_grad_norm(self, cp: ContextParallel, max_norm: float = 1.0) -> torch.Tensor:
        squared_norm = torch.zeros((), device=self.dense[0].device, dtype=torch.float32)
        for parameters, replicas in ((self.dense, cp.size), (self.experts, 1)):
            for parameter in parameters:
                if parameter.grad is not None:
                    squared_norm += local_tensor(parameter.grad).float().square().sum() / replicas
        dist.all_reduce(squared_norm)
        norm = squared_norm.sqrt()
        coefficient = (max_norm / (norm + 1.0e-6)).clamp(max=1.0)
        for parameter in (*self.dense, *self.experts):
            if parameter.grad is not None:
                local_tensor(parameter.grad).mul_(coefficient)
        return norm
