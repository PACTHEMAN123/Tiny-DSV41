"""Sparse MoE routing, token dispatch, and expert computation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
import torch.nn.functional as F
from torch import nn

from .config import DeepSeekV41Config

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh


@dataclass
class _Dispatch:
    token_count: int
    topk: int
    inverse_order: torch.Tensor | None = None
    send_splits: list[int] | None = None
    recv_splits: list[int] | None = None


class TokenDispatcher:
    """Keep token routing local when EP is disabled."""

    def expert_range(self, num_experts: int) -> tuple[int, int]:
        return 0, num_experts

    def dispatch(
        self,
        x: torch.Tensor,
        expert_ids: torch.Tensor,
        weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, _Dispatch]:
        topk = expert_ids.shape[-1]
        tokens = x[:, None, :].expand(-1, topk, -1).reshape(-1, x.shape[-1])
        metadata = _Dispatch(token_count=x.shape[0], topk=topk)
        return tokens, expert_ids.reshape(-1), weights.reshape(-1), metadata

    def combine(self, x: torch.Tensor, metadata: _Dispatch) -> torch.Tensor:
        return x.view(metadata.token_count, metadata.topk, -1).sum(dim=1)


class AllToAllTokenDispatcher(TokenDispatcher):
    """Move each token assignment to the rank that owns its expert."""

    def __init__(self, mesh: DeviceMesh) -> None:
        if mesh.ndim != 1:
            raise ValueError("the EP mesh must be one-dimensional")
        self.mesh = mesh
        self.size = mesh.size()
        self.rank = mesh.get_local_rank()
        self.group = mesh.get_group()
        self.num_experts: int | None = None

    def expert_range(self, num_experts: int) -> tuple[int, int]:
        if num_experts < 1:
            raise ValueError("num_experts must be positive")
        if num_experts % self.size:
            raise ValueError(f"num_experts ({num_experts}) must divide EP size ({self.size})")
        self.num_experts = num_experts
        local = num_experts // self.size
        start = self.rank * local
        return start, start + local

    def dispatch(
        self,
        x: torch.Tensor,
        expert_ids: torch.Tensor,
        weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, _Dispatch]:
        if self.num_experts is None:
            raise RuntimeError("attach the dispatcher to RoutedExperts before use")
        start, stop = self.expert_range(self.num_experts)
        local_experts = stop - start
        topk = expert_ids.shape[-1]
        flat_ids = expert_ids.reshape(-1)
        if flat_ids.numel() and (flat_ids.min() < 0 or flat_ids.max() >= self.num_experts):
            raise ValueError("expert ids are outside the configured expert range")

        owners = torch.div(flat_ids, local_experts, rounding_mode="floor")
        order = torch.argsort(owners, stable=True)
        send_counts = torch.bincount(owners, minlength=self.size).to(torch.int64)
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts, group=self.group)
        send_splits = send_counts.tolist()
        recv_splits = recv_counts.tolist()

        tokens = x[:, None, :].expand(-1, topk, -1).reshape(-1, x.shape[-1])[order]
        routed_ids = self._exchange(flat_ids[order].contiguous(), recv_splits, send_splits)
        routed_x = self._exchange(tokens.contiguous(), recv_splits, send_splits, autograd=True)
        routed_weights = self._exchange(
            weights.reshape(-1)[order].contiguous(), recv_splits, send_splits, autograd=True
        )
        metadata = _Dispatch(
            token_count=x.shape[0],
            topk=topk,
            inverse_order=torch.argsort(order),
            send_splits=send_splits,
            recv_splits=recv_splits,
        )
        return routed_x, routed_ids - start, routed_weights, metadata

    def combine(self, x: torch.Tensor, metadata: _Dispatch) -> torch.Tensor:
        assert metadata.send_splits is not None
        assert metadata.recv_splits is not None
        assert metadata.inverse_order is not None
        returned = self._exchange(
            x.contiguous(),
            metadata.send_splits,
            metadata.recv_splits,
            autograd=True,
        )
        returned = returned[metadata.inverse_order]
        return returned.view(metadata.token_count, metadata.topk, -1).sum(dim=1)

    def _exchange(
        self,
        tensor: torch.Tensor,
        output_splits: list[int],
        input_splits: list[int],
        *,
        autograd: bool = False,
    ) -> torch.Tensor:
        output = tensor.new_empty((sum(output_splits), *tensor.shape[1:]))
        if autograd:
            return dist_nn.all_to_all_single(
                output,
                tensor,
                output_split_sizes=output_splits,
                input_split_sizes=input_splits,
                group=self.group,
            )
        dist.all_to_all_single(
            output,
            tensor,
            output_split_sizes=output_splits,
            input_split_sizes=input_splits,
            group=self.group,
        )
        return output


def _activation(name: str, x: torch.Tensor) -> torch.Tensor:
    if name == "sqrtsoftplus":
        return torch.sqrt(F.softplus(x))
    if name == "softmax":
        return torch.softmax(x, dim=-1)
    return torch.sigmoid(x)


class TopKRouter(nn.Module):
    def __init__(self, config: DeepSeekV41Config) -> None:
        super().__init__()
        self.topk = config.num_experts_per_tok
        self.num_experts = config.n_routed_experts
        self.score_name = config.scoring_func
        self.temperature = config.gate_temp
        self.normalize = config.norm_topk_prob
        self.scale = config.routed_scaling_factor
        self.weight = nn.Parameter(torch.empty(self.num_experts, config.hidden_size))
        self.register_buffer("selection_bias", torch.zeros(self.num_experts))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = F.linear(x.flatten(0, 1).float(), self.weight.float()) / self.temperature
        scores = _activation(self.score_name, logits)
        indices = (scores + self.selection_bias).topk(self.topk, dim=-1).indices
        weights = scores.gather(1, indices)
        if self.normalize and self.topk > 1:
            weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
        return logits, weights * self.scale, indices


def _clamped_swiglu(gate: torch.Tensor, up: torch.Tensor, limit: float) -> torch.Tensor:
    gate, up = gate.float(), up.float()
    if limit > 0:
        gate = gate.clamp(max=limit)
        up = up.clamp(-limit, limit)
    return F.silu(gate) * up


class RoutedExperts(nn.Module):
    def __init__(self, config: DeepSeekV41Config, dispatcher: TokenDispatcher) -> None:
        super().__init__()
        self.dispatcher = dispatcher
        self.global_num_experts = config.n_routed_experts
        self.expert_start, expert_stop = dispatcher.expert_range(config.n_routed_experts)
        self.num_experts = expert_stop - self.expert_start
        self.intermediate = config.moe_intermediate_size
        self.limit = config.swiglu_limit
        self.gate_up = nn.Parameter(
            torch.empty(self.num_experts, 2 * self.intermediate, config.hidden_size)
        )
        self.down = nn.Parameter(
            torch.empty(self.num_experts, config.hidden_size, self.intermediate)
        )

    def forward(
        self, x: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        x, indices, weights, metadata = self.dispatcher.dispatch(x, indices, weights)
        output = torch.zeros_like(x, dtype=torch.float32)
        for expert_id in indices.unique():
            token_ids = torch.where(indices == expert_id)[0]
            current = x[token_ids]
            gate, up = F.linear(current, self.gate_up[expert_id]).chunk(2, dim=-1)
            current = _clamped_swiglu(gate, up, self.limit)
            current = current * weights[token_ids, None]
            current = F.linear(current.to(x.dtype), self.down[expert_id])
            output.index_add_(0, token_ids, current.float())
        return self.dispatcher.combine(output, metadata)


class SharedExpert(nn.Module):
    def __init__(self, config: DeepSeekV41Config) -> None:
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.moe_intermediate_size, bias=False)
        self.up = nn.Linear(config.hidden_size, config.moe_intermediate_size, bias=False)
        self.down = nn.Linear(config.moe_intermediate_size, config.hidden_size, bias=False)
        self.limit = config.swiglu_limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(_clamped_swiglu(self.gate(x), self.up(x), self.limit).to(x.dtype))


class SparseMoE(nn.Module):
    def __init__(self, config: DeepSeekV41Config, dispatcher: TokenDispatcher) -> None:
        super().__init__()
        self.router = TopKRouter(config)
        self.routed = RoutedExperts(config, dispatcher)
        self.shared = SharedExpert(config)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shape = x.shape
        flat = x.flatten(0, 1)
        logits, weights, indices = self.router(x)
        output = self.routed(flat, indices, weights) + self.shared(flat).float()
        return output.to(x.dtype).view(shape), logits


__all__ = [
    "AllToAllTokenDispatcher",
    "RoutedExperts",
    "SparseMoE",
    "TokenDispatcher",
    "TopKRouter",
]
