"""Sparse MoE routing, token dispatch, and expert computation."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from ...dispatch import TokenDispatcher
from .config import DeepSeekV41Config


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
        self.gate_up = nn.ParameterList(
            nn.Parameter(torch.empty(2 * self.intermediate, config.hidden_size))
            for _ in range(self.num_experts)
        )
        self.down = nn.ParameterList(
            nn.Parameter(torch.empty(config.hidden_size, self.intermediate))
            for _ in range(self.num_experts)
        )
        self.gradient_group = None

    @property
    def local_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in (*self.gate_up, *self.down))

    def set_gradient_group(self, group) -> None:
        self.gradient_group = group

    def forward(
        self, x: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        x, indices, weights, metadata = self.dispatcher.dispatch(x, indices, weights)
        output = torch.zeros_like(x, dtype=torch.float32) + x.float().sum() * 0
        used = torch.zeros(self.num_experts, dtype=torch.int32, device=indices.device)
        used.scatter_(0, indices, 1)
        if self.gradient_group is not None:
            dist.all_reduce(used, op=dist.ReduceOp.MAX, group=self.gradient_group)
        for expert_id in used.nonzero().flatten().tolist():
            token_ids = torch.where(indices == expert_id)[0]
            if token_ids.numel() == 0:
                anchor = (
                    self.gate_up[expert_id].flatten()[0]
                    + self.down[expert_id].flatten()[0]
                )
                output = output + anchor.to(output.dtype) * 0
                continue
            current = x[token_ids]
            weight = self.gate_up[expert_id]
            gate = F.linear(current, weight[: self.intermediate])
            up = F.linear(current, weight[self.intermediate :])
            current = _clamped_swiglu(gate, up, self.limit)
            current = current * weights[token_ids, None]
            current = F.linear(current.to(x.dtype), self.down[expert_id])
            output.index_add_(0, token_ids, current.float())
        output = self.dispatcher.combine(output, metadata)
        return output


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
    "RoutedExperts",
    "SparseMoE",
    "TopKRouter",
]
