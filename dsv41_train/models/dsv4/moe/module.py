"""DSV4 routers and grouped expert modules."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..config import DeepSeekV41Config
from .dispatch import TokenDispatcher
from .kernels import clamped_swiglu, clamped_swiglu_reference


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


def _grouped_mm(
    x: torch.Tensor, weight: torch.Tensor, counts: torch.Tensor
) -> torch.Tensor:
    if x.is_cuda:
        offsets = torch.cumsum(counts, dim=0, dtype=torch.int32)
        return torch._grouped_mm(x, weight.transpose(-2, -1), offs=offsets)

    outputs = []
    start = 0
    for expert, count in enumerate(counts.tolist()):
        stop = start + count
        if stop > start:
            outputs.append(F.linear(x[start:stop], weight[expert]))
        start = stop
    if outputs:
        return torch.cat(outputs)
    return x.new_empty((0, weight.shape[1]))


class RoutedExperts(nn.Module):
    def __init__(self, config: DeepSeekV41Config, dispatcher: TokenDispatcher) -> None:
        super().__init__()
        self.dispatcher = dispatcher
        self.global_num_experts = config.n_routed_experts
        self.expert_start, expert_stop = dispatcher.expert_range(config.n_routed_experts)
        self.num_experts = expert_stop - self.expert_start
        self.hidden_size = config.hidden_size
        self.intermediate = config.moe_intermediate_size
        self.limit = config.swiglu_limit
        self.gate_up = nn.Parameter(
            torch.empty(self.num_experts, self.intermediate, 2, self.hidden_size)
        )
        self.down = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_size, self.intermediate)
        )

    @property
    def local_parameter_count(self) -> int:
        return self.gate_up.numel() + self.down.numel()

    def forward(
        self, x: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        x, counts, weights, metadata = self.dispatcher.dispatch(x, indices, weights)
        if x.shape[0] == 0:
            anchor = self.gate_up.flatten()[0] + self.down.flatten()[0]
            output = x.new_empty((0, self.hidden_size)) + anchor.to(x.dtype) * 0
        else:
            gate_up = _grouped_mm(
                x,
                self.gate_up.reshape(
                    self.num_experts, 2 * self.intermediate, self.hidden_size
                ),
                counts,
            )
            gate, up = gate_up.reshape(-1, self.intermediate, 2).unbind(-1)
            hidden = clamped_swiglu(gate, up, weights, self.limit)
            output = _grouped_mm(hidden, self.down, counts)
        return self.dispatcher.combine(output, metadata)


class SharedExpert(nn.Module):
    def __init__(self, config: DeepSeekV41Config) -> None:
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.moe_intermediate_size, bias=False)
        self.up = nn.Linear(config.hidden_size, config.moe_intermediate_size, bias=False)
        self.down = nn.Linear(config.moe_intermediate_size, config.hidden_size, bias=False)
        self.limit = config.swiglu_limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = clamped_swiglu_reference(self.gate(x), self.up(x), self.limit)
        return self.down(hidden.to(x.dtype))


class SparseMoE(nn.Module):
    def __init__(self, config: DeepSeekV41Config, dispatcher: TokenDispatcher) -> None:
        super().__init__()
        self.router = TopKRouter(config)
        self.routed = RoutedExperts(config, dispatcher)
        self.shared = SharedExpert(config)
        self._router_aux_gradient: torch.Tensor | None = None

    def stage_router_aux_gradient(self, gradient: torch.Tensor) -> None:
        self._router_aux_gradient = gradient.detach()

    def _add_router_aux_gradient(self, gradient: torch.Tensor) -> torch.Tensor:
        auxiliary, self._router_aux_gradient = self._router_aux_gradient, None
        return gradient if auxiliary is None else gradient + auxiliary.to(gradient)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shape = x.shape
        flat = x.flatten(0, 1)
        logits, weights, indices = self.router(x)
        if logits.requires_grad:
            logits.register_hook(self._add_router_aux_gradient)
        output = self.routed(flat, indices, weights) + self.shared(flat).float()
        return output.to(x.dtype).view(shape), logits


__all__ = ["RoutedExperts", "SparseMoE", "TopKRouter"]
