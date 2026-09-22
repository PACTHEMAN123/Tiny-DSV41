"""Autograd-aware token dispatch for expert-parallel MoE models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh


@dataclass
class DispatchMetadata:
    token_count: int
    topk: int
    inverse_order: torch.Tensor | None = None
    send_splits: list[int] | None = None
    recv_splits: list[int] | None = None


class TokenDispatcher:
    """Keep token routing local when expert parallelism is disabled."""

    def expert_range(self, num_experts: int) -> tuple[int, int]:
        return 0, num_experts

    def dispatch(
        self,
        hidden: torch.Tensor,
        expert_ids: torch.Tensor,
        weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, DispatchMetadata]:
        topk = expert_ids.shape[-1]
        tokens = hidden[:, None, :].expand(-1, topk, -1).reshape(-1, hidden.shape[-1])
        metadata = DispatchMetadata(token_count=hidden.shape[0], topk=topk)
        return tokens, expert_ids.reshape(-1), weights.reshape(-1), metadata

    def combine(self, hidden: torch.Tensor, metadata: DispatchMetadata) -> torch.Tensor:
        return hidden.view(metadata.token_count, metadata.topk, -1).sum(dim=1)


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
        hidden: torch.Tensor,
        expert_ids: torch.Tensor,
        weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, DispatchMetadata]:
        if self.num_experts is None:
            raise RuntimeError("attach the dispatcher to an expert module before use")
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

        tokens = hidden[:, None, :].expand(-1, topk, -1).reshape(-1, hidden.shape[-1])[order]
        routed_ids = self._exchange(flat_ids[order].contiguous(), recv_splits, send_splits)
        routed_hidden = self._exchange(tokens.contiguous(), recv_splits, send_splits, autograd=True)
        routed_weights = self._exchange(
            weights.reshape(-1)[order].contiguous(),
            recv_splits,
            send_splits,
            autograd=True,
        )
        metadata = DispatchMetadata(
            token_count=hidden.shape[0],
            topk=topk,
            inverse_order=torch.argsort(order),
            send_splits=send_splits,
            recv_splits=recv_splits,
        )
        return routed_hidden, routed_ids - start, routed_weights, metadata

    def combine(self, hidden: torch.Tensor, metadata: DispatchMetadata) -> torch.Tensor:
        assert metadata.send_splits is not None
        assert metadata.recv_splits is not None
        assert metadata.inverse_order is not None
        returned = self._exchange(
            hidden.contiguous(),
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


__all__ = ["AllToAllTokenDispatcher", "DispatchMetadata", "TokenDispatcher"]
