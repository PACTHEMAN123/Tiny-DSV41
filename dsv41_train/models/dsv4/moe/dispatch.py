"""Expert-major token dispatch for DSV4 grouped experts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from torch.distributed._functional_collectives import all_to_all_single

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh


@dataclass
class DispatchMetadata:
    token_count: int
    token_indices: torch.Tensor
    rank_major_order: torch.Tensor | None = None
    send_splits: list[int] | None = None
    recv_splits: list[int] | None = None


class TokenDispatcher:
    """Sort local token assignments into expert-major order."""

    def __init__(self) -> None:
        self.num_experts: int | None = None

    def expert_range(self, num_experts: int) -> tuple[int, int]:
        if num_experts < 1:
            raise ValueError("num_experts must be positive")
        self.num_experts = num_experts
        return 0, num_experts

    def dispatch(
        self,
        hidden: torch.Tensor,
        expert_ids: torch.Tensor,
        weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, DispatchMetadata]:
        if self.num_experts is None:
            raise RuntimeError("attach the dispatcher to an expert module before use")
        topk = expert_ids.shape[-1]
        flat_ids = expert_ids.reshape(-1)
        order = torch.argsort(flat_ids, stable=True)
        token_indices = torch.div(order, topk, rounding_mode="floor")
        counts = torch.bincount(flat_ids, minlength=self.num_experts).to(torch.int32)
        metadata = DispatchMetadata(hidden.shape[0], token_indices)
        return hidden[token_indices], counts, weights.reshape(-1)[order], metadata

    def combine(self, hidden: torch.Tensor, metadata: DispatchMetadata) -> torch.Tensor:
        output = torch.zeros(
            metadata.token_count,
            hidden.shape[-1],
            device=hidden.device,
            dtype=torch.float32,
        )
        return output.index_add(0, metadata.token_indices, hidden.float())


class AllToAllTokenDispatcher(TokenDispatcher):
    """Dispatch assignments by owner rank, then reorder them by local expert."""

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

        order = torch.argsort(flat_ids, stable=True)
        token_indices = torch.div(order, topk, rounding_mode="floor")
        counts = torch.bincount(flat_ids, minlength=self.num_experts)
        send_counts = counts.view(self.size, local_experts).contiguous()
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts, group=self.group)
        send_splits = send_counts.sum(1).tolist()
        recv_splits = recv_counts.sum(1).tolist()

        routed_hidden = self._exchange(
            hidden[token_indices].contiguous(), recv_splits, send_splits
        )
        routed_weights = self._exchange(
            weights.reshape(-1)[order].contiguous(),
            recv_splits,
            send_splits,
        )
        expert_order = self._expert_major_order(recv_counts, routed_hidden.shape[0])
        metadata = DispatchMetadata(
            token_count=hidden.shape[0],
            token_indices=token_indices,
            rank_major_order=torch.argsort(expert_order),
            send_splits=send_splits,
            recv_splits=recv_splits,
        )
        return (
            routed_hidden[expert_order],
            recv_counts.sum(0).to(torch.int32),
            routed_weights[expert_order],
            metadata,
        )

    def combine(self, hidden: torch.Tensor, metadata: DispatchMetadata) -> torch.Tensor:
        assert metadata.rank_major_order is not None
        assert metadata.send_splits is not None
        assert metadata.recv_splits is not None
        returned = self._exchange(
            hidden[metadata.rank_major_order].contiguous(),
            metadata.send_splits,
            metadata.recv_splits,
        )
        output = torch.zeros(
            metadata.token_count,
            returned.shape[-1],
            device=returned.device,
            dtype=torch.float32,
        )
        return output.index_add(0, metadata.token_indices, returned.float())

    @staticmethod
    def _expert_major_order(counts: torch.Tensor, total: int) -> torch.Tensor:
        flat_counts = counts.reshape(-1)
        input_starts = (flat_counts.cumsum(0) - flat_counts).view_as(counts)
        segment_lengths = counts.t().reshape(-1)
        input_starts = input_starts.t().reshape(-1)
        segment_ids = torch.arange(
            segment_lengths.shape[0], device=counts.device
        ).repeat_interleave(segment_lengths, output_size=total)
        output_starts = segment_lengths.cumsum(0) - segment_lengths
        return (
            input_starts[segment_ids]
            + torch.arange(total, device=counts.device)
            - output_starts[segment_ids]
        )

    def _exchange(
        self,
        tensor: torch.Tensor,
        output_splits: list[int],
        input_splits: list[int],
    ) -> torch.Tensor:
        return all_to_all_single(
            tensor,
            output_split_sizes=output_splits,
            input_split_sizes=input_splits,
            group=self.mesh,
        )


__all__ = ["AllToAllTokenDispatcher", "DispatchMetadata", "TokenDispatcher"]
