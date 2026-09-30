"""Context-parallel collectives and packed-sequence metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh


class _AllGather(torch.autograd.Function):
    """Autograd all-gather whose backward stays inside the given subgroup."""

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, dim: int, group, size: int) -> torch.Tensor:
        ctx.dim = dim
        ctx.group = group
        ctx.size = size
        parts = [torch.empty_like(tensor) for _ in range(size)]
        dist.all_gather(parts, tensor.contiguous(), group=group)
        return torch.cat(parts, dim=dim)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_output = grad_output.movedim(ctx.dim, 0).contiguous()
        exchanged = torch.empty_like(grad_output)
        dist.all_to_all_single(exchanged, grad_output, group=ctx.group)
        local_grad = torch.stack(exchanged.chunk(ctx.size, dim=0)).sum(dim=0)
        return local_grad.movedim(0, ctx.dim).contiguous(), None, None, None


def _ring_exchange(
    tensor: torch.Tensor,
    *,
    send_rank: int,
    recv_rank: int,
    group,
    dim: int,
) -> torch.Tensor:
    """Exchange one fixed-size slice with neighboring ranks."""

    size = dist.get_world_size(group)
    flattened = tensor.movedim(dim, 0).contiguous()
    tokens = flattened.shape[0]
    input_splits = [0] * size
    output_splits = [0] * size
    input_splits[send_rank] = tokens
    output_splits[recv_rank] = tokens
    received = torch.empty_like(flattened)
    dist.all_to_all_single(
        received,
        flattened,
        output_split_sizes=output_splits,
        input_split_sizes=input_splits,
        group=group,
    )
    return received.movedim(0, dim).contiguous()


class _LeftHalo(torch.autograd.Function):
    """Prepend the previous CP rank's tail and route its gradient back."""

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, length: int, dim: int, group) -> torch.Tensor:
        rank = dist.get_rank(group)
        size = dist.get_world_size(group)
        ctx.rank = rank
        ctx.size = size
        ctx.group = group
        ctx.length = length
        ctx.dim = dim

        tail = tensor.narrow(dim, tensor.shape[dim] - length, length)
        prefix = _ring_exchange(
            tail,
            send_rank=(rank + 1) % size,
            recv_rank=(rank - 1) % size,
            group=group,
            dim=dim,
        )
        if rank == 0:
            return tensor
        return torch.cat((prefix, tensor), dim=dim)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if ctx.rank == 0:
            prefix_shape = list(grad_output.shape)
            prefix_shape[ctx.dim] = ctx.length
            prefix_grad = grad_output.new_zeros(prefix_shape)
            local_grad = grad_output.contiguous()
        else:
            prefix_grad = grad_output.narrow(ctx.dim, 0, ctx.length).contiguous()
            local_grad = grad_output.narrow(
                ctx.dim, ctx.length, grad_output.shape[ctx.dim] - ctx.length
            ).contiguous()

        received = _ring_exchange(
            prefix_grad,
            send_rank=(ctx.rank - 1) % ctx.size,
            recv_rank=(ctx.rank + 1) % ctx.size,
            group=ctx.group,
            dim=ctx.dim,
        )
        result = local_grad.clone()
        if ctx.rank < ctx.size - 1:
            result.narrow(ctx.dim, result.shape[ctx.dim] - ctx.length, ctx.length).add_(
                received
            )
        return result, None, None, None


class ContextParallel:
    """Shard tokens and communicate context required by sequence modules."""

    def __init__(self, mesh: DeviceMesh | None = None) -> None:
        self.mesh = mesh
        self.size = 1 if mesh is None else mesh.size()
        self.rank = 0 if mesh is None else mesh.get_local_rank()
        self.group = None if mesh is None else mesh.get_group()

    @property
    def enabled(self) -> bool:
        return self.size > 1

    def shard(self, tensor: torch.Tensor, dim: int = 1) -> torch.Tensor:
        if not self.enabled:
            return tensor
        if tensor.shape[dim] % self.size:
            raise ValueError(
                f"tensor dimension {dim} ({tensor.shape[dim]}) must divide CP size ({self.size})"
            )
        return tensor.chunk(self.size, dim=dim)[self.rank].contiguous()

    def gather(self, tensor: torch.Tensor, dim: int = 1) -> torch.Tensor:
        if not self.enabled:
            return tensor
        tensor = tensor.contiguous()
        if tensor.requires_grad:
            return _AllGather.apply(tensor, dim, self.group, self.size)
        parts = [torch.empty_like(tensor) for _ in range(self.size)]
        dist.all_gather(parts, tensor, group=self.group)
        return torch.cat(parts, dim=dim)

    def redistribute_selected(
        self,
        tensor: torch.Tensor,
        source_indices: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Select global source tokens and shard the compacted result over CP."""

        if tensor.ndim < 2 or source_indices.ndim != 2:
            raise ValueError(
                "selected CP tensors must include batch and sequence dimensions"
            )
        if (
            tensor.shape[0] != source_indices.shape[0]
            or token_mask.shape != source_indices.shape
        ):
            raise ValueError("selected token metadata must match the tensor batch dimension")

        local_length = tensor.shape[1]
        start = self.rank * local_length
        local_indices = (source_indices - start).clamp(0, max(local_length - 1, 0))
        gather_indices = local_indices.view(
            *local_indices.shape, *([1] * (tensor.ndim - 2))
        ).expand(*local_indices.shape, *tensor.shape[2:])
        selected = tensor.gather(1, gather_indices)
        owned = (
            token_mask
            & source_indices.ge(start)
            & source_indices.lt(start + local_length)
        )
        selected = selected.masked_fill(
            ~owned.view(*owned.shape, *([1] * (tensor.ndim - 2))), 0
        )
        selected = self.sum(selected, autograd=tensor.requires_grad)
        return self.shard(selected)

    def left_halo(
        self, tensor: torch.Tensor, length: int, dim: int = 1
    ) -> torch.Tensor:
        """Prepend only the context required from the previous CP rank."""

        if not self.enabled or length == 0:
            return tensor
        if length < 0 or length > tensor.shape[dim]:
            raise ValueError(
                f"halo length ({length}) must be in [0, {tensor.shape[dim]}]"
            )
        return _LeftHalo.apply(tensor.contiguous(), length, dim, self.group)

    def sum(self, tensor: torch.Tensor, *, autograd: bool = False) -> torch.Tensor:
        if not self.enabled:
            return tensor
        if autograd:
            return dist_nn.all_reduce(tensor, op=dist.ReduceOp.SUM, group=self.group)
        result = tensor.clone()
        dist.all_reduce(result, op=dist.ReduceOp.SUM, group=self.group)
        return result

    def mean_loss(self, local_sum: torch.Tensor, local_count: torch.Tensor) -> torch.Tensor:
        """Return the global CP mean while keeping FSDP's averaging correct."""

        if not self.enabled:
            return local_sum / local_count.clamp_min(1)

        total = self.sum(local_sum.detach())
        count = self.sum(local_count.detach()).clamp_min(1)
        backward_loss = local_sum * (self.size / count)
        value = total / count
        return backward_loss + (value - backward_loss.detach())


@dataclass(frozen=True)
class ModelContext:
    parallel: ContextParallel
    input_ids: torch.Tensor
    positions: torch.Tensor
    token_mask: torch.Tensor
    sequence_ids: torch.Tensor
    attention_indices: torch.Tensor
    attention_mask: torch.Tensor
    attention_halo_length: int

    @classmethod
    def build(
        cls,
        parallel: ContextParallel,
        input_ids: torch.Tensor,
        token_mask: torch.Tensor,
        sequence_ids: torch.Tensor,
        window: int,
        positions: torch.Tensor | None = None,
    ) -> "ModelContext":
        length = input_ids.shape[1]
        indices = torch.arange(length, device=input_ids.device).expand_as(input_ids)
        if positions is None:
            positions = sequence_positions(token_mask, sequence_ids)
        elif positions.shape != input_ids.shape:
            raise ValueError("positions must have the same shape as input_ids")
        positions = positions.masked_fill(~token_mask, 0)
        local_indices = parallel.shard(indices)
        global_key_indices = local_indices.unsqueeze(-1) - torch.arange(
            window - 1, -1, -1, device=input_ids.device
        )
        valid = global_key_indices >= 0
        global_key_indices = global_key_indices.clamp_min(0)
        key_sequences = sequence_ids.gather(
            1, global_key_indices.flatten(1)
        ).view_as(global_key_indices)
        key_mask = token_mask.gather(
            1, global_key_indices.flatten(1)
        ).view_as(global_key_indices)
        local_input, local_positions, local_mask, local_sequences = map(
            parallel.shard, (input_ids, positions, token_mask, sequence_ids)
        )
        local_length = local_input.shape[1]
        halo_length = (
            window - 1 if parallel.enabled and window - 1 <= local_length else 0
        )
        if halo_length:
            local_start = parallel.rank * local_length
            prefix_length = 0 if parallel.rank == 0 else halo_length
            attention_indices = global_key_indices - local_start + prefix_length
        else:
            attention_indices = global_key_indices
        attention_mask = (
            valid
            & key_mask
            & local_mask.unsqueeze(-1)
            & (local_sequences.unsqueeze(-1) == key_sequences)
        )
        return cls(
            parallel,
            local_input,
            local_positions,
            local_mask,
            local_sequences,
            attention_indices,
            attention_mask.unsqueeze(1),
            halo_length,
        )

    def gather(self, tensor: torch.Tensor, dim: int = 1) -> torch.Tensor:
        return self.parallel.gather(tensor, dim)

    def gather_attention(self, tensor: torch.Tensor, dim: int = 1) -> torch.Tensor:
        if self.attention_halo_length:
            return self.parallel.left_halo(tensor, self.attention_halo_length, dim)
        return self.parallel.gather(tensor, dim)


@dataclass(frozen=True)
class ReplaySelection:
    input_ids: torch.Tensor
    positions: torch.Tensor
    token_mask: torch.Tensor
    sequence_ids: torch.Tensor
    source_indices: torch.Tensor


def sequence_positions(
    token_mask: torch.Tensor, sequence_ids: torch.Tensor
) -> torch.Tensor:
    """Return zero-based positions for each contiguous packed sequence."""

    if token_mask.shape != sequence_ids.shape:
        raise ValueError("token mask and sequence IDs must have matching shapes")
    length = token_mask.shape[1]
    indices = torch.arange(length, device=token_mask.device).expand_as(sequence_ids)
    starts = token_mask & (
        sequence_ids != F.pad(sequence_ids[:, :-1], (1, 0), value=-1)
    )
    return (
        indices - torch.where(starts, indices, 0).cummax(-1).values
    ).masked_fill(~token_mask, 0)


def bounded_replay_selection(
    input_ids: torch.Tensor,
    token_mask: torch.Tensor,
    sequence_ids: torch.Tensor,
    positions: torch.Tensor,
    window: int,
    shard_size: int = 1,
) -> ReplaySelection:
    """Pack the last replay window of every sequence while preserving positions."""

    if window < 1 or shard_size < 1:
        raise ValueError("replay window and shard size must be positive")
    if not (
        input_ids.shape
        == token_mask.shape
        == sequence_ids.shape
        == positions.shape
    ):
        raise ValueError("replay inputs must have matching batch and sequence shapes")

    length = input_ids.shape[1]
    reverse_ids = sequence_ids.flip(1)
    reverse_mask = token_mask.flip(1)
    reverse_indices = torch.arange(length, device=input_ids.device).expand_as(input_ids)
    reverse_starts = reverse_mask & (
        reverse_ids != F.pad(reverse_ids[:, :-1], (1, 0), value=-1)
    )
    reverse_positions = (
        reverse_indices
        - torch.where(reverse_starts, reverse_indices, 0).cummax(-1).values
    ).masked_fill(~reverse_mask, 0)
    replay_mask = token_mask & reverse_positions.flip(1).lt(window)

    counts = replay_mask.sum(-1)
    max_count = max(int(counts.max().item()), 1)
    padded_count = ((max_count + shard_size - 1) // shard_size) * shard_size
    source_indices = torch.zeros(
        input_ids.shape[0], padded_count, device=input_ids.device, dtype=torch.long
    )
    rows, columns = replay_mask.nonzero(as_tuple=True)
    destinations = replay_mask.long().cumsum(-1)[rows, columns] - 1
    source_indices[rows, destinations] = columns
    compact_mask = (
        torch.arange(padded_count, device=input_ids.device).unsqueeze(0)
        < counts.unsqueeze(1)
    )

    selected_ids = input_ids.gather(1, source_indices).masked_fill(~compact_mask, 0)
    selected_positions = positions.gather(1, source_indices).masked_fill(~compact_mask, 0)
    selected_sequences = sequence_ids.gather(1, source_indices).masked_fill(
        ~compact_mask, -1
    )
    return ReplaySelection(
        selected_ids,
        selected_positions,
        compact_mask,
        selected_sequences,
        source_indices,
    )


def pack_sequences(input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten a batch and preserve its original sequence boundaries."""

    batch, length = input_ids.shape
    sequence_ids = torch.arange(batch, device=input_ids.device).repeat_interleave(length)
    return input_ids.reshape(1, -1), sequence_ids.reshape(1, -1)


__all__ = [
    "ContextParallel",
    "ModelContext",
    "ReplaySelection",
    "bounded_replay_selection",
    "pack_sequences",
    "sequence_positions",
]
