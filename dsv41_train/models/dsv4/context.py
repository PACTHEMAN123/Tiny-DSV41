"""Packed-sequence metadata and context-parallel communication."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ...parallel import ContextParallel


@dataclass(frozen=True)
class ModelContext:
    parallel: ContextParallel
    input_ids: torch.Tensor
    positions: torch.Tensor
    token_mask: torch.Tensor
    sequence_ids: torch.Tensor
    attention_indices: torch.Tensor
    attention_mask: torch.Tensor

    @classmethod
    def build(
        cls,
        parallel: ContextParallel,
        input_ids: torch.Tensor,
        token_mask: torch.Tensor,
        sequence_ids: torch.Tensor,
        window: int,
    ) -> "ModelContext":
        length = input_ids.shape[1]
        indices = torch.arange(length, device=input_ids.device).expand_as(input_ids)
        starts = token_mask & (sequence_ids != F.pad(sequence_ids[:, :-1], (1, 0), value=-1))
        positions = (indices - torch.where(starts, indices, 0).cummax(-1).values).masked_fill(
            ~token_mask, 0
        )
        local_indices = parallel.shard(indices)
        key_indices = local_indices.unsqueeze(-1) - torch.arange(
            window - 1, -1, -1, device=input_ids.device
        )
        valid = key_indices >= 0
        key_indices = key_indices.clamp_min(0)
        key_sequences = sequence_ids.gather(1, key_indices.flatten(1)).view_as(key_indices)
        key_mask = token_mask.gather(1, key_indices.flatten(1)).view_as(key_indices)
        local_input, local_positions, local_mask, local_sequences = map(
            parallel.shard, (input_ids, positions, token_mask, sequence_ids)
        )
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
            key_indices,
            attention_mask.unsqueeze(1),
        )

    def gather(self, tensor: torch.Tensor, dim: int = 1) -> torch.Tensor:
        return self.parallel.gather(tensor, dim)


def pack_sequences(input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten a batch and preserve its original sequence boundaries."""

    batch, length = input_ids.shape
    sequence_ids = torch.arange(batch, device=input_ids.device).repeat_interleave(length)
    return input_ids.reshape(1, -1), sequence_ids.reshape(1, -1)


__all__ = ["ModelContext", "pack_sequences"]
