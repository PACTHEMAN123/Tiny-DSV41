"""Triton fused score kernel for the DeepSeek V4 sparse indexer."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _fused_index_scores_kernel(
        query,
        keys,
        weights,
        key_starts,
        key_ends,
        output,
        sequence_length: tl.constexpr,
        key_length: tl.constexpr,
        heads: tl.constexpr,
        head_dim: tl.constexpr,
        output_width: tl.constexpr,
        scale: tl.constexpr,
        block_keys: tl.constexpr,
    ):
        row = tl.program_id(0)
        key_block = tl.program_id(1)
        batch = row // sequence_length
        sequence = row - batch * sequence_length

        key_offsets = key_block * block_keys + tl.arange(0, block_keys)
        head_offsets = tl.arange(0, heads)
        dim_offsets = tl.arange(0, head_dim)
        start = tl.load(key_starts + row)
        end = tl.load(key_ends + row)
        key_indices = start + key_offsets
        valid = (key_offsets < output_width) & (key_indices < end)

        query_base = ((batch * sequence_length + sequence) * heads) * head_dim
        query_tile = tl.load(
            query
            + query_base
            + dim_offsets[:, None] + head_offsets[None, :] * head_dim
        )
        key_base = (batch * key_length) * head_dim
        key_tile = tl.load(
            keys
            + key_base
            + key_indices[:, None] * head_dim
            + dim_offsets[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        dots = tl.dot(key_tile, query_tile)
        dots = tl.maximum(dots, 0.0) * scale
        head_weights = tl.load(
            weights
            + (batch * sequence_length + sequence) * heads
            + head_offsets
        )
        scores = tl.sum(dots * head_weights[None, :], axis=1)
        scores = tl.where(valid, scores, float("-inf"))
        tl.store(output + row * output_width + key_offsets, scores, mask=key_offsets < output_width)


def fused_index_scores(
    query: torch.Tensor,
    keys: torch.Tensor,
    weights: torch.Tensor,
    key_starts: torch.Tensor,
    key_ends: torch.Tensor,
    output_width: int,
) -> torch.Tensor:
    """Fuse QK, ReLU, head weighting, and head reduction without a head score tensor."""

    if triton is None:
        raise RuntimeError("the sparse indexer requires Triton")
    if query.device.type != "cuda":
        raise RuntimeError("the sparse indexer requires CUDA")
    if query.ndim != 4 or keys.ndim != 3 or weights.shape != query.shape[:-1]:
        raise ValueError("invalid fused indexer tensor shapes")
    batch, sequence_length, heads, head_dim = query.shape
    if (
        keys.shape[0] != batch
        or keys.shape[2] != head_dim
        or key_starts.shape != (batch, sequence_length)
        or key_ends.shape != key_starts.shape
    ):
        raise ValueError("fused indexer inputs disagree on batch or sequence dimensions")
    if heads % 16 or head_dim % 16:
        raise ValueError("fused indexer heads and head dimension must be multiples of 16")
    if output_width <= 0:
        return query.new_empty((batch, sequence_length, 0), dtype=torch.float32)

    query = query.contiguous()
    keys = keys.contiguous()
    weights = weights.float().contiguous()
    starts = key_starts.to(torch.int32).contiguous()
    ends = key_ends.to(torch.int32).contiguous()
    output = torch.empty(
        (batch, sequence_length, output_width),
        device=query.device,
        dtype=torch.float32,
    )
    block_keys = 64
    grid = (batch * sequence_length, triton.cdiv(output_width, block_keys))
    _fused_index_scores_kernel[grid](
        query,
        keys,
        weights,
        starts,
        ends,
        output,
        sequence_length=sequence_length,
        key_length=keys.shape[1],
        heads=heads,
        head_dim=head_dim,
        output_width=output_width,
        scale=head_dim**-0.5,
        block_keys=block_keys,
        num_warps=4,
        num_stages=3,
    )
    return output


__all__ = ["fused_index_scores"]
