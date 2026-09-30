"""Triton sparse attention for DeepSeek V4 sliding and compressed KV."""

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
    def _attention_forward_kernel(
        query,
        kv,
        window_indices,
        window_mask,
        compressed_kv,
        compressed_indices,
        sinks,
        output,
        logsumexp,
        sequence_length: tl.constexpr,
        kv_length: tl.constexpr,
        compressed_length: tl.constexpr,
        heads: tl.constexpr,
        head_dim: tl.constexpr,
        window: tl.constexpr,
        selected: tl.constexpr,
        scale: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        sequence_row = tl.program_id(0)
        head_block = tl.program_id(1)
        batch = sequence_row // sequence_length
        sequence = sequence_row - batch * sequence_length
        head_offsets = head_block * BLOCK_H + tl.arange(0, BLOCK_H)
        dim_offsets = tl.arange(0, BLOCK_D)
        valid_heads = head_offsets < heads
        valid_dims = dim_offsets < head_dim

        query_offsets = (
            ((batch * heads + head_offsets[:, None]) * sequence_length + sequence)
            * head_dim
            + dim_offsets[None, :]
        )
        q = tl.load(
            query + query_offsets,
            mask=valid_heads[:, None] & valid_dims[None, :],
            other=0.0,
        )
        sink = tl.load(sinks + head_offsets, mask=valid_heads, other=0.0).to(tl.float32)
        maximum = sink
        denominator = tl.full((BLOCK_H,), 1.0, tl.float32)
        accumulator = tl.zeros((BLOCK_H, BLOCK_D), tl.float32)

        for start in range(0, window, BLOCK_N):
            candidate_offsets = start + tl.arange(0, BLOCK_N)
            candidate_valid = candidate_offsets < window
            index_base = (batch * sequence_length + sequence) * window
            indices = tl.load(
                window_indices + index_base + candidate_offsets,
                mask=candidate_valid,
                other=0,
            )
            candidate_valid &= tl.load(
                window_mask + index_base + candidate_offsets,
                mask=candidate_valid,
                other=0,
            ).to(tl.int1)
            candidate_valid &= (indices >= 0) & (indices < kv_length)
            kv_offsets = (
                (batch * kv_length + indices[:, None]) * head_dim
                + dim_offsets[None, :]
            )
            values = tl.load(
                kv + kv_offsets,
                mask=candidate_valid[:, None] & valid_dims[None, :],
                other=0.0,
            )
            scores = tl.dot(q, tl.trans(values)) * scale
            scores = tl.where(
                valid_heads[:, None] & candidate_valid[None, :],
                scores,
                float("-inf"),
            )
            block_maximum = tl.max(scores, axis=1)
            new_maximum = tl.maximum(maximum, block_maximum)
            alpha = tl.exp(maximum - new_maximum)
            probabilities = tl.exp(scores - new_maximum[:, None])
            accumulator = accumulator * alpha[:, None] + tl.dot(
                probabilities.to(values.dtype), values
            )
            denominator = denominator * alpha + tl.sum(probabilities, axis=1)
            maximum = new_maximum

        for start in range(0, selected, BLOCK_N):
            candidate_offsets = start + tl.arange(0, BLOCK_N)
            candidate_valid = candidate_offsets < selected
            index_base = (batch * sequence_length + sequence) * selected
            indices = tl.load(
                compressed_indices + index_base + candidate_offsets,
                mask=candidate_valid,
                other=-1,
            )
            candidate_valid &= (indices >= 0) & (indices < compressed_length)
            kv_offsets = (
                (batch * compressed_length + indices[:, None]) * head_dim
                + dim_offsets[None, :]
            )
            values = tl.load(
                compressed_kv + kv_offsets,
                mask=candidate_valid[:, None] & valid_dims[None, :],
                other=0.0,
            )
            scores = tl.dot(q, tl.trans(values)) * scale
            scores = tl.where(
                valid_heads[:, None] & candidate_valid[None, :],
                scores,
                float("-inf"),
            )
            block_maximum = tl.max(scores, axis=1)
            new_maximum = tl.maximum(maximum, block_maximum)
            alpha = tl.exp(maximum - new_maximum)
            probabilities = tl.exp(scores - new_maximum[:, None])
            accumulator = accumulator * alpha[:, None] + tl.dot(
                probabilities.to(values.dtype), values
            )
            denominator = denominator * alpha + tl.sum(probabilities, axis=1)
            maximum = new_maximum

        accumulator /= denominator[:, None]
        tl.store(
            output + query_offsets,
            accumulator,
            mask=valid_heads[:, None] & valid_dims[None, :],
        )
        lse_offsets = (batch * heads + head_offsets) * sequence_length + sequence
        tl.store(
            logsumexp + lse_offsets,
            maximum + tl.log(denominator),
            mask=valid_heads,
        )

    @triton.jit
    def _attention_backward_kernel(
        query,
        kv,
        window_indices,
        window_mask,
        compressed_kv,
        compressed_indices,
        sinks,
        output,
        logsumexp,
        grad_output,
        grad_query,
        grad_kv,
        grad_compressed_kv,
        grad_sinks,
        sequence_length: tl.constexpr,
        kv_length: tl.constexpr,
        compressed_length: tl.constexpr,
        heads: tl.constexpr,
        head_dim: tl.constexpr,
        window: tl.constexpr,
        selected: tl.constexpr,
        scale: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        sequence_row = tl.program_id(0)
        head_block = tl.program_id(1)
        batch = sequence_row // sequence_length
        sequence = sequence_row - batch * sequence_length
        head_offsets = head_block * BLOCK_H + tl.arange(0, BLOCK_H)
        dim_offsets = tl.arange(0, BLOCK_D)
        valid_heads = head_offsets < heads
        valid_dims = dim_offsets < head_dim
        query_offsets = (
            ((batch * heads + head_offsets[:, None]) * sequence_length + sequence)
            * head_dim
            + dim_offsets[None, :]
        )
        tensor_mask = valid_heads[:, None] & valid_dims[None, :]
        q = tl.load(query + query_offsets, mask=tensor_mask, other=0.0)
        o = tl.load(output + query_offsets, mask=tensor_mask, other=0.0)
        do = tl.load(grad_output + query_offsets, mask=tensor_mask, other=0.0)
        delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
        lse_offsets = (batch * heads + head_offsets) * sequence_length + sequence
        lse = tl.load(logsumexp + lse_offsets, mask=valid_heads, other=0.0)
        sink = tl.load(sinks + head_offsets, mask=valid_heads, other=0.0).to(tl.float32)
        sink_probability = tl.exp(sink - lse)
        tl.atomic_add(
            grad_sinks + head_offsets,
            -sink_probability * delta,
            mask=valid_heads,
        )
        dq = tl.zeros((BLOCK_H, BLOCK_D), tl.float32)

        for start in range(0, window, BLOCK_N):
            candidate_offsets = start + tl.arange(0, BLOCK_N)
            candidate_valid = candidate_offsets < window
            index_base = (batch * sequence_length + sequence) * window
            indices = tl.load(
                window_indices + index_base + candidate_offsets,
                mask=candidate_valid,
                other=0,
            )
            candidate_valid &= tl.load(
                window_mask + index_base + candidate_offsets,
                mask=candidate_valid,
                other=0,
            ).to(tl.int1)
            candidate_valid &= (indices >= 0) & (indices < kv_length)
            kv_offsets = (
                (batch * kv_length + indices[:, None]) * head_dim
                + dim_offsets[None, :]
            )
            values = tl.load(
                kv + kv_offsets,
                mask=candidate_valid[:, None] & valid_dims[None, :],
                other=0.0,
            )
            scores = tl.dot(q, tl.trans(values)) * scale
            probabilities = tl.exp(scores - lse[:, None])
            probabilities = tl.where(
                valid_heads[:, None] & candidate_valid[None, :],
                probabilities,
                0.0,
            )
            grad_probabilities = tl.dot(do, tl.trans(values))
            grad_scores = probabilities * (grad_probabilities - delta[:, None])
            dq += tl.dot(grad_scores.to(values.dtype), values) * scale
            grad_values = tl.dot(tl.trans(probabilities.to(do.dtype)), do)
            grad_values += tl.dot(tl.trans(grad_scores.to(q.dtype)), q) * scale
            tl.atomic_add(
                grad_kv + kv_offsets,
                grad_values,
                mask=candidate_valid[:, None] & valid_dims[None, :],
            )

        for start in range(0, selected, BLOCK_N):
            candidate_offsets = start + tl.arange(0, BLOCK_N)
            candidate_valid = candidate_offsets < selected
            index_base = (batch * sequence_length + sequence) * selected
            indices = tl.load(
                compressed_indices + index_base + candidate_offsets,
                mask=candidate_valid,
                other=-1,
            )
            candidate_valid &= (indices >= 0) & (indices < compressed_length)
            kv_offsets = (
                (batch * compressed_length + indices[:, None]) * head_dim
                + dim_offsets[None, :]
            )
            values = tl.load(
                compressed_kv + kv_offsets,
                mask=candidate_valid[:, None] & valid_dims[None, :],
                other=0.0,
            )
            scores = tl.dot(q, tl.trans(values)) * scale
            probabilities = tl.exp(scores - lse[:, None])
            probabilities = tl.where(
                valid_heads[:, None] & candidate_valid[None, :],
                probabilities,
                0.0,
            )
            grad_probabilities = tl.dot(do, tl.trans(values))
            grad_scores = probabilities * (grad_probabilities - delta[:, None])
            dq += tl.dot(grad_scores.to(values.dtype), values) * scale
            grad_values = tl.dot(tl.trans(probabilities.to(do.dtype)), do)
            grad_values += tl.dot(tl.trans(grad_scores.to(q.dtype)), q) * scale
            tl.atomic_add(
                grad_compressed_kv + kv_offsets,
                grad_values,
                mask=candidate_valid[:, None] & valid_dims[None, :],
            )

        tl.store(grad_query + query_offsets, dq, mask=tensor_mask)


class _SparseAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        kv: torch.Tensor,
        window_indices: torch.Tensor,
        window_mask: torch.Tensor,
        compressed_kv: torch.Tensor,
        compressed_indices: torch.Tensor,
        sinks: torch.Tensor,
    ) -> torch.Tensor:
        batch, heads, sequence_length, head_dim = query.shape
        window = window_indices.shape[-1]
        selected = compressed_indices.shape[-1]
        output = torch.empty_like(query)
        logsumexp = torch.empty(
            (batch, heads, sequence_length), device=query.device, dtype=torch.float32
        )
        block_h = 16
        block_n = 16
        block_d = triton.next_power_of_2(head_dim)
        grid = (batch * sequence_length, triton.cdiv(heads, block_h))
        _attention_forward_kernel[grid](
            query,
            kv,
            window_indices,
            window_mask,
            compressed_kv,
            compressed_indices,
            sinks,
            output,
            logsumexp,
            sequence_length=sequence_length,
            kv_length=kv.shape[1],
            compressed_length=compressed_kv.shape[1],
            heads=heads,
            head_dim=head_dim,
            window=window,
            selected=selected,
            scale=head_dim**-0.5,
            BLOCK_H=block_h,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            num_warps=8,
            num_stages=2,
        )
        ctx.save_for_backward(
            query,
            kv,
            window_indices,
            window_mask,
            compressed_kv,
            compressed_indices,
            sinks,
            output,
            logsumexp,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (
            query,
            kv,
            window_indices,
            window_mask,
            compressed_kv,
            compressed_indices,
            sinks,
            output,
            logsumexp,
        ) = ctx.saved_tensors
        batch, heads, sequence_length, head_dim = query.shape
        grad_query = torch.empty_like(query)
        grad_kv_fp32 = torch.zeros_like(kv, dtype=torch.float32)
        grad_compressed_fp32 = torch.zeros_like(compressed_kv, dtype=torch.float32)
        grad_sinks = torch.zeros_like(sinks, dtype=torch.float32)
        block_h = 16
        block_n = 16
        block_d = triton.next_power_of_2(head_dim)
        grid = (batch * sequence_length, triton.cdiv(heads, block_h))
        _attention_backward_kernel[grid](
            query,
            kv,
            window_indices,
            window_mask,
            compressed_kv,
            compressed_indices,
            sinks,
            output,
            logsumexp,
            grad_output.contiguous(),
            grad_query,
            grad_kv_fp32,
            grad_compressed_fp32,
            grad_sinks,
            sequence_length=sequence_length,
            kv_length=kv.shape[1],
            compressed_length=compressed_kv.shape[1],
            heads=heads,
            head_dim=head_dim,
            window=window_indices.shape[-1],
            selected=compressed_indices.shape[-1],
            scale=head_dim**-0.5,
            BLOCK_H=block_h,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            num_warps=8,
            num_stages=2,
        )
        return (
            grad_query,
            grad_kv_fp32.to(kv.dtype),
            None,
            None,
            grad_compressed_fp32.to(compressed_kv.dtype),
            None,
            grad_sinks.to(sinks.dtype),
        )


def sparse_attention(
    query: torch.Tensor,
    kv: torch.Tensor,
    window_indices: torch.Tensor,
    window_mask: torch.Tensor,
    compressed_kv: torch.Tensor | None,
    compressed_indices: torch.Tensor | None,
    sinks: torch.Tensor,
) -> torch.Tensor:
    """Apply fused sparse attention without materializing score tensors."""

    if triton is None:
        raise RuntimeError("Triton attention requires Triton")
    if query.device.type != "cuda":
        raise RuntimeError("Triton attention requires CUDA")
    if query.ndim != 4 or kv.ndim != 3:
        raise ValueError("query and KV must have shapes [B,H,L,D] and [B,S,D]")
    batch, _, sequence_length, head_dim = query.shape
    if kv.shape[0] != batch or kv.shape[2] != head_dim:
        raise ValueError("query and KV dimensions do not match")
    if window_indices.shape[:2] != (batch, sequence_length):
        raise ValueError("window indices do not match query dimensions")
    if window_mask.shape != window_indices.shape:
        raise ValueError("window mask must match window indices")
    if head_dim % 16 or head_dim > 512:
        raise ValueError("Triton attention requires a head dimension in 16..512")

    if compressed_kv is None:
        compressed_kv = kv.new_empty((batch, 0, head_dim))
        compressed_indices = window_indices.new_empty((batch, sequence_length, 0))
    elif compressed_indices is None:
        raise ValueError("compressed indices are required with compressed KV")
    if compressed_kv.shape[0] != batch or compressed_kv.shape[2] != head_dim:
        raise ValueError("compressed KV dimensions do not match query")
    if compressed_indices.shape[:2] != (batch, sequence_length):
        raise ValueError("compressed indices do not match query dimensions")

    return _SparseAttention.apply(
        query.contiguous(),
        kv.contiguous(),
        window_indices.to(torch.int32).contiguous(),
        window_mask.bool().contiguous(),
        compressed_kv.contiguous(),
        compressed_indices.to(torch.int32).contiguous(),
        sinks.contiguous(),
    )


def is_available(tensor: torch.Tensor) -> bool:
    return triton is not None and tensor.device.type == "cuda"


__all__ = ["is_available", "sparse_attention"]
