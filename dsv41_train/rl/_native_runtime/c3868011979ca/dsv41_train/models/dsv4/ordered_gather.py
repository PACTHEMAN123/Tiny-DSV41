"""First-order compressed-KV gather with fixed-order FP32 gradient accumulation."""
from __future__ import annotations

import torch
from torch.autograd.function import once_differentiable

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = tl = None


if triton is not None:
    @triton.jit
    def _reduce_gather_gradient(upstream, order, offsets, output,
                                dim: tl.constexpr, BLOCK_E: tl.constexpr,
                                BLOCK_D: tl.constexpr):
        # One program owns each output tile; stable groups of eight edges match
        # the main attention's ordered_kv reduction. No floating atomics.
        owner = tl.program_id(0)
        dims = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
        begin, end = tl.load(offsets + owner), tl.load(offsets + owner + 1)
        total = tl.full((BLOCK_D,), 0, tl.float32)
        lanes = tl.arange(0, BLOCK_E)
        for position in range(begin, end, BLOCK_E):
            present = position + lanes < end
            edges = tl.load(order + position + lanes, mask=present, other=0)
            values = tl.load(upstream + edges[:, None] * dim + dims[None, :],
                            mask=present[:, None] & (dims[None, :] < dim), other=0).to(tl.float32)
            total += tl.sum(values, axis=0)
        tl.store(output + owner * dim + dims, total, mask=dims < dim)


class _MaskedGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, entries, indices):
        ctx.save_for_backward(indices)
        ctx.entry_shape, ctx.entry_dtype = entries.shape, entries.dtype
        rows = torch.arange(entries.shape[0], device=entries.device)[:, None, None]
        selected = entries[rows, indices.clamp_min(0)]
        return selected.masked_fill_(indices.lt(0).unsqueeze(-1), 0)

    @staticmethod
    @once_differentiable
    def backward(ctx, upstream):
        (indices,) = ctx.saved_tensors
        batch, keys, dim = ctx.entry_shape
        owners = batch * keys
        batch_offset = torch.arange(batch, device=indices.device, dtype=torch.int64)[:, None, None] * keys
        # Invalid outputs are constant zero; they do not own a gradient edge.
        encoded = torch.where(indices >= 0, indices.to(torch.int64) + batch_offset, owners).reshape(-1)
        order = torch.argsort(encoded, stable=True)
        counts = torch.bincount(encoded, minlength=owners + 1)
        offsets = torch.cat((counts.new_zeros(1), counts[:owners].cumsum(0)))
        gradient = torch.zeros(ctx.entry_shape, device=upstream.device, dtype=ctx.entry_dtype)
        if indices.numel():
            _reduce_gather_gradient[(owners, triton.cdiv(dim, 128))](
                upstream.contiguous(), order, offsets, gradient, dim=dim,
                BLOCK_E=8, BLOCK_D=128, num_warps=4)
        return gradient, None


def gather_compressed_kv(entries, indices):
    """Gather [B,K,D] into [B,S,T,D]; negative IDs return constant zero.

    CUDA BF16/FP32 backward uses the main attention's stable edge order and FP32
    reduction. The optimized path supports first-order training gradients only.
    """
    if entries.is_cuda and entries.dtype in (torch.bfloat16, torch.float32) and triton is not None:
        return _MaskedGather.apply(entries, indices)
    rows = torch.arange(entries.shape[0], device=entries.device)[:, None, None]
    source = entries.float() if entries.dtype in (torch.bfloat16, torch.float16) else entries
    selected = source[rows, indices.clamp_min(0)].to(entries.dtype)
    return selected.masked_fill_(indices.lt(0).unsqueeze(-1), 0)
