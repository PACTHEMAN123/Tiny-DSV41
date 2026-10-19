"""Fixed-order sparse KV gradients with bounded per-query edge storage.

Each edge keeps the shared native Tensor Core formula. Stable inverse adjacency
retains duplicates, and one program owns each output tile. Chunk order is fixed.
"""
from __future__ import annotations

import torch

from .sparse_attention import tl, triton


def _inverse_adjacency(indices, mask, key_length):
    """Stable key ownership; retain duplicate slots and exclude invalid edges."""
    if indices.ndim != 3 or key_length < 0:
        raise ValueError('indices must be [B,L,slots] with nonnegative key length')
    if mask is not None and mask.shape != indices.shape:
        raise ValueError('mask shape differs from indices')
    batch = indices.shape[0]
    owners = batch * key_length
    valid = (indices >= 0) & (indices < key_length)
    if mask is not None:
        valid &= mask.bool()
    batch_offset = torch.arange(batch, device=indices.device, dtype=torch.int64)[:, None, None] * key_length
    encoded = torch.where(valid, indices.to(torch.int64) + batch_offset, owners).reshape(-1)
    order = torch.argsort(encoded, stable=True)
    counts = torch.bincount(encoded, minlength=owners + 1)
    offsets = torch.cat((counts.new_zeros(1), counts[:owners].cumsum(0)))
    return order, offsets



if triton is not None:
    @triton.jit
    def _reduce_edge_values(
        edge_values, edge_order, owner_offsets, grad_kv,
        chunk_start, chunk_length: tl.constexpr, chunk_capacity: tl.constexpr,
        sequence_length: tl.constexpr, kv_length: tl.constexpr,
        slots: tl.constexpr, combined_slots: tl.constexpr, branch_offset: tl.constexpr,
        head_blocks: tl.constexpr, head_dim: tl.constexpr,
        BLOCK_E: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        # Exactly one program owns each (batch,key,dimension tile). Kernel launches
        # are stream-ordered across chunks. No floating-point atomic is used.
        owner = tl.program_id(0)
        dim_block = tl.program_id(1)
        batch = owner // kv_length
        first = tl.load(owner_offsets + owner)
        last = tl.load(owner_offsets + owner + 1)
        lower_edge = (batch * sequence_length + chunk_start) * slots
        upper_edge = (batch * sequence_length + chunk_start + chunk_length) * slots
        # Stable sort by owner preserves ascending original edge ID within each
        # owner, enabling bounds without rereading every edge in every chunk.
        lo, hi = first, last
        while lo < hi:
            mid = (lo + hi) // 2
            value = tl.load(edge_order + mid)
            before = value < lower_edge
            lo = tl.where(before, mid + 1, lo)
            hi = tl.where(before, hi, mid)
        begin = lo
        lo, hi = begin, last
        while lo < hi:
            mid = (lo + hi) // 2
            value = tl.load(edge_order + mid)
            before = value < upper_edge
            lo = tl.where(before, mid + 1, lo)
            hi = tl.where(before, hi, mid)
        end = lo
        dims = dim_block * BLOCK_D + tl.arange(0, BLOCK_D)
        valid_dims = dims < head_dim
        total = tl.load(grad_kv + owner * head_dim + dims, mask=valid_dims, other=0.0)
        lanes = tl.arange(0, BLOCK_R)
        edge_lane = lanes // head_blocks
        head_block = lanes % head_blocks
        # Fixed groups of eight ascending (query,slot) edges, with head blocks
        # in ascending order; fixed tl.sum tree per group, then fixed chunk order.
        for position in range(begin, end, BLOCK_E):
            positions = position + edge_lane
            valid_edge = (edge_lane < BLOCK_E) & (positions < end)
            edge = tl.load(edge_order + positions, mask=valid_edge, other=0)
            local_query = edge // slots - batch * sequence_length - chunk_start
            slot = edge % slots + branch_offset
            offsets = ((((batch * chunk_capacity + local_query[:, None])
                         * combined_slots + slot[:, None]) * head_blocks
                        + head_block[:, None]) * head_dim + dims[None, :])
            contribution = tl.load(edge_values + offsets,
                mask=valid_edge[:, None] & valid_dims[None, :], other=0.0)
            total += tl.sum(contribution, axis=0)
        tl.store(grad_kv + owner * head_dim + dims, total, mask=valid_dims)


def ordered_backward(ctx, grad_output):
    """Native seven-result backward contract, with bounded edge storage."""
    if triton is None:
        raise RuntimeError('Triton required')
    from .sparse_attention import _attention_backward_kernel
    chunk_queries = 128
    (query, kv, wi, wm, compressed, ci, sinks, output, lse) = ctx.saved_tensors
    batch, heads, length, dim = query.shape
    if query.device.type != 'cuda':
        raise RuntimeError('CUDA required')
    head_blocks = triton.cdiv(heads, 16)
    capacity = min(length, chunk_queries)
    window, selected = wi.shape[-1], ci.shape[-1]
    dq = torch.empty_like(query)
    dk = torch.zeros_like(kv, dtype=torch.float32)
    dc = torch.zeros_like(compressed, dtype=torch.float32)
    ds = torch.empty_like(lse)
    upstream = grad_output.contiguous()
    edge_values = torch.empty((batch, capacity, window + selected, head_blocks, dim),
                              dtype=torch.float32, device=query.device)
    adjacency = []
    for keys, indices, mask, target, branch in (
        (kv, wi, wm, dk, 0), (compressed, ci, None, dc, window)):
        if keys.numel() and indices.numel():
            order, offsets = _inverse_adjacency(indices, mask, keys.shape[1])
            adjacency.append((order, offsets, target, indices.shape[-1], branch))
    for start in range(0, length, chunk_queries):
        size = min(capacity, length - start)
        _attention_backward_kernel[(batch * size, head_blocks)](
            query, kv, wi, wm, compressed, ci, sinks, output, lse,
            upstream, dq, None, None, ds, edge_values,
            ORDERED_KV=True,
            chunk_start=start, chunk_length=size, chunk_capacity=capacity,
            head_blocks=head_blocks, sequence_length=length,
            kv_length=kv.shape[1], compressed_length=compressed.shape[1],
            heads=heads, head_dim=dim, window=window, selected=selected,
            scale=dim**-0.5, BLOCK_H=16, BLOCK_N=16,
            BLOCK_D=triton.next_power_of_2(dim), num_warps=8, num_stages=2)
        for order, offsets, target, slots, branch in adjacency:
            _reduce_edge_values[(batch * target.shape[1], triton.cdiv(dim, 128))](
                edge_values, order, offsets, target,
                chunk_start=start, chunk_length=size, chunk_capacity=capacity,
                sequence_length=length, kv_length=target.shape[1], slots=slots,
                combined_slots=window + selected, branch_offset=branch,
                head_blocks=head_blocks, head_dim=dim, BLOCK_E=8,
                BLOCK_R=triton.next_power_of_2(8 * head_blocks), BLOCK_D=128,
                num_warps=4)
    return (dq, dk.to(kv.dtype), None, None, dc.to(compressed.dtype), None,
            ds.sum(dim=(0, 2), dtype=torch.float32).to(sinks.dtype))
