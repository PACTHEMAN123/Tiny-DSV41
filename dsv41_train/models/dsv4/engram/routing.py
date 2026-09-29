"""Owner routing, independent of the device holding the table."""

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn


class _ScaleGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, scale):
        ctx.scale = scale
        return tensor

    @staticmethod
    def backward(ctx, gradient):
        return gradient * ctx.scale, None


class _ExpandRows(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rows, inverse):
        ctx.save_for_backward(inverse)
        ctx.shape = rows.shape
        return rows[inverse]

    @staticmethod
    def backward(ctx, gradient):
        (inverse,) = ctx.saved_tensors
        # Match sparse SGD coalescing without repeated low-precision rounding.
        values = gradient.float() if gradient.dtype in (torch.float16, torch.bfloat16) else gradient
        result = values.new_zeros(ctx.shape).index_add_(0, inverse, values)
        return result.to(gradient.dtype), None


@dataclass
class RowRequest:
    shape: torch.Size
    local_ids: torch.Tensor
    inverse: torch.Tensor | None
    order: torch.Tensor | None
    send: list[int]
    recv: list[int]


class RowRouter:
    def __init__(self, num_embeddings, mesh=None):
        if mesh is not None and mesh.ndim != 1:
            raise ValueError("the embedding mesh must be one-dimensional")
        self.size = 1 if mesh is None else mesh.size()
        self.rank = 0 if mesh is None else mesh.get_local_rank()
        self.group = None if mesh is None else mesh.get_group()
        if num_embeddings < self.size:
            raise ValueError("embedding rows must be at least the embedding mesh size")
        self.num_embeddings = num_embeddings
        base, remainder = divmod(num_embeddings, self.size)
        self.row_start = self.rank * base + min(self.rank, remainder)
        self.row_stop = self.row_start + base + int(self.rank < remainder)

    def prepare(self, indices, deduplicate=False):
        flat = indices.reshape(-1)
        if flat.numel() and (flat.min() < 0 or flat.max() >= self.num_embeddings):
            raise IndexError("embedding index is outside the configured row range")
        inverse = None
        if deduplicate:
            flat, inverse = torch.unique(flat, sorted=True, return_inverse=True)
        if self.size == 1:
            return RowRequest(indices.shape, flat, inverse, None, [], [])
        base, remainder = divmod(self.num_embeddings, self.size)
        owners = torch.where(
            flat < (base + 1) * remainder,
            torch.div(flat, base + 1, rounding_mode="floor"),
            remainder + torch.div(flat - (base + 1) * remainder, base, rounding_mode="floor"),
        )
        order = torch.argsort(owners, stable=True)
        send = torch.bincount(owners, minlength=self.size).to(torch.int64)
        recv = torch.empty_like(send)
        dist.all_to_all_single(recv, send, group=self.group)
        send, recv = send.tolist(), recv.tolist()
        routed = self.exchange(flat[order], recv, send, autograd=False)
        return RowRequest(indices.shape, routed - self.row_start, inverse, order, send, recv)

    def complete(self, rows, request):
        if self.size > 1:
            rows = _ScaleGradient.apply(rows, 1.0 / self.size)
            rows = self.exchange(rows.contiguous(), request.send, request.recv, autograd=True)
            rows = rows[torch.argsort(request.order)]
        if request.inverse is not None:
            rows = _ExpandRows.apply(rows, request.inverse)
        return rows.view(*request.shape, rows.shape[-1])

    def exchange(self, tensor, output_splits, input_splits, *, autograd):
        output = tensor.new_empty((sum(output_splits), *tensor.shape[1:]))
        options = dict(output_split_sizes=output_splits, input_split_sizes=input_splits, group=self.group)
        if autograd:
            return dist_nn.all_to_all_single(output, tensor, **options)
        dist.all_to_all_single(output, tensor, **options)
        return output
