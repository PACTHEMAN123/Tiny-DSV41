"""Triton kernels and PyTorch references for manifold hyper-connections."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .config import DeepSeekV41Config

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _sinkhorn_forward_kernel(
        logits,
        output,
        initial,
        rows,
        eps: tl.constexpr,
        iterations: tl.constexpr,
    ):
        row = tl.program_id(0)
        row_offsets = tl.arange(0, 4)
        column_offsets = tl.arange(0, 4)
        offsets = row_offsets[:, None] * 4 + column_offsets[None, :]
        mask = row < rows
        matrix = tl.load(logits + row * 16 + offsets, mask=mask, other=0.0)
        maximum = tl.max(matrix, axis=1)
        matrix = tl.exp2((matrix - maximum[:, None]) * 1.4426950408889634)
        tl.store(initial + row * 16 + offsets, matrix, mask=mask)

        row_sum = tl.sum(matrix, axis=1)
        matrix = matrix / row_sum[:, None] + eps
        column_sum = tl.sum(matrix, axis=0)
        matrix = matrix / (column_sum[None, :] + eps)
        for _ in range(iterations - 1):
            row_sum = tl.sum(matrix, axis=1)
            matrix = matrix / (row_sum[:, None] + eps)
            column_sum = tl.sum(matrix, axis=0)
            matrix = matrix / (column_sum[None, :] + eps)
        tl.store(output + row * 16 + offsets, matrix, mask=mask)

    @triton.jit
    def _sinkhorn_backward_kernel(
        grad_output,
        initial,
        grad_logits,
        matrices,
        row_sums,
        column_sums,
        rows,
        eps: tl.constexpr,
        iterations: tl.constexpr,
    ):
        row = tl.program_id(0)
        row_offsets = tl.arange(0, 4)
        column_offsets = tl.arange(0, 4)
        offsets = row_offsets[:, None] * 4 + column_offsets[None, :]
        mask = row < rows
        matrix = tl.load(initial + row * 16 + offsets, mask=mask, other=0.0)
        matrix_base = row * 2 * iterations * 16
        sum_base = row * iterations * 4

        for iteration in range(iterations):
            before_row = matrix_base + 2 * iteration * 16
            tl.store(matrices + before_row + offsets, matrix, mask=mask)
            row_sum = tl.sum(matrix, axis=1)
            tl.store(row_sums + sum_base + iteration * 4 + row_offsets, row_sum, mask=mask)
            if iteration == 0:
                matrix = matrix / row_sum[:, None] + eps
            else:
                matrix = matrix / (row_sum[:, None] + eps)

            after_row = matrix_base + (2 * iteration + 1) * 16
            tl.store(matrices + after_row + offsets, matrix, mask=mask)
            column_sum = tl.sum(matrix, axis=0)
            tl.store(
                column_sums + sum_base + iteration * 4 + column_offsets,
                column_sum,
                mask=mask,
            )
            matrix = matrix / (column_sum[None, :] + eps)

        gradient = tl.load(
            grad_output + row * 16 + offsets, mask=mask, other=0.0
        )
        for reverse_iteration in range(iterations):
            iteration = iterations - 1 - reverse_iteration
            column_sum = tl.load(
                column_sums + sum_base + iteration * 4 + column_offsets,
                mask=mask,
                other=1.0,
            )
            gradient = gradient / (column_sum[None, :] + eps)
            gradient -= tl.sum(gradient * matrix, axis=0)[None, :]

            after_row = matrix_base + (2 * iteration + 1) * 16
            matrix = tl.load(matrices + after_row + offsets, mask=mask, other=0.0)
            row_sum = tl.load(
                row_sums + sum_base + iteration * 4 + row_offsets,
                mask=mask,
                other=1.0,
            )
            if iteration == 0:
                gradient = gradient / row_sum[:, None]
                gradient -= tl.sum(gradient * (matrix - eps), axis=1)[:, None]
            else:
                gradient = gradient / (row_sum[:, None] + eps)
                gradient -= tl.sum(gradient * matrix, axis=1)[:, None]

            before_row = matrix_base + 2 * iteration * 16
            matrix = tl.load(matrices + before_row + offsets, mask=mask, other=0.0)

        gradient *= tl.load(initial + row * 16 + offsets, mask=mask, other=0.0)
        tl.store(grad_logits + row * 16 + offsets, gradient, mask=mask)

    @triton.jit
    def _collapse_forward_kernel(
        streams,
        weights,
        output,
        rows,
        hidden: tl.constexpr,
        block: tl.constexpr,
    ):
        row = tl.program_id(0)
        columns = tl.program_id(1) * block + tl.arange(0, block)
        mask = (row < rows) & (columns < hidden)
        stream_base = streams + row * 4 * hidden + columns
        weight_base = weights + row * 4
        accumulator = tl.zeros((block,), tl.float32)
        for stream in tl.static_range(4):
            value = tl.load(stream_base + stream * hidden, mask=mask, other=0.0)
            weight = tl.load(weight_base + stream, mask=row < rows, other=0.0)
            accumulator += value.to(tl.float32) * weight
        tl.store(output + row * hidden + columns, accumulator, mask=mask)

    @triton.jit
    def _collapse_backward_kernel(
        grad_output,
        streams,
        weights,
        grad_streams,
        grad_weights,
        rows,
        hidden: tl.constexpr,
        block: tl.constexpr,
    ):
        row = tl.program_id(0)
        columns = tl.program_id(1) * block + tl.arange(0, block)
        mask = (row < rows) & (columns < hidden)
        gradient = tl.load(
            grad_output + row * hidden + columns, mask=mask, other=0.0
        ).to(tl.float32)
        stream_base = streams + row * 4 * hidden + columns
        weight_base = weights + row * 4
        for stream in tl.static_range(4):
            value = tl.load(stream_base + stream * hidden, mask=mask, other=0.0)
            weight = tl.load(weight_base + stream, mask=row < rows, other=0.0)
            tl.store(
                grad_streams + row * 4 * hidden + stream * hidden + columns,
                gradient * weight,
                mask=mask,
            )
            tl.atomic_add(
                grad_weights + row * 4 + stream,
                tl.sum(gradient * value.to(tl.float32), axis=0),
                mask=row < rows,
            )

    @triton.jit
    def _expand_forward_kernel(
        value,
        residual,
        output_weights,
        residual_weights,
        output,
        rows,
        hidden: tl.constexpr,
        block: tl.constexpr,
    ):
        row = tl.program_id(0)
        columns = tl.program_id(1) * block + tl.arange(0, block)
        mask = (row < rows) & (columns < hidden)
        branch = tl.load(value + row * hidden + columns, mask=mask, other=0.0)
        residual_base = residual + row * 4 * hidden + columns
        residual0 = tl.load(residual_base, mask=mask, other=0.0).to(tl.float32)
        residual1 = tl.load(residual_base + hidden, mask=mask, other=0.0).to(tl.float32)
        residual2 = tl.load(residual_base + 2 * hidden, mask=mask, other=0.0).to(tl.float32)
        residual3 = tl.load(residual_base + 3 * hidden, mask=mask, other=0.0).to(tl.float32)
        matrix = residual_weights + row * 16
        for stream in tl.static_range(4):
            result = branch.to(tl.float32) * tl.load(output_weights + row * 4 + stream)
            result += residual0 * tl.load(matrix + stream)
            result += residual1 * tl.load(matrix + 4 + stream)
            result += residual2 * tl.load(matrix + 8 + stream)
            result += residual3 * tl.load(matrix + 12 + stream)
            tl.store(
                output + row * 4 * hidden + stream * hidden + columns,
                result,
                mask=mask,
            )

    @triton.jit
    def _expand_backward_kernel(
        grad_output,
        value,
        residual,
        output_weights,
        residual_weights,
        grad_value,
        grad_residual,
        grad_output_weights,
        grad_residual_weights,
        rows,
        hidden: tl.constexpr,
        block: tl.constexpr,
    ):
        row = tl.program_id(0)
        columns = tl.program_id(1) * block + tl.arange(0, block)
        mask = (row < rows) & (columns < hidden)
        grad_base = grad_output + row * 4 * hidden + columns
        grad0 = tl.load(grad_base, mask=mask, other=0.0).to(tl.float32)
        grad1 = tl.load(grad_base + hidden, mask=mask, other=0.0).to(tl.float32)
        grad2 = tl.load(grad_base + 2 * hidden, mask=mask, other=0.0).to(tl.float32)
        grad3 = tl.load(grad_base + 3 * hidden, mask=mask, other=0.0).to(tl.float32)
        branch = tl.load(value + row * hidden + columns, mask=mask, other=0.0)
        post = output_weights + row * 4
        grad_branch = grad0 * tl.load(post)
        grad_branch += grad1 * tl.load(post + 1)
        grad_branch += grad2 * tl.load(post + 2)
        grad_branch += grad3 * tl.load(post + 3)
        tl.store(grad_value + row * hidden + columns, grad_branch, mask=mask)

        residual_base = residual + row * 4 * hidden + columns
        matrix = residual_weights + row * 16
        gradients = (grad0, grad1, grad2, grad3)
        for source in tl.static_range(4):
            source_matrix = matrix + source * 4
            grad_residual_value = grad0 * tl.load(source_matrix)
            grad_residual_value += grad1 * tl.load(source_matrix + 1)
            grad_residual_value += grad2 * tl.load(source_matrix + 2)
            grad_residual_value += grad3 * tl.load(source_matrix + 3)
            tl.store(
                grad_residual + row * 4 * hidden + source * hidden + columns,
                grad_residual_value,
                mask=mask,
            )
            residual_value = tl.load(
                residual_base + source * hidden, mask=mask, other=0.0
            ).to(tl.float32)
            for target in tl.static_range(4):
                tl.atomic_add(
                    grad_residual_weights + row * 16 + source * 4 + target,
                    tl.sum(residual_value * gradients[target], axis=0),
                    mask=row < rows,
                )

        branch = branch.to(tl.float32)
        tl.atomic_add(
            grad_output_weights + row * 4,
            tl.sum(branch * grad0, axis=0),
            mask=row < rows,
        )
        tl.atomic_add(
            grad_output_weights + row * 4 + 1,
            tl.sum(branch * grad1, axis=0),
            mask=row < rows,
        )
        tl.atomic_add(
            grad_output_weights + row * 4 + 2,
            tl.sum(branch * grad2, axis=0),
            mask=row < rows,
        )
        tl.atomic_add(
            grad_output_weights + row * 4 + 3,
            tl.sum(branch * grad3, axis=0),
            mask=row < rows,
        )


def _sinkhorn_forward(
    logits: torch.Tensor, iterations: int, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    leading = logits.shape[:-2]
    rows = logits.numel() // 16
    flat = logits.contiguous().view(rows, 4, 4)
    output = torch.empty_like(flat, dtype=torch.float32)
    initial = torch.empty_like(output)
    _sinkhorn_forward_kernel[(rows,)](
        flat,
        output,
        initial,
        rows,
        eps=eps,
        iterations=iterations,
        num_warps=1,
    )
    return output.view(*leading, 4, 4), initial.view(*leading, 4, 4)


def _sinkhorn_backward(
    grad_output: torch.Tensor,
    initial: torch.Tensor,
    iterations: int,
    eps: float,
) -> torch.Tensor:
    leading = grad_output.shape[:-2]
    rows = grad_output.numel() // 16
    grad_output = grad_output.contiguous().view(rows, 4, 4)
    initial = initial.contiguous().view(rows, 4, 4)
    grad_logits = torch.empty_like(initial)
    matrices = torch.empty(
        rows * 2 * iterations * 16, device=grad_output.device, dtype=torch.float32
    )
    row_sums = torch.empty(
        rows * iterations * 4, device=grad_output.device, dtype=torch.float32
    )
    column_sums = torch.empty_like(row_sums)
    _sinkhorn_backward_kernel[(rows,)](
        grad_output,
        initial,
        grad_logits,
        matrices,
        row_sums,
        column_sums,
        rows,
        eps=eps,
        iterations=iterations,
        num_warps=1,
    )
    return grad_logits.view(*leading, 4, 4)


class _Sinkhorn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, iterations, eps):
        output, initial = _sinkhorn_forward(logits, iterations, eps)
        ctx.save_for_backward(initial)
        ctx.iterations = iterations
        ctx.eps = eps
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (initial,) = ctx.saved_tensors
        return _sinkhorn_backward(grad_output, initial, ctx.iterations, ctx.eps), None, None


class _Collapse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, streams, weights):
        leading, hidden = streams.shape[:-2], streams.shape[-1]
        rows = streams.numel() // (4 * hidden)
        streams = streams.contiguous().view(rows, 4, hidden)
        weights = weights.contiguous().view(rows, 4)
        output = torch.empty((rows, hidden), device=streams.device, dtype=streams.dtype)
        block = 256
        _collapse_forward_kernel[(rows, triton.cdiv(hidden, block))](
            streams, weights, output, rows, hidden=hidden, block=block, num_warps=8
        )
        ctx.save_for_backward(streams, weights)
        ctx.leading = leading
        return output.view(*leading, hidden)

    @staticmethod
    def backward(ctx, grad_output):
        streams, weights = ctx.saved_tensors
        rows, _, hidden = streams.shape
        grad_output = grad_output.contiguous().view(rows, hidden)
        grad_streams = torch.empty_like(streams)
        grad_weights = torch.zeros_like(weights)
        block = 256
        _collapse_backward_kernel[(rows, triton.cdiv(hidden, block))](
            grad_output,
            streams,
            weights,
            grad_streams,
            grad_weights,
            rows,
            hidden=hidden,
            block=block,
            num_warps=8,
        )
        return grad_streams.view(*ctx.leading, 4, hidden), grad_weights.view(*ctx.leading, 4)


class _Expand(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, residual, output_weights, residual_weights):
        leading, hidden = residual.shape[:-2], residual.shape[-1]
        rows = residual.numel() // (4 * hidden)
        value = value.contiguous().view(rows, hidden)
        residual = residual.contiguous().view(rows, 4, hidden)
        output_weights = output_weights.contiguous().view(rows, 4)
        residual_weights = residual_weights.contiguous().view(rows, 4, 4)
        output = torch.empty_like(residual)
        block = 256
        _expand_forward_kernel[(rows, triton.cdiv(hidden, block))](
            value,
            residual,
            output_weights,
            residual_weights,
            output,
            rows,
            hidden=hidden,
            block=block,
            num_warps=8,
        )
        ctx.save_for_backward(value, residual, output_weights, residual_weights)
        ctx.leading = leading
        return output.view(*leading, 4, hidden)

    @staticmethod
    def backward(ctx, grad_output):
        value, residual, output_weights, residual_weights = ctx.saved_tensors
        rows, _, hidden = residual.shape
        grad_output = grad_output.contiguous().view(rows, 4, hidden)
        grad_value = torch.empty_like(value)
        grad_residual = torch.empty_like(residual)
        grad_output_weights = torch.zeros_like(output_weights)
        grad_residual_weights = torch.zeros_like(residual_weights)
        block = 256
        _expand_backward_kernel[(rows, triton.cdiv(hidden, block))](
            grad_output,
            value,
            residual,
            output_weights,
            residual_weights,
            grad_value,
            grad_residual,
            grad_output_weights,
            grad_residual_weights,
            rows,
            hidden=hidden,
            block=block,
            num_warps=8,
        )
        return (
            grad_value.view(*ctx.leading, hidden),
            grad_residual.view(*ctx.leading, 4, hidden),
            grad_output_weights.view(*ctx.leading, 4),
            grad_residual_weights.view(*ctx.leading, 4, 4),
        )


def _mixing_weights(
    projected: torch.Tensor,
    base: torch.Tensor,
    scale: torch.Tensor,
    hc: int,
    iterations: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pre, post, comb = projected.float().split([hc, hc, hc * hc], dim=-1)
    pre_bias, post_bias, comb_bias = base.float().split([hc, hc, hc * hc])
    pre_scale, post_scale, comb_scale = scale.float()
    pre = torch.sigmoid(pre * pre_scale + pre_bias) + eps
    post = 2 * torch.sigmoid(post * post_scale + post_bias)
    comb = comb.view(*comb.shape[:-1], hc, hc) * comb_scale
    comb = comb + comb_bias.view(hc, hc)
    if comb.is_cuda and triton is not None and hc == 4 and comb.numel() > 0:
        comb = _Sinkhorn.apply(comb, iterations, eps)
    else:
        comb = torch.softmax(comb.float(), dim=-1) + eps
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
        for _ in range(iterations - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


def _collapse(streams: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if (
        streams.is_cuda
        and triton is not None
        and streams.shape[-2] == 4
        and streams.numel() > 0
    ):
        return _Collapse.apply(streams, weights)
    return (weights.unsqueeze(-1) * streams.float()).sum(-2).to(streams.dtype)


def _expand(
    value: torch.Tensor,
    residual: torch.Tensor,
    output_weights: torch.Tensor,
    residual_weights: torch.Tensor,
) -> torch.Tensor:
    if (
        residual.is_cuda
        and triton is not None
        and residual.shape[-2] == 4
        and residual.numel() > 0
    ):
        return _Expand.apply(value, residual, output_weights, residual_weights)
    mixed = torch.einsum(
        "...jk,...jd->...kd", residual_weights.float(), residual.float()
    )
    return (
        output_weights.unsqueeze(-1) * value.float().unsqueeze(-2) + mixed
    ).to(residual.dtype)


class HyperConnection(nn.Module):
    """Prepare and finish one four-stream mHC branch."""

    def __init__(self, config: DeepSeekV41Config) -> None:
        super().__init__()
        self.hc_mult = config.hc_mult
        self.sinkhorn_iters = config.hc_sinkhorn_iters
        self.eps = config.hc_eps
        mix_size = (2 + config.hc_mult) * config.hc_mult
        self.fn = nn.Parameter(torch.empty(mix_size, config.hc_mult * config.hidden_size))
        self.base = nn.Parameter(torch.zeros(mix_size))
        self.scale = nn.Parameter(torch.ones(3))
        self.norm_eps = config.rms_norm_eps

    def forward(
        self, streams: torch.Tensor, input_weights: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = streams.flatten(2).float()
        flat = flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.norm_eps)
        projected = F.linear(flat, self.fn.float())
        pre, post, residual_weights = _mixing_weights(
            projected,
            self.base,
            self.scale,
            self.hc_mult,
            self.sinkhorn_iters,
            self.eps,
        )
        return _collapse(streams, input_weights), pre, post, residual_weights

    @staticmethod
    def finish(
        value: torch.Tensor,
        residual: torch.Tensor,
        output_weights: torch.Tensor,
        residual_weights: torch.Tensor,
    ) -> torch.Tensor:
        return _expand(value, residual, output_weights, residual_weights)

    @staticmethod
    def collapse(streams: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return _collapse(streams, weights)


def is_available(tensor: torch.Tensor) -> bool:
    return tensor.is_cuda and triton is not None


__all__ = [
    "HyperConnection",
    "is_available",
]
