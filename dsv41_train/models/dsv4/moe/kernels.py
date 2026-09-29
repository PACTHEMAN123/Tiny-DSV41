"""Triton kernels and PyTorch references for DSV4 expert activations."""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


def clamped_swiglu_reference(
    gate: torch.Tensor, up: torch.Tensor, limit: float
) -> torch.Tensor:
    gate, up = gate.float(), up.float()
    if limit > 0:
        gate = gate.clamp(max=limit)
        up = up.clamp(-limit, limit)
    return F.silu(gate) * up


if triton is not None:

    @triton.jit
    def _clamped_swiglu_forward_kernel(
        gate,
        up,
        weights,
        output,
        rows: tl.constexpr,
        columns: tl.constexpr,
        gate_row_stride: tl.constexpr,
        gate_column_stride: tl.constexpr,
        up_row_stride: tl.constexpr,
        up_column_stride: tl.constexpr,
        output_row_stride: tl.constexpr,
        output_column_stride: tl.constexpr,
        limit: tl.constexpr,
        block: tl.constexpr,
    ):
        row = tl.program_id(0)
        columns_offset = tl.program_id(1) * block + tl.arange(0, block)
        mask = (row < rows) & (columns_offset < columns)
        gate_value = tl.load(
            gate
            + row * gate_row_stride
            + columns_offset * gate_column_stride,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        up_value = tl.load(
            up + row * up_row_stride + columns_offset * up_column_stride,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        if limit > 0:
            gate_value = tl.minimum(gate_value, limit)
            up_value = tl.maximum(tl.minimum(up_value, limit), -limit)
        silu = gate_value * tl.sigmoid(gate_value)
        weight = tl.load(weights + row).to(tl.float32)
        tl.store(
            output
            + row * output_row_stride
            + columns_offset * output_column_stride,
            silu * up_value * weight,
            mask=mask,
        )

    @triton.jit
    def _clamped_swiglu_backward_kernel(
        grad_output,
        gate,
        up,
        weights,
        grad_gate,
        grad_up,
        grad_weights,
        rows: tl.constexpr,
        columns: tl.constexpr,
        grad_output_row_stride: tl.constexpr,
        grad_output_column_stride: tl.constexpr,
        gate_row_stride: tl.constexpr,
        gate_column_stride: tl.constexpr,
        up_row_stride: tl.constexpr,
        up_column_stride: tl.constexpr,
        grad_gate_row_stride: tl.constexpr,
        grad_gate_column_stride: tl.constexpr,
        grad_up_row_stride: tl.constexpr,
        grad_up_column_stride: tl.constexpr,
        limit: tl.constexpr,
        block: tl.constexpr,
    ):
        row = tl.program_id(0)
        columns_offset = tl.program_id(1) * block + tl.arange(0, block)
        mask = (row < rows) & (columns_offset < columns)
        grad = tl.load(
            grad_output
            + row * grad_output_row_stride
            + columns_offset * grad_output_column_stride,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        raw_gate = tl.load(
            gate
            + row * gate_row_stride
            + columns_offset * gate_column_stride,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        raw_up = tl.load(
            up + row * up_row_stride + columns_offset * up_column_stride,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        gate_value = raw_gate
        up_value = raw_up
        gate_mask = 1.0
        up_mask = 1.0
        if limit > 0:
            gate_value = tl.minimum(raw_gate, limit)
            up_value = tl.maximum(tl.minimum(raw_up, limit), -limit)
            gate_mask = raw_gate <= limit
            up_mask = (raw_up >= -limit) & (raw_up <= limit)
        sigmoid = tl.sigmoid(gate_value)
        silu = gate_value * sigmoid
        silu_grad = sigmoid * (1.0 + gate_value * (1.0 - sigmoid))
        weight = tl.load(weights + row).to(tl.float32)
        tl.store(
            grad_gate
            + row * grad_gate_row_stride
            + columns_offset * grad_gate_column_stride,
            grad * weight * up_value * silu_grad * gate_mask,
            mask=mask,
        )
        tl.store(
            grad_up
            + row * grad_up_row_stride
            + columns_offset * grad_up_column_stride,
            grad * weight * silu * up_mask,
            mask=mask,
        )
        tl.atomic_add(grad_weights + row, tl.sum(grad * silu * up_value, axis=0))

    def _clamped_swiglu_forward(
        gate: torch.Tensor,
        up: torch.Tensor,
        weights: torch.Tensor,
        limit: float,
    ) -> torch.Tensor:
        output = torch.empty_like(gate, memory_format=torch.contiguous_format)
        block = min(1024, triton.next_power_of_2(gate.shape[1]))
        grid = (gate.shape[0], triton.cdiv(gate.shape[1], block))
        _clamped_swiglu_forward_kernel[grid](
            gate,
            up,
            weights,
            output,
            rows=gate.shape[0],
            columns=gate.shape[1],
            gate_row_stride=gate.stride(0),
            gate_column_stride=gate.stride(1),
            up_row_stride=up.stride(0),
            up_column_stride=up.stride(1),
            output_row_stride=output.stride(0),
            output_column_stride=output.stride(1),
            limit=limit,
            block=block,
            num_warps=8,
        )
        return output

    def _clamped_swiglu_backward(
        grad_output: torch.Tensor,
        gate: torch.Tensor,
        up: torch.Tensor,
        weights: torch.Tensor,
        limit: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        grad_gate = torch.empty_like(gate, memory_format=torch.contiguous_format)
        grad_up = torch.empty_like(up, memory_format=torch.contiguous_format)
        grad_weights = torch.zeros_like(weights, dtype=torch.float32)
        block = min(1024, triton.next_power_of_2(gate.shape[1]))
        grid = (gate.shape[0], triton.cdiv(gate.shape[1], block))
        _clamped_swiglu_backward_kernel[grid](
            grad_output,
            gate,
            up,
            weights,
            grad_gate,
            grad_up,
            grad_weights,
            rows=gate.shape[0],
            columns=gate.shape[1],
            grad_output_row_stride=grad_output.stride(0),
            grad_output_column_stride=grad_output.stride(1),
            gate_row_stride=gate.stride(0),
            gate_column_stride=gate.stride(1),
            up_row_stride=up.stride(0),
            up_column_stride=up.stride(1),
            grad_gate_row_stride=grad_gate.stride(0),
            grad_gate_column_stride=grad_gate.stride(1),
            grad_up_row_stride=grad_up.stride(0),
            grad_up_column_stride=grad_up.stride(1),
            limit=limit,
            block=block,
            num_warps=8,
        )
        return grad_gate, grad_up, grad_weights

    @torch.library.custom_op(
        "dsv41::clamped_swiglu", mutates_args=(), device_types="cuda"
    )
    def _clamped_swiglu_op(
        gate: torch.Tensor,
        up: torch.Tensor,
        weights: torch.Tensor,
        limit: float,
    ) -> torch.Tensor:
        return _clamped_swiglu_forward(gate, up, weights, limit)

    @_clamped_swiglu_op.register_fake
    def _clamped_swiglu_fake(
        gate: torch.Tensor,
        up: torch.Tensor,
        weights: torch.Tensor,
        limit: float,
    ) -> torch.Tensor:
        return torch.empty_like(gate, memory_format=torch.contiguous_format)

    @torch.library.custom_op(
        "dsv41::clamped_swiglu_backward", mutates_args=(), device_types="cuda"
    )
    def _clamped_swiglu_backward_op(
        grad_output: torch.Tensor,
        gate: torch.Tensor,
        up: torch.Tensor,
        weights: torch.Tensor,
        limit: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _clamped_swiglu_backward(grad_output, gate, up, weights, limit)

    @_clamped_swiglu_backward_op.register_fake
    def _clamped_swiglu_backward_fake(
        grad_output: torch.Tensor,
        gate: torch.Tensor,
        up: torch.Tensor,
        weights: torch.Tensor,
        limit: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch.empty_like(gate), torch.empty_like(up), torch.empty_like(weights)

    def _setup_context(ctx, inputs, output) -> None:
        gate, up, weights, limit = inputs
        ctx.save_for_backward(gate, up, weights)
        ctx.limit = limit

    def _autograd_backward(ctx, grad_output):
        gate, up, weights = ctx.saved_tensors
        grad_gate, grad_up, grad_weights = _clamped_swiglu_backward_op(
            grad_output, gate, up, weights, ctx.limit
        )
        return grad_gate, grad_up, grad_weights, None

    _clamped_swiglu_op.register_autograd(
        _autograd_backward, setup_context=_setup_context
    )


def clamped_swiglu(
    gate: torch.Tensor,
    up: torch.Tensor,
    weights: torch.Tensor,
    limit: float,
) -> torch.Tensor:
    if gate.is_cuda and triton is not None:
        return _clamped_swiglu_op(gate, up, weights, limit)
    return (clamped_swiglu_reference(gate, up, limit) * weights[:, None]).to(gate.dtype)


__all__ = ["clamped_swiglu", "clamped_swiglu_reference"]
