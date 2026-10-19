# Portions Copyright 2026 The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0.
"""Reference cache quantization with an explicit straight-through training gradient."""

import torch


_FP4_MAX = 6.0
_FP4_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)
_FP8_MAX = 448.0


def _pow2_ceil_scale(values: torch.Tensor) -> torch.Tensor:
    bits = values.contiguous().view(torch.int32)
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    power = exponent - 127 + (mantissa != 0).to(torch.int32)
    return torch.exp2(power.to(torch.float32))


def _e2m1_codes(values: torch.Tensor) -> torch.Tensor:
    magnitude = values.abs()
    negative = torch.signbit(values)
    boundaries = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
        dtype=torch.float32,
        device=values.device,
    )
    ties_up = torch.tensor(
        [False, True, False, True, False, True, False], device=values.device
    )
    thresholds = torch.where(
        ties_up,
        boundaries,
        torch.nextafter(boundaries, torch.full_like(boundaries, float("inf"))),
    )
    codes = (magnitude.unsqueeze(-1) >= thresholds).sum(-1).to(torch.uint8)
    return codes | (negative.to(torch.uint8) << 3)


def _fake_quant_fp4_block(
    tensor: torch.Tensor, block_size: int, *, e4m3_scales: bool = False
) -> torch.Tensor:
    width = tensor.shape[-1]
    if width % block_size:
        return tensor
    blocks = tensor.float().view(*tensor.shape[:-1], width // block_size, block_size)
    maximum = blocks.abs().amax(-1)
    if e4m3_scales:
        scale = (maximum.clamp_min(_FP4_MAX * 2.0**-9) / _FP4_MAX).to(
            torch.float8_e4m3fn
        ).float()
    else:
        scale = _pow2_ceil_scale(
            maximum.clamp_min(_FP4_MAX * 2.0**-126) / _FP4_MAX
        )
    quantized = (blocks / scale.unsqueeze(-1)).clamp(-_FP4_MAX, _FP4_MAX)
    lookup = torch.tensor(_FP4_VALUES, device=tensor.device, dtype=torch.float32)
    values = lookup[_e2m1_codes(quantized).long()]
    return (values * scale.unsqueeze(-1)).view_as(tensor).to(tensor.dtype)


def _fake_quant_fp8_block(tensor: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    width = tensor.shape[-1]
    if width % block_size:
        return tensor
    blocks = tensor.float().view(*tensor.shape[:-1], width // block_size, block_size)
    maximum = blocks.abs().amax(-1).clamp_min(1e-4)
    scale = _pow2_ceil_scale(maximum / _FP8_MAX)
    quantized = (blocks / scale.unsqueeze(-1)).clamp(-_FP8_MAX, _FP8_MAX)
    values = quantized.to(torch.float8_e4m3fn).float() * scale.unsqueeze(-1)
    return values.view_as(tensor).to(tensor.dtype)


class _StraightThrough(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, quantized):
        return quantized

    @staticmethod
    def backward(ctx, gradient):
        return gradient, None


def quantize_activation(tensor: torch.Tensor, kind: str) -> torch.Tensor:
    with torch.no_grad():
        if kind == "fp8":
            quantized = _fake_quant_fp8_block(tensor)
        elif kind == "fp4_index":
            quantized = _fake_quant_fp4_block(tensor, 32)
        elif kind == "fp4_kv":
            quantized = _fake_quant_fp4_block(tensor, 16, e4m3_scales=True)
        else:
            raise ValueError(f"unknown quantization kind: {kind}")
    return _StraightThrough.apply(tensor, quantized)
