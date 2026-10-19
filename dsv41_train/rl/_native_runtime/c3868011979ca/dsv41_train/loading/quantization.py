"""Checkpoint weight decoding, independent of model naming and placement."""

from functools import lru_cache

import torch


_FP4_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


@lru_cache(maxsize=1)
def _cpu_fp8_values():
    # Derive every encoding from PyTorch, including signed zero and NaN.
    return torch.arange(256, device="cpu").to(torch.uint8).view(torch.float8_e4m3fn).float()


def _fp8_float(weight):
    if weight.device.type == "cpu" and weight.dtype == torch.float8_e4m3fn:
        indices = weight.view(torch.uint8).reshape(-1).long()
        return torch.index_select(_cpu_fp8_values(), 0, indices).view(weight.shape)
    return weight.float()


@torch.no_grad()
def dequantize_blockwise_fp8(weight, scale, block_size=(128, 128), *, dtype=torch.bfloat16):
    """Decode HF block-scaled FP8, including partial edge blocks."""
    if weight.ndim != 2 or scale.ndim != 2 or len(block_size) != 2 or min(block_size) < 1:
        raise ValueError("blockwise FP8 requires matrices and two positive block dimensions")
    rows, columns = weight.shape
    height, width = block_size
    if tuple(scale.shape) != ((rows + height - 1) // height, (columns + width - 1) // width):
        raise ValueError("FP8 block scale shape does not match the weight")
    expanded = scale.float().repeat_interleave(height, 0).repeat_interleave(width, 1)
    return (_fp8_float(weight) * expanded[:rows, :columns]).to(dtype)


@torch.no_grad()
def dequantize_fp8_blocks(
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize a 32x32 E4M3 weight using its E8M0 block scales."""

    if weight.ndim != 2 or scale.ndim != 2:
        raise ValueError("FP8 weight and scale must both be matrices")
    out_features, in_features = weight.shape
    out_blocks, in_blocks = scale.shape
    if out_features != out_blocks * 32 or in_features != in_blocks * 32:
        raise ValueError(
            f"FP8 shape mismatch: weight={tuple(weight.shape)}, scale={tuple(scale.shape)}"
        )
    blocks = _fp8_float(weight).view(out_blocks, 32, in_blocks, 32)
    return (blocks * scale.float()[:, None, :, None]).reshape(weight.shape).to(dtype)


@torch.no_grad()
def dequantize_fp8_rows(
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize E4M3 rows with one E8M0 scale per 32 columns."""

    if weight.ndim != 2 or scale.ndim != 2:
        raise ValueError("FP8 weight and scale must both be matrices")
    rows, features = weight.shape
    if scale.shape[0] != rows or features != scale.shape[1] * 32:
        raise ValueError(
            f"row-scaled FP8 shape mismatch: weight={tuple(weight.shape)}, "
            f"scale={tuple(scale.shape)}"
        )
    blocks = _fp8_float(weight).view(rows, scale.shape[1], 32)
    return (blocks * scale.float().unsqueeze(-1)).flatten(1).to(dtype)


@torch.no_grad()
def dequantize_fp4_rows(
    packed: torch.Tensor,
    scale: torch.Tensor,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Unpack E2M1 pairs and apply one E8M0 scale per 32 input values."""

    if packed.dtype != torch.int8 or packed.ndim != 2 or scale.ndim != 2:
        raise ValueError("FP4 weight must be an int8 matrix with a matrix of scales")
    out_features, packed_features = packed.shape
    if scale.shape[0] != out_features or packed_features * 2 != scale.shape[1] * 32:
        raise ValueError(
            f"FP4 shape mismatch: weight={tuple(packed.shape)}, scale={tuple(scale.shape)}"
        )
    values = torch.tensor(_FP4_VALUES, device=packed.device, dtype=torch.float32)
    raw = packed.view(torch.uint8)
    low = values[(raw & 0x0F).long()]
    high = values[((raw >> 4) & 0x0F).long()]
    unpacked = torch.stack((low, high), dim=-1).flatten(1)
    blocks = unpacked.view(out_features, scale.shape[1], 32)
    return (blocks * scale.float().unsqueeze(-1)).flatten(1).to(dtype)
