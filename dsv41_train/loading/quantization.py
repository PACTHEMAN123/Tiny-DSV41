"""Checkpoint weight decoding, independent of model naming and placement."""

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
    blocks = weight.view(out_blocks, 32, in_blocks, 32).float()
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
    blocks = weight.view(rows, scale.shape[1], 32).float()
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
