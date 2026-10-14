"""Read bounded CPU samples from a real DSV4 quantized checkpoint; never write weights."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from dsv41_train.loading import ShardedSafeTensorReader
from dsv41_train.loading.quantization import (
    dequantize_fp4_rows, dequantize_fp8_blocks, dequantize_fp8_rows,
)


SAMPLES = (
    ("attention_fp8_block", "layers.0.attn.wq_a", "F8_E4M3", dequantize_fp8_blocks),
    ("expert_fp4_rows", "layers.0.ffn.experts.0.w1", "I8", dequantize_fp4_rows),
    ("engram_fp8_rows", "layers.1.engram.embed", "F8_E4M3", dequantize_fp8_rows),
)
MAX_SAMPLE_ROWS = 32
MAX_SOURCE_BYTES_PER_SAMPLE = 8 * 1024**2


def _metadata(reader, name):
    metadata = reader.metadata(name)
    shape = metadata["shape"]
    if len(shape) != 2 or any(type(size) is not int or size < 1 for size in shape):
        raise ValueError(f"{name} must be a nonempty matrix")
    size = metadata["data_offsets"][1] - metadata["data_offsets"][0]
    if size <= 0 or size % math.prod(shape):
        raise ValueError(f"{name} has an invalid encoded byte range")
    return {"dtype": metadata["dtype"], "shape": list(shape),
            "encoded_bytes": size, "row_bytes": size // shape[0]}


def inspect(folder):
    """Return format/decode diagnostics, not model or numerical-reference parity."""
    report = {
        "model_path": str(Path(folder)), "torch": torch.__version__, "device": "cpu",
        "scope": "bounded tensor-format and decode checks; not model/reference parity",
        "max_rows": MAX_SAMPLE_ROWS,
        "max_source_bytes_per_sample": MAX_SOURCE_BYTES_PER_SAMPLE,
        "samples": [],
    }
    torch.set_num_threads(1)
    with ShardedSafeTensorReader(folder) as reader:
        for kind, prefix, expected_dtype, decode in SAMPLES:
            sample = {"kind": kind, "source": prefix, "passed": False}
            report["samples"].append(sample)
            started = reader.bytes_read
            try:
                weight_info = _metadata(reader, prefix + ".weight")
                scale_info = _metadata(reader, prefix + ".scale")
                sample.update(weight=weight_info, scale=scale_info)
                if weight_info["dtype"] != expected_dtype:
                    raise ValueError(f"expected weight dtype {expected_dtype}, got {weight_info['dtype']}")
                rows = min(MAX_SAMPLE_ROWS, weight_info["shape"][0])
                scale_rows = 1 if decode is dequantize_fp8_blocks else rows
                if decode is dequantize_fp8_blocks and rows != 32:
                    raise ValueError("block FP8 requires at least one complete 32-row block")
                planned = rows * weight_info["row_bytes"] + scale_rows * scale_info["row_bytes"]
                if planned > MAX_SOURCE_BYTES_PER_SAMPLE:
                    raise ValueError(f"sample exceeds bounded read budget: {planned} bytes")
                weight = reader.tensor_rows(prefix + ".weight", 0, rows, "cpu")
                scale = reader.tensor_rows(prefix + ".scale", 0, scale_rows, "cpu")
                decoded = decode(weight, scale, dtype=torch.bfloat16)
                finite = bool(torch.isfinite(decoded).all())
                sample.update(
                    weight_sample_shape=list(weight.shape), scale_sample_shape=list(scale.shape),
                    torch_weight_dtype=str(weight.dtype), torch_scale_dtype=str(scale.dtype),
                    decoded_shape=list(decoded.shape), decoded_dtype=str(decoded.dtype),
                    all_finite=finite, nonzero_values=int(torch.count_nonzero(decoded)),
                    max_abs=float(decoded.float().abs().max()) if finite else None,
                    passed=finite,
                )
                if not finite:
                    sample["error"] = "decoded sample contains nonfinite values"
                del weight, scale, decoded
            except (KeyError, ValueError, TypeError, IndexError, RuntimeError) as error:
                sample["error"] = f"{type(error).__name__}: {error}"
            sample["logical_bytes_read"] = reader.bytes_read - started
        report["logical_bytes_read"] = reader.bytes_read
    report["passed"] = all(sample["passed"] for sample in report["samples"])
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="/mnt/fuse/deepseek-ai/DeepSeek-V4.1-Flash")
    args = parser.parse_args(argv)
    report = inspect(args.model_path)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
