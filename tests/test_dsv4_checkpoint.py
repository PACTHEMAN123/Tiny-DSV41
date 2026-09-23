import json
import struct
import tempfile
import unittest
from pathlib import Path

import torch

from dsv41_train.models.dsv4.checkpoint import (
    ShardedSafeTensorReader,
    dequantize_fp4_rows,
    dequantize_fp8_blocks,
)


def write_safetensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    header = {}
    payload = bytearray()
    for name, tensor in tensors.items():
        raw = tensor.contiguous().view(torch.uint8).numpy().tobytes()
        start = len(payload)
        payload.extend(raw)
        dtype = {torch.float32: "F32", torch.int8: "I8"}[tensor.dtype]
        header[name] = {
            "dtype": dtype,
            "shape": list(tensor.shape),
            "data_offsets": [start, len(payload)],
        }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


class DSV4CheckpointTest(unittest.TestCase):
    def test_reads_an_indexed_safetensors_file_without_safetensors_package(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            expected = torch.arange(6, dtype=torch.float32).view(2, 3)
            write_safetensors(folder / "model.safetensors", {"weight": expected})
            (folder / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"weight": "model.safetensors"}}),
                encoding="utf-8",
            )

            with ShardedSafeTensorReader(folder) as checkpoint:
                actual = checkpoint.tensor("weight", "cpu")

        torch.testing.assert_close(actual, expected)

    def test_dequantizes_fp8_blocks(self):
        weight = torch.ones(32, 32, dtype=torch.float8_e4m3fn)
        scale = torch.full((1, 1), 2.0)

        actual = dequantize_fp8_blocks(weight, scale, dtype=torch.float32)

        torch.testing.assert_close(actual, torch.full((32, 32), 2.0))

    def test_dequantizes_packed_fp4_rows_low_nibble_first(self):
        packed = torch.full((1, 16), 0x21, dtype=torch.int8)
        scale = torch.ones(1, 1)

        actual = dequantize_fp4_rows(packed, scale, dtype=torch.float32)

        expected = torch.tensor([0.5, 1.0] * 16).view(1, 32)
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
