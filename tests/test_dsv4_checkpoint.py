import json
import struct
import tempfile
import unittest
from pathlib import Path

import torch

from dsv41_train.models.dsv4 import DeepSeekV41Config
from dsv41_train.models.dsv4.checkpoint import (
    ShardedSafeTensorReader,
    _materialize_rotary,
    _prefix_config,
    dequantize_fp4_rows,
    dequantize_fp8_blocks,
    dequantize_fp8_rows,
)
from dsv41_train.models.dsv4.model import RotaryEmbedding


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
    def test_materializes_checkpoint_rotary_with_yarn_scaling(self):
        values = DeepSeekV41Config.tiny().to_dict()
        values["rope_scaling"] = {
            "rope_type": "yarn",
            "factor": 4,
            "beta_fast": 32,
            "beta_slow": 1,
            "original_max_position_embeddings": 16,
        }
        config = DeepSeekV41Config(**values)
        with torch.device("meta"):
            actual = RotaryEmbedding(config)

        _materialize_rotary(actual, config, torch.device("cpu"))
        expected = RotaryEmbedding(config)

        self.assertFalse(actual.main.is_meta)
        self.assertFalse(actual.compressed.is_meta)
        torch.testing.assert_close(actual.main, expected.main, rtol=0, atol=0)
        torch.testing.assert_close(actual.compressed, expected.compressed, rtol=0, atol=0)

    def test_prefix_config_includes_the_second_engram_layer(self):
        config = DeepSeekV41Config(
            engram_layer_ids=[1, 14],
            engram_num_embeddings=[100, 200],
        )
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / "config.json").write_text(
                json.dumps(config.to_dict()), encoding="utf-8"
            )

            prefix = _prefix_config(folder, config.num_hidden_layers)

            self.assertEqual(prefix.num_hidden_layers, config.num_hidden_layers)
            self.assertEqual(prefix.engram_layer_ids, [1, 14])
            self.assertEqual(prefix.engram_num_embeddings, [100, 200])
            self.assertEqual(
                prefix.candidate_source_layer_id, config.candidate_source_layer_id
            )
            with self.assertRaisesRegex(NotImplementedError, "40"):
                _prefix_config(folder, config.num_hidden_layers + 1)

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
                sliced = checkpoint.tensor_rows("weight", 1, 2, "cpu")

        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(sliced, expected[1:2])

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

    def test_dequantizes_row_scaled_fp8(self):
        weight = torch.ones(2, 64, dtype=torch.float8_e4m3fn)
        scale = torch.tensor([[2.0, 3.0], [4.0, 5.0]])

        actual = dequantize_fp8_rows(weight, scale, dtype=torch.float32)

        expected = torch.tensor([[2.0] * 32 + [3.0] * 32, [4.0] * 32 + [5.0] * 32])
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
