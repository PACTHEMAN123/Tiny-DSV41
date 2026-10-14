"""Cache QDQ preserves the signed zero produced by official FP8/FP4 casts."""
import unittest

import torch

from dsv41_train.models.dsv4.quantization import (
    _fake_quant_fp4_block,
    _fake_quant_fp8_block,
    quantize_activation,
)


class CacheQuantizationSignedZeroTest(unittest.TestCase):
    def test_rounded_and_explicit_negative_zero_survive_forward_and_ste(self):
        helpers = {
            "fp8": _fake_quant_fp8_block,
            "fp4_index": lambda x: _fake_quant_fp4_block(x, 32),
            "fp4_kv": lambda x: _fake_quant_fp4_block(x, 16, e4m3_scales=True),
        }
        for dtype in (torch.float32, torch.bfloat16):
            storage_dtype = torch.int32 if dtype == torch.float32 else torch.int16
            for kind, helper in helpers.items():
                with self.subTest(dtype=dtype, kind=kind):
                    values = torch.zeros((3, 32), dtype=dtype)
                    # Fix scale=1; check explicit zero, underflow and a tie rounded to zero.
                    values[0, 0] = 448 if kind == "fp8" else 6
                    values[0, 1] = -0.0
                    tiny = 2.0**-12 if kind == "fp8" else 0.125
                    tie = 2.0**-10 if kind == "fp8" else 0.25
                    values[0, 3:7] = torch.tensor([-tiny, tiny, -tie, tie], dtype=dtype)
                    values[1] = -values[1]  # Entire negative-zero groups exercise the scale floor.
                    values.requires_grad_(True)
                    expected = helper(values.detach())
                    output = quantize_activation(values, kind)
                    self.assertTrue(torch.equal(output.view(storage_dtype), expected.view(storage_dtype)))
                    self.assertTrue((output[0, 1:7] == 0).all())
                    self.assertEqual(torch.signbit(output[0, 1:7]).tolist(), [True, False, True, False, True, False])
                    self.assertTrue(torch.signbit(output[1]).all())
                    self.assertFalse(torch.signbit(output[2]).any())
                    upstream = torch.linspace(-1, 1, output.numel()).reshape_as(output).to(dtype)
                    upstream[0, 1] = -0.0
                    output.backward(upstream)
                    self.assertTrue(torch.equal(values.grad.view(storage_dtype), upstream.view(storage_dtype)))


if __name__ == "__main__":
    unittest.main()
