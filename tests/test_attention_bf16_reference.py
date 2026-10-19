"""Small analytic CPU checks for the independent quantized attention KV oracle."""
import math
import unittest

import torch

from attention_bf16_reference import quantized_kv_reference


def inputs(ids=(0,), mask=None, compressed_ids=()):
    q = torch.zeros(1, 1, 1, 2, dtype=torch.bfloat16)
    kv = torch.ones(1, 1, 2, dtype=torch.bfloat16)
    compressed = kv.clone() if compressed_ids else torch.empty(1, 0, 2, dtype=torch.bfloat16)
    ids = torch.tensor([[ids]], dtype=torch.int32)
    mask = torch.ones_like(ids, dtype=torch.bool) if mask is None else torch.tensor([[mask]])
    picked = torch.tensor([[compressed_ids]], dtype=torch.int32)
    count = int(((ids == 0) & mask).sum() + (picked == 0).sum())
    output = torch.full_like(q, count / (count + 1))
    lse = torch.full((1, 1, 1), math.log(count + 1), dtype=torch.float32)
    return (q, kv, ids, mask, compressed, picked, torch.zeros(1), output, lse), torch.ones_like(q)


class BF16AttentionReferenceTest(unittest.TestCase):
    def test_saved_output_and_quantized_score_gradient(self):
        saved, upstream = inputs()
        saved = list(saved)
        saved[0] = torch.tensor([[[[1., 2.]]]], dtype=torch.bfloat16)
        saved[1] = torch.tensor([[[.5, -.25]]], dtype=torch.bfloat16)
        saved[7] = torch.tensor([[[[.2, .3]]]], dtype=torch.bfloat16)
        upstream = torch.tensor([[[[3., -1.]]]], dtype=torch.bfloat16)
        # Q dot K=0, so P=1/2. dP=7/4 and delta uses rounded saved O.
        delta = 3 * float(saved[7][0, 0, 0, 0]) - float(saved[7][0, 0, 0, 1])
        ds = float(torch.tensor(.5 * (1.75 - delta), dtype=torch.bfloat16))
        expected = torch.tensor([[[1.5 + ds / math.sqrt(2), -.5 + 2 * ds / math.sqrt(2)]]], dtype=torch.float64)
        result = quantized_kv_reference(saved, upstream)
        torch.testing.assert_close(result['fp64_sums']['grad_kv'], expected, rtol=0, atol=1e-15)
        self.assertTrue(torch.equal(result['cast_to_kv_dtype']['grad_kv'], expected.bfloat16()))

    def test_duplicate_masked_slots_and_branches_are_preserved(self):
        saved, upstream = inputs((0, 0, -1, 1, 0), (True, True, True, True, False), (0,))
        before = [t.clone() for t in (*saved, upstream)]
        rng = torch.get_rng_state()
        result = quantized_kv_reference(saved, upstream)
        self.assertEqual(result['valid_slot_occurrences'], {'window': 2, 'compressed': 1})
        self.assertEqual(result['fp64_sums']['grad_kv'].tolist(), [[[.5, .5]]])
        self.assertEqual(result['fp64_sums']['grad_compressed'].tolist(), [[[.25, .25]]])
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, (*saved, upstream))))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_empty_and_invalid_branches_are_zero(self):
        for args in (((), None, ()), ((-1, 1), None, (-1, 1))):
            with self.subTest(args=args):
                saved, upstream = inputs(*args)
                result = quantized_kv_reference(saved, upstream)
                self.assertEqual(result['valid_slot_occurrences'], {'window': 0, 'compressed': 0})
                self.assertTrue(all(not torch.count_nonzero(t) for t in result['fp64_sums'].values()))


if __name__ == '__main__':
    unittest.main()
