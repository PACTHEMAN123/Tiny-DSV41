import unittest

import torch

from dsv41_train.models.dsv4.moe.kernels import clamped_swiglu


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TritonMoETest(unittest.TestCase):
    def test_clamped_swiglu_stabilizes_negative_infinity(self):
        gate = torch.tensor(
            [[-float("inf"), -100.0, 0.0, 10.0]],
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        up = torch.ones_like(gate, requires_grad=True)
        weights = torch.ones(1, device="cuda", requires_grad=True)

        output = clamped_swiglu(gate, up, weights, 10.0)
        output.sum().backward()

        torch.testing.assert_close(
            output.float(), torch.tensor([[0.0, 0.0, 0.0, 10.0]], device="cuda")
        )
        self.assertTrue(torch.isfinite(gate.grad).all())
        self.assertTrue(torch.isfinite(up.grad).all())
        self.assertTrue(torch.isfinite(weights.grad).all())

if __name__ == "__main__":
    unittest.main()
