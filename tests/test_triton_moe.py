import unittest

import torch

from dsv41_train.models.dsv4.moe.kernels import clamped_swiglu, clamped_swiglu_reference


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TritonMoETest(unittest.TestCase):
    def test_changing_routed_rows_matches_reference_forward_and_backward(self):
        torch.manual_seed(42)
        for rows in (1, 7, 32, 19):
            for limit in (0.0, 2.0):
                with self.subTest(rows=rows, limit=limit):
                    # Interleaved gate/up reproduces the grouped expert layout;
                    # 1031 columns also exercises a masked tail block.
                    packed = torch.randn(rows, 1031, 2, device="cuda", dtype=torch.bfloat16)
                    gate, up = [v.detach().requires_grad_() for v in packed.unbind(-1)]
                    weights = torch.randn(rows, device="cuda", requires_grad=True)
                    expected_inputs = [v.detach().clone().requires_grad_() for v in (gate, up, weights)]
                    output = clamped_swiglu(gate, up, weights, limit)
                    expected = (clamped_swiglu_reference(*expected_inputs[:2], limit)
                                * expected_inputs[2][:, None]).to(gate.dtype)
                    gradient = torch.randn_like(output)
                    output.backward(gradient)
                    expected.backward(gradient)
                    torch.testing.assert_close(output, expected, rtol=0.02, atol=0.03)
                    for actual, reference in zip((gate, up, weights), expected_inputs):
                        torch.testing.assert_close(actual.grad, reference.grad, rtol=0.02, atol=0.03)

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
