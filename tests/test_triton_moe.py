import unittest

import torch

from dsv41_train.models.dsv4.moe import kernels
from dsv41_train.models.dsv4.moe.kernels import clamped_swiglu, clamped_swiglu_reference


@unittest.skipUnless(kernels.triton is not None, "Triton registration is required")
class TritonMoEFakeTest(unittest.TestCase):
    def test_backward_fake_matches_fp32_weight_gradient_and_contiguous_layout(self):
        for rows in (0, 7):
            for dtype in (torch.bfloat16, torch.float32):
                with self.subTest(rows=rows, weights=dtype):
                    gate = torch.empty(2304, rows, dtype=torch.bfloat16, device="meta").t()
                    weights = torch.empty(rows, dtype=dtype, device="meta")
                    gradients = kernels._clamped_swiglu_backward_fake(gate, gate, gate, weights, 10.)
                    for value, gradient in zip((gate, gate, weights), gradients):
                        self.assertEqual(value.shape, gradient.shape)
                        self.assertTrue(gradient.is_contiguous())
                    self.assertEqual(gradients[0].dtype, gate.dtype)
                    self.assertEqual(gradients[1].dtype, gate.dtype)
                    self.assertEqual(gradients[2].dtype, torch.float32)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TritonMoETest(unittest.TestCase):
    def test_three_block_weight_gradient_is_repeatable(self):
        generator = torch.Generator().manual_seed(42)
        packed = torch.randn(256, 2304, 2, generator=generator).bfloat16().cuda()
        gate, up = [v.detach().requires_grad_() for v in packed.unbind(-1)]
        weights = torch.randn(256, generator=generator).cuda().requires_grad_()
        upstream = torch.randn(256, 2304, generator=generator).bfloat16().cuda()
        first = None
        for _ in range(30):
            output = clamped_swiglu(gate, up, weights, 10.)
            gradients = torch.autograd.grad(output, (gate, up, weights), upstream)
            values = (output, *gradients)
            self.assertTrue(all(bool(torch.isfinite(value).all()) for value in values))
            if first is None:
                first = tuple(value.detach().clone() for value in values)
            else:
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(values, first)))
        reference = clamped_swiglu_reference(gate, up, 10.) * weights[:, None]
        expected, = torch.autograd.grad(reference, weights, upstream.float())
        torch.testing.assert_close(first[-1], expected, rtol=1e-4, atol=2e-5)

    def test_empty_routed_rows_forward_and_backward(self):
        gate = torch.empty(0, 2304, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        up = torch.empty_like(gate, requires_grad=True)
        weights = torch.empty(0, device="cuda", requires_grad=True)
        output = clamped_swiglu(gate, up, weights, 10.)
        gradients = torch.autograd.grad(output.sum(), (gate, up, weights))
        for value, gradient in zip((gate, up, weights), gradients):
            self.assertEqual(value.shape, gradient.shape)
            self.assertEqual(value.dtype, gradient.dtype)

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
