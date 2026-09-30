import unittest

import torch

from dsv41_train.models.dsv4.mhc import (
    collapse,
    collapse_reference,
    expand,
    expand_reference,
    is_available,
    mixing_weights,
    mixing_weights_reference,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TritonMHCTest(unittest.TestCase):
    def setUp(self):
        if not is_available(torch.empty((), device="cuda")):
            self.skipTest("Triton is unavailable")

    def test_mixing_weights_forward_and_backward_match_reference(self):
        generator = torch.Generator(device="cuda").manual_seed(11)
        projected = torch.randn(2, 7, 24, generator=generator, device="cuda")
        base = torch.randn(24, generator=generator, device="cuda") * 0.2
        scale = torch.tensor([0.8, 1.1, 0.6], device="cuda")
        upstream = (
            torch.randn(2, 7, 4, generator=generator, device="cuda"),
            torch.randn(2, 7, 4, generator=generator, device="cuda"),
            torch.randn(2, 7, 4, 4, generator=generator, device="cuda"),
        )
        actual_inputs = [
            value.detach().clone().requires_grad_()
            for value in (projected, base, scale)
        ]
        expected_inputs = [
            value.detach().clone().requires_grad_()
            for value in (projected, base, scale)
        ]
        actual = mixing_weights(*actual_inputs, 4, 20, 1.0e-6)
        expected = mixing_weights_reference(*expected_inputs, 4, 20, 1.0e-6)
        torch.autograd.backward(actual, upstream)
        torch.autograd.backward(expected, upstream)

        for actual_output, expected_output in zip(actual, expected):
            torch.testing.assert_close(
                actual_output, expected_output, rtol=2e-5, atol=2e-5
            )
        for actual_input, expected_input in zip(actual_inputs, expected_inputs):
            torch.testing.assert_close(
                actual_input.grad, expected_input.grad, rtol=3e-4, atol=3e-4
            )

    def test_collapse_forward_and_backward_match_reference(self):
        generator = torch.Generator(device="cuda").manual_seed(13)
        streams = torch.randn(
            2,
            5,
            4,
            320,
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        weights = torch.randn(2, 5, 4, generator=generator, device="cuda")
        upstream = torch.randn(
            2,
            5,
            320,
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        actual_inputs = [
            value.detach().clone().requires_grad_() for value in (streams, weights)
        ]
        expected_inputs = [
            value.detach().clone().requires_grad_() for value in (streams, weights)
        ]
        actual = collapse(*actual_inputs)
        expected = collapse_reference(*expected_inputs)
        actual.backward(upstream)
        expected.backward(upstream)

        torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.01)
        torch.testing.assert_close(
            actual_inputs[0].grad, expected_inputs[0].grad, rtol=0.01, atol=0.01
        )
        torch.testing.assert_close(
            actual_inputs[1].grad, expected_inputs[1].grad, rtol=5e-4, atol=5e-4
        )

    def test_expand_forward_and_backward_match_reference(self):
        generator = torch.Generator(device="cuda").manual_seed(17)
        value = torch.randn(
            2,
            5,
            320,
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        residual = torch.randn(
            2,
            5,
            4,
            320,
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        output_weights = torch.randn(2, 5, 4, generator=generator, device="cuda")
        residual_weights = torch.randn(2, 5, 4, 4, generator=generator, device="cuda")
        upstream = torch.randn_like(residual)
        values = (value, residual, output_weights, residual_weights)
        actual_inputs = [item.detach().clone().requires_grad_() for item in values]
        expected_inputs = [item.detach().clone().requires_grad_() for item in values]
        actual = expand(*actual_inputs)
        expected = expand_reference(*expected_inputs)
        actual.backward(upstream)
        expected.backward(upstream)

        torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.01)
        for index, (actual_input, expected_input) in enumerate(
            zip(actual_inputs, expected_inputs)
        ):
            tolerance = 0.01 if index < 2 else 8e-4
            torch.testing.assert_close(
                actual_input.grad, expected_input.grad, rtol=tolerance, atol=tolerance
            )


if __name__ == "__main__":
    unittest.main()
