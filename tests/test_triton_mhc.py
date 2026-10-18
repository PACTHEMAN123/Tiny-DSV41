import copy
import unittest

import torch

import dsv41_train.models.dsv4.mhc as mhc
from dsv41_train.models.dsv4.config import DeepSeekV41Config
from dsv41_train.models.dsv4.mhc import HyperConnection, is_available


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TritonMHCTest(unittest.TestCase):
    def test_primitive_backward_repeats_with_tail_and_native_hidden_size(self):
        if not is_available(torch.empty((), device="cuda")):
            self.skipTest("Triton is unavailable")
        generator = torch.Generator(device="cuda").manual_seed(31)
        for dtype in (torch.bfloat16, torch.float32):
            for hidden in (513, 5120):
                streams = torch.randn(1, 17, 4, hidden, generator=generator,
                                      device="cuda", dtype=dtype)
                value = torch.randn(1, 17, hidden, generator=generator,
                                    device="cuda", dtype=dtype)
                weights = torch.randn(1, 17, 4, generator=generator, device="cuda").sigmoid()
                combine = torch.randn(1, 17, 4, 4, generator=generator, device="cuda").softmax(-1)
                for name, inputs, output_shape in (
                    ("_collapse", (streams, weights), value.shape),
                    ("_expand", (value, streams, weights, combine), streams.shape),
                ):
                    with self.subTest(operation=name, dtype=dtype, hidden=hidden):
                        upstream = torch.randn(output_shape, generator=generator,
                                               device="cuda", dtype=dtype)

                        def run(reference=False):
                            variables = tuple(x.detach().requires_grad_() for x in inputs)
                            saved = mhc.triton
                            if reference:
                                mhc.triton = None
                            try:
                                output = getattr(mhc, name)(*variables)
                                gradients = torch.autograd.grad(output, variables, upstream)
                                return (output.detach(), *gradients)
                            finally:
                                mhc.triton = saved

                        expected = run(reference=True)
                        first = run()
                        for actual, reference in zip(first, expected):
                            torch.testing.assert_close(actual, reference, rtol=0.01, atol=0.01)
                        for variable, gradient in zip(inputs, first[1:]):
                            self.assertEqual(gradient.shape, variable.shape)
                            self.assertEqual(gradient.dtype, variable.dtype)
                        for _ in range(5):
                            for actual, baseline in zip(run(), first):
                                self.assertTrue(bool(torch.isfinite(actual).all()))
                                self.assertTrue(torch.equal(actual.contiguous().view(torch.uint8),
                                                            baseline.contiguous().view(torch.uint8)))

    def test_complete_site_matches_torch_fallback(self):
        if not is_available(torch.empty((), device="cuda")):
            self.skipTest("Triton is unavailable")
        generator = torch.Generator(device="cuda").manual_seed(11)
        config = DeepSeekV41Config.tiny()
        actual = HyperConnection(config).cuda().to(torch.bfloat16)
        torch.nn.init.normal_(actual.fn, std=config.initializer_range)
        reference = copy.deepcopy(actual)
        streams = torch.randn(
            2,
            5,
            4,
            config.hidden_size,
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        input_weights = torch.randn(2, 5, 4, generator=generator, device="cuda")
        output_gradient = torch.randn_like(streams)
        pre_gradient = torch.randn(2, 5, 4, generator=generator, device="cuda")

        actual_streams = streams.detach().clone().requires_grad_()
        actual_weights = input_weights.detach().clone().requires_grad_()
        collapsed, pre, post, residual_weights = actual(
            actual_streams, actual_weights
        )
        value = torch.sin(collapsed.float()).to(collapsed.dtype)
        output = actual.finish(value, actual_streams, post, residual_weights)
        torch.autograd.backward((output, pre), (output_gradient, pre_gradient))
        actual_outputs = tuple(
            tensor.detach().clone()
            for tensor in (collapsed, pre, post, residual_weights, output)
        )
        actual_grads = [actual_streams.grad, actual_weights.grad]
        actual_grads.extend(parameter.grad for parameter in actual.parameters())

        reference_streams = streams.detach().clone().requires_grad_()
        reference_weights = input_weights.detach().clone().requires_grad_()
        saved_triton = mhc.triton
        mhc.triton = None
        try:
            collapsed, pre, post, residual_weights = reference(
                reference_streams, reference_weights
            )
            value = torch.sin(collapsed.float()).to(collapsed.dtype)
            output = reference.finish(
                value, reference_streams, post, residual_weights
            )
            torch.autograd.backward(
                (output, pre), (output_gradient, pre_gradient)
            )
        finally:
            mhc.triton = saved_triton

        reference_outputs = (collapsed, pre, post, residual_weights, output)
        reference_grads = [reference_streams.grad, reference_weights.grad]
        reference_grads.extend(parameter.grad for parameter in reference.parameters())
        for actual_output, reference_output in zip(
            actual_outputs, reference_outputs
        ):
            torch.testing.assert_close(
                actual_output, reference_output, rtol=0.01, atol=0.01
            )
        for actual_grad, reference_grad in zip(actual_grads, reference_grads):
            torch.testing.assert_close(
                actual_grad, reference_grad, rtol=0.01, atol=0.01
            )


class MHCEmptyFallbackTest(unittest.TestCase):
    def test_empty_tokens_preserve_output_and_gradient_shapes_and_dtypes(self):
        for dtype in (torch.bfloat16, torch.float32):
            streams = torch.empty(2, 0, 4, 5120, dtype=dtype, requires_grad=True)
            value = torch.empty(2, 0, 5120, dtype=dtype, requires_grad=True)
            weights = torch.empty(2, 0, 4, dtype=torch.float32, requires_grad=True)
            combine = torch.empty(2, 0, 4, 4, dtype=torch.float32, requires_grad=True)
            for name, inputs, shape in (
                ("_collapse", (streams, weights), value.shape),
                ("_expand", (value, streams, weights, combine), streams.shape),
            ):
                with self.subTest(operation=name, dtype=dtype):
                    output = getattr(mhc, name)(*inputs)
                    gradients = torch.autograd.grad(output, inputs, torch.empty_like(output))
                    self.assertEqual(output.shape, shape)
                    self.assertEqual(output.dtype, dtype)
                    for parameter, gradient in zip(inputs, gradients):
                        self.assertEqual(gradient.shape, parameter.shape)
                        self.assertEqual(gradient.dtype, parameter.dtype)
                        self.assertEqual(gradient.numel(), 0)


if __name__ == "__main__":
    unittest.main()
