import unittest
from types import SimpleNamespace

from attention_bf16_reference import quantized_kv_reference

import torch

from dsv41_train.models.dsv4.attention.sparse_attention import _SparseAttention

from dsv41_train.models.dsv4.attention import (
    fused_index_scores,
    is_available,
    sparse_attention,
)


def reference_attention(query, kv, indices, mask, compressed_kv, compressed_indices, sinks):
    rows = torch.arange(query.shape[0], device=query.device).view(-1, 1, 1)
    window_kv = kv[rows, indices]
    logits = torch.einsum("bhld,blwd->bhlw", query, window_kv)
    logits = logits * query.shape[-1] ** -0.5
    logits = logits.masked_fill(~mask.unsqueeze(1), torch.finfo(logits.dtype).min)
    compressed = compressed_kv[rows, compressed_indices.clamp_min(0)]
    picked = torch.einsum("bhld,blkd->bhlk", query, compressed.to(query.dtype))
    picked = picked * query.shape[-1] ** -0.5
    picked = picked.masked_fill(~(compressed_indices >= 0).unsqueeze(1), float("-inf"))
    logits = torch.cat((logits, picked), dim=-1)
    sink_logits = sinks.view(1, -1, 1, 1).expand(*logits.shape[:-1], 1)
    probabilities = torch.softmax(
        torch.cat((logits.float(), sink_logits), dim=-1), dim=-1
    )[..., :-1].to(query.dtype)
    window = indices.shape[-1]
    output = torch.einsum("bhlw,blwd->bhld", probabilities[..., :window], window_kv)
    return output + torch.einsum(
        "bhlk,blkd->bhld", probabilities[..., window:], compressed
    )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TritonAttentionTest(unittest.TestCase):
    def test_sink_backward_repeats_and_matches_reference(self):
        device = torch.device("cuda")
        if not is_available(torch.empty((), device=device)):
            self.skipTest("Triton is unavailable")
        generator = torch.Generator(device=device).manual_seed(47)
        for batch, heads, length, selected in ((2, 17, 257, 5), (1, 64, 4116, 0)):
            with self.subTest(batch=batch, heads=heads, length=length, selected=selected):
                dim, window, compressed_length = 64, 7, 11
                query = torch.randn(batch, heads, length, dim, generator=generator,
                                    device=device, dtype=torch.bfloat16)
                kv = torch.randn(batch, length, dim, generator=generator,
                                 device=device, dtype=torch.bfloat16)
                compressed = torch.randn(batch, compressed_length, dim,
                                         generator=generator, device=device, dtype=torch.bfloat16)
                indices = torch.arange(length, device=device)[:, None] - torch.arange(window, device=device)
                mask = (indices >= 0).expand(batch, -1, -1).contiguous()
                indices = indices.clamp_min(0).expand(batch, -1, -1).contiguous()
                picked = torch.randint(compressed_length, (batch, length, selected),
                                       generator=generator, device=device)
                if selected:
                    picked[:, ::3, -1] = -1
                sinks = torch.randn(heads, generator=generator, device=device)
                upstream = torch.randn(query.shape, generator=generator,
                                       device=device, dtype=query.dtype)
                first = None
                for _ in range(6):
                    current = sinks.detach().clone().requires_grad_()
                    output = sparse_attention(query, kv, indices, mask, compressed, picked, current)
                    gradient, = torch.autograd.grad(output, current, upstream)
                    self.assertEqual(gradient.shape, sinks.shape)
                    self.assertEqual(gradient.dtype, sinks.dtype)
                    self.assertTrue(torch.isfinite(gradient).all())
                    bits = gradient.detach().cpu().contiguous().view(torch.uint8)
                    if first is None:
                        first = bits.clone()
                    else:
                        self.assertTrue(torch.equal(bits, first), "sink gradient changed across repeats")
                expected_sinks = sinks.detach().clone().requires_grad_()
                expected = reference_attention(query, kv, indices, mask, compressed, picked, expected_sinks)
                expected_gradient, = torch.autograd.grad(expected, expected_sinks, upstream)
                torch.testing.assert_close(gradient, expected_gradient, rtol=0.04, atol=0.04)

    def test_ordered_kv_gradients_repeat_and_match_reference(self):
        device = torch.device("cuda")
        if not is_available(torch.empty((), device=device)):
            self.skipTest("Triton is unavailable")
        generator = torch.Generator(device=device).manual_seed(53)
        enabled = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        self.addCleanup(torch.use_deterministic_algorithms, enabled, warn_only=warn_only)
        torch.use_deterministic_algorithms(False)
        # Cross a 128-query chunk, exercise partial head/dimension blocks, then
        # the released H64/D512 dot shape. The remaining cases cover empty edges.
        cases = ((2, 17, 129, 80, 5, 3, False), (1, 64, 129, 512, 7, 5, False),
                 (1, 3, 5, 16, 3, 0, False), (1, 3, 5, 16, 3, 2, True),
                 (1, 3, 5, 16, 0, 0, False))
        for batch, heads, length, dim, window, selected, all_masked in cases:
            with self.subTest(batch=batch, heads=heads, dim=dim, window=window, selected=selected,
                              all_masked=all_masked):
                def random(shape, dtype=torch.bfloat16):
                    return torch.randn(shape, generator=generator, device=device, dtype=dtype)
                inputs = [random((batch, heads, length, dim)), random((batch, 23, dim)),
                          random((batch, 11 if selected else 0, dim)), random((heads,), torch.float32)]
                indices = torch.randint(23, (batch, length, window), generator=generator, device=device)
                mask = torch.rand((batch, length, window), generator=generator, device=device) > .2
                picked = torch.randint(11, (batch, length, selected), generator=generator, device=device)
                if window >= 2:
                    indices[..., :2] = 0
                    mask[..., :2] = True
                if selected >= 2:
                    picked[..., :2] = 0
                    picked[:, ::3, -1] = -1
                if all_masked:
                    mask.zero_()
                    picked.fill_(-1)
                upstream = random(inputs[0].shape)

                def run(deterministic):
                    leaves = [value.detach().requires_grad_() for value in inputs]
                    output = sparse_attention(leaves[0], leaves[1], indices, mask, leaves[2], picked,
                                              leaves[3], deterministic=deterministic)
                    return (output.detach(), *torch.autograd.grad(output, leaves, upstream))

                native = run(False)
                first = run(True)
                for _ in range(5):
                    current = run(True)
                    for actual, expected in zip(current, first):
                        self.assertTrue(torch.isfinite(actual).all())
                        self.assertTrue(torch.equal(actual.contiguous().view(torch.uint8),
                                                    expected.contiguous().view(torch.uint8)))
                for index in (0, 1, 4):  # Forward, query gradient and sink gradient retain native bits.
                    self.assertTrue(torch.equal(first[index].contiguous().view(torch.uint8),
                                                native[index].contiguous().view(torch.uint8)))
                expected_inputs = [value.detach().double().requires_grad_() for value in inputs]
                expected_output = reference_attention(expected_inputs[0], expected_inputs[1], indices, mask,
                                                       expected_inputs[2], picked, expected_inputs[3])
                expected = (expected_output, *torch.autograd.grad(expected_output, expected_inputs, upstream.double()))
                # Native backward uses its rounded saved O and BF16 P/dScore.
                # Ideal autograd remains the reference for output/dQ/dSink.
                for index in (0, 1, 4):
                    tolerance = .03 if index == 0 else .04
                    torch.testing.assert_close(first[index].double(), expected[index],
                                               rtol=tolerance, atol=tolerance)
                ctx = SimpleNamespace()
                ctx.save_for_backward = lambda *values: setattr(ctx, "saved_tensors", values)
                saved_output = _SparseAttention.forward(ctx, inputs[0], inputs[1], indices.int(), mask,
                                                        inputs[2], picked.int(), inputs[3])
                self.assertTrue(torch.equal(saved_output.contiguous().view(torch.uint8),
                                            native[0].contiguous().view(torch.uint8)))
                reference = quantized_kv_reference(tuple(t.detach().cpu() for t in ctx.saved_tensors),
                                                   upstream.detach().cpu())
                for index, name in ((2, "grad_kv"), (3, "grad_compressed")):
                    for mode in ("fp64_sums", "cast_to_kv_dtype"):
                        torch.testing.assert_close(first[index].detach().cpu().double(),
                                                   reference[mode][name].double(), rtol=.04, atol=.04)
                if all_masked or not (window or selected):
                    self.assertTrue(all(torch.count_nonzero(value) == 0 for value in first))
                # The global policy selects the class at forward time; changing
                # the flag before backward must not revert to atomic reductions.
                for enabled_during_backward in (True, False):
                    leaves = [value.detach().requires_grad_() for value in inputs]
                    torch.use_deterministic_algorithms(True)
                    output = sparse_attention(leaves[0], leaves[1], indices, mask, leaves[2], picked, leaves[3])
                    torch.use_deterministic_algorithms(enabled_during_backward)
                    global_result = (output.detach(), *torch.autograd.grad(output, leaves, upstream))
                    torch.use_deterministic_algorithms(False)
                    for actual, expected_value in zip(global_result, first):
                        self.assertTrue(torch.equal(actual.contiguous().view(torch.uint8),
                                                    expected_value.contiguous().view(torch.uint8)))

    def test_reindex_candidate_mask_matches_reference(self):
        device = torch.device("cuda")
        if not is_available(torch.empty((), device=device)):
            self.skipTest("Triton is unavailable")
        generator = torch.Generator(device=device).manual_seed(5)
        batch, length, heads, head_dim, key_length = 2, 7, 16, 64, 13
        query = torch.randn(
            batch, length, heads, head_dim,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        keys = torch.randn(
            batch, key_length, head_dim,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        weights = torch.randn(
            batch, length, heads, generator=generator, device=device
        )
        starts = torch.randint(
            0, 4, (batch, length), generator=generator, device=device
        )
        ends = torch.randint(
            7, key_length + 1, (batch, length), generator=generator, device=device
        )
        width = key_length
        candidates = torch.rand(
            batch, length, width, generator=generator, device=device
        ) > 0.35

        actual = fused_index_scores(
            query, keys, weights, starts, ends, width, candidates
        )
        offsets = torch.arange(width, device=device)
        key_indices = starts.unsqueeze(-1) + offsets
        valid = (key_indices < ends.unsqueeze(-1)) & candidates
        gathered = keys[
            torch.arange(batch, device=device).view(-1, 1, 1),
            key_indices.clamp_max(key_length - 1),
        ]
        dots = torch.einsum("blhd,blkd->blkh", query, gathered)
        expected = (
            dots.clamp_min(0).float()
            * head_dim**-0.5
            * weights.unsqueeze(2)
        ).sum(-1)
        expected = expected.masked_fill(~valid, float("-inf"))
        torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.03)

    def test_forward_and_backward_match_reference(self):
        device = torch.device("cuda")
        if not is_available(torch.empty((), device=device)):
            self.skipTest("Triton is unavailable")
        generator = torch.Generator(device=device).manual_seed(7)
        batch, heads, length, head_dim = 2, 16, 17, 64
        kv_length, window = 23, 7
        compressed_length, selected = 11, 5
        indices = torch.randint(
            kv_length, (batch, length, window), generator=generator, device=device
        )
        mask = torch.rand(
            batch, length, window, generator=generator, device=device
        ) > 0.2
        compressed_indices = torch.randint(
            compressed_length,
            (batch, length, selected),
            generator=generator,
            device=device,
        )
        compressed_indices[:, ::3, -1] = -1
        upstream = torch.randn(
            batch, heads, length, head_dim,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )

        inputs = [
            torch.randn(
                batch, heads, length, head_dim,
                generator=generator,
                device=device,
                dtype=torch.bfloat16,
            ),
            torch.randn(
                batch, kv_length, head_dim,
                generator=generator,
                device=device,
                dtype=torch.bfloat16,
            ),
            torch.randn(
                batch, compressed_length, head_dim,
                generator=generator,
                device=device,
                dtype=torch.bfloat16,
            ),
            torch.randn(heads, generator=generator, device=device),
        ]
        actual_inputs = [value.detach().clone().requires_grad_() for value in inputs]
        expected_inputs = [value.detach().clone().requires_grad_() for value in inputs]

        actual = sparse_attention(
            actual_inputs[0],
            actual_inputs[1],
            indices,
            mask,
            actual_inputs[2],
            compressed_indices,
            actual_inputs[3],
        )
        expected = reference_attention(
            expected_inputs[0],
            expected_inputs[1],
            indices,
            mask,
            expected_inputs[2],
            compressed_indices,
            expected_inputs[3],
        )
        actual.backward(upstream)
        expected.backward(upstream)

        torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.03)
        for actual_input, expected_input in zip(actual_inputs, expected_inputs):
            torch.testing.assert_close(
                actual_input.grad, expected_input.grad, rtol=0.04, atol=0.04
            )


if __name__ == "__main__":
    unittest.main()
