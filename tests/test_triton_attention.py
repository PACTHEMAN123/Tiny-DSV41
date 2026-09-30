import unittest

import torch

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
