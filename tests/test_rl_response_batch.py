"""Physical batch scoring, row isolation and native adapter-gradient regression."""
from dataclasses import replace
from pathlib import Path
import types
import unittest
from unittest.mock import patch

import torch
from torch import nn

from dsv41_train.rl.algorithm import response_log_probs, response_log_probs_batch
from dsv41_train.rl.native_lora import import_native, scoring_model_class


class CausalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(13, 8)
        self.head = nn.Linear(8, 13)
        self.calls = []

    def forward(self, input_ids, attention_mask):
        self.calls.append((input_ids.detach().clone(), attention_mask.detach().clone()))
        values = self.embedding(input_ids) * attention_mask.unsqueeze(-1)
        hidden = values.cumsum(1) / attention_mask.cumsum(1).clamp_min(1).unsqueeze(-1)
        return types.SimpleNamespace(logits=self.head(hidden))


class BatchScoringTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        self.model = CausalModel()
        self.prompts = [[0, 3, 4], [2]]
        self.responses = [[7, 0, 2], [5, 2]]

    def test_one_forward_ragged_alignment_zero_token_eos_and_gradients(self):
        expected = [response_log_probs(self.model, p, r, .8) for p, r in zip(self.prompts, self.responses)]
        expected_loss = -torch.stack([value.mean() for value in expected]).mean()
        expected_gradients = torch.autograd.grad(expected_loss, tuple(self.model.parameters()))
        self.model.calls.clear()
        actual, receipt = response_log_probs_batch(self.model, self.prompts, self.responses, .8,
                                                   return_metadata=True)
        self.assertEqual(len(self.model.calls), 1)
        ids, mask = self.model.calls[0]
        self.assertEqual(ids.tolist(), [[0, 3, 4, 7, 0], [2, 5, 0, 0, 0]])
        self.assertEqual(mask.tolist(), [[True] * 5, [True, True, False, False, False]])
        self.assertEqual(receipt["input_shape"], [2, 5])
        self.assertEqual(receipt["valid_input_tokens"], 7)
        self.assertEqual(receipt["padded_input_tokens"], 3)
        self.assertEqual(receipt["response_lengths"], [3, 2])
        for got, want in zip(actual, expected):
            torch.testing.assert_close(got, want)
        actual_loss = -torch.stack([value.mean() for value in actual]).mean()
        actual_gradients = torch.autograd.grad(actual_loss, tuple(self.model.parameters()))
        for got, want in zip(actual_gradients, expected_gradients):
            torch.testing.assert_close(got, want)
        changed = response_log_probs_batch(self.model, [self.prompts[0], [9]], [self.responses[0], [8, 1]], .8)
        torch.testing.assert_close(changed[0], actual[0])

    def test_invalid_batches_fail_before_model_construction_or_forward(self):
        cases = [([], [], .8), ([[0]], [], .8), ([[]], [[1]], .8),
                 ([[0]], [[]], .8), ([[0]], [[-1]], .8), ([[0]], [[1]], float("nan"))]
        for prompts, responses, temperature in cases:
            with self.subTest(prompts=prompts, responses=responses), self.assertRaises(ValueError):
                response_log_probs_batch(self.model, prompts, responses, temperature)
        self.assertFalse(self.model.calls)
        self.model.response_log_probs = lambda *args: None
        with self.assertRaisesRegex(ValueError, "sampling-prefix"):
            response_log_probs_batch(self.model, [[0]], [[1]])
        self.assertFalse(self.model.calls)

    def test_nonfinite_or_wrong_shape_model_output_rejected(self):
        for output in (torch.zeros(1, 1, 13), torch.full((2, 5, 13), float("nan"))):
            with self.subTest(shape=output.shape), patch.object(self.model, "forward", return_value=output):
                with self.assertRaises(ValueError):
                    response_log_probs_batch(self.model, self.prompts, self.responses)


class NativeBatchScoringTests(unittest.TestCase):
    def test_native_six_layer_compression_engram_and_checkpoint_gradients(self):
        native, _ = import_native()
        config = native("models.dsv4.config").DeepSeekV41Config.from_json(
            Path(__file__).resolve().parents[1] / "examples/rl/tiny-dsv41/config.json")
        config = replace(config, cache_quantization="hf", attention_dropout=0.)
        threads = torch.get_num_threads()
        model = None
        try:
            torch.set_num_threads(1)
            torch.manual_seed(31)
            model = scoring_model_class(native("models.dsv4.model").DeepSeekV41ForCausalLM)(config)
            lora = native("lora")
            lora.inject_lora(model, lora.LoRAConfig(8, 8, 0, ("attention.q_a",)))
            with torch.no_grad():
                for module in model.modules():
                    if hasattr(module, "lora_b"):
                        module.lora_b.normal_(0, .02)
            model.train()
            prompts, responses = [[0, 1, 3, 4], [6], [2, 7]], [[7, 0, 2], [3, 5], [0, 2]]
            parameters = tuple(lora.adapter_parameters(model))
            serial = [response_log_probs(model, p, r, .8) for p, r in zip(prompts, responses)]
            expected_gradients = torch.autograd.grad(-torch.stack([x.mean() for x in serial]).mean(), parameters)
            # Force multiple bounded head chunks while preserving one backbone.
            model.scoring_head_chunk_tokens = 2
            observed = []
            handle = model.model.register_forward_hook(lambda _m, args, _o: observed.append(list(args[0].shape)))
            for checkpointing in (False, True):
                if checkpointing:
                    model.gradient_checkpointing_enable()
                observed.clear()
                values, metadata = response_log_probs_batch(model, prompts, responses, .8, return_metadata=True)
                self.assertEqual(observed, [[3, 6]])
                self.assertEqual(metadata["input_shape"], [3, 6])
                self.assertEqual(metadata["response_lengths"], [3, 2, 2])
                gradients = torch.autograd.grad(-torch.stack([x.mean() for x in values]).mean(), parameters)
                for got, want in zip(values, serial):
                    torch.testing.assert_close(got, want, atol=2e-6, rtol=2e-6)
                for got, want in zip(gradients, expected_gradients):
                    self.assertTrue(torch.isfinite(got).all())
                    torch.testing.assert_close(got, want, atol=2e-6, rtol=2e-5)
                self.assertTrue(all(p.grad is None for p in model.parameters() if not p.requires_grad))
            with torch.no_grad():
                values, _ = response_log_probs_batch(model, prompts, responses, .8, return_metadata=True)
            self.assertTrue(all(not value.requires_grad for value in values))
            handle.remove()
            with self.assertRaisesRegex(ValueError, "right-padded"):
                model(torch.tensor([[1, 0, 2]]), torch.tensor([[1, 0, 1]]),
                      target_ids=torch.tensor([[-100, -100, 2]]), target_mask=torch.tensor([[0, 0, 1]]))
            with self.assertRaisesRegex(ValueError, "live tokens"):
                model(torch.tensor([[1, 0]]), torch.tensor([[1, 0]]),
                      target_ids=torch.tensor([[-100, 2]]), target_mask=torch.tensor([[0, 1]]))
        finally:
            if model is not None:
                model.close()
            torch.set_num_threads(threads)


if __name__ == "__main__":
    unittest.main()
