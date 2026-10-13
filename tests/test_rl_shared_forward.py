import unittest

import torch

from dsv41_train.rl.algorithm import response_log_probs
from dsv41_train.rl.config import RLConfig
from dsv41_train.rl.policy import DSV41Config
from dsv41_train.rl.runtime import make_model
from test_rl_policy import MODEL_DIR


class SharedForwardTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.config = DSV41Config.from_model_dir(MODEL_DIR)

    def policy(self, **settings):
        return make_model(MODEL_DIR, self.config, RLConfig(seed=37, **settings), ("cpu",), True)

    def assert_gradients_match(self, actual, expected):
        expected_parameters = dict(expected.named_parameters())
        for name, parameter in actual.named_parameters():
            reference = expected_parameters[name]
            with self.subTest(parameter=name):
                if reference.grad is None:
                    self.assertIsNone(parameter.grad)
                else:
                    self.assertIsNotNone(parameter.grad)
                    torch.testing.assert_close(parameter.grad, reference.grad, rtol=2e-4, atol=2e-6)

    def test_selected_scores_match_full_logits_and_gradients_with_padding(self):
        reference, policy = self.policy(), self.policy()
        tokens = torch.tensor([[1, 4, 7, 3, 8, 9], [1, 5, 0, 6, 0, 0]])
        attention = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]], dtype=torch.bool)
        targets = torch.tensor([[-100, -100, 3, 8, 9, 2], [-100, 0, 6, 2, -100, -100]])
        requested = torch.tensor([[0, 0, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1]], dtype=torch.bool)
        valid = requested & attention
        logits = reference(tokens, attention_mask=attention)
        expected = torch.log_softmax(logits[valid].float() / 0.8, dim=-1).gather(
            -1, targets[valid].unsqueeze(-1)).squeeze(-1)
        output = policy.backend(tokens, attention_mask=attention, target_ids=targets,
                                target_mask=requested, temperature=0.8, compute_aux_loss=False)
        self.assertIsNone(output.logits)
        self.assertIsNone(output.loss)
        self.assertIsNone(output.aux_loss)
        self.assertEqual(output.token_log_probs.shape, tokens.shape)
        self.assertEqual(output.token_log_probs.dtype, torch.float32)
        torch.testing.assert_close(output.token_log_probs[valid], expected)
        torch.testing.assert_close(output.token_log_probs[~valid], torch.zeros(5))
        expected.square().sum().backward()
        output.token_log_probs.square().sum().backward()
        self.assert_gradients_match(policy, reference)
        policy.assert_engram_invariants()

    def test_rollout_and_response_scoring_enter_managed_root_without_aux_loss(self):
        policy = self.policy()
        outputs = []
        head_shapes = []
        handle = policy.backend.register_forward_hook(lambda module, args, output: outputs.append(output))
        head_handle = policy.backend.lm_head.register_forward_pre_hook(
            lambda module, args: head_shapes.append(tuple(args[0].shape)))
        try:
            tokens = torch.tensor([[1, 4, 7, 3, 8]])
            logits = policy(tokens)
            actual = response_log_probs(policy, [1, 4, 7], [3, 8, 2], temperature=0.8)
        finally:
            handle.remove()
            head_handle.remove()
        self.assertEqual(len(outputs), 2)
        self.assertTrue(all(output.aux_loss is None for output in outputs))
        self.assertIsNone(outputs[1].logits)
        self.assertEqual(head_shapes, [(1, 5, self.config.hidden_size), (3, self.config.hidden_size)])
        expected = torch.log_softmax(logits[0, 2:].float() / 0.8, -1)[torch.arange(3), [3, 8, 2]]
        torch.testing.assert_close(actual, expected)

    def test_checkpointing_recomputes_decoder_without_changing_response_gradients(self):
        reference = self.policy()
        policy = self.policy(gradient_checkpointing=True)
        calls = [0] * self.config.num_hidden_layers

        def count(index):
            def record(module, args):
                calls[index] += 1
            return record

        handles = [layer.register_forward_pre_hook(count(i))
                   for i, layer in enumerate(policy.backend.model.layers)]
        try:
            expected = response_log_probs(reference, [1, 4, 7, 3, 8, 9], [5, 6, 2], 0.8)
            actual = response_log_probs(policy, [1, 4, 7, 3, 8, 9], [5, 6, 2], 0.8)
            self.assertEqual(calls, [1] * len(calls))
            torch.testing.assert_close(actual, expected)
            expected.sum().backward()
            actual.sum().backward()
            self.assertTrue(all(value > 1 for value in calls), calls)
            self.assert_gradients_match(policy, reference)
        finally:
            for handle in handles:
                handle.remove()

    def test_empty_score_mask_has_zero_finite_gradients(self):
        policy = self.policy()
        tokens = torch.tensor([[1, 4, 7]])
        output = policy.backend(tokens, target_ids=torch.full_like(tokens, -100),
                                target_mask=torch.zeros_like(tokens, dtype=torch.bool),
                                compute_aux_loss=False)
        torch.testing.assert_close(output.token_log_probs, torch.zeros(1, 3))
        output.token_log_probs.sum().backward()
        for parameter in policy.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertEqual(parameter.grad.count_nonzero(), 0)

    def test_score_inputs_reject_ambiguous_alignment_and_invalid_targets(self):
        policy = self.policy()
        tokens = torch.tensor([[1, 4, 7]])
        for kwargs in (
            {"target_ids": tokens, "labels": tokens},
            {"target_mask": torch.ones_like(tokens)},
            {"target_ids": tokens[:, :2]},
            {"target_ids": tokens, "target_mask": torch.ones(1, 2)},
            {"target_ids": tokens.float()},
            {"target_ids": torch.tensor([[1, 4, 32]])},
            {"target_ids": torch.tensor([[1, -1, 7]])},
            {"target_ids": tokens, "temperature": 0},
            {"target_ids": tokens, "temperature": float("nan")},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                policy.backend(tokens, **kwargs)

    def test_sft_labels_still_shift_and_mask_packed_boundaries(self):
        policy = self.policy()
        policy.eval()
        tokens = torch.tensor([[1, 4, 7, 3, 8, 0]])
        mask = torch.tensor([[1, 1, 1, 1, 1, 0]], dtype=torch.bool)
        sequences = torch.tensor([[0, 0, 0, 1, 1, -1]])
        output = policy.backend(tokens, attention_mask=mask, labels=tokens, sequence_ids=sequences)
        expected = torch.nn.functional.cross_entropy(output.logits[0, [0, 1, 3]].float(),
                                                     torch.tensor([4, 7, 8]))
        torch.testing.assert_close(output.loss, expected + self.config.router_aux_loss_coef * output.aux_loss.detach())
        self.assertIsNone(output.token_log_probs)


if __name__ == "__main__":
    unittest.main()
