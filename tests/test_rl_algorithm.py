import math
import unittest

import torch
from torch import nn

from dsv41_train.rl.algorithm import group_advantages, grpo_loss, response_log_probs


class TransitionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.transitions = nn.Parameter(torch.tensor([[0.0, 1.0, 2.0], [2.0, 0.0, 1.0], [1.0, 2.0, 0.0]]))

    def forward(self, input_ids, attention_mask):
        if not torch.equal(attention_mask, torch.ones_like(input_ids)):
            raise AssertionError("real token zero must not be masked")
        return self.transitions[input_ids]


class AlgorithmTests(unittest.TestCase):
    def test_group_advantages_are_normalized_and_constant_group_is_zero(self):
        result = group_advantages([1.0, 3.0])
        torch.testing.assert_close(result, torch.tensor([-1.0, 1.0]), atol=2e-6, rtol=0)
        torch.testing.assert_close(group_advantages([4.0, 4.0]), torch.zeros(2))
        torch.testing.assert_close(group_advantages([4.0]), torch.zeros(1))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            group_advantages([1.0, float("nan")])

    def test_response_alignment_temperature_and_zero_token(self):
        model = TransitionModel()
        actual = response_log_probs(model, [2, 0], [1, 2], temperature=0.5)
        expected = torch.log_softmax(model.transitions[[0, 1]] / 0.5, -1)[[0, 1], [1, 2]]
        torch.testing.assert_close(actual, expected)
        actual.sum().backward()
        self.assertGreater(model.transitions.grad[:2].abs().sum().item(), 0)
        self.assertEqual(model.transitions.grad[2].abs().sum().item(), 0)

    def test_clipping_handles_positive_and_negative_advantages(self):
        behavior = [[math.log(0.25)], [math.log(0.25)]]
        current = [
            torch.tensor([math.log(0.25 * 1.5)], requires_grad=True),
            torch.tensor([math.log(0.25 * 0.5)], requires_grad=True),
        ]
        loss, metrics = grpo_loss(current, behavior, [1.0, -1.0])
        self.assertAlmostEqual(loss.item(), (-1.2 + 0.8) / 2, places=6)
        self.assertEqual(metrics["clip_fraction"], 1.0)
        loss.backward()
        for value in current:
            self.assertEqual(value.grad.item(), 0.0)

    def test_loss_averages_each_response_before_the_batch(self):
        behavior = [[math.log(0.25)], [math.log(0.25)] * 3]
        current = [torch.tensor([math.log(0.25 * 1.1)]), torch.tensor([math.log(0.25 * 0.7)] * 3)]
        loss, _ = grpo_loss(current, behavior, [1.0, 1.0])
        self.assertAlmostEqual(loss.item(), -0.9, places=6)

    def test_unclipped_policy_gradient_has_expected_direction(self):
        current = [torch.tensor([-1.0], requires_grad=True), torch.tensor([-1.0], requires_grad=True)]
        behavior = [value.detach().clone() for value in current]
        advantage = torch.tensor([1.0, -1.0], requires_grad=True)
        loss, _ = grpo_loss(current, behavior, advantage)
        loss.backward()
        self.assertLess(current[0].grad.item(), 0)
        self.assertGreater(current[1].grad.item(), 0)
        self.assertIsNone(advantage.grad)

    def test_off_policy_weighted_kl_matches_exact_value_and_gradient(self):
        theta = torch.tensor([0.3, -0.5], requires_grad=True)
        log_pi = torch.log_softmax(theta, -1)
        behavior = torch.tensor([0.7, 0.3])
        reference = torch.tensor([0.4, 0.6])
        expected_loss = 0
        for token in range(2):
            sampled_loss, _ = grpo_loss(
                [log_pi[token : token + 1]],
                [behavior[token : token + 1].log()],
                [0.0],
                ref_log_probs=[reference[token : token + 1].log()],
                kl_beta=1.0,
            )
            expected_loss = expected_loss + behavior[token] * sampled_loss
        exact_kl = (log_pi.exp() * (log_pi - reference.log())).sum()
        torch.testing.assert_close(expected_loss, exact_kl)
        actual_gradient = torch.autograd.grad(expected_loss, theta, retain_graph=True)[0]
        exact_gradient = torch.autograd.grad(exact_kl, theta)[0]
        torch.testing.assert_close(actual_gradient, exact_gradient)

    def test_invalid_probability_shapes_and_numerics_fail(self):
        with self.assertRaisesRegex(ValueError, "response lengths"):
            grpo_loss([torch.tensor([-1.0, -2.0])], [[-1.0]], [1.0])
        with self.assertRaisesRegex(ValueError, "requires reference"):
            grpo_loss([torch.tensor([-1.0])], [[-1.0]], [1.0], kl_beta=0.1)
        with self.assertRaisesRegex(ValueError, "importance ratios"):
            grpo_loss([torch.tensor([-1.0])], [[-1000.0]], [1.0])
        with self.assertRaisesRegex(ValueError, "nonempty"):
            grpo_loss([torch.empty(0)], [[]], [1.0])
        with self.assertRaisesRegex(ValueError, "temperature"):
            response_log_probs(TransitionModel(), [0], [1], temperature=0)


if __name__ == "__main__":
    unittest.main()
