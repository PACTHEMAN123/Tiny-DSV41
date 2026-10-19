import math
import unittest

import torch

from dsv41_train.rl.algorithm import decoupled_token_tis_loss, grpo_loss


class TokenTISTests(unittest.TestCase):
    def test_initial_native_ratio_one_has_hand_gradient_and_detached_correction(self):
        old = [torch.tensor([-2., -2.], requires_grad=True), torch.tensor([-2.], requires_grad=True)]
        behavior = [torch.tensor([-2. - math.log(3), -2. - math.log(.25)], requires_grad=True),
                    torch.tensor([-2. - math.log(.5)], requires_grad=True)]
        current = [v.detach().clone().requires_grad_() for v in old]
        advantage = torch.tensor([1., -1.], requires_grad=True)
        loss, m = decoupled_token_tis_loss(current, old, behavior, advantage, is_cap=2)
        # Mean responses, not mean all3 tokens: -(2+.25)/2/2 + .5/2.
        self.assertAlmostEqual(loss.item(), -.3125, places=6)
        loss.backward()
        torch.testing.assert_close(current[0].grad, torch.tensor([-.5, -.0625]))
        torch.testing.assert_close(current[1].grad, torch.tensor([.25]))
        self.assertTrue(all(v.grad is None for v in old + behavior + [advantage]))
        self.assertEqual(m['native_ppo_clip_fraction'], 0)
        self.assertEqual(m['is_clipped_tokens'], 1)
        self.assertAlmostEqual(m['is_clipped_fraction'], 1/3)
        self.assertAlmostEqual(m['is_weight_sum'], 2.75, places=6)
        self.assertAlmostEqual(m['is_raw_weight_sum'], 3.75, places=6)
        self.assertAlmostEqual(m['is_dropped_weight_mass'], 1, places=6)
        self.assertAlmostEqual(m['is_token_weight_ess'], 2.75**2/(4+.25**2+.5**2), places=6)

    def test_hand_clipped_and_unclipped_derivatives_for_both_advantage_signs(self):
        # Positive A clips only high r; negative A clips only low r.
        for sign, ratio, derivative in [(1, 1.5, 0), (1, .5, -.5),
                                        (-1, .5, 0), (-1, 1.5, 1.5)]:
            with self.subTest(sign=sign, ratio=ratio):
                current = torch.tensor([-3. + math.log(ratio)], requires_grad=True)
                loss, m = decoupled_token_tis_loss([current], [[-3.]], [[-3.]], [sign])
                expected = -min(ratio*sign, max(.8, min(1.2, ratio))*sign)
                self.assertAlmostEqual(loss.item(), expected, places=6)
                loss.backward()
                self.assertAlmostEqual(current.grad.item(), derivative, places=6)
                self.assertEqual(m['native_ppo_clip_fraction'], 1)

    def test_upper_IS_cap_is_separate_from_PPO_clipping_without_lower_clip(self):
        current = torch.tensor([-3., -3.], requires_grad=True)
        loss, m = decoupled_token_tis_loss([current], [[-3., -3.]],
                                          [[-3.-math.log(4), -3.-math.log(.1)]], [1.], is_cap=2)
        loss.backward()
        torch.testing.assert_close(current.grad, torch.tensor([-1., -.05]))
        self.assertAlmostEqual(loss.item(), -1.05, places=6)
        self.assertAlmostEqual(m['is_weight_min'], .1, places=6)
        self.assertEqual(m['is_weight_max'], 2)
        self.assertEqual(m['native_ppo_clip_fraction'], 0)
        # A cap8 changes only correction weights, never the r=1 PPO ratio.
        _, other = decoupled_token_tis_loss([current], [[-3., -3.]],
                                           [[-3.-math.log(4), -3.-math.log(.1)]], [1.], is_cap=8)
        self.assertEqual(other['is_clipped_fraction'], 0)
        self.assertEqual(other['native_ppo_clip_fraction'], 0)

    def test_matching_native_and_behavior_reduces_to_existing_objective(self):
        current = [torch.tensor([-2., -3.]), torch.tensor([-1.5])]
        old = [[-2.1, -2.9], [-1.4]]
        expected, _ = grpo_loss(current, old, [1., -1.])
        actual, m = decoupled_token_tis_loss(current, old, old, [1., -1.])
        torch.testing.assert_close(actual, expected)
        self.assertEqual(m['is_token_weight_ess'], 3)
        self.assertEqual(m['is_dropped_weight_mass'], 0)

    def test_invalid_shapes_probabilities_and_finite_guards(self):
        valid = ([torch.tensor([-2.])], [[-2.]], [[-2.]], [1.])
        for kwargs in ({'is_cap': 0}, {'is_cap': float('inf')}, {'clip_eps': 1}, {'clip_eps': float('nan')}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                decoupled_token_tis_loss(*valid, **kwargs)
        for current, old, behavior, advantage in [([], [], [], []),
                ([torch.empty(0)], [[]], [[]], [1.]),
                ([torch.tensor([-2.])], [[-2., -3.]], [[-2.]], [1.]),
                ([torch.tensor([-2.])], [[-2.]], [[-2.]], [float('nan')]),
                ([torch.tensor([.1])], [[-2.]], [[-2.]], [1.]),
                ([torch.tensor([-2.])], [[float('nan')]], [[-2.]], [1.]),
                ([torch.tensor([-2.])], [[-2.]], [[float('-inf')]], [1.]),
                ([torch.tensor([-2.])], [[-1000.]], [[-2.]], [1.]),
                ([torch.tensor([-1000.])], [[-2.]], [[-2.]], [1.]),
                ([torch.tensor([-2.])], [[-2.]], [[-1000.]], [1.])]:
            with self.subTest(current=current, old=old, behavior=behavior), self.assertRaises(ValueError):
                decoupled_token_tis_loss(current, old, behavior, advantage)


if __name__ == '__main__':
    unittest.main()
