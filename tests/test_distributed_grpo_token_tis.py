"""Version-bound correction admission and actual CPU optimizer regression."""
import math
import types
import unittest
from unittest.mock import patch

import torch

from dsv41_train.rl.algorithm import decoupled_token_tis_loss, response_log_probs
from dsv41_train.rl.distributed_grpo import (aggregate_token_tis_metrics, initial_admission,
                                           train_group, validate_correction_configuration)
from dsv41_train.rl.native_lora import scoring_model_class
from tests.test_distributed_grpo import NativeBase


class DistributedTokenTISTests(unittest.TestCase):
    def args(self):
        return types.SimpleNamespace(temperature=.8, kl_beta=0., clip_eps=.2,
            rollout_correction="decoupled-token-tis", rollout_is_cap=2., disable_prefetch=True)

    def test_explicit_mode_preserves_strict_admission_and_rejects_stale_configuration(self):
        with self.assertRaisesRegex(ValueError, "admission bound"):
            initial_admission(.882, .1, "none")
        evidence = initial_admission(.882, .1, "decoupled-token-tis")
        self.assertFalse(evidence["within_strict_admission_bound"])
        self.assertIn("not assumed", evidence["admission_basis"])
        for value in (math.nan, math.inf):
            with self.assertRaises(ValueError):
                initial_admission(value, .1, "decoupled-token-tis")
        args = self.args()
        for name, value in (("kl_beta", .01), ("disable_prefetch", False), ("rollout_is_cap", math.inf)):
            bad = types.SimpleNamespace(**vars(args)); setattr(bad, name, value)
            with self.assertRaises(ValueError):
                validate_correction_configuration(bad)

    def test_group_statistics_pool_token_sums_while_loss_pools_response_means(self):
        currents = [torch.tensor([-1., -2.], requires_grad=True), torch.tensor([-3.], requires_grad=True)]
        olds = [current.detach().clone() for current in currents]
        behavior = [[-2., -2.], [-1.]]
        rows = [decoupled_token_tis_loss([c], [o], [b], [a])[1]
                for c, o, b, a in zip(currents, olds, behavior, [1., -1.])]
        _, expected = decoupled_token_tis_loss(currents, olds, behavior, [1., -1.])
        actual = aggregate_token_tis_metrics(rows)
        for key, value in expected.items():
            self.assertAlmostEqual(actual[key], value, places=6, msg=key)
        self.assertEqual(actual["token_count"], 3)
        self.assertNotAlmostEqual(actual["is_token_weight_ess"], sum(row["is_token_weight_ess"] for row in rows) / 2)

    def test_actual_optimizer_uses_no_grad_native_anchor_and_immutable_behavior(self):
        torch.manual_seed(4)
        model = scoring_model_class(NativeBase)()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                     lr=.03, weight_decay=0, foreach=False)
        ack = {"version": 0, "digest": "fixed", "lineage": "test", "adapter_id": 1}
        results = []
        for actions in ([3, 4], [5, 6]):
            with torch.no_grad():
                old = response_log_probs(model, [0, 2], actions, .8).tolist()
            results.append({"context_token_ids": [0, 2], "action_token_ids": actions,
                "behavior_log_probs": [v - .5 for v in old], "generation_metrics": {"policy_ack": ack}})
        results = results * 16
        behavior_before = [row["behavior_log_probs"][:] for row in results]
        calls = []
        def score(*args, **kwargs):
            calls.append(torch.is_grad_enabled())
            return response_log_probs(*args, **kwargs)
        policy = types.SimpleNamespace(begin_step=lambda: None, normalize=lambda: None)
        with patch("dsv41_train.rl.distributed_grpo.response_log_probs", side_effect=score):
            row = train_group(model, optimizer, policy, results, [1., -1.] * 16, self.args(), 1,
                              expected_ack=ack)
        self.assertEqual(calls, [False, True] * 32)
        self.assertGreater(row["gradient_norm"], 0)
        self.assertGreater(row["changed_adapter_elements"], 0)
        self.assertEqual(row["native_old_policy_ack"], ack)
        self.assertAlmostEqual(row["native_ppo_ratio_mean"], 1.)
        self.assertEqual(row["native_ppo_clip_fraction"], 0)
        self.assertAlmostEqual(row["is_weight_mean"], math.exp(.5), places=5)
        self.assertEqual([r["behavior_log_probs"] for r in results], behavior_before)
        with patch("dsv41_train.rl.distributed_grpo.response_log_probs") as forward:
            with self.assertRaisesRegex(ValueError, "current acknowledged adapter"):
                train_group(model, optimizer, policy, results, [1., -1.] * 16, self.args(), 2,
                            expected_ack=ack)
            forward.assert_not_called()


if __name__ == "__main__":
    unittest.main()
