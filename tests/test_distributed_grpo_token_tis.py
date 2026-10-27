"""Version-bound correction admission and actual CPU optimizer regression."""
import copy
import math
import types
import unittest
from unittest.mock import patch

import torch

from dsv41_train.rl.algorithm import decoupled_token_tis_loss, response_log_probs
from dsv41_train.rl.distributed_grpo import (aggregate_token_tis_metrics, initial_admission,
                                           train_group, validate_correction_configuration)
from dsv41_train.rl.native_lora import (adapter_parameters_at, adapter_snapshot,
                                      scoring_model_class)
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
        args.disable_prefetch = False
        self.assertEqual(validate_correction_configuration(args), "decoupled-token-tis")
        for name, value in (("kl_beta", .01), ("rollout_is_cap", math.inf)):
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
            with self.assertRaisesRegex(ValueError, "generating policy"):
                train_group(model, optimizer, policy, results, [1., -1.] * 16, self.args(), 2,
                            expected_ack=ack)
            forward.assert_not_called()

    def test_stale_anchor_gradient_and_adamw_match_independent_generating_model(self):
        for microbatch in (1, 4):
            with self.subTest(microbatch=microbatch):
                torch.manual_seed(41)
                model = scoring_model_class(NativeBase)()
                old_model = scoring_model_class(NativeBase)()
                old_model.load_state_dict(model.state_dict())
                parameters = [p for p in model.parameters() if p.requires_grad]
                optimizer = torch.optim.AdamW(parameters, lr=.03, weight_decay=0, foreach=False)
                old_ack = {"version": 0, "digest": "old", "lineage": "test", "adapter_id": 1}
                current_ack = {**old_ack, "version": 1, "digest": "current", "adapter_id": 2}
                snapshot = {"ack": old_ack, "parameters": adapter_snapshot(optimizer)}
                # Advance weights and Adam moments before consuming version-zero samples.
                for parameter in parameters:
                    parameter.grad = torch.full_like(parameter, .05)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                oracle = scoring_model_class(NativeBase)()
                oracle.load_state_dict(model.state_dict())
                oracle_parameters = [p for p in oracle.parameters() if p.requires_grad]
                oracle_optimizer = torch.optim.AdamW(oracle_parameters, lr=.03, weight_decay=0, foreach=False)
                oracle_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
                results, old_scores = [], []
                for index in range(32):
                    actions = [3, 4] if index % 2 == 0 else [5, 6, 7]
                    with torch.no_grad():
                        old = response_log_probs(old_model, [0, 2], actions, .8)
                    old_scores.append(old)
                    results.append({"context_token_ids": [0, 2], "action_token_ids": actions,
                        "behavior_log_probs": (old - .5).tolist(), "generation_metrics": {"policy_ack": old_ack}})
                original_results = copy.deepcopy(results)
                advantages = [1., -1.] * 16
                oracle_optimizer.zero_grad(set_to_none=True)
                losses = []
                for result, old, advantage in zip(results, old_scores, advantages):
                    current = response_log_probs(oracle, result["context_token_ids"], result["action_token_ids"], .8)
                    loss, _ = decoupled_token_tis_loss([current], [old], [result["behavior_log_probs"]], [advantage])
                    losses.append(loss)
                oracle_loss = torch.stack(losses).mean()
                oracle_loss.backward()
                oracle_optimizer.step()
                args = self.args()
                args.disable_prefetch = False
                args.learner_microbatch_size = microbatch
                policy = types.SimpleNamespace(begin_step=lambda: None, normalize=lambda: None)
                row = train_group(model, optimizer, policy, results, advantages, args, 2,
                                  expected_ack=current_ack, native_old_snapshot=snapshot)
                self.assertEqual(row["policy_lag"], 1)
                self.assertEqual(row["native_old_policy_ack"], old_ack)
                self.assertEqual(row["learner_policy_ack"], current_ack)
                self.assertAlmostEqual(row["loss"], oracle_loss.item(), places=6)
                self.assertAlmostEqual(row["is_weight_mean"], math.exp(.5), places=5)
                self.assertNotAlmostEqual(row["native_ppo_ratio_mean"], 1., places=5)
                for actual, expected in zip(parameters, oracle_parameters):
                    torch.testing.assert_close(actual.grad, expected.grad, rtol=2e-5, atol=2e-7)
                    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-7)
                    for name, value in optimizer.state[actual].items():
                        torch.testing.assert_close(value, oracle_optimizer.state[expected][name], rtol=2e-5, atol=2e-7)
                self.assertEqual(results, original_results)
                for value, expected in zip(snapshot["parameters"], [p for p in old_model.parameters() if p.requires_grad]):
                    torch.testing.assert_close(value, expected, rtol=0, atol=0)
                # No stale group may enter without its immutable snapshot, in
                # synchronous mode, across lineages, or more than one version late.
                bad_calls = [({}, current_ack, 2, None),
                    ({"disable_prefetch": True}, current_ack, 2, snapshot),
                    ({}, {**current_ack, "lineage": "other"}, 2, snapshot),
                    ({}, {**current_ack, "version": 2}, 3, snapshot)]
                for overrides, ack, update, state in bad_calls:
                    bad_args = types.SimpleNamespace(**(vars(args) | overrides))
                    with patch("dsv41_train.rl.distributed_grpo.response_log_probs_batch") as score:
                        with self.assertRaises(ValueError):
                            train_group(model, optimizer, policy, results, advantages, bad_args, update,
                                        expected_ack=ack, native_old_snapshot=state)
                        score.assert_not_called()

    def _strict_case(self, current_behavior=False):
        torch.manual_seed(7)
        model = scoring_model_class(NativeBase)()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                     lr=.03, weight_decay=0, foreach=False)
        old_ack = {"version": 0, "digest": "old", "lineage": "strict-test"}
        ack = {**old_ack, "version": 1, "digest": "current"}
        results = []
        for actions in ([3, 4], [5, 6]):
            with torch.no_grad():
                behavior = response_log_probs(model, [0, 2], actions, .8).tolist()
            results.append({"context_token_ids": [0, 2], "action_token_ids": actions,
                "behavior_log_probs": behavior, "generation_metrics": {"policy_ack": old_ack}})
        with torch.no_grad():
            model.model.adapter.lora_b.add_(.01)
            if current_behavior:
                for result in results:
                    result["behavior_log_probs"] = response_log_probs(
                        model, [0, 2], result["action_token_ids"], .8).tolist()
                    result["generation_metrics"] = {"policy_ack": ack}
        args = self.args()
        args.rollout_correction, args.disable_prefetch = "none", False
        policy = types.SimpleNamespace(begin_step=lambda: None, normalize=lambda: None)
        return model, optimizer, policy, results * 16, args, ack

    def test_strict_policy_lag_uses_behavior_version_and_preserves_unknown_identity(self):
        for current_behavior, known_identity, expected in ((False, True, 1), (True, True, 0), (True, False, None)):
            with self.subTest(current_behavior=current_behavior, known_identity=known_identity):
                model, optimizer, policy, results, args, ack = self._strict_case(current_behavior)
                row = train_group(model, optimizer, policy, results, [1., -1.] * 16, args, 2,
                                  expected_ack=ack if known_identity else None)
                self.assertEqual(row["policy_lag"], expected)
                self.assertIsNone(row["native_old_policy_ack"])
                self.assertGreater(row["gradient_norm"], 0)
                self.assertGreater(row["changed_adapter_elements"], 0)

    def test_strict_policy_lag_rejects_mixed_or_excessively_stale_groups_before_scoring(self):
        for invalid in ("mixed", "lag_two", "synchronous_lag_one"):
            with self.subTest(invalid=invalid):
                model, optimizer, policy, results, args, ack = self._strict_case()
                update = 2
                if invalid == "mixed":
                    results[0] = {**results[0], "generation_metrics": {"policy_ack": ack}}
                elif invalid == "lag_two":
                    ack, update = {**ack, "version": 2}, 3
                else:
                    args.disable_prefetch = True
                with patch("dsv41_train.rl.distributed_grpo.response_log_probs") as score:
                    with self.assertRaisesRegex(ValueError, "strict"):
                        train_group(model, optimizer, policy, results, [1., -1.] * 16, args, update,
                                    expected_ack=ack)
                    score.assert_not_called()

    def test_generating_adapter_context_restores_parameters_and_gradients_on_failure(self):
        model = scoring_model_class(NativeBase)()
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters)
        for parameter in parameters:
            parameter.grad = torch.ones_like(parameter)
        before = adapter_snapshot(optimizer)
        old = [value + .1 for value in before]
        with self.assertRaisesRegex(RuntimeError, "scorer failed"):
            with adapter_parameters_at(model, optimizer, old):
                self.assertFalse(torch.is_grad_enabled())
                for parameter, value in zip(parameters, old):
                    torch.testing.assert_close(parameter, value, rtol=0, atol=0)
                raise RuntimeError("scorer failed")
        for parameter, value in zip(parameters, before):
            torch.testing.assert_close(parameter, value, rtol=0, atol=0)
            torch.testing.assert_close(parameter.grad, torch.ones_like(parameter), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
