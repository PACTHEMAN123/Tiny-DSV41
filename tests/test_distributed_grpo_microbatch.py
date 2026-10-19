"""Physical learner batches preserve the complete G32 response objective."""
import copy
import types
import unittest
from unittest.mock import patch

import torch

from dsv41_train.rl.algorithm import response_log_probs, response_log_probs_batch
from dsv41_train.rl.distributed_grpo import train_group
from dsv41_train.rl.native_lora import scoring_model_class
from tests.test_distributed_grpo import NativeBase


class LearnerMicrobatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def fixture(self):
        torch.manual_seed(19)
        model = scoring_model_class(NativeBase)()
        ack = {"version": 0, "digest": "fixed", "lineage": "microbatch"}
        results = []
        for index in range(32):
            prompt = [0, 2, 3][:2 + index % 2]
            actions = ([3], [4, 5, 6], [7, 0])[index % 3]
            with torch.no_grad():
                behavior = response_log_probs(model, prompt, actions, .8).tolist()
            results.append({"context_token_ids": prompt, "action_token_ids": actions,
                "behavior_log_probs": [value - .5 for value in behavior],
                "generation_metrics": {"policy_ack": dict(ack)}})
        args = types.SimpleNamespace(temperature=.8, kl_beta=0., clip_eps=.2,
            rollout_correction="decoupled-token-tis", rollout_is_cap=2., disable_prefetch=True,
            learner_microbatch_size=1)
        return model, results, [1. if index % 2 else -1. for index in range(32)], args, ack

    def execute(self, model, results, advantages, args, ack):
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.03)
        policy = types.SimpleNamespace(begin_step=lambda: None, normalize=lambda: None)
        calls = []
        hook = model.register_forward_pre_hook(
            lambda module, inputs: calls.append((torch.is_grad_enabled(), list(inputs[0].shape))))
        try:
            with patch.object(optimizer, "step", wraps=optimizer.step) as step:
                metrics = train_group(model, optimizer, policy, results, advantages, args, 1,
                                      expected_ack=ack)
            self.assertEqual(step.call_count, 1)
        finally:
            hook.remove()
        return metrics, calls

    def test_actual_batches_match_serial_gradients_and_keep_unequal_response_weighting(self):
        for size in (8, 7):  # 7 exercises a smaller final physical batch.
            with self.subTest(size=size):
                model, results, advantages, args, ack = self.fixture()
                batched = copy.deepcopy(model)
                frozen = {name: p.detach().clone() for name, p in batched.named_parameters()
                          if not p.requires_grad}
                serial, _ = self.execute(model, results, advantages, args, ack)
                args.learner_microbatch_size = size
                actual, calls = self.execute(batched, results, advantages, args, ack)
                for (name, left), (_, right) in zip(model.named_parameters(), batched.named_parameters()):
                    torch.testing.assert_close(left, right, atol=2e-7, rtol=2e-6, msg=name)
                    if left.requires_grad:
                        torch.testing.assert_close(left.grad, right.grad, atol=2e-7, rtol=2e-6, msg=name)
                    else:
                        self.assertIsNone(right.grad)
                        torch.testing.assert_close(right, frozen[name], atol=0, rtol=0)
                self.assertAlmostEqual(actual["loss"], serial["loss"], places=6)
                self.assertAlmostEqual(actual["gradient_norm"], serial["gradient_norm"], places=6)
                evidence = actual["learner_batching"]
                widths = [min(size, 32 - start) for start in range(0, 32, size)]
                self.assertEqual(evidence["physical_batch_sizes"], widths)
                self.assertEqual(evidence["backward_calls"], len(widths))
                self.assertEqual(evidence["model_forward_calls"],
                                 {"native_old": len(widths), "current": len(widths), "reference": 0})
                self.assertEqual([enabled for enabled, shape in calls], [False, True] * len(widths))
                self.assertEqual([shape[0] for enabled, shape in calls],
                                 [width for width in widths for _ in range(2)])
                self.assertGreater(evidence["padding_fraction"], 0.)

    def test_reference_batches_preserve_original_objective(self):
        model, results, advantages, args, ack = self.fixture()
        batched = copy.deepcopy(model)
        args.rollout_correction, args.kl_beta = "none", .01
        serial, _ = self.execute(model, results, advantages, args, ack)
        args.learner_microbatch_size = 8
        actual, calls = self.execute(batched, results, advantages, args, ack)
        self.assertAlmostEqual(actual["loss"], serial["loss"], places=6)
        for left, right in zip(model.parameters(), batched.parameters()):
            torch.testing.assert_close(left, right, atol=2e-7, rtol=2e-6)
        self.assertEqual(len(calls), 8)
        self.assertEqual(actual["learner_batching"]["model_forward_calls"],
                         {"native_old": 0, "current": 4, "reference": 4})

    def test_length_qualified_plan_restores_response_objective_gradients_and_indices(self):
        for correction, beta in (("decoupled-token-tis", 0.), ("none", .01)):
            with self.subTest(correction=correction):
                model, results, advantages, args, ack = self.fixture()
                batched = copy.deepcopy(model)
                args.rollout_correction, args.kl_beta = correction, beta
                serial, _ = self.execute(model, results, advantages, args, ack)
                args.learner_batch_shapes = "7:5,16:3,32:2"
                actual, calls = self.execute(batched, results, advantages, args, ack)
                self.assertAlmostEqual(actual["loss"], serial["loss"], places=6)
                for (name, left), (_, right) in zip(model.named_parameters(), batched.named_parameters()):
                    torch.testing.assert_close(left, right, atol=2e-7, rtol=2e-6, msg=name)
                    if left.requires_grad:
                        torch.testing.assert_close(left.grad, right.grad, atol=2e-7, rtol=2e-6, msg=name)
                self.assertEqual([shape[0] for enabled, shape in calls if enabled],
                                 actual["learner_batching"]["physical_batch_sizes"])

    def test_uncovered_actual_length_rejected_before_optimizer_or_model_work(self):
        model, results, advantages, args, ack = self.fixture()
        args.learner_batch_shapes = "32:2"
        policy = types.SimpleNamespace(begin_step=lambda: None, normalize=lambda: None)
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.03)
        with patch.object(optimizer, "step") as step, \
                patch("dsv41_train.rl.distributed_grpo.response_log_probs_batch") as forward:
            with self.assertRaisesRegex(ValueError, "cover every actual response"):
                train_group(model, optimizer, policy, results, advantages, args, 1, expected_ack=ack)
            step.assert_not_called()
            forward.assert_not_called()

    def test_invalid_group_metadata_and_oom_fail_before_any_optimizer_step(self):
        model, results, advantages, args, ack = self.fixture()
        policy = types.SimpleNamespace(begin_step=lambda: None, normalize=lambda: None)
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.03)
        with patch.object(optimizer, "step") as step:
            for size in (0, 33, True, 2.5):
                args.learner_microbatch_size = size
                with self.assertRaisesRegex(ValueError, "microbatch size"):
                    train_group(model, optimizer, policy, results, advantages, args, 1, expected_ack=ack)
            args.learner_microbatch_size = 8
            with self.assertRaisesRegex(ValueError, "all 32"):
                train_group(model, optimizer, policy, results[:-1], advantages, args, 1, expected_ack=ack)
            bad = list(advantages); bad[0] = float("nan")
            with self.assertRaisesRegex(ValueError, "finite advantages"):
                train_group(model, optimizer, policy, results, bad, args, 1, expected_ack=ack)
            def wrong_shape(*values, **kwargs):
                vectors, metadata = response_log_probs_batch(*values, **kwargs)
                metadata["input_shape"][0] = 1
                return vectors, metadata
            with patch("dsv41_train.rl.distributed_grpo.response_log_probs_batch", side_effect=wrong_shape):
                with self.assertRaisesRegex(ValueError, "physical aligned"):
                    train_group(model, optimizer, policy, results, advantages, args, 1, expected_ack=ack)
            with patch("dsv41_train.rl.distributed_grpo.response_log_probs_batch",
                       side_effect=torch.OutOfMemoryError("capacity gate required")) as forward:
                with self.assertRaises(torch.OutOfMemoryError):
                    train_group(model, optimizer, policy, results, advantages, args, 1, expected_ack=ack)
                self.assertEqual(forward.call_count, 1)
            step.assert_not_called()


if __name__ == "__main__":
    unittest.main()
