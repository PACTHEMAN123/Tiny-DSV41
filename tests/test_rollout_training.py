import json
import pickle
import subprocess
import sys
import unittest
from dataclasses import replace
from pathlib import Path

from rollout_manager.training import (PolicyRef, TrainingGroup, TrainingTrajectory, TurnTrace,
                                      dumps, from_rollout_group, loads, to_rollout_group, token_digest)


class TrainingFixture:
    def setUp(self):
        self.policy = PolicyRef(3, "a" * 64, "run-one", {"tokenizer_sha256": "b" * 64},
                                {"temperature": 0.7, "top_p": 1, "top_k": 0})
        self.context = [1, 0]

    def sample(self, index, **changes):
        turn = TurnTrace(1, f"request-{index}", self.context, [2, 1], [-0.2, -1.1], self.policy,
                         "eos", 1)
        values = dict(group_id=8, task_id="task-a", initial_context_sha256=token_digest(self.context),
                      reset_sha256="c" * 64, sample_index=index, turns=(turn,), reward=float(index),
                      evidence={"verifier": {"result": "verified"}})
        values.update(changes)
        return TrainingTrajectory(**values)

    def group(self, samples=None, **changes):
        values = dict(group_id=8, task_id="task-a", initial_context_sha256=token_digest(self.context),
                      reset_sha256="c" * 64, policy=self.policy, expected_size=2,
                      samples=samples if samples is not None else (self.sample(0), self.sample(1)))
        values.update(changes)
        return TrainingGroup(**values)


class TrainingContractTests(TrainingFixture, unittest.TestCase):
    def test_shared_contract_imports_without_torch_or_site_packages(self):
        root = str(Path(__file__).resolve().parents[1])
        script = "import sys; from rollout_manager.training import PolicyRef; assert 'torch' not in sys.modules"
        subprocess.run([sys.executable, "-B", "-S", "-c", script], cwd=root, check=True)

    def test_durable_codec_and_cpu_ipc_preserve_complete_group(self):
        group = self.group(samples=(self.sample(1), self.sample(0)))
        self.assertEqual([sample.sample_index for sample in group.samples], [0, 1])
        for record in (self.policy, group.samples[0].turns[0], group.samples[0], group):
            self.assertEqual(loads(dumps(record)), record)
            self.assertEqual(pickle.loads(pickle.dumps(record)), record)
        self.assertEqual(loads(dumps(group)).samples[0].reward, 0.0)
        with self.assertRaises(TypeError):
            self.policy.sampling["temperature"] = 0
        value = json.loads(dumps(group))
        value["record"]["unexpected"] = 1
        with self.assertRaises(ValueError):
            loads(json.dumps(value))
        with self.assertRaises(ValueError):
            loads('{"schema_version":1,"schema_version":1,"kind":"PolicyRef","record":{}}')

    def test_exact_action_alignment_and_full_categorical_policy_are_required(self):
        for changes in ({"action_token_ids": [True]}, {"context_token_ids": []},
                        {"behavior_log_probs": [-0.2]}, {"behavior_log_probs": [-0.2, float("nan")]},
                        {"behavior_log_probs": [-0.2, 0.1]}, {"stop_token_count": 3}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(self.sample(0).turns[0], **changes)
        for sampling in ({"temperature": 0, "top_p": 1, "top_k": 0},
                         {"temperature": 1, "top_p": 0.9, "top_k": 0},
                         {"temperature": 1, "top_p": 1, "top_k": 5}):
            with self.subTest(sampling=sampling), self.assertRaises(ValueError):
                replace(self.policy, sampling=sampling)

    def test_missing_reward_excludes_whole_group_but_verified_zero_is_valid(self):
        self.assertEqual(self.group().samples[0].reward, 0)
        for sibling in (self.sample(1, reward=None),
                        self.sample(1, reward=0, reward_eligible=False, exclusion_reason="cleanup_unknown")):
            self.assertEqual(loads(dumps(sibling)), sibling)
            with self.assertRaisesRegex(ValueError, "G eligible"):
                self.group((self.sample(0), sibling))
        with self.assertRaisesRegex(ValueError, "exactly G"):
            self.group((self.sample(0),))
        with self.assertRaisesRegex(ValueError, ">= 2"):
            self.group((self.sample(0),), expected_size=1)

    def test_resource_waves_and_mixed_policy_cannot_be_training_groups(self):
        sample = self.sample(1)
        different = replace(self.policy, version=4)
        other_context = [0, 1]
        cases = [replace(sample, task_id="another-task"), replace(sample, reset_sha256="d" * 64),
                 replace(sample, turns=(replace(sample.turns[0], policy=different),)),
                 replace(sample, initial_context_sha256=token_digest(other_context),
                         turns=(replace(sample.turns[0], context_token_ids=other_context),)),
                 replace(sample, sample_id=self.sample(0).sample_id),
                 replace(sample, turns=(replace(sample.turns[0], request_id="request-0"),))]
        for sibling in cases:
            with self.subTest(sibling=sibling), self.assertRaises(ValueError):
                self.group((self.sample(0), sibling))
        second = TurnTrace(2, "second-turn", [0, 1, 2], [0], [-0.1], different)
        with self.assertRaisesRegex(ValueError, "one policy"):
            replace(self.sample(0), turns=(*self.sample(0).turns, second))


class TrajectoryLossTests(TrainingFixture, unittest.TestCase):
    def model(self):
        import torch
        from torch import nn
        class CausalModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.transitions = nn.Parameter(torch.tensor([[0., 1., 2.], [2., 0., 1.], [1., 2., 0.]]))
                self.calls = []
            def forward(self, input_ids, attention_mask):
                self.calls.append(input_ids.tolist()[0])
                assert torch.equal(attention_mask, torch.ones_like(input_ids))
                return self.transitions[input_ids]
        return CausalModel()

    def test_single_turn_conversion_preserves_sampler_and_original_loss(self):
        import torch
        from dsv41_train.rl.algorithm import grpo_loss, response_log_probs
        from dsv41_train.rl.data import PromptRecord
        from dsv41_train.rl.rollout import generate_group
        from dsv41_train.rl.trajectory import trajectory_log_probs
        model = self.model()
        original = generate_group(model, PromptRecord(self.context, [2]), 2, 4, 0.7, None, 3, 8,
                                  torch.Generator().manual_seed(42))
        group = from_rollout_group(original, self.policy, "task-a", "c" * 64)
        self.assertEqual(to_rollout_group(loads(dumps(group))), original)
        original_probs = [response_log_probs(model, sample.prompt_ids, sample.response_ids, 0.7)
                          for sample in original.samples]
        adapted = [trajectory_log_probs(model, sample, 0.7) for sample in group.samples]
        for old, new in zip(original_probs, adapted):
            torch.testing.assert_close(old, new)
        old_loss, _ = grpo_loss(original_probs, [s.behavior_log_probs for s in original.samples], [-1, 1])
        new_loss, _ = grpo_loss(adapted, [s.behavior_log_probs for s in group.samples], [-1, 1])
        torch.testing.assert_close(old_loss, new_loss)
        old_grad = torch.autograd.grad(old_loss, model.transitions)[0]
        new_grad = torch.autograd.grad(new_loss, model.transitions)[0]
        torch.testing.assert_close(old_grad, new_grad)
        torch.testing.assert_close(trajectory_log_probs(model, original.samples[0], 0.7), original_probs[0])

    def test_multi_turn_uses_actual_contexts_and_token_weighted_action_loss(self):
        import torch
        from dsv41_train.rl.algorithm import grpo_loss, response_log_probs
        from dsv41_train.rl.trajectory import trajectory_log_probs
        model = self.model()
        first = TurnTrace(1, "first", self.context, [2], [-0.2], self.policy, "tool_calls")
        # The intervening tool observation is context, not a sampled action.
        second = TurnTrace(2, "second", [1, 0, 2, 0, 1], [0, 2, 1], [-0.4, -0.6, -0.8], self.policy)
        sample = self.sample(0, turns=(first, second))
        current = trajectory_log_probs(model, sample, 0.7)
        self.assertEqual(model.calls, [[1, 0], [1, 0, 2, 0, 1, 0, 2]])
        self.assertEqual(current.numel(), 4)
        expected = torch.cat([response_log_probs(model, first.context_token_ids, first.action_token_ids, 0.7),
                              response_log_probs(model, second.context_token_ids, second.action_token_ids, 0.7)])
        torch.testing.assert_close(current, expected)
        loss, _ = grpo_loss([current], [sample.behavior_log_probs], [1], clip_eps=0.99)
        ratio = (current - torch.tensor(sample.behavior_log_probs)).exp()
        torch.testing.assert_close(loss, -torch.minimum(ratio, ratio.clamp(0.01, 1.99)).mean())
        self.assertTrue(torch.autograd.grad(loss, model.transitions)[0].isfinite().all())
        with self.assertRaisesRegex(ValueError, "per-turn contexts"):
            _ = sample.response_ids
        with self.assertRaisesRegex(ValueError, "per-turn contexts"):
            to_rollout_group(self.group((sample, self.sample(1))))
        with self.assertRaisesRegex(ValueError, "temperature"):
            trajectory_log_probs(model, sample, 1.0)


if __name__ == "__main__":
    unittest.main()
