"""Tiny native learner tests; actor behavior numbers are explicit contract fixtures."""
from dataclasses import replace
import math
import unittest

import torch
from dsv41_train.models.qwen import Qwen3MoeForCausalLM
from dsv41_train.rl.algorithm import response_log_probs
from dsv41_train.rl.config import RLConfig
from dsv41_train.rl.qwen_policy import QwenPolicy
from dsv41_train.rl.runtime import train_groups
from rollout_manager.backends.vllm_profile import vllm_policy_identities
from rollout_manager.training import PolicyRef, TrainingGroup, TrainingTrajectory, TurnTrace, token_digest
from qwen_helpers import tiny_config


class VLLMLearnerContractTests(unittest.TestCase):
    def fixture(self, rewards):
        torch.set_num_threads(1)
        torch.manual_seed(471)
        model = QwenPolicy(Qwen3MoeForCausalLM(tiny_config()).to(dtype=torch.bfloat16), trainable_scope="lm_head")
        model.qwen_last_position_only = model.qwen_kv_cache = True
        model.behavior_engine = "vllm"
        config = RLConfig(dtype="bfloat16", trainable_scope="lm_head", qwen_last_position_only=True,
            qwen_kv_cache=True, backprop_chunk_tokens=0, temperature=.8, learning_rate=.01,
            kl_beta=0, weight_decay=0)
        policy = PolicyRef(0, "a" * 64, "fixture", {
            "qwen_logits_path": "last_position_head_v1", **vllm_policy_identities(model.config.max_position_embeddings)},
            {"temperature": .8, "top_p": 1, "top_k": 0})
        prompt, actions = [1, 2], [3, 4]
        actual = response_log_probs(model, prompt, actions, .8).detach().tolist()
        # Explicit different behavior denominator: this is no live vLLM claim.
        behavior = [value - .1 for value in actual]
        context = token_digest(prompt)
        samples = [TrainingTrajectory(0, "fixture", context, "b" * 64, index,
            (TurnTrace(1, "%032x" % (index + 1), prompt, actions, behavior, policy),), reward)
            for index, reward in enumerate(rewards)]
        group = TrainingGroup(0, "fixture", context, "b" * 64, policy, len(samples), samples)
        optimizer = torch.optim.AdamW([model.backend.lm_head.weight], lr=.01, weight_decay=0, foreach=False)
        return model, optimizer, config, group

    def test_actual_behavior_denominator_is_not_replaced_by_learner_replay(self):
        model, optimizer, config, group = self.fixture([0., 1.] * 4)
        metrics = train_groups(model, None, optimizer, [group], config)
        self.assertAlmostEqual(metrics["importance_ratio"], math.exp(.1), places=5)
        self.assertEqual(metrics["nonzero_advantages"], 8)
        self.assertEqual(float(optimizer.state[model.backend.lm_head.weight]["step"]), 1.)

    def test_native_provider_does_not_silently_accept_vllm_receipts(self):
        model, optimizer, config, group = self.fixture([0.] * 8)
        model.behavior_engine = "native-pytorch"
        with self.assertRaisesRegex(ValueError, "selected sampling profile"):
            train_groups(model, None, optimizer, [group], config)

    def test_zero_variance_vllm_group_retains_zero_weight_update(self):
        model, optimizer, config, group = self.fixture([0.] * 8)
        before = model.backend.lm_head.weight.detach().clone()
        metrics = train_groups(model, None, optimizer, [group], config)
        self.assertEqual(metrics["grad_norm"], 0.)
        self.assertEqual(metrics["nonzero_advantages"], 0)
        self.assertTrue(torch.equal(before, model.backend.lm_head.weight))
        self.assertEqual(float(optimizer.state[model.backend.lm_head.weight]["step"]), 1.)


if __name__ == "__main__":
    unittest.main()
