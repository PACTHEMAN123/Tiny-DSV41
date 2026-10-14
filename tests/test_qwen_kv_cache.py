import unittest
import json
from pathlib import Path
import tempfile

import torch

from dsv41_train.models.qwen import Qwen3MoeForCausalLM
from dsv41_train.rl.algorithm import response_log_probs
from dsv41_train.rl.config import RLConfig
from dsv41_train.rl.data import PromptRecord
from dsv41_train.rl.qwen_policy import QwenPolicy
from dsv41_train.rl.rollout import generate_group
from dsv41_train.rl.runtime import run_training
from rollout_manager.backends.qwen_kv_cache import (
    QwenCachedSampler, QwenRequestKVCache, QWEN_KV_SAMPLING_PATH,
    generate_cached_qwen_group, qualify_qwen_cache)
from qwen_helpers import tiny_config, write_hf_checkpoint


class QwenKVCacheTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(47)

    def policy(self, *, sparse=True, dtype=torch.float32):
        backend = Qwen3MoeForCausalLM(tiny_config(mlp_only_layers=[] if sparse else [0, 1],
                                                 max_position_embeddings=64)).to(dtype=dtype)
        policy = QwenPolicy(backend, trainable_scope="lm_head").eval()
        policy.qwen_last_position_only = True
        policy.qwen_kv_cache = True
        return policy

    def test_dense_and_moe_prefill_then_decode_logits_and_probabilities(self):
        for sparse in (False, True):
            for dtype in (torch.float32, torch.bfloat16):
                with self.subTest(sparse=sparse, dtype=dtype), torch.no_grad():
                    policy = self.policy(sparse=sparse, dtype=dtype)
                    cached = QwenCachedSampler(policy)
                    prefix = [1, 0, 4, 9, 3]
                    lengths = []
                    cached_lengths = []
                    handle = policy.backend.model.layers[0].mlp.register_forward_pre_hook(
                        lambda module, inputs: lengths.append(inputs[0].shape[1]))
                    try:
                        for token in [7, 2, 5, 0, 11, 6, 3]:
                            tokens = torch.tensor([prefix])
                            actual = cached.next_token_logits(tokens, torch.ones_like(tokens))
                            cached_lengths.append(lengths.pop())
                            expected = policy.next_token_logits(tokens, torch.ones_like(tokens))
                            lengths.pop()
                            tolerance = 2e-6 if dtype == torch.float32 else .004
                            torch.testing.assert_close(actual.float(), expected.float(), rtol=0, atol=tolerance)
                            torch.testing.assert_close((actual.float() / .8).log_softmax(-1),
                                                       (expected.float() / .8).log_softmax(-1),
                                                       rtol=0, atol=tolerance * 2)
                            prefix.append(token)
                    finally:
                        handle.remove()
                    self.assertEqual(cached.cache._keys_values[0][0].shape[2], len(prefix) - 1)
                    self.assertEqual(cached_lengths, [5] + [1] * 6)

    def test_cached_sampling_replay_matches_and_backpropagates_only_head(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                policy = self.policy(dtype=dtype)
                lengths = []
                handle = policy.backend.model.layers[0].mlp.register_forward_pre_hook(
                    lambda module, inputs: lengths.append(inputs[0].shape[1]))
                sample = generate_group(QwenCachedSampler(policy), PromptRecord([1, 0, 4, 9, 3]),
                    1, 8, .8, None, 0, 0, torch.Generator().manual_seed(53),
                    reward_fn=lambda record, ids: 0.).samples[0]
                handle.remove()
                self.assertEqual(lengths, [5] + [1] * 7)
                self.assertFalse(policy.training)
                probabilities = response_log_probs(policy, sample.prompt_ids, sample.response_ids, .8)
                torch.testing.assert_close(probabilities, torch.tensor(sample.behavior_log_probs), rtol=0, atol=0)
                (-probabilities.mean()).backward()
                self.assertGreater(float(policy.backend.lm_head.weight.grad.abs().sum()), 0)
                self.assertTrue(all(parameter.grad is None for name, parameter in policy.named_parameters()
                                    if name != "backend.lm_head.weight"))

    def test_cache_isolation_input_guards_and_weight_boundary(self):
        policy = self.policy()
        cache = QwenRequestKVCache(policy)
        prefix = torch.tensor([[1, 0, 4, 9]])
        expected = cache.next_hidden(prefix)
        cache.next_hidden(torch.tensor([[1, 0, 4, 9, 3]]))
        torch.testing.assert_close(QwenRequestKVCache(policy).next_hidden(prefix), expected, rtol=0, atol=0)
        for invalid in (prefix, torch.tensor([[1, 0, 4, 2, 3, 6]])):
            with self.assertRaisesRegex(ValueError, "unchanged-prefix"):
                cache.next_hidden(invalid)
        with self.assertRaisesRegex(ValueError, "padding"):
            QwenRequestKVCache(policy).next_hidden(prefix, torch.tensor([[1, 0, 1, 1]]))
        with torch.no_grad():
            policy.backend.lm_head.weight.add_(.01)
        with self.assertRaisesRegex(RuntimeError, "policy weight update"):
            cache.next_hidden(torch.tensor([[1, 0, 4, 9, 3, 6]]))
        QwenRequestKVCache(policy).next_hidden(prefix)

    def test_independent_group_samples_preserve_shared_rng_progression(self):
        policy = self.policy()
        record = PromptRecord([1, 0, 4])
        kwargs = dict(max_new_tokens=5, temperature=.8, eos_token_id=None, policy_version=0,
                      group_id=0, reward_fn=lambda record, ids: float(sum(ids)))
        generator = torch.Generator().manual_seed(54)
        group = generate_cached_qwen_group(policy, record, 3, generator=generator, **kwargs)
        serial_generator = torch.Generator().manual_seed(54)
        expected = [generate_group(QwenCachedSampler(policy), record, 1,
                                   generator=serial_generator, **kwargs).samples[0] for _ in range(3)]
        self.assertEqual(group.samples, expected)
        self.assertTrue(torch.equal(generator.get_state(), serial_generator.get_state()))
        automatic = generate_group(policy, record, 3, generator=torch.Generator().manual_seed(54), **kwargs)
        self.assertEqual(group, automatic)

    def test_diagnostic_distinguishes_actual_replay_from_legacy_rounding(self):
        result = qualify_qwen_cache(self.policy(dtype=torch.bfloat16), [1, 0, 4, 9], decode_tokens=8)
        self.assertTrue(result["cached_replay_qualified"])
        self.assertEqual(result["cached_sampling_vs_cached_replay_max_logprob_error"], 0.)
        self.assertFalse(result["benchmark_task"])
        self.assertIsNone(result["benchmark_reward"])
        self.assertEqual(result["decoder_timings"]["decoder_calls"], 8)
        self.assertEqual(result["decoder_timings"]["decode_tokens"], 7)

    def test_configuration_profile_is_explicit_and_preserves_defaults(self):
        self.assertFalse(RLConfig().qwen_kv_cache)
        settings = dict(qwen_kv_cache=True, trainable_scope="lm_head", qwen_last_position_only=True)
        RLConfig.from_dict(settings)
        for changes in (dict(qwen_kv_cache=1), dict(trainable_scope="all"),
                        dict(qwen_last_position_only=False), dict(backprop_chunk_tokens=16)):
            with self.assertRaises(ValueError):
                RLConfig.from_dict(settings | changes)

    def test_native_manager_trains_and_publishes_the_bound_cached_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = self.policy()
            write_hf_checkpoint(root, policy.backend)
            record = PromptRecord([1, 0, 4, 9])
            config = RLConfig(rollout_engine="native-pytorch", max_steps=1, max_new_tokens=5, group_size=2, queue_capacity=1,
                trainable_scope="lm_head", qwen_last_position_only=True, qwen_kv_cache=True,
                temperature=.8, learning_rate=.01, output_dir=str(root / "training"))
            target = generate_group(policy, record, 1, config.max_new_tokens, config.temperature,
                None, 0, 0, torch.Generator().manual_seed(config.seed + 1),
                reward_fn=lambda record, ids: 0.).samples[0].response_ids
            data = root / "data.jsonl"
            data.write_text(json.dumps({"prompt_ids": record.prompt_ids, "target_ids": target}) + "\n")
            run_training(model_dir=root, data_path=data, config=config)
            state = torch.load(root / "training/step-00000001.pt", weights_only=True)
            self.assertEqual(state["identity"]["qwen_sampling_path"], QWEN_KV_SAMPLING_PATH)
            self.assertTrue(state["config"]["qwen_kv_cache"])
            metrics = json.loads((root / "training/metrics.jsonl").read_text().splitlines()[0])
            self.assertAlmostEqual(metrics["importance_ratio"], 1., places=7)
            self.assertEqual(metrics["clip_fraction"], 0.)
            self.assertGreater(metrics["nonzero_advantages"], 0)
            self.assertGreater(metrics["grad_norm"], 0.)
            stages = [json.loads(row) for row in (root / "training/learner-stage-timings.jsonl")
                      .read_text().splitlines()]
            self.assertEqual([row["stage"] for row in stages], ["model_startup", "initial_policy_publish",
                "manager_startup", "wait_eligible_groups", "actor_pause_before_training", "train_groups",
                "actor_pause_after_group_commit", "policy_publish", "checkpoint_write", "actor_activation",
                "actor_resume", "manager_close"])
            self.assertTrue(all(row["status"] == "completed" and row["duration_seconds"] >= 0 for row in stages))
            self.assertEqual(stages[5]["completion"], "gradient_norm_cpu_read_after_optimizer_step")


if __name__ == "__main__":
    unittest.main()
