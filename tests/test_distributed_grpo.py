import types
import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from dsv41_train.rl.algorithm import response_log_probs
from dsv41_train.rl.distributed_grpo import (rank_zero_call, run, semantic_configuration,
                                           train_group, validate_results)
from dsv41_train.rl.native_lora import (adapter_snapshot, gradient_norm,
    default_native_source, import_native, PINNED_NATIVE_MANIFEST_SHA256,
    reference_base, scoring_model_class, verify_optimizer_step)


class NativeRuntimeTests(unittest.TestCase):
    def test_default_bundle_is_pinned_and_imports_privately(self):
        native, provenance = import_native()
        self.assertEqual(provenance["python_files"], 30)
        self.assertEqual(provenance["manifest_sha256"], PINNED_NATIVE_MANIFEST_SHA256)
        self.assertEqual(provenance["source_kind"], "bundled_pinned_native_runtime")
        self.assertTrue(Path(native("models.dsv4.materialize").__file__).is_relative_to(default_native_source()))

    def test_altered_default_manifest_is_rejected_before_import(self):
        original = (default_native_source() / "source-manifest.json").read_text()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            changed = json.loads(original)
            changed["files"] = {}
            (root / "source-manifest.json").write_text(json.dumps(changed))
            with patch("dsv41_train.rl.native_lora.default_native_source", return_value=root):
                with self.assertRaisesRegex(ValueError, "pinned native runtime manifest digest"):
                    import_native()

    def test_native_six_layer_checkpoint_scores_and_adapter_gradients_match(self):
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
            def score():
                model.zero_grad(set_to_none=True)
                values = response_log_probs(model, [0, 1, 3, 4], [7, 2], .8)
                (-values.mean()).backward()
                self.assertTrue(all(p.grad is None for p in model.parameters() if not p.requires_grad))
                gradients = [p.grad.clone() for p in lora.adapter_parameters(model)]
                self.assertEqual(len(gradients), 12)
                self.assertTrue(all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients))
                return values.detach(), gradients
            expected, expected_gradients = score()
            model.gradient_checkpointing_enable()
            actual, actual_gradients = score()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for actual, expected in zip(actual_gradients, expected_gradients):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        finally:
            if model is not None:
                model.close()
            torch.set_num_threads(threads)


class Adapter(nn.Linear):
    def __init__(self):
        super().__init__(8, 8, bias=False)
        self.weight.requires_grad_(False)
        self.lora_a = nn.Parameter(torch.randn(2, 8) * .1)
        self.lora_b = nn.Parameter(torch.zeros(8, 2))
        self.adapters_enabled = True

    def forward(self, value):
        return nn.functional.linear(value, self.weight) + (
            nn.functional.linear(nn.functional.linear(value, self.lora_a), self.lora_b)
            if self.adapters_enabled else 0)


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(11, 8).requires_grad_(False)
        self.adapter = Adapter()

    def forward(self, ids, mask):
        self.hidden = self.adapter(self.embedding(ids))
        if self.hidden.requires_grad:
            self.hidden.retain_grad()
        return self.hidden, ()


class NativeBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(vocab_size=11)
        self.context_parallel = types.SimpleNamespace(size=1)
        self.model = Backbone()
        self.lm_head = nn.Linear(8, 11, bias=False).requires_grad_(False)


class DistributedGRPOTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.model = scoring_model_class(NativeBase)()
        self.optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad],
                                          lr=.03, weight_decay=0, foreach=False)

    def test_response_alignment_and_prompt_hidden_mask(self):
        prompt, response = [0, 2, 3], [4, 5]
        actual = response_log_probs(self.model, prompt, response, .8)
        hidden = self.model.model.hidden
        expected = self.model.lm_head(hidden).float().div(.8).log_softmax(-1)[0, 2:]
        expected = expected.gather(-1, torch.tensor(response)[:, None]).squeeze(-1)
        torch.testing.assert_close(actual, expected)
        actual.sum().backward()
        self.assertEqual(float(hidden.grad[0, :2].abs().sum()), 0)
        self.assertGreater(float(hidden.grad[0, 2:].abs().sum()), 0)

    def test_actual_cpu_optimizer_and_reference_bypass(self):
        with torch.no_grad():
            old = response_log_probs(self.model, [0, 2], [3, 4], .8).tolist()
        result = {"context_token_ids": [0, 2], "action_token_ids": [3, 4],
                  "behavior_log_probs": old}
        policy = types.SimpleNamespace(begin_step=lambda: None, normalize=lambda: None)
        args = types.SimpleNamespace(temperature=.8, kl_beta=.01, clip_eps=.2)
        # Two opposite advantages on distinct actions produce genuine gradients.
        other = {**result, "action_token_ids": [5, 6]}
        with torch.no_grad():
            other["behavior_log_probs"] = response_log_probs(self.model, [0, 2], [5, 6], .8).tolist()
        metrics = train_group(self.model, self.optimizer, policy, [result, other] * 16,
                              torch.tensor([1., -1.] * 16), args, 1)
        self.assertGreater(metrics["gradient_norm"], 0)
        self.assertGreater(metrics["changed_adapter_elements"], 0)
        self.assertTrue(all(p.grad is None for p in self.model.parameters() if not p.requires_grad))
        with reference_base(self.model):
            self.assertFalse(self.model.model.adapter.adapters_enabled)
        self.assertTrue(self.model.model.adapter.adapters_enabled)

    def test_rejects_missing_gradient_and_no_parameter_change(self):
        with self.assertRaisesRegex(ValueError, "no gradient"):
            gradient_norm(self.optimizer)
        before = adapter_snapshot(self.optimizer)
        with self.assertRaisesRegex(ValueError, "changed no"):
            verify_optimizer_step(self.model, self.optimizer, before)

    def test_generation_failure_or_wrong_policy_cannot_enter_reward_path(self):
        record = types.SimpleNamespace(prompt_ids=[0, 2])
        with self.assertRaisesRegex(ValueError, "complete"):
            validate_results([], record, {"version": 0}, 32)
        result = {"context_token_ids": [0, 2], "action_token_ids": [3],
            "behavior_log_probs": [-1.], "logprob_convention": "sampled_distribution",
            "generation_metrics": {"policy_ack": {"version": 1}}, "finish_reason": "eos",
            "stop_token_count": 1}
        with self.assertRaisesRegex(ValueError, "identity"):
            validate_results([result], record, {"version": 0}, 1)

    def test_local_paths_do_not_change_distributed_semantic_identity(self):
        a = types.SimpleNamespace(native_source="/a/native", output_dir="/a/run", seed=42, kl_beta=0.)
        b = types.SimpleNamespace(native_source="/b/native", output_dir="/b/run", seed=42, kl_beta=0.)
        self.assertEqual(semantic_configuration(a), semantic_configuration(b))
        b.kl_beta = .01
        self.assertNotEqual(semantic_configuration(a), semantic_configuration(b))

    def test_unsupported_context_or_evaluation_seed_fails_before_model_loading(self):
        args = types.SimpleNamespace(learner_batch_shapes=None, rollout_correction="none",
            updates=3, max_groups=64, group_size=32, eval_questions=1319,
            max_prompt_tokens=256, max_new_tokens=256, max_model_len=2048,
            max_initial_logprob_delta=.1, learning_rate=1e-4, kl_beta=0.,
            temperature=.8, eval_temperature=.01, clip_eps=.2, num_threads=2,
            learner_microbatch_size=4, seed=42)
        for name, value in (("max_model_len", 2049), ("seed", 2**63 - 2 - 64 * 32)):
            invalid = types.SimpleNamespace(**vars(args)); setattr(invalid, name, value)
            with self.subTest(name=name), patch("dsv41_train.rl.distributed_grpo.ModelTokenizer") as load:
                with self.assertRaisesRegex(ValueError, "invalid GRPO"):
                    run(invalid)
                load.assert_not_called()

    def test_rank_zero_io_error_is_a_broadcast_error_not_a_reward(self):
        with patch("dsv41_train.rl.distributed_grpo.dist.get_rank", return_value=0), \
                patch("dsv41_train.rl.distributed_grpo.dist.broadcast_object_list") as broadcast:
            with self.assertRaisesRegex(RuntimeError, "actual actor failure"):
                rank_zero_call(lambda: (_ for _ in ()).throw(ValueError("actual actor failure")))
            self.assertEqual(broadcast.call_args.args[0][0]["error_type"], "ValueError")


if __name__ == "__main__":
    unittest.main()
