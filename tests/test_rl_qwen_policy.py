import json
import tempfile
import unittest
from pathlib import Path

import torch

from dsv41_train.models.qwen import Qwen3MoeForCausalLM
from dsv41_train.rl.algorithm import grpo_loss, response_log_probs
from dsv41_train.rl.checkpoint import load_policy_artifact, policy_state, publish_policy_artifact
from dsv41_train.rl.config import RLConfig
from dsv41_train.rl.data import PromptRecord
from dsv41_train.rl.family import build_policy, load_model_config
from dsv41_train.rl.qwen_policy import QwenPolicy
from dsv41_train.rl.rollout import generate_group
from dsv41_train.rl.runtime import train_groups
from qwen_helpers import tiny_config, write_hf_checkpoint


class ShapeSensitivePolicy(torch.nn.Module):
    """Expose sequence-shape drift deterministically without requiring a GPU."""
    response_log_probs = QwenPolicy.response_log_probs

    def __init__(self):
        super().__init__()
        self.transitions = torch.nn.Parameter(torch.tensor([[0., 1., 2.], [2., 0., 1.], [1., 2., 0.]]))

    def forward(self, input_ids, attention_mask):
        if not torch.equal(attention_mask, torch.ones_like(input_ids)):
            raise AssertionError("sampling masks must match")
        shape_offset = input_ids.shape[1] * self.transitions.new_tensor([0., .4, -.6])
        return self.transitions[input_ids] + shape_offset


class QwenRLPolicyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(19)
        self.source = Qwen3MoeForCausalLM(tiny_config())
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        write_hf_checkpoint(self.root, self.source)

    def policy(self, scope="lm_head"):
        return build_policy(self.root, load_model_config(self.root), random_init=False,
                            devices=("cpu",), dtype=torch.float32, trainable_scope=scope)

    def test_pretrained_native_logits_gradients_and_sampling_parity(self):
        policy = self.policy()
        tokens = torch.tensor([[1, 4, 0, 9]])
        mask = torch.ones_like(tokens)
        actual, expected = policy(tokens, mask), self.source(tokens, mask).logits
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.square().mean().backward()
        expected.square().mean().backward()
        torch.testing.assert_close(policy.backend.lm_head.weight.grad, self.source.lm_head.weight.grad,
                                   rtol=0, atol=0)
        self.assertTrue(all(parameter.grad is None for name, parameter in policy.named_parameters()
                            if name != "backend.lm_head.weight"))
        group = generate_group(policy, PromptRecord(tokens[0].tolist()), group_size=4,
            max_new_tokens=3, temperature=0.8, eos_token_id=None, policy_version=0, group_id=0,
            generator=torch.Generator().manual_seed(21), reward_fn=lambda record, ids: sum(ids))
        for sample in group.samples:
            probabilities = response_log_probs(policy, sample.prompt_ids, sample.response_ids, 0.8)
            torch.testing.assert_close(probabilities, torch.tensor(sample.behavior_log_probs),
                                       rtol=1e-5, atol=1e-5)

    def test_prefix_replay_avoids_shape_drift_and_initial_false_clipping(self):
        policy = ShapeSensitivePolicy()
        group = generate_group(policy, PromptRecord([2, 0]), group_size=2, max_new_tokens=4,
            temperature=1.3, eos_token_id=None, policy_version=0, group_id=0,
            generator=torch.Generator().manual_seed(27), reward_fn=lambda record, ids: sum(ids))
        current, batched = [], []
        for sample in group.samples:
            replayed = response_log_probs(policy, sample.prompt_ids, sample.response_ids, 1.3)
            torch.testing.assert_close(replayed, torch.tensor(sample.behavior_log_probs), rtol=0, atol=0)
            current.append(replayed)
            tokens = torch.tensor([sample.prompt_ids + sample.response_ids[:-1]])
            logits = policy(tokens, torch.ones_like(tokens))[0, len(sample.prompt_ids) - 1:]
            batched.append((logits / 1.3).log_softmax(-1).gather(
                -1, torch.tensor(sample.response_ids)[:, None])[:, 0])
        behavior = [sample.behavior_log_probs for sample in group.samples]
        _, previous = grpo_loss(batched, behavior, [1., -1.])
        self.assertGreater(previous["clip_fraction"], 0)
        loss, metrics = grpo_loss(current, behavior, [1., -1.])
        self.assertEqual(metrics["importance_ratio"], 1)
        self.assertEqual(metrics["approx_kl"], 0)
        self.assertEqual(metrics["clip_fraction"], 0)
        loss.backward()
        self.assertGreater(float(policy.transitions.grad.abs().sum()), 0)

    def test_pretrained_prefix_probability_gradients_for_both_scopes(self):
        prompt, actions, temperature = [1, 4, 0], [7, 0, 2], .8
        for scope in ("lm_head", "all"):
            with self.subTest(scope=scope):
                policy = self.policy(scope)
                self.source.zero_grad(set_to_none=True)
                actual = response_log_probs(policy, prompt, actions, temperature)
                expected = []
                for index, token in enumerate(actions):
                    tokens = torch.tensor([prompt + actions[:index]])
                    logits = self.source(tokens, torch.ones_like(tokens)).logits[0, -1]
                    expected.append((logits.float() / temperature).log_softmax(-1)[token])
                expected = torch.stack(expected)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                actual.sum().backward()
                expected.sum().backward()
                torch.testing.assert_close(policy.backend.lm_head.weight.grad, self.source.lm_head.weight.grad,
                                           rtol=0, atol=0)
                self.assertGreater(float(policy.backend.lm_head.weight.grad.abs().sum()), 0)
                if scope == "all":
                    torch.testing.assert_close(policy.backend.model.embed_tokens.weight.grad,
                                               self.source.model.embed_tokens.weight.grad, rtol=0, atol=0)
                    self.assertGreater(float(policy.backend.model.embed_tokens.weight.grad.abs().sum()), 0)
                else:
                    self.assertTrue(all(parameter.grad is None for name, parameter in policy.named_parameters()
                                        if name != "backend.lm_head.weight"))

    def test_real_grpo_head_update_and_immutable_artifact_restore(self):
        policy = self.policy()
        frozen = {name: parameter.detach().clone() for name, parameter in policy.named_parameters()
                  if not parameter.requires_grad}
        before = policy.backend.lm_head.weight.detach().clone()
        receipt = policy.frozen_parameter_fingerprint()
        group = generate_group(policy, PromptRecord([1, 4, 7]), group_size=4, max_new_tokens=3,
            temperature=0.8, eos_token_id=None, policy_version=0, group_id=0,
            generator=torch.Generator().manual_seed(21), reward_fn=lambda record, ids: sum(ids))
        optimizer = torch.optim.AdamW([parameter for parameter in policy.parameters()
                                       if parameter.requires_grad], lr=0.01, foreach=False)
        metrics = train_groups(policy, None, optimizer, [group], RLConfig(rollout_engine="native-pytorch", temperature=0.8))
        self.assertGreater(metrics["nonzero_advantages"], 0)
        self.assertFalse(torch.equal(before, policy.backend.lm_head.weight))
        self.assertEqual(receipt, policy.frozen_parameter_fingerprint())
        for name, parameter in policy.named_parameters():
            if name in frozen:
                torch.testing.assert_close(parameter, frozen[name], rtol=0, atol=0)
                self.assertIsNone(parameter.grad)
        state = policy_state(policy)
        self.assertEqual(set(state["trainable"]), {"backend.lm_head.weight"})
        self.assertEqual(state["frozen"], {})
        artifact = publish_policy_artifact(policy, 1, self.root / "policies", lineage="qwen-test",
            identity={"base": "mounted-fixture", "trainable_scope": "lm_head"},
            sampling={"temperature": 0.8, "top_p": 1, "top_k": 0})
        restored = self.policy()
        ack = load_policy_artifact(restored, artifact)
        self.assertEqual(ack, {key: artifact[key] for key in ("version", "digest", "lineage")})
        tokens = torch.tensor([[1, 4, 7, 9]])
        torch.testing.assert_close(restored(tokens), policy(tokens), rtol=0, atol=0)
        self.assertEqual(restored.frozen_parameter_fingerprint(), receipt)
        with self.assertRaisesRegex(ValueError, "configured scope"):
            load_policy_artifact(self.policy("all"), artifact)

    def test_frozen_mutations_and_unsupported_buffers_fail_closed(self):
        policy = self.policy()
        with torch.no_grad():
            policy.backend.model.layers[0].mlp.experts.down_proj.add_(1)
        with self.assertRaisesRegex(RuntimeError, "immutable Qwen base"):
            policy_state(policy)
        policy = self.policy()
        policy.backend.model.norm.weight.data.add_(1)
        with self.assertRaisesRegex(RuntimeError, "immutable Qwen normalization"):
            policy_state(policy)
        policy = self.policy()
        policy.backend.register_buffer("changed_state", torch.tensor(1))
        with self.assertRaisesRegex(RuntimeError, "mutable buffers"):
            policy_state(policy)

    def test_freezing_reference_recaptures_invariants_without_widening_scope(self):
        for scope in ("lm_head", "all"):
            with self.subTest(scope=scope):
                reference = self.policy(scope)
                weights = {name: parameter.detach().clone() for name, parameter in reference.named_parameters()}
                self.assertIs(reference.eval().requires_grad_(False), reference)
                receipt = reference.frozen_parameter_fingerprint()
                self.assertEqual(receipt["trainable_parameter_count"], 0)
                self.assertEqual(receipt["frozen_parameter_count"], sum(value.numel() for value in weights.values()))
                tokens = torch.tensor([[1, 4, 0]])
                self.assertFalse(reference(tokens, torch.ones_like(tokens)).requires_grad)
                probabilities = response_log_probs(reference, [1, 4, 0], [7, 2], .8)
                self.assertFalse(probabilities.requires_grad)
                self.assertEqual(receipt, reference.frozen_parameter_fingerprint())
                for name, parameter in reference.named_parameters():
                    torch.testing.assert_close(parameter, weights[name], rtol=0, atol=0)
        policy = self.policy("lm_head").requires_grad_(True)
        with self.assertRaisesRegex(RuntimeError, "frozen parameter set changed"):
            policy.assert_engram_invariants()
        with self.assertRaisesRegex(RuntimeError, "frozen parameter set changed"):
            policy.requires_grad_(False)

    def test_unsupported_scope_and_rope_configs_fail_before_load(self):
        with self.assertRaisesRegex(ValueError, "trainable_scope"):
            QwenPolicy(self.source, trainable_scope="routers")
        values = self.source.config.to_dict()
        values["rope_scaling"] = {"type": "linear", "factor": 2}
        (self.root / "config.json").write_text(json.dumps(values), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "RoPE scaling"):
            load_model_config(self.root)


if __name__ == "__main__":
    unittest.main()
