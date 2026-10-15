from dataclasses import replace
import json
import tempfile
import unittest
from pathlib import Path

import torch

from dsv41_train.models.qwen import Qwen3MoeForCausalLM
from dsv41_train.rl.algorithm import response_log_probs
from dsv41_train.rl.config import RLConfig
from dsv41_train.rl.data import PromptRecord
from dsv41_train.rl.family import build_policy, load_model_config
from dsv41_train.rl.qwen_policy import LAST_POSITION_LOGITS_PATH
from dsv41_train.rl.rollout import generate_group
from dsv41_train.rl.runtime import make_model, run_training, train_groups
from rollout_manager.training import PolicyRef, TrainingGroup, TrainingTrajectory, TurnTrace, token_digest
from qwen_helpers import tiny_config, write_hf_checkpoint


class QwenMemoryProfileTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(41)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        write_hf_checkpoint(self.root, Qwen3MoeForCausalLM(tiny_config(max_position_embeddings=128)))
        self.config = RLConfig(rollout_engine="native-pytorch", temperature=.8, trainable_scope="lm_head", qwen_last_position_only=True,
                               learning_rate=.01, max_grad_norm=100., kl_beta=.05)

    def policy(self, *, last=True, dtype=torch.float32):
        model = build_policy(self.root, load_model_config(self.root), random_init=False,
                             devices=("cpu",), dtype=dtype, trainable_scope="lm_head")
        model.qwen_last_position_only = last
        return model

    def sample(self, model, prompt, count, seed):
        return generate_group(model, PromptRecord(prompt), 1, count, .8, None, 0, 0,
                              torch.Generator().manual_seed(seed), reward_fn=lambda record, ids: 0.).samples[0]

    def group(self, model):
        context = [1, 4, 0]
        policy = PolicyRef(0, "a" * 64, "memory-test", {"qwen_logits_path": LAST_POSITION_LOGITS_PATH},
                           {"temperature": .8, "top_p": 1, "top_k": 0})
        trajectories = []
        for index in range(2):
            first = self.sample(model, context, index + 1, 100 + index)
            observed = context + first.response_ids + [7, 11, 4]
            second = self.sample(model, observed, 5 - index, 200 + index)
            turns = tuple(TurnTrace(turn_index, f"sample-{index}-turn-{turn_index}", sample.prompt_ids,
                sample.response_ids, sample.behavior_log_probs, policy, sample.finish_reason, sample.stop_token_count)
                for turn_index, sample in enumerate((first, second), 1))
            trajectories.append(TrainingTrajectory(0, "fixture", token_digest(context), "b" * 64,
                                                   index, turns, float(index)))
        return TrainingGroup(0, "fixture", token_digest(context), "b" * 64, policy, 2, trajectories)

    def test_default_sampler_preserves_full_head_behavior_and_opt_in_replays_exactly(self):
        class ForwardOnly(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, input_ids, attention_mask):
                return self.model(input_ids, attention_mask)
        default = self.policy(last=False)
        self.assertEqual(self.sample(default, [1, 4, 0], 6, 51),
                         self.sample(ForwardOnly(default), [1, 4, 0], 6, 51))
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                model = self.policy(dtype=dtype)
                sample = self.sample(model, [1, 4, 0], 6, 52)
                actual = response_log_probs(model, sample.prompt_ids, sample.response_ids, .8)
                torch.testing.assert_close(actual, torch.tensor(sample.behavior_log_probs), rtol=0, atol=0)
                actual.sum().backward()
                self.assertGreater(float(model.backend.lm_head.weight.grad.abs().sum()), 0)
                self.assertEqual(model(torch.tensor([[1, 4, 0]])).shape, (1, 3, 32))

    def test_last_row_head_saves_only_its_own_storage(self):
        model = self.policy()
        saved = []
        def pack(tensor):
            saved.append((tuple(tensor.shape), tensor.untyped_storage().nbytes()))
            return tensor
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            values = response_log_probs(model, [1] * 32, [3] * 8, .8)
        inputs = [size for shape, size in saved if shape == (1, 16)]
        self.assertEqual(inputs, [16 * 4] * 8)
        self.assertFalse(any(len(shape) == 2 and shape[0] > 1 and shape[1] == 16 for shape, _ in saved))
        values.sum().backward()
        self.assertGreater(float(model.backend.lm_head.weight.grad.abs().sum()), 0)

    def test_multiturn_chunked_loss_metrics_and_gradients_preserve_token_weights(self):
        base = self.policy()
        group = self.group(base)
        reference = self.policy().eval().requires_grad_(False)
        with torch.no_grad():
            base.backend.lm_head.weight.add_(torch.randn_like(base.backend.lm_head.weight) * .08)
        expected_gradient, expected_metrics = None, None
        for width in (0, 1, 2, 3, 32):
            with self.subTest(width=width):
                model = self.policy()
                model.load_trainable_state_dict(base.trainable_state_dict())
                optimizer = torch.optim.SGD([model.backend.lm_head.weight], lr=0.)
                metrics = train_groups(model, reference, optimizer, [group],
                                       replace(self.config, backprop_chunk_tokens=width))
                gradient = model.backend.lm_head.weight.grad
                self.assertGreater(float(gradient.abs().sum()), 0)
                if expected_gradient is None:
                    expected_gradient, expected_metrics = gradient.clone(), metrics
                    self.assertGreater(metrics["clip_fraction"], 0)
                    self.assertGreater(metrics["kl"], 0)
                else:
                    torch.testing.assert_close(gradient, expected_gradient, rtol=2e-6, atol=2e-7)
                    for key in metrics:
                        self.assertAlmostEqual(metrics[key], expected_metrics[key], delta=2e-6)
        # The two unequal turn lengths must retain whole-trajectory token weighting.
        self.assertEqual([len(turn.action_token_ids) for turn in group.samples[0].turns], [1, 5])

    def test_profile_gate_rejects_untagged_receipts_and_invalid_configuration(self):
        model = self.policy()
        group = self.group(model)
        old = replace(group.policy, identities={})
        samples = tuple(replace(sample, turns=tuple(replace(turn, policy=old) for turn in sample.turns))
                        for sample in group.samples)
        stale = replace(group, policy=old, samples=samples)
        optimizer = torch.optim.SGD([model.backend.lm_head.weight], lr=0.)
        with self.assertRaisesRegex(ValueError, "receipt.*logits profile"):
            train_groups(model, None, optimizer, [stale], replace(self.config, kl_beta=0))
        configured = make_model(self.root, load_model_config(self.root), replace(self.config, learning_rate=.01),
                                ("cpu",), False)
        self.assertTrue(configured.qwen_last_position_only)
        for settings in ({"qwen_last_position_only": 1}, {"backprop_chunk_tokens": -1}, {"backprop_chunk_tokens": True}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                RLConfig.from_dict(settings)

    def test_native_manager_activates_the_same_opt_in_sampler_profile(self):
        data = self.root / "native.jsonl"
        target = self.sample(self.policy(), [1, 4, 0], 3, self.config.seed + 1).response_ids
        data.write_text(json.dumps({"prompt_ids": [1, 4, 0], "target_ids": target}) + "\n")
        output = self.root / "native-training"
        config = replace(self.config, max_steps=2, max_new_tokens=3, group_size=2,
                         kl_beta=0., backprop_chunk_tokens=2, output_dir=str(output))
        run_training(model_dir=self.root, data_path=data, config=config)
        rows = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
        self.assertAlmostEqual(rows[0]["importance_ratio"], 1., places=12)
        self.assertEqual(rows[0]["clip_fraction"], 0.)
        self.assertGreater(rows[0]["nonzero_advantages"], 0)
        for version in (1, 2):
            state = torch.load(output / f"step-{version:08d}.pt", weights_only=True)
            self.assertEqual(state["identity"]["qwen_logits_path"], LAST_POSITION_LOGITS_PATH)
            self.assertTrue(state["config"]["qwen_last_position_only"])
            self.assertEqual(state["config"]["backprop_chunk_tokens"], 2)


if __name__ == "__main__":
    unittest.main()
