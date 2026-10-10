import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from dsv41_train.rl.algorithm import response_log_probs
from dsv41_train.rl.data import PromptRecord, build_reward, load_jsonl
from dsv41_train.rl.rollout import RolloutGroup, Sample, generate_group


class CausalModel(nn.Module):
    def __init__(self, always_eos=False):
        super().__init__()
        values = [[0.0, 1.0, 2.0], [2.0, 0.0, 1.0], [1.0, 2.0, 0.0]]
        if always_eos:
            values = [[-100.0, -100.0, 100.0]] * 3
        self.transitions = nn.Parameter(torch.tensor(values))

    def forward(self, input_ids, attention_mask):
        if not torch.equal(attention_mask, torch.ones_like(input_ids)):
            raise AssertionError("mask must include token zero")
        return self.transitions[input_ids]


class RolloutTests(unittest.TestCase):
    def generate(self, model=None, **kwargs):
        defaults = dict(
            model=model if model is not None else CausalModel(),
            record=PromptRecord([1, 0], [2]),
            group_size=2,
            max_new_tokens=4,
            temperature=0.7,
            eos_token_id=None,
            policy_version=3,
            group_id=8,
            generator=torch.Generator().manual_seed(42),
        )
        defaults.update(kwargs)
        return generate_group(**defaults)

    def test_recorded_behavior_probabilities_match_teacher_forcing(self):
        model = CausalModel()
        group = self.generate(model)
        self.assertTrue(model.training)
        self.assertEqual(group.group_id, 8)
        self.assertEqual(group.policy_version, 3)
        for sample in group.samples:
            self.assertEqual(len(sample.response_ids), 4)
            expected = response_log_probs(model, sample.prompt_ids, sample.response_ids, temperature=0.7)
            torch.testing.assert_close(torch.tensor(sample.behavior_log_probs), expected)

    def test_eos_is_included_and_stops_generation(self):
        group = self.generate(CausalModel(always_eos=True), eos_token_id=2)
        for sample in group.samples:
            self.assertEqual(sample.response_ids, [2])
            self.assertEqual(len(sample.behavior_log_probs), 1)
            self.assertEqual(sample.reward, 1.0)

    def test_multiple_eos_and_custom_stop_preserve_training_tokens(self):
        group = self.generate(CausalModel(always_eos=True), eos_token_id=[0, 2])
        self.assertEqual(group.samples[0].finish_reason, "eos")
        self.assertEqual(group.samples[0].content_ids, [])
        model = CausalModel()
        with torch.no_grad():
            model.transitions.copy_(torch.tensor([[-100., 100., -100.],
                                                 [-100., -100., 100.], [100., -100., -100.]]))
        sample = self.generate(model, record=PromptRecord([0], [1, 2]),
                               stop_token_sequences=((1, 2),)).samples[0]
        self.assertEqual(sample.response_ids, [1, 2])
        self.assertEqual(sample.finish_reason, "stop")
        self.assertEqual(sample.stop_token_count, 2)
        self.assertEqual(sample.reward, 1.0)
        torch.testing.assert_close(torch.tensor(sample.behavior_log_probs),
                                   response_log_probs(model, [0], [1, 2], 0.7))

    def test_truncation_reward_and_content_scoring_are_explicit(self):
        reward = lambda record, ids: float(ids == [])
        reward.strip_stop_tokens = True
        stopped = self.generate(CausalModel(always_eos=True), eos_token_id=2,
                                reward_fn=reward).samples[0]
        self.assertEqual(stopped.reward, 1.0)
        truncated = self.generate(CausalModel(always_eos=True), max_new_tokens=2,
                                  truncated_reward=-1.0,
                                  reward_fn=lambda record, ids: 0.5).samples[0]
        self.assertEqual(truncated.finish_reason, "length")
        self.assertEqual(truncated.response_ids, [2, 2])
        self.assertEqual(truncated.raw_reward, 0.5)
        self.assertEqual(truncated.reward, -1.0)

    def test_rng_state_replays_the_same_group(self):
        generator = torch.Generator().manual_seed(58)
        state = generator.get_state().clone()
        first = self.generate(generator=generator)
        generator.set_state(state)
        replay = self.generate(generator=generator)
        self.assertEqual(first, replay)

    def test_sampling_support_includes_every_vocabulary_token(self):
        model = CausalModel()
        with torch.no_grad():
            model.transitions.zero_()
        group = self.generate(model, group_size=30, max_new_tokens=3)
        self.assertEqual({token for sample in group.samples for token in sample.response_ids}, {0, 1, 2})

    def test_reward_failure_restores_training_mode(self):
        model = CausalModel()
        with self.assertRaisesRegex(ValueError, "reward must be finite"):
            self.generate(model, reward_fn=lambda record, response: float("nan"))
        self.assertTrue(model.training)

    def test_version_mixing_and_different_prompts_are_rejected(self):
        first = Sample([0], [1], [-1.0], 1.0, 0)
        second = Sample([0], [2], [-1.0], 0.0, 1)
        with self.assertRaisesRegex(ValueError, "policy_version"):
            RolloutGroup([first, second], 0, 0)
        with self.assertRaisesRegex(ValueError, "same prompt"):
            RolloutGroup([first, Sample([1], [2], [-1.0], 0.0, 0)], 0, 0)

    def test_invalid_inputs_fail_before_sampling(self):
        with self.assertRaisesRegex(ValueError, "max_new_tokens"):
            self.generate(max_new_tokens=0)
        with self.assertRaisesRegex(ValueError, "outside the model vocabulary"):
            self.generate(eos_token_id=3)
        with self.assertRaisesRegex(ValueError, "policy_version"):
            self.generate(policy_version=-1)
        with self.assertRaisesRegex(ValueError, "nonnegative integer"):
            PromptRecord([True])
        with self.assertRaisesRegex(ValueError, "requires one behavior"):
            Sample([0], [1], [], 0.0, 0)


class DataRewardTests(unittest.TestCase):
    def test_text_records_keep_answers_out_of_prompts_and_check_lengths(self):
        class Encoder:
            def encode_prompt(self, prompt=None, messages=None):
                self.input = (prompt, messages)
                return [1, 4, 5]

        encoder = Encoder()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            value = {"messages": [{"role": "user", "content": "What is 2+2?"}],
                     "answer": "4", "task": "math"}
            path.write_text(json.dumps(value) + "\n")
            record = load_jsonl(path, tokenizer=encoder)[0]
            self.assertEqual(encoder.input, (None, value["messages"]))
            self.assertEqual(record.prompt_ids, [1, 4, 5])
            self.assertEqual(record.metadata["answer"], "4")
            self.assertEqual(record.metadata["messages"], value["messages"])
            with self.assertRaisesRegex(ValueError, "tokenizer"):
                load_jsonl(path)
            with self.assertRaisesRegex(ValueError, r"data.jsonl:1:.*length"):
                load_jsonl(path, tokenizer=encoder, max_prompt_tokens=2)
            with path.open("a") as handle:
                handle.write(json.dumps({"prompt_ids": [1]}) + "\n")
            with self.assertWarnsRegex(UserWarning, "skipped"):
                records = load_jsonl(path, tokenizer=encoder, max_prompt_tokens=2,
                                     length_policy="skip")
            self.assertEqual([r.prompt_ids for r in records], [[1]])
            with self.assertRaisesRegex(ValueError, "exactly one"):
                PromptRecord.from_mapping({"prompt_ids": [1], "prompt": "ambiguous"}, tokenizer=encoder)

    def test_token_rewards_count_extra_tokens_and_eos(self):
        record = PromptRecord([0], [1, 2])
        reward = build_reward("token_accuracy")
        self.assertEqual(reward(record, [1, 2]), 1.0)
        self.assertEqual(reward(record, [1]), 0.5)
        self.assertEqual(reward(record, [1, 2, 0]), 2 / 3)
        self.assertEqual(build_reward("exact_match")(record, [1, 2]), 1.0)
        self.assertEqual(build_reward("exact_match")(record, [1]), 0.0)
        self.assertEqual(build_reward("toy_token_score")(PromptRecord([0]), [1, 3]), 2.0)
        with self.assertRaisesRegex(ValueError, "requires target_ids"):
            reward(PromptRecord([0]), [1])

    def test_jsonl_validates_records_with_line_numbers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text('\n' + json.dumps({"prompt_ids": [0, 1], "target_ids": [2], "task": "unit"}) + '\n', encoding="utf-8")
            records = load_jsonl(path)
            self.assertEqual(records, [PromptRecord([0, 1], [2], {"task": "unit"})])
            path.write_text('{"prompt_ids": [0]}\n{"prompt_ids": []}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"data.jsonl:2: prompt_ids must not be empty"):
                load_jsonl(path)

    def test_reward_plugin_receives_metadata_and_checks_finite_values(self):
        plugin = types.ModuleType("test_reward_plugin")
        plugin.score = lambda record, response: record.metadata["scale"] * len(response)
        plugin.bad = lambda record, response: float("inf")
        with patch.dict(sys.modules, {plugin.__name__: plugin}):
            score = build_reward("test_reward_plugin:score")
            self.assertEqual(score(PromptRecord([0], metadata={"scale": 2}), [1, 2]), 4.0)
            with self.assertRaisesRegex(ValueError, "non-finite"):
                build_reward("test_reward_plugin:bad")(PromptRecord([0]), [1])


if __name__ == "__main__":
    unittest.main()
