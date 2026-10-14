"""CPU checks against the actual local Qwen and DeepSeek template resources."""

import json
from pathlib import Path
import tempfile
import unittest

from scripts.prepare_identity_sft import (FORMAT, TARGET, TRAIN_QUESTIONS, TemplateCodec,
                                          digest, make_example, prepare)


ROOT = Path(__file__).resolve().parents[1]
METADATA = ROOT / "artifacts/new-l20d-20261014-longseq/model-metadata"
HAS_METADATA = all((METADATA / model / "tokenizer.json").is_file() for model in ("qwen", "dsv4"))


class IdentityInputValidationTest(unittest.TestCase):
    def test_rejects_short_sequence_or_existing_destination_before_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            with self.assertRaisesRegex(ValueError, "at least 4096"):
                prepare(path / "missing", path / "out", 2048)
            with self.assertRaisesRegex(ValueError, "already exists"):
                prepare(path / "missing", path, 4096)
            self.assertFalse((path / "out").exists())
            with self.assertRaisesRegex(ValueError, "nonempty fallback reason"):
                prepare(path / "missing", path / "out", 2048, allow_shorter_sequence=True)
            with self.assertRaisesRegex(ValueError, "nonempty fallback reason"):
                prepare(path / "missing", path / "out", 2048, allow_shorter_sequence=True, fallback_reason="  ")


@unittest.skipUnless(HAS_METADATA, "actual model tokenizer/template metadata is not present")
class ActualIdentityTemplateTest(unittest.TestCase):
    def test_explicit_shorter_fallback_rebuilds_complete_targets(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "shorter"
            manifest = prepare(METADATA / "dsv4", output, 2048, allow_shorter_sequence=True,
                               fallback_reason="CPU regression fixture: simulated 4096 capacity rejection")
            self.assertEqual(manifest["default_target_sequence_length"], 4096)
            self.assertEqual(manifest["requested_sequence_length"], 2048)
            self.assertEqual(manifest["effective_sequence_length"], 2048)
            self.assertTrue(manifest["fallback_authorized"])
            self.assertIn("CPU regression fixture", manifest["fallback_reason"])
            codec = TemplateCodec(METADATA / "dsv4")
            for line in (output / "train.jsonl").read_text().splitlines():
                row = json.loads(line)
                self.assertEqual(len(row["input_ids"]), 2048)
                self.assertEqual(sum(row["attention_mask"]), 2048)
                labels = [token for token in row["labels"] if token != -100]
                self.assertEqual(codec.decode(labels[:-1]), TARGET)
                self.assertIn(labels[-1], codec.eos_token_ids)
                # Prefix truncation of an old 4096 record would lose every label.
                self.assertEqual(labels, codec.encode(TARGET).ids + [codec.eos_token_ids[0]])

    def test_actual_no_thinking_templates_and_exact_stop_boundaries(self):
        expected = {
            "qwen": "<|im_start|>user\n你是谁<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            "dsv4": "<｜begin▁of▁sentence｜><｜User｜>你是谁<｜Assistant｜></think>",
        }
        for model in expected:
            with self.subTest(model=model):
                codec = TemplateCodec(METADATA / model)
                record, evaluation, _ = make_example(codec, "short", "你是谁", seq_len=4096)
                self.assertEqual(codec.render("你是谁"), expected[model])
                self.assertEqual(evaluation["prompt_text"], expected[model])
                self.assertEqual(codec.decode(evaluation["target_ids"][:-1]), TARGET)
                self.assertEqual(len(evaluation["target_ids"]), 5)
                self.assertIn(evaluation["target_ids"][-1], codec.eos_token_ids)
                self.assertEqual([token for token in record["labels"] if token != -100], evaluation["target_ids"])
                self.assertEqual(codec.decode(record["input_ids"]), codec.render("你是谁", TARGET))
                if model == "qwen":
                    self.assertEqual(record["labels"][-1], -100)  # Template newline after EOS.
                    self.assertNotEqual(record["input_ids"][0], 151643)  # No extra BOS.
                else:
                    self.assertEqual(record["input_ids"].count(0), 1)  # Template owns BOS.

    def test_32_real_4096_examples_and_disjoint_generation_checks(self):
        for model in ("qwen", "dsv4"):
            with self.subTest(model=model), tempfile.TemporaryDirectory() as folder:
                output = Path(folder) / "data"
                manifest = prepare(METADATA / model, output)
                codec = TemplateCodec(METADATA / model)
                rows = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
                stress = [json.loads(line) for line in (output / "long_context_stress.jsonl").read_text().splitlines()]
                heldout = json.loads((output / "heldout.json").read_text())
                self.assertEqual(len(rows), 32)
                self.assertEqual(len(stress), 2)
                for row in rows + stress:
                    self.assertEqual(len(row["input_ids"]), 4096)
                    self.assertEqual(len(row["labels"]), 4096)
                    self.assertEqual(row["attention_mask"], [1] * 4096)
                    self.assertEqual(row["encoding"]["format"], FORMAT)
                    decoded = codec.decode(row["input_ids"])
                    self.assertNotIn("\ufffd", decoded)
                    self.assertEqual(codec.encode(decoded).ids, row["input_ids"])
                    supervised = [item for item in row["labels"] if item != -100]
                    self.assertEqual(codec.decode(supervised[:-1]), TARGET)
                    self.assertIn(supervised[-1], codec.eos_token_ids)
                    start = next(index for index, item in enumerate(row["labels"]) if item != -100)
                    self.assertNotIn("RTP", codec.decode(row["input_ids"][:start]))
                    self.assertEqual(row["input_ids"][start:start + len(supervised)], supervised)
                self.assertEqual(len(heldout["prompts"]), 5)
                self.assertEqual([p["seen_question_in_train"] for p in heldout["prompts"]], [True, False, False, False, False])
                self.assertEqual(heldout["prompts"][0]["question"], "你是谁")
                for prompt in heldout["prompts"]:
                    self.assertEqual(codec.decode(prompt["input_ids"]), prompt["prompt_text"])
                    self.assertEqual(codec.encode(prompt["prompt_text"]).ids, prompt["input_ids"])
                    self.assertNotIn("RTP", prompt["prompt_text"])
                    self.assertEqual(prompt["target_text"], TARGET)
                    self.assertIn(prompt["target_ids"][-1], heldout["eos_token_ids"])
                    self.assertLessEqual(len(prompt["input_ids"]) + heldout["max_new_tokens"], codec.max_positions)
                    if not prompt["seen_question_in_train"]:
                        self.assertNotIn(prompt["question"], TRAIN_QUESTIONS)
                self.assertEqual(len(heldout["prompts"][-1]["input_ids"]), 4096)
                self.assertTrue(heldout["prompts"][-1]["long_context"])
                self.assertEqual(manifest["train_effective_tokens_min"], 4096)
                self.assertEqual(manifest["train_effective_tokens_max"], 4096)
                self.assertEqual(manifest["train_supervised_tokens_min"], 5)
                training_contexts = {item["context_sha256"] for item in manifest["train_audit"]}
                independent = [manifest["heldout_long_audit"], *manifest["long_context_stress_audit"]]
                self.assertEqual(len({item["context_sha256"] for item in independent}), 3)
                self.assertTrue(all(item["context_sha256"] not in training_contexts for item in independent))
                for name, info in manifest["files"].items():
                    content = (output / name).read_bytes()
                    self.assertEqual(digest(content), info["sha256"])
                    self.assertEqual(len(content), info["bytes"])
                # Repeated CPU preprocessing produces identical reviewed artifacts.
                second = Path(folder) / "repeat"
                prepare(METADATA / model, second)
                for name in (*manifest["files"], "manifest.json"):
                    self.assertEqual((output / name).read_bytes(), (second / name).read_bytes())

    def test_rejects_prompt_leakage_and_insufficient_generation_positions(self):
        codec = TemplateCodec(METADATA / "qwen")
        for question, context in ((TARGET, None), ("你是谁", "RTP背景")):
            with self.assertRaisesRegex(ValueError, "leaked"):
                make_example(codec, "bad", question, seq_len=4096, context=context)
        codec.max_positions = 8
        with self.assertRaisesRegex(ValueError, "16 generated"):
            make_example(codec, "bad", "你是谁", seq_len=4096)

    def test_unknown_template_globals_fail_without_guessed_rendering(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for filename in ("config.json", "tokenizer.json"):
                (root / filename).symlink_to(METADATA / "qwen" / filename)
            config = json.loads((METADATA / "qwen/tokenizer_config.json").read_text())
            config["chat_template"] = "{{ unsupported_runtime_function(messages) }}"
            (root / "tokenizer_config.json").write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "unsupported chat template globals"):
                TemplateCodec(root)


if __name__ == "__main__":
    unittest.main()
