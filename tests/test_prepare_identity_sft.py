import io
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from dsv41_train.sft_data import encode_example
from scripts import prepare_identity_sft as prepare


class CharacterTokenizer:
    """Deterministic text fixture; actual model token counts are checked remotely."""

    eos_token_ids = (2,)

    def encode_completion(self, messages, answer, **kwargs):
        prompt = [1, *[ord(char) + 3 for message in messages for char in message["content"]]]
        return prompt, [*prompt, *[ord(char) + 3 for char in answer], 2]


class IdentityDataTests(unittest.TestCase):
    def test_optional_short_rows_preserve_long_rows_and_fixed_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            reports, rows = {}, {}
            for mode, options in (("long", []), ("mixed", ["--include-short-training"]),
                                  ("diverse", ["--diverse-training"])):
                folder = Path(directory) / mode
                argv = ["prepare_identity_sft", "--model-path", "fixture", "--output-dir", str(folder),
                        "--min-active-tokens", "128", "--max-length", "256", *options]
                with patch("sys.argv", argv), patch.object(prepare, "ModelTokenizer", return_value=CharacterTokenizer()), redirect_stdout(io.StringIO()) as output:
                    prepare.main()
                reports[mode] = json.loads(output.getvalue())
                rows[mode] = [json.loads(line) for line in (folder / "train.jsonl").read_text().splitlines()]
            original = Path(directory) / "long"
            mixed = Path(directory) / "mixed"
            diverse = Path(directory) / "diverse"
            self.assertEqual((original / "eval.jsonl").read_bytes(), (mixed / "eval.jsonl").read_bytes())
            self.assertEqual((original / "eval.jsonl").read_bytes(), (diverse / "eval.jsonl").read_bytes())
            self.assertEqual((original / "train.jsonl").read_bytes(), b"".join((mixed / "train.jsonl").read_bytes().splitlines(keepends=True)[:4]))
            self.assertEqual(len(rows["long"]), 4)
            self.assertEqual(rows["mixed"][4:], [{"prompt": q, "answer": prepare.TARGET} for q in prepare.QUESTIONS])
            lengths = []
            for row in rows["mixed"]:
                self.assertNotIn(prepare.TARGET, row["prompt"])
                encoded = encode_example(row, CharacterTokenizer(), max_length=256)
                self.assertEqual(encoded["attention_mask"], [1] * len(encoded["input_ids"]))
                lengths.append(len(encoded["input_ids"]))
            self.assertTrue(all(length >= 128 for length in lengths[:4]))
            self.assertTrue(all(length < 128 for length in lengths[4:]))
            self.assertEqual(reports["mixed"]["active_tokens"], lengths)
            for name, values in (("long", lengths[:4]), ("short", lengths[4:])):
                self.assertEqual(reports["mixed"]["training_groups"][name],
                                 {"examples": 4, "min_active_tokens": min(values), "max_active_tokens": max(values)})
            self.assertEqual(reports["long"]["training_groups"]["short"],
                             {"examples": 0, "min_active_tokens": None, "max_active_tokens": None})
            self.assertEqual(reports["mixed"]["eval_roles"][0], "seen_short_prompt")
            self.assertEqual(reports["long"]["eval_roles"][0], "seen_question_new_short_context")
            self.assertEqual(reports["mixed"]["eval_roles"][1:],
                             ["unseen_short_question"] * 3 + ["unseen_long_context_and_question"])
            self.assertEqual(reports["diverse"]["eval_roles"], reports["mixed"]["eval_roles"])
            self.assertEqual(len(rows["diverse"]), 32)
            self.assertEqual((mixed / "train.jsonl").read_bytes(),
                             b"".join((diverse / "train.jsonl").read_bytes().splitlines(keepends=True)[:8]))
            self.assertEqual(sum(row["prompt"] == "你是谁" for row in rows["diverse"]), 2)
            lengths, groups = [], {"long": [], "short": []}
            for index, row in enumerate(rows["diverse"]):
                self.assertNotIn(prepare.TARGET, row["prompt"])
                self.assertTrue(all(unseen not in row["prompt"] for unseen in prepare.UNSEEN))
                encoded = encode_example(row, CharacterTokenizer(), max_length=256)
                self.assertTrue(all(encoded["attention_mask"]))
                self.assertEqual([value for value in encoded["labels"] if value != -100],
                                 [*[ord(char) + 3 for char in prepare.TARGET], 2])
                length = len(encoded["input_ids"])
                lengths.append(length)
                group = "long" if index % 8 < 4 else "short"
                groups[group].append(length)
                self.assertEqual(length >= 128, group == "long")
            for start, question_start in ((8, 0), (16, 4)):
                questions = prepare.DIVERSE_QUESTIONS[question_start:question_start + 4]
                self.assertEqual([row["prompt"] for row in rows["diverse"][start + 4:start + 8]], list(questions))
                for row, question in zip(rows["diverse"][start:start + 4], questions):
                    self.assertTrue(row["prompt"].endswith("请回答最后的问题：" + question))
                    self.assertTrue(all(unseen not in question for unseen in prepare.UNSEEN))
            self.assertEqual([row["prompt"] for row in rows["diverse"][28:]],
                             ["你是谁", "你是谁", "大家平时叫你什么？", "这个智能助手叫什么名字？"])
            for row, question, context in zip(rows["diverse"][24:28], prepare.EXTRA_LONG_QUESTIONS, prepare.EXTRA_BACKGROUNDS):
                self.assertTrue(row["prompt"].endswith("请回答最后的问题：" + question))
                self.assertIn("记录1：" + context[0], row["prompt"])
                self.assertNotIn(prepare.NEW_BACKGROUND[0], row["prompt"])
            self.assertEqual(reports["diverse"]["active_tokens"], lengths)
            self.assertEqual(reports["diverse"]["training_groups"],
                             {name: {"examples": 16, "min_active_tokens": min(values), "max_active_tokens": max(values)}
                              for name, values in groups.items()})


if __name__ == "__main__":
    unittest.main()
