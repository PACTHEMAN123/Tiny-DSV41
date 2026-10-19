import json
import tempfile
import unittest
from pathlib import Path

import torch
from tokenizers import Tokenizer, models, pre_tokenizers

from dsv41_train.rl.tokenizer import ModelTokenizer
from dsv41_train.sft_data import collate_examples, encode_example, load_jsonl


class SFTDataTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        tokens = ["[UNK]", "[BOS]", "[EOS]", "[STOP]", "user", "assistant", "hello", "reply", "suffix"]
        tokenizer = Tokenizer(models.WordLevel(dict(zip(tokens, range(len(tokens)))), unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        tokenizer.add_special_tokens(tokens[:4])
        tokenizer.save(str(self.directory / "tokenizer.json"))
        (self.directory / "config.json").write_text(json.dumps({"vocab_size": len(tokens), "eos_token_id": 2}))
        (self.directory / "tokenizer_config.json").write_text(json.dumps({
            "tokenizer_class": "PreTrainedTokenizerFast", "bos_token": "[BOS]", "eos_token": "[EOS]",
        }))
        self.template = '{{ bos_token }} {% for message in messages %}{{ message.role }} {{ message.content }} {{ eos_token }} {% endfor %}{% if add_generation_prompt %}assistant{% else %}suffix{% endif %}'
        self.write_template(self.template)

    def write_template(self, template):
        (self.directory / "chat_template.jinja").write_text(template)
        self.tokenizer = ModelTokenizer(self.directory)

    def encode(self, record=None, **kwargs):
        return encode_example(record or {"prompt": "hello", "answer": "reply"}, self.tokenizer,
                              **{"max_length": 64, **kwargs})

    def test_only_final_answer_and_eos_are_supervised_without_shift(self):
        example = self.encode({"messages": [
            {"role": "user", "content": "hello"}, {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "hello"},
        ], "answer": "reply"})
        prompt = example["prompt_ids"]
        self.assertEqual(example["input_ids"], [*prompt, 7, 2, 8])
        self.assertEqual(example["labels"], [-100] * len(prompt) + [7, 2, -100])
        self.assertEqual(example["attention_mask"], [1] * len(example["input_ids"]))
        self.assertEqual(self.encode(), self.encode({"messages": [{"role": "user", "content": "hello"}], "answer": "reply"}))

    def test_eos_comes_from_template_and_must_match_checkpoint_stop_ids(self):
        (self.directory / "generation_config.json").write_text(json.dumps({"eos_token_id": [3, 2]}))
        self.write_template(self.template.replace("{{ eos_token }}", "[STOP]"))
        self.assertEqual([value for value in self.encode()["labels"] if value != -100], [7, 3])
        self.write_template(self.template.replace("{{ eos_token }}", ""))
        with self.assertRaisesRegex(ValueError, "EOS"):
            self.encode()
        self.write_template(self.template)
        with self.assertRaisesRegex(ValueError, "before EOS"):
            self.encode({"prompt": "hello", "answer": "[EOS]"})

    def test_right_padding_masks_labels_even_when_pad_is_eos_or_a_real_input_token(self):
        examples = [self.encode(), self.encode({"prompt": "unknown hello hello", "answer": "reply"})]
        for pad_id in (0, 2):
            with self.subTest(pad_id=pad_id):
                batch = collate_examples(examples, pad_token_id=pad_id, pad_to_multiple_of=4)
                self.assertEqual(set(batch), {"input_ids", "labels", "attention_mask"})
                self.assertEqual(batch["input_ids"].shape, (2, 12))
                self.assertTrue(all(value.dtype == torch.long for value in batch.values()))
                for index, example in enumerate(examples):
                    size = len(example["input_ids"])
                    self.assertEqual(batch["input_ids"][index, :size].tolist(), example["input_ids"])
                    self.assertEqual(batch["labels"][index, :size].tolist(), example["labels"])
                    self.assertTrue((batch["input_ids"][index, size:] == pad_id).all())
                    self.assertTrue((batch["labels"][index, size:] == -100).all())
                    self.assertEqual(batch["attention_mask"][index].tolist(), [1] * size + [0] * (12 - size))

    def test_length_limit_is_actual_tokens_and_never_truncates(self):
        size = len(self.encode()["input_ids"])
        self.assertEqual(len(self.encode(max_length=size)["input_ids"]), size)
        with self.assertRaisesRegex(ValueError, "truncation is disabled"):
            self.encode(max_length=size - 1)
        for length in (True, 1, 2.5):
            with self.subTest(length=length), self.assertRaisesRegex(ValueError, "max_length"):
                self.encode(max_length=length)

    def test_4096_context_contains_real_tokens_instead_of_padding(self):
        example = self.encode({"prompt": " ".join(["hello"] * 4089), "answer": "reply"}, max_length=4096)
        batch = collate_examples([example], pad_token_id=2)
        self.assertEqual(batch["input_ids"].shape, (1, 4096))
        self.assertEqual(batch["attention_mask"].sum().item(), 4096)
        self.assertEqual((batch["labels"][:, 1:] != -100).sum().item(), 2)

    def test_invalid_records_and_nontext_history_are_rejected(self):
        records = [None, {}, {"prompt": "hello"}, {"prompt": "hello", "messages": [], "answer": "reply"},
                   {"prompt": "", "answer": "reply"}, {"prompt": "hello", "answer": ""},
                   {"messages": [{"role": "assistant", "content": "reply"}], "answer": "reply"},
                   {"messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}], "answer": "reply"},
                   {"messages": [{"role": "tool", "content": "hello"}], "answer": "reply"}]
        for record in records:
            with self.subTest(record=record), self.assertRaises(ValueError):
                encode_example(record, self.tokenizer, max_length=64)
        with self.assertRaisesRegex(ValueError, "boolean"):
            self.encode(enable_thinking="false")

    def test_jsonl_reports_source_line_and_rejects_empty_dataset(self):
        path = self.directory / "data.jsonl"
        path.write_text('\n{"prompt":"hello","answer":"reply"}\n')
        self.assertEqual(load_jsonl(path, self.tokenizer, max_length=64), [self.encode()])
        for invalid in ("not json", '{"prompt":"hello"}'):
            path.write_text('\n' + invalid + '\n')
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, r"data.jsonl:2:"):
                load_jsonl(path, self.tokenizer, max_length=64)
        path.write_text("\n \n")
        with self.assertRaisesRegex(ValueError, "at least one example"):
            load_jsonl(path, self.tokenizer, max_length=64)

    def test_collator_rejects_empty_batches_and_invalid_padding(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            collate_examples([], pad_token_id=0)
        for kwargs in ({"pad_token_id": -1}, {"pad_token_id": True},
                       {"pad_token_id": 0, "pad_to_multiple_of": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                collate_examples([self.encode()], **kwargs)


if __name__ == "__main__":
    unittest.main()
