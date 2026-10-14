import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from dsv41_train.sft_data import SFTDataset


class CharacterTokenizer:
    """An offset-aware deterministic tokenizer; no model files or downloads."""

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return SimpleNamespace(ids=[ord(char) + 3 for char in text],
                               offsets=[(index, index + 1) for index in range(len(text))])

    def to_str(self):
        return "test-character-tokenizer-v1"


class SFTDataTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "data.jsonl"

    def dataset(self, records, **kwargs):
        self.path.write_text("\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n")
        return SFTDataset.from_jsonl(self.path, **kwargs)

    def test_prompt_response_masks_prompt_padding_and_counts_shifted_targets(self):
        dataset = self.dataset([{"prompt": "问", "response": "回答"}],
                               tokenizer=CharacterTokenizer(), max_length=10,
                               pad_token_id=0, bos_token_id=1, eos_token_id=2)
        self.assertEqual(dataset.input_ids[0].tolist(), [1, ord("问") + 3, 13, 13,
                                                       ord("回") + 3, ord("答") + 3, 2, 0, 0, 0])
        self.assertEqual(dataset.labels[0].tolist(), [-100] * 4 + [ord("回") + 3, ord("答") + 3, 2]
                         + [-100] * 3)
        self.assertEqual(dataset.attention_mask[0].tolist(), [True] * 7 + [False] * 3)
        self.assertEqual(dataset.batch([0])["supervised_tokens"].item(), 3)

    def test_alpaca_and_equivalent_prompt_use_identical_explicit_format(self):
        tokenizer = CharacterTokenizer()
        first = self.dataset([{"instruction": "Do", "input": "this", "output": "yes"}],
                             tokenizer=tokenizer, max_length=20)
        second = self.dataset([{"prompt": "Do\n\nthis", "response": "yes"}],
                              tokenizer=tokenizer, max_length=20)
        torch.testing.assert_close(first.input_ids, second.input_ids)
        torch.testing.assert_close(first.labels, second.labels)

    def test_truncation_retains_answer_prefix_and_rejects_lost_answer(self):
        dataset = self.dataset([{"prompt": "p", "response": "answer"}],
                               tokenizer=CharacterTokenizer(), max_length=5)
        self.assertEqual(dataset.supervised_tokens.tolist(), [2])
        with self.assertRaisesRegex(ValueError, "data.jsonl:1:.*no supervised"):
            SFTDataset.from_jsonl(self.path, tokenizer=CharacterTokenizer(), max_length=3)

    def test_pretokenized_labels_are_causally_aligned_and_padding_is_masked(self):
        dataset = self.dataset([{"input_ids": [5, 6, 7, 0], "labels": [5, -100, 7, 0],
                                 "attention_mask": [1, 1, 1, 0]}], max_length=6)
        self.assertEqual(dataset.labels.tolist(), [[-100, -100, 7, -100, -100, -100]])
        self.assertEqual(dataset.supervised_tokens.tolist(), [1])
        self.assertEqual(dataset.identity["tokenizer_sha256"], None)

    def test_only_first_position_supervised_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no supervised"):
            self.dataset([{"input_ids": [4, 5], "labels": [4, -100]}], max_length=2)

    def test_token_crossing_answer_boundary_is_masked(self):
        class CrossingTokenizer(CharacterTokenizer):
            def encode(self, text, *, add_special_tokens):
                return SimpleNamespace(ids=[8, 9, 10], offsets=[(0, 1), (1, 4), (4, 5)])
        dataset = self.dataset([{"prompt": "p", "response": "ab"}],
                               tokenizer=CrossingTokenizer(), max_length=4)
        self.assertEqual(dataset.labels.tolist(), [[-100, -100, 10, -100]])

    def test_invalid_records_and_shapes_fail_with_source_line(self):
        cases = [
            {"input_ids": [1, 2], "labels": [2]},
            {"input_ids": [1, True], "labels": [-100, 2]},
            {"input_ids": [1, 2], "labels": [-100, 2], "attention_mask": [0, 1]},
            {"input_ids": [1, 2], "labels": [-100, 2], "attention_mask": [1, 2]},
            {"input_ids": [1, 2], "labels": [-100, -5]},
            {"input_ids": [1, 2], "labels": [-100.0, 2]},
            {"prompt": "p", "response": ""},
        ]
        for record in cases:
            with self.subTest(record=record), self.assertRaisesRegex(ValueError, "data.jsonl:1:"):
                self.dataset([record], tokenizer=CharacterTokenizer(), max_length=8)

    def test_identity_tracks_source_tokenizer_and_processing_not_path(self):
        dataset = self.dataset([{"prompt": "p", "response": "a"}],
                               tokenizer=CharacterTokenizer(), max_length=8)
        other_path = self.path.with_name("same-content.jsonl")
        other_path.write_bytes(self.path.read_bytes())
        same = SFTDataset.from_jsonl(other_path, tokenizer=CharacterTokenizer(), max_length=8)
        self.assertEqual(dataset.identity, same.identity)
        shorter = SFTDataset.from_jsonl(other_path, tokenizer=CharacterTokenizer(), max_length=7)
        self.assertNotEqual(dataset.identity["fingerprint"], shorter.identity["fingerprint"])
        changed = self.dataset([{"prompt": "p", "response": "b"}],
                               tokenizer=CharacterTokenizer(), max_length=8)
        self.assertNotEqual(dataset.identity["fingerprint"], changed.identity["fingerprint"])

    def test_distributed_microbatches_partition_epoch_and_resume_exactly(self):
        records = [{"input_ids": [1, index + 3], "labels": [-100, index + 3]} for index in range(12)]
        dataset = self.dataset(records, max_length=2)
        options = dict(batch_size=2, accumulation_steps=3, data_world_size=2, seed=17)
        first_epoch = [index for micro in range(3) for rank in range(2)
                       for index in dataset.indices_for_step(0, micro, data_rank=rank, **options)]
        self.assertEqual(sorted(first_epoch), list(range(12)))
        fresh = SFTDataset.from_jsonl(self.path, max_length=2)
        for step in range(1, 4):
            for micro in range(3):
                for rank in range(2):
                    expected = dataset.batch_for_step(step, micro, data_rank=rank, **options)
                    resumed = fresh.batch_for_step(step, micro, data_rank=rank, **options)
                    for name in expected:
                        torch.testing.assert_close(expected[name], resumed[name])

    def test_batch_crosses_epoch_boundary_without_losing_examples(self):
        dataset = self.dataset([{"input_ids": [1, index + 3], "labels": [-100, index + 3]}
                                for index in range(5)], max_length=2)
        options = dict(batch_size=3, accumulation_steps=1, seed=42)
        indices = dataset.indices_for_step(0, 0, **options) + dataset.indices_for_step(1, 0, **options)
        self.assertEqual(sorted(indices[:5]), list(range(5)))
        self.assertEqual(dataset.batch_for_step(1, 0, **options)["supervised_tokens"].item(), 3)
        with self.assertRaisesRegex(ValueError, "micro_step"):
            dataset.indices_for_step(0, 1, **options)

    def test_text_needs_explicit_local_tokenizer_and_empty_source_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "local tokenizer"):
            self.dataset([{"prompt": "p", "response": "a"}], max_length=8)
        self.path.write_text("\n")
        with self.assertRaisesRegex(ValueError, "no records"):
            SFTDataset.from_jsonl(self.path, max_length=8)

    def test_local_tokenizer_saved_padding_and_truncation_do_not_override_dataset(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "p": 3, "answer": 4, "word": 5}, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.enable_truncation(max_length=2)
        tokenizer.enable_padding(length=8, pad_id=0)
        tokenizer_path = self.path.with_name("tokenizer.json")
        tokenizer.save(str(tokenizer_path))
        saved = tokenizer.to_str()
        dataset = self.dataset([{"prompt": "p", "response": "answer word"}],
                               tokenizer_path=tokenizer_path, max_length=6, eos_token_id=2)
        self.assertEqual(dataset.input_ids.tolist(), [[3, 4, 5, 2, 0, 0]])
        self.assertEqual(dataset.labels.tolist(), [[-100, 4, 5, 2, -100, -100]])
        same = SFTDataset.from_jsonl(self.path, tokenizer=tokenizer, max_length=6, eos_token_id=2)
        torch.testing.assert_close(dataset.input_ids, same.input_ids)
        self.assertEqual(tokenizer.to_str(), saved)


if __name__ == "__main__":
    unittest.main()
