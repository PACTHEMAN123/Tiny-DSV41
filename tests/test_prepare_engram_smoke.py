import json
from pathlib import Path
import tempfile
import unittest

import torch

from dsv41_train.sft_data import SFTDataset
from scripts.prepare_engram_smoke import main, pack_dataset, prepare
from qwen_helpers import tiny_config


class PrepareEngramSmokeTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.data_path = self.folder / "data.jsonl"

    def source(self, records, max_length=8):
        self.data_path.write_text("\n".join(json.dumps(row) for row in records) + "\n")
        return SFTDataset.from_jsonl(self.data_path, max_length=max_length)

    def test_continuous_packing_preserves_masks_removes_padding_and_repeats(self):
        dataset = self.source([
            {"input_ids": [3, 4, 0, 0], "labels": [-100, 4, 0, -100], "attention_mask": [1, 1, 1, 0]},
            {"input_ids": [5, 6, 0, 0], "labels": [-100, 6, -100, -100], "attention_mask": [1, 1, 0, 0]},
        ], max_length=4)
        result = pack_dataset(dataset, seq_len=4, packs=3)
        self.assertEqual(result["input_ids"].tolist(), [[3, 4, 0, 5], [6, 3, 4, 0], [5, 6, 3, 4]])
        self.assertEqual(result["labels"].tolist(), [[-100, 4, 0, -100], [-100, -100, 4, 0], [-100, 6, -100, 4]])
        metadata = result["metadata"]
        self.assertEqual(metadata["source_padding_tokens_removed"], 3)
        self.assertEqual(metadata["packing_padding_tokens"], 0)
        self.assertEqual(metadata["supervised_tokens"], 6)
        self.assertTrue(metadata["source_repeated"])
        self.assertFalse(metadata["quality_evaluation_data"])
        self.assertFalse(metadata["representative_performance_data"])
        self.assertEqual(metadata["source_data_identity"], dataset.identity)

    def test_empty_packs_are_skipped_and_every_accepted_pack_has_targets(self):
        dataset = self.source([{"input_ids": [1, 2, 3, 4, 5, 6],
                                "labels": [-100, -100, -100, -100, 5, -100]}])
        result = pack_dataset(dataset, seq_len=3, packs=4)
        self.assertEqual(result["metadata"]["empty_packs_discarded"], 4)
        self.assertEqual(result["labels"].tolist(), [[-100, 5, -100]] * 4)

    def test_boundary_only_targets_reject_instead_of_looping_forever(self):
        dataset = self.source([{"input_ids": [1, 2, 3, 4, 5, 6],
                                "labels": [-100, -100, -100, 4, -100, -100]}])
        with self.assertRaisesRegex(ValueError, "no supervised tokens remain"):
            pack_dataset(dataset, seq_len=3, packs=2)

    def test_all_masked_source_is_rejected(self):
        dataset = SFTDataset([[1, 2]], [[-100, -100]], [[True, True]], {})
        with self.assertRaisesRegex(ValueError, "no supervised tokens"):
            pack_dataset(dataset, seq_len=2, packs=2)

    def model_fixture(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        config = tiny_config()
        (self.folder / "config.json").write_text(json.dumps(config.to_dict()))
        tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "p": 3, "answer": 4, "word": 5}, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.save(str(self.folder / "tokenizer.json"))
        self.data_path.write_text('{"prompt":"p","response":"answer word"}\n')
        return config

    def test_local_text_encoding_matches_sftdataset_and_output_loads_safely(self):
        config = self.model_fixture()
        source = SFTDataset.from_jsonl(self.data_path, max_length=16,
                                      tokenizer_path=self.folder / "tokenizer.json",
                                      bos_token_id=config.bos_token_id, eos_token_id=config.eos_token_id,
                                      pad_token_id=config.eos_token_id)
        wanted = pack_dataset(source, seq_len=4, packs=64)
        result = prepare(self.data_path, self.folder, seq_len=4, packs=64, source_max_length=16)
        for key in ("input_ids", "labels"):
            torch.testing.assert_close(result[key], wanted[key])
        output = self.folder / "smoke.pt"
        main(["--model-path", str(self.folder), "--data", str(self.data_path),
              "--output", str(output), "--seq-len", "4", "--packs", "64", "--source-max-length", "16"])
        saved = torch.load(output, weights_only=True)
        torch.testing.assert_close(saved["labels"], wanted["labels"])
        self.assertEqual(saved["metadata"]["purpose"], "execution-only")
        self.assertEqual(json.loads(output.with_suffix(".json").read_text()), saved["metadata"])

    def test_source_truncation_and_invalid_vocabulary_are_rejected(self):
        self.model_fixture()
        with self.assertRaisesRegex(ValueError, "reaches source_max_length"):
            prepare(self.data_path, self.folder, seq_len=4, source_max_length=4)
        self.data_path.write_text('{"input_ids":[1,999],"labels":[-100,999]}\n')
        with self.assertRaisesRegex(ValueError, "outside model vocabulary"):
            prepare(self.data_path, self.folder, seq_len=4)

    def test_determinism_and_size_validation(self):
        dataset = self.source([{"input_ids": [1, 2, 3], "labels": [-100, 2, 3]}])
        left = pack_dataset(dataset, seq_len=128, packs=64)
        right = pack_dataset(dataset, seq_len=128, packs=64)
        for key in ("input_ids", "labels"):
            torch.testing.assert_close(left[key], right[key])
        self.assertEqual(left["metadata"], right["metadata"])
        self.assertEqual(tuple(left["input_ids"].shape), (64, 128))
        self.assertTrue(left["labels"][:, 0].eq(-100).all())
        self.assertTrue(left["labels"][:, 1:].ne(-100).any(dim=1).all())
        for kwargs in ({"seq_len": 1}, {"packs": 0}, {"packs": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                pack_dataset(dataset, **kwargs)


if __name__ == "__main__":
    unittest.main()
