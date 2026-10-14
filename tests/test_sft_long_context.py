"""Effective long-context inputs and bounded probes remain different contracts."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch

from dsv41_train.sft_data import SFTDataset
from train_sft import _bounded_lora_probe


class LongContextDataTests(unittest.TestCase):
    def test_padding_does_not_satisfy_effective_length(self):
        dataset = SFTDataset([[1, 2, 0, 0]], [[-100, 2, -100, -100]], [[1, 1, 0, 0]], {})
        self.assertEqual(dataset.length_statistics()['padding_positions'], 2)
        with self.assertRaisesRegex(ValueError, 'padding is not long-context input'):
            dataset.require_active_tokens(4)
        dataset.require_active_tokens(2)

    def test_long_sequence_probe_retains_tail_target_without_shortening_training(self):
        ids = torch.arange(8192).reshape(1, -1)
        labels = torch.full_like(ids, -100)
        labels[:, -5:] = ids[:, -5:]
        batch = {'input_ids': ids, 'labels': labels, 'attention_mask': torch.ones_like(ids, dtype=torch.bool)}
        probe = _bounded_lora_probe(batch, 128)
        self.assertEqual(batch['input_ids'].shape, (1, 8192))
        self.assertEqual(probe['input_ids'].shape, (1, 128))
        self.assertEqual(int(probe['labels'][:, 1:].ne(-100).sum()), 5)
        self.assertEqual(probe['labels'][0, -1], 8191)
        self.assertEqual(probe['labels'][0, 0], -100)

    def test_default_small_probe_preserves_old_inputs_and_labels(self):
        batch = {'input_ids': torch.tensor([[1, 2, 3, 0]]), 'labels': torch.tensor([[-100, -100, 3, -100]]),
                 'attention_mask': torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)}
        probe = _bounded_lora_probe(batch, 4096)
        for key in batch:
            torch.testing.assert_close(batch[key], probe[key])

    def test_preencoded_chat_identity_binds_actual_tokenizer(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            tokenizer = folder / 'tokenizer.json'
            tokenizer.write_text('local tokenizer fixture bytes')
            encoding = {'format': 'identity-chat-template-v1', 'tokenizer_sha256': hashlib.sha256(tokenizer.read_bytes()).hexdigest(),
                        'config_sha256': '1' * 64, 'template_sha256': '2' * 64, 'model_type': 'qwen', 'template_options': {'enable_thinking': False}}
            path = folder / 'data.jsonl'
            record = {'input_ids': [1, 2, 3, 4], 'labels': [-100, -100, 3, 4], 'encoding': encoding}
            path.write_text(json.dumps(record) + '\n')
            dataset = SFTDataset.from_jsonl(path, max_length=4, tokenizer_path=tokenizer)
            dataset.require_active_tokens(4)
            self.assertEqual(dataset.identity['encoding'], encoding)
            self.assertEqual(dataset.length_statistics()['fully_active_examples'], 1)
            tokenizer.write_text('different tokenizer bytes')
            with self.assertRaisesRegex(ValueError, 'matching local tokenizer'):
                SFTDataset.from_jsonl(path, max_length=4, tokenizer_path=tokenizer)


if __name__ == '__main__':
    unittest.main()
