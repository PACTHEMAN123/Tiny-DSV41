"""Special-token inheritance from released nested DeepSeek checkpoint configs."""

import json
from pathlib import Path
import tempfile
import unittest

from dsv41_train.models.dsv4.config import DeepSeekV41Config
from dsv41_train.models.dsv4.checkpoint import _prefix_config


class NestedDeepSeekConfigTest(unittest.TestCase):
    def read(self, value):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            return DeepSeekV41Config.from_json(path)

    def test_released_config_and_prefix_preserve_outer_special_tokens(self):
        folder = Path(__file__).resolve().parents[1] / "dsv41_train/models/dsv4"
        for config in (DeepSeekV41Config.from_json(folder / "config.json"),
                       _prefix_config(folder, 1), _prefix_config(folder, 2)):
            self.assertEqual((config.bos_token_id, config.eos_token_id, config.pad_token_id),
                             (0, 1, 2))

    def test_explicit_text_values_override_outer_defaults_individually(self):
        config = self.read({
            "bos_token_id": 0, "eos_token_id": 1, "pad_token_id": 2,
            "vocab_size": 999,
            "text_config": {"bos_token_id": 7, "vocab_size": 64,
                            "engram_pad_token_id": 3},
        })
        self.assertEqual((config.bos_token_id, config.eos_token_id, config.pad_token_id),
                         (7, 1, 2))
        self.assertEqual(config.vocab_size, 64)
        self.assertEqual(config.engram_pad_id, 3)

    def test_flat_configs_and_missing_outer_ids_retain_existing_behavior(self):
        explicit = self.read({"bos_token_id": 4, "eos_token_id": 5, "pad_token_id": 6})
        self.assertEqual((explicit.bos_token_id, explicit.eos_token_id, explicit.pad_token_id),
                         (4, 5, 6))
        nested = self.read({"text_config": {"vocab_size": 64}})
        defaults = DeepSeekV41Config()
        self.assertEqual((nested.bos_token_id, nested.eos_token_id, nested.pad_token_id),
                         (defaults.bos_token_id, defaults.eos_token_id, defaults.pad_token_id))


if __name__ == "__main__":
    unittest.main()
