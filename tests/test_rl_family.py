import json
import tempfile
import unittest
from pathlib import Path

from dsv41_train.rl.family import load_model_config
from dsv41_train.rl.policy import DSV41Config


ROOT = Path(__file__).resolve().parents[1]


class ModelFamilyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dsv = json.loads((ROOT / "examples/rl/tiny-dsv41/config.json").read_text())
        self.qwen = json.loads((ROOT / "dsv41_train/models/qwen/config.json").read_text())

    def load(self, values):
        (self.root / "config.json").write_text(json.dumps(values), encoding="utf-8")
        return load_model_config(self.root)

    def test_dsv_flat_nested_and_legacy_types_remain_compatible(self):
        legacy = {name: value for name, value in self.dsv.items() if name != "model_type"}
        cases = [legacy, self.dsv, self.dsv | {"model_type": "deepseek_v41_text"},
                 {"model_type": "deepseek_v41", "text_config": self.dsv | {"model_type": "deepseek_v41_text"}}]
        released = json.loads((ROOT / "dsv41_train/models/dsv4/config.json").read_text())
        for values in cases + [released]:
            with self.subTest(root_type=values.get("model_type"), nested="text_config" in values):
                config = self.load(values)
                self.assertIsInstance(config, DSV41Config)
                self.assertEqual(config.hidden_size, values.get("text_config", values)["hidden_size"])

    def test_unknown_root_or_nested_type_never_maps_to_dsv(self):
        cases = [self.dsv | {"model_type": "llama"},
                 {"model_type": "deepseek_v41", "text_config": self.dsv | {"model_type": "qwen3"}},
                 {"model_type": "other_wrapper", "text_config": self.dsv}]
        for values in cases:
            with self.subTest(values=values.get("model_type")):
                with self.assertRaisesRegex(ValueError, "unsupported .*model_type"):
                    self.load(values)

    def test_nested_qwen_cannot_silently_select_default_architecture(self):
        for values in ({"text_config": self.qwen}, self.qwen | {"text_config": self.qwen}):
            with self.assertRaisesRegex(ValueError, "flat qwen3_moe"):
                self.load(values)
        config = self.load(self.qwen)
        self.assertEqual(config.model_type, "qwen3_moe")
        self.assertEqual(config.hidden_size, self.qwen["hidden_size"])

    def test_unsupported_native_qwen_metadata_fails_before_weight_loading(self):
        cases = [{"quantization_config": {"quant_method": "fp8"}}, {"attention_dropout": 0.1},
                 {"attention_bias": True}, {"hidden_act": "gelu"},
                 {"rope_scaling": {"rope_type": "linear", "factor": 2}},
                 {"use_sliding_window": True, "sliding_window": 128}, {"tie_word_embeddings": True}]
        for fields in cases:
            with self.subTest(fields=fields):
                with self.assertRaises(ValueError):
                    self.load(self.qwen | fields)


if __name__ == "__main__":
    unittest.main()
