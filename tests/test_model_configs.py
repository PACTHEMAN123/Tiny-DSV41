import json
import subprocess
import sys
import unittest
from pathlib import Path

from dsv41_train.models.dsv4 import DeepSeekV41Config
from dsv41_train.models.qwen import Qwen3MoeConfig


ROOT = Path(__file__).resolve().parents[1]


class ModelConfigTest(unittest.TestCase):
    def test_imports_keep_framework_and_model_families_independent(self):
        cases = (
            (
                "import dsv41_train.checkpoint, dsv41_train.dispatch, "
                "dsv41_train.parallel, dsv41_train.runtime",
                ("dsv41_train.models",),
            ),
            ("import dsv41_train.models.dsv4", ("dsv41_train.models.qwen",)),
            ("import dsv41_train.models.qwen", ("dsv41_train.models.dsv4",)),
        )
        for statement, forbidden in cases:
            with self.subTest(statement=statement):
                result = subprocess.run(
                    [sys.executable, "-c", f"{statement}; import json, sys; print(json.dumps(list(sys.modules)))"],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                loaded = json.loads(result.stdout)
                self.assertFalse([
                    name for name in loaded
                    if any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden)
                ])

    def test_dsv4_config_remains_loadable_after_model_reorganization(self):
        config = DeepSeekV41Config.from_json(
            ROOT / "dsv41_train" / "models" / "dsv4" / "config.json"
        )

        self.assertEqual(config.hidden_size, 5120)
        self.assertEqual(config.num_hidden_layers, 40)

    def test_qwen_config_matches_the_packaged_reference(self):
        path = ROOT / "dsv41_train" / "models" / "qwen" / "config.json"
        config = Qwen3MoeConfig.from_json(path)

        with path.open(encoding="utf-8") as file:
            reference = json.load(file)

        self.assertEqual(config.to_dict(), reference)
        self.assertEqual(config.model_type, "qwen3_moe")
        self.assertEqual(config.num_experts, 128)
        self.assertEqual(config.num_experts_per_tok, 8)

    def test_qwen_config_rejects_more_selected_experts_than_available(self):
        with self.assertRaisesRegex(ValueError, "num_experts_per_tok"):
            Qwen3MoeConfig(num_experts=4, num_experts_per_tok=8)


if __name__ == "__main__":
    unittest.main()
