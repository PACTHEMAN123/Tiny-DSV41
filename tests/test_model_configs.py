import json
import subprocess
import sys
import tempfile
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

    def test_dsv4_checkpoint_metadata_is_shared_by_native_and_rl_loaders(self):
        from dsv41_train.rl.policy import DSV41Config

        path = ROOT / "dsv41_train/models/dsv4/config.json"
        native = DeepSeekV41Config.from_json(path)
        rl = DSV41Config.from_model_dir(path.parent)
        self.assertEqual((native.bos_token_id, native.eos_token_id, native.pad_token_id), (0, 1, 2))
        self.assertEqual(len(native.compress_ratios), native.num_hidden_layers)
        self.assertEqual(native.engram_pad_id, 2)
        self.assertEqual(native.to_dict(), {key: rl.to_dict()[key] for key in native.to_dict()})

    def test_dsv4_missing_structure_never_falls_back_to_a_preset(self):
        from dsv41_train.rl.policy import DSV41Config

        original = json.loads((ROOT / "examples/rl/tiny-dsv41/config.json").read_text())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            for name in ("hidden_size", "num_hidden_layers", "n_routed_experts",
                         "compress_ratios", "kv_source_layer_ids", "engram_head_dim"):
                values = dict(original)
                del values[name]
                path.write_text(json.dumps(values))
                for loader in (lambda: DeepSeekV41Config.from_json(path),
                               lambda: DSV41Config.from_model_dir(path.parent)):
                    with self.subTest(field=name, loader=loader):
                        with self.assertRaisesRegex(ValueError, name):
                            loader()

    def test_dsv4_compression_layout_only_accepts_declared_mtp_suffix(self):
        values = json.loads((ROOT / "examples/rl/tiny-dsv41/config.json").read_text())
        ratios = values["compress_ratios"][:]
        values["compress_ratios"] += [0, 0]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            path.write_text(json.dumps(values))
            with self.assertRaisesRegex(ValueError, "compress_ratios"):
                DeepSeekV41Config.from_json(path)
            values["num_nextn_predict_layers"] = 2
            path.write_text(json.dumps(values))
            config = DeepSeekV41Config.from_json(path)
            self.assertEqual(config.compress_ratios, ratios)

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
