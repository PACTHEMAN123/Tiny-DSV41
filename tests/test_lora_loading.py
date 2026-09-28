"""Adapter injection and restore must finish before a layer is sharded."""

import tempfile
import unittest
from pathlib import Path

import torch

from dsv41_train.lora import LoRAConfig, adapter_parameters, save_adapter
from dsv41_train.models.dsv4 import load_dsv41_backbone_window
from lora_helpers import synthetic_checkpoint


class LoRALoadingTest(unittest.TestCase):
    def test_loader_injects_adapters_before_sharding_and_freezes_root(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            synthetic_checkpoint(folder)
            seen = []

            def loaded(layer):
                seen.append(layer.attention.q_a.lora_b.shape)
                self.assertFalse(layer.attention.q_a.weight.requires_grad)

            model = load_dsv41_backbone_window(folder, num_layers=3, dtype=torch.float32,
                                               lora_config=LoRAConfig(), layer_loaded=loaded)
            self.assertEqual(len(seen), 3)
            for name, parameter in model.named_parameters():
                self.assertEqual(parameter.requires_grad, "lora_" in name, name)

    def test_adapter_is_restored_before_each_layer_callback(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            synthetic_checkpoint(folder)
            config = LoRAConfig()
            original = load_dsv41_backbone_window(folder, num_layers=3, dtype=torch.float32,
                                                  lora_config=config)
            with torch.no_grad():
                for parameter in adapter_parameters(original):
                    parameter.fill_(0.125)
            path = folder / "adapter.pt"
            save_adapter(original, path)
            seen = []

            def loaded(layer):
                for parameter in adapter_parameters(layer):
                    torch.testing.assert_close(parameter, torch.full_like(parameter, 0.125))
                seen.append(layer.layer_id)

            restored = load_dsv41_backbone_window(folder, num_layers=3, dtype=torch.float32,
                                                  lora_config=config, adapter_path=path,
                                                  layer_loaded=loaded)
            self.assertEqual(seen, [0, 1, 2])
            tokens = torch.randint(3, 32, (1, 12))
            torch.testing.assert_close(restored(tokens).logits, original(tokens).logits, rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, "layer window"):
                load_dsv41_backbone_window(folder, num_layers=2, lora_config=config, adapter_path=path)


if __name__ == "__main__":
    unittest.main()
