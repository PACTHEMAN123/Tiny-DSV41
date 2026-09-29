import json
import tempfile
import unittest
import weakref
from pathlib import Path
from unittest.mock import patch

import torch

from dsv41_train.lora import LoRAConfig, adapter_parameters, save_adapter
from dsv41_train.models.dsv4 import DeepSeekV41ForCausalLM, load_dsv41_backbone_window
from dsv41_train.models.dsv4.checkpoint import ShardedSafeTensorReader
from post_training_helpers import small_config
from test_dsv4_checkpoint import write_safetensors


def synthetic_checkpoint(folder: Path, config=None):
    if config is None:
        config = small_config(q_lora_rank=32, o_lora_rank=32, engram_layer_ids=[], engram_num_embeddings=[])
    template = DeepSeekV41ForCausalLM(config)
    tensors = {
        "embed.weight": template.model.embedding.weight.detach(),
        "norm.weight": template.model.norm.weight.detach(),
        "head.weight": template.lm_head.weight.detach(),
    }
    expected = {"model.embedding.weight": tensors["embed.weight"],
                "model.norm.weight": tensors["norm.weight"], "lm_head.weight": tensors["head.weight"]}

    def add(source, target, tensor, quantized=False, fp4=False):
        tensor = tensor.detach()
        if fp4:
            rows, cols = tensor.shape
            tensors[source + ".weight"] = torch.full((rows, cols // 2), 0x21, dtype=torch.int8)
            tensors[source + ".scale"] = torch.full((rows, cols // 32), 0.01)
            value = torch.tensor([0.005, 0.01] * (cols // 2)).expand(rows, -1).clone()
        elif quantized:
            value = tensor.reshape(-1, tensor.shape[-1]).to(torch.float8_e4m3fn)
            tensors[source + ".weight"] = value
            tensors[source + ".scale"] = torch.ones((value.shape[0] + 31) // 32, (value.shape[1] + 31) // 32)
            value = value.float().reshape_as(tensor)
        else:
            tensors[source] = tensor
            value = tensor
        expected[target] = value

    for layer in template.model.layers:
        prefix, target = f"layers.{layer.layer_id}", f"model.layers.{layer.layer_id}"
        state = layer.state_dict()
        mapping = {
            "attention_hc.base": "hc_attn_base", "attention_hc.fn": "hc_attn_fn",
            "attention_hc.scale": "hc_attn_scale", "moe_hc.base": "hc_ffn_base",
            "moe_hc.fn": "hc_ffn_fn", "moe_hc.scale": "hc_ffn_scale",
            "attention.sinks.weight": "attn.attn_sink",
            "attention.q_norm.weight": "attn.q_norm.weight",
            "attention.kv_norm.weight": "attn.kv_norm.weight",
            "input_norm.weight": "attn_norm.weight", "post_attention_norm.weight": "ffn_norm.weight",
            "moe.router.weight": "ffn.gate.weight", "moe.router.selection_bias": "ffn.gate.bias",
        }
        quantized = {
            "attention.q_a.weight": "attn.wq_a", "attention.q_b.weight": "attn.wq_b",
            "attention.kv_proj.weight": "attn.wkv", "attention.o_a.weight": "attn.wo_a",
            "attention.o_b.weight": "attn.wo_b", "moe.shared.gate.weight": "ffn.shared_experts.w1",
            "moe.shared.up.weight": "ffn.shared_experts.w3", "moe.shared.down.weight": "ffn.shared_experts.w2",
        }
        if layer.attention.compressor is not None:
            mapping.update({"attention.compressor.kv_proj.weight": "attn.compressor.wkv.weight",
                            "attention.compressor.norm.weight": "attn.compressor.norm.weight"})
            if layer.attention.compressor.gate_proj is not None:
                mapping["attention.compressor.gate_proj.weight"] = "attn.compressor.wgate.weight"
        if layer.attention.indexer is not None:
            quantized["attention.indexer.q_proj.weight"] = "attn.indexer.wq_b"
            mapping["attention.indexer.weight_proj.weight"] = "attn.indexer.weights_proj.weight"
            if layer.attention.indexer.owns_keys:
                mapping.update({"attention.indexer.k_proj.weight": "attn.indexer.wk.weight",
                                "attention.indexer.k_norm.weight": "attn.indexer.k_norm.weight"})
        if layer.engram is not None:
            mapping.update({"engram.q_weight": "engram.q_weight", "engram.k_weight": "engram.k_weight"})
            quantized["engram.proj.weight"] = "engram.wkv"
            table = template.model.engram_tables[str(layer.layer_id)].weight.detach()
            value = table.to(torch.float8_e4m3fn)
            tensors[f"{prefix}.engram.embed.weight"] = value
            tensors[f"{prefix}.engram.embed.scale"] = torch.ones(table.shape[0], table.shape[1] // 32)
            expected[f"model.engram_tables.{layer.layer_id}.weight"] = value.float()
        for name, source in mapping.items():
            add(f"{prefix}.{source}", f"{target}.{name}", state[name])
        for name, source in quantized.items():
            add(f"{prefix}.{source}", f"{target}.{name}", state[name], quantized=True)
        for index in range(config.n_routed_experts):
            expert = f"{prefix}.ffn.experts.{index}"
            weight = state[f"moe.routed.gate_up.{index}"]
            add(expert + ".w1", "gate", weight[:config.moe_intermediate_size], fp4=True)
            add(expert + ".w3", "up", weight[config.moe_intermediate_size:], fp4=True)
            expected[f"{target}.moe.routed.gate_up.{index}"] = torch.cat((expected.pop("gate"), expected.pop("up")))
            add(expert + ".w2", f"{target}.moe.routed.down.{index}", state[f"moe.routed.down.{index}"], fp4=True)
    write_safetensors(folder / "model.safetensors", tensors)
    (folder / "config.json").write_text(json.dumps(config.to_dict()), encoding="utf-8")
    (folder / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "model.safetensors" for name in tensors}}), encoding="utf-8",
    )
    if config.engram_layer_ids:
        from tokenizers import Tokenizer, models
        tokenizer = Tokenizer(models.WordLevel({str(i): i for i in range(config.vocab_size)}))
        tokenizer.save(str(folder / "tokenizer.json"))
    return config, expected


class LoadingTest(unittest.TestCase):
    def test_loading_releases_temporaries_before_callback_and_preserves_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            config, expected = synthetic_checkpoint(folder)
            references = []
            read = ShardedSafeTensorReader.tensor

            def tracked(reader, name, device):
                tensor = read(reader, name, device)
                # Buffers intentionally retain the original tensor object after assign=True.
                if name.startswith("layers.") and not name.endswith("ffn.gate.bias"):
                    references.append(weakref.ref(tensor))
                return tensor

            callbacks = []

            def loaded(layer):
                self.assertTrue(all(reference() is None for reference in references),
                                "layer loading retained temporary tensors across the shard callback")
                callbacks.append(layer.layer_id)

            with patch.object(ShardedSafeTensorReader, "tensor", tracked):
                model = load_dsv41_backbone_window(folder, num_layers=5, dtype=torch.float32,
                                                   layer_loaded=loaded)
            self.assertEqual(callbacks, list(range(5)))
            for name, tensor in model.state_dict().items():
                torch.testing.assert_close(tensor, expected[name], rtol=1e-6, atol=1e-8, msg=name)
            tokens = torch.randint(3, config.vocab_size, (1, 12))
            model(tokens, labels=tokens).loss.backward()
            self.assertTrue(torch.isfinite(model.model.embedding.weight.grad).all())

    def test_loader_injects_adapters_before_sharding_and_freezes_root(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            synthetic_checkpoint(folder)
            seen = []

            def loaded(layer):
                seen.append(layer.attention.q_a.lora_b.shape)
                self.assertFalse(layer.attention.q_a.weight.requires_grad)

            model = load_dsv41_backbone_window(folder, num_layers=3, dtype=torch.float32,
                                               lora_config=LoRAConfig(), layer_loaded=loaded,
                                               cache_quantization="hf")
            self.assertEqual(len(seen), 3)
            self.assertEqual(model.config.cache_quantization, "hf")
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
