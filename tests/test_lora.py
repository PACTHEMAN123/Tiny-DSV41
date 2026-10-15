import copy
import os
import shutil
import tempfile
import unittest
import warnings
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn

from dsv41_train.lora import (
    LoRAConfig, LoRALinear, adapter_base_identity, adapter_parameters, inject_lora, load_adapter,
    load_adapter_state, merge_adapters, read_adapter, save_adapter, unmerge_adapters,
)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layer_ids = [2, 3]
        self.model.layers = nn.ModuleList([nn.ModuleDict({"proj": nn.Linear(4, 4)}) for _ in range(2)])
        self.head = nn.Linear(4, 2)

    def forward(self, inputs):
        for layer in self.model.layers:
            inputs = layer["proj"](inputs).tanh()
        return self.head(inputs)


class LoRATest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.config = LoRAConfig(rank=2, alpha=4, targets=("proj",))

    def test_identity_freezing_and_optimizer_updates(self):
        model = TinyModel()
        inputs = torch.randn(3, 4)
        before = {name: value.detach().clone() for name, value in model.named_parameters()}
        base_parameters = dict(model.named_parameters())
        expected = model(inputs).detach()
        inject_lora(model, self.config)
        torch.testing.assert_close(model(inputs), expected, rtol=0, atol=0)
        for name, value in base_parameters.items():
            self.assertIs(value, model.get_parameter(name))
        optimizer = torch.optim.AdamW(adapter_parameters(model), lr=.01, weight_decay=0)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            model(inputs).square().mean().backward()
            for name, parameter in model.named_parameters():
                if "lora_" in name:
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                    if step or name.endswith("lora_b"):
                        self.assertGreater(float(parameter.grad.abs().max()), 0)
                    else:
                        self.assertEqual(int(parameter.grad.count_nonzero()), 0)
                else:
                    self.assertFalse(parameter.requires_grad)
                    self.assertIsNone(parameter.grad)
            optimizer.step()
        for name, value in before.items():
            torch.testing.assert_close(model.get_parameter(name), value, rtol=0, atol=0)
        self.assertFalse(torch.equal(model(inputs), expected))

    def test_explicit_dense_formula_and_input_adapter_gradients(self):
        layer = LoRALinear(nn.Linear(7, 5, dtype=torch.float64), LoRAConfig(rank=3, alpha=7))
        with torch.no_grad():
            layer.lora_b.normal_()
        inputs = torch.randn(2, 7, dtype=torch.float64, requires_grad=True)
        reference = [value.detach().clone().requires_grad_() for value in (inputs, layer.lora_a, layer.lora_b)]
        x, a, b = reference
        expected = nn.functional.linear(x, layer.weight + layer.scaling * (b @ a), layer.bias)
        actual = layer(inputs)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        actual.square().sum().backward()
        expected.square().sum().backward()
        for value, target in zip((inputs, layer.lora_a, layer.lora_b), reference):
            torch.testing.assert_close(value.grad, target.grad, rtol=1e-12, atol=1e-12)

    def test_targets_are_validated_before_mutating_the_model(self):
        model = TinyModel()
        for targets in (("proj", "missing"), ("model.layers",)):
            with self.subTest(targets=targets), self.assertRaises(ValueError):
                inject_lora(model, LoRAConfig(targets=targets))
            self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))
            self.assertFalse(any(isinstance(module, LoRALinear) for module in model.modules()))
        with self.assertRaisesRegex(ValueError, "after loading"):
            inject_lora(nn.ModuleDict({"proj": nn.Linear(2, 2, device="meta")}), self.config)
        inject_lora(model, self.config)
        with self.assertRaisesRegex(ValueError, "already installed"):
            inject_lora(model, self.config)

    def test_dsv4_default_and_qwen_configurable_target_names(self):
        for prefix, names in (("attention", ("q_a", "q_b", "kv_proj", "o_b")),
                              ("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj"))):
            model = nn.ModuleDict({prefix: nn.ModuleDict({name: nn.Linear(4, 4) for name in names})})
            config = LoRAConfig() if prefix == "attention" else LoRAConfig(
                targets=tuple(f"{prefix}.{name}" for name in names))
            self.assertEqual(set(inject_lora(model, config)), set(config.targets))

    def test_portable_roundtrip_and_layerwise_restore(self):
        model = TinyModel()
        restored = copy.deepcopy(model)
        inject_lora(model, self.config)
        inject_lora(restored, self.config)
        with torch.no_grad():
            for parameter in adapter_parameters(model):
                parameter.normal_(std=.1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter.pt"
            save_adapter(model, path)
            payload = read_adapter(path)
            self.assertEqual(payload["layer_ids"], [2, 3])
            self.assertEqual(len(payload["weights"]), 4)
            self.assertTrue(all(name.endswith(("lora_a", "lora_b")) for name in payload["weights"]))
            self.assertEqual(payload["format_version"], 1)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", RuntimeWarning)
                load_adapter(restored, path)
            self.assertTrue(any(issubclass(item.category, RuntimeWarning)
                                and "base weights and tokenizer cannot be verified" in str(item.message)
                                for item in caught))
            inputs = torch.randn(2, 4)
            torch.testing.assert_close(restored(inputs), model(inputs), rtol=0, atol=0)
            for index, layer in enumerate(restored.model.layers):
                with torch.no_grad():
                    for parameter in adapter_parameters(layer):
                        parameter.zero_()
                load_adapter_state(layer, payload, prefix=f"model.layers.{index}")
            torch.testing.assert_close(restored(inputs), model(inputs), rtol=0, atol=0)
            restored.model.layer_ids = [0, 1]
            with self.assertRaisesRegex(ValueError, "layer window"):
                load_adapter(restored, path)

    def test_bound_adapter_rejects_source_changes_before_mutation(self):
        model, restored = TinyModel(), TinyModel()
        for item in (model, restored):
            inject_lora(item, self.config)
        before = copy.deepcopy(restored.state_dict())
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "base"
            folder.mkdir()
            (folder / "config.json").write_text('{"hidden_size":4}')
            (folder / "tokenizer.json").write_text('{"vocab":["a"]}')
            torch.save(TinyModel().state_dict(), folder / "pytorch_model.bin")
            identity = adapter_base_identity(folder)
            path = folder / "adapter.pt"
            save_adapter(model, path, base_identity=identity)
            payload = read_adapter(path)
            self.assertEqual(payload["format_version"], 2)
            self.assertEqual(payload["base_identity"], identity)
            for changed in (None, "config.json", "tokenizer.json", "pytorch_model.bin"):
                wrong = None
                if changed:
                    other = Path(directory) / changed
                    shutil.copytree(folder, other)
                    self.assertEqual(adapter_base_identity(other), identity)
                    if changed == "pytorch_model.bin":
                        state = torch.load(other / changed, weights_only=True)
                        state["head.weight"].add_(1)  # Same architecture, different frozen base weights.
                        torch.save(state, other / changed)
                        stat = (folder / changed).stat()
                        os.utime(other / changed, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
                    else:
                        (other / changed).write_text('{"hidden_size":4,"revision":2}' if changed == "config.json"
                                                    else '{"vocab":["b"]}')
                    wrong = adapter_base_identity(other)
                with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, "base identity"):
                    load_adapter(restored, path, base_identity=wrong)
                for name, parameter in restored.state_dict().items():
                    torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
            # Inspecting a payload is allowed; applying it directly or one layer
            # at a time must enforce the same binding as the file loader.
            with self.assertRaisesRegex(ValueError, "base identity"):
                load_adapter_state(restored, payload)
            for index, layer in enumerate(restored.model.layers):
                load_adapter_state(layer, payload, prefix=f"model.layers.{index}", base_identity=identity)
            load_adapter(restored, path, base_identity=identity)
            for name, value in payload["weights"].items():
                torch.testing.assert_close(restored.get_parameter(name), value, rtol=0, atol=0)
            self.assertNotEqual(adapter_base_identity(folder, random_init=True, seed=1),
                                adapter_base_identity(folder, random_init=True, seed=2))

    def test_adapter_validation_is_atomic(self):
        model = TinyModel()
        inject_lora(model, self.config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter.pt"
            save_adapter(model, path)
            payload = read_adapter(path)
        names = list(payload["weights"])
        before = [parameter.detach().clone() for parameter in adapter_parameters(model)]
        for failure in ("missing", "extra", "shape", "nonfinite", "config"):
            bad = copy.deepcopy(payload)
            bad["weights"][names[0]].fill_(99)
            if failure == "missing":
                del bad["weights"][names[-1]]
            elif failure == "extra":
                bad["weights"]["unexpected.lora_a"] = torch.ones(1)
            elif failure == "shape":
                bad["weights"][names[-1]] = torch.zeros(1)
            elif failure == "nonfinite":
                bad["weights"][names[-1]].fill_(float("nan"))
            else:
                next(iter(bad["modules"].values()))["alpha"] = 3
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                load_adapter_state(model, bad)
            for value, expected in zip(adapter_parameters(model), before):
                torch.testing.assert_close(value, expected, rtol=0, atol=0)

    def test_single_linear_adapter_roundtrip_and_merged_load_guard(self):
        layer = LoRALinear(nn.Linear(4, 4), LoRAConfig(rank=2)).eval()
        restored = copy.deepcopy(layer)
        with torch.no_grad():
            layer.lora_b.fill_(.2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter.pt"
            save_adapter(layer, path)
            load_adapter(restored, path)
            inputs = torch.randn(2, 4)
            torch.testing.assert_close(restored(inputs), layer(inputs), rtol=0, atol=0)
            restored.merge()
            with self.assertRaisesRegex(ValueError, "unmerge"):
                load_adapter(restored, path)

    def test_merged_state_dict_roundtrip_and_bfloat16_unmerge(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                layer = LoRALinear(nn.Linear(7, 5, dtype=dtype), LoRAConfig(rank=2)).eval()
                with torch.no_grad():
                    layer.lora_b.normal_(std=.1)
                inputs = torch.randn(2, 7, dtype=dtype)
                base = layer.weight.detach().clone()
                expected = layer(inputs).detach()
                restored = copy.deepcopy(layer)
                merge_adapters(layer)
                state = copy.deepcopy(layer.state_dict())
                torch.testing.assert_close(state["weight"], base, rtol=0, atol=0)
                restored.load_state_dict(state)
                torch.testing.assert_close(restored(inputs), expected, rtol=0, atol=0)
                layer.load_state_dict(state)
                self.assertFalse(layer.merged)
                torch.testing.assert_close(layer(inputs), expected, rtol=0, atol=0)
                merge_adapters(layer)
                unmerge_adapters(layer)
                torch.testing.assert_close(layer.weight, base, rtol=0, atol=0)
                merge_adapters(layer)
                layer.train()
                self.assertFalse(layer.merged)

    def test_adapter_bypass_including_merged_eval(self):
        layer = LoRALinear(nn.Linear(4, 4), LoRAConfig(rank=2, dropout=.5)).eval()
        with torch.no_grad():
            layer.lora_b.normal_()
        inputs = torch.randn(3, 4)
        expected = nn.functional.linear(inputs, layer.weight, layer.bias).detach()
        for merged in (False, True):
            if merged:
                layer.merge()
            layer.adapters_enabled = False
            torch.testing.assert_close(layer(inputs), expected, rtol=0, atol=0)
            layer.adapters_enabled = True
            self.assertFalse(torch.equal(layer(inputs), expected))

    def test_configuration_rejects_invalid_rank_scaling_dropout_and_targets(self):
        for values in ({"rank": 1.5}, {"rank": True}, {"rank": 0}, {"alpha": float("inf")},
                       {"alpha": 0}, {"dropout": 1}, {"dropout": float("nan")}, {"targets": "proj"}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                LoRAConfig(**values)

    def test_dsv4_shared_forward_checkpointed_training_and_reload(self):
        from dsv41_train.models.dsv4 import DeepSeekV41Config, DeepSeekV41ForCausalLM

        previous_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous_threads)
        torch.set_num_threads(1)
        config = replace(DeepSeekV41Config.tiny(), hidden_size=16, num_attention_heads=2,
                         head_dim=8, q_lora_rank=8, qk_rope_head_dim=4, index_n_heads=2,
                         index_head_dim=8, o_groups=1, o_lora_rank=8, moe_intermediate_size=32,
                         n_routed_experts=4, hc_mult=2, hc_sinkhorn_iters=5, engram_head_dim=4)
        model = DeepSeekV41ForCausalLM(config)
        restored = copy.deepcopy(model)
        tokens = torch.randint(3, config.vocab_size, (2, 10))
        labels = tokens.clone()
        labels[:, :5] = -100
        expected = model(tokens, labels=labels, compute_aux_loss=False).loss.detach()
        base = {name: value.detach().clone() for name, value in model.named_parameters()}
        lora = LoRAConfig(rank=2, alpha=4)
        inject_lora(model, lora)
        model.gradient_checkpointing_enable()
        torch.testing.assert_close(model(tokens, labels=labels, compute_aux_loss=False).loss,
                                   expected, rtol=0, atol=0)
        before = {name: value.detach().clone() for name, value in model.named_parameters() if value.requires_grad}
        optimizer = torch.optim.AdamW(adapter_parameters(model), lr=.01, weight_decay=0)
        calls = []
        handle = model.model.layers[0].register_forward_pre_hook(lambda module, inputs: calls.append(1))
        self.addCleanup(handle.remove)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            calls.clear()
            output = model(tokens, labels=labels, compute_aux_loss=False)
            self.assertIsNone(output.aux_loss)
            output.loss.backward()
            self.assertGreater(len(calls), 1, "decoder activation checkpoint did not recompute")
            for name, parameter in model.named_parameters():
                if name in before:
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                    if step or name.endswith("lora_b"):
                        self.assertGreater(float(parameter.grad.abs().max()), 0, name)
                else:
                    self.assertFalse(parameter.requires_grad, name)
                    self.assertIsNone(parameter.grad, name)
            optimizer.step()
        for name, parameter in model.named_parameters():
            if name in base:
                torch.testing.assert_close(parameter, base[name], rtol=0, atol=0, msg=name)
            else:
                self.assertFalse(torch.equal(parameter, before[name]), name)
        inject_lora(restored, lora)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter.pt"
            save_adapter(model, path)
            load_adapter(restored, path)
        model.eval()
        restored.eval()
        actual = model(tokens, labels=labels, compute_aux_loss=False)
        loaded = restored(tokens, labels=labels, compute_aux_loss=False)
        torch.testing.assert_close(actual.loss, loaded.loss, rtol=0, atol=0)
        torch.testing.assert_close(actual.logits, loaded.logits, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
