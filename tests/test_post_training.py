import copy
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch

from dsv41_train.activation import ActivationCheckpointConfig
from dsv41_train.checkpoint import CheckpointManager, TrainingState
from dsv41_train.lora import (
    LoRAConfig, adapter_parameters, inject_lora, load_adapter,
    merge_adapters, save_adapter, unmerge_adapters,
)
from dsv41_train.models.dsv4 import DeepSeekV41ForCausalLM
from dsv41_train.models.dsv4.quantization import (
    _fake_quant_fp4_block, _fake_quant_fp8_block, quantize_activation,
)
from post_training_helpers import small_config


class PostTrainingTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.tokens = torch.randint(3, 32, (2, 12))
        self.mask = torch.ones_like(self.tokens)
        self.mask[0, :2] = 0
        self.mask[1, -3:] = 0

    def assert_gradients(self, actual, expected, scale=1.0):
        for (name, parameter), (ref_name, reference) in zip(
            actual.named_parameters(), expected.named_parameters(),
        ):
            self.assertEqual(name, ref_name)
            if parameter.grad is None or reference.grad is None:
                self.assertIs(parameter.grad, reference.grad, name)
            else:
                torch.testing.assert_close(parameter.grad, reference.grad * scale,
                                           rtol=2e-4, atol=2e-6, msg=name)

    def test_checkpoint_outputs_and_gradients_with_shared_state_padding_and_dropout(self):
        for quantization in ("none", "hf"):
            direct = DeepSeekV41ForCausalLM(small_config(attention_dropout=0.15,
                                                       cache_quantization=quantization))
            for mode in ("full", "selective"):
                with self.subTest(mode=mode, quantization=quantization):
                    candidate = copy.deepcopy(direct)
                    candidate.set_activation_checkpointing(
                        ActivationCheckpointConfig(mode, ("attention", "moe", "hc")),
                    )
                    direct.zero_grad(set_to_none=True)
                    torch.manual_seed(73)
                    expected = direct(self.tokens, self.mask, labels=self.tokens)
                    expected.loss.backward()
                    torch.manual_seed(73)
                    actual = candidate(self.tokens, self.mask, labels=self.tokens)
                    actual.loss.backward()
                    torch.testing.assert_close(actual.logits, expected.logits, rtol=0, atol=0)
                    torch.testing.assert_close(actual.loss, expected.loss, rtol=0, atol=0)
                    self.assert_gradients(candidate, direct)

    def test_loss_scaling_and_accumulation_with_lora_and_checkpointing(self):
        for mode in ("none", "full", "selective"):
            for lora in (False, True):
                with self.subTest(mode=mode, lora=lora):
                    model = DeepSeekV41ForCausalLM(small_config())
                    if lora:
                        inject_lora(model, LoRAConfig())
                    model.set_activation_checkpointing(ActivationCheckpointConfig(mode))
                    scaled, accumulated = copy.deepcopy(model), copy.deepcopy(model)
                    model(self.tokens, self.mask, labels=self.tokens).loss.backward()
                    (scaled(self.tokens, self.mask, labels=self.tokens).loss * 0.5).backward()
                    self.assert_gradients(scaled, model, 0.5)
                    # Two live forward graphs must not share mutable routing or KV state.
                    first = accumulated(self.tokens, self.mask, labels=self.tokens)
                    second = accumulated(self.tokens, self.mask, labels=self.tokens)
                    (first.loss / 2).backward()
                    (second.loss / 2).backward()
                    self.assert_gradients(accumulated, model)

    def test_lora_identity_freezing_and_optimizer_updates(self):
        model = DeepSeekV41ForCausalLM(small_config())
        original = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        expected = model(self.tokens).logits.detach()
        inject_lora(model, LoRAConfig())
        torch.testing.assert_close(model(self.tokens).logits, expected, rtol=0, atol=0)
        adapters = {name: parameter.detach().clone() for name, parameter in model.named_parameters()
                    if "lora_" in name}
        optimizer = torch.optim.AdamW(adapter_parameters(model), lr=0.01, weight_decay=0)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            model(self.tokens, labels=self.tokens).loss.backward()
            for name, parameter in model.named_parameters():
                if name not in adapters:
                    continue
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                if step or name.endswith("lora_b"):
                    self.assertGreater(parameter.grad.abs().max().item(), 0, name)
                else:
                    self.assertEqual(torch.count_nonzero(parameter.grad).item(), 0, name)
            optimizer.step()
        for name, parameter in model.named_parameters():
            if name in original:
                self.assertFalse(parameter.requires_grad)
                self.assertIsNone(parameter.grad)
                torch.testing.assert_close(parameter, original[name], rtol=0, atol=0)
            elif name in adapters:
                self.assertFalse(torch.equal(parameter, adapters[name]), name)

    def test_lora_matches_explicit_dense_update_and_gradients(self):
        model = torch.nn.Sequential(torch.nn.Linear(7, 5, dtype=torch.float64))
        config = LoRAConfig(rank=3, alpha=7, targets=("0",))
        inject_lora(model, config)
        with torch.no_grad():
            model[0].lora_b.normal_(std=0.2)
        inputs = torch.randn(2, 4, 7, dtype=torch.float64, requires_grad=True)
        reference_inputs = inputs.detach().clone().requires_grad_()
        reference_a = model[0].lora_a.detach().clone().requires_grad_()
        reference_b = model[0].lora_b.detach().clone().requires_grad_()
        dense_weight = model[0].weight + (config.alpha / config.rank) * reference_b @ reference_a
        expected = torch.nn.functional.linear(reference_inputs, dense_weight, model[0].bias)
        actual = model(inputs)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        upstream = torch.randn_like(actual)
        actual.backward(upstream)
        expected.backward(upstream)
        for parameter, reference in ((inputs, reference_inputs), (model[0].lora_a, reference_a),
                                     (model[0].lora_b, reference_b)):
            torch.testing.assert_close(parameter.grad, reference.grad, rtol=1e-12, atol=1e-12)
        self.assertIsNone(model[0].weight.grad)
        self.assertIsNone(model[0].bias.grad)

    def test_lora_dropout_checkpoint_gradients_after_adapter_updates(self):
        direct = DeepSeekV41ForCausalLM(small_config(attention_dropout=0.15))
        inject_lora(direct, LoRAConfig(dropout=0.2))
        with torch.no_grad():
            for name, parameter in direct.named_parameters():
                if name.endswith("lora_b"):
                    parameter.normal_(std=0.03)
        for mode in ("full", "selective"):
            with self.subTest(mode=mode):
                candidate = copy.deepcopy(direct)
                candidate.set_activation_checkpointing(
                    ActivationCheckpointConfig(mode, ("attention", "moe", "hc")),
                )
                direct.zero_grad(set_to_none=True)
                torch.manual_seed(73)
                expected = direct(self.tokens, self.mask, labels=self.tokens)
                expected.loss.backward()
                torch.manual_seed(73)
                actual = candidate(self.tokens, self.mask, labels=self.tokens)
                actual.loss.backward()
                torch.testing.assert_close(actual.logits, expected.logits, rtol=0, atol=0)
                torch.testing.assert_close(actual.loss, expected.loss, rtol=0, atol=0)
                self.assert_gradients(candidate, direct)

    def test_adapter_roundtrip_and_merge(self):
        model = DeepSeekV41ForCausalLM(small_config())
        restored = copy.deepcopy(model)
        config = LoRAConfig(dropout=0.1)
        inject_lora(model, config)
        inject_lora(restored, config)
        with torch.no_grad():
            for parameter in adapter_parameters(model):
                parameter.normal_(std=0.03)
        model.eval()
        restored.eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter.pt"
            save_adapter(model, path)
            load_adapter(restored, path)
            self.assertLess(path.stat().st_size, 100000)
        expected = model(self.tokens).logits.detach()
        torch.testing.assert_close(restored(self.tokens).logits, expected, rtol=0, atol=0)
        base = {name: parameter.clone() for name, parameter in restored.named_parameters()
                if name.endswith("weight")}
        merge_adapters(restored)
        torch.testing.assert_close(restored(self.tokens).logits, expected, rtol=2e-5, atol=2e-6)
        unmerge_adapters(restored)
        for name, parameter in restored.named_parameters():
            if name in base:
                torch.testing.assert_close(parameter, base[name], rtol=0, atol=0)
        torch.testing.assert_close(restored(self.tokens).logits, expected, rtol=0, atol=0)
        merge_adapters(restored)
        restored.train()
        self.assertFalse(any(module.merged for module in restored.modules() if hasattr(module, "merged")))

    def test_unsupported_lora_target_does_not_freeze_model(self):
        model = DeepSeekV41ForCausalLM(small_config())
        for target in ("attention.o_a", "missing"):
            with self.assertRaises(ValueError):
                inject_lora(model, LoRAConfig(targets=(target,)))
            self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))

    def test_bfloat16_unmerge_restores_exact_base_weights(self):
        model = torch.nn.Sequential(torch.nn.Linear(64, 32, dtype=torch.bfloat16))
        inject_lora(model, LoRAConfig(targets=("0",)))
        with torch.no_grad():
            model[0].lora_b.normal_(std=0.02)
        model.eval()
        inputs = torch.randn(3, 64, dtype=torch.bfloat16)
        original_weight = model[0].weight.clone()
        expected = model(inputs)
        merge_adapters(model)
        torch.testing.assert_close(model(inputs), expected, rtol=0.03, atol=0.01)
        unmerge_adapters(model)
        torch.testing.assert_close(model[0].weight, original_weight, rtol=0, atol=0)
        torch.testing.assert_close(model(inputs), expected, rtol=0, atol=0)

    def test_quantization_preserves_reference_forward_and_identity_gradient(self):
        for kind, reference in (
            ("fp8", _fake_quant_fp8_block),
            ("fp4_index", lambda x: _fake_quant_fp4_block(x, 32)),
            ("fp4_kv", lambda x: _fake_quant_fp4_block(x, 16, e4m3_scales=True)),
        ):
            with self.subTest(kind=kind):
                values = torch.randn(2, 3, 64, requires_grad=True)
                upstream = torch.randn_like(values)
                output = quantize_activation(values, kind)
                torch.testing.assert_close(output, reference(values.detach()), rtol=0, atol=0)
                output.backward(upstream)
                torch.testing.assert_close(values.grad, upstream, rtol=0, atol=0)

    def test_resume_restores_lora_optimizer_and_dropout_rng(self):
        model = DeepSeekV41ForCausalLM(small_config(attention_dropout=0.1))
        lora = LoRAConfig(dropout=0.2)
        inject_lora(model, lora)
        optimizer = torch.optim.AdamW(adapter_parameters(model), lr=1e-3)
        generator = torch.Generator().manual_seed(17)
        state = TrainingState(model, optimizer, model_config=model.config.to_dict(),
                              data_generator=generator, training_config={"lora": asdict(lora)})

        def step():
            optimizer.zero_grad(set_to_none=True)
            output = model(self.tokens, labels=self.tokens)
            output.loss.backward()
            optimizer.step()
            return output.loss.detach()

        step()
        with tempfile.TemporaryDirectory() as directory:
            manager = CheckpointManager(directory, state)
            manager.save(1)
            expected_loss = step()
            expected = [parameter.detach().clone() for parameter in adapter_parameters(model)]
            self.assertEqual(manager.load(), 1)
            actual_loss = step()
        torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
        for actual, reference in zip(adapter_parameters(model), expected):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
