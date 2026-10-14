"""Exercise trainer LoRA assertions and exact full-state continuation on CPU."""

from pathlib import Path
import tempfile
import unittest

import torch

from dsv41_train.checkpoint import CheckpointManager, TrainingState
from dsv41_train.lora import LoRAConfig, adapter_parameters, inject_lora
from dsv41_train.models.dsv4 import DeepSeekV41ForCausalLM
from dsv41_train.optimization import OptimizerPolicy, build_optimizer
from dsv41_train.runtime import Runtime
from dsv41_train.sharded_state import StateAdapter
from post_training_helpers import small_config
from scripts.verify_lora_adapter import capture_probe
from train_sft import LoRAVerifier, _sample_parameter


class LoRAVerifierTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(173)
        self.runtime = Runtime(torch.device("cpu"))

    def linear(self):
        model = torch.nn.Sequential(torch.nn.Linear(33, 19))
        inject_lora(model, LoRAConfig(rank=3, alpha=6, targets=("0",)))
        optimizer = torch.optim.AdamW(adapter_parameters(model), lr=.01, weight_decay=0)
        return model, optimizer

    def test_first_zero_a_gradient_then_a_and_b_updates_and_bounded_base_samples(self):
        model, optimizer = self.linear()
        verifier = LoRAVerifier(model, optimizer, self.runtime)
        inputs = torch.randn(2, 7, 33)
        base = {name: parameter.detach().clone() for name, parameter in verifier.base}
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            model(inputs).square().mean().backward()
            gradients = verifier.gradients()
            self.assertGreater(gradients["b"]["gradient_nonzero"], 0)
            self.assertEqual(gradients["a"]["gradient_nonzero"] == 0, step == 0)
            optimizer.step()
            metrics = verifier.after_step(gradients)
            self.assertTrue(metrics["base_requires_grad_false"])
            self.assertTrue(metrics["base_grad_none"])
            self.assertEqual(metrics["base_probe_delta_max"], 0)
            self.assertGreater(metrics["b"]["step_parameter_probe_delta_max"], 0)
            self.assertEqual(metrics["a"]["parameter_probe_delta_max"] == 0, step == 0)
        summary = verifier.finish()
        self.assertTrue(summary["verified"])
        self.assertEqual(summary["adapter_parameters_global"], (33 + 19) * 3)
        self.assertEqual(summary["adapter_bytes_global"], (33 + 19) * 3 * 4)
        self.assertGreater(summary["a_parameter_probe_delta_max"], 0)
        self.assertEqual(summary["base_sampled_values_global"], 256 + 19)
        for name, parameter in verifier.base:
            torch.testing.assert_close(parameter, base[name], rtol=0, atol=0)
        for _, _, indices, before in verifier.base_probes:
            self.assertLessEqual(indices.numel(), 256)
            self.assertLessEqual(before.numel(), 256)

    def test_rejects_unfrozen_base_or_optimizer_contamination_omission_and_duplicates(self):
        for invalid in ("unfrozen", "base", "omitted", "duplicate", "frozen_adapter"):
            with self.subTest(invalid=invalid):
                model, optimizer = self.linear()
                group = optimizer.param_groups[0]["params"]
                if invalid == "unfrozen":
                    model[0].weight.requires_grad_(True)
                elif invalid == "base":
                    group.append(model[0].weight)
                elif invalid == "omitted":
                    group.pop()
                elif invalid == "duplicate":
                    group.append(group[0])
                else:
                    model[0].lora_a.requires_grad_(False)
                with self.assertRaisesRegex(RuntimeError, "LoRA verification failed"):
                    LoRAVerifier(model, optimizer, self.runtime)

    def test_catches_base_grad_mutation_and_nonfinite_or_missing_adapter_gradients(self):
        for invalid, message in (("base_grad", "base has gradients"),
                                 ("base_update", "base samples changed"),
                                 ("nan_gradient", "non-finite adapter gradient"),
                                 ("missing_gradient", "adapter gradient is missing"),
                                 ("nan_parameter", "non-finite adapter parameter")):
            with self.subTest(invalid=invalid):
                model, optimizer = self.linear()
                verifier = LoRAVerifier(model, optimizer, self.runtime)
                model(torch.randn(2, 3, 33)).square().mean().backward()
                if invalid == "base_grad":
                    model[0].weight.grad = torch.ones_like(model[0].weight)
                elif invalid == "nan_gradient":
                    model[0].lora_b.grad[0, 0] = float("nan")
                elif invalid == "missing_gradient":
                    model[0].lora_a.grad = None
                if invalid in ("base_grad", "nan_gradient", "missing_gradient"):
                    with self.assertRaisesRegex(RuntimeError, message):
                        verifier.gradients()
                    continue
                gradients = verifier.gradients()
                optimizer.step()
                with torch.no_grad():
                    if invalid == "base_update":
                        model[0].weight[0, 0] += 1
                    else:
                        model[0].lora_b[0, 0] = float("nan")
                with self.assertRaisesRegex(RuntimeError, message):
                    verifier.after_step(gradients)

    def test_no_b_update_fails_even_with_finite_zero_gradients(self):
        model, optimizer = self.linear()
        verifier = LoRAVerifier(model, optimizer, self.runtime)
        model(torch.randn(2, 3, 33)).sum().mul(0).backward()
        gradients = verifier.gradients()
        optimizer.step()
        verifier.after_step(gradients)
        with self.assertRaisesRegex(RuntimeError, "no B parameter update"):
            verifier.finish()

    def test_noncontiguous_base_sampling_does_not_need_a_full_weight_copy(self):
        tensor = torch.arange(1024 * 7).reshape(1024, 7).t()
        self.assertFalse(tensor.is_contiguous())
        offsets = torch.tensor([0, 17, tensor.numel() - 1])
        actual = _sample_parameter(tensor, offsets)
        self.assertEqual(actual.tolist(), [0, 17 * 7, 1024 * 7 - 1])

    def test_dsv4_lora_adamw_dcp_resume_matches_uninterrupted_with_dropout(self):
        config = small_config(num_hidden_layers=1, compress_ratios=[0],
                              kv_source_layer_ids=[], index_source_layer_ids=[],
                              candidate_source_layer_id=-1, engram_layer_ids=[],
                              engram_num_embeddings=[])
        def make_model():
            model = DeepSeekV41ForCausalLM(config)
            self.addCleanup(model.close)
            inject_lora(model, LoRAConfig(rank=2, alpha=4, dropout=.2))
            optimizer = build_optimizer(model, OptimizerPolicy("adamw", .01))
            return model, optimizer
        model, optimizer = make_model()
        verifier = LoRAVerifier(model, optimizer, self.runtime)
        original_base = {name: value.detach().clone() for name, value in verifier.base}
        ids = torch.tensor([[1, 3, 4, 5, 2, 0], [1, 5, 4, 3, 2, 0]])
        labels = ids.masked_fill(ids.eq(0), -100)
        labels[:, :2] = -100
        batch = {"input_ids": ids, "attention_mask": ids.ne(0), "labels": labels}
        def update(current, current_optimizer, checks):
            current_optimizer.zero_grad(set_to_none=True)
            loss = current(ids, attention_mask=ids.ne(0), labels=labels).loss
            loss.backward()
            gradients = checks.gradients()
            current_optimizer.step()
            return loss.detach(), checks.after_step(gradients)
        update(model, optimizer, verifier)
        update(model, optimizer, verifier)
        with tempfile.TemporaryDirectory() as directory:
            def manager(current, current_optimizer):
                return CheckpointManager(Path(directory), TrainingState(
                    current, current_optimizer, model_config=current.config.to_dict(),
                    data_generator=torch.Generator(), training_config={"lora": True},
                    state_adapter=StateAdapter(current)))
            rng_before_eval = torch.get_rng_state().clone()
            capture_probe(model, batch, self.runtime)
            torch.testing.assert_close(torch.get_rng_state(), rng_before_eval, rtol=0, atol=0)
            manager(model, optimizer).save(2)
            expected_losses = [update(model, optimizer, verifier)[0] for _ in range(2)]
            restored, restored_optimizer = make_model()
            self.assertEqual(manager(restored, restored_optimizer).load(), 2)
            capture_probe(restored, batch, self.runtime)
            resumed_verifier = LoRAVerifier(restored, restored_optimizer, self.runtime)
            self.assertEqual(set(resumed_verifier.maxima.values()), {0.0})
            for expected in expected_losses:
                loss, metrics = update(restored, restored_optimizer, resumed_verifier)
                torch.testing.assert_close(loss, expected, rtol=0, atol=0)
                self.assertGreater(metrics["a"]["gradient_nonzero"], 0)
            for (name, parameter), (restored_name, restored_parameter) in zip(
                    model.named_parameters(), restored.named_parameters()):
                self.assertEqual(name, restored_name)
                torch.testing.assert_close(parameter, restored_parameter, rtol=0, atol=0)
                if name in original_base:
                    torch.testing.assert_close(parameter, original_base[name], rtol=0, atol=0)
            expected_state = optimizer.state_dict()["state"]
            for key, state in restored_optimizer.state_dict()["state"].items():
                for name, value in state.items():
                    torch.testing.assert_close(value, expected_state[key][name], rtol=0, atol=0)
            self.assertTrue(resumed_verifier.finish()["verified"])


if __name__ == "__main__":
    unittest.main()
