"""Qwen LoRA over real tiny HF checkpoints, including exact resume and fresh reload."""

import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import torch

from dsv41_train.checkpoint import CheckpointManager, TrainingState
from dsv41_train.dispatch import TokenDispatcher
from dsv41_train.lora import LoRAConfig, LoRALinear, save_adapter
from dsv41_train.models.qwen import Qwen3MoeForCausalLM, load_qwen3_moe
from dsv41_train.models.qwen.checkpoint import QWEN_LORA_TARGETS
from dsv41_train.optimization import OptimizerPolicy, build_optimizer
from dsv41_train.parallel import ContextParallel
from dsv41_train.runtime import Runtime
from dsv41_train.sharded_state import StateAdapter
from qwen_helpers import tiny_config, write_hf_checkpoint
from scripts.compare_lora_runs import compare_runs
from scripts.verify_lora_adapter import capture_probe, compare_probes, make_reference, verify
from train_sft import LoRAVerifier, _source_identity


class QwenLoRATest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(217)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.base = self.folder / "base"
        self.base.mkdir()
        self.source = Qwen3MoeForCausalLM(tiny_config(mlp_only_layers=[0]))
        write_hf_checkpoint(self.base, self.source)
        self.lora = LoRAConfig(rank=2, alpha=4, dropout=.1, targets=QWEN_LORA_TARGETS)
        self.runtime = Runtime(torch.device("cpu"))
        ids = torch.tensor([[1, 4, 3, 5, 2, 0], [1, 3, 5, 4, 2, 0]])
        labels = ids.masked_fill(ids.eq(0), -100)
        labels[:, :2] = -100
        self.batch = {"input_ids": ids, "labels": labels, "attention_mask": ids.ne(0)}

    def load(self, **kwargs):
        return load_qwen3_moe(self.base, dtype=torch.bfloat16, lora_config=self.lora, **kwargs)

    def test_zero_b_preserves_pretrained_forward_and_injection_precedes_sharding_callback(self):
        callbacks = []
        def materialize(name, module):
            callbacks.append(name)
            if name.startswith("model.layers."):
                self.assertIsInstance(module.self_attn.q_proj, LoRALinear)
                self.assertEqual(sum(isinstance(child, LoRALinear) for child in module.modules()), 4)
                self.assertTrue(module.self_attn.q_proj.lora_a.requires_grad)
                self.assertFalse(module.self_attn.q_proj.weight.requires_grad)
            else:
                self.assertTrue(all(not parameter.requires_grad for parameter in module.parameters()))
        loaded = self.load(module_callback=materialize)
        baseline = load_qwen3_moe(self.base, dtype=torch.bfloat16)
        self.assertEqual(len(callbacks), 5)
        self.assertEqual(loaded.model.layer_ids, [0, 1])
        self.assertTrue(compare_probes(capture_probe(loaded, self.batch, self.runtime),
                                      capture_probe(baseline, self.batch, self.runtime), atol=0, rtol=0)["passed"])
        expected = dict(baseline.named_parameters())
        for name, parameter in loaded.named_parameters():
            if name in expected:
                self.assertFalse(parameter.requires_grad, name)
                torch.testing.assert_close(parameter, expected[name], rtol=0, atol=0)
            else:
                self.assertTrue(parameter.requires_grad, name)
                self.assertTrue(name.endswith(("lora_a", "lora_b")), name)

    def test_invalid_adapter_layout_and_payload_fail_before_any_sharding_callback(self):
        model = self.load()
        adapter = self.folder / "adapter.pt"
        save_adapter(model, adapter)
        original = torch.load(adapter, weights_only=True)
        for kind in ("layers", "extra_module", "missing_weight", "wrong_rank"):
            with self.subTest(kind=kind):
                payload = copy.deepcopy(original)
                if kind == "layers":
                    payload["layer_ids"] = [0]
                elif kind == "extra_module":
                    payload["modules"]["model.layers.9.self_attn.q_proj"] = asdict(self.lora)
                elif kind == "missing_weight":
                    payload["weights"].pop(next(iter(payload["weights"])))
                else:
                    payload["modules"][next(iter(payload["modules"]))]["rank"] = 3
                torch.save(payload, adapter)
                callbacks = []
                with self.assertRaises(ValueError):
                    self.load(adapter_path=adapter, module_callback=lambda *args: callbacks.append(args))
                self.assertEqual(callbacks, [])
        with self.assertRaisesRegex(ValueError, "requires lora_config"):
            load_qwen3_moe(self.base, adapter_path=adapter)
        with self.assertRaisesRegex(ValueError, "targets only"):
            load_qwen3_moe(self.base, lora_config=LoRAConfig(targets=("mlp.experts",)))

    def test_adamw_dropout_exact_resume_full_adapter_compare_and_independent_reload(self):
        self._resume_and_reload("full")

    def test_compact_checkpoint_exact_resume_and_independent_reload(self):
        self._resume_and_reload("lora")

    def _resume_and_reload(self, checkpoint_state):
        context = {"model_type": "qwen", "source": _source_identity(self.base),
                   "data": {"kind": "authored tiny test"}, "lora": asdict(self.lora),
                   "world_size": 1, "cp_size": 1, "ep_size": 1,
                   "start_layer": 0, "num_layers": 2, "optimizer": "adamw",
                   "router_aux_loss_coef": 0., "cache_quantization": "none"}
        context = json.loads(json.dumps(context))
        def setup():
            model = self.load()
            model.gradient_checkpointing_enable()
            optimizer = build_optimizer(model, OptimizerPolicy("adamw", .01))
            return model, optimizer
        def manager(folder, model, optimizer):
            return CheckpointManager(folder, TrainingState(
                model, optimizer, model_config=model.config.to_dict(),
                data_generator=torch.Generator(), training_config=context,
                state_adapter=StateAdapter(model), checkpoint_state=checkpoint_state))
        def update(model, optimizer, verifier, step):
            optimizer.zero_grad(set_to_none=True)
            loss = model(**self.batch).loss
            loss.backward()
            gradients = verifier.gradients()
            optimizer.step()
            metrics = verifier.after_step(gradients)
            self.assertGreater(metrics["b"]["gradient_nonzero"], 0)
            self.assertEqual(metrics["a"]["gradient_nonzero"] == 0, step == 1)
            return {"step": step, "loss": loss.item(), "ce_loss": loss.item(), "aux_loss": 0.,
                    "input_tokens": int(self.batch["attention_mask"].sum()),
                    "supervised_tokens": int(self.batch["labels"][:, 1:].ne(-100).sum())}
        def export(folder, model, verifier, initial, rows, start):
            folder.mkdir(exist_ok=True)
            save_adapter(model, folder / "adapter.pt")
            trained = capture_probe(model, self.batch, self.runtime)
            reference = make_reference(context, initial, trained, folder / "adapter.pt")
            summary = {**rows[-1], **{key: context[key] for key in
                       ("model_type", "source", "world_size", "cp_size", "ep_size", "data")},
                       "steps": len(rows), "start_step": start,
                       "lora_verification": verifier.finish(),
                       "trained_input_tokens": sum(row["input_tokens"] for row in rows[start:]),
                       "trained_supervised_tokens": sum(row["supervised_tokens"] for row in rows[start:])}
            for filename, record in (("run_config.json", context), ("summary.json", summary),
                                     ("lora_eval_reference.json", reference)):
                (folder / filename).write_text(json.dumps(record))
            (folder / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            return reference
        model, optimizer = setup()
        verifier = LoRAVerifier(model, optimizer, self.runtime)
        initial = capture_probe(model, self.batch, self.runtime)
        rows = [update(model, optimizer, verifier, step) for step in (1, 2)]
        split_folder = self.folder / "split"
        manager(split_folder, model, optimizer).save(2)
        rows.extend(update(model, optimizer, verifier, step) for step in (3, 4))
        continuous = self.folder / "continuous"
        reference = export(continuous, model, verifier, initial, rows, 0)
        restored, restored_optimizer = setup()
        self.assertEqual(manager(split_folder, restored, restored_optimizer).load(), 2)
        resumed_verifier = LoRAVerifier(restored, restored_optimizer, self.runtime)
        resumed_initial = capture_probe(restored, self.batch, self.runtime)
        resumed_rows = rows[:2] + [update(restored, restored_optimizer, resumed_verifier, step) for step in (3, 4)]
        self.assertEqual(resumed_rows, rows)
        for (name, expected), (other_name, actual) in zip(model.named_parameters(), restored.named_parameters()):
            self.assertEqual(name, other_name)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for key, state in restored_optimizer.state_dict()["state"].items():
            for name, value in state.items():
                torch.testing.assert_close(value, optimizer.state_dict()["state"][key][name], rtol=0, atol=0)
        export(split_folder, restored, resumed_verifier, resumed_initial, resumed_rows, 2)
        comparison = compare_runs(continuous, split_folder)
        self.assertTrue(comparison["validation_passed"], comparison["errors"])
        self.assertEqual(comparison["adapter_tensor_count"], 16)
        self.assertEqual(comparison["adapter_max_abs_difference"], 0)

        child_output = self.folder / "child.json"
        code = """
import json, sys, torch
from pathlib import Path
from dsv41_train.lora import LoRAConfig
from dsv41_train.models.qwen import load_qwen3_moe
from dsv41_train.runtime import Runtime
from scripts.verify_lora_adapter import capture_probe
reference = json.loads(Path(sys.argv[2]).read_text())
lora = dict(reference['context']['lora'])
lora['targets'] = tuple(lora['targets'])
model = load_qwen3_moe(sys.argv[1], dtype=torch.bfloat16,
    lora_config=LoRAConfig(**lora), adapter_path=sys.argv[3])
batch = {key: torch.tensor(value) for key,value in reference['trained']['ranks'][0]['inputs'].items()}
Path(sys.argv[4]).write_text(json.dumps(capture_probe(model,batch,Runtime(torch.device('cpu')))))
"""
        root = Path(__file__).resolve().parents[1]
        child = subprocess.run([sys.executable, "-c", code, str(self.base),
                                str(continuous / "lora_eval_reference.json"), str(continuous / "adapter.pt"),
                                str(child_output)], cwd=root,
                               env=dict(os.environ, PYTHONPATH=str(root), OMP_NUM_THREADS="1"),
                               capture_output=True, text=True, timeout=60)
        self.assertEqual(child.returncode, 0, child.stderr)
        self.assertTrue(compare_probes(json.loads(child_output.read_text()), reference["trained"],
                                      atol=0, rtol=0)["passed"])
        # Exercise the real validator's Qwen selection; only CUDA FSDP/topology
        # setup is isolated on this CPU host, not checkpoint/adapter loading.
        def gather(rows, value):
            rows[0] = value
        with patch("scripts.verify_lora_adapter.build_parallelism",
                   return_value=(SimpleNamespace(), ContextParallel(), TokenDispatcher())), \
             patch("scripts.verify_lora_adapter.shard_qwen_component"), \
             patch("scripts.verify_lora_adapter.shard_qwen_root"), \
             patch("scripts.verify_lora_adapter.dist.all_gather_object", side_effect=gather):
            result = verify(SimpleNamespace(reference=str(continuous / "lora_eval_reference.json"),
                            adapter_in=str(continuous / "adapter.pt"), model_path=None,
                            output=str(self.folder / "replay.json"), atol=0., rtol=0.), self.runtime)
        self.assertTrue(result["validation_passed"])


if __name__ == "__main__":
    unittest.main()
