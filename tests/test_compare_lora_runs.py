"""Focused CPU checks for full-adapter and resumed-trajectory comparison."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from scripts.compare_lora_runs import compare_runs, main
from scripts.verify_lora_adapter import file_sha256, input_sha256


def write_run(folder, *, start_step):
    folder.mkdir()
    targets = ["attention.q_a", "attention.q_b", "attention.kv_proj", "attention.o_b"]
    lora = {"rank": 2, "alpha": 4., "dropout": .1, "targets": targets}
    config = {"model_type": "dsv4", "lora": lora, "source": {"path": "/same/base"},
              "world_size": 2, "ep_size": 2, "cp_size": 1, "data": "same-data-hash",
              "start_layer": 0, "num_layers": 1, "seed": 42}
    module_config = dict(lora, targets=tuple(targets))
    modules = {f"model.layers.0.{target}": module_config for target in targets}
    weights = {}
    for index, name in enumerate(modules):
        weights[name + ".lora_a"] = torch.full((2, 512), .01 * (index + 1))
        weights[name + ".lora_b"] = torch.full((512, 2), .02 * (index + 1))
    payload = {"format_version": 1, "layer_ids": [0], "modules": modules, "weights": weights,
               "extra_serialization_metadata": f"independent-export-{start_step}"}
    torch.save(payload, folder / "adapter.pt")
    inputs = {"input_ids": [[1, 2]], "labels": [[-100, 2]], "attention_mask": [[True, True]]}
    ranks = [{"rank": rank, "inputs": inputs, "input_sha256": input_sha256(inputs), "loss": 3.,
              "shifted_targets": 1, "logits_shape": [1, 2, 16], "logit_indices": [0, 31],
              "logit_values": [.5, .6]} for rank in range(2)]
    probe = {"format_version": 1, "world_size": 2, "ranks": ranks}
    initial = copy.deepcopy(probe)
    for row in initial["ranks"]:
        row["loss"] = 4. - start_step / 10
        row["logit_values"] = [.1 + start_step / 100, .2]
    reference = {"format_version": 1, "context": config, "initial": initial, "trained": probe,
                 "adapter": {"path": str(folder / "adapter.pt"), "sha256": file_sha256(folder / "adapter.pt")}}
    metrics = [{"step": step, "loss": 3. - .1 * step, "ce_loss": 3. - .1 * step,
                "aux_loss": .1, "input_tokens": 128, "supervised_tokens": 16 + step} for step in range(1, 6)]
    summary = {**metrics[-1], **{key: config[key] for key in ("model_type", "source", "world_size", "cp_size", "ep_size", "data")},
               "steps": 5, "start_step": start_step, "lora_verification": {"verified": True},
               "trained_input_tokens": sum(row["input_tokens"] for row in metrics[start_step:]),
               "trained_supervised_tokens": sum(row["supervised_tokens"] for row in metrics[start_step:])}
    for name, value in (("run_config.json", config), ("lora_eval_reference.json", reference), ("summary.json", summary)):
        (folder / name).write_text(json.dumps(value))
    (folder / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in metrics))


class LoRARunComparisonTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.continuous, self.resumed = self.root / "continuous", self.root / "resumed"
        write_run(self.continuous, start_step=0)
        write_run(self.resumed, start_step=3)

    def edit_json(self, name, mutate):
        path = self.resumed / name
        value = json.loads(path.read_text())
        mutate(value)
        path.write_text(json.dumps(value))

    def edit_adapter(self, mutate):
        path = self.resumed / "adapter.pt"
        payload = torch.load(path, weights_only=True)
        mutate(payload)
        torch.save(payload, path)
        self.edit_json("lora_eval_reference.json", lambda value: value["adapter"].update(sha256=file_sha256(path)))

    def test_exact_full_eight_tensor_success_with_different_initials_and_file_hashes(self):
        result = compare_runs(self.continuous, self.resumed)
        self.assertTrue(result["validation_passed"], result["errors"])
        self.assertTrue(result["full_adapter_tensors_compared"])
        self.assertEqual(result["adapter_tensor_count"], 8)
        self.assertEqual(result["adapter_elements_compared"], 8192)
        self.assertEqual(result["adapter_max_abs_difference"], 0.)
        self.assertEqual(len(result["steps"]), 5)
        self.assertNotEqual(*result["adapter_file_sha256"].values())

    def test_single_element_anywhere_is_checked_and_nonzero_tolerance_is_explicit(self):
        self.edit_adapter(lambda value: value["weights"]["model.layers.0.attention.q_a.lora_a"][0].__setitem__(257, .0101))
        result = compare_runs(self.continuous, self.resumed)
        self.assertFalse(result["validation_passed"])
        self.assertGreater(result["adapter_max_abs_difference"], 0.)
        tolerant = compare_runs(self.continuous, self.resumed, atol=.001)
        self.assertTrue(tolerant["validation_passed"], tolerant["errors"])
        self.assertFalse(tolerant["exact_equality_required"])

    def test_nonfinite_adapter_and_missing_tensor_fail(self):
        self.edit_adapter(lambda value: value["weights"]["model.layers.0.attention.q_a.lora_b"][0].__setitem__(0, float("nan")))
        self.assertFalse(compare_runs(self.continuous, self.resumed)["validation_passed"])
        self.edit_adapter(lambda value: value["weights"].pop("model.layers.0.attention.q_a.lora_b"))
        result = compare_runs(self.continuous, self.resumed)
        self.assertTrue(any("tensor names" in error for error in result["errors"]))

    def test_complete_step_sequence_and_count_deltas(self):
        path = self.resumed / "metrics.jsonl"
        original = path.read_text()
        rows = [json.loads(line) for line in original.splitlines()]
        path.write_text("\n".join(json.dumps(row) for row in rows[1:]))
        self.assertFalse(compare_runs(self.continuous, self.resumed)["validation_passed"])
        rows[1]["input_tokens"] += 1
        path.write_text("\n".join(json.dumps(row) for row in rows))
        result = compare_runs(self.continuous, self.resumed)
        self.assertFalse(result["validation_passed"])
        self.assertEqual(result["steps"][1]["counts"]["input_tokens"]["delta"], 1)
        rows[1]["input_tokens"] -= 1
        rows[1]["ce_loss"] += .02
        path.write_text("\n".join(json.dumps(row) for row in rows))
        result = compare_runs(self.continuous, self.resumed)
        self.assertGreater(result["steps"][1]["loss_abs_differences"]["ce_loss"], 0.)

    def test_final_probe_is_compared_and_initial_is_not(self):
        self.edit_json("lora_eval_reference.json", lambda value: value["initial"]["ranks"][0].update(loss=1000.))
        self.assertTrue(compare_runs(self.continuous, self.resumed)["validation_passed"])
        self.edit_json("lora_eval_reference.json", lambda value: value["trained"]["ranks"][1].update(loss=3.001))
        result = compare_runs(self.continuous, self.resumed)
        self.assertFalse(result["validation_passed"])
        self.assertFalse(result["final_eval"]["passed"])

    def test_config_resume_marker_and_cli_failure_report(self):
        self.edit_json("run_config.json", lambda value: value.update(seed=43))
        result = compare_runs(self.continuous, self.resumed)
        self.assertFalse(result["validation_passed"])
        self.edit_json("run_config.json", lambda value: value.update(seed=42))
        self.edit_json("summary.json", lambda value: value.update(start_step=0))
        result = compare_runs(self.continuous, self.resumed)
        self.assertTrue(any("start_step>0" in error for error in result["errors"]))
        output = self.root / "comparison.json"
        with patch("builtins.print"):
            code = main(["--continuous", str(self.continuous), "--resumed", str(self.resumed), "--output", str(output)])
        self.assertEqual(code, 1)
        self.assertFalse(json.loads(output.read_text())["validation_passed"])


if __name__ == "__main__":
    unittest.main()
