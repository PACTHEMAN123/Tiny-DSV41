"""CPU tests for LoRA eval references; no checkpoint downloads or CUDA needed."""

import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from dsv41_train.lora import LoRAConfig, inject_lora, load_adapter, save_adapter
from dsv41_train.runtime import Runtime
from scripts.verify_lora_adapter import (
    capture_probe, compare_probes, file_sha256, make_reference, parse_args,
    probe_indices, validate_probe, validate_reference,
)


class TinyLoRA(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layer_ids = [0]
        self.model.embedding = nn.Embedding(16, 8)
        self.lm_head = nn.Linear(8, 16, bias=False)
        self.config = LoRAConfig(rank=2, alpha=4, dropout=.25, targets=("lm_head",))
        inject_lora(self, self.config)

    def forward(self, input_ids, labels, attention_mask):
        # Deliberately consume RNG even in eval to test capture's restoration.
        torch.rand(3)
        logits = self.lm_head(self.model.embedding(input_ids))
        target = labels[:, 1:].masked_fill(~attention_mask[:, 1:].bool(), -100)
        loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, 16), target.reshape(-1))
        return SimpleNamespace(loss=loss, logits=logits)


class AdapterProbeTest(unittest.TestCase):
    def setUp(self):
        self.runtime = Runtime(torch.device("cpu"))
        self.batch = {"input_ids": torch.tensor([[1, 2, 3, 4]]),
                      "labels": torch.tensor([[-100, -100, 3, 4]]),
                      "attention_mask": torch.ones(1, 4, dtype=torch.int64)}
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        (self.folder / "config.json").write_text("{}")
        (self.folder / "model.safetensors.index.json").write_text("{}")
        self.adapter = self.folder / "adapter.pt"

    def model(self):
        torch.manual_seed(123)
        return TinyLoRA()

    def trained_reference(self):
        model = self.model()
        initial = capture_probe(model, self.batch, self.runtime)
        optimizer = torch.optim.SGD([parameter for parameter in model.parameters() if parameter.requires_grad], lr=.5)
        model(**self.batch).loss.backward()
        optimizer.step()
        final = capture_probe(model, self.batch, self.runtime)
        save_adapter(model, self.adapter)
        context = {"model_type": "dsv4", "lora": asdict(model.config), "world_size": 1,
                   "cp_size": 1, "ep_size": 1, "start_layer": 0, "num_layers": 1,
                   "source": {"path": str(self.folder),
                              "config_sha256": file_sha256(self.folder / "config.json"),
                              "index_sha256": file_sha256(self.folder / "model.safetensors.index.json")}}
        reference = make_reference(context, initial, final, self.adapter)
        return model, reference

    def test_capture_restores_rng_and_mixed_module_training_flags(self):
        model = self.model()
        model.model.embedding.eval()
        states = [module.training for module in model.modules()]
        rng = torch.random.get_rng_state().clone()
        probe = capture_probe(model, self.batch, self.runtime)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertEqual(states, [module.training for module in model.modules()])
        self.assertEqual(probe["ranks"][0]["shifted_targets"], 2)
        self.assertIs(probe["ranks"][0]["inputs"]["attention_mask"][0][0], True)
        repeated = capture_probe(model, self.batch, self.runtime)
        self.assertEqual(probe, repeated)

    def test_actual_adapter_export_reloads_identical_eval_and_changes_initial(self):
        _, reference = self.trained_reference()
        folder, config = validate_reference(reference, self.adapter)
        self.assertEqual(folder, self.folder)
        self.assertEqual(config.rank, 2)
        fresh = self.model()
        load_adapter(fresh, self.adapter)
        replay = capture_probe(fresh, self.batch, self.runtime)
        self.assertTrue(compare_probes(replay, reference["trained"], atol=0, rtol=0)["passed"])
        self.assertTrue(compare_probes(replay, reference["initial"], atol=0, rtol=0)["output_changed"])
        self.assertEqual(reference, json.loads(json.dumps(reference)))

    def test_reference_without_export_is_allowed_but_cannot_claim_adapter_reload(self):
        _, reference = self.trained_reference()
        no_export = make_reference(reference["context"], reference["initial"], reference["trained"])
        self.assertIsNone(no_export["adapter"])
        with self.assertRaisesRegex(ValueError, "no adapter export"):
            validate_reference(no_export, self.adapter)

    def test_adapter_reload_matches_in_an_independent_python_process(self):
        _, reference = self.trained_reference()
        inputs = self.folder / "inputs.json"
        inputs.write_text(json.dumps(reference["trained"]["ranks"][0]["inputs"]))
        result = self.folder / "child-probe.json"
        code = """
import json, sys, torch
from pathlib import Path
from test_verify_lora_adapter import TinyLoRA
from dsv41_train.lora import load_adapter
from dsv41_train.runtime import Runtime
from scripts.verify_lora_adapter import capture_probe
torch.manual_seed(123)
model = TinyLoRA()
load_adapter(model, sys.argv[1])
batch = {key: torch.tensor(value) for key, value in json.loads(Path(sys.argv[2]).read_text()).items()}
probe = capture_probe(model, batch, Runtime(torch.device('cpu')))
Path(sys.argv[3]).write_text(json.dumps(probe))
"""
        root = Path(__file__).resolve().parents[1]
        environment = dict(os.environ, PYTHONPATH=str(root) + os.pathsep + str(root / "tests"), OMP_NUM_THREADS="1")
        child = subprocess.run([sys.executable, "-c", code, str(self.adapter), str(inputs), str(result)],
                               cwd=self.folder, env=environment, capture_output=True, text=True, timeout=60)
        self.assertEqual(child.returncode, 0, child.stderr)
        replay = json.loads(result.read_text())
        self.assertTrue(compare_probes(replay, reference["trained"], atol=0, rtol=0)["passed"])

    def test_unchanged_probe_cannot_be_claimed_as_trained_reference(self):
        _, reference = self.trained_reference()
        with self.assertRaisesRegex(ValueError, "did not change"):
            make_reference(reference["context"], reference["initial"], reference["initial"], self.adapter)

    def test_file_source_and_adapter_config_identity_mismatches_fail(self):
        _, reference = self.trained_reference()
        original = self.adapter.read_bytes()
        self.adapter.write_bytes(original + b"different export")
        with self.assertRaisesRegex(ValueError, "SHA256 differs"):
            validate_reference(reference, self.adapter)
        self.adapter.write_bytes(original)
        wrong = copy.deepcopy(reference)
        wrong["context"]["lora"]["rank"] = 3
        with self.assertRaisesRegex(ValueError, "configuration differs"):
            validate_reference(wrong, self.adapter)
        (self.folder / "config.json").write_text('{"changed": true}')
        with self.assertRaisesRegex(ValueError, "config.json identity differs"):
            validate_reference(reference, self.adapter)

    def test_token_identity_sample_indices_and_nonfinite_values_are_rejected(self):
        _, reference = self.trained_reference()
        mutations = (
            lambda row: row["inputs"]["input_ids"][0].__setitem__(0, 2),
            lambda row: row["logit_indices"].__setitem__(0, 10000),
            lambda row: row["logit_values"].__setitem__(0, float("nan")),
            lambda row: row.__setitem__("shifted_targets", 0),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                broken = copy.deepcopy(reference["trained"])
                mutation(broken["ranks"][0])
                with self.assertRaises(ValueError):
                    validate_probe(broken)

    def test_zero_shifted_target_capture_fails(self):
        labels = torch.tensor([[1, -100, -100, -100]])
        with self.assertRaisesRegex(ValueError, "shifted supervised"):
            capture_probe(self.model(), dict(self.batch, labels=labels), self.runtime)

    def test_rank_alignment_and_tolerance_report(self):
        _, reference = self.trained_reference()
        expected = copy.deepcopy(reference["trained"])
        expected["world_size"] = 2
        expected["ranks"].append(copy.deepcopy(expected["ranks"][0]))
        expected["ranks"][1]["rank"] = 1
        actual = copy.deepcopy(expected)
        actual["ranks"].reverse()
        actual["ranks"][0]["logit_values"][0] += 1e-4
        self.assertTrue(compare_probes(actual, expected, atol=1e-3, rtol=0)["passed"])
        self.assertFalse(compare_probes(actual, expected, atol=0, rtol=0)["passed"])
        actual["ranks"][0]["loss"] += 1.
        self.assertFalse(compare_probes(actual, expected)["passed"])

    def test_exact_integer_indices_large_sizes_and_cli_tolerances(self):
        maximum = 2**63 - 1
        indices = probe_indices(maximum)
        self.assertEqual(indices[0].item(), 0)
        self.assertEqual(indices[-1].item(), maximum - 1)
        self.assertTrue((indices[1:] > indices[:-1]).all())
        required = ["--reference", "reference.json", "--adapter-in", "adapter.pt", "--output", "result.json"]
        self.assertEqual(parse_args(required).rtol, .01)
        with self.assertRaises(SystemExit):
            parse_args(required + ["--atol", "nan"])


if __name__ == "__main__":
    unittest.main()
