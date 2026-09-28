"""Target selection, adapter file compatibility, and CLI error handling."""

import argparse
import copy
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from dsv41_train.lora import (
    LoRAConfig, LoRALinear, adapter_parameters, inject_lora, load_adapter,
    load_adapter_state, merge_adapters,
)
from dsv41_train.lora.cli import adapter_config, add_lora_options


class LoRAAPITest(unittest.TestCase):
    def test_target_selection_preserves_base_names_and_parameter_identity(self):
        model = nn.ModuleDict({"left": nn.Sequential(nn.Linear(4, 3)),
                               "right": nn.Sequential(nn.Linear(4, 3))})
        original = dict(model.named_parameters())
        names = inject_lora(model, LoRAConfig(targets=("0", "left.0")))
        self.assertEqual(names, ["left.0", "right.0"])
        for name, parameter in original.items():
            self.assertIs(dict(model.named_parameters())[name], parameter)
            self.assertFalse(parameter.requires_grad)
        self.assertEqual(len(list(adapter_parameters(model))), 4)
        with self.assertRaisesRegex(ValueError, "already installed"):
            inject_lora(model, LoRAConfig(targets=("0",)))

    def test_all_targets_are_validated_before_mutation(self):
        for invalid in ("missing", "1"):
            model = nn.Sequential(nn.Linear(4, 3), nn.ReLU())
            with self.assertRaises(ValueError):
                inject_lora(model, LoRAConfig(targets=("0", invalid)))
            self.assertNotIsInstance(model[0], LoRALinear)
            self.assertTrue(all(p.requires_grad for p in model.parameters()))
        model = nn.Sequential(nn.Linear(4, 3, device="meta"))
        with self.assertRaisesRegex(ValueError, "after loading"):
            inject_lora(model, LoRAConfig(targets=("0",)))

    def test_loads_version_one_adapter_without_a_model_dependency(self):
        config = LoRAConfig(rank=2, alpha=4, targets=("0",))
        model = nn.Sequential(nn.Linear(4, 3))
        inject_lora(model, config)
        payload = {
            "format_version": 1, "layer_ids": None,
            "modules": {"0": asdict(config)},
            "weights": {"0.lora_a": torch.full((2, 4), 0.25),
                        "0.lora_b": torch.full((3, 2), 0.5)},
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "adapter.pt"
            torch.save(payload, path)
            load_adapter(model, path)
        for name, tensor in payload["weights"].items():
            torch.testing.assert_close(dict(model.named_parameters())[name], tensor, rtol=0, atol=0)

        before = [p.clone() for p in adapter_parameters(model)]
        malformed = copy.deepcopy(payload)
        malformed["weights"]["0.lora_b"] = torch.ones(1, 2)
        with self.assertRaisesRegex(ValueError, "shape"):
            load_adapter_state(model, malformed)
        for actual, expected in zip(adapter_parameters(model), before):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        model.eval()
        merge_adapters(model)
        with self.assertRaisesRegex(ValueError, "unmerge"):
            load_adapter_state(model, payload)

    def test_cli_defaults_and_conflicts(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--resume", action="store_true")
        add_lora_options(parser)
        self.assertIsNone(adapter_config(parser.parse_args([])))
        self.assertEqual(adapter_config(parser.parse_args(["--lora"])), LoRAConfig())
        selected = adapter_config(parser.parse_args([
            "--lora", "--lora-targets", " attention.q_a, attention.o_b ",
        ]))
        self.assertEqual(selected.targets, ("attention.q_a", "attention.o_b"))
        for arguments in (
            ["--adapter-in", "adapter.pt"],
            ["--adapter-out", "adapter.pt"],
            ["--lora", "--adapter-in", "adapter.pt", "--resume"],
            ["--lora", "--lora-rank", "0"],
            ["--lora", "--lora-targets", "attention.q_a,"],
        ):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                adapter_config(parser.parse_args(arguments))


if __name__ == "__main__":
    unittest.main()
