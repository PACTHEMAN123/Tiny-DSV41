"""Engram initialization must not perturb the caller's training RNG streams."""

from copy import deepcopy
from dataclasses import replace
import unittest
from unittest.mock import patch

import torch

from dsv41_train.models.dsv4.engram import EngramPolicy
from dsv41_train.models.qwen import Qwen3MoeForCausalLM
from dsv41_train.models.qwen.engram import QwenEngramConfig, attach_engram
from qwen_helpers import tiny_config


class QwenEngramRNGTest(unittest.TestCase):
    def setUp(self):
        self.config = QwenEngramConfig(
            (0, 1), vocab_per_head=17, head_dim=4, heads=2, max_ngram=3, seed=207,
        )

    def attach(self, model, config=None, *, device="cpu", policy=None):
        attach_engram(model, config or self.config, policy or EngramPolicy("cpu"),
                      device=device, dtype=torch.float32)
        for table in model.model.engram_tables.values():
            self.addCleanup(table.close)

    @staticmethod
    def augmentation(model):
        return {name: value.detach().cpu().clone() for name, value in model.named_parameters()
                if ".engram." in name or ".engram_tables." in name}

    def test_cpu_call_preserves_next_draw_and_never_reseeds_cuda(self):
        model = Qwen3MoeForCausalLM(tiny_config())
        before = torch.get_rng_state().clone()
        expected = torch.randn(12, generator=torch.Generator().set_state(before))
        cuda_initialized = torch.cuda.is_initialized()
        # This catches the regression even on CPU-only CI: torch.manual_seed
        # schedules CUDA reseeding even when CUDA is not initialized yet.
        with patch("torch.cuda.manual_seed_all", side_effect=AssertionError("CUDA reseeded")):
            self.attach(model)
        self.assertEqual(torch.cuda.is_initialized(), cuda_initialized)
        torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
        torch.testing.assert_close(torch.randn(12), expected, rtol=0, atol=0)

    def test_fixed_seed_is_independent_of_caller_rng_and_default_device(self):
        base = Qwen3MoeForCausalLM(tiny_config())
        first, second, different = deepcopy(base), deepcopy(base), deepcopy(base)
        self.attach(first)
        torch.rand(19)
        before = torch.get_rng_state().clone()
        # The initialization contract is CPU-based even within another default
        # device context. Meta exercises this on hosts without a CUDA device.
        with torch.device("meta"):
            self.attach(second)
        torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
        expected, actual = self.augmentation(first), self.augmentation(second)
        self.assertTrue(expected)
        self.assertEqual(expected.keys(), actual.keys())
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0, msg=name)
        self.attach(different, replace(self.config, seed=self.config.seed + 1))
        changed = self.augmentation(different)
        for name in expected:
            if name.endswith("proj.weight") or ".engram_tables." in name:
                self.assertFalse(torch.equal(changed[name], expected[name]), name)

    def test_cpu_rng_is_restored_when_initialization_raises(self):
        model = Qwen3MoeForCausalLM(tiny_config())
        before = torch.get_rng_state().clone()
        with patch("torch.nn.init.normal_", side_effect=RuntimeError("initialization failed")):
            with self.assertRaisesRegex(RuntimeError, "initialization failed"):
                self.attach(model)
        torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_preserves_all_initialized_cuda_streams_and_matches_cpu_weights(self):
        devices = list(range(torch.cuda.device_count()))
        with torch.random.fork_rng(devices=devices):
            base = Qwen3MoeForCausalLM(tiny_config())
            reference = deepcopy(base)
            self.attach(reference)
            expected = self.augmentation(reference)
            # Distinct seeds and offsets on every GPU detect changing devices
            # other than the target GPU as well as resetting the active stream.
            for index in devices:
                with torch.cuda.device(index):
                    torch.cuda.manual_seed(500 + index)
                    torch.rand(7 + index, device=f"cuda:{index}")
            for placement in ("cpu", "cuda"):
                with self.subTest(placement=placement):
                    model = deepcopy(base)
                    cpu_before = torch.get_rng_state().clone()
                    cuda_before = torch.cuda.get_rng_state_all()
                    with torch.device("cuda:0"):
                        self.attach(model, device="cuda:0", policy=EngramPolicy(placement))
                    torch.testing.assert_close(torch.get_rng_state(), cpu_before, rtol=0, atol=0)
                    for state, original in zip(torch.cuda.get_rng_state_all(), cuda_before):
                        torch.testing.assert_close(state, original, rtol=0, atol=0)
                    actual = self.augmentation(model)
                    for name in expected:
                        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0, msg=name)


if __name__ == "__main__":
    unittest.main()
