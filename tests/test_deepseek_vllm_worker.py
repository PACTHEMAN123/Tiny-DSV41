"""CPU buffer tests; no vLLM engine or GPU execution is claimed here."""
import importlib
import sys
import types
import unittest
from unittest.mock import patch

import torch


class CPUWrapper:
    def __init__(self):
        self.tp_size, self.n_slices = 1, 2
        self.lora_a_stacked = tuple(torch.zeros(2, 1, 8, 16, dtype=torch.bfloat16) for _ in range(2))
        self.lora_b_stacked = tuple(torch.zeros(2, 1, output, 8, dtype=torch.bfloat16) for output in (8, 4))


class DeepseekWorkerReadbackTests(unittest.TestCase):
    def setUp(self):
        # Only import-time external registration and the exact wrapper class
        # are substituted. verify_readback executes unchanged on real buffers.
        with patch("rollout_manager.backends.deepseek_v41_lora.register_deepseek_v41_lora"):
            worker = importlib.import_module("rollout_manager.backends.vllm_lora_worker")
        self.verify = worker.verify_readback
        module = types.ModuleType("vllm.lora.layers.column_parallel_linear")
        module.MergedColumnParallelLinearWithLoRA = CPUWrapper
        self.patch = patch.dict(sys.modules, {module.__name__: module})
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.text = {"text_config": dict(hidden_size=16, q_lora_rank=8, head_dim=4, num_hidden_layers=2)}
        self.names = [f"language_model.model.layers.{i}.attn.fused_wqa_wkv" for i in range(2)]
        self.manager = types.SimpleNamespace(lora_index_to_id=[None, 7],
            modules={name: CPUWrapper() for name in self.names})
        self.source = self.names[1].rsplit(".", 1)[0] + ".wq_a"
        self.a = torch.arange(128).reshape(8, 16).to(torch.bfloat16)
        self.b = torch.arange(64).reshape(8, 8).to(torch.bfloat16)
        self.tensors = {self.source + ".lora_A.weight": self.a,
                        self.source + ".lora_B.weight": self.b}
        wrapper = self.manager.modules[self.names[1]]
        wrapper.lora_a_stacked[0][1, 0].copy_(self.a)
        wrapper.lora_b_stacked[0][1, 0].copy_(self.b)

    def test_checks_actual_buffers_including_untouched_layer_and_kv(self):
        receipt = self.verify(self.manager, self.tensors, 7, self.text)
        self.assertEqual(receipt["slot"], 1)
        self.assertEqual(receipt["target_count"], 2)
        self.assertEqual(receipt["adapted_modules"], [self.source])

    def test_rejects_wrong_query_stale_kv_and_untouched_layer(self):
        for layer, part in ((1, 0), (1, 1), (0, 0)):
            with self.subTest(layer=layer, part=part):
                buffer = self.manager.modules[self.names[layer]].lora_b_stacked[part]
                original = buffer.clone()
                buffer[1, 0, 0, 0] += 1
                with self.assertRaises(RuntimeError):
                    self.verify(self.manager, self.tensors, 7, self.text)
                buffer.copy_(original)

    def test_rejects_ambiguous_slot_extra_target_and_wrapper_subclass(self):
        self.manager.lora_index_to_id = [7, 7]
        with self.assertRaisesRegex(RuntimeError, "unique"):
            self.verify(self.manager, self.tensors, 7, self.text)
        self.manager.lora_index_to_id = [None, 7]
        self.manager.modules["unexpected"] = CPUWrapper()
        with self.assertRaisesRegex(RuntimeError, "wrappers"):
            self.verify(self.manager, self.tensors, 7, self.text)
        del self.manager.modules["unexpected"]
        self.manager.modules[self.names[0]] = type("UnexpectedSubclass", (CPUWrapper,), {})()
        with self.assertRaisesRegex(RuntimeError, "layout"):
            self.verify(self.manager, self.tensors, 7, self.text)


if __name__ == "__main__":
    unittest.main()
