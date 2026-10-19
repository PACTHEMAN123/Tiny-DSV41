import importlib
import sys
import types
import unittest
from unittest.mock import patch

import torch
from dsv41_train.rl.lora_config import DistributedLoRAConfig

from rollout_manager.backends.deepseek_v41_lora import (
    LORA_ARCHITECTURE, PACKED_MODULES_MAPPING, lora_target_modules, register_deepseek_v41_lora,
    validate_adapter_tensors, validate_deployment,
)


class DeepseekLoRAContractTests(unittest.TestCase):
    def setUp(self):
        self.text = dict(hidden_size=16, q_lora_rank=8, head_dim=4, num_hidden_layers=2)
        self.module = "language_model.model.layers.1.attn.wq_a"
        self.tensors = {
            self.module + ".lora_A.weight": torch.ones(8, 16, dtype=torch.bfloat16),
            self.module + ".lora_B.weight": torch.zeros(8, 8, dtype=torch.bfloat16),
        }
        self.config = types.SimpleNamespace(
            lora_config=types.SimpleNamespace(target_modules=lora_target_modules(self.text),
                max_lora_rank=8, lora_dtype=torch.bfloat16, fully_sharded_loras=False,
                enable_tower_connector_lora=False),
            parallel_config=types.SimpleNamespace(pipeline_parallel_size=1),
            speculative_config=None,
            model_config=types.SimpleNamespace(hf_config=self.text),
            additional_config={"rollout_manager_lora": DistributedLoRAConfig().to_dict()},
        )

    def test_exact_scope_and_both_config_layouts(self):
        self.assertEqual(PACKED_MODULES_MAPPING["gate_up_proj"], ["w1", "w3"])
        validate_deployment(self.config)
        for config in (self.text, {"text_config": self.text}):
            self.assertEqual(validate_adapter_tensors(self.tensors, config), (self.module,))
            self.assertEqual(lora_target_modules(config), [
                f"language_model.model.layers.{i}.attn.fused_wqa_wkv" for i in range(2)])
        for targets in (["gate_up_proj"], ["fused_wqa_wkv"], lora_target_modules(self.text)[:1]):
            self.config.lora_config.target_modules = targets
            with self.assertRaises(ValueError):
                validate_deployment(self.config)

    def test_configured_subset_rank_and_exact_inventory(self):
        profile = DistributedLoRAConfig(rank=16, alpha=32, layers=(1,))
        self.config.additional_config["rollout_manager_lora"] = profile.to_dict()
        self.config.lora_config.target_modules = lora_target_modules(self.text, profile)
        self.config.lora_config.max_lora_rank = profile.rank
        validate_deployment(self.config)
        tensors = {self.module + ".lora_A.weight": torch.ones(16, 16, dtype=torch.bfloat16),
                   self.module + ".lora_B.weight": torch.ones(8, 16, dtype=torch.bfloat16)}
        self.assertEqual(validate_adapter_tensors(tensors, self.text, lora_config=profile),
                         (self.module,))
        for changed in (DistributedLoRAConfig(rank=8, layers=(1,)),
                        DistributedLoRAConfig(rank=16, layers=(0,))):
            with self.subTest(profile=changed), self.assertRaises(ValueError):
                validate_adapter_tensors(tensors, self.text, lora_config=changed)
        with self.assertRaisesRegex(ValueError, "overflows"):
            validate_adapter_tensors(tensors, self.text,
                                     lora_config=DistributedLoRAConfig(16, 1e41, (1,)))

    def test_installed_target_filter_excludes_standalone_engram_wkv(self):
        try:
            from vllm.lora.utils import is_in_target_modules
        except ImportError:
            self.skipTest("requires the inspected vLLM runtime")
        attention = "language_model.model.layers.1.attn.fused_wqa_wkv"
        engram = "language_model.model.layers.1.engram.wkv"
        self.assertTrue(is_in_target_modules(engram, ["fused_wqa_wkv"], PACKED_MODULES_MAPPING))
        targets = lora_target_modules(self.text)
        self.assertTrue(is_in_target_modules(attention, targets, PACKED_MODULES_MAPPING))
        self.assertFalse(is_in_target_modules(engram, targets, PACKED_MODULES_MAPPING))

    def test_rejects_partial_wrong_scope_dimensions_dtype_and_nonfinite(self):
        a, b = self.tensors
        bad = [
            {a: self.tensors[a]},
            {k.replace("wq_a", "wkv"): v for k, v in self.tensors.items()},
            {k.replace("layers.1", "layers.2"): v for k, v in self.tensors.items()},
            {**self.tensors, b: torch.zeros(4, 8, dtype=torch.bfloat16)},
            {**self.tensors, b: self.tensors[b].float()},
            {**self.tensors, b: torch.full((8, 8), float("nan"), dtype=torch.bfloat16)},
        ]
        for tensors in bad:
            with self.subTest(keys=list(tensors)), self.assertRaises(ValueError):
                validate_adapter_tensors(tensors, self.text)

    def test_rejects_broad_deployment_before_base_model_construction(self):
        self.config.lora_config.target_modules = None
        base = type("Base", (), {"__init__": lambda *a, **k: self.fail("allocated model")})
        modules = {}
        for name, attr, value in (
            ("vllm.model_executor.layers.linear", "MergedColumnParallelLinear", type("Linear", (), {})),
            ("vllm.model_executor.models.interfaces", "SupportsLoRA", type("SupportsLoRA", (), {})),
            ("vllm.models.deepseek_v41.nvidia.vl_model", "DeepseekV41ForCausalLM", base),
        ):
            modules[name] = types.ModuleType(name)
            setattr(modules[name], attr, value)
        name = "rollout_manager.backends.deepseek_v41_lora_model"
        with patch.dict(sys.modules, modules):
            sys.modules.pop(name, None)
            try:
                cls = importlib.import_module(name).DeepseekV41LoRAForCausalLM
                with self.assertRaisesRegex(ValueError, "target_modules"):
                    cls(vllm_config=self.config)
            finally:
                sys.modules.pop(name, None)

    def test_registration_is_lazy_and_version_specific(self):
        calls = []
        fake = types.ModuleType("vllm")
        fake.__version__ = "0.31.0"
        fake.ModelRegistry = types.SimpleNamespace(register_model=lambda *args: calls.append(args))
        with patch.dict(sys.modules, {"vllm": fake}), patch(
                "rollout_manager.backends.vllm_lora_triton.configure_lora_triton_source") as source_guard:
            register_deepseek_v41_lora()
            source_guard.assert_called_once_with()
            self.assertEqual(LORA_ARCHITECTURE, "DeepseekV41ForCausalLM")
            self.assertEqual(calls[0], ("DeepseekV41ForCausalLM",
                "rollout_manager.backends.deepseek_v41_lora_model:DeepseekV41LoRAForCausalLM"))
            fake.__version__ = "0.30.0"
            with self.assertRaises(RuntimeError):
                register_deepseek_v41_lora()


if __name__ == "__main__":
    unittest.main()
