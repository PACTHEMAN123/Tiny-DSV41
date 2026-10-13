"""Qwen augmentation, sparse update, and FP8 loading regression checks."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from dsv41_train.loading.quantization import dequantize_blockwise_fp8
from dsv41_train.models.qwen import Qwen3MoeForCausalLM, load_qwen3_moe
from dsv41_train.models.qwen.engram import QwenEngramConfig, attach_engram
from dsv41_train.models.dsv4.engram import EngramPolicy
from dsv41_train.optimization import OptimizerPolicy, build_optimizer
from qwen_helpers import tiny_config, to_hf_state_dict


class QwenEngramTest(unittest.TestCase):
    def test_fsdp_can_discover_output_tensors(self):
        from torch.utils._pytree import tree_flatten, tree_unflatten
        from dsv41_train.models.qwen.model import Qwen3MoeCausalLMOutput
        loss, logits = torch.tensor(1.), torch.randn(2, 3, 4)
        output = Qwen3MoeCausalLMOutput(loss, logits, None, None)
        flat, structure = tree_flatten(output)
        self.assertIs(flat[0], loss)
        self.assertIs(flat[1], logits)
        self.assertIs(tree_unflatten(flat, structure).logits, logits)

    def test_fp8_tail_blocks_and_layerwise_loader(self):
        from safetensors.torch import save_file
        source = Qwen3MoeForCausalLM(tiny_config())
        state = to_hf_state_dict(source)
        expected = {}
        for name, value in list(state.items()):
            if value.ndim != 2:
                expected[name] = value
                continue
            rows, columns = value.shape
            scales = torch.rand((rows+2)//3, (columns+4)//5) + .5
            raw = value.to(torch.float8_e4m3fn)
            state[name] = raw
            state[name + "_scale_inv"] = scales
            expanded = scales.repeat_interleave(3, 0).repeat_interleave(5, 1)[:rows, :columns]
            expected[name] = raw.float() * expanded
            torch.testing.assert_close(dequantize_blockwise_fp8(raw, scales, (3, 5), dtype=torch.float32),
                                       expected[name], rtol=0, atol=0)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = source.config.to_dict()
            config["quantization_config"] = {"quant_method": "fp8", "weight_block_size": [3, 5]}
            (root / "config.json").write_text(json.dumps(config))
            (root / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": dict.fromkeys(state, "weights.safetensors")}))
            save_file(state, root / "weights.safetensors")
            seen = []
            loaded = load_qwen3_moe(root, dtype=torch.float32,
                                    module_callback=lambda name, module: seen.append(name))
            actual = to_hf_state_dict(loaded)
            self.assertEqual(len(seen), source.config.num_hidden_layers + 3)
            for name, value in expected.items():
                torch.testing.assert_close(actual[name], value, rtol=0, atol=0)

    def exercise(self, device, dtype=torch.float32):
        torch.manual_seed(77)
        base = Qwen3MoeForCausalLM(tiny_config()).to(device=device, dtype=dtype)
        engram = QwenEngramConfig((0, 1), vocab_per_head=17, head_dim=4, heads=2, max_ngram=3)
        policies = [EngramPolicy("cuda"), EngramPolicy("cuda", cache_bytes=4096),
                    EngramPolicy("cpu"), EngramPolicy("cpu", "async", cache_bytes=4096)]
        models = [deepcopy(base) for _ in policies]
        for model, policy in zip(models, policies):
            attach_engram(model, engram, policy, device=device, dtype=dtype)
            model.gradient_checkpointing_enable()
        optimizers = [build_optimizer(model, OptimizerPolicy("sgd", .01)) for model in models]
        ids = torch.tensor([[1, 3, 3, 3, 5, 2], [1, 4, 4, 4, 2, 0]], device=device)
        mask = ids.ne(0)
        labels = ids.clone().masked_fill(~mask, -100)
        labels[:, :2] = -100
        tolerance = dict(rtol=.02, atol=.002) if dtype == torch.bfloat16 else dict(rtol=2e-5, atol=2e-6)
        try:
            for _ in range(3):
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)
                for _ in range(2):
                    outputs = [model(ids, attention_mask=mask, labels=labels) for model in models]
                    for output in outputs[1:]:
                        torch.testing.assert_close(output.logits, outputs[0].logits, **tolerance)
                    for output in outputs:
                        (output.loss * .37 / 2).backward()
                reference = dict(models[0].named_parameters())
                for model in models[1:]:
                    for name, parameter in model.named_parameters():
                        grad = parameter.grad
                        expected_grad = reference[name].grad
                        if grad is not None:
                            left, right = grad.to_dense().float().cpu(), expected_grad.to_dense().float().cpu()
                            if dtype == torch.bfloat16:
                                self.assertLessEqual(float((left-right).norm()), float(right.norm())*.05+1e-8)
                            else:
                                torch.testing.assert_close(left, right, rtol=2e-4, atol=2e-6)
                            if "engram_tables" in name:
                                self.assertGreater(float(left.abs().max()), 0)
                for optimizer in optimizers:
                    optimizer.step()
                for model in models[1:]:
                    for name, parameter in model.named_parameters():
                        torch.testing.assert_close(parameter.cpu(), reference[name].cpu(), **tolerance)
        finally:
            for model in models:
                for table in model.model.engram_tables.values():
                    table.close()

    def test_cpu_checkpoint_accumulation_sparse_updates(self):
        self.exercise(torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_four_policy_update_parity(self):
        self.exercise(torch.device("cuda", 0))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_bfloat16_four_policy_update_parity(self):
        self.exercise(torch.device("cuda", 0), torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
