import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch
from torch.nn import functional as F

from dsv41_train.lora import LoRAConfig, inject_lora
from dsv41_train.models.dsv4 import DeepSeekV41Config, DeepSeekV41ForCausalLM
from dsv41_train.models.qwen import Qwen3MoeForCausalLM
from dsv41_train.runtime import Runtime
from dsv41_train.sft_data import collate_examples
from qwen_helpers import tiny_config
from train_sft import evaluate, make_batches, parse_args, sft_loss, train_update, validate_args


def small_model(family):
    if family == "qwen":
        return Qwen3MoeForCausalLM(tiny_config(router_aux_loss_coef=10.0))
    config = replace(DeepSeekV41Config.tiny(), hidden_size=16, num_attention_heads=2,
                     head_dim=8, q_lora_rank=8, qk_rope_head_dim=4, index_n_heads=2,
                     index_head_dim=8, o_groups=1, o_lora_rank=8, moe_intermediate_size=32,
                     n_routed_experts=4, hc_mult=2, hc_sinkhorn_iters=5, engram_head_dim=4,
                     router_aux_loss_coef=10.0)
    return DeepSeekV41ForCausalLM(config)


def examples():
    return [{"input_ids": [1, 4, 7, 8, 2], "labels": [-100, -100, -100, 8, 2], "attention_mask": [1] * 5},
            {"input_ids": [1, 5, 3, 6, 9, 2], "labels": [-100, -100, 3, 6, 9, 2], "attention_mask": [1] * 6}]


class SFTTrainTests(unittest.TestCase):
    def setUp(self):
        previous = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous)
        torch.set_num_threads(1)
        torch.manual_seed(31)
        self.runtime = Runtime(torch.device("cpu"))

    def test_native_family_losses_are_shifted_answer_ce_without_router_aux(self):
        batch = collate_examples(examples(), pad_token_id=0)
        for family in ("dsv4", "qwen"):
            with self.subTest(family=family):
                model = small_model(family).eval()
                options = {"compute_aux_loss": False} if family == "dsv4" else {"output_router_logits": False}
                logits = model(batch["input_ids"], attention_mask=batch["attention_mask"], **options).logits
                expected = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                                           batch["labels"][:, 1:].reshape(-1), ignore_index=-100)
                torch.testing.assert_close(sft_loss(model, batch, family), expected)
                torch.testing.assert_close(torch.tensor(evaluate(model, [batch], family=family, runtime=self.runtime)), expected.detach())
                self.assertFalse(model.training)

    def test_accumulation_matches_one_batch_with_unequal_answer_lengths_full_and_lora(self):
        for family in ("dsv4", "qwen"):
            for mode in ("full", "lora"):
                with self.subTest(family=family, mode=mode):
                    accumulated = small_model(family)
                    if mode == "lora":
                        targets = LoRAConfig().targets if family == "dsv4" else ("self_attn.q_proj", "self_attn.v_proj")
                        inject_lora(accumulated, LoRAConfig(rank=2, alpha=4, targets=targets))
                    combined = copy.deepcopy(accumulated)
                    optimizers = [torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.01)
                                  for model in (accumulated, combined)]
                    sync = Mock()
                    micro = [collate_examples([example], pad_token_id=0) for example in examples()]
                    actual = train_update(accumulated, optimizers[0], micro, family=family, runtime=self.runtime, synchronize=sync)
                    expected = train_update(combined, optimizers[1], [collate_examples(examples(), pad_token_id=0)],
                                            family=family, runtime=self.runtime)
                    sync.assert_called_once()
                    self.assertAlmostEqual(actual["loss"], expected["loss"], places=5)
                    self.assertEqual(actual["supervised_tokens"], 6)
                    self.assertEqual(actual["input_tokens"], 11)
                    self.assertEqual((actual["input_tokens_per_example_min"], actual["input_tokens_per_example_max"]), (5, 6))
                    for (name, parameter), (_, reference) in zip(accumulated.named_parameters(), combined.named_parameters()):
                        torch.testing.assert_close(parameter, reference, rtol=2e-5, atol=2e-7, msg=name)
                        if not parameter.requires_grad:
                            self.assertIsNone(parameter.grad)

    def test_data_cursor_duplicates_only_cp_peers_and_resumes_exact_indices(self):
        data = [{"input_ids": [1, index, 2], "labels": [-100, index, 2], "attention_mask": [1] * 3}
                for index in range(10)]
        batches = []
        for rank in range(4):
            result, cursor = make_batches(data, 8, batch_size=1, count=2,
                                          runtime=Runtime(torch.device("cpu"), rank=rank, world_size=4),
                                          cp_size=2, pad_token_id=0)
            self.assertEqual(cursor, 12)
            batches.append([batch["input_ids"][0, 1].item() for batch in result])
        self.assertEqual(batches, [[8, 0], [8, 0], [9, 1], [9, 1]])

    def test_nonfinite_gradients_fail_before_optimizer_step(self):
        class BadGradient(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(()))
                self.weight.register_hook(lambda gradient: gradient * float("inf"))

            def forward(self, **kwargs):
                return SimpleNamespace(loss=self.weight.square())

        model = BadGradient()
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        optimizer.step = Mock(wraps=optimizer.step)
        with self.assertRaisesRegex(RuntimeError, "non-finite gradient"):
            train_update(model, optimizer, [collate_examples(examples(), pad_token_id=0)], family="qwen", runtime=self.runtime)
        optimizer.step.assert_not_called()
        self.assertEqual(model.weight.item(), 1)

    def test_lora_adamw_resume_preserves_optimizer_rng_and_next_update(self):
        from dsv41_train.sft_checkpoint import load_checkpoint, save_checkpoint

        model = small_model("qwen")
        inject_lora(model, LoRAConfig(rank=2, alpha=4, dropout=.2, targets=("self_attn.q_proj", "self_attn.v_proj")))
        restored = copy.deepcopy(model)
        optimizer, restored_optimizer = [torch.optim.AdamW([p for p in value.parameters() if p.requires_grad],
                                                            lr=.01, weight_decay=0, foreach=False)
                                          for value in (model, restored)]
        batches = [collate_examples([example], pad_token_id=0) for example in examples()]
        options = dict(family="qwen", runtime=self.runtime)
        train_update(model, optimizer, batches, **options)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "step-1"
            identity = {"test": "qwen-lora-sft"}
            save_checkpoint(path, model, optimizer, step=1, identity=identity, data_position=2)
            expected = train_update(model, optimizer, batches, **options)
            self.assertEqual(load_checkpoint(path, restored, restored_optimizer, identity=identity), (1, 2))
            actual = train_update(restored, restored_optimizer, batches, **options)
        self.assertEqual(actual, expected)
        for (name, parameter), (_, reference) in zip(restored.named_parameters(), model.named_parameters()):
            torch.testing.assert_close(parameter, reference, rtol=0, atol=0, msg=name)

    def test_default_dsv4_selection_is_full_backbone_and_optimizer_is_mode_specific(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            config = DeepSeekV41Config.tiny()
            (path / "config.json").write_text(json.dumps(config.to_dict()))
            (path / "model.safetensors.index.json").write_text("{}")
            for mode, optimizer in (("full", "sgd"), ("lora", "adamw")):
                args = parse_args(["--model-path", directory, "--data", "data.jsonl", "--output-dir", directory, "--mode", mode])
                validate_args(args, 1)
                self.assertEqual(args.num_layers, config.num_hidden_layers)
                self.assertEqual(args.optimizer, optimizer)
                self.assertEqual(args.steps, 20)
                self.assertEqual(args.seq_len, 4096)

            base = ["--model-path", directory, "--data", "data.jsonl", "--output-dir", directory]
            for options, optimizer in ((["--engram-offload"], "adamw"),
                                       (["--engram-offload", "--train-engram"], "sgd"),
                                       (["--engram-offload", "--mode", "full"], "sgd")):
                args = parse_args(base + options)
                validate_args(args, 1)
                self.assertEqual(args.optimizer, optimizer)
            for options in (["--engram-offload", "--train-engram", "--optimizer", "adamw"],
                            ["--engram-offload", "--mode", "full", "--optimizer", "adamw"]):
                with self.assertRaisesRegex(ValueError, "requires SGD"):
                    validate_args(parse_args(base + options), 1)
            with self.assertRaisesRegex(ValueError, "require the DSV4"):
                validate_args(parse_args(base + ["--family", "qwen", "--engram-offload"]), 1)
            for options in (["--engram-cache-mib", "1"],
                            ["--engram-offload", "--engram-cache-mib", "-1"]):
                with self.assertRaisesRegex(ValueError, "engram-cache-mib"):
                    validate_args(parse_args(base + options), 1)
            cached = parse_args(base + ["--engram-offload", "--engram-cache-mib", "32"])
            validate_args(cached, 1)
            self.assertEqual(cached.engram_cache_mib, 32)
            self.assertFalse(cached.optimizer_in_backward)
            enabled = base + ["--mode", "full", "--optimizer-in-backward"]
            validate_args(parse_args(enabled), 1)
            for options, message in ((["--gradient-accumulation-steps", "2"], "accumulation"),
                                     (["--optimizer", "adamw"], "requires DSV4 and SGD")):
                with self.assertRaisesRegex(ValueError, message):
                    validate_args(parse_args(enabled + options), 1)


if __name__ == "__main__":
    unittest.main()
