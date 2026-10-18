import copy
from contextlib import ExitStack, redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from torch.nn import functional as F

from dsv41_train.lora import LoRAConfig, inject_lora
from dsv41_train.models.dsv4 import DeepSeekV41Config, DeepSeekV41ForCausalLM
from dsv41_train.models.qwen import Qwen3MoeForCausalLM
from dsv41_train.runtime import Runtime
from dsv41_train.sft_data import collate_examples
from qwen_helpers import tiny_config
from train_sft import evaluate, load_model, make_batches, parse_args, sft_loss, train, train_update, validate_args


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

    def test_native_loaders_set_sft_indexer_policy_before_wrapping(self):
        for family in ("dsv4", "qwen"):
            for mode in ("full", "lora"):
                with self.subTest(family=family, mode=mode), ExitStack() as stack:
                    model = small_model(family)
                    args = parse_args(["--model-path", "unused", "--data", "unused", "--output-dir", "unused",
                                       "--family", family, "--mode", mode])
                    args.optimizer, args.num_layers = "sgd", model.config.num_hidden_layers
                    meshes = SimpleNamespace(engram=None)
                    prefix = f"dsv41_train.models.{family}"
                    stack.enter_context(patch(prefix + ".parallel.build_parallelism", return_value=(meshes, None, None)))
                    if family == "dsv4":
                        indexers = [layer.attention.indexer for layer in model.model.layers
                                    if layer.attention.indexer is not None]
                        self.assertTrue(indexers)
                        self.assertTrue(all(not indexer.stable_topk for indexer in indexers))

                        def released_loader(*_args, layer_loaded, **_kwargs):
                            for layer in model.model.layers:
                                layer_loaded(layer)
                            return model

                        def wrap_layer(layer, _meshes):
                            if layer.attention.indexer is not None:
                                self.assertTrue(layer.attention.indexer.stable_topk)

                        loader = stack.enter_context(patch(prefix + ".load_dsv41_backbone_window", side_effect=released_loader))
                        wrapped = stack.enter_context(patch(prefix + ".parallel.apply_fsdp2_layer", side_effect=wrap_layer))
                        stack.enter_context(patch(prefix + ".parallel.apply_fsdp2_root"))
                    else:
                        loader = stack.enter_context(patch(prefix + ".load_qwen3_moe", return_value=model))
                        wrapped = stack.enter_context(patch(prefix + ".parallel.apply_fsdp2"))
                        stack.enter_context(patch(prefix + ".parallel.ParallelParameters.collect",
                                                  return_value=SimpleNamespace(synchronize=lambda cp: None)))
                    loaded, _ = load_model(args, self.runtime)
                    self.assertIs(loaded, model)
                    loader.assert_called_once()
                    self.assertEqual(wrapped.call_count, len(model.model.layers) if family == "dsv4" else 1)

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

    def test_training_entry_no_save_and_default_checkpoint_exports(self):
        from dsv41_train.engram_offload import install_engram_sgd_hooks
        from dsv41_train.sft_checkpoint import load_checkpoint, save_checkpoint

        resident_results = []
        for family, mode, options in (("qwen", "full", []), ("qwen", "full", ["--save-every", "1"]),
                                      ("qwen", "full", ["--no-save"]), ("qwen", "lora", []),
                                      ("qwen", "lora", ["--no-save"]), ("dsv4", "lora", ["--no-save"]),
                                      ("dsv4", "full", []), ("dsv4", "full", ["--no-save"]),
                                      ("dsv4", "full", ["--no-save", "--train-engram"])):
            with self.subTest(family=family, mode=mode, options=options), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                base, output = root / "base", root / "output"
                base.mkdir()
                torch.manual_seed(31)
                model = small_model(family)
                if family == "dsv4":
                    model = DeepSeekV41ForCausalLM(model.config, sparse_engram_gradients=True, engram_offload=mode == "lora")
                if mode == "lora":
                    targets = LoRAConfig().targets if family == "dsv4" else ("self_attn.q_proj", "self_attn.v_proj")
                    inject_lora(model, LoRAConfig(rank=2, alpha=4, targets=targets))
                    if family == "dsv4":
                        model.model.engram_tables.requires_grad_(True)
                (base / "config.json").write_text(json.dumps(model.config.to_dict()))
                (base / "model.safetensors.index.json").write_text("{}")
                data = root / "data.jsonl"
                data.write_text('{"prompt":"test","answer":"answer"}\n')
                command = ["--model-path", str(base), "--data", str(data), "--output-dir", str(output),
                           "--family", family, "--mode", mode, "--steps", "2", "--seq-len", "16", *options]
                if mode == "lora":
                    command += ["--lora-rank", "2", "--lora-alpha", "4", "--lora-targets", ",".join(targets)]
                if family == "dsv4" and mode == "lora":
                    command += ["--train-engram", "--engram-offload"]
                initial = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
                with ExitStack() as stack, redirect_stdout(io.StringIO()):
                    for name in ("reset_peak_memory_stats", "synchronize", "max_memory_allocated", "max_memory_reserved"):
                        stack.enter_context(patch(f"torch.cuda.{name}", return_value=0))
                    stack.enter_context(patch("train_sft.ModelTokenizer", return_value=SimpleNamespace(eos_token_ids=[2])))
                    stack.enter_context(patch("train_sft.load_jsonl", return_value=examples()))
                    stack.enter_context(patch("train_sft.load_model", return_value=(model, lambda: None)))
                    save = stack.enter_context(patch("dsv41_train.sft_checkpoint.save_checkpoint", wraps=save_checkpoint))
                    bundle = stack.enter_context(patch("dsv41_train.sft_export.export_sft_bundle"))
                    hooks = stack.enter_context(patch("dsv41_train.engram_offload.install_engram_sgd_hooks",
                                                      wraps=install_engram_sgd_hooks))
                    summary = train(parse_args(command), self.runtime)
                    self.assertEqual(hooks.call_count, int(family == "dsv4"))
                    if family == "dsv4" and mode == "full":
                        self.assertIsNone(model._sft_engram_checkpoint)
                        resident_results.append({name: parameter.detach().clone() for name, parameter in model.named_parameters()})
                    disabled = "--no-save" in options
                    self.assertEqual(summary["checkpoint_saving"], not disabled)
                    self.assertEqual((summary["step"], summary["data_position"]), (2, 2))
                    self.assertTrue(any(not torch.equal(parameter, initial[name]) for name, parameter in model.named_parameters()))
                    bundle.assert_not_called()
                    if disabled:
                        save.assert_not_called()
                        self.assertEqual({path.name for path in output.iterdir()}, {"metrics.jsonl", "summary.json"})
                    else:
                        self.assertEqual(save.call_count, 2 if "--save-every" in options else 1)
                        self.assertEqual((output / "adapter.pt").is_file(), mode == "lora")
                        checkpoint = output / "checkpoint-00000002"
                        self.assertTrue((checkpoint / "manifest.json").is_file())
                        identity = json.loads((checkpoint / "manifest.json").read_text())["identity"]
                        if family == "dsv4":
                            self.assertEqual(identity["training_config"]["indexer_selection"],
                                             "stable_score_descending_index_ascending_v1")
                            legacy = copy.deepcopy(identity)
                            del legacy["training_config"]["indexer_selection"]
                            optimizer = save.call_args.args[2]
                            save_checkpoint(root / "legacy", model, optimizer, step=2, identity=legacy, data_position=2)
                            before_rejected_resume = {name: p.detach().clone() for name, p in model.named_parameters()}
                            with self.assertRaisesRegex(ValueError, "checkpoint identity differs"):
                                load_checkpoint(root / "legacy", model, optimizer, identity=identity)
                            for name, parameter in model.named_parameters():
                                torch.testing.assert_close(parameter, before_rejected_resume[name], rtol=0, atol=0)
                        else:
                            self.assertNotIn("indexer_selection", identity["training_config"])
                            if mode == "full":
                                self.assertEqual(identity["training_config"], {
                                    "family": "qwen", "mode": "full", "num_layers": None, "batch_size": 1,
                                    "gradient_accumulation_steps": 1, "seq_len": 16, "learning_rate": 1e-4,
                                    "optimizer": "sgd", "cp_size": 1, "ep_size": 1, "seed": 42,
                                    "gradient_checkpointing": False, "world_size": 1, "enable_thinking": False, "lora": None})
                        load = stack.enter_context(patch("dsv41_train.sft_checkpoint.load_checkpoint", wraps=load_checkpoint))
                        resumed = train(parse_args(command + ["--save-every", "0", "--no-save", "--resume", str(checkpoint),
                                                              "--steps", "3", "--output-dir", str(root / "resumed")]), self.runtime)
                        load.assert_called_once()
                        self.assertEqual((resumed["initial_step"], resumed["step"], resumed["data_position"]), (2, 3, 3))
                        self.assertEqual({path.name for path in (root / "resumed").iterdir()}, {"metrics.jsonl", "summary.json"})
        for name, parameter in resident_results[0].items():
            torch.testing.assert_close(parameter, resident_results[1][name], rtol=0, atol=0, msg=name)

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
                self.assertFalse(args.no_save)

            base = ["--model-path", directory, "--data", "data.jsonl", "--output-dir", directory]
            with self.assertRaisesRegex(ValueError, "no-save cannot be combined"):
                validate_args(parse_args(base + ["--no-save", "--save-every", "1"]), 1)
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
            self.assertEqual(parse_args(enabled).sgd_fp32_master_parameters, [])
            selected = parse_args(enabled + ["--sgd-fp32-master-parameter", "lm_head.weight",
                                             "--sgd-fp32-master-parameter", "model.embedding.weight"])
            validate_args(selected, 1)
            self.assertEqual(selected.sgd_fp32_master_parameters, ["lm_head.weight", "model.embedding.weight"])
            for command, message in ((base + ["--sgd-fp32-master-parameter", "lm_head.weight"], "requires optimizer-in-backward"),
                                      (enabled + ["--sgd-fp32-master-parameter", ""], "unique exact"),
                                      (enabled + ["--sgd-fp32-master-parameter", "lm_head.weight"] * 2, "unique exact")):
                with self.assertRaisesRegex(ValueError, message):
                    validate_args(parse_args(command), 1)
            for options, message in ((["--gradient-accumulation-steps", "2"], "accumulation"),
                                     (["--optimizer", "adamw"], "requires DSV4 and SGD")):
                with self.assertRaisesRegex(ValueError, message):
                    validate_args(parse_args(enabled + options), 1)


if __name__ == "__main__":
    unittest.main()
