"""Actual SFT updates and snapshots initialize native RL without losing table edits."""

from copy import deepcopy
from contextlib import ExitStack, redirect_stdout
from dataclasses import asdict, replace
from datetime import timedelta
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from dsv41_train.engram_offload import install_engram_sgd_hooks
from dsv41_train.lora import LoRAConfig, adapter_base_identity, inject_lora, read_adapter
from dsv41_train.models.dsv4 import DeepSeekV41ForCausalLM
from dsv41_train.models.dsv4.attention import AttentionSinks
from dsv41_train.models.dsv4.mhc import HyperConnection
from dsv41_train.models.dsv4.parallel import apply_fsdp2, build_parallelism, synchronize_cp_gradients
from dsv41_train.rl.checkpoint import (add_engram_identity, add_lora_identity, atomic_save,
    file_sha256, run_identity, save_training_checkpoint)
from dsv41_train.rl.config import RLConfig
from dsv41_train.rl.family import load_model_config
from dsv41_train.rl.engram_state import baseline_identity
from dsv41_train.rl.runtime import make_model, train_groups
from dsv41_train.rl.sample import load_checkpoint_model
from dsv41_train.runtime import Runtime
from dsv41_train.sft_checkpoint import save_checkpoint
from dsv41_train.sft_engram import EngramCheckpoint
from dsv41_train.sft_export import _global_rows, export_sft_bundle, resolve_initial_bundle
from dsv41_train.rl.algorithm import response_log_probs
from dsv41_train.rl.rollout import RolloutGroup, Sample
from train_sft import parse_args, train, train_update
from engram_offload_cuda_parity import configuration, reference_torch

ROOT = Path(__file__).resolve().parents[1]


def place_bf16(model, device):
    islands = {id(p) for module in model.modules() if isinstance(module, (HyperConnection, AttentionSinks))
               for p in module.parameters(recurse=False)}
    for name, parameter in model.named_parameters():
        parameter.data = parameter.data.to(device="cpu" if ".engram_tables." in name else device,
            dtype=torch.float32 if id(parameter) in islands else torch.bfloat16)
    for module in model.modules():
        for name, buffer in module.named_buffers(recurse=False):
            setattr(module, name, buffer.to(device))


def create_base(folder, cuda=False):
    if cuda:
        (folder / "config.json").write_text(json.dumps({**asdict(configuration(16)), "dtype": "bfloat16",
                                                       "checkpoint_format": "native"}))
    else:
        (folder / "config.json").write_bytes((ROOT / "examples/rl/tiny-dsv41/config.json").read_bytes())
    torch.manual_seed(17)
    model = DeepSeekV41ForCausalLM(load_model_config(folder))
    if cuda:
        place_bf16(model, "cpu")
    torch.save(model.state_dict(), folder / "pytorch_model.bin")
    torch.save(model.model.hash.token_map, folder / "engram_token_map.pt")
    (folder / "data.jsonl").write_text('{"prompt_ids":[1,4,7]}\n')


def build(folder, distributed=False, device=torch.device("cpu")):
    config = load_model_config(folder)
    meshes, cp, dispatcher = build_parallelism(1, 1, device.type) if distributed else (None, None, None)
    model = DeepSeekV41ForCausalLM(config, cp, dispatcher, None if meshes is None else meshes.engram,
                                 sparse_engram_gradients=True, engram_offload=True)
    source = torch.load(folder / "pytorch_model.bin", weights_only=True)
    owned = dict(source)
    for name, table in model.model.engram_tables.items():
        key = f"model.engram_tables.{name}.weight"
        owned[key] = source[key][table.row_start:table.row_stop]
    model.load_state_dict(owned)
    inject_lora(model, LoRAConfig(rank=2, alpha=4))
    model.model.engram_tables.requires_grad_(True)
    if device.type == "cuda":
        place_bf16(model, device)
    if distributed:
        apply_fsdp2(model, meshes)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.1, foreach=False)
    install_engram_sgd_hooks(model, optimizer)
    model._sft_engram_checkpoint = EngramCheckpoint(model, optimizer, lambda name, ids: source[name][ids])
    return model, optimizer, cp


def update_and_save(folder, model, optimizer, cp=None):
    device = model.model.embedding.weight.device
    ids = torch.tensor([[1, 4, 7, 9, 3, 2] if device.type == "cpu" else [1, 4, 7, 9] * 4])
    labels = ids.clone()
    labels[:, :3] = -100
    runtime = Runtime(device, rank=dist.get_rank() if dist.is_initialized() else 0,
                      world_size=dist.get_world_size() if dist.is_initialized() else 1)
    for _ in range(2):
        train_update(model, optimizer, [dict(input_ids=ids, labels=labels, attention_mask=torch.ones_like(ids))],
                     family="dsv4", runtime=runtime,
                     synchronize=lambda: synchronize_cp_gradients(model, cp) if cp is not None else None)
    identity = run_identity(folder, folder / "data.jsonl", False)
    identity["training_config"] = dict(family="dsv4", mode="lora")
    checkpoint = folder / "step2"
    save_checkpoint(checkpoint, model, optimizer, step=2, identity=identity, data_position=2)
    return checkpoint


def distributed_export_worker(rank, directory, cuda):
    torch.set_num_threads(1)
    if sys.platform == "darwin":
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0")
    folder = Path(directory)
    device = torch.device("cuda", rank) if cuda else torch.device("cpu")
    if cuda:
        reference_torch()
        torch.cuda.set_device(device)
    dist.init_process_group("nccl" if cuda else "gloo", init_method=f"file://{folder}/rendezvous", rank=rank,
                            world_size=2, timeout=timedelta(seconds=90))
    try:
        model, optimizer, cp = build(folder, True, device)
        checkpoint = update_and_save(folder, model, optimizer, cp)
        export_sft_bundle(model, checkpoint, folder, folder / "distributed-bundle")
        import dsv41_train.sft_export as exporter
        original = exporter.save_adapter
        def failing_write(*args, **kwargs):
            original(*args, **kwargs)  # Both ranks first execute the actual adapter collectives.
            if rank == 0:
                raise OSError("injected rank-zero adapter write failure")
        with patch.object(exporter, "save_adapter", failing_write):
            try:
                export_sft_bundle(model, checkpoint, folder, folder / "failed-bundle")
            except ValueError as error:
                assert "rank-zero adapter write failure" in str(error), error
            else:
                raise AssertionError("rank-zero write failure was not broadcast")
        assert not (folder / "failed-bundle/manifest.json").exists()
    finally:
        dist.destroy_process_group()


class SFTBundleTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        create_base(self.folder)
        self.model, self.optimizer, _ = build(self.folder)
        self.checkpoint = update_and_save(self.folder, self.model, self.optimizer)
        self.bundle = self.folder / "bundle"

    def config(self, bundle=None):
        return RLConfig(rollout_engine="native-pytorch", optimizer="sgd", train_engram=True,
                        trainable_scope="lora", lora_rank=2, lora_alpha=4, learning_rate=.1,
                        initial_adapter=str(bundle or self.bundle), kl_beta=.03)

    def test_sft_checkpoint_bundle_three_roles_and_rl_update(self):
        export_sft_bundle(self.model, self.checkpoint, self.folder, self.bundle)
        payload = read_adapter(self.bundle / "adapter.pt", base_identity=adapter_base_identity(self.folder))
        self.assertEqual(payload["format_version"], 2)
        config = self.config()
        model_config = load_model_config(self.folder)
        learner = make_model(self.folder, model_config, config, ("cpu",), False)
        baseline = learner.export_engram_baseline()
        reference = make_model(self.folder, model_config, config, ("cpu",), False,
                               engram_weights=baseline, role="reference").eval().requires_grad_(False)
        actor = make_model(self.folder, model_config, config, ("cpu",), False,
                           engram_weights=baseline, role="actor").eval()
        tokens = torch.tensor([[1, 4, 7, 9]])
        with torch.no_grad():
            wanted = self.model.eval()(tokens, compute_aux_loss=False).logits
            for policy in (learner, reference, actor):
                for name, parameter in policy.backend.named_parameters():
                    torch.testing.assert_close(parameter, dict(self.model.named_parameters())[name], rtol=0, atol=0)
                torch.testing.assert_close(policy(tokens), wanted, rtol=0, atol=0)
            samples = [Sample([1, 4, 7], response, response_log_probs(actor, [1, 4, 7], response).tolist(), reward, 0)
                       for response, reward in (([3, 4], 0.), ([5, 6], 1.))]
        initial = {name: p.detach().clone() for name, p in reference.named_parameters()}
        optim = torch.optim.SGD([p for p in learner.parameters() if p.requires_grad], lr=.1, foreach=False)
        install_engram_sgd_hooks(learner, optim)
        metrics = train_groups(learner, reference, optim, [RolloutGroup(samples, 0, 0)], config)
        self.assertEqual(metrics["nonzero_advantages"], 2)
        self.assertTrue(any(not torch.equal(p, initial[name]) for name, p in learner.named_parameters()))
        for name, parameter in reference.named_parameters():
            torch.testing.assert_close(parameter, initial[name], rtol=0, atol=0)
        with torch.no_grad():
            torch.testing.assert_close(reference(tokens), wanted, rtol=0, atol=0)
        resolved = resolve_initial_bundle(config)
        identity = add_engram_identity(add_lora_identity(
            run_identity(self.folder, self.folder / "data.jsonl", False), resolved, model_config), resolved)
        checkpoint = self.folder / "rl-step1.pt"
        save_training_checkpoint(checkpoint, learner, optim, step=1, config=config, identity=identity,
            next_group_id=1, rollout_rng_state=torch.get_rng_state(), dropped_groups=0)
        restored, restored_config, version, _ = load_checkpoint_model(model_dir=self.folder, checkpoint=checkpoint)
        self.assertEqual(version, 1)
        self.assertTrue(Path(restored_config.initial_adapter).is_file())
        for name, parameter in restored.named_parameters():
            torch.testing.assert_close(parameter, dict(learner.named_parameters())[name], rtol=0, atol=0)
        with torch.no_grad():
            torch.testing.assert_close(restored(tokens), learner(tokens), rtol=0, atol=0)

    def test_prefix_training_entry_finishes_without_bundle_export(self):
        (self.folder / "model.safetensors.index.json").write_text("{}")
        args = parse_args(["--model-path", str(self.folder), "--data", str(self.folder / "data.jsonl"),
            "--output-dir", str(self.folder / "prefix-run"), "--num-layers", "1", "--steps", "2",
            "--resume", str(self.checkpoint), "--train-engram", "--engram-offload"])
        # The real prefix loader also narrows model.config, which must not imply a full backbone.
        from dsv41_train.models.dsv4.checkpoint import _prefix_config
        self.model.config = _prefix_config(self.folder, 1)
        example = dict(input_ids=[1, 4, 7, 2], labels=[-100, -100, 7, 2], attention_mask=[1] * 4)
        with ExitStack() as stack, redirect_stdout(io.StringIO()):
            for name in ("reset_peak_memory_stats", "synchronize"):
                stack.enter_context(patch(f"torch.cuda.{name}"))
            for name in ("max_memory_allocated", "max_memory_reserved"):
                stack.enter_context(patch(f"torch.cuda.{name}", return_value=0))
            for name, value in (("ModelTokenizer", SimpleNamespace(eos_token_ids=[2])),
                                ("load_jsonl", [example]), ("evaluate", 1.),
                                ("load_model", (self.model, lambda: None))):
                stack.enter_context(patch(f"train_sft.{name}", return_value=value))
            stack.enter_context(patch("dsv41_train.sft_checkpoint.load_checkpoint", return_value=(2, 2)))
            stack.enter_context(patch("dsv41_train.sft_engram.build_engram_checkpoint"))
            export = stack.enter_context(patch("dsv41_train.sft_export.export_sft_bundle"))
            summary = train(args, Runtime(torch.device("cpu")))
        self.assertEqual(summary["step"], 2)
        export.assert_not_called()
        events = (self.folder / "prefix-run/metrics.jsonl").read_text()
        self.assertIn('"event": "sft_bundle_unavailable"', events)

    def test_reject_identity_full_partial_layers_dropout_and_stale_live_state(self):
        manifest_path = self.checkpoint / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        changed = deepcopy(manifest)
        changed["identity"]["model_config_sha256"] = "wrong"
        manifest_path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "identity"):
            export_sft_bundle(self.model, self.checkpoint, self.folder, self.bundle)
        manifest_path.write_text(json.dumps(manifest))
        self.model.model.embedding.weight.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "full SFT"):
            export_sft_bundle(self.model, self.checkpoint, self.folder, self.bundle)
        self.model.model.embedding.weight.requires_grad_(False)
        self.model.model.layer_ids = [0]
        with self.assertRaisesRegex(ValueError, "same-layer"):
            export_sft_bundle(self.model, self.checkpoint, self.folder, self.bundle)
        self.model.model.layer_ids = list(range(self.model.config.num_hidden_layers))
        module = next(module for module in self.model.modules() if hasattr(module, "lora_a"))
        module.config = replace(module.config, dropout=.1)
        with self.assertRaisesRegex(ValueError, "dropout"):
            export_sft_bundle(self.model, self.checkpoint, self.folder, self.bundle)
        module.config = replace(module.config, dropout=0)
        with torch.no_grad():
            module.lora_b.add_(1)
        with self.assertRaisesRegex(ValueError, "live adapter"):
            export_sft_bundle(self.model, self.checkpoint, self.folder, self.bundle)
        self.assertFalse(self.bundle.exists())

    def test_incomplete_corrupt_or_mismatched_bundle_rejected(self):
        export_sft_bundle(self.model, self.checkpoint, self.folder, self.bundle)
        for kind in ("manifest", "adapter", "engram", "identity"):
            folder = self.folder / kind
            shutil.copytree(self.bundle, folder)
            if kind == "manifest":
                (folder / "manifest.json").unlink()
            elif kind == "adapter":
                with (folder / "adapter.pt").open("ab") as handle:
                    handle.write(b"mismatched-adapter")
            elif kind == "engram":
                (folder / "engram_delta.pt").unlink()
            else:
                path = folder / "manifest.json"
                value = json.loads(path.read_text())
                value["base"]["source"]["model_config_sha256"] = "wrong"
                path.write_text(json.dumps(value))
            with self.subTest(kind=kind), self.assertRaises((ValueError, FileNotFoundError)):
                resolve_initial_bundle(self.config(folder))
        with self.assertRaisesRegex(ValueError, "conflicting"):
            resolve_initial_bundle(replace(self.config(), initial_engram_delta="other.pt"))
        config_path = self.folder / "config.json"
        config_path.write_text(config_path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "adapter base identity"):
            make_model(self.folder, load_model_config(self.folder), self.config(), ("cpu",), False)

    def test_global_uneven_empty_owners_missing_overlap_and_partial_files(self):
        manifest = json.loads((self.checkpoint / "manifest.json").read_text())
        source = torch.load(self.checkpoint / "rank-0.pt", weights_only=True)
        base = baseline_identity(self.folder, load_model_config(self.folder), torch.float32, False, 0)
        # Repartition only the touched rows. Owner 1 has no hits; no full table is read.
        owners = []
        for rank, bounds in enumerate(((0, 255), (255, 256))):
            payload, record = deepcopy(source), deepcopy(manifest["ranks"][0])
            for name, layout in payload["engram_rows"]["schema"].items():
                start, stop = bounds
                rows = payload["engram_rows"]["tables"][name]
                mask = (rows["indices"] >= start) & (rows["indices"] < stop)
                rows.update(indices=rows["indices"][mask] - start, values=rows["values"][mask])
                layout.update(row_start=start, row_stop=stop, local_shape=[stop - start, layout["global_shape"][1]])
            record["schema"]["engram_rows"] = deepcopy(payload["engram_rows"]["schema"])
            path = self.checkpoint / f"rank-{rank}.pt"
            atomic_save(payload, path)
            record.update(file=path.name, bytes=path.stat().st_size, sha256=file_sha256(path))
            owners.append(record)
        manifest.update(world_size=2, ranks=owners)
        merged = _global_rows(self.checkpoint, manifest, base)
        for name, record in merged["tables"].items():
            original = source["engram_rows"]["tables"][f"model.engram_tables.{name}.weight"]
            torch.testing.assert_close(record["indices"], original["indices"], rtol=0, atol=0)
            torch.testing.assert_close(record["values"], original["values"], rtol=0, atol=0)
        missing = deepcopy(manifest)
        missing["ranks"] = missing["ranks"][:1]
        with self.assertRaisesRegex(ValueError, "cover"):
            _global_rows(self.checkpoint, missing, base)
        bad = torch.load(self.checkpoint / "rank-1.pt", weights_only=True)
        for layout in bad["engram_rows"]["schema"].values():
            layout.update(row_start=254, local_shape=[2, layout["global_shape"][1]])
        atomic_save(bad, self.checkpoint / "rank-1.pt")
        owners[1].update(bytes=(self.checkpoint / "rank-1.pt").stat().st_size,
                         sha256=file_sha256(self.checkpoint / "rank-1.pt"))
        owners[1]["schema"]["engram_rows"] = bad["engram_rows"]["schema"]
        with self.assertRaisesRegex(ValueError, "overlap"):
            _global_rows(self.checkpoint, manifest, base)
        (self.checkpoint / "rank-1.pt").unlink()
        with self.assertRaises(FileNotFoundError):
            _global_rows(self.checkpoint, manifest, base)

    def test_real_collectives_and_rank_zero_write_error_broadcast(self):
        cuda = torch.cuda.device_count() >= 2
        if not cuda and tuple(int(part) for part in torch.__version__.split(".")[:2]) < (2, 14):
            self.skipTest("requires two CUDA devices or PyTorch CPU FSDP support")
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            create_base(folder, cuda)
            mp.spawn(distributed_export_worker, args=(directory, cuda), nprocs=2, join=True)
            self.assertTrue((folder / "distributed-bundle/manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
