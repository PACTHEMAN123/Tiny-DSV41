"""Real full SFT snapshots must initialize a native model without source ranks."""

import datetime
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.tensor import DTensor

from dsv41_train.engram_offload import install_engram_sgd_hooks
from dsv41_train.models.dsv4 import DeepSeekV41Config, DeepSeekV41ForCausalLM
from dsv41_train.models.dsv4.parallel import apply_fsdp2, build_parallelism, synchronize_cp_gradients
from dsv41_train.rl.checkpoint import file_sha256, run_identity
from dsv41_train.runtime import Runtime
from dsv41_train.sft_checkpoint import save_checkpoint
from dsv41_train.sft_engram import EngramCheckpoint
from dsv41_train.sft_source import SFTCheckpointSource
from train_sft import train_update


ROOT = Path(__file__).resolve().parents[1]


def owner_start(model, name):
    parent = model.get_submodule(name.rsplit(".", 1)[0])
    return getattr(parent, "expert_start", getattr(parent, "row_start", None))


def save_fixture(rank, folder):
    torch.set_num_threads(1)
    if sys.platform == "darwin":
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0")
    dist.init_process_group("gloo", init_method=f"file://{folder}/rendezvous", rank=rank,
                            world_size=4, timeout=datetime.timedelta(seconds=120))
    try:
        root = Path(folder)
        base = torch.load(root / "pytorch_model.bin", weights_only=True)
        config = DeepSeekV41Config.from_json(root / "config.json")
        for cp_size, sparse in ((1, False), (2, True)):
            meshes, cp, dispatcher = build_parallelism(cp_size, 2, "cpu")
            model = DeepSeekV41ForCausalLM(config, cp, dispatcher, meshes.engram,
                                           sparse_engram_gradients=True, engram_offload=sparse)
            owned = {}
            for name, value in model.state_dict().items():
                offset = owner_start(model, name)
                owned[name] = base[name] if offset is None else base[name][offset:offset + value.shape[0]]
            model.load_state_dict(owned)
            apply_fsdp2(model, meshes)
            optimizer = torch.optim.SGD(model.parameters(), lr=.05, foreach=False)
            install_engram_sgd_hooks(model, optimizer)
            if sparse:
                model._sft_engram_checkpoint = EngramCheckpoint(model, optimizer, lambda name, ids: base[name][ids])
            runtime = Runtime(torch.device("cpu"), rank=rank, world_size=4)
            for step in range(2):
                ids = ((torch.arange(16) + 3 + rank // cp_size * 5 + step) % config.vocab_size).unsqueeze(0)
                labels = ids.clone()
                labels[:, :8] = -100
                train_update(model, optimizer, [{"input_ids": ids, "labels": labels, "attention_mask": torch.ones_like(ids)}],
                             family="dsv4", runtime=runtime, cp_size=cp_size,
                             synchronize=lambda: synchronize_cp_gradients(model, cp))
            path = root / f"cp{cp_size}"
            identity = run_identity(root, root / "data.jsonl", False)
            identity["training_config"] = {"family": "dsv4", "mode": "full", "num_layers": config.num_hidden_layers}
            save_checkpoint(path, model, optimizer, step=2, identity=identity, data_position=2)
            # Independent tiny-model oracle: native DTensor reconstruction, then
            # concatenate the EP/Engram owner regions. Production never gathers.
            local = {name: ((value.full_tensor() if isinstance(value, DTensor) else value).detach().clone(),
                            owner_start(model, name)) for name, value in model.state_dict().items()}
            states = [None] * 4
            dist.all_gather_object(states, local)
            if rank == 0:
                expected = {}
                for name in local:
                    parts = {}
                    for state in states:
                        value, offset = state[name]
                        if offset in parts:
                            assert torch.equal(parts[offset], value), name
                        parts[offset] = value
                    expected[name] = parts[None] if None in parts else torch.cat([parts[start] for start in sorted(parts)])
                torch.save(expected, root / f"expected-cp{cp_size}.pt")
            dist.barrier()
    finally:
        dist.destroy_process_group()


class SFTSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        config_values = json.loads((ROOT / "examples/rl/tiny-dsv41/config.json").read_text())
        config_values["vocab_size"] = 33  # Uneven dense FSDP shards; FP32 scales also exercise empty shards.
        (cls.root / "config.json").write_text(json.dumps(config_values))
        config = DeepSeekV41Config.from_json(cls.root / "config.json")
        torch.manual_seed(41)
        torch.save(DeepSeekV41ForCausalLM(config).state_dict(), cls.root / "pytorch_model.bin")
        (cls.root / "data.jsonl").write_text('{"prompt_ids":[1,4,7]}\n')
        mp.spawn(save_fixture, args=(str(cls.root),), nprocs=4, join=True)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_full_and_sparse_engram_snapshots_reconstruct_every_tensor(self):
        for cp in (1, 2):
            with self.subTest(cp=cp):
                source = SFTCheckpointSource(self.root / f"cp{cp}", self.root, chunk_bytes=128)
                expected = torch.load(self.root / f"expected-cp{cp}.pt", weights_only=True)
                original = torch.load(self.root / "pytorch_model.bin", weights_only=True)
                self.assertEqual(source.identity["step"], 2)
                self.assertEqual(source.identity["manifest_sha256"], file_sha256(self.root / f"cp{cp}/manifest.json"))
                for name in source.tensor_names:
                    result = torch.empty_like(expected[name])
                    source.copy_tensor(name, result)
                    torch.testing.assert_close(result, expected[name], rtol=0, atol=0, msg=name)
                for layer, mode in source.engram_modes.items():
                    name = f"model.engram_tables.{layer}.weight"
                    result = original[name].clone() if mode == "sparse" else torch.empty_like(expected[name])
                    for ids, rows in source.iter_engram_rows(layer):
                        self.assertLessEqual(rows.numel() * rows.element_size(), 128)
                        result.index_copy_(0, ids, rows)
                    torch.testing.assert_close(result, expected[name], rtol=0, atol=0)
                    self.assertFalse(torch.equal(result, original[name]))
                with self.assertRaisesRegex(ValueError, "destination"):
                    source.copy_tensor("lm_head.weight", torch.empty(1))

    def changed(self, edit, *, payload_rank=None):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "checkpoint"
        shutil.copytree(self.root / "cp2", path)
        manifest = json.loads((path / "manifest.json").read_text())
        if payload_rank is None:
            edit(manifest)
        else:
            file = path / f"rank-{payload_rank}.pt"
            payload = torch.load(file, weights_only=True)
            edit(payload)
            torch.save(payload, file)
            manifest["ranks"][payload_rank].update(bytes=file.stat().st_size, sha256=file_sha256(file))
        (path / "manifest.json").write_text(json.dumps(manifest))
        return path

    def test_rejects_base_hash_layout_and_owner_changes(self):
        variants = [
            (lambda m: m["identity"].update(model_config_sha256="bad"), "base identity"),
            (lambda m: m["ranks"][0].update(sha256="bad"), "checksum"),
            (lambda m: m["ranks"][0]["schema"]["parameters"]["lm_head.weight"].update(placements=["S(1)"]), "placement"),
            (lambda m: m["ranks"][1]["schema"]["ownership"]["model.layers.0.moe.routed"].update(expert_start=0), "mesh members|owners"),
            (lambda m: m["identity"]["training_config"].update(mode="lora"), "full DSV4"),
        ]
        for edit, message in variants:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    SFTCheckpointSource(self.changed(edit), self.root)

    def test_rejects_disagreeing_replicas_and_invalid_sparse_rows(self):
        name = "model.layers.0.attention_hc.fn"
        with self.assertRaisesRegex(ValueError, "replicated tensors differ"):
            SFTCheckpointSource(self.changed(lambda p: p["parameters"][name].add_(1), payload_rank=1), self.root)
        def corrupt_rows(payload):
            record = next(iter(payload["engram_rows"]["tables"].values()))
            record["indices"][0] = -1
        with self.assertRaisesRegex(ValueError, "Engram indices"):
            SFTCheckpointSource(self.changed(corrupt_rows, payload_rank=0), self.root)


if __name__ == "__main__":
    unittest.main()
