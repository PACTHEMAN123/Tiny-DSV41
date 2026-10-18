"""Selective local FP32 masters must survive strict, same-topology resume."""
import datetime
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

from dsv41_train.optim import install_sgd_in_backward, sgd_parameter_groups
from dsv41_train.rl.checkpoint import file_sha256, run_identity
from dsv41_train.sft_checkpoint import _collect, _validate_master, load_checkpoint, save_checkpoint

IDENTITY = {"base": "tiny-master-checkpoint", "data": "fixed"}


def fixture(mesh):
    model = nn.Module()
    model.head = nn.Parameter(distribute_tensor(torch.ones(4, 3, dtype=torch.bfloat16), mesh, [Shard(0)]))
    model.plain = nn.Parameter(distribute_tensor(torch.full((4, 3), .5, dtype=torch.bfloat16), mesh, [Shard(0)]))
    model.register_buffer("bias", torch.ones(1))
    optimizer = torch.optim.SGD(sgd_parameter_groups(model, ["head"]), lr=.01, foreach=False)
    hooks = install_sgd_in_backward(model, optimizer)
    return model, optimizer, hooks


def update(model):
    (model.head.sum() * .125 + model.plain.sum() * .5).backward()


def snapshot(model, optimizer):
    return [model.head.to_local().clone(), model.plain.to_local().clone(), model.bias.clone(),
            optimizer.state[model.head]["master_weight"].clone(), torch.get_rng_state().clone()]


def rewrite(path, edit, rank=0):
    file = path / f"rank-{rank}.pt"
    payload = torch.load(file, weights_only=True)
    edit(payload)
    torch.save(payload, file)
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["ranks"][rank].update(bytes=file.stat().st_size, sha256=file_sha256(file))
    (path / "manifest.json").write_text(json.dumps(manifest))


def distributed_reject(rank, folder):
    if sys.platform == "darwin":
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0")
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{folder}/rendezvous", rank=rank,
                            world_size=2, timeout=datetime.timedelta(seconds=60))
    try:
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("fsdp",))
        model, optimizer, _ = fixture(mesh)
        path = Path(folder) / "saved"
        save_checkpoint(path, model, optimizer, step=0, identity=IDENTITY, data_position=0)
        if rank == 1:
            rewrite(path, lambda p: p["optimizer"].__setitem__("head", {}), rank=1)
        dist.barrier()
        with torch.no_grad():
            model.head.to_local().fill_(99)
            optimizer.state[model.head]["master_weight"].fill_(77)
        before = snapshot(model, optimizer)
        try:
            load_checkpoint(path, model, optimizer, identity=IDENTITY)
        except ValueError as error:
            assert "invalid optimizer fields: head" in str(error)
        else:
            raise AssertionError("missing rank1 master accepted")
        assert all(torch.equal(a, b) for a, b in zip(snapshot(model, optimizer), before))
    finally:
        dist.destroy_process_group()


class SFTMasterCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.process = tempfile.TemporaryDirectory()
        if sys.platform == "darwin":
            os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0")
        dist.init_process_group("gloo", init_method=f"file://{cls.process.name}/rendezvous",
                                rank=0, world_size=1, timeout=datetime.timedelta(seconds=60))
        cls.mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("fsdp",))

    @classmethod
    def tearDownClass(cls):
        dist.destroy_process_group()
        cls.process.cleanup()

    def test_resume_restores_residual_and_dynamic_next_update_exactly(self):
        with tempfile.TemporaryDirectory() as folder:
            model, optimizer, hooks = fixture(self.mesh)
            update(model)
            master = optimizer.state[model.head]["master_weight"]
            self.assertFalse(torch.equal(master, model.head.to_local().float()))
            path = Path(folder) / "step1"
            save_checkpoint(path, model, optimizer, step=1, identity=IDENTITY, data_position=1)
            saved = snapshot(model, optimizer)
            payload = torch.load(path / "rank-0.pt", weights_only=True)
            self.assertEqual(payload["optimizer"]["plain"], {})
            self.assertNotIn("mesh", payload["optimizer"]["head"]["master_weight"]["layout"])
            update(model)
            expected, expected_draw = snapshot(model, optimizer), torch.rand(3)
            restored, other, restored_hooks = fixture(self.mesh)
            eager_master = other.state[restored.head]["master_weight"]
            self.assertEqual(load_checkpoint(path, restored, other, identity=IDENTITY), (1, 1))
            self.assertIsNot(other.state[restored.head]["master_weight"], eager_master)
            self.assertNotIsInstance(other.state[restored.head]["master_weight"], DTensor)
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(snapshot(restored, other), saved)))
            update(restored)
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(snapshot(restored, other), expected)))
            self.assertFalse(torch.equal(restored.head.to_local(), saved[0]))
            self.assertTrue(torch.equal(torch.rand(3), expected_draw))
            hooks.remove(); restored_hooks.remove()

    def test_invalid_live_master_rejected_before_creating_checkpoint(self):
        edits = {
            "missing": lambda m, o: o.state[m.head].clear(),
            "dtype": lambda m, o: o.state[m.head].update(master_weight=torch.ones(4, 3, dtype=torch.bfloat16)),
            "shape": lambda m, o: o.state[m.head].update(master_weight=torch.ones(12)),
            "nan": lambda m, o: o.state[m.head]["master_weight"].fill_(float("nan")),
            "mismatch": lambda m, o: o.state[m.head]["master_weight"].add_(1),
            "dtensor": lambda m, o: o.state[m.head].update(master_weight=m.head.detach().float()),
            "extra": lambda m, o: o.state[m.head].update(momentum_buffer=torch.zeros(4, 3)),
            "ordinary_extra": lambda m, o: o.state[m.plain].update(master_weight=torch.ones(4, 3)),
            "momentum": lambda m, o: o.param_groups[-1].update(momentum=.9),
        }
        with tempfile.TemporaryDirectory() as folder:
            for name, edit in edits.items():
                with self.subTest(name=name):
                    model, optimizer, hooks = fixture(self.mesh)
                    edit(model, optimizer)
                    path = Path(folder) / name
                    with self.assertRaises(ValueError):
                        save_checkpoint(path, model, optimizer, step=0, identity=IDENTITY, data_position=0)
                    self.assertFalse(path.exists())
                    hooks.remove()

    def test_invalid_saved_master_rejected_before_any_live_mutation(self):
        def state(payload): return payload["optimizer"]["head"]["master_weight"]
        edits = {
            "empty_fields": lambda p: p["optimizer"].__setitem__("head", {}),
            "missing_name": lambda p: p["optimizer"].pop("head"),
            "dtype": lambda p: state(p).update(value=state(p)["value"].to(torch.bfloat16)),
            "shape": lambda p: state(p).update(value=state(p)["value"].flatten()),
            "nan": lambda p: state(p)["value"].fill_(float("nan")),
            "inf": lambda p: state(p)["value"].fill_(float("inf")),
            "mismatch": lambda p: state(p)["value"].add_(1),
            "saved_weight": lambda p: p["parameters"]["head"].add_(1),
            "mesh": lambda p: state(p)["layout"].update(mesh=[0]),
            "device": lambda p: state(p)["layout"].update(device="cuda:0"),
            "extra": lambda p: p["optimizer"]["head"].update(step=state(p)),
            "record_extra": lambda p: state(p).update(unexpected=True),
            "ordinary_extra": lambda p: p["optimizer"]["plain"].update(master_weight=state(p)),
        }
        with tempfile.TemporaryDirectory() as folder:
            for name, edit in edits.items():
                with self.subTest(name=name):
                    model, optimizer, hooks = fixture(self.mesh)
                    path = Path(folder) / name
                    save_checkpoint(path, model, optimizer, step=0, identity=IDENTITY, data_position=0)
                    rewrite(path, edit)
                    with torch.no_grad():
                        model.head.to_local().fill_(99); model.bias.fill_(7)
                        optimizer.state[model.head]["master_weight"].fill_(77)
                    before = snapshot(model, optimizer)
                    with self.assertRaises(ValueError): load_checkpoint(path, model, optimizer, identity=IDENTITY)
                    self.assertTrue(all(torch.equal(a, b) for a, b in zip(snapshot(model, optimizer), before)))
                    hooks.remove()

    def test_comparison_casts_are_bounded_for_noncontiguous_master(self):
        weight = torch.ones(1025, 1025, dtype=torch.bfloat16).T
        master = weight.float()
        equal, sizes = torch.equal, []
        def observed(a, b): sizes.append(a.numel()); return equal(a, b)
        with patch("dsv41_train.sft_checkpoint.torch.equal", side_effect=observed):
            _validate_master(master, weight, "wide")
        self.assertGreater(len(sizes), 1)
        self.assertLessEqual(max(sizes), 2**20)

    def test_missing_master_on_one_rank_rejects_before_both_mutate(self):
        with tempfile.TemporaryDirectory() as folder:
            mp.spawn(distributed_reject, args=(folder,), nprocs=2, join=True)

    def test_full_sft_initialization_reads_weights_without_optimizer_resume(self):
        from dsv41_train.models.dsv4 import DeepSeekV41Config, DeepSeekV41ForCausalLM
        from dsv41_train.sft_source import SFTCheckpointSource
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            config_path = Path(__file__).resolve().parents[1] / "examples/rl/tiny-dsv41/config.json"
            (base / "config.json").write_bytes(config_path.read_bytes())
            model = DeepSeekV41ForCausalLM(DeepSeekV41Config.from_json(base / "config.json"))
            model.lm_head.weight = nn.Parameter(model.lm_head.weight.detach().to(torch.bfloat16))
            torch.save(model.state_dict(), base / "pytorch_model.bin")
            model.lm_head.weight = nn.Parameter(distribute_tensor(model.lm_head.weight.detach(), self.mesh, [Shard(0)]))
            optimizer = torch.optim.SGD(sgd_parameter_groups(model, ["lm_head.weight"]), lr=.01, foreach=False)
            hooks = install_sgd_in_backward(model, optimizer)
            (model.lm_head.weight.sum() * .125).backward()
            identity = run_identity(base, base / "config.json", False)
            identity["training_config"] = {"family": "dsv4", "mode": "full", "num_layers": model.config.num_hidden_layers}
            path = base / "saved"
            save_checkpoint(path, model, optimizer, step=1, identity=identity, data_position=1)
            source = SFTCheckpointSource(path, base)
            result = torch.empty_like(model.lm_head.weight.to_local())
            source.copy_tensor("lm_head.weight", result)
            self.assertTrue(torch.equal(result, model.lm_head.weight.to_local()))
            hooks.remove()


def cuda_validation(folder):
    """Two-rank torchrun gate: actual FSDP2 backward, exact resume, collective rejection."""
    from torch.distributed.fsdp import fully_shard
    rank = int(os.environ["RANK"])
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device, timeout=datetime.timedelta(seconds=180))
    try:
        if dist.get_world_size() != 2:
            raise ValueError("CUDA checkpoint gate requires exactly two ranks")
        mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("fsdp",))
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.head = nn.Linear(4, 4, bias=False, device=device, dtype=torch.bfloat16)
                self.plain = nn.Linear(4, 4, bias=False, device=device, dtype=torch.bfloat16)
                with torch.no_grad():
                    self.head.weight.fill_(1); self.plain.weight.fill_(.5)
            def forward(self, x): return self.head(x) + self.plain(x)
        def create():
            model = Model(); fully_shard(model, mesh=mesh)
            optimizer = torch.optim.SGD(sgd_parameter_groups(model, ["head.weight"]), lr=.01, foreach=False)
            hooks = install_sgd_in_backward(model, optimizer)
            return model, optimizer, hooks
        def step(model, hooks):
            hooks.reset()
            (model(torch.ones(1, 4, dtype=torch.bfloat16, device=device)).sum() * .125).backward()
            _collect({"error": None if bool(hooks.local_finite(device)) else "nonfinite actual FSDP update"})
        def values(model, optimizer):
            return [*(p.to_local().detach().cpu().clone() for p in model.parameters()),
                    optimizer.state[model.head.weight]["master_weight"].cpu().clone(),
                    torch.get_rng_state().clone(), torch.cuda.get_rng_state(device).clone()]
        def equal(actual, expected, label):
            _collect({"error": None if all(torch.equal(a, b) for a, b in zip(actual, expected)) else label})
        model, optimizer, hooks = create()
        initial = model.head.weight.to_local().detach().clone()
        step(model, hooks)
        master = optimizer.state[model.head.weight]["master_weight"]
        _collect({"error": None if (torch.equal(model.head.weight.to_local(), initial)
                    and not torch.equal(master, initial.float())) else "first sub-ULP residual was lost"})
        path = Path(folder) / "step1"
        save_checkpoint(path, model, optimizer, step=1, identity=IDENTITY, data_position=2)
        saved = values(model, optimizer)
        step(model, hooks); expected = values(model, optimizer)
        resumed, other, resumed_hooks = create()
        load_checkpoint(path, resumed, other, identity=IDENTITY)
        equal(values(resumed, other), saved, "restore differs")
        step(resumed, resumed_hooks)
        equal(values(resumed, other), expected, "next native FSDP update differs")
        _collect({"error": "accumulated sub-ULP updates did not change BF16" if
                    torch.equal(resumed.head.weight.to_local(), initial) else None})
        if rank == 1:
            rewrite(path, lambda p: p["optimizer"]["head.weight"]["master_weight"]["layout"].update(mesh=[1]), rank=1)
        dist.barrier()
        before = values(resumed, other)
        rejected = False
        try: load_checkpoint(path, resumed, other, identity=IDENTITY)
        except ValueError as error: rejected = "FP32 master layout differs" in str(error)
        _collect({"error": None if rejected else "corrupt rank1 master metadata accepted"})
        equal(values(resumed, other), before, "invalid load mutated live state")
        if rank == 0:
            with (Path(folder) / "results.json").open("x") as f:
                json.dump(dict(passed=True, native_fsdp2=True, world_size=2, sub_ulp_accumulation=True,
                    restored_exact=True, next_update_exact=True, invalid_rank1_rejected_without_mutation=True,
                    fresh_model_optimizer=True, separate_process_restart=False), f, indent=2)
                f.write("\n")
        hooks.remove(); resumed_hooks.remove()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    if "--cuda-dir" in sys.argv:
        import argparse
        parser = argparse.ArgumentParser(); parser.add_argument("--cuda-dir", type=Path, required=True)
        cuda_validation(parser.parse_args().cuda_dir)
    else:
        unittest.main()
