"""Resume must preserve rank-owned weights, optimizer history and the next draw."""

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

from dsv41_train.sft_checkpoint import load_checkpoint, save_checkpoint


IDENTITY = {"base": "tiny-original-checkpoint", "data": "fixed-data", "seed": 41}


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Parameter(torch.tensor([3.0, 4.0]), requires_grad=False)
        self.adapter = nn.Parameter(torch.tensor([0.5, 0.25]))
        self.register_buffer("routing_bias", torch.zeros(2))
        self.register_buffer("scratch", torch.ones(2), persistent=False)


def _step(model, optimizer):
    optimizer.zero_grad(set_to_none=True)
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.grad = torch.randn_like(parameter)
    optimizer.step()


def _local(value):
    return value.to_local() if isinstance(value, DTensor) else value


def _snapshot(model, optimizer):
    return ({name: _local(p).detach().clone() for name, p in model.named_parameters()},
            {name: {key: _local(value).clone() for key, value in optimizer.state.get(p, {}).items()}
             for name, p in model.named_parameters()})


def _assert_snapshot(actual, expected):
    for actual_section, expected_section in zip(actual, expected):
        assert actual_section.keys() == expected_section.keys()
        for name, value in actual_section.items():
            if isinstance(value, dict):
                assert value.keys() == expected_section[name].keys()
                for key, tensor in value.items():
                    torch.testing.assert_close(tensor, expected_section[name][key], rtol=0, atol=0)
            else:
                torch.testing.assert_close(value, expected_section[name], rtol=0, atol=0)


def _distributed_worker(rank, folder):
    torch.set_num_threads(1)
    if sys.platform == "darwin":
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0")
    dist.init_process_group("gloo", init_method=f"file://{folder}/rendezvous", rank=rank,
                            world_size=2, timeout=datetime.timedelta(seconds=60))
    try:
        mesh = init_device_mesh("cpu", (1, 2), mesh_dim_names=("expert_fsdp", "ep"))["expert_fsdp"]

        def model():
            result = nn.Module()
            result.table = nn.Embedding(3, 2)
            result.table.row_start, result.table.row_stop = rank * 3, (rank + 1) * 3
            result.table.global_num_embeddings = 6
            result.experts = nn.Module()
            result.experts.expert_start, result.experts.num_experts = rank, 1
            result.experts.global_num_experts = 2
            result.experts.weight = nn.Parameter(distribute_tensor(torch.full((2, 2), 10.0 + rank), mesh, [Shard(0)]))
            with torch.no_grad():
                result.table.weight.fill_(20 + rank)
            result.register_buffer("bias", torch.tensor(float(rank)))
            return result

        for kind in (torch.optim.SGD, torch.optim.AdamW):
            torch.manual_seed(41 + rank)
            original = model()
            options = {"lr": .01, "foreach": False}
            if kind is torch.optim.SGD:
                options["momentum"] = .9
            optimizer = kind(original.parameters(), **options)
            _step(original, optimizer)
            saved = _snapshot(original, optimizer)
            path = Path(folder) / kind.__name__
            save_checkpoint(path, original, optimizer, step=1, identity=IDENTITY,
                            data_position={"sample": rank + 7})
            _step(original, optimizer)
            expected = _snapshot(original, optimizer)
            expected_draw = torch.rand(3)
            resumed = model()
            resumed_optimizer = kind(resumed.parameters(), **options)
            parameter_ids = [id(p) for p in resumed.parameters()]
            step, cursor = load_checkpoint(path, resumed, resumed_optimizer, identity=IDENTITY)
            assert step == 1 and cursor == {"sample": rank + 7}
            assert parameter_ids == [id(p) for p in resumed.parameters()]
            _assert_snapshot(_snapshot(resumed, resumed_optimizer), saved)
            assert resumed.bias.item() == rank
            for value in resumed_optimizer.state[resumed.experts.weight].values():
                if value.ndim:
                    assert isinstance(value, DTensor)
            _step(resumed, resumed_optimizer)
            _assert_snapshot(_snapshot(resumed, resumed_optimizer), expected)
            torch.testing.assert_close(torch.rand(3), expected_draw, rtol=0, atol=0)

            # Only one rank has an incompatible ownership map: both must reject
            # before either rank mutates live tensors or optimizer state.
            with torch.no_grad():
                resumed.table.weight.fill_(99)
            if rank == 1:
                resumed.table.row_start = 0
            try:
                load_checkpoint(path, resumed, resumed_optimizer, identity=IDENTITY)
            except ValueError as exc:
                assert "layout differs" in str(exc)
            else:
                raise AssertionError("changed ownership was accepted")
            assert bool((resumed.table.weight == 99).all())
            dist.barrier()
        if rank == 1:
            with patch("dsv41_train.sft_checkpoint.atomic_save", side_effect=OSError("injected write failure")):
                try:
                    save_checkpoint(Path(folder) / "failed", resumed, resumed_optimizer, step=2,
                                    identity=IDENTITY, data_position=0)
                except ValueError:
                    pass
        else:
            try:
                save_checkpoint(Path(folder) / "failed", resumed, resumed_optimizer, step=2,
                                identity=IDENTITY, data_position=0)
            except ValueError:
                pass
        assert not (Path(folder) / "failed/manifest.json").exists()
    finally:
        dist.destroy_process_group()


class SFTCheckpointTests(unittest.TestCase):
    def test_adam_resume_matches_next_step_and_rng_without_copying_base(self):
        with tempfile.TemporaryDirectory() as folder:
            model = Model()
            optimizer = torch.optim.AdamW([model.adapter], lr=.01, foreach=False)
            torch.manual_seed(17)
            _step(model, optimizer)
            model.routing_bias.add_(2)
            path = Path(folder) / "step-1"
            save_checkpoint(path, model, optimizer, step=1, identity=IDENTITY, data_position={"next": 2})
            payload = torch.load(path / "rank-0.pt", weights_only=True)
            self.assertEqual(set(payload["parameters"]), {"adapter"})
            self.assertEqual(set(payload["buffers"]), {"routing_bias"})
            _step(model, optimizer)
            expected = _snapshot(model, optimizer)
            draw = torch.rand(5)
            restored = Model()
            other = torch.optim.AdamW([restored.adapter], lr=.01, foreach=False)
            self.assertEqual(load_checkpoint(path, restored, other, identity=IDENTITY), (1, {"next": 2}))
            torch.testing.assert_close(restored.routing_bias, torch.full((2,), 2.0))
            _step(restored, other)
            _assert_snapshot(_snapshot(restored, other), expected)
            torch.testing.assert_close(torch.rand(5), draw, rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, "File exists"):
                save_checkpoint(path, model, optimizer, step=2, identity=IDENTITY, data_position=3)

    def test_invalid_identity_file_or_layout_rejects_before_mutation(self):
        with tempfile.TemporaryDirectory() as folder:
            model = Model()
            optimizer = torch.optim.SGD([model.adapter], lr=.1)
            path = Path(folder) / "step"
            save_checkpoint(path, model, optimizer, step=0, identity=IDENTITY, data_position=0)
            with torch.no_grad():
                model.adapter.fill_(88)
            rng = torch.get_rng_state().clone()
            with self.assertRaisesRegex(ValueError, "identity differs"):
                load_checkpoint(path, model, optimizer, identity={"base": "wrong"})
            manifest = json.loads((path / "manifest.json").read_text())
            manifest["ranks"][0]["schema"]["parameters"]["adapter"]["dtype"] = "torch.float16"
            (path / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "layout differs"):
                load_checkpoint(path, model, optimizer, identity=IDENTITY)
            manifest["ranks"][0]["schema"]["parameters"]["adapter"]["dtype"] = "torch.float32"
            (path / "manifest.json").write_text(json.dumps(manifest))
            with (path / "rank-0.pt").open("r+b") as handle:
                handle.write(b"BAD!")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_checkpoint(path, model, optimizer, identity=IDENTITY)
            self.assertTrue(bool((model.adapter == 88).all()))
            self.assertTrue(torch.equal(torch.get_rng_state(), rng))
            (path / "manifest.json").unlink()
            with self.assertRaises(ValueError):
                load_checkpoint(path, model, optimizer, identity=IDENTITY)

    def test_rank_local_rows_and_subgroup_dtensor_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            mp.spawn(_distributed_worker, args=(folder,), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
