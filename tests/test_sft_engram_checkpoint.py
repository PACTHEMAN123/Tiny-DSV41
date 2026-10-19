"""Sparse Engram snapshots use the same rank validation barrier as dense SFT."""

from copy import deepcopy
import datetime
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from dsv41_train.rl.checkpoint import atomic_save, file_sha256
from dsv41_train.sft_checkpoint import load_checkpoint, save_checkpoint
import test_sft_engram as fixtures


IDENTITY = {"base": "verified-tiny-released-fp8", "mode": "sft", "topology": "fixed"}


def build(fixture):
    model, optimizer, manager = fixture.build()
    model._sft_engram_checkpoint = manager
    model.register_buffer("routing_bias", torch.zeros(1))
    return model, optimizer, manager


def step(model, optimizer, row):
    optimizer.zero_grad(set_to_none=True)
    model.adapter.grad = torch.ones_like(model.adapter)
    for table in model.model.engram_tables.values():
        table._lookup(torch.tensor([row, row])).sum().backward()
    optimizer.step()
    model.routing_bias.add_(1)


def snapshot(model):
    return {name: value.detach().clone() for name, value in (*model.named_parameters(), *model.named_buffers())}


def rewrite_payload(path, owner, payload):
    filename = f"rank-{owner}.pt"
    atomic_save(payload, path / filename)
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["ranks"][owner].update(sha256=file_sha256(path / filename), bytes=(path / filename).stat().st_size)
    (path / "manifest.json").write_text(json.dumps(manifest))


def rank_validation_worker(rank, folder):
    if sys.platform == "darwin":
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0")
    dist.init_process_group("gloo", init_method=f"file://{folder}/rendezvous", rank=rank,
                            world_size=2, timeout=datetime.timedelta(seconds=60))
    fixture = fixtures.SFTEngramTests()
    fixture.setUp()
    try:
        model, optimizer, manager = build(fixture)
        step(model, optimizer, 0)
        path = Path(folder) / "sparse"
        save_checkpoint(path, model, optimizer, step=1, identity=IDENTITY, data_position=2)
        if rank == 1:
            payload = torch.load(path / "rank-1.pt", weights_only=True)
            record = next(iter(payload["engram_rows"]["tables"].values()))
            record["indices"][0] = 3  # outside this owner's local [0, 3) rows
            rewrite_payload(path, rank, payload)
        dist.barrier()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(88)
        before, rng = snapshot(model), torch.get_rng_state().clone()
        try:
            load_checkpoint(path, model, optimizer, identity=IDENTITY)
        except ValueError as exc:
            assert "rank 1" in str(exc) and "Engram row" in str(exc)
        else:
            raise AssertionError("one rank's invalid sparse delta was accepted")
        for name, value in snapshot(model).items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    finally:
        fixture.doCleanups()
        dist.destroy_process_group()


class SparseSFTCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.SFTEngramTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def assert_snapshot(self, model, expected):
        for name, value in snapshot(model).items():
            torch.testing.assert_close(value, expected[name], rtol=0, atol=0)

    def test_sparse_save_fresh_restore_rollback_and_next_step(self):
        model, optimizer, manager = build(self.fixture)
        step(model, optimizer, 0)
        first = snapshot(model)
        path = self.fixture.folder / "step1"
        save_checkpoint(path, model, optimizer, step=1, identity=IDENTITY, data_position=2)
        payload = torch.load(path / "rank-0.pt", weights_only=True)
        schema = json.loads((path / "manifest.json").read_text())["ranks"][0]["schema"]
        self.assertEqual(set(payload["parameters"]), {"adapter"})
        self.assertEqual(set(payload["buffers"]), {"routing_bias"})
        self.assertEqual(schema["engram_rows"], manager.schema)
        self.assertTrue(manager.parameter_names.issubset(schema["parameters"]))
        self.assertTrue(manager.parameter_names.issubset(schema["optimizer"]["groups"][0]["params"]))
        for name in manager.parameter_names:
            self.assertEqual(payload["optimizer"][name], {})
            self.assertEqual(payload["engram_rows"]["tables"][name]["values"].shape, (1, 64))
        step(model, optimizer, 2)
        second = snapshot(model)
        restored, restored_optimizer, restored_manager = build(self.fixture)
        ids = [id(parameter) for parameter in restored.parameters()]
        self.assertEqual(load_checkpoint(path, restored, restored_optimizer, identity=IDENTITY), (1, 2))
        self.assertEqual(ids, [id(parameter) for parameter in restored.parameters()])
        self.assert_snapshot(restored, first)
        step(restored, restored_optimizer, 2)
        self.assert_snapshot(restored, second)
        load_checkpoint(path, model, optimizer, identity=IDENTITY)
        self.assert_snapshot(model, first)
        step(model, optimizer, 2)
        self.assert_snapshot(model, second)

    def test_rehashed_bad_rows_do_not_mutate_dense_rows_optimizer_or_rng(self):
        model, optimizer, manager = build(self.fixture)
        step(model, optimizer, 0)
        path = self.fixture.folder / "step1"
        save_checkpoint(path, model, optimizer, step=1, identity=IDENTITY, data_position=2)
        good = torch.load(path / "rank-0.pt", weights_only=True)
        step(model, optimizer, 2)
        before, rng = snapshot(model), torch.get_rng_state().clone()
        dirty = {name: indices.clone() for name, indices in manager.dirty.items()}
        for kind in ("dtype", "missing", "full_table"):
            bad = deepcopy(good)
            bad["parameters"]["adapter"].fill_(123)
            name = next(iter(manager.parameter_names))
            if kind == "dtype":
                bad["engram_rows"]["tables"][name]["values"] = bad["engram_rows"]["tables"][name]["values"].double()
            elif kind == "missing":
                del bad["engram_rows"]
            else:
                bad["parameters"][name] = model.model.engram_tables[name.split(".")[-2]].weight.detach().clone()
            rewrite_payload(path, 0, bad)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                load_checkpoint(path, model, optimizer, identity=IDENTITY)
            self.assert_snapshot(model, before)
            self.assertFalse(optimizer.state)
            torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
            for key, indices in manager.dirty.items():
                torch.testing.assert_close(indices, dirty[key], rtol=0, atol=0)

    def test_one_rank_bad_rows_abort_all_ranks_before_mutation(self):
        with tempfile.TemporaryDirectory() as folder:
            mp.spawn(rank_validation_worker, args=(folder,), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
