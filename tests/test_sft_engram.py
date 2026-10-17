from copy import deepcopy
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn

from dsv41_train.engram_offload import install_engram_sgd_hooks
from dsv41_train.models.dsv4.checkpoint import ShardedSafeTensorReader
from dsv41_train.models.dsv4.model import RowShardedEmbedding
from dsv41_train.sft_engram import EngramCheckpoint, ReleasedEngramRows, build_engram_checkpoint


class SFTEngramTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.header, self.payload, self.original = {}, bytearray(), {}
        for layer in (1, 4):
            values = (torch.arange(7).float()[:, None].expand(7, 64) + layer).to(torch.float8_e4m3fn)
            scales = torch.tensor([[2., 4.]]).expand(7, 2).to(torch.float8_e8m0fnu)
            self.original[str(layer)] = (values.float().reshape(7, 2, 32) * scales.float()[:, :, None]).flatten(1).bfloat16()
            for suffix, tensor, dtype in (("weight", values, "F8_E4M3"), ("scale", scales, "F8_E8M0")):
                start = len(self.payload)
                self.payload.extend(tensor.contiguous().view(torch.uint8).numpy().tobytes())
                self.header[f"layers.{layer}.engram.embed.{suffix}"] = dict(dtype=dtype, shape=list(tensor.shape),
                                                                          data_offsets=[start, len(self.payload)])
        self.write_base()
        (self.folder / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {name: "rows.safetensors" for name in self.header}}))

    def write_base(self):
        encoded = json.dumps(self.header).encode()
        (self.folder / "rows.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + self.payload)

    def build(self, *, empty_owner=False):
        model = nn.Module()
        model.adapter = nn.Parameter(torch.ones(1))
        model.model = nn.Module()
        model.model.engram_tables = nn.ModuleDict()
        for name in ("1", "4"):
            mesh = Mock(ndim=1)
            mesh.size.return_value, mesh.get_local_rank.return_value = 2, 1
            table = RowShardedEmbedding(7, 64, mesh, sparse_gradients=True, engram_offload=True).bfloat16()
            with torch.no_grad():
                table.weight.copy_(self.original[name][4:7])
            if empty_owner:
                table.row_start = table.row_stop = 7
                table.weight = nn.Parameter(torch.empty(0, 64, dtype=torch.bfloat16))
            model.model.engram_tables[name] = table
        optimizer = torch.optim.SGD(model.parameters(), lr=.125, foreach=False)
        for handle in install_engram_sgd_hooks(model, optimizer):
            self.addCleanup(handle.remove)
        manager = build_engram_checkpoint(model, self.folder, optimizer)
        self.addCleanup(manager.remove)
        return model, optimizer, manager

    @staticmethod
    def update(model, optimizer, microbatches, *, device="cpu"):
        optimizer.zero_grad(set_to_none=True)
        for ids, coefficients in microbatches:
            ids = torch.tensor(ids, device=device)
            for table in model.model.engram_tables.values():
                values = table._lookup(ids)
                values.backward(torch.tensor(coefficients, dtype=values.dtype, device=device)[:, None].expand_as(values))
        optimizer.step()

    def assert_rows(self, actual, expected):
        for name, table in actual.model.engram_tables.items():
            torch.testing.assert_close(table.weight, expected[name], rtol=0, atol=0)

    def test_repeated_rows_microbatches_fresh_restore_and_older_rollback(self):
        model, optimizer, manager = self.build()
        initial = manager.state_dict()
        self.assertEqual(manager.parameter_names, {f"model.engram_tables.{layer}.weight" for layer in (1, 4)})
        self.update(model, optimizer, [([0, 0], [256, 1]), ([0, 0], [1, -256])])
        first = manager.state_dict()
        expected = {name: self.original[name][4:7].clone() for name in ("1", "4")}
        for values in expected.values():
            values[0] -= .25  # FP32 occurrence accumulation before BF16 SGD.
        self.assert_rows(model, expected)
        for record in first["tables"].values():
            self.assertEqual(record["indices"].tolist(), [0])
            self.assertEqual(record["values"].shape, (1, 64))
        fresh, fresh_optimizer, fresh_manager = self.build()
        fresh_manager.read_base_rows = Mock(side_effect=AssertionError("fresh restore should only apply saved rows"))
        fresh_manager.load_state_dict(first)
        self.assert_rows(fresh, expected)
        self.update(model, optimizer, [([2, 2], [1, 1])])
        self.update(fresh, fresh_optimizer, [([2, 2], [1, 1])])
        for name in expected:
            torch.testing.assert_close(model.model.engram_tables[name].weight,
                                       fresh.model.engram_tables[name].weight, rtol=0, atol=0)
        reads, read = [], manager.read_base_rows
        def observe(name, ids):
            reads.append((name, ids.tolist()))
            return read(name, ids)
        manager.read_base_rows = observe
        manager.load_state_dict(first)
        self.assert_rows(model, expected)
        self.assertEqual([ids for _, ids in reads], [[6], [6]])
        manager.load_state_dict(initial)
        self.assert_rows(model, {name: self.original[name][4:7] for name in expected})
        self.assertTrue(all(not ids.numel() for ids in manager.dirty.values()))

    def test_bad_payload_is_rejected_before_any_mutation_or_base_read(self):
        model, optimizer, manager = self.build()
        self.update(model, optimizer, [([0, 2], [1, 1])])
        good = manager.state_dict()
        before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        manager.read_base_rows = Mock(side_effect=AssertionError("bad payload must not read base rows"))
        name = next(iter(manager.parameter_names))
        for kind in ("schema", "dtype", "shape", "duplicate", "negative", "range", "nan", "missing", "extra", "format"):
            bad = deepcopy(good)
            record = bad["tables"][name]
            if kind == "schema":
                bad["schema"][name]["row_start"] = 0
            elif kind == "dtype":
                record["values"] = record["values"].float()
            elif kind == "shape":
                record["values"] = record["values"][:, :1]
            elif kind == "duplicate":
                record["indices"][1] = record["indices"][0]
            elif kind == "negative":
                record["indices"][0] = -1
            elif kind == "range":
                record["indices"][1] = 3
            elif kind == "nan":
                record["values"][0, 0] = float("nan")
            elif kind == "missing":
                del bad["tables"][name]
            elif kind == "extra":
                bad["tables"]["unknown.weight"] = record
            else:
                bad["format"] = True
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                manager.load_state_dict(bad)
            for key, parameter in model.named_parameters():
                torch.testing.assert_close(parameter, before[key], rtol=0, atol=0)
            for key, ids in manager.dirty.items():
                torch.testing.assert_close(ids, good["tables"][key]["indices"], rtol=0, atol=0)
        manager.read_base_rows.assert_not_called()

    def test_empty_owner_no_grad_and_frozen_models(self):
        model, optimizer, manager = self.build(empty_owner=True)
        optimizer.step()
        snapshot = manager.state_dict()
        manager.read_base_rows = Mock(side_effect=AssertionError("empty owner has no base reads"))
        manager.load_state_dict(snapshot)
        for record in snapshot["tables"].values():
            self.assertEqual(tuple(record["values"].shape), (0, 64))
        for table in model.model.engram_tables.values():
            table.requires_grad_(False)
        self.assertIsNone(build_engram_checkpoint(model, self.folder, optimizer))
        for table in model.model.engram_tables.values():
            table.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "native SGD"):
            EngramCheckpoint(model, torch.optim.AdamW(model.parameters()), None)

    def test_released_reads_only_requested_global_ranges_and_strict_metadata(self):
        _, _, manager = self.build()
        name = next(iter(manager.parameter_names))
        reader = ReleasedEngramRows(self.folder, manager.schema)
        original, calls = ShardedSafeTensorReader.tensor_rows, []
        def read(source, tensor_name, start, stop, device):
            calls.append((tensor_name.rsplit(".", 1)[-1], start, stop))
            return original(source, tensor_name, start, stop, device)
        with patch.object(ShardedSafeTensorReader, "tensor_rows", read):
            actual = reader(name, torch.tensor([0, 1, 6]))
        layer = name.split(".")[-2]
        torch.testing.assert_close(actual, self.original[layer][[0, 1, 6]], rtol=0, atol=0)
        self.assertEqual(calls, [(suffix, start, stop) for start, stop in ((0, 2), (6, 7)) for suffix in ("weight", "scale")])
        self.header[f"layers.{layer}.engram.embed.scale"]["dtype"] = "F8_E4M3"
        self.write_base()
        with self.assertRaisesRegex(ValueError, "released Engram layout"):
            reader(name, torch.tensor([4]))
        with self.assertRaisesRegex(ValueError, "bounded row chunk"):
            reader(name, torch.arange(3000))

    def _warm_cache_restore_and_next_update(self, device):
        model, optimizer, manager = self.build()
        for table in model.model.engram_tables.values():
            table.configure_cache(2**20, device)
        self.update(model, optimizer, [([0, 0, 2], [1, 1, 1])], device=device)
        snapshot = manager.state_dict()
        expected = {name: nn.Embedding.from_pretrained(table.weight.detach().float().clone(), freeze=False)
                    for name, table in model.model.engram_tables.items()}
        self.update(model, optimizer, [([1, 2], [1, 1])], device=device)
        with torch.no_grad():
            for table in model.model.engram_tables.values():
                table._lookup(torch.tensor([0, 1, 2], device=device))
        manager.load_state_dict(snapshot)
        ids = torch.tensor([0, 1, 1, 2], device=device)
        for name, table in model.model.engram_tables.items():
            torch.testing.assert_close(table._lookup(ids), expected[name](ids.cpu()).bfloat16().to(device), rtol=0, atol=0)
        self.update(model, optimizer, [(ids.tolist(), [1, 1, 1, 1])], device=device)
        for name, dense in expected.items():
            dense_optimizer = torch.optim.SGD(dense.parameters(), lr=.125)
            dense(ids.cpu()).sum().backward()
            dense_optimizer.step()
            torch.testing.assert_close(model.model.engram_tables[name].weight, dense.weight.bfloat16(), rtol=0, atol=0)

    def test_warm_cpu_cache_restore_and_next_update_match_dense_oracle(self):
        self._warm_cache_restore_and_next_update("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for GPU row cache")
    def test_warm_gpu_cache_restore_and_next_update_match_dense_oracle(self):
        self._warm_cache_restore_and_next_update("cuda")


if __name__ == "__main__":
    unittest.main()
