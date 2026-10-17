"""Single-GPU synchronous Engram cache gate; run with PYTHONPATH=. python FILE -v."""

from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from dsv41_train.engram_offload import install_engram_sgd_hooks
from dsv41_train.models.dsv4.model import RowShardedEmbedding
from dsv41_train.sft_checkpoint import load_checkpoint, save_checkpoint


@unittest.skipUnless(torch.cuda.is_available(), "requires one CUDA GPU")
class EngramCacheCudaTest(unittest.TestCase):
    width = 32
    slot_bytes = width * 2 + 8  # BF16 values and one CPU int64 tag.

    def setUp(self):
        self.device = torch.device("cuda", torch.cuda.current_device())

    def tearDown(self):
        torch.cuda.synchronize(self.device)

    def table(self, *, frozen=False, slots=3):
        table = RowShardedEmbedding(7, self.width, sparse_gradients=True, engram_offload=True).bfloat16()
        with torch.no_grad():
            table.weight.copy_(torch.arange(7 * self.width).reshape(7, self.width) / 1024)
        table.requires_grad_(not frozen)
        table.configure_cache(slots * self.slot_bytes, self.device)
        self.addCleanup(table.synchronize)
        return table

    def ids(self, values):
        return torch.tensor(values, device=self.device, dtype=torch.long)

    def optimizer(self, table):
        optimizer = torch.optim.SGD(table.parameters(), lr=.125, foreach=False)
        for handle in install_engram_sgd_hooks(table, optimizer):
            self.addCleanup(handle.remove)
        return optimizer

    @staticmethod
    def metrics(table):
        return table._cpu_lookup.cache_metrics

    def equal(self, actual, expected):
        torch.testing.assert_close(actual.detach().cpu(), expected.detach().cpu(), rtol=0, atol=0)

    def update(self, table, optimizer):
        optimizer.zero_grad(set_to_none=True)
        for coefficients in ((256., 1.), (1., -256.)):
            output = table(self.ids([0, 0]))
            output.backward(torch.tensor(coefficients, device=self.device, dtype=torch.bfloat16)
                            .view(2, 1).expand_as(output).contiguous())
        gradient = table.weight.grad
        self.assertEqual(gradient.device.type, "cpu")
        self.assertEqual(gradient._nnz(), 4)
        self.equal(gradient.float().coalesce().values(), torch.full((1, self.width), 2.))
        optimizer.step()

    def test_hits_collisions_duplicate_gradients_and_empty_lookup(self):
        table = self.table()
        with torch.no_grad():
            old = table(self.ids([[0, 1, 1]]))
            self.equal(table(self.ids([[0, 1, 1]])), old)
        ids = [3, 0, 6, 1, 3]  # Hits 0/1 must be read before 3/6 replace slot 0.
        output = table(self.ids(ids))
        self.equal(output, table.weight[ids])
        self.equal(old, table.weight[[0, 1, 1]].unsqueeze(0))
        gradient = torch.arange(1, 6, device=self.device, dtype=torch.bfloat16)[:, None].expand_as(output)
        output.backward(gradient)
        self.assertEqual(table.weight.grad._indices().tolist(), [ids])
        self.equal(table.weight.grad._values(), gradient)
        self.assertEqual(self.metrics(table), dict(hits=5, misses=6, transferred_rows=4,
                                                  resident_bytes=3 * self.slot_bytes))
        table.zero_grad(set_to_none=True)
        table(torch.empty((2, 0), device=self.device, dtype=torch.long)).sum().backward()
        self.assertEqual(table.weight.grad._nnz(), 0)

    def test_two_microbatch_sgd_matches_uncached_and_invalidates(self):
        cached, uncached = self.table(), self.table(slots=0)
        optimizers = [self.optimizer(table) for table in (cached, uncached)]
        expected = cached.weight[0].detach().clone()
        for _ in range(2):
            for table, optimizer in zip((cached, uncached), optimizers):
                self.update(table, optimizer)
            expected = (expected.float() - .25).bfloat16()
            self.equal(cached.weight, uncached.weight)
            self.equal(cached.weight[0], expected)
            before = self.metrics(cached)
            self.equal(cached(self.ids([0])), cached.weight[[0]])
            self.assertEqual(self.metrics(cached)["misses"], before["misses"] + 1)
            self.assertGreater(self.metrics(cached)["hits"], 0)

    def test_frozen_cache_survives_other_parameter_steps(self):
        table = self.table(frozen=True)
        other = nn.Parameter(torch.ones(1, device=self.device))
        optimizer = torch.optim.SGD([other], lr=.125, foreach=False)
        self.assertEqual(install_engram_sgd_hooks(table, optimizer), ())
        old = table(self.ids([0, 1]))
        for _ in range(2):
            other.grad = torch.ones_like(other)
            optimizer.step()
            self.equal(table(self.ids([0, 1])), old)
        self.assertEqual(self.metrics(table)["hits"], 4)
        self.assertEqual(self.metrics(table)["transferred_rows"], 2)
        self.assertIsNone(table.weight.grad)

    def test_two_streams_clear_reconfigure_without_borrowed_outputs(self):
        table = self.table(frozen=True)
        first, second = torch.cuda.Stream(device=self.device), torch.cuda.Stream(device=self.device)
        outputs = []
        def read(ids):
            outputs.append((table(self.ids(ids)), table.weight[ids].clone()))
        # No user-side wait between these calls: cache lifecycle owns its events.
        with torch.cuda.stream(first):
            read([0, 1, 1])
            read([0, 1])
        with torch.cuda.stream(second):
            read([3, 0, 6, 1, 3])
            table.clear_cache()
            read([0, 1])
            table.configure_cache(self.slot_bytes, self.device)
            read([2, 5, 2])
        with torch.cuda.stream(first):
            table.configure_cache(0, self.device)
            read([0, 1])
            table.configure_cache(3 * self.slot_bytes, self.device)
            read([0, 1])
        torch.cuda.synchronize(self.device)
        for output, expected in outputs:
            self.equal(output, expected)
        self.assertEqual(len({output.data_ptr() for output, _ in outputs}), len(outputs))
        self.assertEqual(self.metrics(table)["resident_bytes"], 3 * self.slot_bytes)

    def test_inference_warmup_then_train_miss_backward_and_clear(self):
        table = self.table()
        with torch.inference_mode():
            table(self.ids([0]))
        table(self.ids([0, 0, 3])).sum().backward()
        self.assertEqual(table.weight.grad._indices().tolist(), [[0, 0, 3]])
        self.equal(table.weight.grad._values(), torch.ones((3, self.width), dtype=torch.bfloat16))
        table.clear_cache()
        self.equal(table(self.ids([0])), table.weight[[0]])

    def test_warm_checkpoint_restore_then_next_update_matches(self):
        source, restored = self.table(), self.table()
        source_optimizer, restored_optimizer = self.optimizer(source), self.optimizer(restored)
        with torch.no_grad():
            old = restored(self.ids([0]))
            restored(self.ids([0]))
        original = restored.weight[[0]].detach().clone()
        self.update(source, source_optimizer)
        with tempfile.TemporaryDirectory() as temporary:
            path, identity = Path(temporary) / "step1", {"test": "CUDA Engram cache"}
            save_checkpoint(path, source, source_optimizer, step=1, identity=identity, data_position=2)
            before = self.metrics(restored)
            self.assertEqual(load_checkpoint(path, restored, restored_optimizer, identity=identity), (1, 2))
            self.equal(restored(self.ids([0])), source.weight[[0]])
            self.assertEqual(self.metrics(restored)["misses"], before["misses"] + 1)
            self.equal(old, original)
            self.update(source, source_optimizer)
            self.update(restored, restored_optimizer)
            self.equal(restored.weight, source.weight)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
