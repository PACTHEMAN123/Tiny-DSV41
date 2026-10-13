"""Cache coherence and sparse gradients, including eviction before backward."""

from collections import defaultdict
from copy import deepcopy
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch

from dsv41_train.models.dsv4.engram.cache import RowCache
from dsv41_train.models.dsv4.engram import EngramPolicy, RowShardedEmbedding
from dsv41_train.models.dsv4.engram.routing import RowRouter
from dsv41_train.optimization import build_optimizer, OptimizerPolicy
from dsv41_train.checkpoint import CheckpointManager, TrainingState
from dsv41_train.sharded_state import StateAdapter


class RowCacheTest(unittest.TestCase):
    def exercise(self, device, dtype):
        torch.manual_seed(19)
        weight = torch.nn.Parameter(torch.randn(17, 8, dtype=dtype))
        reference = torch.nn.Embedding(17, 8, sparse=True, dtype=dtype)
        reference.weight.data.copy_(weight)
        metrics = defaultdict(int)
        cache = RowCache(5 * 8 * weight.element_size(), metrics)
        optimizer = torch.optim.SGD([weight], lr=.125)
        expected_optimizer = torch.optim.SGD(reference.parameters(), lr=.125)
        for step in range(4):
            optimizer.zero_grad(set_to_none=True)
            expected_optimizer.zero_grad(set_to_none=True)
            losses, expected_losses = [], []
            for values in ([0, 1, 1, 6, 16], [], [5, 10, 15, 2], [0, 1, 1, 6, 16]):
                ids = torch.tensor(values, dtype=torch.long)
                actual = cache.lookup(weight, ids, device, lambda rows, target: rows.to(target))
                expected = reference(ids)
                torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
                losses.append(actual.float().square().sum() * .125)
                expected_losses.append(expected.float().square().sum() * .125)
            # Several evictions happen before any backward consumes saved rows.
            sum(losses).backward()
            sum(expected_losses).backward()
            self.assertTrue(weight.grad.is_sparse)
            torch.testing.assert_close(weight.grad.to_dense(), reference.weight.grad.to_dense(), rtol=0, atol=0)
            for parameter in (weight, reference.weight):
                parameter.grad = parameter.grad.float().coalesce().to(dtype)
            cache.before_step(weight)
            optimizer.step()
            expected_optimizer.step()
            if step % 2 == 0:
                cache.after_step(weight)
            # The other iterations test version-based fallback for external SGD.
            torch.testing.assert_close(weight, reference.weight, rtol=0, atol=0)
            self.assertLessEqual(cache.allocated_bytes, cache.budget)
        self.assertGreater(metrics["cache_hits"], 0)
        self.assertGreater(metrics["cache_invalidated_rows"], 0)

    def test_cpu_sparse_gradients_eviction_and_external_updates(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                self.exercise(torch.device("cpu"), dtype)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_sparse_gradients_eviction_and_external_updates(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                self.exercise(torch.device("cuda", 0), dtype)

    def test_sparse_update_preserves_unmodified_cache_rows(self):
        weight = torch.nn.Parameter(torch.randn(8, 4))
        metrics = defaultdict(int)
        cache = RowCache(8 * 4 * 4, metrics)
        lookup = lambda ids: cache.lookup(weight, torch.tensor(ids), torch.device("cpu"), lambda rows, _: rows)
        lookup([1, 2, 3])
        lookup([2, 2]).sum().backward()
        cache.before_step(weight)
        torch.optim.SGD([weight], lr=.1).step()
        cache.after_step(weight)
        self.assertEqual(metrics["cache_invalidated_rows"], 1)
        before = metrics["cache_hits"]
        torch.testing.assert_close(lookup([1, 2, 3]), weight[[1, 2, 3]])
        self.assertEqual(metrics["cache_hits"] - before, 2)

    def test_copy_and_parameter_replacement_invalidate_cache(self):
        metrics = defaultdict(int)
        cache = RowCache(64, metrics)
        weight = torch.nn.Parameter(torch.ones(8, 4))
        ids = torch.tensor([1, 2])
        lookup = lambda: cache.lookup(weight, ids, torch.device("cpu"), lambda rows, _: rows)
        lookup()
        with torch.no_grad():
            weight.copy_(torch.full_like(weight, 3))
        torch.testing.assert_close(lookup(), weight[ids])
        weight = torch.nn.Parameter(torch.full_like(weight, 7))
        torch.testing.assert_close(lookup(), weight[ids])
        cache.clear()
        self.assertEqual(cache.allocated_bytes, 0)

    def test_policy_rejects_invalid_placement_and_budget(self):
        self.assertEqual(EngramPolicy(cache_bytes=4096).device, "cuda")
        with self.assertRaises(ValueError):
            EngramPolicy("cpu", cache_bytes=-1)

    def test_batched_route_preserves_bounds_and_uneven_ownership(self):
        mesh = Mock(ndim=1)
        mesh.size.return_value = 3
        mesh.get_local_rank.return_value = 1
        mesh.get_group.return_value = None
        router = RowRouter(10, mesh)
        with patch("torch.distributed.all_to_all_single", side_effect=lambda out, data, **kw: out.copy_(data)), \
             patch.object(router, "exchange", side_effect=lambda data, *a, **kw: data):
            request = router.prepare(torch.tensor([[6, 4, 4]]))
            self.assertEqual(request.local_ids.tolist(), [2, 0, 0])
            self.assertEqual(request.send, [0, 3, 0])
            self.assertEqual(router.prepare(torch.empty(0, dtype=torch.long)).recv, [0, 0, 0])
            for ids in ([-1, 4], [4, 10]):
                with self.assertRaises(IndexError):
                    router.prepare(torch.tensor(ids))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_store_prefetch_checkpoint_and_pinned_budget(self):
        for mode in ("sync", "async"):
            with self.subTest(mode=mode):
                policy = EngramPolicy("cpu", mode, staging_bytes=4096, cache_bytes=2048)
                table = RowShardedEmbedding(67, 8, sparse_gradients=True, policy=policy).bfloat16()
                reference = RowShardedEmbedding(67, 8, sparse_gradients=True, policy=EngramPolicy("cpu")).bfloat16()
                self.addCleanup(table.close)
                self.addCleanup(reference.close)
                with torch.no_grad():
                    table.weight.normal_()
                    reference.weight.copy_(table.weight)
                optimizer = build_optimizer(table, OptimizerPolicy("sgd", .03125))
                ref_optimizer = build_optimizer(reference, OptimizerPolicy("sgd", .03125))
                saved = None
                for step in range(6):
                    optimizer.zero_grad(set_to_none=True)
                    ref_optimizer.zero_grad(set_to_none=True)
                    for ids in (torch.arange(700).remainder(67), torch.tensor([1, 1, 2, 3]),
                                torch.empty(0, dtype=torch.long), torch.tensor([1, 1, 2, 3])):
                        actual = table(ids.cuda())
                        expected = reference(ids.cuda())
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        (actual.float().square().sum() * .0625).backward()
                        (expected.float().square().sum() * .0625).backward()
                    torch.testing.assert_close(table.weight.grad.to_dense(), reference.weight.grad.to_dense(), rtol=0, atol=0)
                    optimizer.step()
                    ref_optimizer.step()
                    torch.testing.assert_close(table.weight, reference.weight, rtol=0, atol=0)
                    self.assertLessEqual(table.store.stager.allocated_bytes, 4096)
                    self.assertLessEqual(table.store.cache.allocated_bytes, 2048)
                    if step == 0:
                        saved = deepcopy(table.state_dict())
                    if step == 2:
                        with torch.no_grad():
                            table(torch.tensor([1, 2, 3], device="cuda"))
                        table.load_state_dict(saved)
                        reference.load_state_dict(saved)
                        self.assertEqual(table.store.cache.allocated_bytes, 0)
                self.assertEqual(set(table.state_dict()), {"weight"})
                self.assertGreater(table.metrics["cache_hits"], 0)
                self.assertLess(table.metrics["h2d_bytes"], reference.metrics["h2d_bytes"])
                table.close()
                self.assertEqual(table.store.cache.allocated_bytes, 0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_dcp_restores_over_live_cache_and_rejects_pending_lookup(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.table = RowShardedEmbedding(67, 8, sparse_gradients=True,
                    policy=EngramPolicy("cpu", "async", cache_bytes=4096))

            def synchronize_engram(self):
                self.table.synchronize()

        model = Model()
        self.addCleanup(model.table.close)
        model.table.weight.data.normal_()
        optimizer = build_optimizer(model, OptimizerPolicy("sgd", .125))
        ids = torch.tensor([1, 1, 3, 7], device="cuda")
        with tempfile.TemporaryDirectory() as folder:
            manager = CheckpointManager(folder, TrainingState(model, optimizer, model_config={},
                data_generator=torch.Generator(), state_adapter=StateAdapter(model)))
            model.table(ids).square().sum().backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            manager.save(1)
            expected = model.table.weight.detach().clone()
            with torch.no_grad():
                model.table.weight.add_(3)
                model.table(ids)
            self.assertGreater(model.table.store.cache.allocated_bytes, 0)
            manager.load(1)
            torch.testing.assert_close(model.table.weight, expected, rtol=0, atol=0)
            self.assertEqual(model.table.store.cache.allocated_bytes, 0)
            torch.testing.assert_close(model.table(ids).cpu(), expected[ids.cpu()], rtol=0, atol=0)
            ticket = model.table.prefetch(ids)
            with self.assertRaisesRegex(RuntimeError, "unconsumed"):
                manager.load(1)
            model.table.resolve(ticket)


if __name__ == "__main__":
    unittest.main()
