"""Ticket lifecycle on CPU; real stream/DMA regressions require CUDA."""

import unittest
import weakref
from unittest.mock import patch

import torch
from torch import nn

from dsv41_train.engram_offload import _RowCache, install_engram_sgd_hooks
from dsv41_train.models.dsv4.model import RowShardedEmbedding


class Event:
    def __init__(self):
        self.waits = 0

    def synchronize(self):
        self.waits += 1


def ticket(cache, weight):
    event = Event()
    result = (cache, cache._generation, weakref.ref(weight), cache._signature(weight),
              weight._version, 1, event)
    cache._ticket = result
    cache._copy_done = event
    return result


class EngramTicketTest(unittest.TestCase):
    def setUp(self):
        self.cache = _RowCache(36, torch.device("cpu"))
        self.weight = nn.Parameter(torch.arange(14).reshape(7, 2).bfloat16())
        self.cache._validate(self.weight)

    def test_clear_and_drain_discard_delayed_tickets(self):
        for operation in (self.cache.clear, self.cache.wait):
            pending = ticket(self.cache, self.weight)
            operation()
            self.assertIsNone(self.cache._consume_transfer(self.weight, pending))
            self.assertIsNone(self.cache._ticket)
            self.assertTrue((self.cache.tags == -1).all())
            self.assertGreater(pending[-1].waits, 0)

    def test_version_storage_parameter_and_owner_changes_reject_old_rows(self):
        for mutation in ("version", "storage", "parameter", "owner"):
            with self.subTest(mutation=mutation):
                pending = ticket(self.cache, self.weight)
                weight, cache = self.weight, self.cache
                with torch.no_grad():
                    if mutation == "version":
                        weight.add_(1)
                    elif mutation == "storage":
                        weight.data = weight.detach().clone()
                    elif mutation == "parameter":
                        weight = nn.Parameter(weight.detach())
                    else:
                        cache = _RowCache(36, torch.device("cpu"))
                self.assertIsNone(cache._consume_transfer(weight, pending))
                self.assertTrue((self.cache.tags == -1).all())
                self.assertGreater(pending[-1].waits, 0)

    def test_normal_consumer_only_enqueues_a_stream_dependency(self):
        pending = ticket(self.cache, self.weight)
        self.cache._staging = torch.ones(2, 2)
        with patch("torch.cuda.current_stream") as current:
            rows = self.cache._consume_transfer(self.weight, pending)
        current.return_value.wait_event.assert_called_once_with(pending[-1])
        self.assertEqual(pending[-1].waits, 0)
        self.assertEqual(rows.shape, (1, 2))

    def test_sgd_drains_before_update_and_invalidates_after(self):
        module = RowShardedEmbedding(7, 2, sparse_gradients=True, engram_offload=True).bfloat16()
        with torch.no_grad():
            module.weight.zero_()
        module.configure_cache(36, "cpu")
        optimizer = torch.optim.SGD(module.parameters(), lr=.5, foreach=False)
        install_engram_sgd_hooks(module, optimizer)
        module(torch.tensor([0])).sum().backward()
        lookup = module._cpu_lookup
        observations = []
        drain, clear = lookup.synchronize, lookup.clear_cache
        def before():
            observations.append(("before", module.weight[0].tolist()))
            drain()
        def after():
            observations.append(("after", module.weight[0].tolist()))
            clear()
        with patch.object(lookup, "synchronize", before), patch.object(lookup, "clear_cache", after):
            optimizer.step()
        self.assertEqual(observations, [("before", [0., 0.]), ("after", [-.5, -.5])])


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for pinned DMA and stream validation")
class EngramAsyncCudaTest(unittest.TestCase):
    def table(self, *, cached=True, trainable=True):
        module = RowShardedEmbedding(31, 2, sparse_gradients=True, engram_offload=True).bfloat16()
        with torch.no_grad():
            module.weight.copy_(torch.arange(62).reshape(31, 2) / 16)
        module.requires_grad_(trainable)
        module.configure_cache(36 if cached else 0, "cuda")
        self.addCleanup(module.synchronize)
        return module

    def test_chunked_transfers_and_hit_gradients_match_uncached_updates(self):
        cached, uncached = self.table(), self.table(cached=False)
        with torch.inference_mode():
            cached(torch.tensor([1], device="cuda"))
        optimizers = [torch.optim.SGD(m.parameters(), lr=.125, foreach=False) for m in (cached, uncached)]
        for module, optimizer in zip((cached, uncached), optimizers):
            install_engram_sgd_hooks(module, optimizer)
        for step in range(3):
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            for ids in (torch.arange(25), torch.tensor([24, 24, 22, 0, 0, 5])):
                ids = ids.cuda()
                outputs = [module(ids) for module in (cached, uncached)]
                torch.testing.assert_close(*outputs, rtol=0, atol=0)
                gradient = torch.arange(ids.numel() * 2, device="cuda").reshape(-1, 2).bfloat16() / 8
                for output in outputs:
                    output.backward(gradient)
            for optimizer in optimizers:
                optimizer.step()
            torch.testing.assert_close(cached.weight, uncached.weight, rtol=0, atol=0)
        cache = cached._cpu_lookup._cache
        self.assertTrue(cache._pinned.is_pinned())
        self.assertIsNone(cache._ticket)
        self.assertLessEqual(cache._pinned.numel() * cache._pinned.element_size(), 36)
        self.assertLessEqual(cache._staging.numel() * cache._staging.element_size(), 36)
        self.assertLessEqual(cache.metrics["resident_bytes"], 36)
        self.assertGreater(cache.metrics["hits"], 0)

    def test_cross_stream_collision_clear_and_reconfigure(self):
        module = self.table(trainable=False)
        first, second = torch.cuda.Stream(), torch.cuda.Stream()
        with torch.cuda.stream(first):
            original = module(torch.tensor([0, 1, 1], device="cuda"))
        with torch.cuda.stream(second):
            collision = module(torch.tensor([3, 0, 6, 1, 3], device="cuda"))
        module.clear_cache()  # Drains both copy and consumer events.
        first.synchronize()
        second.synchronize()
        torch.testing.assert_close(original.cpu(), module.weight[[0, 1, 1]], rtol=0, atol=0)
        torch.testing.assert_close(collision.cpu(), module.weight[[3, 0, 6, 1, 3]], rtol=0, atol=0)
        with torch.no_grad():
            module.weight.add_(1)
        module.configure_cache(24, "cuda")
        with torch.cuda.stream(first):
            refreshed = module(torch.tensor([0, 1], device="cuda"))
        module.synchronize()
        torch.testing.assert_close(refreshed.cpu(), module.weight[[0, 1]], rtol=0, atol=0)

    def test_real_inflight_copy_cannot_publish_after_update_or_clear(self):
        module = self.table(trainable=False)
        cache = module._cpu_lookup._cache
        cache._validate(module.weight)
        for invalidation in ("update", "clear"):
            pending = cache._begin_transfer(module.weight, torch.tensor([0, 1]))
            if invalidation == "update":
                with torch.no_grad():
                    module.weight.add_(1)
            else:
                module.clear_cache()
            self.assertIsNone(cache._consume_transfer(module.weight, pending))
            self.assertTrue((cache.tags == -1).all())
        torch.testing.assert_close(module(torch.tensor([0, 1], device="cuda")).cpu(),
                                   module.weight[[0, 1]], rtol=0, atol=0)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
