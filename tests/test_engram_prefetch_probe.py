from concurrent.futures import Future
import threading
import unittest
from unittest.mock import patch

import torch

from dsv41_train.models.dsv4.engram import EngramPolicy, RowShardedEmbedding
from engram_prefetch_probe import prefetch_probe
from summarize_engram_prefetch import diagnostics


class PrefetchProbeTest(unittest.TestCase):
    def make_table(self, mode):
        table = RowShardedEmbedding(11, 4, sparse_gradients=True,
                                    policy=EngramPolicy("cpu", mode))
        table.weight.data.normal_()
        self.addCleanup(table.close)
        return table

    def test_ready_lookup_preserves_rows_gradients_and_restores_methods(self):
        for mode in ("sync", "async"):
            table = self.make_table(mode)
            ids = torch.tensor([1, 3, 3, 8])
            expected = table(ids)
            expected.square().sum().backward()
            gradient = table.weight.grad.to_dense()
            table.weight.grad = None
            samples = []
            with prefetch_probe({"3": table}, samples):
                ticket = table.prefetch(ids)
                if isinstance(ticket.result, Future):
                    ticket.result.result(timeout=5)
                actual = table.resolve(ticket)
                actual.square().sum().backward()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(table.weight.grad.to_dense(), gradient, rtol=0, atol=0)
            self.assertTrue(samples[0]["ready_before_resolve"])
            self.assertGreaterEqual(samples[0]["ready_margin_ms"], 0)
            for owner, method in ((table, "prefetch"), (table, "resolve"),
                                  (table.store, "lookup"), (table.router, "complete")):
                self.assertNotIn(method, owner.__dict__)

    def test_late_lookup_is_not_counted_ready(self):
        table = self.make_table("async")
        entered, release = threading.Event(), threading.Event()
        lookup = table.store.lookup

        def delayed(*args, **kwargs):
            entered.set()
            if not release.wait(5):
                raise RuntimeError("test lookup timed out")
            return lookup(*args, **kwargs)

        samples = []
        try:
            with patch.object(table.store, "lookup", delayed), prefetch_probe({"1": table}, samples):
                ticket = table.prefetch(torch.tensor([2, 2]))
                self.assertTrue(entered.wait(5))
                original = ticket.result.result

                def release_on_consumption(*args, **kwargs):
                    release.set()
                    return original(*args, **kwargs)

                with patch.object(ticket.result, "result", release_on_consumption):
                    table.resolve(ticket).sum().backward()
            self.assertFalse(samples[0]["ready_before_resolve"])
            self.assertFalse(samples[0]["future_done_at_resolve"])
            self.assertLess(samples[0]["ready_margin_ms"], 0)
            self.assertGreater(samples[0]["future_wait_ms"], 0)
        finally:
            release.set()

    def test_restores_on_failure(self):
        table = self.make_table("sync")
        with self.assertRaisesRegex(RuntimeError, "diagnostic failure"):
            with prefetch_probe({"1": table}, []):
                raise RuntimeError("diagnostic failure")
        self.assertNotIn("prefetch", table.__dict__)
        self.assertNotIn("lookup", table.store.__dict__)

    def test_summary_keeps_readiness_separate_from_routing(self):
        ranks = []
        for rank in range(2):
            samples = []
            for step in range(2):
                sample = dict(table="1", step=step, request=step, prefetch_return=1,
                              store_ready=2, ready_before_resolve=rank == 0,
                              future_done_at_resolve=rank == 0, ready_margin_ms=1 if rank == 0 else -1,
                              future_wait_ms=rank + step, route_host_ms=10, prefetch_host_ms=3, store_ms=2)
                samples.append(sample)
            ranks.append({"prefetch_diagnostic": {"steps": 2, "step_ms": [20, 30], "samples": samples}})
        report = diagnostics([{"ranks": ranks, "microbatches": 1, "config": {"engram_layer_ids": [1]}}])
        self.assertEqual(report["tables"]["1"]["ready_fraction"], .5)
        self.assertEqual(report["tables"]["1"]["ready_before_decoder_dispatch_fraction"], 0)
        self.assertEqual(report["max_rank_future_wait_ms_per_step"]["mean"], 1.5)
        self.assertEqual(report["max_rank_route_host_ms_per_step"]["mean"], 10)


if __name__ == "__main__":
    unittest.main()
