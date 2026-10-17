"""CPU algorithm regressions; CUDA stream/NCCL behavior needs the GPU gate."""
from copy import deepcopy
import tempfile
from pathlib import Path
import unittest

import torch
from torch import nn

from dsv41_train.engram_offload import install_engram_sgd_hooks
from dsv41_train.models.dsv4.model import RowShardedEmbedding
from dsv41_train.sft_checkpoint import load_checkpoint, save_checkpoint


def table(*, trainable=True, budget=36):
    module = RowShardedEmbedding(7, 2, sparse_gradients=True, engram_offload=True).bfloat16()
    with torch.no_grad():
        module.weight.copy_(torch.arange(14).reshape(7, 2) / 16)
    module.requires_grad_(trainable)
    # Each BF16 row is four bytes; each CPU int64 tag is eight bytes.
    module._cpu_lookup.configure_cache(budget, "cpu:0")
    return module


def backward_rows(module, values):
    module(torch.tensor([0, 0])).backward(
        torch.tensor(values, dtype=module.weight.dtype).view(2, 1).expand(2, 2).contiguous())


class EngramCacheTest(unittest.TestCase):
    def test_cold_hot_mixed_duplicates_collisions_and_output_ownership(self):
        module = table(trainable=False)
        lookup = module._cpu_lookup
        ids = torch.tensor([[0, 1, 1]], dtype=torch.int32)
        first = module(ids)
        torch.testing.assert_close(first, module.weight[ids], rtol=0, atol=0)
        self.assertEqual(lookup.cache_metrics, dict(hits=0, misses=3, transferred_rows=2, resident_bytes=36))
        torch.testing.assert_close(module(ids), first, rtol=0, atol=0)
        self.assertEqual(lookup.cache_metrics["hits"], 3)
        # IDs 3 and 6 both evict slot 0. Its hit must be copied before eviction.
        mixed = torch.tensor([3, 0, 6, 1, 3])
        torch.testing.assert_close(module(mixed), module.weight[mixed], rtol=0, atol=0)
        torch.testing.assert_close(first, module.weight[ids], rtol=0, atol=0)
        self.assertEqual(lookup.cache_metrics, dict(hits=5, misses=6, transferred_rows=4, resident_bytes=36))
        torch.testing.assert_close(module(torch.tensor([6, 0])), module.weight[[6, 0]], rtol=0, atol=0)
        self.assertEqual(lookup.cache_metrics["hits"], 6)
        self.assertEqual(lookup.cache_metrics["transferred_rows"], 5)
        # Returning a metrics copy prevents callers from altering cache state.
        lookup.cache_metrics["hits"] = -1
        self.assertEqual(lookup.cache_metrics["hits"], 6)

    def test_empty_and_every_hit_duplicate_gets_a_sparse_occurrence(self):
        module = table()
        ids = torch.tensor([[1, 1, 2]])
        with torch.no_grad():
            module(ids)
        output = module(ids)
        values = torch.arange(6).reshape(1, 3, 2).bfloat16()
        output.backward(values)
        gradient = module.weight.grad
        self.assertTrue(gradient.is_sparse)
        self.assertFalse(gradient.is_coalesced())
        self.assertEqual(gradient._indices().tolist(), [[1, 1, 2]])
        torch.testing.assert_close(gradient._values(), values.reshape(3, 2), rtol=0, atol=0)
        self.assertEqual(module._cpu_lookup.cache_metrics["hits"], 3)
        module.zero_grad(set_to_none=True)
        empty = module(torch.empty((2, 0), dtype=torch.long))
        self.assertEqual(empty.shape, (2, 0, 2))
        empty.sum().backward()
        self.assertEqual(module.weight.grad._nnz(), 0)
        self.assertEqual(module.weight.grad.shape, module.weight.shape)

    def test_bf16_microbatch_update_parity_and_step_invalidation(self):
        cached, uncached = table(), table(budget=0)
        optimizers = [torch.optim.SGD(m.parameters(), lr=.125, foreach=False) for m in (cached, uncached)]
        for module, optimizer in zip((cached, uncached), optimizers):
            for handle in install_engram_sgd_hooks(module, optimizer):
                self.addCleanup(handle.remove)
        for step in range(3):
            for module, optimizer in zip((cached, uncached), optimizers):
                optimizer.zero_grad(set_to_none=bool(step % 2))
                backward_rows(module, (128., 128.))
                optimizer.zero_grad(set_to_none=bool(step % 2))
                backward_rows(module, (256., 1.))
                backward_rows(module, (1., -256.))
                self.assertEqual(module.weight.grad._nnz(), 4)
                self.assertEqual(module.weight.grad.float().coalesce().values().tolist(), [[2., 2.]])
                optimizer.step()
            torch.testing.assert_close(cached.weight, uncached.weight, rtol=0, atol=0)
            before = cached._cpu_lookup.cache_metrics
            torch.testing.assert_close(cached(torch.tensor([0])), cached.weight[[0]], rtol=0, atol=0)
            after = cached._cpu_lookup.cache_metrics
            self.assertEqual(after["misses"], before["misses"] + 1)
            self.assertGreater(after["hits"], 0)
            self.assertEqual(after["resident_bytes"], 36)

    def test_copy_restore_replacement_dtype_stride_and_explicit_clear(self):
        module = table(trainable=False)
        ids = torch.tensor([0, 1])
        def read():
            torch.testing.assert_close(module(ids), module.weight[ids], rtol=0, atol=0)
        read()
        with torch.no_grad():
            module.weight.add_(1)
        read()
        module.load_state_dict({"weight": torch.full_like(module.weight, 3)})
        read()
        module.weight = nn.Parameter(torch.full_like(module.weight, 4), requires_grad=False)
        read()
        module.float()
        read()
        self.assertLessEqual(module._cpu_lookup.cache_metrics["resident_bytes"], 36)
        # Storage/dtype/stride can change while the Parameter object stays fixed.
        module.weight.data = torch.arange(14).reshape(2, 7).float().t()
        read()
        module.weight.data.fill_(9)  # Deliberately bypasses the version counter.
        before = module._cpu_lookup.cache_metrics
        module._cpu_lookup.clear_cache()
        self.assertEqual(module._cpu_lookup.cache_metrics, before)
        read()

    def test_optimizer_partial_failure_cannot_reuse_stale_rows(self):
        module = table()
        other = nn.Parameter(torch.ones(1))
        optimizer = torch.optim.SGD([{"params": [module.weight]}, {"params": [other]}], lr=.125, foreach=False)
        for handle in install_engram_sgd_hooks(module, optimizer):
            self.addCleanup(handle.remove)
        backward_rows(module, (1., 1.))
        previous = module.weight.detach().clone()
        other.grad = torch.ones_like(other)
        optimizer.param_groups[1]["lr"] = "invalid after the first group updates"
        with self.assertRaises(TypeError):
            optimizer.step()
        self.assertFalse(torch.equal(previous, module.weight))
        # Post-step invalidation did not run, so this exercises tensor _version.
        before = module._cpu_lookup.cache_metrics
        torch.testing.assert_close(module(torch.tensor([0])), module.weight[[0]], rtol=0, atol=0)
        self.assertEqual(module._cpu_lookup.cache_metrics["misses"], before["misses"] + 1)

    def test_checkpoint_reload_invalidates_warm_cache_and_next_update_matches(self):
        source, restored = table(), table()
        optimizers = [torch.optim.SGD(m.parameters(), lr=.125, foreach=False) for m in (source, restored)]
        for module, optimizer in zip((source, restored), optimizers):
            install_engram_sgd_hooks(module, optimizer)
        def update(module, optimizer):
            optimizer.zero_grad(set_to_none=True)
            backward_rows(module, (256., 1.))
            backward_rows(module, (1., -256.))
            optimizer.step()
        restored(torch.tensor([0]))  # Hold an old row across loading step 1.
        update(source, optimizers[0])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "step1"
            identity = {"test": "cache CPU checkpoint"}
            save_checkpoint(path, source, optimizers[0], step=1, identity=identity, data_position=2)
            self.assertEqual(load_checkpoint(path, restored, optimizers[1], identity=identity), (1, 2))
            torch.testing.assert_close(restored(torch.tensor([0])), source.weight[[0]], rtol=0, atol=0)
            for module, optimizer in zip((source, restored), optimizers):
                update(module, optimizer)
            torch.testing.assert_close(source.weight, restored.weight, rtol=0, atol=0)

    def test_frozen_cache_persists_and_deepcopy_has_independent_state(self):
        module = table(trainable=False)
        other = nn.Parameter(torch.ones(1))
        optimizer = torch.optim.SGD([other], lr=.125)
        self.assertEqual(install_engram_sgd_hooks(module, optimizer), ())
        first = module(torch.tensor([0]))
        other.grad = torch.ones_like(other)
        optimizer.step()
        torch.testing.assert_close(module(torch.tensor([0])), first, rtol=0, atol=0)
        self.assertEqual(module._cpu_lookup.cache_metrics["hits"], 1)
        copied = deepcopy(module)
        self.assertEqual(copied._cpu_lookup.cache_metrics["resident_bytes"], 0)
        with torch.no_grad():
            copied.weight.add_(1)
        torch.testing.assert_close(copied(torch.tensor([0])), first + 1, rtol=0, atol=0)
        torch.testing.assert_close(module(torch.tensor([0])), first, rtol=0, atol=0)

    def test_inference_warmup_then_training_and_clear(self):
        module = table()
        with torch.inference_mode():
            module(torch.tensor([0]))
        module(torch.tensor([3])).sum().backward()
        self.assertEqual(module.weight.grad._indices().tolist(), [[3]])
        module._cpu_lookup.clear_cache()
        torch.testing.assert_close(module(torch.tensor([0])), module.weight[[0]], rtol=0, atol=0)

    def test_budget_configuration_and_index_validation(self):
        module = table(budget=10_000)
        module(torch.tensor([0]))
        lookup = module._cpu_lookup
        self.assertEqual(lookup.cache_metrics["resident_bytes"], 7 * 12)
        for budget in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                lookup.configure_cache(budget, "cpu")
        lookup.configure_cache(11, "cpu")
        with self.assertRaisesRegex(ValueError, "complete row"):
            module(torch.tensor([0]))
        lookup.configure_cache(12, "cpu")
        for ids in (torch.tensor([-1]), torch.tensor([7])):
            with self.assertRaises(IndexError):
                module(ids)
        with self.assertRaises(TypeError):
            module(torch.tensor([1.0]))
        lookup.configure_cache(0, "cpu")
        self.assertEqual(lookup.cache_metrics, dict(hits=0, misses=0, transferred_rows=0, resident_bytes=0))
        torch.testing.assert_close(module(torch.tensor([0])), module.weight[[0]], rtol=0, atol=0)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
