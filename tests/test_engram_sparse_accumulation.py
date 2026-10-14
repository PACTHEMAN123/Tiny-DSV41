"""Preserve low-precision COO occurrences before CPU AccumulateGrad can merge."""

import json
from pathlib import Path
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from dsv41_train.models.dsv4.engram import EngramPolicy, RowShardedEmbedding
from dsv41_train.models.dsv4.engram.storage import CachedRowStore
from dsv41_train.models.dsv4.parallel import build_parallelism
from dsv41_train.models.dsv4.policies import GradientPolicy
from dsv41_train.optimization import build_optimizer, OptimizerPolicy


def make_table():
    table = RowShardedEmbedding(7, 2, sparse_gradients=True,
                                policy=EngramPolicy("cpu")).to(dtype=torch.bfloat16)
    with torch.no_grad():
        table.weight.zero_()
    return table


def canceling_microbatches(table):
    ids = torch.tensor([0, 0])
    for values in ((256., 1.), (1., -256.)):
        gradient = torch.tensor(values, dtype=torch.bfloat16).view(-1, 1).expand(-1, 2).contiguous()
        table(ids).backward(gradient)


def distributed_worker(rank, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank, world_size=2)
    try:
        meshes, _, _ = build_parallelism(1, 2, device_type="cpu")
        model = nn.Module()
        model.engram_tables = nn.ModuleDict({"1": RowShardedEmbedding(
            7, 2, meshes.engram, sparse_gradients=True, policy=EngramPolicy("cpu")).to(dtype=torch.bfloat16)})
        table = model.engram_tables["1"]
        with torch.no_grad():
            table.weight.zero_()
        policy = GradientPolicy(model, meshes)
        optimizer = build_optimizer(model, OptimizerPolicy("sgd", .125, True))
        for set_to_none in (True, False):
            optimizer.zero_grad(set_to_none=set_to_none)
            policy.begin_step()
            # Each owner receives two ranks' [256,1] then [1,-256] after
            # RowRouter's 1/world scaling. Correct local sum is 4, not 2.
            ids = torch.tensor([0, 0, 4, 4])
            for values in ((512., 2., 512., 2.), (2., -512., 2., -512.)):
                table(ids).backward(torch.tensor(values, dtype=torch.bfloat16).view(-1, 1).expand(-1, 2).contiguous())
            before = table.weight.grad.float().coalesce().to_dense()
            policy.normalize()
            table.synchronize()
            torch.testing.assert_close(table.weight.grad.float().coalesce().to_dense(), before, rtol=0, atol=0)
            expected = torch.zeros_like(before)
            expected[(0 if rank == 0 else 4) - table.row_start] = 4
            torch.testing.assert_close(before, expected, rtol=0, atol=0)
            optimizer.step()
        expected = torch.zeros_like(table.weight)
        expected[(0 if rank == 0 else 4) - table.row_start] = -1
        torch.testing.assert_close(table.weight, expected, rtol=0, atol=0)
        Path(output, f"rank{rank}.json").write_text(json.dumps({
            "rank": rank, "ep_size": meshes.ep.size(), "table_gradient_skipped_by_normalization": True,
            "local_updated_row": (0 if rank == 0 else 4) - table.row_start,
            "updated_values": [-1., -1.], "zero_grad_modes": [True, False]}))
        table.close()
    finally:
        dist.destroy_process_group()


class SparseAccumulationTest(unittest.TestCase):
    def test_real_two_backward_calls_preserve_all_occurrences_before_fp32_coalesce(self):
        table = make_table()
        self.addCleanup(table.close)
        canceling_microbatches(table)
        self.assertEqual(table.weight.grad.dtype, torch.bfloat16)
        self.assertEqual(table.weight.grad._nnz(), 4)
        self.assertEqual(table.weight.grad._values()[:, 0].tolist(), [256., 1., 1., -256.])
        self.assertEqual(table.weight.grad.float().coalesce().values().tolist(), [[2., 2.]])
        # Coalescing only after the CPU's native BF16 addition is too late.
        native = nn.Parameter(torch.zeros(7, 2, dtype=torch.bfloat16))
        for values in ((256., 1.), (1., -256.)):
            nn.functional.embedding(torch.tensor([0, 0]), native, sparse=True).backward(
                torch.tensor(values, dtype=torch.bfloat16).view(-1, 1).expand(-1, 2).contiguous())
        self.assertEqual(native.grad.float().coalesce().values().tolist(), [[1., 1.]])

    def test_optimizer_and_both_zero_grad_modes_do_not_reuse_or_renormalize_history(self):
        for foreach in (False, True):
            for set_to_none in (False, True):
                with self.subTest(foreach=foreach, set_to_none=set_to_none):
                    table = make_table()
                    optimizer = build_optimizer(table, OptimizerPolicy("sgd", .125, foreach))
                    try:
                        self.assertFalse(optimizer.param_groups[0]["foreach"])
                        for step in range(3):
                            optimizer.zero_grad(set_to_none=set_to_none)
                            canceling_microbatches(table)
                            optimizer.step()
                            self.assertEqual(table.weight[0].tolist(), [-.25 * (step + 1)] * 2)
                            self.assertEqual(table.weight[1:].count_nonzero().item(), 0)
                    finally:
                        table.close()

    def test_cached_backward_and_empty_microbatch_keep_same_raw_gradient_semantics(self):
        table = make_table()
        self.addCleanup(table.close)
        # Exercise the real cached autograd function on CPU, without CUDA staging.
        table.store = CachedRowStore(16, table.metrics)
        canceling_microbatches(table)
        table(torch.empty(0, dtype=torch.long)).sum().backward()
        self.assertEqual(table.weight.grad.float().coalesce().values().tolist(), [[2., 2.]])
        build_optimizer(table, OptimizerPolicy("sgd", .125)).step()
        self.assertEqual(table.weight[0].tolist(), [-.25, -.25])
        torch.testing.assert_close(table(torch.tensor([0])), table.weight[[0]], rtol=0, atol=0)

    def test_parameter_replacement_and_dtype_change_remove_old_hook(self):
        table = make_table()
        self.addCleanup(table.close)
        canceling_microbatches(table)
        old = table.weight
        self.assertEqual(len(old._backward_hooks), 1)
        table.weight = nn.Parameter(torch.zeros_like(old))
        canceling_microbatches(table)
        self.assertFalse(old._backward_hooks)
        self.assertEqual(table.weight.grad.float().coalesce().values().tolist(), [[2., 2.]])
        table.zero_grad(set_to_none=True)
        table.to(dtype=torch.float32)
        table(torch.tensor([0])).sum().backward()
        self.assertIsNone(table._sparse_accumulation_handle)
        self.assertFalse(table.weight._backward_hooks)

    def test_existing_scaling_hook_sees_only_new_microbatch_gradient(self):
        table = make_table()
        self.addCleanup(table.close)
        seen = []
        def scale(gradient):
            seen.append(gradient._nnz())
            return gradient * .5
        handle = table.weight.register_hook(scale)
        self.addCleanup(handle.remove)
        canceling_microbatches(table)
        self.assertEqual(seen, [2, 2])
        self.assertEqual(table.weight.grad.float().coalesce().values().tolist(), [[1., 1.]])

    def test_later_scaling_hook_is_rejected_instead_of_scaling_old_gradient_twice(self):
        table = make_table()
        self.addCleanup(table.close)
        table(torch.tensor([0])).sum().backward()
        output = table(torch.tensor([0]))
        handle = table.weight.register_hook(lambda gradient: gradient * .5)
        self.addCleanup(handle.remove)
        with self.assertRaisesRegex(RuntimeError, "gradient-transform hooks"):
            output.sum().backward()

    def test_multiple_lookup_branches_in_one_backward_fail_closed(self):
        table = make_table()
        self.addCleanup(table.close)
        a, b = table(torch.tensor([0, 0])), table(torch.tensor([0, 0]))
        ga = torch.tensor([[256., 256.], [1., 1.]], dtype=torch.bfloat16)
        gb = torch.tensor([[1., 1.], [-256., -256.]], dtype=torch.bfloat16)
        with self.assertRaisesRegex(RuntimeError, "one lookup per table per backward"):
            torch.autograd.backward((a, b), (ga, gb))
        self.assertIsNone(table.weight.grad)

    def test_higher_order_backward_is_explicitly_rejected(self):
        table = make_table()
        self.addCleanup(table.close)
        with self.assertRaisesRegex(RuntimeError, "first-order backward only"):
            table(torch.tensor([0])).square().sum().backward(create_graph=True)
        self.assertIsNone(table.weight.grad)

    def test_two_rank_ep_router_and_gradient_policy_normalize_once(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(distributed_worker, args=(str(Path(directory, "store")), directory), nprocs=2, join=True)
            records = [json.loads(Path(directory, f"rank{rank}.json").read_text()) for rank in (0, 1)]
            self.assertTrue(all(r["ep_size"] == 2 and r["table_gradient_skipped_by_normalization"] for r in records))


if __name__ == "__main__":
    unittest.main()
