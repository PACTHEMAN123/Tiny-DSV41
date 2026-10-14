"""Real two-rank CPU collectives for the benchmark's complete dense proof."""

import datetime
import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from dsv41_train.runtime import Runtime
from qwen_engram_sft_benchmark import (
    DenseUpdateValidationError, _capture_dense_updates, _verify_dense_updates,
)


def _worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=2, init_method="file://" + rendezvous,
                            timeout=datetime.timedelta(seconds=30))
    runtime = Runtime(torch.device("cpu"), rank, rank, 2)
    try:
        # An empty local shard is valid when a peer owns and updates the tensor.
        parameter = nn.Parameter(torch.ones(0 if rank == 0 else 9, dtype=torch.bfloat16))
        selected = {"lm_head.weight": parameter}
        snapshots = _capture_dense_updates(selected, runtime, chunk_elements=4)
        if rank == 1:
            with torch.no_grad():
                parameter[-1] -= .125
        proof = _verify_dense_updates(selected, snapshots, runtime, chunk_elements=4)
        assert proof["global_elements_checked"] == 9
        assert proof["global_changed_elements"] == 1
        assert proof["global_max_abs_delta"] == .125
        assert proof["snapshot_allocated_bytes_global"] == 18
        assert proof["local_changed_elements"] == rank
        # A peer's nonfinite value must make both ranks fail at the same check.
        snapshots = _capture_dense_updates(selected, runtime, chunk_elements=4)
        if rank == 1:
            with torch.no_grad():
                parameter[1] = float("inf")
        try:
            _verify_dense_updates(selected, snapshots, runtime, chunk_elements=4)
        except RuntimeError as error:
            assert "non-finite weight/update" in str(error)
        else:
            raise AssertionError("peer nonfinite value did not fail on every rank")
        try:
            _capture_dense_updates(selected, runtime, chunk_elements=4)
        except RuntimeError as error:
            assert "before optimizer" in str(error)
        else:
            raise AssertionError("peer nonfinite snapshot did not fail on every rank")
        if rank == 1:
            with torch.no_grad():
                parameter[1] = 1.
        snapshots = _capture_dense_updates(selected, runtime, chunk_elements=4)
        try:
            _verify_dense_updates(selected, snapshots, runtime, chunk_elements=4)
        except DenseUpdateValidationError as error:
            assert error.diagnostics["global_changed_elements"] == 0
            assert error.diagnostics["global_elements_checked"] == 9
        else:
            raise AssertionError("global no-op update must fail on every rank")
    finally:
        dist.destroy_process_group()


class DistributedDenseProofTest(unittest.TestCase):
    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "Gloo required")
    def test_empty_shard_global_proof_and_peer_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_worker, args=(str(Path(directory) / "gloo"),), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
