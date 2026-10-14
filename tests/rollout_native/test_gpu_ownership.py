import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from rollout_manager.gpu_ownership import GpuOwnershipError, ReviewedGpuOwners


class GpuOwnershipTests(unittest.TestCase):
    def test_a_slow_query_does_not_block_another_gpu(self):
        owners = ReviewedGpuOwners()
        entered, release = threading.Event(), threading.Event()

        def query(gpu):
            if gpu == "gpu-a":
                entered.set()
                self.assertTrue(release.wait(timeout=2))
            return []

        with owners.lease("gpu-a", Path("a")) as guard_a, owners.lease("gpu-b", Path("b")) as guard_b:
            with patch("rollout_manager.gpu_ownership.gpu_processes", side_effect=query):
                with ThreadPoolExecutor(max_workers=2) as pool:
                    first = pool.submit(guard_a)
                    try:
                        self.assertTrue(entered.wait(timeout=1))
                        self.assertTrue(pool.submit(guard_b).result(timeout=1))
                    finally:
                        release.set()
                    self.assertTrue(first.result(timeout=1))
        self.assertEqual(len(owners.checks), 2)
        self.assertTrue(all(row["accepted"] for row in owners.checks))

    def test_only_registered_peer_harness_contexts_are_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            own, peer = root / "own", root / "peer"
            own.mkdir()
            peer.mkdir()
            proc = root / "123"
            proc.mkdir()
            (proc / "cwd").symlink_to(peer)
            (proc / "stat").write_text("123 (python) " + " ".join(["0"] * 19 + ["42"]))
            (proc / "cmdline").write_bytes(b"python\0-m\0rollout_manager.adapters.harness_child\0verify\0")
            (proc / "environ").write_bytes(b"CUDA_VISIBLE_DEVICES=gpu\0")
            owners = ReviewedGpuOwners()
            owners.proc = root
            with patch("rollout_manager.gpu_ownership.gpu_processes", return_value=["gpu, 123, 100"]):
                with owners.lease("gpu", own) as guard:
                    with self.assertRaises(GpuOwnershipError):
                        guard()
                    with owners.lease("gpu", peer):
                        self.assertTrue(guard())
                        (proc / "cmdline").write_bytes(b"unrelated-program\0")
                        with self.assertRaises(GpuOwnershipError):
                            guard()
                    with self.assertRaises(GpuOwnershipError):
                        guard()

    def test_uninspectable_reported_context_is_not_cleanup_proof(self):
        with tempfile.TemporaryDirectory() as directory:
            owners = ReviewedGpuOwners(settle_seconds=0)
            owners.proc = Path(directory)
            with owners.lease("gpu", Path(directory)) as guard:
                with patch("rollout_manager.gpu_ownership.gpu_processes", return_value=["gpu, 123, 100"]):
                    with self.assertRaises(GpuOwnershipError):
                        guard()
                with patch("rollout_manager.gpu_ownership.gpu_processes", side_effect=[["gpu, 123, 100"], []]):
                    self.assertTrue(guard())

    def test_delayed_nvml_exit_requires_observed_disappearance(self):
        with tempfile.TemporaryDirectory() as directory:
            owners = ReviewedGpuOwners(settle_seconds=.2)
            owners.proc = Path(directory)
            with owners.lease("gpu", Path(directory)) as guard:
                with patch("rollout_manager.gpu_ownership.gpu_processes",
                           side_effect=[["gpu, 123, 100"], ["gpu, 123, 100"], []]):
                    self.assertTrue(guard())
            self.assertTrue(owners.observations[-1]["context_disappeared"])
