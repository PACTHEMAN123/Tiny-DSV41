from concurrent.futures import ThreadPoolExecutor
import threading
import tracemalloc
import unittest
from unittest.mock import Mock, patch

import torch

from dsv41_train.loading import LoadPolicy, WeightSource


class TensorReader:
    def __init__(self, tensor):
        self.tensor = tensor

    def metadata(self, name):
        return {"shape": self.tensor.shape}

    def tensor_rows(self, name, start, stop, device):
        return self.tensor[start:stop].clone().to(device)


class LoadingPipelineTest(unittest.TestCase):
    def test_chunk_metadata_is_bounded_before_the_first_read(self):
        target = torch.empty((100000, 4), device="meta")
        reader = TensorReader(target)
        policy = LoadPolicy("direct_shard", chunk_rows=1)
        with patch.object(reader, "tensor_rows", side_effect=RuntimeError("first read")):
            tracemalloc.start()
            try:
                with self.assertRaisesRegex(RuntimeError, "first read"):
                    WeightSource(("weight",)).fill(reader, target, (0, 0), target.shape, policy)
                _, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
        self.assertLess(peak, 1024**2, "chunk metadata grew with the entire row count")

    def test_single_and_empty_chunks_do_not_submit_worker_tasks(self):
        expected = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        reader = TensorReader(expected)
        worker = Mock()
        worker.submit.side_effect = AssertionError("single chunk dispatched to worker")
        policy = LoadPolicy("direct_shard", chunk_rows=8, pipeline=True)
        for start, stop in ((0, 6), (2, 5), (6, 6)):
            actual = torch.empty(stop-start, 4)
            WeightSource(("weight",)).fill(reader, actual, (start, 0), expected.shape, policy, executor=worker)
            torch.testing.assert_close(actual, expected[start:stop], rtol=0, atol=0)
        worker.submit.assert_not_called()
        worker.shutdown.assert_not_called()

    def test_borrowed_executor_survives_multiple_fills_and_read_failure(self):
        expected = torch.arange(28, dtype=torch.float32).reshape(7, 4)
        reader = TensorReader(expected)
        source = WeightSource(("weight",))
        policy = LoadPolicy("direct_shard", chunk_rows=2, pipeline=True)
        with ThreadPoolExecutor(max_workers=1) as worker:
            for start, stop in ((0, 7), (1, 6)):
                actual = torch.empty(stop-start, 4)
                source.fill(reader, actual, (start, 0), expected.shape, policy, executor=worker)
                torch.testing.assert_close(actual, expected[start:stop], rtol=0, atol=0)
            with patch.object(reader, "tensor_rows", side_effect=RuntimeError("read failure")):
                with self.assertRaisesRegex(RuntimeError, "read failure"):
                    source.fill(reader, torch.empty_like(expected), (0, 0), expected.shape, policy, executor=worker)
            self.assertEqual(worker.submit(lambda: 42).result(timeout=2), 42)

    def test_copy_failure_drains_prefetch_before_returning_reader_to_caller(self):
        expected = torch.ones(8, 4)
        reader = TensorReader(expected)
        second_started, release, second_finished = (threading.Event() for _ in range(3))
        original = reader.tensor_rows

        def read(name, start, stop, device):
            if start == 4:
                second_started.set()
                if not release.wait(timeout=2):
                    raise RuntimeError("test did not release the pending read")
                try:
                    return original(name, start, stop, device)
                finally:
                    second_finished.set()
            return original(name, start, stop, device)

        def fail_copy(*args, **kwargs):
            self.assertTrue(second_started.wait(timeout=2))
            release.set()
            raise RuntimeError("copy failure")

        with ThreadPoolExecutor(max_workers=1) as worker:
            try:
                with patch.object(reader, "tensor_rows", side_effect=read), patch.object(torch.Tensor, "copy_", side_effect=fail_copy):
                    with self.assertRaisesRegex(RuntimeError, "copy failure"):
                        WeightSource(("weight",)).fill(reader, torch.empty_like(expected), (0, 0), expected.shape,
                            LoadPolicy("direct_shard", chunk_rows=4, pipeline=True), executor=worker)
                self.assertTrue(second_finished.is_set())
                self.assertEqual(worker.submit(lambda: 42).result(timeout=2), 42)
            finally:
                release.set()


if __name__ == "__main__":
    unittest.main()
