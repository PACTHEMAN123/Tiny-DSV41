import hashlib
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from rollout_manager.host_memory import ByteBudget, GIB, MemoryAdmissionCancelled, available_bytes, evaluator_estimates, evaluator_memory
from rollout_manager.runner import Journal, make_pool, reservation


class EvaluatorMemoryTests(unittest.TestCase):
    def test_cgroup_remaining_memory_limits_proc_headroom(self):
        files = {"/proc/meminfo": f"MemAvailable: {480 * 1024**2} kB\n",
                 "/sys/fs/cgroup/memory.max": str(128*GIB),
                 "/sys/fs/cgroup/memory.current": str(32*GIB)}

        def read(path):
            if str(path) not in files:
                raise FileNotFoundError(path)
            return files[str(path)]
        with patch.object(Path, "read_text", autospec=True, side_effect=read):
            self.assertEqual(available_bytes(), 96*GIB)

    def test_model_and_input_estimate_uses_pinned_sizes(self):
        source = "model_name = 'gpt2'\nconfig = AutoConfig.from_pretrained(model_name)\n"
        sha = hashlib.sha256(source.encode()).hexdigest()
        cases = [{"task": {"task_id": "model", "code": source, "source_sha256": sha}}]
        registry = {"contracts": {sha: {"source_sha256": sha, "arena_bytes": 2*GIB}}, "tasks": []}
        models = [{"model": "gpt2", "files": [{"name": "model.safetensors", "bytes": GIB},
                                               {"name": "config.json", "bytes": 500}]}]
        self.assertEqual(evaluator_estimates(cases, registry, models)["model"]["bytes"], 15*GIB)
        with self.assertRaisesRegex(ValueError, "pinned model"):
            evaluator_estimates(cases, registry, [])
        cases[0]["task"]["code"] += "# changed"
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            evaluator_estimates(cases, registry, models)

    def test_waiting_for_memory_does_not_take_gpu_and_cancellation_unblocks(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = Journal(Path(directory))
            budget = ByteBudget(10)
            gpus = make_pool(["gpu"])
            budget.enqueue("owner", 8)
            budget.acquire("owner", lambda: False)
            queued, cancel = threading.Event(), threading.Event()
            errors = []

            def waiter():
                queued.set()
                try:
                    with evaluator_memory(budget, {"bytes": 7}, journal, "job", "evaluation", cancel.is_set):
                        with reservation(gpus, journal, "job", "evaluation"):
                            self.fail("cancelled evaluator acquired GPU")
                except MemoryAdmissionCancelled:
                    pass
                except BaseException as exc:
                    errors.append(exc)
            thread = threading.Thread(target=waiter)
            thread.start()
            self.assertTrue(queued.wait(1))
            self.assertEqual(gpus.qsize(), 1)
            cancel.set()
            thread.join(2)
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(gpus.qsize(), 1)
            self.assertEqual(budget.used, 8)
            self.assertEqual(budget.snapshot()["waiting_jobs"], 0)
            budget.release("owner")
            journal.close()

    def test_memory_released_after_cleanup_but_retained_for_quarantine(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = Journal(Path(directory))
            budget = ByteBudget(10)
            with evaluator_memory(budget, {"bytes": 7}, journal, "job", "evaluation", lambda: False):
                self.assertEqual(budget.used, 7)
            self.assertEqual(budget.used, 0)
            with evaluator_memory(budget, {"bytes": 7}, journal, "job", "final", lambda: False) as timing:
                timing["quarantined"] = True
            self.assertEqual(budget.snapshot()["state_bytes"]["quarantined"], 7)
            journal.close()

    def test_native_memory_plan_and_measured_legacy_estimate_agree(self):
        source = "def get_inputs(): return []"
        sha = hashlib.sha256(source.encode()).hexdigest()
        cases = [{"task": {"task_id": "task", "code": source, "source_sha256": sha}}]
        modern = {"plans": {sha: {"source_sha256": sha, "allocation_bytes": GIB}}, "tasks": []}
        legacy = {"contracts": {sha: {"source_sha256": sha, "arena_bytes": GIB}}, "tasks": []}
        self.assertEqual(evaluator_estimates(cases, modern, []), evaluator_estimates(cases, legacy, []))

    def test_fifo_head_cancellation_unblocks_without_oversubscription(self):
        budget = ByteBudget(10)
        budget.enqueue("owner", 6)
        budget.acquire("owner", lambda: False)
        budget.enqueue("large", 8)
        budget.enqueue("small", 4)
        ready = threading.Event()
        def waiter():
            budget.acquire("small", lambda: False)
            ready.set()
        thread = threading.Thread(target=waiter)
        thread.start()
        try:
            self.assertFalse(ready.wait(.05))
            budget.cancel("large")
            self.assertTrue(ready.wait(1))
            self.assertEqual(budget.used, 10)
        finally:
            budget.release("owner")
            budget.release("small")
            thread.join(1)
        self.assertFalse(thread.is_alive())

    def test_impossible_and_duplicate_reservations_fail(self):
        budget = ByteBudget(10)
        with self.assertRaises(ValueError):
            budget.enqueue("large", 11)
        budget.enqueue("job", 8)
        with self.assertRaises(ValueError):
            budget.enqueue("job", 1)
        with self.assertRaises(MemoryAdmissionCancelled):
            budget.acquire("job", lambda: True)
        self.assertEqual(budget.used, 0)
        self.assertEqual(budget.snapshot()["waiting_jobs"], 0)
