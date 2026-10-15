import asyncio
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from rollout_manager.contracts import (CancellationToken, EpisodeIdentity, EpisodePlan,
                                       EpisodeResult, ResourceRequirements, TaskSpec)
from rollout_manager.coordinator import Coordinator
from rollout_manager.events import EventStore


def plan(sample=0, group="g", benchmark="kernelbench", executor="cuda", cpu=1, accounting="episode"):
    task_id = f"{benchmark}/{group}/{sample}"
    payload = {"task_path": f"tasks/{sample}"} if benchmark == "terminal-bench" else {"native_case": {"task": task_id}}
    task = TaskSpec(benchmark, task_id, "a" * 64, payload,
                    ResourceRequirements(executor, cpu=cpu, accounting=accounting))
    return EpisodePlan(EpisodeIdentity("run", benchmark, task_id, group, sample, sampling_seed=42 + sample), task)


def result(plan, **changes):
    values = dict(identity=plan.identity, execution_state="completed", verification_state="verified",
                  score_eligible=True, cleanup_state="verified", native_outcome={"turns": 1}, native_artifacts={})
    values.update(changes)
    return EpisodeResult(**values)


class Backend:
    def __init__(self, execute=None):
        self.calls = []
        self.function = execute

    async def execute(self, item, cancellation):
        self.calls.append(item.identity)
        if self.function:
            return await self.function(item, cancellation)
        await asyncio.sleep(0)
        return result(item)


class CoordinatorTests(unittest.IsolatedAsyncioTestCase):
    async def test_eight_samples_are_independent_and_group_replenishment_waits(self):
        first_finished = set()

        async def execute(item, token):
            if item.identity.group_id == "second":
                self.assertEqual(len(first_finished), 8)
            await asyncio.sleep(.002)
            if item.identity.group_id == "first":
                first_finished.add(item.identity.sample_index)
            return result(item)

        backend = Backend(execute)
        coordinator = Coordinator({"kernelbench": backend}, max_workers=3, max_groups=1)
        items = [plan(i, group) for group in ("first", "second") for i in range(8)]
        observed = await coordinator.run(items)
        self.assertEqual(len(observed), 16)
        self.assertEqual(len(set(backend.calls)), 16)
        self.assertTrue(all(item.native_outcome["turns"] == 1 for item in observed))
        self.assertEqual(coordinator.summary["groups"], 2)

    async def test_backend_stage_resources_are_not_charged_twice(self):
        coordinator = Coordinator({"kernelbench": Backend()},
                                  capacities={"cuda": ResourceRequirements("cuda", cpu=1)})
        observed = await coordinator.run([plan(cpu=99, accounting="backend-stage")])
        self.assertEqual(observed[0].execution_state, "completed")
        self.assertEqual(coordinator.peak_reserved, {})

    async def test_backend_stage_executor_has_one_owner_and_shared_lifecycle_initializes_once(self):
        lifecycle = []

        class StageBackend(Backend):
            async def start(self):
                lifecycle.append("start")

            async def close(self):
                lifecycle.append("close")

        items = [plan(accounting="backend-stage"),
                 plan(benchmark="cuda-agent", accounting="backend-stage")]
        coordinator = Coordinator({"kernelbench": StageBackend(), "cuda-agent": StageBackend()})
        with self.assertRaisesRegex(ValueError, "one authoritative backend owner"):
            await coordinator.run(items)
        self.assertEqual(lifecycle, [])
        self.assertEqual(coordinator.reserved, {})
        shared = StageBackend()
        coordinator = Coordinator({"kernelbench": shared, "cuda-agent": shared})
        observed = await coordinator.run(items)
        self.assertTrue(all(item.execution_state == "completed" for item in observed))
        self.assertEqual(lifecycle, ["start", "close"])
        self.assertEqual(coordinator.peak_reserved, {})

    async def test_executor_capacity_and_fifo_are_preserved(self):
        order = []

        async def execute(item, token):
            order.append(item.identity.sample_index)
            await asyncio.sleep(.015)
            return result(item)

        coordinator = Coordinator({"kernelbench": Backend(execute)}, max_workers=3,
                                  capacities={"cuda": ResourceRequirements("cuda", cpu=2)})
        await coordinator.run([plan(0, cpu=1), plan(1, cpu=2), plan(2, cpu=1)])
        self.assertEqual(order, [0, 1, 2])
        self.assertEqual(coordinator.peak_reserved["cuda"]["cpu"], 2)
        self.assertEqual(coordinator.reserved["cuda"]["cpu"], 0)

    async def test_separate_cpu_executor_never_needs_cuda_lease(self):
        entered, both_entered = set(), asyncio.Event()

        async def execute(item, token):
            entered.add(item.task.resources.executor)
            if len(entered) == 2:
                both_entered.set()
            await asyncio.wait_for(both_entered.wait(), 1)
            return result(item)

        coordinator = Coordinator({"kernelbench": Backend(execute), "terminal-bench": Backend(execute)},
                                  max_workers=2, max_groups=2,
                                  capacities={"cuda": ResourceRequirements("cuda", cpu=1),
                                              "cpu-sandbox": ResourceRequirements("cpu-sandbox", cpu=1)})
        await coordinator.run([plan(), plan(benchmark="terminal-bench", executor="cpu-sandbox")])
        self.assertEqual(entered, {"cuda", "cpu-sandbox"})
        self.assertEqual(coordinator.peak_reserved["cpu-sandbox"]["gpu_count"], 0)

    async def test_cancellation_drains_real_thread_before_releasing_resources(self):
        entered, finished = threading.Event(), threading.Event()

        async def execute(item, token):
            def worker():
                entered.set()
                token.event.wait(1)
                time.sleep(.03)
                finished.set()
            await asyncio.to_thread(worker)
            return result(item, execution_state="cancelled", score_eligible=False)

        coordinator = Coordinator({"kernelbench": Backend(execute)}, drain_timeout=1)
        task = asyncio.create_task(coordinator.run([plan(), plan(1)]))
        while not entered.is_set():
            await asyncio.sleep(.001)
        task.cancel()
        observed = await task
        self.assertTrue(finished.is_set())
        self.assertTrue(coordinator.summary["drain_complete"])
        self.assertEqual([item.execution_state for item in observed], ["cancelled", "not_admitted"])
        self.assertEqual(coordinator.reserved["cuda"]["cpu"], 0)

    async def test_unproved_drain_quarantines_and_does_not_claim_thread_stopped(self):
        entered, release = threading.Event(), threading.Event()

        async def execute(item, token):
            def worker():
                entered.set()
                release.wait(2)
            await asyncio.to_thread(worker)
            return result(item)

        coordinator = Coordinator({"kernelbench": Backend(execute)}, drain_timeout=.02)
        task = asyncio.create_task(coordinator.run([plan(), plan(1)]))
        try:
            while not entered.is_set():
                await asyncio.sleep(.001)
            task.cancel()
            observed = await task
            self.assertFalse(coordinator.summary["drain_complete"])
            self.assertEqual(coordinator.quarantined["cuda"]["cpu"], 1)
            self.assertEqual(observed[0].cleanup_state, "unknown")
            self.assertEqual(observed[1].execution_state, "not_admitted")
            self.assertTrue(any(not worker.done() for worker in coordinator.draining_tasks))
        finally:
            release.set()
            await asyncio.gather(*coordinator.draining_tasks)

    async def test_uncertain_request_is_not_replayed_and_blocks_next_admission(self):
        async def execute(item, token):
            return result(item, request_state={"state": "published_outcome_unknown", "request_id": "unique"})

        backend = Backend(execute)
        coordinator = Coordinator({"kernelbench": backend})
        observed = await coordinator.run([plan(), plan(1)])
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(observed[1].execution_state, "not_admitted")
        self.assertEqual(coordinator.quarantined["cuda"]["cpu"], 1)
        self.assertEqual(coordinator.summary["admission_stopped_reason"], "uncertain_model_request")

    async def test_terminal_wave_holds_finished_sibling_resources_until_closure(self):
        entered, release = asyncio.Event(), asyncio.Event()
        calls = []
        coordinator = None

        class WaveBackend(Backend):
            async def admit_group(self, selected):
                calls.append(("admit", selected[0].identity.group_id))

            async def finish_group(self, selected, observed):
                self_test.assertEqual(coordinator.reserved["cpu"]["cpu"], len(selected))
                calls.append(("finish", selected[0].identity.group_id))
                return {"spool_terminal": True, "remote_cancellation_proven": False}

        async def execute(item, token):
            if item.identity.group_id == "first" and item.identity.sample_index == 1:
                entered.set()
                await release.wait()
            return result(item)

        self_test = self
        coordinator = Coordinator({"terminal-bench": WaveBackend(execute)}, max_workers=2,
                                  policy="terminal-waves", capacities={"cpu": ResourceRequirements("cpu", cpu=2)})
        items = [plan(i, "first", "terminal-bench", "cpu") for i in range(2)]
        items.append(plan(0, "second", "terminal-bench", "cpu"))
        task = asyncio.create_task(coordinator.run(items))
        await entered.wait()
        await asyncio.sleep(.01)
        self.assertEqual(coordinator.reserved["cpu"]["cpu"], 2)
        self.assertEqual(calls, [("admit", "first")])
        release.set()
        await task
        self.assertEqual(calls, [("admit", "first"), ("finish", "first"),
                                 ("admit", "second"), ("finish", "second")])

    async def test_terminal_review_and_null_reward_remain_distinct(self):
        async def execute(item, token):
            return result(item, verification_state="review_required", score_eligible=False,
                          native_outcome={"official_reward": 0, "reward": None})

        coordinator = Coordinator({"terminal-bench": Backend(execute)}, policy="terminal-waves")
        observed = await coordinator.run([plan(0, "first", "terminal-bench"),
                                          plan(0, "next", "terminal-bench")])
        self.assertIsNone(observed[0].native_outcome["reward"])
        self.assertEqual(observed[0].native_outcome["official_reward"], 0)
        self.assertEqual(observed[1].execution_state, "not_admitted")
        self.assertEqual(coordinator.summary["admission_stopped_reason"], "verifier_review_required")

    async def test_unsafe_wave_closure_quarantines_entire_wave(self):
        class UnclosedBackend(Backend):
            async def finish_group(self, selected, observed):
                return {"spool_terminal": False}

        coordinator = Coordinator({"terminal-bench": UnclosedBackend()}, max_workers=2,
                                  policy="terminal-waves")
        items = [plan(i, "first", "terminal-bench", "cpu") for i in range(2)]
        items.append(plan(0, "next", "terminal-bench", "cpu"))
        observed = await coordinator.run(items)
        self.assertEqual(coordinator.quarantined["cpu"]["cpu"], 2)
        self.assertEqual(observed[-1].execution_state, "not_admitted")

    async def test_partial_start_failure_closes_initialized_backends_and_writes_evidence(self):
        actions = []

        class LifecycleBackend(Backend):
            def __init__(self, name, fails=False):
                super().__init__()
                self.name, self.fails = name, fails

            async def start(self):
                actions.append("start-" + self.name)
                if self.fails:
                    raise OSError("baseline failed")

            async def close(self):
                actions.append("close-" + self.name)

        with tempfile.TemporaryDirectory() as temporary:
            store = EventStore(Path(temporary) / "run", "run")
            coordinator = Coordinator({"kernelbench": LifecycleBackend("first"),
                                       "terminal-bench": LifecycleBackend("partial", True)}, event_store=store)
            observed = await coordinator.run([plan(), plan(benchmark="terminal-bench")])
            store.close()
            self.assertEqual(actions, ["start-first", "start-partial", "close-partial", "close-first"])
            self.assertTrue(all(item.execution_state == "not_admitted" for item in observed))
            self.assertTrue(all(not item.score_eligible for item in observed))
            summary = json.loads((store.root / "summary.json").read_text())
            self.assertIn("baseline failed", summary["lifecycle_failures"][0])
            self.assertEqual(len(list((store.root / "results").glob("*.json"))), 2)

    async def test_close_failure_preserves_results_and_records_unproved_cleanup(self):
        class CloseFailure(Backend):
            async def close(self):
                raise OSError("cleanup receipt unavailable")

        coordinator = Coordinator({"kernelbench": CloseFailure()})
        observed = await coordinator.run([plan()])
        self.assertEqual(observed[0].execution_state, "completed")
        self.assertFalse(coordinator.summary["cleanup_complete"])
        self.assertIn("cleanup receipt unavailable", coordinator.summary["lifecycle_failures"][0])

    async def test_cancellation_during_wave_closure_retains_wave_resources(self):
        closing = asyncio.Event()

        class CancelledClosure(Backend):
            async def finish_group(self, selected, observed):
                closing.set()
                await asyncio.Future()

        coordinator = Coordinator({"terminal-bench": CancelledClosure()}, policy="terminal-waves")
        task = asyncio.create_task(coordinator.run([plan(benchmark="terminal-bench", executor="cpu"),
                                                  plan(group="later", benchmark="terminal-bench", executor="cpu")]))
        await closing.wait()
        task.cancel()
        observed = await task
        self.assertEqual(coordinator.quarantined["cpu"]["cpu"], 1)
        self.assertEqual(observed[-1].execution_state, "not_admitted")

    async def test_cancelled_before_start_does_not_initialize_backend(self):
        class UnstartedBackend(Backend):
            async def start(self):
                raise AssertionError("must not initialize after cancellation")

        token = CancellationToken()
        token.cancel("already_cancelled")
        coordinator = Coordinator({"kernelbench": UnstartedBackend()})
        observed = await coordinator.run([plan()], token)
        self.assertEqual(observed[0].execution_state, "not_admitted")
        self.assertEqual(coordinator.summary["admission_stopped_reason"], "already_cancelled")


class EventContractTests(unittest.TestCase):
    def test_new_records_are_immutable_and_missing_nested_measurements_stay_missing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "new-run"
            store = EventStore(root, "run")
            item = result(plan(), usage=None,
                          timings={"parent_seconds": 4, "native_samples_ms": [2, 3], "missing": None},
                          native_outcome={"development_correct": True, "final_correct": None,
                                          "verified_correct": False, "reward": None})
            path = store.write_result(item)
            with self.assertRaises(FileExistsError):
                store.write_result(item)
            store.append("native_evidence", item.identity, turn_id=2, request_id="request",
                         tool_call_id="tool", clock_domain="cuda_event", duration_seconds=None)
            store.close()
            saved = json.loads(path.read_text())
            self.assertIsNone(saved["timings"]["missing"])
            self.assertEqual(saved["timings"]["parent_seconds"], 4)
            self.assertIsNone(saved["native_outcome"]["final_correct"])
            events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
            self.assertEqual(events[-1]["clock_domain"], "coordinator_monotonic")
            self.assertEqual(events[-1]["details"]["clock_domain"], "cuda_event")
            self.assertIsNone(events[-1]["details"]["duration_seconds"])
            self.assertEqual((events[-1]["turn_id"], events[-1]["request_id"], events[-1]["tool_call_id"]),
                             (2, "request", "tool"))
            with self.assertRaises(FileExistsError):
                EventStore(root, "another-run")

    def test_terminal_payload_does_not_require_reference_source(self):
        item = plan(benchmark="terminal-bench", executor="cpu")
        self.assertNotIn("code", item.task.payload)
        with self.assertRaises(ValueError):
            TaskSpec("terminal-bench", "task", "a" * 64, {"task_path": "task", "code": "cuda"},
                     ResourceRequirements("cpu"))


if __name__ == "__main__":
    unittest.main()
