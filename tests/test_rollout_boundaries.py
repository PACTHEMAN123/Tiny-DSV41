"""Offline integration checks for installed adapters and transport ownership."""
import asyncio
import contextlib
import copy
from dataclasses import replace
import hashlib
import json
import io
import tempfile
import threading
import time
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from rollout_manager.adapters.datasets import DATASETS
from rollout_manager.backends.native import NativeBackend
from rollout_manager.contracts import CancellationToken, EpisodeResult, ResourceRequirements
from rollout_manager.coordinator import Coordinator
from rollout_manager.datasets import CudaAgentAdapter, KernelBenchAdapter, TerminalBenchAdapter
from rollout_manager.events import EventStore
from rollout_manager.transports import RequestLedger, RpcTransport


def native_case(benchmark="kernelbench"):
    source = "class Model: pass\n"
    return {"task": {"task_id": benchmark + "/level_2/0", "dataset": benchmark,
                     "revision": DATASETS[benchmark]["revision"], "code": source,
                     "source_sha256": hashlib.sha256(source.encode()).hexdigest()},
            "request": {"model": "fixture-model", "messages": [{"role": "user", "content": "public"}]},
            "files": {}}


class DatasetBoundaryTests(unittest.TestCase):
    def test_native_pinned_identity_and_seeds_ignore_completion_order(self):
        adapter = KernelBenchAdapter([native_case()])
        plans = adapter.plan("run", rollouts=8, passes=1, seed=42)
        self.assertEqual(len(plans), 8)
        other = CudaAgentAdapter([native_case("cuda-agent")]).plan("run", rollouts=8)
        self.assertFalse({p.identity.trial_id for p in plans} & {p.identity.trial_id for p in other})
        for item in plans:
            identity = f"42:0:{item.identity.task_id}:{item.identity.sample_index}"
            expected = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:4], "big")
            self.assertEqual(item.identity.sampling_seed, expected)
            self.assertEqual(item.payload["request"]["seed"], expected)
        mutated = copy.deepcopy(native_case())
        mutated["task"]["code"] += "# changed"
        with self.assertRaisesRegex(ValueError, "digest"):
            KernelBenchAdapter([mutated]).enumerate()
        mutated = copy.deepcopy(native_case())
        mutated["task"]["revision"] = "unreviewed"
        with self.assertRaisesRegex(ValueError, "pinned"):
            KernelBenchAdapter([mutated]).enumerate()

    def test_terminal_catalog_contains_only_external_verifier_references(self):
        adapter = TerminalBenchAdapter(executor="cpu-only")
        tasks = adapter.enumerate()
        self.assertEqual(len(tasks), 89)
        self.assertEqual(len({task.task_id for task in tasks}), 89)
        for task in tasks:
            self.assertEqual(task.resources.gpu_count, 0)
            self.assertNotIn("code", task.payload)
            self.assertEqual(task.budgets["attempts"], 1)
            self.assertEqual(task.budgets["retries"], 0)
            for reference in task.provenance["source_files"].values():
                self.assertEqual(set(reference), {"bytes", "sha256"})
            self.assertIsNone(task.payload["task_path"])
        with self.assertRaises(ValueError):
            TerminalBenchAdapter(executor="cpu-only", task_ids=["unregistered"])

    def test_cuda_agent_budget_does_not_claim_kernelbench_trial_protocol(self):
        task = CudaAgentAdapter([native_case("cuda-agent")]).enumerate()[0]
        self.assertNotEqual(task.budgets.get("correctness_trials"), 5,
                            "CUDA-Agent uses original verifier, not KernelBench's num_correct_trials argument")


class TransportBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_durable_claim_prevents_replay_after_client_or_process_restart(self):
        calls = []

        class BrokenClient:
            def __init__(self, *args, **kwargs): pass
            def call(self, path, payload):
                calls.append(path)
                error = OSError("response was lost")
                error.rpc_failure = {"dispatch_may_have_occurred": True, "phase": "response_body"}
                raise error
            def close(self): pass

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            transport = RpcTransport("https://example.test", root / "credentials", root / "ledger", client_type=BrokenClient)
            request = {"request_id": "episode-t1", "payload": {"model": "fixture"}, "timeout": 1}
            with self.assertRaises(OSError):
                await transport.submit(request)
            saved = json.loads((root / "ledger/episode-t1.json").read_text())
            self.assertEqual(saved["state"], "outcome_unknown")
            restarted = RpcTransport("https://example.test", root / "credentials", root / "ledger", client_type=BrokenClient)
            with self.assertRaises(FileExistsError) as caught:
                await restarted.submit(request)
            self.assertEqual(calls, ["/generate/episode-t1"])
            self.assertFalse(saved["automatic_replay"])
            self.assertFalse(caught.exception.rpc_failure["dispatch_may_have_occurred"])

    async def test_cancelled_rpc_waiter_retains_uncertainty_until_actual_response(self):
        entered, release = threading.Event(), threading.Event()

        class DelayedClient:
            def __init__(self, *args, **kwargs): pass
            def call(self, path, payload):
                entered.set()
                release.wait(2)
                return {"response": {"usage": None}}
            def close(self): pass

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            transport = RpcTransport("https://example.test", root / "credentials", root / "ledger", client_type=DelayedClient)
            request = {"request_id": "unique", "payload": {"model": "fixture"}, "timeout": 1}
            task = asyncio.create_task(transport.submit(request))
            try:
                deadline = asyncio.get_running_loop().time() + 1
                while not entered.is_set() and not task.done() and asyncio.get_running_loop().time() < deadline:
                    await asyncio.sleep(.001)
                if task.done():
                    await task  # Surface failed dispatch rather than wait for a nonexistent worker.
                self.assertTrue(entered.is_set(), "fixture RPC worker was never dispatched")
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                state_path = root / "ledger/unique.json"
                unknown = json.loads(state_path.read_text())
                self.assertTrue(unknown["uncertain"])
                self.assertNotEqual(unknown.get("remote_cancellation_proven"), True)
                with self.assertRaises(FileExistsError):
                    RequestLedger(root / "ledger").claim("unique", "route", {}, 1)
                release.set()
                for _ in range(100):
                    saved = json.loads(state_path.read_text())
                    if saved["state"] == "response_received":
                        break
                    await asyncio.sleep(.005)
                self.assertEqual(saved["state"], "response_received")
                self.assertFalse(saved["uncertain"])
            finally:
                release.set()


class NativeBridgeTests(unittest.IsolatedAsyncioTestCase):
    def backend(self, root, runner):
        args = Namespace(schedule="full-suite", concurrency=[1], turns=1, root=root / "native")
        transport = RpcTransport("https://example.test", root / "credentials", root / "ledger")
        return NativeBackend({"args": args}, transport, runner_function=runner)

    async def test_stale_native_identity_or_source_is_rejected_before_thread_dispatch(self):
        plan = KernelBenchAdapter([native_case()]).plan("run", rollouts=1)[0]
        with tempfile.TemporaryDirectory() as temporary:
            backend = self.backend(Path(temporary), lambda *args, **kwargs: None)
            backend.native_cancelled = lambda: False
            backend.cases[(plan.identity.task_id, 0, 0)] = plan.payload
            bad_plans = [replace(plan, identity=replace(plan.identity, group_id="stale-group")),
                         replace(plan, identity=replace(plan.identity, sampling_seed=0)),
                         replace(plan, task=replace(plan.task, source_sha256="wrong-source"))]
            for bad in bad_plans:
                with self.assertRaisesRegex(ValueError, "identity/source"):
                    await backend.execute(bad, CancellationToken())
            self.assertIsNone(backend.pool)

    async def test_native_lifecycle_initializes_once_and_shared_groups_run_sixteen_samples(self):
        cases = [native_case(), native_case()]
        cases[1]["task"]["task_id"] = "kernelbench/level_2/1"
        plans = KernelBenchAdapter(cases).plan("run", rollouts=8)
        from rollout_manager.workloads import suite_groups
        lock = threading.Lock()
        calls, active_groups, completed_groups = [], {}, {}
        lifecycle_count, peak_workers = 0, 0

        def runner(args, *, episode_driver, **kwargs):
            nonlocal lifecycle_count, peak_workers
            lifecycle_count += 1

            def execute(item, job, profile=False):
                nonlocal peak_workers
                group = item["rollout"]["group_id"]
                with lock:
                    # At group replenishment every prior group's eight actual
                    # workers have finished, including their return records.
                    if group not in completed_groups and completed_groups:
                        self.assertTrue(all(value == 8 for value in completed_groups.values()))
                    calls.append((job, group, item["rollout"]["sample_index"], profile))
                    active_groups[group] = active_groups.get(group, 0) + 1
                    self.assertEqual(len(active_groups), 1)
                    peak_workers = max(peak_workers, sum(active_groups.values()))
                time.sleep(.003)
                with lock:
                    completed_groups[group] = completed_groups.get(group, 0) + 1
                    active_groups[group] -= 1
                    if not active_groups[group]:
                        del active_groups[group]
                return {"completed": True, "cleanup_state": "verified", "score_eligible": True,
                        "final_correct": True, "verified_correct": True, "stages": {},
                        "rollout": item["rollout"]}

            return episode_driver(execute, suite_groups(cases, 8, 1, 42), 3,
                                  lambda: False, time.monotonic() + 30)

        with tempfile.TemporaryDirectory() as temporary:
            backend = self.backend(Path(temporary), runner)
            backend.args.concurrency = [3]
            coordinator = Coordinator({"kernelbench": backend}, max_workers=3, max_groups=1)
            observed = await coordinator.run(plans)
            self.assertEqual(lifecycle_count, 1)
            self.assertEqual(len(calls), 16)
            self.assertEqual(len({call[0] for call in calls}), 16)
            self.assertEqual(sum(call[3] for call in calls), 1)
            self.assertEqual(len(backend.records), 16)
            self.assertTrue(all(item.execution_state == "completed" for item in observed))
            self.assertLessEqual(peak_workers, 3)
            self.assertEqual(completed_groups, {group: 8 for group in {call[1] for call in calls}})
            self.assertFalse(backend.thread.is_alive())

    async def test_native_backend_ready_fails_if_lifecycle_ends_before_driver(self):
        with tempfile.TemporaryDirectory() as temporary:
            backend = self.backend(Path(temporary), lambda *args, **kwargs: None)
            try:
                with self.assertRaises(RuntimeError):
                    await asyncio.wait_for(backend.start(), .15)
            finally:
                await backend.close()

    async def test_cancelled_native_waiter_drains_actual_pool_worker_and_lifecycle(self):
        entered, release, finished = threading.Event(), threading.Event(), threading.Event()
        case = native_case()
        selected = KernelBenchAdapter([case]).plan("run", rollouts=1)[0]
        from rollout_manager.workloads import suite_groups

        def runner(args, *, episode_driver, cancel_event, **kwargs):
            def execute(item, job, profile=False):
                entered.set()
                cancel_event.wait(2)
                release.wait(2)
                finished.set()
                return {"completed": False, "cleanup_state": "verified", "score_eligible": False,
                        "failure_class": "cancelled", "stages": {}, "rollout": item["rollout"]}
            episode_driver(execute, suite_groups([case], 1, 1, 42), 1, cancel_event.is_set,
                           time.monotonic() + 30)

        with tempfile.TemporaryDirectory() as temporary:
            backend = self.backend(Path(temporary), runner)
            await backend.start()
            token = CancellationToken()
            task = asyncio.create_task(backend.execute(selected, token))
            closing = None
            try:
                while not entered.is_set() and not task.done():
                    await asyncio.sleep(.001)
                self.assertTrue(entered.is_set())
                task.cancel()
                await asyncio.sleep(.01)
                self.assertTrue(token.cancelled())
                self.assertTrue(backend.cancel_event.is_set())
                self.assertFalse(task.done())
                task.cancel()  # Repeated caller cancellation must preserve the actual future.
                closing = asyncio.create_task(backend.close())
                await asyncio.sleep(.01)
                self.assertFalse(closing.done())
                self.assertFalse(finished.is_set())
                release.set()
                observed = await asyncio.wait_for(task, 1)
                await asyncio.wait_for(closing, 1)
                self.assertTrue(finished.is_set())
                self.assertEqual(observed.cleanup_state, "verified")
                self.assertEqual(len(backend.records), 1)
                self.assertFalse(backend.thread.is_alive())
            finally:
                release.set()
                await asyncio.gather(task, return_exceptions=True)
                await backend.close()
                if closing is not None:
                    await asyncio.gather(closing, return_exceptions=True)

    async def test_native_ttl_refusal_closes_shared_admission_without_launch(self):
        executed = []
        case = native_case()
        plans = KernelBenchAdapter([case]).plan("run", rollouts=2)
        from rollout_manager.workloads import suite_groups

        def runner(args, *, episode_driver, **kwargs):
            def execute(item, job, profile=False):
                executed.append(job)
                raise AssertionError("TTL must prevent dispatch")
            episode_driver(execute, suite_groups([case], 2, 1, 42), 1, lambda: True, time.monotonic() - 1)

        with tempfile.TemporaryDirectory() as temporary:
            backend = self.backend(Path(temporary), runner)
            coordinator = Coordinator({"kernelbench": backend})
            observed = await coordinator.run(plans)
            self.assertEqual(executed, [])
            self.assertEqual([item.execution_state for item in observed], ["not_admitted", "not_admitted"])
            self.assertIsNotNone(coordinator.summary["admission_stopped_reason"])

    async def test_native_events_keep_episode_and_request_namespaces_separate(self):
        case = native_case()
        plans = KernelBenchAdapter([case]).plan("run", rollouts=1)
        from rollout_manager.workloads import suite_groups

        def runner(args, *, episode_driver, event_sink, **kwargs):
            def execute(item, job, profile=False):
                event_sink({"job": job, "stage": "inference", "event": "finish", "turn_id": 2,
                            "request_id": job + "-t2", "tool_call_id": "shell-observation"})
                return {"completed": True, "score_eligible": True, "final_correct": True, "stages": {},
                        "verified_correct": True, "rollout": item["rollout"]}
            episode_driver(execute, suite_groups([case], 1, 1, 42), 1, lambda: False, time.monotonic() + 30)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = EventStore(root / "run", "run")
            backend = self.backend(root, runner)
            backend.event_store = store
            coordinator = Coordinator({"kernelbench": backend}, event_store=store)
            await coordinator.run(plans)
            store.close()
            native = next(json.loads(line) for line in (store.root / "events.jsonl").read_text().splitlines()
                          if json.loads(line)["event"] == "native_event")
            self.assertEqual(native["identity"]["trial_id"], plans[0].identity.trial_id)
            self.assertEqual(native["request_id"], plans[0].identity.trial_id + "-t2")
            self.assertEqual(native["turn_id"], 2)
            self.assertEqual(native["tool_call_id"], "shell-observation")


class CliDrainTests(unittest.IsolatedAsyncioTestCase):
    async def test_cli_keeps_event_store_open_through_unproved_and_late_worker_drain(self):
        from rollout_manager.cli import run_prepared
        entered, release, finished = threading.Event(), threading.Event(), threading.Event()
        plans = KernelBenchAdapter([native_case()]).plan("run", rollouts=2)
        plans = [replace(item, task=replace(item.task,
                  resources=ResourceRequirements("fixture-executor", cpu=1))) for item in plans]

        class WriterBackend:
            def __init__(self):
                self.store, self.close_count = None, 0

            async def execute(self, item, cancellation):
                def worker():
                    entered.set()
                    cancellation.event.wait(2)
                    release.wait(2)
                    self.store.append("late_native_evidence", item.identity, request_id="retained-request")
                    finished.set()
                    return EpisodeResult(item.identity, "completed", "verified", True, "verified", {}, {})
                return await asyncio.to_thread(worker)

            async def close(self):
                self_test.assertTrue(finished.is_set())
                self.close_count += 1
                self.store.append("backend_closed_after_worker")

        backend, self_test = WriterBackend(), self

        def factory(*args, event_store=None, **kwargs):
            backend.store = event_store
            return backend

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "run"
            config = {"run_id": "run", "benchmark": "kernelbench", "backend": "native-cuda",
                      "transport": "tls-rpc", "drain_timeout": .02}
            runtime = {"args": Namespace(url="https://example.test", credentials=root / "external", rpc_timeout=1)}
            limits = {"max_workers": 1, "max_groups": 1,
                      "capacities": {"fixture-executor": ResourceRequirements("fixture-executor", cpu=1)}}
            prepared = (config, output, "b" * 64, None, plans, runtime, limits)
            with patch("rollout_manager.cli.create_backend", side_effect=factory), contextlib.redirect_stdout(io.StringIO()):
                task = asyncio.create_task(run_prepared(prepared))
                try:
                    while not entered.is_set() and not task.done():
                        await asyncio.sleep(.001)
                    self.assertTrue(entered.is_set())
                    task.cancel()
                    for _ in range(200):
                        if (output / "summary.json").exists():
                            break
                        await asyncio.sleep(.001)
                    initial_bytes = (output / "summary.json").read_bytes()
                    initial = json.loads(initial_bytes)
                    self.assertFalse(initial["drain_complete"])
                    self.assertEqual(initial["quarantined"]["fixture-executor"]["cpu"], 1)
                    self.assertEqual(backend.close_count, 0)
                    task.cancel()  # A second cancellation must not close the live writer's journal.
                    await asyncio.sleep(.01)
                    self.assertFalse(task.done())
                    release.set()
                    self.assertEqual(await asyncio.wait_for(task, 1), 2)
                    self.assertTrue(finished.is_set())
                    self.assertEqual(backend.close_count, 1)
                    self.assertEqual((output / "summary.json").read_bytes(), initial_bytes)
                    events = [json.loads(line)["event"] for line in (output / "events.jsonl").read_text().splitlines()]
                    self.assertLess(events.index("run_finished"), events.index("late_native_evidence"))
                    self.assertLess(events.index("late_native_evidence"), events.index("backend_closed_after_worker"))
                    self.assertEqual(events[-1], "late_drain_finished")
                finally:
                    release.set()
                    await asyncio.gather(task, return_exceptions=True)


if __name__ == "__main__":
    unittest.main()
