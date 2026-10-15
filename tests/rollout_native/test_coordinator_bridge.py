"""Offline regressions for the shared coordinator's native lifecycle bridge."""
import asyncio
from dataclasses import replace
import hashlib
import json
import tempfile
import threading
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from rollout_manager import runner
from rollout_manager.adapters.datasets import DATASETS
from rollout_manager.backends.native import NativeBackend
from rollout_manager.contracts import CancellationToken
from rollout_manager.datasets import KernelBenchAdapter
from rollout_manager.generation_training import TrainingGenerationContract, evidence_digest
from rollout_manager.training import PolicyRef
from rollout_manager.transports import RpcTransport
from rollout_manager.workloads import external_schedule_coverage, suite_groups


class NativeCoordinatorBridgeTests(unittest.TestCase):
    def args(self, base):
        (base / "manifest.json").write_text(json.dumps({"upstream_files": {}, "architecture": "10.0"}))
        cases = [{"task": {"task_id": "kernelbench/example"}, "files": {}, "request": {}}]
        (base / "cases.json").write_text(json.dumps(cases))
        return Namespace(root=base / "run", assets=base, credentials=base, catalog=base / "catalog.json",
            cases=base / "cases.json", mode="inference", gpus="", build_slots=1, cores_per_worker=1,
            ttl=60, turns=1, repeats=1, concurrency=[1], url="https://example.invalid",
            review_timeout=1, instrumentation="none", schedule="full-suite", rollouts_per_task=2,
            max_inflight_groups=1, rpc_timeout=1860)

    def test_external_driver_is_only_scheduler_and_clients_close_after_drain(self):
        clients, events = [], []

        class Client:
            def __init__(self, url, credentials, timeout):
                self.closed = False
                self.calls = []
                clients.append(self)

            def call(self, route, request):
                self.calls.append(route)
                return {"response": {"choices": [{"finish_reason": "stop", "message": {"content": "{}"}}],
                                     "usage": {"completion_tokens": 3}}, "communication": {}}

            def close(self):
                self.closed = True

        def dispatch(execute, groups, level, cancelled, deadline):
            records = [execute(case, f"bridge-{index}") for _, samples in groups
                       for index, case in enumerate(samples)]
            self.assertFalse(cancelled())
            self.assertFalse(clients[0].closed)
            self.assertEqual(len(clients), 1)
            return records

        with tempfile.TemporaryDirectory() as directory:
            args = self.args(Path(directory))
            with patch.object(runner.os, "sched_getaffinity", return_value=set(range(16)), create=True), \
                    patch.object(runner, "check_headroom"), patch.object(runner, "monitor"), \
                    patch.object(runner, "probe_loop"), \
                    patch.object(runner, "GroupSchedule", side_effect=AssertionError("duplicate scheduler")):
                runner.run(args, episode_driver=dispatch, client_factory=Client, event_sink=events.append)
            summary = json.loads((args.root / "summary.json").read_text())
            journal = [json.loads(line) for line in (args.root / "events.jsonl").read_text().splitlines()]
        self.assertTrue(clients[0].closed)
        self.assertEqual(len(clients[0].calls), 2)
        self.assertEqual(events, journal)
        self.assertTrue(summary["requested_levels_completed"])
        self.assertEqual(summary["levels"][0]["coverage"]["expected_episodes"], 2)

    def test_uncertain_dispatch_cancels_coordinator_without_replay(self):
        cancellation, calls = threading.Event(), []

        class Client:
            def __init__(self, *args, **kwargs):
                pass

            def call(self, route, request):
                calls.append(route)
                error = TimeoutError("recorded uncertain response")
                error.rpc_failure = {"phase": "response_headers", "dispatch_may_have_occurred": True,
                                     "automatic_replay": False}
                raise error

            def close(self):
                pass

        def dispatch(execute, groups, level, cancelled, deadline):
            result = execute(groups[0][1][0], "uncertain")
            self.assertTrue(cancelled())
            return [result]

        with tempfile.TemporaryDirectory() as directory:
            args = self.args(Path(directory))
            with patch.object(runner.os, "sched_getaffinity", return_value=set(range(16)), create=True), \
                    patch.object(runner, "check_headroom"), patch.object(runner, "monitor"), \
                    patch.object(runner, "probe_loop"):
                runner.run(args, episode_driver=dispatch, cancel_event=cancellation, client_factory=Client)
            summary = json.loads((args.root / "summary.json").read_text())
        self.assertTrue(cancellation.is_set())
        self.assertEqual(calls, ["/generate/uncertain"])
        self.assertFalse(summary["requested_levels_completed"])
        self.assertEqual(summary["levels"][0]["results"][0]["failure_class"], "infrastructure")

    def test_external_coverage_rejects_changed_or_duplicate_sample_identity(self):
        groups = suite_groups([{"task": {"task_id": "task"}, "request": {}}], 2, 1, 42)
        case = groups[0][1][0]
        record = {"task": "task", "rollout": case["rollout"], "job": "j",
                  "started_unix_ns": 10, "finished_unix_ns": 20}
        partial = external_schedule_coverage(groups, [record])
        self.assertEqual(partial["incomplete_groups"][groups[0][0]]["remaining"], 1)
        for records in ([record, record], [record | {"rollout": record["rollout"] | {"sampling_seed": 1}}]):
            with self.assertRaises(ValueError):
                external_schedule_coverage(groups, records)

    def test_native_tool_failure_uses_actual_cleanup_proof(self):
        class Client:
            def __init__(self, *args, **kwargs):
                pass

            def call(self, *args):
                return {"response": {"choices": [{"finish_reason": "stop", "message": {"content": "{}"}}]},
                        "communication": {}}

            def close(self):
                pass

        for failure, category, cleanup in (
            ("compile_bad", "candidate_compile", "verified"),
            ("compile_cleanup", "infrastructure", "unknown"),
            ("evaluation_deadline", "infrastructure", "verified"),
            ("evaluation_cleanup", "infrastructure", "unknown"),
        ):
            class Harness:
                def __init__(self, *args, **kwargs):
                    pass

                def compile(self, *args):
                    return {"compiled": failure != "compile_bad", "feedback": "fixture compiler",
                            "process_group_stopped": failure != "compile_cleanup", "stopped_for": None}

                def evaluate(self, *args, **kwargs):
                    return {"correct": False, "feedback": "fixture native", "process_group_stopped": True,
                            "gpu_quiescent": failure != "evaluation_cleanup", "processes": [
                                {"stopped_for": "deadline" if failure == "evaluation_deadline" else None}]}

            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                args = self.args(Path(directory))
                args.mode, args.schedule, args.gpus = "coupled", "cyclic", "GPU-fixture"
                case = json.loads(args.cases.read_text())[0]
                case["task"]["dataset"] = "kernelbench"
                args.cases.write_text(json.dumps([case]))
                with patch.object(runner.os, "sched_getaffinity", return_value=set(range(16)), create=True), \
                        patch.object(runner, "check_headroom"), patch.object(runner, "monitor"), \
                        patch.object(runner, "probe_loop"), patch.object(runner, "gpu_processes", return_value=[]), \
                        patch.object(runner, "HarnessRunner", Harness), patch.object(runner, "prepare_workspace"), \
                        patch.object(runner, "parse_candidate", return_value={}), \
                        patch.object(runner, "await_review", return_value=0):
                    runner.run(args, client_factory=Client)
                record = json.loads((args.root / "level-1.json").read_text())["results"][0]
                self.assertEqual(record["failure_class"], category)
                self.assertEqual(record["cleanup_state"], cleanup)

    def test_auxiliary_or_client_close_failure_is_saved_and_surfaced(self):
        for failed_component in ("telemetry", "client_close"):
            class Client:
                def __init__(self, *args, **kwargs):
                    pass

                def call(self, *args):
                    return {"response": {"choices": [{"finish_reason": "stop", "message": {"content": "{}"}}]},
                            "communication": {}}

                def close(self):
                    if failed_component == "client_close":
                        raise OSError("fixture close failure")

            def dispatch(execute, groups, *args):
                return [execute(case, f"job-{i}") for _, samples in groups for i, case in enumerate(samples)]

            with self.subTest(failed_component=failed_component), tempfile.TemporaryDirectory() as directory:
                args = self.args(Path(directory))
                with patch.object(runner.os, "sched_getaffinity", return_value=set(range(16)), create=True), \
                        patch.object(runner, "check_headroom"), patch.object(runner, "probe_loop"), \
                        patch.object(runner, "monitor", side_effect=(OSError("fixture telemetry failure")
                                     if failed_component == "telemetry" else None)):
                    with self.assertRaises(runner.NativeInfrastructureError):
                        runner.run(args, episode_driver=dispatch, client_factory=Client)
                cleanup = json.loads((args.root / "cleanup-summary.json").read_text())
                self.assertTrue(cleanup["probe_thread_stopped"])
                self.assertTrue(cleanup["telemetry_thread_stopped"])
                self.assertTrue(cleanup["telemetry_failure"] if failed_component == "telemetry"
                                else cleanup["client_close_errors"])


class NativeTrainingBridgeTests(unittest.IsolatedAsyncioTestCase):
    def fixture(self, root, *, invalid_evidence=False, turns=1):
        policy = PolicyRef(2, "a" * 64, "fixture", identities={
            "tokenizer_sha256": "b" * 64, "chat_template_sha256": "c" * 64,
            "run_identity_sha256": "d" * 64})
        def capabilities(value):
            return {"schema_version": 1, "policy": value.to_dict(), "logprob_convention": "sampled_distribution",
                    **{name: True for name in TrainingGenerationContract.REQUIRED}}
        source = "class Model: pass\n"
        case = {"task": {"task_id": "kernelbench/level_2/0", "dataset": "kernelbench",
            "revision": DATASETS["kernelbench"]["revision"], "code": source,
            "source_sha256": hashlib.sha256(source.encode()).hexdigest()}, "files": {},
            "request": {"model": "fixture-model", "temperature": 1, "max_tokens": 16,
                        "messages": [{"role": "user", "content": "public task"}]}}
        samples = KernelBenchAdapter([case]).plan("run", rollouts=2)
        leased = replace(policy, version=3, digest="e" * 64,
                         identities={**dict(policy.identities), "run_identity_sha256": "f" * 64})
        samples = [replace(p, payload={**p.payload, "training_policy": leased.to_dict(),
                    "training_capabilities": capabilities(leased)}) for p in samples]
        calls = []
        class Client:
            def __init__(self, *args, **kwargs): pass
            def close(self): pass
            def call(self, route, request):
                calls.append(route)
                response = {"model": request["model"], "choices": [{"finish_reason": "stop",
                    "message": {"role": "assistant", "content": "candidate"}}]}
                if not invalid_evidence:
                    response["training_evidence"] = {"schema_version": 1,
                        "request_id": route.removeprefix("/generate/"),
                        "request_payload_sha256": evidence_digest(request),
                        "response_sha256": evidence_digest(response), "policy": leased.to_dict(),
                        "context_token_ids": [1, 2], "action_token_ids": [3],
                        "behavior_log_probs": [-.5], "finish_reason": "stop", "stop_token_count": 0,
                        "context_truncation": "none", "logprob_convention": "sampled_distribution"}
                return {"response": response, "communication": {}}
        transport = RpcTransport("https://example.invalid", root / "credentials", root / "ledger", client_type=Client)
        def run(args, *, episode_driver, client_factory, cancel_event, **kwargs):
            def execute(case, job, profile):
                client = client_factory(transport.url, transport.credentials)
                record = {"completed": True, "score_eligible": True, "cleanup_state": "verified",
                    "final_correct": True, "stages": {}, "turns": []}
                try:
                    for index in range(turns):
                        request_id = job if turns == 1 else job + f"-t{index + 1}"
                        client.call("/generate/" + request_id, case["request"])
                        if turns > 1:
                            record["turns"].append({"job": request_id, "stages": {}})
                except Exception as error:
                    record.update(completed=False, final_correct=None, failure_class="infrastructure",
                                  communication_failure=error.rpc_failure)
                finally:
                    client.close()
                return record
            return episode_driver(execute, [("group", [p.payload for p in samples])], 2,
                                  cancel_event.is_set, float("inf"))
        args = Namespace(schedule="full-suite", concurrency=[2], turns=turns, root=root / "native", profile_worker=-1)
        backend = NativeBackend({"args": args, "execution_profile": "training",
            "training_policy": policy.to_dict(), "training_capabilities": capabilities(policy)}, transport,
            runner_function=run)
        return backend, samples, leased, calls

    async def test_independent_native_samples_capture_current_plan_lease_and_feedback_turns(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend, samples, leased, calls = self.fixture(Path(tmp), turns=2)
            await backend.start()
            try:
                results = await asyncio.gather(*(backend.execute(plan, CancellationToken()) for plan in samples))
            finally:
                await backend.close()
            self.assertEqual(len(calls), 4)
            for plan, result in zip(samples, results):
                traces = result.native_outcome["training_turns"]
                self.assertEqual([turn["turn_id"] for turn in traces], [1, 2])
                self.assertEqual([turn["request_id"] for turn in traces],
                    [plan.identity.trial_id + "-t1", plan.identity.trial_id + "-t2"])
                self.assertEqual([turn["policy"] for turn in traces], [leased.to_dict()] * 2)
                self.assertFalse(result.request_state["uncertain"])
                self.assertTrue(result.score_eligible)
                with self.assertRaisesRegex(ValueError, "closed"):
                    backend.transport.client_factory(backend.transport.url, backend.transport.credentials).call(
                        "/generate/" + plan.identity.trial_id + "-t3", plan.payload["request"])

    async def test_known_response_invalid_tokens_stop_training_without_remote_uncertainty(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend, samples, _, calls = self.fixture(Path(tmp), invalid_evidence=True)
            await backend.start()
            try:
                result = await backend.execute(samples[0], CancellationToken())
            finally:
                await backend.close()
            self.assertEqual(len(calls), 1)
            self.assertEqual(result.native_outcome["training_turns"], [])
            self.assertEqual(result.failure_category, "infrastructure")
            self.assertFalse(result.request_state["uncertain"])
            states = list(result.request_state["requests"].values())
            self.assertEqual([(s["state"], s["training_evidence"]) for s in states], [("response_received", "rejected")])
            self.assertFalse(result.score_eligible)
