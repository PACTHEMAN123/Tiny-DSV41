"""A persistent native lifecycle whose batch admission belongs to Coordinator."""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import itertools
import threading
from pathlib import Path

from ..contracts import EpisodeResult


class NativeBackend:
    def __init__(self, config, transport, *, event_store=None, runner_function=None):
        self.args = config["args"]
        self.config = config
        self.transport, self.event_store = transport, event_store
        self.runner_function = runner_function
        self.execution_profile = config.get("execution_profile", "evaluation")
        if self.execution_profile not in {"evaluation", "training"}:
            raise ValueError("native execution_profile must be evaluation or training")
        if self.execution_profile == "training":
            from ..generation_training import TrainingGenerationContract
            self.training_contract = TrainingGenerationContract(
                config.get("training_policy"), config.get("training_capabilities"))
            if not hasattr(transport, "bind_episode") or not hasattr(transport, "episode_traces"):
                raise ValueError("native training requires an episode-bound generation transport")
        self.cancel_event, self.stop_driver = threading.Event(), threading.Event()
        self.records, self.cases, self.jobs = [], {}, {}
        self.pool = None
        self.worker_state = threading.local()

    async def start(self):
        from ..runner import run
        if self.args.schedule != "full-suite" or len(self.args.concurrency) != 1:
            raise ValueError("shared native admission requires full-suite and one concurrency level; use native compatibility CLI for cyclic sweeps")
        loop = asyncio.get_running_loop()
        self.ready, self.done = loop.create_future(), loop.create_future()

        def resolve(future, value=None, error=None):
            if not future.done():
                if error is not None:
                    future.set_exception(error)
                else:
                    future.set_result(value)

        def driver(execute, groups, level, cancelled, deadline):
            self.execute_native, self.native_cancelled = execute, cancelled
            for _, cases in groups:
                for case in cases:
                    r = case["rollout"]
                    self.cases[(case["task"]["task_id"], r["pass_index"], r["sample_index"])] = case
            worker_indices = itertools.count()
            def initialize_worker():
                self.worker_state.index = next(worker_indices)
                self.worker_state.started = False
            self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=level, thread_name_prefix="native-episode",
                                                             initializer=initialize_worker)
            loop.call_soon_threadsafe(resolve, self.ready)
            self.stop_driver.wait()
            # shutdown waits for actual native workers, including RPC/process cleanup.
            self.pool.shutdown(wait=True, cancel_futures=False)
            return list(self.records)

        def event(record):
            if self.event_store:
                job = record.get("job", "")
                identity = self.jobs.get(job)
                if identity is None and "-t" in job:
                    identity = self.jobs.get(job.rsplit("-t", 1)[0])
                self.event_store.append("native_event", identity, native_record=record,
                    trial_id=identity.trial_id if identity else None,
                    turn_id=record.get("turn_id", job.rsplit("-t", 1)[1] if "-t" in job else None),
                    request_id=record.get("request_id", job if record.get("stage") == "inference" else None),
                    tool_call_id=record.get("tool_call_id"))

        def lifecycle():
            try:
                (self.runner_function or run)(self.args, episode_driver=driver,
                    cancel_event=self.cancel_event, client_factory=self.transport.client_factory,
                    event_sink=event)
            except BaseException as error:
                def failed(error=error):
                    if not self.ready.done():
                        resolve(self.ready, error=error)
                    resolve(self.done, error=error)
                loop.call_soon_threadsafe(failed)
            else:
                def finished():
                    if not self.ready.done():
                        resolve(self.ready, error=RuntimeError("native lifecycle stopped before episode readiness"))
                    resolve(self.done)
                loop.call_soon_threadsafe(finished)
        self.thread = threading.Thread(target=lifecycle, name="native-lifecycle", daemon=True)
        self.thread.start()
        await asyncio.shield(self.ready)

    async def execute(self, plan, cancellation):
        if cancellation.cancelled() or self.native_cancelled():
            return EpisodeResult(plan.identity, "not_admitted", "not_performed", False, "not_required",
                {"kind": "native-cuda", "reason": "native cancellation or TTL before dispatch"}, {},
                source_sha256=plan.task.source_sha256)
        key = (plan.identity.task_id, plan.identity.pass_index, plan.identity.sample_index)
        case = self.cases[key]  # Includes validated frozen payload substitutions from the native loader.
        rollout = case["rollout"]
        if (rollout["group_id"] != plan.identity.group_id or
                rollout["sampling_seed"] != plan.identity.sampling_seed or
                case["task"]["source_sha256"] != plan.task.source_sha256 or
                case["task"]["dataset"] != plan.identity.benchmark):
            raise ValueError("episode identity/source differs from pinned native case")
        if not plan.identity.trial_id:
            raise ValueError("native episode requires a durable trial/request identity")
        job = plan.identity.trial_id
        self.jobs[job] = plan.identity
        if self.execution_profile == "training":
            from ..generation_training import TrainingGenerationContract
            policy = plan.payload.get("training_policy")
            capabilities = plan.payload.get("training_capabilities")
            if (policy is None) != (capabilities is None):
                raise ValueError("native training lease requires both policy and capabilities")
            contract = (TrainingGenerationContract(policy, capabilities) if policy is not None
                        else self.training_contract)
            fixed_identities = lambda policy: {key: value for key, value in policy.identities.items()
                                               if key != "run_identity_sha256"}
            if (contract.policy.lineage != self.training_contract.policy.lineage or
                    fixed_identities(contract.policy) != fixed_identities(self.training_contract.policy) or
                    dict(contract.policy.sampling) != dict(self.training_contract.policy.sampling)):
                raise ValueError("native training lease changed run or sampling identities")
            self.transport.bind_episode(plan.identity, contract=contract)
        def execute_on_worker():
            profile = (self.worker_state.index == getattr(self.args, "profile_worker", 0)
                       and not self.worker_state.started)
            self.worker_state.started = True
            return self.execute_native(case, job, profile)
        future = asyncio.wrap_future(self.pool.submit(execute_on_worker))
        try:
            while not future.done():
                if cancellation.cancelled():
                    self.cancel_event.set()
                await asyncio.wait({future}, timeout=.05)
        except asyncio.CancelledError:
            cancellation.cancel("native_waiter_cancelled")
            self.cancel_event.set()
            # Caller cancellation is a request to stop, never evidence that this
            # compiler/evaluator/RPC thread stopped. Retain its actual future.
            while not future.done():
                try:
                    await asyncio.shield(future)
                except asyncio.CancelledError:
                    pass
        try:
            record = future.result()
        finally:
            training_turns = (self.transport.episode_traces(plan.identity, close=True)
                              if self.execution_profile == "training" else None)
        self.records.append(record)
        failure = record.get("communication_failure")
        infrastructure = record.get("failure_class") == "infrastructure"
        uncertain = bool(failure and failure.get("dispatch_may_have_occurred") and
                         not failure.get("response_received"))
        # Read durable transport evidence, including every feedback-turn request.
        states = {}
        state_evidence_failed = False
        if self.transport.ledger_root.exists():
            for path in self.transport.ledger_root.glob(job + "*.json"):
                try:
                    states[path.stem] = json.loads(path.read_text())
                except (OSError, ValueError):
                    state_evidence_failed = True
        uncertain = uncertain or state_evidence_failed or any(state.get("uncertain") is True for state in states.values())
        ownership = any(t.get("quarantined") for t in record.get("stages", {}).values())
        cleanup = record.get("cleanup_state", "unknown" if ownership else "not_required")
        usage = record.get("usage")
        # Legacy two-turn totals zero-fill missing usage; retain null and individual observations instead.
        if self.args.turns > 1:
            turns = record.get("turns", [])
            usage = {"turns": [turn.get("usage") for turn in turns], "aggregate": None} if turns else None
        outcome = {"kind": "native-cuda", "development_correct": record.get("development_correct"),
                   "final_correct": record.get("final_correct"), "verified_correct": record.get("verified_correct"),
                   "native_performance": record.get("development_timing"), "record": record}
        if training_turns is not None:
            outcome["training_turns"] = [turn.to_dict() for turn in training_turns]
        outcome["turns"] = [{"turn_id": index + 1,
            "development_correct": turn.get("development_correct"),
            "native_performance": turn.get("development_timing"),
            "request_id": turn.get("job")} for index, turn in enumerate(record.get("turns", []))]
        verified = record.get("final_correct")
        verification = "passed" if verified is True else "failed" if verified is False else "not_performed"
        if record.get("failure_class") == "candidate_policy":
            verification = "source_approval_rejected"
        score_eligible = bool(record.get("score_eligible"))
        if self.execution_profile == "training":
            score_eligible = (score_eligible and bool(record.get("completed")) and
                not record.get("failure_class") and bool(training_turns) and bool(states) and
                not uncertain and cleanup in {"verified", "not_required"} and
                all(state.get("state") == "response_received" and
                    state.get("training_evidence") == "validated" for state in states.values()))
        return EpisodeResult(plan.identity, "completed" if record.get("completed") else "failed", verification,
            score_eligible, cleanup, outcome,
            {"result": str(self.args.root / "jobs" / job / "result.json"),
             "events": str(self.args.root / "events.jsonl")}, usage=usage,
            timings={"clock_domain": "native_executor_monotonic", "nested": True,
                     "wall_seconds": record.get("wall_seconds"),
                     "stages": record.get("stages") if self.args.turns == 1 else
                               {k: v for k, v in record.get("stages", {}).items() if k == "verification"},
                     "communication": record.get("communication") if self.args.turns == 1 else None,
                     "turns": [{"turn_id": index + 1, "stages": turn.get("stages"),
                                "communication": turn.get("communication"), "wall_seconds": turn.get("wall_seconds")}
                               for index, turn in enumerate(record.get("turns", []))]},
            failure_category="infrastructure" if infrastructure else record.get("failure_class"),
            model_identity=case.get("request", {}).get("model"),
            request_state={"uncertain": uncertain, "requests": states},
            configuration_sha256=self.config.get("configuration_sha256"), source_sha256=plan.task.source_sha256)

    async def close(self):
        self.stop_driver.set()
        if hasattr(self, "done"):
            await asyncio.shield(self.done)
