"""One admitted Harbor episode; the shared coordinator owns all wave scheduling."""
from __future__ import annotations

import asyncio
from contextlib import ExitStack
from dataclasses import asdict
import hashlib
from importlib import metadata, resources
import json
from pathlib import Path
import sys
import time
from typing import Mapping

from ..contracts import EpisodeResult


def _identity_key(identity):
    return hashlib.sha256(json.dumps(asdict(identity), sort_keys=True).encode()).hexdigest()


class TerminalBenchBackend:
    """CPU-only Pouch execution with stock Harbor verification and no retries.

    External payload/task copies retain the frozen release's pins. Starting this
    backend acquires the existing browser relay lock; it never starts a service.
    """

    def __init__(self, config: Mapping, transport=None):
        self.config = dict(config)
        self.transport = transport
        self._stack = ExitStack()
        self._admitted = {}
        self._jobs = set()
        self._started = False
        self._stopped = False
        self._seen_episode_ids = set()

    @property
    def training_enabled(self):
        return self.config.get("execution_profile", "evaluation") == "training"

    def _training_contract(self, plan=None):
        from ..generation_training import TrainingGenerationContract
        payload = plan.payload if plan is not None and plan.payload is not None else {}
        base = TrainingGenerationContract(self.config.get("training_policy"), self.config.get("training_capabilities"))
        if not {"training_policy", "training_capabilities"} & set(payload):
            return base
        if not {"training_policy", "training_capabilities"} <= set(payload):
            raise ValueError("A group lease override requires both policy and current serving capabilities")
        contract = TrainingGenerationContract(payload["training_policy"], payload["training_capabilities"])
        def fixed_identity(policy):
            return {key: value for key, value in policy.identities.items() if key != "run_identity_sha256"}
        if (contract.policy.lineage != base.policy.lineage or fixed_identity(contract.policy) != fixed_identity(base.policy) or
                dict(contract.policy.sampling) != dict(base.policy.sampling)):
            raise ValueError("Group lease changes the negotiated model/tokenizer/template/sampling identity")
        return contract

    @staticmethod
    def validate_config(config):
        required = ("executor_root", "output_dir", "relay_root", "model_guard_evidence",
                    "payload_dir", "task_copy_dir", "task_provenance_sha256", "payload_sha256", "bootstrap_qualifications")
        missing = [key for key in required if not config.get(key)]
        if missing:
            raise ValueError("Harbor/Pouch configuration requires: " + ", ".join(missing))
        if config.get("profile", "ubuntu-merge-sparql") != "ubuntu-merge-sparql":
            raise ValueError("Only the selected ubuntu-merge-sparql bootstrap is implemented")
        if config.get("execution_profile", "evaluation") not in {"evaluation", "training"}:
            raise ValueError("execution_profile must be evaluation or explicit training")
        max_turns = config.get("agent_max_turns", 128)
        if type(max_turns) is not int or not 1 <= max_turns <= 1024:
            raise ValueError("agent_max_turns must be an integer in 1..1024")
        serving_profile = config.get("serving_profile", "vllm-v1")
        from ..terminal_bench.controller import TRAINING_PROFILES
        if serving_profile not in {"vllm-v1", *TRAINING_PROFILES}:
            raise ValueError("Unsupported explicit Terminal-Bench serving profile")
        if serving_profile in TRAINING_PROFILES and (config.get("execution_profile") != "training" or
                not isinstance(config.get("model"), str) or not config["model"]):
            raise ValueError("Native serving requires explicit model identity and training lease")
        if config.get("execution_profile") == "training":
            from ..generation_training import TrainingGenerationContract
            contract = TrainingGenerationContract(config.get("training_policy"), config.get("training_capabilities"))
            if contract.policy.sampling["temperature"] > 2:
                raise ValueError("Harbor training temperature exceeds the selected agent bound")
        if config.get("socket_path", "/var/run/pouchd.sock") != "/var/run/pouchd.sock":
            raise ValueError("Preserve the inspected existing Pouch socket")
        if not isinstance(config["bootstrap_qualifications"], dict) or any(
                not isinstance(name, str) or not isinstance(path, str) or not Path(path).is_absolute()
                for name, path in config["bootstrap_qualifications"].items()):
            raise ValueError("bootstrap_qualifications maps each selected task to its absolute existing baseline directory")
        cpu_root = Path(config["executor_root"])
        for path in config["bootstrap_qualifications"].values():
            if ".." in Path(path).parts or cpu_root not in Path(path).parents:
                raise ValueError("bootstrap qualification paths must be children of the explicit CPU executor root")
        reviews = config.get("bootstrap_failure_reviews", {})
        if (not isinstance(reviews, dict) or any(name not in config["bootstrap_qualifications"] or
                not isinstance(path, str) or not Path(path).is_absolute() or ".." in Path(path).parts or
                cpu_root not in Path(path).parents for name, path in reviews.items())):
            raise ValueError("bootstrap_failure_reviews maps qualified tasks to private files below the CPU executor root")
        for key in ("executor_root", "output_dir", "relay_root", "model_guard_evidence",
                    "payload_dir", "task_copy_dir"):
            if not Path(config[key]).is_absolute():
                raise ValueError(key + " must be an explicit absolute executor path")

    def _scoped(self, value):
        root, path = Path(self.config["executor_root"]), Path(value)
        if not path.is_absolute() or ".." in path.parts or root not in path.parents:
            raise ValueError("Execution files must be children of the explicit CPU executor root")
        if any(part.is_symlink() for part in (path, *path.parents)):
            raise ValueError("Linked CPU execution path")
        return path

    async def start(self):
        self.validate_config(self.config)
        if sys.version_info < (3, 12):
            raise RuntimeError("Selected Harbor/Pouch runtime requires Python 3.12+")
        from ..execution_environment import validate_worker_environment
        validate_worker_environment(self.config, self.config.get("configuration_sha256"))
        if self._started:
            raise ValueError("Harbor backend start is single-use")
        try:
            version = metadata.version("harbor")
        except metadata.PackageNotFoundError as error:
            raise RuntimeError("Terminal-Bench requires the optional Harbor 0.23.0 environment (Python 3.12+)") from error
        if version != "0.23.0":
            raise RuntimeError("Selected plugins require Harbor 0.23.0")
        from ..terminal_bench import admission
        for key in ("output_dir", "relay_root", "model_guard_evidence", "payload_dir", "task_copy_dir"):
            self._scoped(self.config[key])
        output = Path(self.config["output_dir"])
        if output.exists() or output.is_symlink() or not output.parent.is_dir():
            raise ValueError("Use a new output directory under an existing CPU parent; never resume/replay")
        root = Path(self.config["relay_root"])
        self._stack.enter_context(admission.lock_experiment(root))
        try:
            admission.check_empty_spool(root)
            output.mkdir(mode=0o700)
        except BaseException:
            self._stack.close()
            raise
        self._started = True

    def _checked_plan(self, plan):
        from ..terminal_bench import bootstrap_payload as payload
        from ..terminal_bench import harbor_bootstrap_ubuntu_merge_sparql as profile
        from ..terminal_bench.task_assets import checked_task, qualification_proof
        from ..terminal_bench.controller import task_row
        if plan.task.benchmark != "terminal-bench" or plan.task.resources.gpu_count:
            raise ValueError("Harbor/Pouch requires a CPU-only Terminal-Bench task")
        if self.training_enabled:
            if (type(plan.identity.sample_index) is not int or plan.identity.sample_index < 0 or
                    plan.identity.pass_index != 0):
                raise ValueError("Training permits independent nonnegative samples, one pass each")
            self._training_contract(plan)
        elif plan.identity.sample_index != 0 or plan.identity.pass_index != 0:
            raise ValueError("Terminal-Bench preserves one attempt; extra samples/passes are invalid")
        if (self.config["task_provenance_sha256"], self.config["payload_sha256"]) != (profile.TASK_PIN, profile.PAYLOAD_PIN):
            raise ValueError("Selected bootstrap requires exact task-copy and payload pins")
        task, _ = checked_task(self.config["task_copy_dir"], profile.TASK_PIN, plan.task.task_id, profile.TASKS)
        if Path(plan.task.payload["task_path"]) != task.paths.task_dir:
            raise ValueError("Dataset task path differs from the pinned original task copy")
        if hashlib.sha256(task.paths.instruction_path.read_bytes()).hexdigest() != plan.task.source_sha256:
            raise ValueError("Dataset instruction source hash differs from the original task copy")
        proof_dir = self.config.get("bootstrap_qualifications", {}).get(plan.task.task_id)
        if not proof_dir:
            raise ValueError("Selected task lacks preserved original bootstrap qualification")
        self._scoped(proof_dir)
        review = self.config.get("bootstrap_failure_reviews", {}).get(plan.task.task_id)
        if review is not None:
            self._scoped(review)
        proof = qualification_proof(proof_dir, plan.task.task_id, profile.TASK_PIN, profile.PAYLOAD_PIN,
                                    failure_review=review)
        payload.verify(self.config["payload_dir"], profile.PAYLOAD_PIN, profile=profile.PROFILE, tasks=profile.PAYLOAD_TASKS)
        profile.verify_overlay(Path(self.config["payload_dir"]).parent / "git-overlay")
        row = task_row(task, plan.task.task_id)
        row["bootstrap_qualification"] = proof
        for shared, original in (("build_seconds", "environment_build"), ("agent_seconds", "agent"),
                                 ("verifier_seconds", "verifier")):
            if plan.task.budgets.get(shared) != row["budgets_seconds"][original]:
                raise ValueError("Shared benchmark budget differs from original Harbor task")
        if plan.task.payload.get("image") != row["image"]:
            raise ValueError("Shared image reference differs from original Harbor task")
        demand = row["resources"]
        declared = plan.task.resources
        if (declared.cpu, declared.memory_bytes, declared.storage_bytes) != (
                demand["cpus"], demand["memory_mb"] * 1024**2, demand["storage_mb"] * 1024**2):
            raise ValueError("Shared reservation differs from original CPU task resources")
        return row

    async def admit_group(self, plans):
        """Fresh original wave gate, with no resource reservation or scheduler."""
        if not self._started or self._stopped or self._admitted:
            raise ValueError("Harbor admission is stopped or a previous wave remains open")
        from ..terminal_bench import admission, controller
        if not plans or len(plans) > 64:
            raise ValueError("Admit 1..64 independently identified original trials")
        keys = [_identity_key(plan.identity) for plan in plans]
        if len(set(keys)) != len(keys) or set(keys) & self._seen_episode_ids:
            raise ValueError("Episode identities cannot be duplicated or replayed")
        if self.training_enabled:
            if (len({(p.identity.run_id, p.identity.group_id, p.task.task_id) for p in plans}) != 1 or
                    len({p.identity.sample_index for p in plans}) != len(plans)):
                raise ValueError("Training admission requires independent samples of one task/group")
            policies = [self._training_contract(plan).policy for plan in plans]
            if any(policy != policies[0] for policy in policies[1:]):
                raise ValueError("Every training sample must retain the same leased policy")
        elif len({p.task.task_id for p in plans}) != len(plans):
            raise ValueError("Evaluation admits unique pinned tasks, one attempt each")
        rows = [self._checked_plan(plan) for plan in plans]
        relay = Path(self.config["relay_root"])
        admission.check_empty_spool(relay)
        required = max(row["required_service_seconds"] for row in rows)
        model = self.config.get("model", controller.MODEL)
        serving_profile = self.config.get("serving_profile", "vllm-v1")
        if serving_profile not in controller.TRAINING_PROFILES and model != controller.MODEL:
            raise ValueError("Selected guard contract requires the recorded served model identity")
        native = {}
        if serving_profile in controller.TRAINING_PROFILES:
            if not self.training_enabled:
                raise ValueError("Native serving admission requires an explicit training lease")
            native = {"serving_profile": serving_profile,
                      "expected_policy": self._training_contract(plans[0]).policy.to_dict()}
        ready = admission.check_readiness(relay, model, len(rows), required, required_generation_seconds=360, **native)
        observed = controller.read_json(Path(self.config["model_guard_evidence"]), 65536)
        guard = controller.guard_gate(observed, ready, required, time.time(),
            **({**native, "model": model} if native else {}))
        record = {"required_seconds": required, "readiness": ready, "model_guard": guard,
                  "episode_ids": [_identity_key(plan.identity) for plan in plans],
                  "tasks": rows,
                  "resources_owned_by": "shared_coordinator"}
        controller.write_new(Path(self.config["output_dir"]) / ("admission-" + record["episode_ids"][0] + ".json"), record)
        self._admitted = {_identity_key(plan.identity): row for plan, row in zip(plans, rows)}
        self._seen_episode_ids.update(keys)
        return record

    def build_config(self, plan, output):
        from harbor.models.job.config import JobConfig
        from ..terminal_bench import harbor_bootstrap_ubuntu_merge_sparql as profile
        map_path = self.config.get("image_map_path")
        if map_path:
            mapping = Path(map_path).read_bytes()
        else:
            mapping = resources.files("rollout_manager.terminal_bench").joinpath("data/harbor-pouch-image-id-map-69671f.json").read_bytes()
        if hashlib.sha256(mapping).hexdigest() != profile.IMAGE_MAP_PIN:
            raise ValueError("Original image identity map pin differs")
        return JobConfig.model_validate({
            "job_name": "episode", "jobs_dir": str(output / "jobs"),
            "n_attempts": 1, "n_concurrent_trials": 1, "timeout_multiplier": 1,
            "retry": {"max_retries": 0}, "quiet": True,
            "environment": {"import_path": "rollout_manager.terminal_bench.harbor_rollout_environment:ProfiledUbuntuMergeSparqlEnvironment",
                "delete": True, "force_build": False, "cpu_enforcement_policy": "limit", "memory_enforcement_policy": "limit",
                "kwargs": {"socket_path": self.config.get("socket_path", "/var/run/pouchd.sock"),
                    "image_id_by_reference": json.loads(mapping), "allow_public_network": False,
                    "bootstrap_payload_dir": str(self.config["payload_dir"]), "bootstrap_manifest_sha256": profile.PAYLOAD_PIN}},
            "verifier": {"disable": False},
            "agents": [{"import_path": "rollout_manager.terminal_bench.harbor_filerelay_agent:FileRelayAgent",
                "model_name": self.config.get("model", "deepseek-ai/DeepSeek-V4.1-Flash"),
                "kwargs": {"relay_root": str(self.config["relay_root"]), "generation_seconds": 360,
                    "max_tokens": 2048, "max_turns": self.config.get("agent_max_turns", 128), "command_seconds": 30,
                    "temperature": self._training_contract(plan).policy.sampling["temperature"] if self.training_enabled else 0}}],
            "tasks": [{"path": plan.task.payload["task_path"]}]})

    async def _run_job(self, job, cancellation):
        task = asyncio.create_task(job.run())
        self._jobs.add(task)
        async def watch():
            while not task.done() and not cancellation.cancelled():
                await asyncio.sleep(0.05)
            if not task.done():
                task.cancel()
        monitor = asyncio.create_task(watch())
        try:
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                if not task.done():
                    cancellation.cancel("coordinator_cancelled")
                    task.cancel()
                # One cancellation enters Harbor finalization. Drain that task;
                # repeated caller cancellation must not interrupt owned cleanup.
                while not task.done():
                    try:
                        await asyncio.shield(task)
                    except asyncio.CancelledError:
                        pass
                if not task.cancelled():
                    task.result()
        finally:
            monitor.cancel()
            await asyncio.gather(monitor, return_exceptions=True)
            self._jobs.discard(task)

    async def execute(self, plan, cancellation):
        from ..terminal_bench import controller
        key = _identity_key(plan.identity)
        if key not in self._admitted:
            raise ValueError("Episode must be admitted by the shared coordinator first")
        if cancellation.cancelled():
            return EpisodeResult(plan.identity, "cancelled", "not_run", False, "not_required",
                {"kind": "terminal-bench", "official_reward": None, "validated_reported_reward": None}, [],
                failure_category="cancelled", source_sha256=plan.task.source_sha256)
        # Revalidate immutable task/payload bytes immediately before Harbor owns
        # the episode; this never uploads hidden tests to the model.
        self._checked_plan(plan)
        output = Path(self.config["output_dir"]) / ("episode-" + key)
        output.mkdir(mode=0o700)
        config = self.build_config(plan, output)
        config_value = config.model_dump(mode="json")
        configuration_sha = hashlib.sha256(json.dumps(config_value, sort_keys=True).encode()).hexdigest()
        controller.write_new(output / "config.json", config_value)
        from ..terminal_bench.harbor_filerelay_agent import relay_factory
        transport = self.transport
        if self.training_enabled:
            from ..transports import FileRelayTransport
            if transport is None:
                transport = FileRelayTransport(self.config["relay_root"])
            if not hasattr(transport, "with_training") or not hasattr(transport, "harbor_relay"):
                raise ValueError("Harbor training requires the established browser FileRelay evidence hook")
            transport = transport.with_training(self._training_contract(plan))
        factory = transport.harbor_relay if transport is not None else None
        token = relay_factory.set(factory)
        error = None
        started = False
        try:
            from harbor.job import Job
            from harbor.trial.hooks import TrialEvent
            job = await Job.create(config)
            async def hook(event):
                row = {"name": event.event.value, "task_id": event.task_name, "trial": event.trial_name,
                       "episode_identity": asdict(plan.identity), "clock_domain": "cpu_monotonic",
                       "ts_ns": time.monotonic_ns(), "wall_unix": time.time()}
                from os import O_APPEND, O_CREAT, O_NOFOLLOW, O_WRONLY, fdopen, open as osopen
                with fdopen(osopen(output / "harbor-phases.jsonl", O_WRONLY | O_CREAT | O_APPEND | O_NOFOLLOW, 0o600), "a") as stream:
                    stream.write(json.dumps(row) + "\n")
            for event in TrialEvent:
                job.add_hook(event, hook)
            started = True
            await self._run_job(job, cancellation)
        except Exception as exc:
            error = type(exc).__name__
        finally:
            relay_factory.reset(token)
        job_dir = output / "jobs/episode"
        trials = [p for p in job_dir.iterdir() if p.is_dir() and not p.is_symlink() and (p / "result.json").is_file()] if job_dir.is_dir() else []
        if len(trials) != 1:
            self._stopped = started
            return EpisodeResult(plan.identity, "infrastructure_error", "not_run", False,
                "uncertain" if started else "not_required", {"kind": "terminal-bench", "official_reward": None,
                "validated_reported_reward": None, "issues": [error or "missing_or_duplicate_trial_result"]},
                {"episode_dir": str(output)}, failure_category="infrastructure_error",
                request_state={"state": "outcome_unknown" if started else "not_submitted"},
                configuration_sha256=configuration_sha, source_sha256=plan.task.source_sha256)
        trial = trials[0]
        row = controller.classify_trial(trial)
        original = controller.read_json(trial / "result.json")
        if row["task_id"] != plan.task.task_id:
            raise ValueError("Harbor trial identity differs from the admitted dataset task")
        try:
            agent = controller.read_json(trial / "agent/file-relay/agent-result.json")
        except (ValueError, OSError):
            agent = {}
        states = {}
        for state_path in (trial / "agent/file-relay").glob("*.state.json"):
            state = controller.read_json(state_path)
            states[state["id"]] = state
        for request_id, state in list(states.items()):
            if state.get("state") != "terminal_failure_received":
                continue
            from ..generation_failure import validate_generation_failure, failure_is_terminal
            try:
                request = controller.read_json(trial / "agent/file-relay" / (request_id + ".request.json"))
                policy = self._training_contract(plan).policy if self.training_enabled else None
                expected_ack = ({key: getattr(policy, key) for key in ("version", "digest", "lineage")}
                                if policy is not None else None)
                receipt = validate_generation_failure(state.get("failure_receipt"), request["payload"],
                                                      request_id, expected_ack)
                if (request.get("id") != request_id or state.get("generation_failure") != "validated"
                        or state.get("request_payload_sha256") != receipt["request_payload_sha256"]
                        or not failure_is_terminal(receipt)):
                    raise ValueError("Durable terminal failure closure differs")
            except (OSError, ValueError, TypeError, KeyError):
                # Derive uncertainty without changing original durable files.
                states[request_id] = {**state, "state": "terminal_failure_closure_unknown"}
        uncertain = (any("unknown" in state.get("state", "") for state in states.values()) or
                     bool(set(row["request_ids"]) - set(states)))
        eligible = row["status"] == "evaluated" and row["cleanup_verified"] and not uncertain and error is None
        training_turns = None
        training_error = None
        if self.training_enabled:
            traces = transport.training_traces
            training_turns = [trace.to_dict() for trace in traces]
            durable_training = True
            for trace in traces:
                record_path = trial / "agent/file-relay" / (trace.request_id + ".training.json")
                state = states.get(trace.request_id)
                try:
                    saved = controller.read_json(record_path)
                except (ValueError, OSError):
                    saved = None
                if not state or state.get("state") != "response_received" or saved != trace.to_dict():
                    durable_training = False
                    continue
                # Derive a common marker from both durable files. The original
                # spool state is never rewritten or treated as a training claim.
                from ..generation_training import evidence_digest
                states[trace.request_id] = {**state, "training_evidence": "validated",
                    "training_record": str(record_path), "training_record_sha256": evidence_digest(saved)}
            if (not traces or len(traces) != agent.get("turns") or
                    not durable_training or
                    len({trace.request_id for trace in traces}) != len(traces) or
                    {trace.request_id for trace in traces} != set(row["request_ids"])):
                training_error = "missing_or_mismatched_training_turn_evidence"
                eligible = False
        if not eligible:
            self._stopped = True
        outcome = {"kind": "terminal-bench", **row, "validated_reported_reward": row["reward"],
                   "review_classification": row["status"], "verification_owned_by_harbor": True,
                   "original_verifier_evidence": str(trial / "original-verifier-terminal.json"),
                   "agent_turns": agent.get("turns"), "request_count": len(row["request_ids"]),
                   "generation_retries": 0, "attempts": 1}
        if self.training_enabled:
            outcome.update(execution_profile="training", training_turns=training_turns,
                           training_evidence_error=training_error,
                           training_policy=self._training_contract(plan).policy.to_dict())
        native = {"episode_dir": str(output), "trial_dir": str(trial), "result": str(trial / "result.json"),
                  "ctrf": str(trial / "verifier/ctrf.json"), "reward": str(trial / "verifier/reward.txt"),
                  "verifier_stdout": str(trial / "verifier/test-stdout.txt"),
                  "cleanup": str(trial / "pouch-environment.json"), "trajectory": str(trial / "agent/file-relay/trajectory.json")}
        context = original.get("agent_result")
        usage = ({name: context.get(name) for name in ("n_input_tokens", "n_output_tokens", "n_cache_tokens")}
                 if isinstance(context, dict) else None)
        timings = {"clock_domain": "cpu_wall_utc", "phases": {name: original.get(name) for name in
                   ("environment_setup", "agent_setup", "agent_execution", "verifier")},
                   "nested_samples": str(trial / "environment-phases.jsonl"), "additive": False}
        return EpisodeResult(plan.identity, "completed" if row["status"] != "infrastructure_error" and training_error is None else "infrastructure_error",
            row["status"], bool(eligible), "verified" if row["cleanup_verified"] else "uncertain",
            outcome, native, usage=usage, timings=timings, model_identity=self.config.get("model", controller.MODEL),
            failure_category="infrastructure_error" if error or training_error or row["status"] == "infrastructure_error" else None,
            request_state={"state": "outcome_unknown" if uncertain else "terminal_failures_closed"
                           if any(state.get("state") == "terminal_failure_received" for state in states.values())
                           else "response_received", "requests": states,
                           "remote_cancellation_proven": False}, configuration_sha256=configuration_sha,
            source_sha256=plan.task.source_sha256)

    async def finish_group(self, plans, results):
        from ..terminal_bench import admission
        if self._jobs:
            self._stopped = True
            return {"spool_terminal": False, "state": "job_cleanup_uncertain"}
        try:
            admission.check_empty_spool(Path(self.config["relay_root"]))
            terminal = True
        except Exception:
            terminal = False
            self._stopped = True
        self._admitted.clear()
        return {"spool_terminal": terminal, "remote_cancellation_proven": False,
                "scope": "CPU durable terminal responses; GPU cancellation is not inferred"}

    async def close(self):
        try:
            if self._jobs:
                await asyncio.gather(*self._jobs, return_exceptions=True)
            if self._started:
                from ..terminal_bench import admission
                admission.check_empty_spool(Path(self.config["relay_root"]))
        finally:
            self._stack.close()
