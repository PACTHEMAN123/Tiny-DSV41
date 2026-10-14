"""Existing CPU Pouch: original semantic groups, serial and never replayed.

``prepare`` exports only public first-turn payloads for A's mounted tokenizer.
``run`` consumes A's immutable lease and uses the real Coordinator/Harbor backend.
No services or machines are created here; each original trial owns a fresh sandbox.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time


def configure_source(config):
    source = Path(config["source_root"])
    if not source.is_absolute() or not source.is_dir() or any(p.is_symlink() for p in (source, *source.parents)):
        raise ValueError("explicit real CPU source_root is required")
    sys.path.insert(0, str(source))
    import rollout_manager
    if source not in Path(rollout_manager.__file__).parents:
        raise ValueError("CPU package import differs from selected source_root")
    if socket.gethostname() != config["expected_hostname"]:
        raise ValueError("CPU executor hostname differs from current inspected identity")


def execution_shape(config):
    from rollout_manager.terminal_bench.sealed import TASKS
    tasks = config.get("task_ids", list(TASKS))
    size = config.get("group_size", 2)
    if (not isinstance(tasks, list) or not tasks or len(set(tasks)) != len(tasks)
            or any(task not in TASKS for task in tasks) or type(size) is not int or not 1 <= size <= 64):
        raise ValueError("select unique prepared tasks and a group size in 1..64")
    if size == 1 and config.get("purpose") != "policy_evaluation":
        raise ValueError("singleton episodes require explicit policy_evaluation purpose")
    return tuple(tasks), size


def public_capsule(config):
    from rollout_manager.terminal_bench.sealed import TASKS, TASK_PIN, PAYLOAD_PIN, file_sha256, initial_payload
    from rollout_manager.datasets import TerminalBenchAdapter
    from rollout_manager.generation_training import evidence_digest
    from rollout_manager.terminal_bench import bootstrap_payload
    from rollout_manager.terminal_bench import harbor_bootstrap_ubuntu_merge_sparql as profile
    from rollout_manager.terminal_bench.task_assets import checked_task, qualification_proof
    from rollout_manager.terminal_bench.controller import task_row
    import rollout_manager.terminal_bench.harbor_filerelay_agent as agent

    runtime = config["terminal_bench"]
    execution_shape(config)
    from rollout_manager.terminal_bench.serving_contracts import TRAINING_PROFILES
    if (config.get("schema_version") != 1 or runtime.get("serving_profile") not in TRAINING_PROFILES
            or runtime.get("execution_profile") != "training"
            or (runtime["task_provenance_sha256"], runtime["payload_sha256"]) != (TASK_PIN, PAYLOAD_PIN)
            or type(config["temperature"]) not in (int, float) or not 0 < config["temperature"] <= 2):
        raise ValueError("CPU experiment requires explicit qualified native training profile and temperature")
    if (not isinstance(config.get("run_id"), str) or not 1 <= len(config["run_id"]) <= 100
            or type(config.get("first_group_id", 0)) is not int or config.get("first_group_id", 0) < 0):
        raise ValueError("a fresh bounded run identity and nonnegative first_group_id are required")
    bootstrap_payload.verify(runtime["payload_dir"], PAYLOAD_PIN, profile=profile.PROFILE, tasks=profile.PAYLOAD_TASKS)
    profile.verify_overlay(Path(runtime["payload_dir"]).parent / "git-overlay")
    adapter = TerminalBenchAdapter(executor=config["executor"],
        task_root=Path(runtime["task_copy_dir"]) / "tasks", task_ids=TASKS)
    rows = {}
    for name in TASKS:
        task, _ = checked_task(runtime["task_copy_dir"], TASK_PIN, name, TASKS)
        spec, instruction = adapter.resolve(name)
        row = task_row(task, name)
        review = runtime.get("bootstrap_failure_reviews", {}).get(name)
        if review is not None:
            from rollout_manager.backends.terminal_bench import TerminalBenchBackend
            TerminalBenchBackend(runtime)._scoped(review)
        baseline = qualification_proof(runtime["bootstrap_qualifications"][name], name, TASK_PIN, PAYLOAD_PIN,
                                       failure_review=review)
        reset = {"schema_version": 1, "kind": "fresh-original-pouch-task-v1", "task": asdict(spec),
                 "task_provenance_sha256": TASK_PIN, "payload_sha256": PAYLOAD_PIN,
                 "bootstrap_qualification": baseline, "network_mode": "none",
                 "original_task_budgets": row["budgets_seconds"],
                 "attempts_per_sample": 1, "retries": 0, "new_owned_container_per_sample": True}
        payload = initial_payload(instruction, runtime["model"], config["temperature"])
        rows[name] = {"instruction_sha256": spec.source_sha256, "payload": payload,
                      "payload_sha256": evidence_digest(payload), "reset_spec": reset,
                      "reset_sha256": evidence_digest(reset)}
    return {"schema_version": 1, "kind": "terminalbench-public-payloads", "tasks": rows,
            "served_model": runtime["model"], "agent_sha256": file_sha256(Path(agent.__file__))["sha256"],
            "task_provenance_sha256": TASK_PIN, "payload_sha256": PAYLOAD_PIN,
            "only_public_instructions_exported": True}


def prepared(config):
    from rollout_manager.terminal_bench.sealed import read_json, TASKS
    from rollout_manager.generation_training import TrainingGenerationContract, evidence_digest
    from rollout_manager.training import PolicyRef
    from rollout_manager.contracts import ResourceRequirements
    from rollout_manager.backends.terminal_bench import TerminalBenchBackend

    capsule, lease = read_json(config["public_capsule"], 1024**2), read_json(config["lease_capsule"], 1024**2)
    current = public_capsule(config)
    if capsule != current or lease.get("public_capsule_sha256") != evidence_digest(current):
        raise ValueError("CPU original task, bootstrap or public payload changed since A prepared the lease")
    if (lease.get("schema_version") != 1 or lease.get("kind") != "terminalbench-policy-lease"
            or lease.get("served_model") != current["served_model"] or lease.get("agent_sha256") != current["agent_sha256"]
            or lease.get("cpu_cleanup_is_gpu_drain") is not False or set(lease["tasks"]) != set(TASKS)):
        raise ValueError("A's pre-admission policy lease is incomplete or unsupported")
    policy = PolicyRef.from_dict(lease["policy"])
    _, size = execution_shape(config)
    if (lease.get("group_size", 2) != size
            or size == 1 and lease.get("purpose") != "policy_evaluation"):
        raise ValueError("CPU group size or evaluation purpose differs from A's immutable lease")
    TrainingGenerationContract(policy, lease["capabilities"])
    if policy.sampling["temperature"] != config["temperature"]:
        raise ValueError("CPU sampling differs from the immutable A lease")
    for task_id in TASKS:
        for key in ("instruction_sha256", "payload_sha256", "reset_sha256"):
            if lease["tasks"][task_id][key] != current["tasks"][task_id][key]:
                raise ValueError("A lease differs from current original task preparation")
    output = Path(config["output_dir"])
    if not output.is_absolute() or output.exists() or output.is_symlink() or not output.parent.is_dir():
        raise ValueError("CPU output_dir must be new under an existing parent")
    runtime = {**config["terminal_bench"], "output_dir": str(output / "terminal-bench"),
               "training_policy": policy.to_dict(), "training_capabilities": lease["capabilities"]}
    digest = evidence_digest({"config": config, "public_capsule": capsule, "lease": lease})
    runtime["configuration_sha256"] = digest
    TerminalBenchBackend.validate_config(runtime)
    backend = TerminalBenchBackend(runtime)
    for key in ("output_dir", "relay_root", "model_guard_evidence", "payload_dir", "task_copy_dir"):
        backend._scoped(runtime[key])
    if set(config["capacity"]) != {"cpu", "memory_bytes", "storage_bytes"}:
        raise ValueError("explicit inspected CPU capacity is required")
    capacity = ResourceRequirements(config["executor"], **config["capacity"])
    if any(getattr(capacity, key) <= 0 for key in ("cpu", "memory_bytes", "storage_bytes")):
        raise ValueError("inspected CPU capacities must be positive")
    return output, runtime, digest, capsule, lease, policy, capacity


async def run(config, values):
    from rollout_manager.terminal_bench.sealed import write_new, write_seals, evaluation_receipt
    from rollout_manager.benchmark_training import GroupAssembler, GroupRegistration, RewardPolicy
    from rollout_manager.contracts import CancellationToken, EpisodeIdentity, EpisodePlan
    from rollout_manager.coordinator import Coordinator
    from rollout_manager.datasets import TerminalBenchAdapter
    from rollout_manager.events import EventStore
    from rollout_manager.training import TrainingGroup
    from rollout_manager.transports import FileRelayTransport
    from rollout_manager.backends.terminal_bench import TerminalBenchBackend
    from rollout_manager.terminal_bench import admission

    output, runtime, digest, capsule, lease, policy, capacity = values
    task_ids, size = execution_shape(config)
    store = EventStore(output, config["run_id"])
    for name, value in (("config.json", config), ("public-capsule.json", capsule), ("lease.json", lease)):
        write_new(output / name, value)
    token = CancellationToken()
    backend = TerminalBenchBackend(runtime, transport=FileRelayTransport(runtime["relay_root"]))
    coordinator = Coordinator({"terminal-bench": backend}, max_workers=1, max_groups=1,
        capacities={capacity.executor: capacity}, event_store=store, policy="terminal-waves")
    loop = asyncio.get_running_loop()
    signals = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        signals[signum] = signal.getsignal(signum)
        loop.add_signal_handler(signum, token.cancel, signal.Signals(signum).name)
    records, closed = [], False
    started = time.time()
    try:
        await coordinator.open_stream(max_pending=size, cancellation=token)
        adapter = TerminalBenchAdapter(executor=capacity.executor,
            task_root=Path(runtime["task_copy_dir"]) / "tasks", task_ids=task_ids)
        for group_id, task in enumerate(adapter.enumerate(), start=config.get("first_group_id", 0)):
            row = lease["tasks"][task.task_id]
            plans = tuple(EpisodePlan(EpisodeIdentity(config["run_id"], "terminal-bench", task.task_id,
                "semantic-%d" % group_id, sample, trial_id=hashlib.sha256(
                    (config["run_id"] + ":%d:%d" % (group_id, sample)).encode()).hexdigest()[:32]), task,
                {"training_policy": policy.to_dict(), "training_capabilities": lease["capabilities"]})
                for sample in range(size))
            registered = None if size == 1 else GroupRegistration(group_id, plans, policy,
                row["initial_context_sha256"], row["reset_sha256"],
                RewardPolicy("terminal-bench", "terminal-validated-verifier-v1"))
            registration_file = "group-%d.registration.json" % group_id
            registration_value = {"group_id": group_id, "plans": [asdict(p) for p in plans],
                "policy": policy.to_dict(), "initial_context_sha256": row["initial_context_sha256"],
                "reset_sha256": row["reset_sha256"]}
            write_new(output / registration_file, registration_value)
            assembler = None if size == 1 else GroupAssembler(max_groups=1, max_episodes=size)
            if assembler is not None:
                assembler.register(registered)
            results = await coordinator.execute_stream_group(plans, sequence=group_id)
            receipt, episode_files = None, []
            for index, result in enumerate(results):
                relative = "group-%d.sample-%d.episode.json" % (group_id, index)
                write_new(output / relative, asdict(result))
                episode_files.append(relative)
                if assembler is not None:
                    receipt = assembler.complete(result)
                else:
                    try:
                        receipt = evaluation_receipt(registration_value, result)
                    except (ValueError, TypeError, KeyError) as error:
                        from rollout_manager.benchmark_training import GroupExclusion
                        receipt = GroupExclusion(group_id, 1, (str(error),), (result,))
            if receipt is None:
                raise RuntimeError("CPU coordinator did not complete every expected sibling")
            receipt_file = "group-%d.receipt.json" % group_id
            write_new(output / receipt_file, receipt.to_dict())
            from rollout_manager.terminal_bench.sealed import PolicyEvaluation
            eligible = isinstance(receipt, (TrainingGroup, PolicyEvaluation))
            records.append({"group_id": group_id, "registration_file": registration_file,
                "episode_files": episode_files, "receipt_file": receipt_file,
                "receipt_kind": ("policy_evaluation" if size == 1 else "training_group")
                    if eligible else "excluded_training_group"})
            print(json.dumps({"event": "group_finished", "group_id": group_id, "task_id": task.task_id,
                "receipt_kind": records[-1]["receipt_kind"], "rewards": [s.reward for s in receipt.samples]
                    if eligible else None}, allow_nan=False), flush=True)
            if token.cancelled():
                break
    finally:
        try:
            await coordinator.close_stream()
            admission.check_empty_spool(Path(runtime["relay_root"]))
            closed = True
        finally:
            for signum, previous in signals.items():
                loop.remove_signal_handler(signum)
                signal.signal(signum, previous)
            store.close()
    closure = {"coordinator_closed": closed, "cpu_spool_terminal": closed,
        "lifecycle_failures": coordinator.lifecycle_failures, "quarantined": coordinator.quarantined,
        "reserved": coordinator.reserved, "admission_stopped_reason": token.reason,
        "worker_hostname": socket.gethostname(), "worker_pid": os.getpid(),
        "started_unix": started, "finished_unix": time.time(), "configuration_sha256": digest,
        "generation_slots": 1, "max_workers": 1, "groups_completed": len(records),
        "gpu_delivery_proven": False}
    write_new(output / "cpu-closure.json", closure)
    manifests = write_seals(output, records, closure)
    publication = config.get("artifact_publication")
    if publication is not None:
        from rollout_manager.terminal_bench.sealed_artifacts import publish_sealed
        if publication.get("relay_root") != runtime["relay_root"] or set(publication.get("keys", {})) != {str(row["group_id"]) for row in records}:
            raise ValueError("automatic publication must identify each original closed group on its executing relay")
        for row in records:
            publish_sealed(runtime["relay_root"], output, row["group_id"], publication["keys"][str(row["group_id"])])
    print(json.dumps({"event": "cpu_sealed", "manifests": list(map(str, manifests)), "closure": closure}), flush=True)
    expected_kind = "policy_evaluation" if size == 1 else "training_group"
    return 0 if len(records) == len(task_ids) and all(r["receipt_kind"] == expected_kind for r in records) and not token.cancelled() else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="new public capsule for prepare")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    configure_source(config)
    if args.action == "prepare":
        if args.output is None:
            parser.error("prepare requires --output")
        from rollout_manager.terminal_bench.sealed import write_new
        write_new(args.output, public_capsule(config))
        print(json.dumps({"public_capsule": str(args.output), "model_generation_executed": False}))
        return 0
    values = prepared(config)
    from rollout_manager.execution_environment import prepare_worker_environment, validate_worker_environment
    if not os.environ.get("ROLLOUT_WORKER_ROOT"):
        environment = prepare_worker_environment(values[1], values[2])
        os.umask(0o077)
        os.execve(sys.executable, [sys.executable, "-B", str(Path(__file__).absolute()), *sys.argv[1:]], environment)
    validate_worker_environment(values[1], values[2])
    return asyncio.run(run(config, values))


if __name__ == "__main__":
    raise SystemExit(main())
