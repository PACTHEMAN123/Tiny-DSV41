"""Finite concurrency experiments using real inference and unchanged native harnesses.

This is a measurement driver, not an adversarial sandbox or an RL trainer.
Each output directory is immutable and every generated bundle requires an exact
review digest before native compilation. Network calls are never retried blindly.
"""
from __future__ import annotations

import argparse
import copy
import concurrent.futures
import hashlib
import json
import os
import queue
import shutil
import signal
import threading
import time
import uuid
from contextlib import contextmanager, nullcontext
from pathlib import Path

from .adapters.candidates import parse_candidate
from .adapters.harnesses import HarnessRunner, prepare_workspace
from .common import save_json
from .adapters.process import gpu_processes
from .common import digest
from .gpu_ownership import GpuOwnershipError, ReviewedGpuOwners
from .network import RpcClient
from .telemetry import monitor
from .host_memory import ByteBudget, available_bytes, evaluator_estimates, evaluator_memory
from .adapters.native_timing import HarnessInstrumentationError
from .cpu_layout import controlled_layout
from .workloads import GroupSchedule, suite_groups


class Journal:
    def __init__(self, root, event_sink=None):
        self.root = root
        self.lock = threading.Lock()
        self.counts = {}
        self.output = (root / "events.jsonl").open("x")
        self.event_sink = event_sink

    def event(self, job, stage, event, **details):
        record = {"unix_ns": time.time_ns(), "monotonic_ns": time.monotonic_ns(),
                  "job": job, "stage": stage, "event": event, **details}
        with self.lock:
            state = self.counts.setdefault(stage, {"queued": 0, "active": 0, "completed": 0})
            if event == "queued":
                state["queued"] += 1
            elif event == "start":
                state["queued"] -= 1
                state["active"] += 1
            elif event == "finish":
                state["active"] -= 1
                state["completed"] += 1
            elif event == "cancel_queued":
                state["queued"] -= 1
            record["queues"] = {k: dict(v) for k, v in self.counts.items()}
            self.output.write(json.dumps(record) + "\n")
            self.output.flush()
        if self.event_sink is not None:
            self.event_sink(record)
        return record

    def close(self):
        self.output.close()

    def snapshot(self):
        with self.lock:
            return {k: dict(v) for k, v in self.counts.items()}


@contextmanager
def reservation(pool, journal, job, stage, cancelled=None):
    queued = time.monotonic_ns()
    journal.event(job, stage, "queued")
    try:
        while True:
            if cancelled is not None and cancelled():
                raise RuntimeError("experiment stopped while waiting for " + stage)
            if time.monotonic_ns() - queued > 1800 * 10**9:
                raise TimeoutError(stage + " admission deadline")
            try:
                item = pool.get(timeout=.2)
                break
            except queue.Empty:
                pass
    except BaseException:
        journal.event(job, stage, "cancel_queued")
        raise
    start = time.monotonic_ns()
    timing = {"queue_seconds": (start - queued) / 1e9, "started_unix_ns": time.time_ns()}
    journal.event(job, stage, "start", **timing)
    try:
        yield item, timing
    finally:
        timing["service_seconds"] = (time.monotonic_ns() - start) / 1e9
        journal.event(job, stage, "finish", **timing)
        if not timing.get("quarantined"):
            pool.put(item)


def make_pool(items):
    pool = queue.Queue(maxsize=len(items))
    for item in items:
        pool.put(item)
    return pool


def check_headroom():
    if available_bytes() < 24 * 1024 ** 3:
        raise RuntimeError("less than 24 GiB host memory available")
    if shutil.disk_usage(".").free < 10 * 1024 ** 3:
        raise RuntimeError("less than 10 GiB storage available")


class CandidateRejected(ValueError):
    def __init__(self, sha256, reason, wait_seconds):
        super().__init__(reason)
        self.decision = {"sha256": sha256, "decision": "reject", "reason": reason,
                         "wait_seconds": wait_seconds}


class NativeInfrastructureError(RuntimeError):
    """A tool deadline or cleanup failure, carrying actual cleanup evidence."""

    def __init__(self, message, *, cleanup_state="unknown", native_evidence=None):
        super().__init__(message)
        self.cleanup_state = cleanup_state
        self.native_evidence = native_evidence


def reviewed(files, catalog):
    entry = json.loads(catalog.read_text()).get(digest(files))
    return (entry is not None and entry.get("scope") == "reviewed-development-pilot"
            and entry.get("decision", "approve") == "approve")


def feedback_request(previous, response, feedback):
    request = copy.deepcopy(previous)
    request["messages"] += [response["choices"][0]["message"], {"role": "user", "content":
        "The unchanged native harness returned this measured feedback:\n" + json.dumps(feedback) +
        "\nProduce an improved complete candidate using the same required file format. "
        "Keep the task interface and tensor shapes unchanged."}]
    return request


def await_review(files, root, catalog, timeout, journal, job, fatal=None):
    sha = digest(files)
    pending = root / "pending" / (job + ".json")
    save_json(pending, {"sha256": sha, "files": files})
    start = time.monotonic()
    journal.event(job, "review", "queued", sha256=sha)
    while True:
        entry = json.loads(catalog.read_text()).get(sha, {})
        if entry.get("scope") == "reviewed-development-pilot":
            decision = entry.get("decision", "approve")
            if decision == "reject":
                reason = entry.get("reason") or "candidate rejected during exact-source review"
                elapsed = time.monotonic() - start
                journal.event(job, "review", "cancel_queued", sha256=sha, reason=reason)
                raise CandidateRejected(sha, reason, elapsed)
            if decision == "approve":
                break
        if fatal is not None and fatal.is_set():
            journal.event(job, "review", "cancel_queued", sha256=sha)
            raise RuntimeError("experiment stopped while awaiting source review")
        if time.monotonic() - start > timeout:
            journal.event(job, "review", "cancel_queued", sha256=sha)
            raise TimeoutError("exact source review was not recorded: " + sha)
        time.sleep(.5)
    journal.event(job, "review", "start", sha256=sha)
    journal.event(job, "review", "finish", wait_seconds=time.monotonic() - start)
    return time.monotonic() - start


def probe_loop(args, root, stop, context, client_factory=None):
    client = (client_factory or RpcClient)(args.url, args.credentials)
    sequence = 0
    try:
        with (root / "probes.jsonl").open("x") as output:
            while not stop.is_set():
                size = (4096, 65536, 196608)[sequence % 3]
                request = {"data": "x" * size}
                start = time.monotonic()
                result = {"unix_ns": time.time_ns(), "level": context.get("level"), "bytes": size}
                try:
                    response = client.call("/echo", request)
                    if response["echo"] != request:
                        raise ValueError("echo mismatch")
                    result["communication"] = response["communication"]
                except Exception as exc:
                    result["error"] = str(exc)
                    context["probe_error_count"] = context.get("probe_error_count", 0) + 1
                output.write(json.dumps(result) + "\n")
                output.flush()
                sequence += 1
                stop.wait(max(0, 1 - (time.monotonic() - start)))
    finally:
        client.close()


class CancellationFlag:
    """Link native cleanup failures and external cancellation in both directions."""

    def __init__(self, external=None):
        self.local = threading.Event()
        self.external = external

    def is_set(self):
        return self.local.is_set() or (self.external is not None and self.external.is_set())

    def set(self):
        self.local.set()
        if self.external is not None:
            self.external.set()


def run(args, *, episode_driver=None, cancel_event=None, client_factory=None, event_sink=None):
    """Initialize stage pools once and preserve the native episode lifecycle.

    A supplied driver owns all batch/group admission and must finish or drain every
    execute_episode call before returning or raising. It receives
    (execute_episode, planned_groups, level, cancelled, deadline), and returns
    native result dictionaries. Cancelling a wrapper alone does not stop an RPC.
    """
    client_factory = client_factory or RpcClient
    args.root = args.root.resolve()
    args.assets = args.assets.resolve()
    args.credentials = args.credentials.resolve()
    args.catalog = args.catalog.resolve()
    args.root.mkdir(parents=True, exist_ok=False)
    cases = json.loads(args.cases.read_text())
    full_suite = getattr(args, "schedule", "cyclic") == "full-suite"
    qualification = None
    if full_suite and args.mode != "inference":
        from .reference_qualification import load_catalog
        qualification = load_catalog(getattr(args, "reference_qualification", None), args.cases, cases,
                                     preserve_integer_inputs=getattr(args, "preserve_integer_inputs", False))
        save_json(args.root / "reference-qualification.json", qualification)
    planned_groups = suite_groups(cases, getattr(args, "rollouts_per_task", 1), args.repeats,
                                 getattr(args, "sampling_seed", 42)) if full_suite else None
    frozen_metadata = None
    if getattr(args, "frozen_rollouts", None):
        if args.mode != "tools" or not full_suite or args.turns != 1:
            raise ValueError("frozen rollouts require one-turn tools-only full-suite scheduling")
        from .frozen_rollouts import load
        planned_groups, frozen_metadata = load(args.frozen_rollouts, args.cases, planned_groups,
            allow_partial=getattr(args, "partial_frozen_qualification", False))
    manifest = json.loads((args.assets / "manifest.json").read_text())
    for relative, expected in manifest["upstream_files"].items():
        if hashlib.sha256((args.assets / relative).read_bytes()).hexdigest() != expected:
            raise ValueError("upstream digest mismatch: " + relative)
    cpus = sorted(os.sched_getaffinity(0))
    gpu_ids = args.gpus.split(",") if args.gpus else []
    if args.mode != "inference" and not gpu_ids:
        raise ValueError("native evaluation needs explicit GPU UUIDs")
    if args.mode != "inference":
        for gpu in gpu_ids:
            if gpu_processes(gpu):
                raise RuntimeError("GPU already has an owner: " + gpu)
    if getattr(args, "gpu_slots", 1) != 1:
        raise ValueError("native performance scoring requires exclusive GPU slots")
    observe_inputs = getattr(args, "observe_inputs", False)
    evaluation_cores = getattr(args, "evaluation_cores", None) or args.cores_per_worker
    gpu_boundary = args.build_slots * args.cores_per_worker + len(gpu_ids) * evaluation_cores
    count = gpu_boundary
    if count > len(cpus) - 8:
        raise ValueError("worker allocation must leave eight CPU cores free")
    boundary = args.build_slots * args.cores_per_worker
    build_chunks = [cpus[i:i + args.cores_per_worker] for i in range(0, boundary, args.cores_per_worker)]
    gpu_chunks = [cpus[i:i + evaluation_cores] for i in range(boundary, gpu_boundary, evaluation_cores)]
    # Include manager/network/telemetry work in the declared logical CPU budget.
    control_cores = cpus[-8:]
    if getattr(args, "bound_cpu_budget", False):
        build_chunks, gpu_chunks, control_cores = controlled_layout(
            cpus, args.build_slots, args.cores_per_worker, len(gpu_ids), evaluation_cores)
    chunks = build_chunks + gpu_chunks
    if getattr(args, "bound_cpu_budget", False):
        os.sched_setaffinity(0, [c for chunk in chunks for c in chunk] + control_cores)
    builds = make_pool(build_chunks)
    gpus = make_pool(list(zip(gpu_ids, gpu_chunks))) if gpu_ids else None
    owners = ReviewedGpuOwners(settle_seconds=getattr(args, "gpu_settle_seconds", 5.0))
    host_capacity = getattr(args, "evaluation_host_gib", 0) * 1024**3
    host_estimates = {}
    cache_client = None
    cache_resident_ram = 0
    reuse_plans = {}
    reuse_qualifications = {}
    if full_suite and args.mode != "inference" and not host_capacity:
        raise ValueError("full-suite CUDA evaluation requires --evaluation-host-gib")
    if host_capacity and args.mode != "inference":
        registry_path = getattr(args, "input_memory_plan", None)
        model_manifest = getattr(args, "model_manifest", None)
        if not registry_path or not model_manifest:
            raise ValueError("evaluator memory planning requires --input-memory-plan and --model-manifest")
        registry = json.loads(Path(registry_path).read_text())
        host_estimates = evaluator_estimates(cases, registry,
                                             json.loads(Path(model_manifest).read_text()))
        if max(item["bytes"] for item in host_estimates.values()) > host_capacity:
            raise ValueError("one task exceeds the evaluator host-memory budget")
        reserved = host_capacity + (4*args.build_slots + 32) * 1024**3
        if getattr(args, "input_cache_socket", None):
            from .input_cache import CacheClient
            if not getattr(args, "input_cache_qualification", None) or not observe_inputs:
                raise ValueError("input reuse requires generator qualification and --observe-inputs")
            cache_client = CacheClient(args.input_cache_socket)
            cache_status = cache_client.control("snapshot")
            if cache_status["consumers"] or cache_status["generating_bytes"]:
                raise ValueError("input cache is not at an idle run boundary")
            reserved += sum(cache_status["capacity_bytes"].values())
            reserved += cache_status["metadata_bound_bytes"]
            cache_resident_ram = cache_status["shared_storage_bytes"]["ram"]
            phase = getattr(args, "input_cache_state", None)
            if phase not in ("cold", "warm") or (phase == "cold" and cache_status["entries"]):
                raise ValueError("declare cold/warm cache state; cold runs require an empty cache")
            plans_by_source = registry.get("plans", registry.get("contracts", {}))
            reuse_plans = {case["task"]["task_id"]: plans_by_source[case["task"]["source_sha256"]]
                           for case in cases if case["task"]["source_sha256"] in plans_by_source}
            reuse_qualifications = json.loads(args.input_cache_qualification.read_text())["tasks"]
            for case in cases:
                task = case["task"]["task_id"]
                if task in reuse_plans and not reuse_qualifications.get(task, {}).get("passed"):
                    raise ValueError("unqualified input reuse: " + task)
            save_json(args.root / "input-cache-initial.json", cache_status)
        if reserved > available_bytes() + cache_resident_ram:
            raise ValueError("evaluator, compiler and system allowances exceed available RAM")
    journal = Journal(args.root, event_sink=event_sink)
    host_budget = ByteBudget(host_capacity, journal.event) if host_estimates else None
    stop = threading.Event()
    fatal = CancellationFlag(cancel_event)
    old_signals = {}
    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGTERM, signal.SIGINT):
            old_signals[signum] = signal.signal(signum, lambda *_: fatal.set())
    context = {}
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    settings.update(started_ns=time.time_ns(), cpu_affinity=cpus, assigned_cpu_chunks=chunks,
                    control_cpu_ids=control_cores,
                    effective_driver_affinity=sorted(os.sched_getaffinity(0)),
                    worker_cpu_budget=count, total_declared_cpu_budget=count+len(control_cores),
                    controlled_scoring=True,
                    explicit_pinned_bytes=0, pageable_driver_staging_allowance_bytes=len(gpu_ids)*6*1024**3,
                    runtime_and_system_allowance_bytes=(2*len(gpu_ids)+4*args.build_slots+32)*1024**3,
                    review_boundary="exact digest; same UID, no adversarial containment",
                    load_model="finite closed-loop clients; each client starts next job after its prior job finishes")
    settings["frozen_inference"] = frozen_metadata
    settings["fresh_inference"] = args.mode != "tools"
    if host_budget is not None:
        settings.update(evaluator_host_budget_bytes=host_capacity, evaluator_host_estimates=host_estimates,
                        pageable_driver_staging_allowance_bytes=None,
                        runtime_and_system_allowance_bytes=(4*args.build_slots+32)*1024**3,
                        evaluator_allowances="per-task input/staging/model/runtime estimates in separate byte pool")
    save_json(args.root / "settings.json", settings)
    def manager_sample():
        return {"queues": journal.snapshot(), "gpu_free_slots": gpus.qsize() if gpus else 0,
                "evaluator_host": host_budget.snapshot() if host_budget else None,
                "input_cache": cache_client.control("snapshot") if cache_client else None}

    def auxiliary(name, function, *positional, **keywords):
        try:
            function(*positional, **keywords)
        except BaseException as exc:
            context[name + "_failure"] = {"error_type": type(exc).__name__, "error": str(exc)}

    telemetry = threading.Thread(target=auxiliary, args=("telemetry", monitor, args.root / "telemetry-B.jsonl", stop),
                                 kwargs={"extra": manager_sample} if observe_inputs or host_budget else {}, daemon=True)
    telemetry.start()
    probe = threading.Thread(target=auxiliary, args=("probe", probe_loop, args, args.root, stop, context),
                             kwargs={"client_factory": client_factory}, daemon=True)
    probe.start()
    run_id = uuid.uuid4().hex[:12]
    deadline = time.monotonic() + args.ttl

    def evaluate(workspace, case, job, stage, *, final=False, profile=False):
        host_lease = (evaluator_memory(host_budget, host_estimates[case["task"]["task_id"]],
                                      journal, job, stage, fatal.is_set)
                      if host_budget is not None else nullcontext({}))
        with host_lease as host_timing, reservation(gpus, journal, job, stage, fatal.is_set) as ((gpu, cores), timing):
            if host_budget is not None:
                timing["host_admission"] = host_timing
            if fatal.is_set():
                raise RuntimeError("experiment stopped before GPU admission")
            check_headroom()
            ownership = owners.lease(gpu, workspace) if observe_inputs else nullcontext(None)
            try:
                with ownership as guard:
                    instrumentation = {"kind": args.instrumentation if profile else "none",
                        "stages": ["evaluation-verify", "evaluation-kernelbench"]}
                    if args.instrumentation == "perf" and profile:
                        instrumentation["executable"] = str(args.perf.resolve())
                    runner = HarnessRunner(args.assets / "upstream", gpu, manifest["architecture"],
                        lambda: None, cores=cores, instrumentation=instrumentation, gpu_guard=guard,
                        cpu_threads=getattr(args, "evaluation_threads", 1),
                        cancelled=fatal.is_set,
                        preserve_integer_inputs=getattr(args, "preserve_integer_inputs", False),
                        isolate_device_inputs=getattr(args, 'isolate_device_inputs', False))
                    options = {"observe_inputs": True} if observe_inputs else {}
                    task = case["task"]["task_id"]
                    if cache_client and task in reuse_plans:
                        options["input_reuse"] = {"socket": str(args.input_cache_socket), "plan": reuse_plans[task],
                            "qualifications": {"kernelbench": reuse_qualifications[task]},
                            "consumer_copy_limit_bytes": host_estimates[task]["input_and_transfer_bytes"]}
                    result = runner.evaluate(workspace, case["task"]["dataset"], final=final, **options)
                    clean = result.get("gpu_contexts_released") if observe_inputs else result["gpu_quiescent"]
                    if not result["process_group_stopped"] or not clean:
                        raise NativeInfrastructureError(stage + " cleanup failed", native_evidence=result)
                    stopped = [process.get("stopped_for") for process in result.get("processes", [])
                               if process.get("stopped_for")]
                    if stopped:
                        raise NativeInfrastructureError(stage + " native child stopped: " + ", ".join(stopped),
                            cleanup_state="verified", native_evidence=result)
            except BaseException as exc:
                if getattr(exc, "cleanup_state", None) != "verified":
                    timing["quarantined"] = True
                    host_timing["quarantined"] = True
                fatal.set()
                raise
        return result, timing, gpu

    def turn(client, case, job, level, payload=None, profile=False):
        check_headroom()
        path = args.root / "jobs" / job
        path.mkdir(parents=True)
        start = time.monotonic_ns()
        record = {"job": job, "level": level, "task": case["task"]["task_id"],
                  "started_unix_ns": time.time_ns(), "stages": {}, "mode": args.mode,
                  "cleanup_state": "not_required"}
        record["score_eligible"] = (qualification["tasks"][record["task"]]["score_eligible"]
                                    if qualification else args.mode != "inference")
        if qualification and not record["score_eligible"]:
            record["score_exclusion_reason"] = qualification["tasks"][record["task"]]["reason"]
        if "rollout" in case:
            record["rollout"] = case["rollout"]
        try:
            files = case["files"]
            if "frozen_rollout" in case:
                frozen = case["frozen_rollout"]
                record["source_inference"] = frozen["source"]
                try:
                    files = parse_candidate(frozen["content"])
                except (ValueError, SyntaxError) as exc:
                    record["candidate_format_error"] = str(exc)
                    record["failure_class"] = "candidate_format"
                    raise
                if digest(files) != frozen.get("candidate_sha256"):
                    raise ValueError("frozen candidate differs from its recorded source digest")
                save_json(path / "candidate.json", files)
                record["candidate_sha256"] = digest(files)
            if args.mode != "tools":
                payload = payload or case["request"]
                save_json(path / "dispatch.json", {"id": job, "status": "uncertain", "request": payload})
                journal.event(job, "inference", "queued")
                journal.event(job, "inference", "start")
                infer_start = time.monotonic_ns()
                try:
                    result = client.call("/generate/" + job, payload)
                    save_json(path / "inference.json", result)
                finally:
                    infer_seconds = (time.monotonic_ns() - infer_start) / 1e9
                    journal.event(job, "inference", "finish", service_seconds=infer_seconds)
                    record["stages"]["inference"] = {"queue_seconds": 0, "service_seconds": infer_seconds}
                record["communication"] = result["communication"]
                record["usage"] = result["response"].get("usage")
                record["finish_reason"] = result["response"]["choices"][0]["finish_reason"]
                try:
                    files = parse_candidate(result["response"]["choices"][0]["message"]["content"])
                    save_json(path / "candidate.json", files)
                    record["candidate_sha256"] = digest(files)
                except (ValueError, SyntaxError) as exc:
                    record["candidate_format_error"] = str(exc)
                    record["failure_class"] = "candidate_format"
                    if args.mode != "inference":
                        raise
            if args.mode != "inference":
                record["review_wait_seconds"] = await_review(files, args.root, args.catalog,
                    args.review_timeout, journal, job, fatal)
                workspace = path / "workspace"
                with reservation(builds, journal, job, "build", fatal.is_set) as (cores, timing):
                    if fatal.is_set():
                        raise RuntimeError("experiment stopped before compiler admission")
                    check_headroom()
                    prepare_workspace(workspace, args.assets / "upstream", case["task"], files)
                    runner = HarnessRunner(args.assets / "upstream", "", manifest["architecture"],
                                           lambda: None, cores=cores, cancelled=fatal.is_set)
                    record["cleanup_state"] = "unknown"
                    built = runner.compile(workspace)
                    if not built["process_group_stopped"]:
                        timing["quarantined"] = True
                        fatal.set()
                        raise NativeInfrastructureError("compiler cleanup failed", native_evidence=built)
                    record["cleanup_state"] = "verified"
                    if built.get("stopped_for"):
                        fatal.set()
                        raise NativeInfrastructureError("compiler child stopped: " + built["stopped_for"],
                            cleanup_state="verified", native_evidence=built)
                record["stages"]["build"] = timing
                if not built["compiled"]:
                    record["compiled"] = False
                    record["failure_class"] = "candidate_compile"
                    record["feedback"] = {"compiled": False, "compiler": built["feedback"]}
                    raise ValueError("candidate failed native compilation")
                record["cleanup_state"] = "unknown"
                dev, timing, gpu = evaluate(workspace, case, job, "evaluation", profile=profile)
                record["cleanup_state"] = "verified"
                record["gpu_uuid"] = gpu
                record["development_correct"] = dev["correct"]
                record["development_timing"] = dev.get("timing", dev.get("native"))
                record["feedback"] = {"compiled": True, "correct": dev["correct"],
                    "timing": record["development_timing"], "harness_output": dev["feedback"][-6000:]}
                record["stages"]["evaluation"] = timing
                if dev["correct"] and args.turns == 1:
                    record["cleanup_state"] = "unknown"
                    final, timing, _ = evaluate(workspace, case, job, "verification", final=True)
                    record["cleanup_state"] = "verified"
                    record["final_correct"] = final["correct"]
                    record["stages"]["verification"] = timing
            record["completed"] = True
        except Exception as exc:
            if isinstance(exc, CandidateRejected):
                record["failure_class"] = "candidate_policy"
                record["source_review_decision"] = exc.decision
                record["review_wait_seconds"] = exc.decision["wait_seconds"]
            if isinstance(exc, (GpuOwnershipError, HarnessInstrumentationError, NativeInfrastructureError, OSError)):
                record["failure_class"] = "infrastructure"
                fatal.set()
            if isinstance(exc, NativeInfrastructureError):
                record["cleanup_state"] = exc.cleanup_state
                record["native_failure"] = exc.native_evidence
            if fatal.is_set() and "failure_class" not in record:
                record["failure_class"] = "cancelled"
            record.update(completed=False, error_type=type(exc).__name__, error=str(exc))
            if hasattr(exc, "rpc_failure"):
                record["communication_failure"] = exc.rpc_failure
                record["failure_class"] = "infrastructure"
                fatal.set()
            journal.event(job, "episode", "error", error=str(exc))
        finally:
            record["wall_seconds"] = (time.monotonic_ns() - start) / 1e9
            record["finished_unix_ns"] = time.time_ns()
            record["verified_correct"] = record["score_eligible"] and bool(record.get("final_correct"))
            save_json(path / "result.json", record)
        return record

    def episode(client, case, job, level, profile=False):
        if args.turns == 1:
            return turn(client, case, job, level, profile=profile)
        start = time.monotonic_ns()
        record = {"job": job, "level": level, "task": case["task"]["task_id"],
                  "started_unix_ns": time.time_ns(), "stages": {}, "mode": args.mode, "turns": [],
                  "cleanup_state": "not_required"}
        if "rollout" in case:
            record["rollout"] = case["rollout"]
        record["score_eligible"] = (qualification["tasks"][record["task"]]["score_eligible"]
                                    if qualification else args.mode != "inference")
        if qualification and not record["score_eligible"]:
            record["score_exclusion_reason"] = qualification["tasks"][record["task"]]["reason"]
        payload, best = case["request"], None
        try:
            for index in range(args.turns):
                child_job = f"{job}-t{index + 1}"
                item = turn(client, case, child_job, level, payload, profile=profile and index == 0)
                record["turns"].append(item)
                if item.get("cleanup_state") != "not_required":
                    record["cleanup_state"] = item.get("cleanup_state", "unknown")
                if item.get("failure_class") == "infrastructure":
                    record["failure_class"] = "infrastructure"
                    if "communication_failure" in item:
                        record["communication_failure"] = item["communication_failure"]
                if fatal.is_set():
                    raise RuntimeError("tool cleanup failed during feedback episode")
                if item.get("development_correct"):
                    best = args.root / "jobs" / child_job / "workspace"
                    record["selected_turn"] = index + 1
                if index + 1 < args.turns:
                    response_path = args.root / "jobs" / child_job / "inference.json"
                    if not response_path.exists() or not item.get("feedback"):
                        raise RuntimeError("cannot continue a turn without measured tool feedback")
                    response = json.loads(response_path.read_text())["response"]
                    payload = feedback_request(payload, response, item["feedback"])
            if best is not None:
                record["cleanup_state"] = "unknown"
                final, timing, _ = evaluate(best, case, job, "verification", final=True)
                record["cleanup_state"] = "verified"
                record["final_correct"] = final["correct"]
                record["stages"]["verification"] = timing
            record["completed"] = True
        except Exception as exc:
            if isinstance(exc, (GpuOwnershipError, HarnessInstrumentationError, NativeInfrastructureError, OSError)):
                record["failure_class"] = "infrastructure"
                fatal.set()
            if isinstance(exc, NativeInfrastructureError):
                record["cleanup_state"] = exc.cleanup_state
                record["native_failure"] = exc.native_evidence
            if fatal.is_set() and "failure_class" not in record:
                record["failure_class"] = "cancelled"
            record.update(completed=False, error_type=type(exc).__name__, error=str(exc))
            journal.event(job, "episode", "error", error=str(exc))
        finally:
            record["review_wait_seconds"] = sum(r.get("review_wait_seconds", 0) for r in record["turns"])
            record["usage"] = {k: sum((r.get("usage") or {}).get(k, 0) for r in record["turns"])
                               for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
            communications = [r["communication"] for r in record["turns"] if "communication" in r]
            record["communication_calls"] = communications
            record["communication"] = {k: sum(r[k] for r in communications) for k in (
                "client_total_seconds", "server_processing_seconds", "residual_seconds", "connect_tls_seconds",
                "request_body_bytes", "response_body_bytes", "tcp_retrans_delta")}
            for item in record["turns"]:
                for stage, values in item["stages"].items():
                    target = record["stages"].setdefault(stage, {"queue_seconds": 0, "service_seconds": 0})
                    for metric in ("queue_seconds", "service_seconds"):
                        target[metric] += values[metric]
            record["wall_seconds"] = (time.monotonic_ns() - start) / 1e9
            record["finished_unix_ns"] = time.time_ns()
            record["verified_correct"] = record["score_eligible"] and bool(record.get("final_correct"))
            save_json(args.root / "jobs" / job / "result.json", record)
        return record

    def worker(index, level, schedule=None):
        client = client_factory(args.url, args.credentials, timeout=getattr(args, "rpc_timeout", 900))
        records = []
        try:
            repeat = 0
            while schedule is not None or repeat < args.repeats:
                if fatal.is_set() or time.monotonic() > deadline:
                    break
                job = f"{run_id}-c{level}-w{index}-r{repeat}"
                case = (schedule.take(lambda: fatal.is_set() or time.monotonic() > deadline)
                        if schedule is not None else cases[(index + repeat) % len(cases)])
                if case is None:
                    break
                result = episode(client, case, job, level,
                    profile=index == getattr(args, "profile_worker", 0) and repeat == 0)
                records.append(result)
                if schedule is not None:
                    schedule.complete(case, result)
                repeat += 1
        finally:
            client.close()
        return records

    thread_clients = threading.local()
    managed_clients = []
    client_lock = threading.Lock()

    def execute_episode(case, job, profile=False):
        if fatal.is_set():
            raise RuntimeError("experiment stopped before episode dispatch")
        if not hasattr(thread_clients, "client"):
            thread_clients.client = client_factory(args.url, args.credentials,
                                                  timeout=getattr(args, "rpc_timeout", 900))
            with client_lock:
                managed_clients.append(thread_clients.client)
        return episode(thread_clients.client, case, job, context["level"], profile=profile)

    try:
        levels = []
        for level in args.concurrency:
            if fatal.is_set() or time.monotonic() > deadline:
                break
            context["level"] = level
            begin = time.monotonic_ns()
            started = time.time_ns()
            journal.event("level", "control", "begin", concurrency=level)
            schedule = None
            if episode_driver is not None:
                results = episode_driver(execute_episode, planned_groups, level,
                    lambda: fatal.is_set() or time.monotonic() > deadline, deadline)
            else:
                schedule = GroupSchedule(planned_groups, getattr(args, "max_inflight_groups", None) or
                    max(1, level // getattr(args, "rollouts_per_task", 1))) if full_suite else None
                with concurrent.futures.ThreadPoolExecutor(max_workers=level) as executor:
                    futures = [executor.submit(worker, index, level, schedule) for index in range(level)]
                    results = [item for future in futures for item in future.result()]
            elapsed = (time.monotonic_ns() - begin) / 1e9
            successful = sum(r["completed"] for r in results)
            tokens = sum((r.get("usage") or {}).get("completion_tokens", 0) for r in results)
            summary = {"concurrency": level, "started_unix_ns": started, "finished_unix_ns": time.time_ns(),
                       "wall_seconds": elapsed, "jobs": len(results), "completed": successful,
                       "errors": len(results) - successful, "jobs_per_second": successful / elapsed,
                       "completion_tokens": tokens, "completion_tokens_per_second": tokens / elapsed,
                       "correct": sum(r["verified_correct"] for r in results),
                       "unscorable": sum(not r["score_eligible"] for r in results), "results": results}
            if schedule is not None:
                summary["coverage"] = schedule.summary()
                save_json(args.root / f"groups-{level}.json", summary["coverage"])
            elif full_suite:
                from .workloads import external_schedule_coverage
                summary["coverage"] = external_schedule_coverage(planned_groups, results)
                save_json(args.root / f"groups-{level}.json", summary["coverage"])
            save_json(args.root / f"level-{level}.json", summary)
            levels.append(summary)
            print(json.dumps({k: v for k, v in summary.items() if k != "results"}), flush=True)
            journal.event("level", "control", "end", concurrency=level)
            if not full_suite and len(results) and successful < len(results) * .8:
                fatal.set()
        save_json(args.root / "summary.json", {"levels": levels, "fatal": fatal.is_set(),
                  "frozen_inference": frozen_metadata,
                  "measurement": "tools_only_frozen_real_inference" if frozen_metadata else args.mode,
                  "requested_levels_completed": len(levels) == len(args.concurrency) and all(
                      level["jobs"] == (level["coverage"]["expected_episodes"] if full_suite else
                                        level["concurrency"] * args.repeats) for level in levels),
                  "finished_ns": time.time_ns(), "no_profiler_during_sweep": args.instrumentation == "none"
                      and getattr(args, "external_instrumentation", "none") == "none"})
    finally:
        # The external driver must drain episode threads before returning. An
        # RPC close alone cannot prove that A stopped an already sent request.
        close_errors = []
        for client in managed_clients:
            try:
                client.close()
            except BaseException as exc:
                close_errors.append({"error_type": type(exc).__name__, "error": str(exc)})
        if cache_client:
            save_json(args.root / "input-cache-final.json", cache_client.control("snapshot"))
        if host_budget is not None:
            save_json(args.root / "evaluator-host-final.json", host_budget.snapshot())
        stop.set()
        probe.join(timeout=10)
        telemetry.join(timeout=10)
        cleanup = {"probe_thread_stopped": not probe.is_alive(),
                   "telemetry_thread_stopped": not telemetry.is_alive(),
                   "client_close_errors": close_errors,
                   "probe_error_count": context.get("probe_error_count", 0),
                   "probe_failure": context.get("probe_failure"),
                   "telemetry_failure": context.get("telemetry_failure")}
        save_json(args.root / "cleanup-summary.json", cleanup)
        journal.close()
        save_json(args.root / "gpu-ownership.json", {"settle_limit_seconds": owners.settle_seconds,
                                                   "observations": owners.observations,
                                                   "checks": owners.checks})
        for signum, previous in old_signals.items():
            signal.signal(signum, previous)
        if (close_errors or not cleanup["probe_thread_stopped"] or not cleanup["telemetry_thread_stopped"]
                or cleanup["probe_failure"] or cleanup["telemetry_failure"]):
            raise NativeInfrastructureError("native auxiliary/client cleanup was not established",
                                            native_evidence=cleanup)


def parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["inference", "tools", "coupled"], required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--credentials", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--frozen-rollouts", type=Path,
                        help="exact collected responses for a controlled tools-only comparison; no fresh inference")
    parser.add_argument("--partial-frozen-qualification", action="store_true",
                        help="allow complete task groups from an incomplete source collection; not full-cohort performance")
    parser.add_argument("--url", required=True, help="authenticated A relay origin, including for echo probes")
    parser.add_argument("--rpc-timeout", type=int, default=900,
                        help="request deadline in seconds; allow margin beyond the relay generation deadline")
    parser.add_argument("--preserve-integer-inputs", action="store_true",
                        help="explicit pinned KernelBench transfer correction; apply identically to comparison arms")
    parser.add_argument("--reference-qualification", type=Path,
                        help="full-suite native qualification; unqualified tasks stay covered but unscorable")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[64])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--schedule", choices=["cyclic", "full-suite"], default="full-suite")
    parser.add_argument("--rollouts-per-task", type=int, default=8,
                        help="samples per task and pass in full-suite mode; each is an independent request")
    parser.add_argument("--max-inflight-groups", type=int,
                        help="active task groups; default is concurrency divided by group size")
    parser.add_argument("--sampling-seed", type=int, default=42)
    parser.add_argument("--turns", type=int, choices=[1, 2], default=1,
                        help="two turns feed real tool feedback back into inference; final verify once")
    parser.add_argument("--gpus", default="")
    parser.add_argument("--build-slots", type=int, default=8)
    parser.add_argument("--cores-per-worker", type=int, default=4)
    parser.add_argument("--evaluation-cores", type=int, choices=range(1, 9),
                        help="CPU affinity size per evaluator; defaults to cores-per-worker")
    parser.add_argument("--evaluation-threads", type=int, choices=range(1, 9), default=1,
                        help="OpenMP/MKL threads per evaluator; must fit its CPU affinity")
    parser.add_argument("--gpu-slots", type=int, choices=[1], default=1,
                        help="one evaluator per GPU for uncontended performance scoring")
    parser.add_argument("--gpu-settle-seconds", type=float, default=5,
                        help="bounded wait for an uninspectable NVML context to actually disappear")
    parser.add_argument("--observe-inputs", action="store_true",
                        help="record native input generation, H2D, execution, validation and cleanup timings")
    parser.add_argument('--isolate-device-inputs', action='store_true',
                        help='new protocol: clone reference/candidate correctness inputs independently; never reinterpret old results')
    parser.add_argument("--input-cache-socket", type=Path,
                        help="opt-in immutable CPU input cache; omitted means native generation on every call")
    parser.add_argument("--input-cache-qualification", type=Path,
                        help="full-size native/RNG equivalence results for every cached generator")
    parser.add_argument("--input-cache-state", choices=["cold", "warm"],
                        help="explicit initial cache state; warm does not imply every input is resident")
    parser.add_argument("--bound-cpu-budget", action="store_true",
                        help="limit driver and descendants to worker CPUs plus eight control CPU IDs")
    parser.add_argument("--evaluation-host-gib", type=int, default=0,
                        help="separate byte pool for evaluator host-memory estimates; required for full-suite CUDA")
    parser.add_argument("--model-manifest", type=Path,
                        help="pinned model-assets.json used for evaluator model-loading estimates")
    parser.add_argument("--input-memory-plan", type=Path,
                        help="source-pinned native input memory estimates; required with evaluator host admission")
    parser.add_argument("--review-timeout", type=int, default=1800)
    parser.add_argument("--ttl", type=int, default=18000)
    parser.add_argument("--instrumentation", choices=["none", "nsys", "perf"], default="none")
    parser.add_argument("--external-instrumentation", choices=["none", "nsys", "perf"], default="none",
                        help="label separately attached serving instrumentation; does not start a profiler")
    parser.add_argument("--perf", type=Path)
    parser.add_argument("--profile-worker", type=int, default=0,
                        help="only this worker's first development verification is instrumented")
    return parser


def parse_args(argv=None):
    command_parser = parser()
    args = command_parser.parse_args(argv)
    if args.input_cache_socket and (not args.evaluation_host_gib or args.mode == "inference"):
        command_parser.error("input reuse requires evaluator host-memory admission")
    if not args.input_cache_socket and (args.input_cache_qualification or args.input_cache_state):
        command_parser.error("cache configuration without --input-cache-socket")
    if (not all(1 <= c <= 1024 for c in args.concurrency) or not 1 <= args.repeats <= 8
            or not 1 <= args.build_slots <= 16 or not 1 <= args.cores_per_worker <= 8
            or args.evaluation_host_gib < 0
            or not 1 <= args.ttl <= 21600 or not 0 <= args.gpu_settle_seconds <= 60):
        command_parser.error("bounded experiment dimensions required")
    if args.turns != 1 and args.mode != "coupled":
        command_parser.error("multiple feedback turns require coupled mode")
    if args.evaluation_threads > (args.evaluation_cores or args.cores_per_worker):
        command_parser.error("evaluation threads must fit evaluator CPU affinity")
    if args.instrumentation == "perf" and (args.perf is None or not args.perf.is_file()):
        command_parser.error("perf instrumentation requires an existing --perf executable")
    if args.profile_worker < 0 or args.profile_worker >= max(args.concurrency):
        command_parser.error("profile worker must exist in the selected concurrency levels")
    return args


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
