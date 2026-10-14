"""Unified CLI. inspect/validate/plan are entirely offline and read-only."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sys
from pathlib import Path

from .backends import COMBINATIONS, create_backend, validate_combination
from .contracts import CancellationToken, ResourceRequirements
from .coordinator import Coordinator
from .datasets import CudaAgentAdapter, KernelBenchAdapter, TerminalBenchAdapter, sha256_json
from .events import EventStore, json_value


def _native_args(settings, output, base):
    from .runner import parse_args, parser
    options = {action.dest: action for action in parser()._actions}
    argv = []
    for key, value in {**settings, "root": str(output / "native")}.items():
        if key not in options or key == "help":
            raise ValueError("unknown native option: " + key)
        action = options[key]
        flag = action.option_strings[0]
        if isinstance(action, argparse._StoreTrueAction):
            if not isinstance(value, bool):
                raise ValueError(key + " must be boolean")
            if value:
                argv.append(flag)
        else:
            values = value if isinstance(value, list) else [value]
            if action.type is Path:
                values = [str(_path(item, base)) for item in values]
            argv.extend([flag, *map(str, values)])
    args = parse_args(argv)
    if args.schedule != "full-suite" or len(args.concurrency) != 1:
        raise ValueError("unified admission uses full-suite and one concurrency level; native compatibility supports legacy sweeps")
    if args.frozen_rollouts and (args.mode != "tools" or args.turns != 1):
        raise ValueError("frozen rollouts require one-turn tools-only full-suite scheduling")
    return args


def _path(value, base):
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def prepare(config_path, benchmark, backend, transport):
    validate_combination(benchmark, backend, transport)
    path = Path(config_path).resolve()
    config = json.loads(path.read_text())
    if config.get("schema_version") != 1:
        raise ValueError("configuration requires schema_version=1")
    for key, value in (("benchmark", benchmark), ("backend", backend), ("transport", transport)):
        if config.get(key) != value:
            raise ValueError("explicit CLI selection differs from configuration " + key)
    run_id = config["run_id"]
    if not isinstance(run_id, str) or not run_id or len(run_id) > 100:
        raise ValueError("run_id must be a nonempty bounded string")
    output = _path(config["output_dir"], path.parent)
    if output.exists():
        raise ValueError("output_dir must be new; runs are never resumed or replayed")
    digest = sha256_json(config)
    if backend == "native-cuda":
        args = _native_args(config["native"], output, path.parent)
        from .transports import validate_rpc_origin
        validate_rpc_origin(args.url)
        adapter = (KernelBenchAdapter if benchmark == "kernelbench" else CudaAgentAdapter)(
            args.cases, executor=config.get("executor", "native-evaluator"), mode=args.mode)
        plans = adapter.plan(run_id, rollouts=args.rollouts_per_task,
                             passes=args.repeats, seed=args.sampling_seed)
        if args.frozen_rollouts:
            from .frozen_rollouts import load
            from .workloads import suite_groups
            groups, _ = load(args.frozen_rollouts, args.cases,
                suite_groups(adapter.cases, args.rollouts_per_task, args.repeats, args.sampling_seed),
                allow_partial=args.partial_frozen_qualification)
            selected = {(c["task"]["task_id"], c["rollout"]["pass_index"], c["rollout"]["sample_index"])
                        for _, cases in groups for c in cases}
            plans = [p for p in plans if (p.identity.task_id,p.identity.pass_index,p.identity.sample_index) in selected]
        runtime = {"args": args, "configuration_sha256": digest}
        limits = {"max_workers": args.concurrency[0],
                  "max_groups": args.max_inflight_groups or max(1, args.concurrency[0] // args.rollouts_per_task),
                  "policy": "native-groups", "capacities": {}}
    else:
        from .backends.terminal_bench import TerminalBenchBackend
        runtime = dict(config["terminal_bench"])
        runtime["output_dir"] = str(output / "terminal-bench")
        runtime["configuration_sha256"] = digest
        for key in ("executor_root", "relay_root", "model_guard_evidence", "payload_dir", "task_copy_dir", "image_map_path"):
            if key in runtime:
                runtime[key] = str(_path(runtime[key], path.parent))
        TerminalBenchBackend.validate_config(runtime)
        scoped = TerminalBenchBackend(runtime)
        for key in ("output_dir", "relay_root", "model_guard_evidence", "payload_dir", "task_copy_dir"):
            scoped._scoped(runtime[key])
        for qualification in runtime["bootstrap_qualifications"].values():
            scoped._scoped(qualification)
        executor = config["executor"]
        demand = config["capacity"]
        if set(demand) != {"cpu", "memory_bytes", "storage_bytes"}:
            raise ValueError("Terminal-Bench capacity requires original CPU/memory/storage admission limits")
        capacity = ResourceRequirements(executor, **demand)
        if not all(getattr(capacity, key) > 0 for key in demand):
            raise ValueError("CPU executor capacities must be positive")
        workers = config.get("max_workers", 2)
        if type(workers) is not int or not 1 <= workers <= 64:
            raise ValueError("Terminal-Bench preserves a bounded 1..64 worker limit")
        adapter = TerminalBenchAdapter(executor=executor, task_root=Path(runtime["task_copy_dir"]) / "tasks",
                                       task_ids=config.get("task_ids", ["merge-diff-arc-agi-task", "sparql-university"]))
        plans = adapter.plan(run_id, max_workers=workers, capacity=capacity)
        limits = {"max_workers": workers, "max_groups": 1, "policy": "terminal-waves",
                  "capacities": {executor: capacity}}
    return config, output, digest, adapter, plans, runtime, limits


async def run_prepared(prepared):
    from .transports import FileRelayTransport, RpcTransport
    config, output, digest, adapter, plans, runtime, limits = prepared
    store = EventStore(output, config["run_id"])
    token = CancellationToken()
    installed_signals = {}
    loop = asyncio.get_running_loop()
    try:
        store.append("configuration", configuration_sha256=digest,
                     benchmark=config["benchmark"], backend=config["backend"], transport=config["transport"])
        if config["transport"] == "tls-rpc":
            args = runtime["args"]
            transport = RpcTransport(args.url, args.credentials, output / "requests", timeout=args.rpc_timeout)
        else:
            transport = FileRelayTransport(runtime["relay_root"])
        backend = create_backend(config["benchmark"], config["backend"], config["transport"], runtime,
                                 transport=transport, event_store=store)
        coordinator = Coordinator({config["benchmark"]: backend}, event_store=store,
                                  drain_timeout=config.get("drain_timeout", 30), **limits)
        for signum in (signal.SIGINT, signal.SIGTERM):
            try:
                previous = signal.getsignal(signum)
                loop.add_signal_handler(signum, token.cancel, signal.Signals(signum).name)
                installed_signals[signum] = previous
            except (NotImplementedError, RuntimeError):
                pass
        await coordinator.run(plans, token)
        print(json.dumps(coordinator.summary, indent=2))
        if coordinator.draining_tasks:
            # Preserve the bounded coordinator's unresolved summary. The CLI
            # must keep journals/lifecycles alive until real workers finish,
            # rather than letting asyncio.run cancel away their cleanup proof.
            actual_workers = asyncio.gather(*coordinator.draining_tasks, return_exceptions=True)
            try:
                store.append("late_drain_waiting", quarantined=coordinator.quarantined)
            finally:
                try:
                    while not actual_workers.done():
                        try:
                            await asyncio.shield(actual_workers)
                        except asyncio.CancelledError:
                            token.cancel("cli_cancelled_during_late_drain")
                finally:
                    try:
                        await backend.close()
                    except Exception as error:
                        store.append("late_drain_close_failed", error_type=type(error).__name__)
                    else:
                        store.append("late_drain_finished", original_summary_remains_unresolved=True)
        return 0 if (coordinator.summary.get("drain_complete") and
                     coordinator.summary.get("cleanup_complete", True) and
                     not coordinator.summary.get("admission_stopped_reason")) else 2
    finally:
        for signum, previous in installed_signals.items():
            loop.remove_signal_handler(signum)
            signal.signal(signum, previous)
        store.close()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "native":
        from .runner import main as native_main
        return native_main(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["inspect", "validate", "plan", "run"],
                        help="inspect/validate/plan never contact services or execute task code")
    parser.add_argument("--benchmark", choices=COMBINATIONS, required=True)
    parser.add_argument("--backend", choices=["native-cuda", "harbor-pouch"], required=True)
    parser.add_argument("--transport", choices=["tls-rpc", "browser-filerelay"], required=True)
    parser.add_argument("--config", type=Path, help="versioned JSON configuration")
    args = parser.parse_args(argv)
    try:
        validate_combination(args.benchmark, args.backend, args.transport)
        if args.action == "inspect" and args.benchmark == "terminal-bench" and args.config is None:
            tasks = TerminalBenchAdapter(executor="unassigned-cpu-executor").enumerate()
            print(json.dumps(json_value(tasks), indent=2)); return 0
        if args.config is None:
            raise ValueError("this command requires --config")
        prepared = prepare(args.config, args.benchmark, args.backend, args.transport)
        if args.action == "run":
            if args.benchmark == "terminal-bench":
                from .execution_environment import prepare_worker_environment, validate_worker_environment
                if not os.environ.get("ROLLOUT_WORKER_ROOT"):
                    environment = prepare_worker_environment(prepared[5], prepared[2])
                    os.umask(0o077)
                    os.execve(sys.executable, [sys.executable, "-B", "-m", "rollout_manager", *argv], environment)
                validate_worker_environment(prepared[5], prepared[2])
            return asyncio.run(run_prepared(prepared))
        if args.action == "inspect":
            value = {"tasks": json_value(prepared[3].enumerate())}
            # Prepared native requests/candidate bodies are omitted from task inspection.
            for task in value["tasks"]:
                task["payload"].pop("native_case", None)
        elif args.action == "plan":
            value = {"configuration_sha256": prepared[2], "admission": json_value(prepared[6]),
                     "episodes": [{"identity": json_value(p.identity), "resources": json_value(p.task.resources),
                                   "budgets": json_value(p.task.budgets)} for p in prepared[4]]}
        else:
            value = {"valid": True, "offline": True, "configuration_sha256": prepared[2],
                     "tasks": len(prepared[3].enumerate()), "episodes": len(prepared[4]),
                     "live_qualification": "not established by offline validation"}
        print(json.dumps(value, indent=2)); return 0
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
