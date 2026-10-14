"""Once-only CPU workflow supervisor for automatic browser artifact handoff.

Initial setup supplies immutable local templates and pairs the browsers once.
Each phase waits for its policy lease, owns one fresh workflow, and publishes
original closed evidence. A pending claim after a crash is never relaunched.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from .browser_relay.artifacts import (
    KEY, checked_path, encoded, pin, read_json, wait_received, write_new,
)
from .sealed_artifacts import publish_sealed


def clean_closed(output, group_id, *, expected_lease=None):
    from .sealed import read_json as read_seal
    output = checked_path(output, directory=True)
    seal = read_seal(output / ("group-%d.seal.json" % group_id))
    closure = read_seal(output / "cpu-closure.json")
    if (seal.get("kind") != "terminalbench-sealed-group" or seal.get("group_id") != group_id
            or seal.get("closure") != closure or closure.get("coordinator_closed") is not True
            or closure.get("cpu_spool_terminal") is not True or closure.get("lifecycle_failures")
            or closure.get("quarantined") or closure.get("admission_stopped_reason") is not None
            or any(value != 0 for row in closure.get("reserved", {}).values() for value in row.values())):
        raise ValueError("original CPU closure is unresolved; no replay or lease advancement")
    if expected_lease is not None and read_seal(output / "lease.json") != expected_lease:
        raise ValueError("existing sealed group does not use the exact delivered original lease")
    return seal


def validate_config(value):
    if (value.get("schema_version") != 1 or value.get("expected_hostname") != socket.gethostname()
            or value.get("expected_uid") != os.getuid() or type(value.get("wait_seconds")) is not int
            or not 1 <= value["wait_seconds"] <= 86400):
        raise ValueError("supervisor requires inspected host/UID and bounded wait")
    output = checked_path(value["output_root"], directory=True)
    if output.stat().st_mode & 0o077:
        raise ValueError("supervisor output root must be private")
    relay = checked_path(value["relay_root"], directory=True)
    source = checked_path(value["source_root"], directory=True)
    phases = value.get("phases")
    if not isinstance(phases, list) or not 1 <= len(phases) <= 64:
        raise ValueError("supervisor needs 1..64 explicit training/evaluation phases")
    keys, outputs, versions = set(), set(), []
    for phase in phases:
        if (not isinstance(phase, dict) or set(phase) - {"lease_key", "seal_key", "config", "existing_output", "policy_version"}
                or not KEY.fullmatch(str(phase.get("lease_key"))) or not KEY.fullmatch(str(phase.get("seal_key")))
                or phase["lease_key"] in keys or phase["seal_key"] in keys
                or type(phase.get("policy_version")) is not int or phase["policy_version"] < 0):
            raise ValueError("supervisor phases require distinct immutable artifact keys")
        keys.update((phase["lease_key"], phase["seal_key"]))
        template = phase["config"]
        if template["source_root"] != str(source) or template["expected_hostname"] != value["expected_hostname"]:
            raise ValueError("phase source/host differs from supervisor")
        destination = Path(template["output_dir"])
        if (not destination.is_absolute() or destination.parent != output or destination in outputs
                or template["terminal_bench"]["relay_root"] != str(relay)
                or type(template.get("first_group_id")) is not int or template["first_group_id"] < 0
                or len(template.get("task_ids", [])) != 1):
            raise ValueError("each phase owns one new task-group output under supervisor root")
        outputs.add(destination)
        from .cpu_workflow import execution_shape
        execution_shape(template)
        versions.append(template["first_group_id"])
    if len(set(versions)) != len(versions):
        raise ValueError("semantic group ids cannot repeat within a supervisor")
    return output, relay, source


def phase_config(template, control, lease_path, relay, seal_key):
    value = copy.deepcopy(template)
    value["lease_capsule"] = str(lease_path)
    value["terminal_bench"]["model_guard_evidence"] = str(control / "model-guard.json")
    value["artifact_publication"] = {"relay_root": str(relay),
        "keys": {str(value["first_group_id"]): seal_key}}
    return value


def qualified_guard(config, lease, timeout):
    from . import admission, serving_contracts as controller
    from .sealed import read_json as read_seal
    capsule = read_seal(config["public_capsule"], 1024**2)
    runtime = config["terminal_bench"]
    required = max(sum(capsule["tasks"][task]["reset_spec"]["original_task_budgets"].values())
                   for task in config["task_ids"])
    deadline = time.monotonic() + timeout
    while True:
        admission.check_empty_spool(Path(runtime["relay_root"]))
        try:
            ready = admission.check_readiness(runtime["relay_root"], runtime["model"], 1, required,
                required_generation_seconds=360,
                serving_profile=runtime["serving_profile"], expected_policy=lease["policy"])
            break
        except (ValueError, OSError, RuntimeError) as error:
            if time.monotonic() >= deadline:
                raise TimeoutError("new policy readiness did not qualify; no episode admitted") from error
            time.sleep(.2)
    key = "native_controller" if runtime["serving_profile"] == "native-pytorch-v1" else "training_controller"
    guard = {"schema_version": 2, "serving_profile": runtime["serving_profile"], "model": runtime["model"],
        key: ready["gpu"][key], "cpu_relay_boot_id": ready["cpu"]["boot_id"],
        "gpu_relay_boot_id": ready["gpu"]["boot_id"],
        "source": "actual authenticated browser readiness; never an operator-supplied proof"}
    controller.guard_gate(guard, ready, required, time.time(), serving_profile=runtime["serving_profile"],
                         model=runtime["model"], expected_policy=lease["policy"])
    return guard


def owned_workflow(config_path, source, control):
    command = [sys.executable, "-B", str(source / "rollout_manager/terminal_bench/cpu_workflow.py"),
               "run", "--config", str(config_path)]
    environment = dict(os.environ, PYTHONPATH=str(source), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    with os.fdopen(os.open(control / "worker.log", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), "wb") as log:
        process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                   cwd=control, start_new_session=True)
        from .browser_relay.relay_server import process_start_ticks
        write_new(control / "worker.json", encoded({"pid": process.pid,
            "pid_start_ticks": process_start_ticks(process.pid),
            "cmdline_sha256": hashlib.sha256(Path("/proc/%d/cmdline" % process.pid).read_bytes()).hexdigest(),
            "started_unix": time.time()}))
        return process.wait()


def run_supervisor(value, *, runner=owned_workflow, guard_factory=qualified_guard):
    output, relay, source = validate_config(value)
    state = output / "supervisor"
    state.mkdir(mode=0o700, exist_ok=True)
    checked_path(state, directory=True)
    descriptor = os.open(state / ".lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        snapshot = state / "config.json"
        if snapshot.exists():
            if read_json(snapshot) != value:
                raise ValueError("supervisor immutable setup changed")
        else:
            write_new(snapshot, encoded(value))
        for phase in value["phases"]:
            control = state / phase["lease_key"]
            control.mkdir(mode=0o700, exist_ok=True)
            manifest, blob = wait_received(relay, "policy_lease", phase["lease_key"], value["wait_seconds"])
            lease = read_json(blob)
            if lease.get("kind") != "terminalbench-policy-lease" or lease.get("schema_version") != 1:
                raise ValueError("automatic lease has unsupported schema or kind")
            from rollout_manager.training import PolicyRef
            from rollout_manager.generation_training import TrainingGenerationContract
            policy = PolicyRef.from_dict(lease["policy"])
            TrainingGenerationContract(policy, lease["capabilities"])
            if policy.version != phase["policy_version"]:
                raise ValueError("automatic lease policy version differs from declared phase")
            lease_path = control / "lease.json"
            if lease_path.exists():
                if pin(lease_path) != pin(blob):
                    raise ValueError("delivered phase lease changed")
            else:
                write_new(lease_path, blob.read_bytes())
            config = phase_config(phase["config"], control, lease_path, relay, phase["seal_key"])
            from .cpu_workflow import execution_shape
            _, size = execution_shape(config)
            if (lease.get("group_size", 2) != size or lease.get("purpose", "training") != config.get("purpose", "training")):
                raise ValueError("delivered lease differs from declared training/evaluation purpose")
            group_id = config["first_group_id"]
            existing = phase.get("existing_output")
            destination = Path(existing or config["output_dir"])
            finished = control / "finished.json"
            if finished.exists():
                previous = read_json(finished)
                if previous["lease_artifact"] != manifest:
                    raise ValueError("completed phase lease changed")
                clean_closed(destination, group_id, expected_lease=lease)
                publish_sealed(relay, destination, group_id, phase["seal_key"])
                continue
            if existing:
                clean_closed(destination, group_id, expected_lease=lease)
            else:
                claim = control / "claim.json"
                if claim.exists() or destination.exists():
                    raise ValueError("phase already claimed or output exists; unresolved work is never relaunched")
                guard = guard_factory(config, lease, value["wait_seconds"])
                write_new(Path(config["terminal_bench"]["model_guard_evidence"]), encoded(guard))
                config_path = control / "worker-config.json"
                write_new(config_path, encoded(config))
                write_new(claim, encoded({"schema_version": 1, "lease_artifact": manifest,
                    "config_sha256": hashlib.sha256(encoded(config)).hexdigest(), "claimed_unix": time.time()}))
                code = runner(config_path, source, control)
                if code != 0:
                    # A closed exclusion remains evidence, never reward zero.
                    if (destination / ("group-%d.seal.json" % group_id)).exists():
                        publish_sealed(relay, destination, group_id, phase["seal_key"])
                    raise RuntimeError("CPU phase failed or was excluded; admission stops without replay")
                clean_closed(destination, group_id, expected_lease=lease)
            artifact = publish_sealed(relay, destination, group_id, phase["seal_key"])
            write_new(finished, encoded({"schema_version": 1, "lease_artifact": manifest,
                "sealed_artifact": artifact, "finished_unix": time.time(), "existing_evidence_reused": bool(existing)}))
            print(json.dumps({"event": "automatic_cpu_phase_published", "key": phase["seal_key"],
                              "group_id": group_id, "artifact_id": artifact["id"]}), flush=True)
        return 0
    finally:
        os.close(descriptor)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    value = read_json(args.config)
    from .cpu_workflow import configure_source
    configure_source(value)
    return run_supervisor(value)


if __name__ == "__main__":
    raise SystemExit(main())
