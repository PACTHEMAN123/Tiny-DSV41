"""Recover an unconsumed original sealed group after a pre-optimizer failure.

Old inference evidence remains historical. A new serving controller never claims
that it sampled the old request identities, and CPU episodes are never rerun.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path

from .controller import read_json
from ..backends.native_qwen_loopback import LocalGPURelayDeliveryProbe, read_private, strict_json, write_new, linux_process_identity
from ..training import PolicyRef


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _file_pin(path):
    from .sealed import file_sha256
    return file_sha256(path)


def _exited(owner):
    pid = owner.get("pid")
    ticks = owner.get("pid_start_ticks", owner.get("start_ticks"))
    digest = owner.get("cmdline_sha256")
    _require(type(pid) is int and pid > 0 and type(ticks) is int and ticks > 0
             and isinstance(digest, str) and len(digest) == 64, "original owner identity is incomplete")
    path = Path("/proc") / str(pid)
    try:
        raw = (path / "stat").read_text()
    except FileNotFoundError:
        return
    fields = raw[raw.rfind(")") + 2:].split()
    _require(int(fields[19]) == ticks and fields[0] == "Z", "original owned process is still active or PID was reused")
    command = (path / "cmdline").read_bytes()
    _require(not command or hashlib.sha256(command).hexdigest() == digest, "original process command differs")


class OriginalGroupConsumer:
    """One durable consumer claim plus the original kernel lock until close."""
    def __init__(self, root, lock, artifact, consumer_output, seal_sha256):
        self.root, self.lock = root, lock
        self.artifact, self.output = artifact, Path(consumer_output).absolute()
        _require(len(seal_sha256) == 64 and all(c in "0123456789abcdef" for c in seal_sha256),
                 "original group seal SHA256 is required")
        self.record = {"kind": "exclusive-original-pre-update-consumer", "policy_version": 0,
            "policy_digest": artifact["digest"], "policy_lineage": artifact["lineage"],
            "group_id": 0, "original_group_seal_sha256": seal_sha256,
            "consumer_output": str(self.output), "owner": linux_process_identity(),
            "crash_or_failed_continuation_allows_automatic_reuse": False}
        write_new(root / ".pre-update-consumer.json", self.record)

    def committed(self, checkpoint, published_policy):
        path = Path(checkpoint).absolute()
        _require(self.lock is not None and path.parent == self.output
            and path.name == "step-00000001.pt" and published_policy["version"] == 1
            and published_policy["lineage"] == self.artifact["lineage"],
            "original group commit requires its real first checkpoint and same-lineage publication")
        write_new(self.root / ".pre-update-consumer-committed.json", {
            "kind": "original-pre-update-group-consumed", "consumer_claim": _file_pin(
                self.root / ".pre-update-consumer.json"), "checkpoint_path": str(path),
            "checkpoint": _file_pin(path), "published_policy": published_policy,
            "optimizer_step_applied": True, "episode_generation_replayed": False})

    def close(self):
        if self.lock is not None:
            fcntl.flock(self.lock, fcntl.LOCK_UN)
            self.lock.close()
            self.lock = None


def original_initial_policy(training_root, workflow_receipt, *, expected_digest,
                            expected_actor_exit_sha256, consumer_output, expected_group_seal_sha256):
    """Prove no previous optimizer/update happened, then recover exact artifact bytes."""
    import torch
    root = Path(training_root).absolute()
    _require(root.is_dir() and not root.is_symlink(), "original training output is required")
    for path in (root, *root.parents):
        _require(not path.is_symlink(), "original recovery root must be real")
    owner = read_json(Path(workflow_receipt))
    _exited(owner)
    lock = (root / ".run.lock").open("r+")
    consumer = None
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stages = [strict_json(line.encode()) for line in (root / "learner-stage-timings.jsonl").read_text().splitlines()]
        _require([row["stage"] for row in stages] == ["model_startup", "initial_policy_publish",
            "manager_startup", "wait_eligible_groups", "manager_close"],
            "recovery requires the exact original pre-optimizer failure boundary")
        _require(all(row["pid"] == owner["pid"] for row in stages)
            and all(row["status"] == "completed" for row in stages if row["stage"] != "wait_eligible_groups")
            and stages[-2]["status"] == "failed" and stages[-2].get("error_type") == "ModuleNotFoundError"
            and stages[-2]["details"] == {"step": 0, "required_groups": 1},
            "original failure is not a closed pre-update import failure")
        _require(not any(root.glob("step-*.pt")) and not any(root.glob("import-group-*.json"))
            and not any(root.glob("activated-policy-*.json"))
            and not (root / "updated-policy-episode.json").exists()
            and not (root / "metrics.jsonl").exists(), "original group may already have been consumed")
        shutdown_path = root / "native-serving/vllm-actor/actor-exit.json"
        _require(_file_pin(shutdown_path)["sha256"] == expected_actor_exit_sha256,
                 "original actor shutdown bytes differ")
        shutdown = read_json(shutdown_path)
        _require(shutdown.get("returncode") == 0 and shutdown.get("drain_complete") is True
                 and shutdown.get("forced_termination", False) is False,
                 "original actor lacks an unforced complete shutdown")
        _exited(shutdown["actor_process"])
        startup_path = root / "native-serving/owned-controller-startup.json"
        startup = read_json(startup_path)
        _require(startup.get("model_pid") == owner["pid"]
            and startup.get("model_pid_start_ticks") == owner["pid_start_ticks"]
            and startup.get("model_cmdline_sha256") == owner["cmdline_sha256"]
            and startup["vllm_engine"]["actor_process"] == shutdown["actor_process"]
            and startup["vllm_engine"].get("workers"), "original model/actor ownership differs")
        for worker in startup["vllm_engine"]["workers"]:
            _exited(worker)
        lease = read_json(root / "cpu-policy-lease.json")
        policy = PolicyRef.from_dict(lease["policy"])
        _require(policy.version == 0 and policy.digest == expected_digest, "original lease policy differs")
        files = list((root / "policies").iterdir())
        _require(len(files) == 1 and files[0].name == "policy-00000000-" + expected_digest + ".pt",
                 "original output does not contain exactly its initial policy artifact")
        artifact_path = files[0]
        _require(_file_pin(artifact_path)["sha256"] == expected_digest, "original artifact bytes differ")
        stored = torch.load(artifact_path, map_location="cpu", weights_only=True)
        _require(_file_pin(artifact_path)["sha256"] == expected_digest, "original artifact changed while loading")
        _require(set(stored) == {"version", "lineage", "identity", "sampling", "trainable", "frozen"}
            and stored["version"] == 0 and stored["lineage"] == policy.lineage
            and stored["sampling"] == dict(policy.sampling)
            and set(stored["trainable"]) == {"backend.lm_head.weight"} and stored["frozen"] == {},
            "original artifact is not the exact qualified pretrained head policy")
        artifact = {key: stored[key] for key in ("version", "lineage", "identity", "sampling")}
        artifact.update(path=str(artifact_path), digest=expected_digest)
        from ..benchmark_training import artifact_policy
        _require(artifact_policy(artifact, identities=dict(policy.identities)) == policy,
                 "original artifact identity differs from the sealed behavior policy")
        qualification = read_json(root / "mounted-vllm-qualification.json")
        _require(qualification.get("parameters_unchanged") is True
            and qualification.get("policy") == policy.to_dict(), "original diagnostic mutated or changed policy")
        proof = {"kind": "original-pre-update-policy-recovery", "original_training_root": str(root),
            "workflow_receipt": _file_pin(Path(workflow_receipt)),
            "stage_timings": _file_pin(root / "learner-stage-timings.jsonl"),
            "actor_shutdown": _file_pin(shutdown_path), "original_artifact": _file_pin(artifact_path),
            "original_controller_startup": _file_pin(startup_path),
            "policy": policy.to_dict(), "optimizer_step_previously_applied": False,
            "original_group_consumed": False, "episode_generation_replayed": False}
        consumer = OriginalGroupConsumer(root, lock, artifact, consumer_output, expected_group_seal_sha256)
        proof["exclusive_consumer_claim"] = _file_pin(root / ".pre-update-consumer.json")
        return artifact, lease, proof, consumer
    finally:
        if consumer is None:
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()


class HistoricalGroupDelivery:
    """Original closed sampler receipts plus the still-owned original relay ACKs."""
    def __init__(self, original_training_root, relay_root, relay_port, policy, model,
                 expected_relay_startup_sha256, expected_relay_cmdline_sha256):
        self.output = Path(original_training_root) / "native-serving"
        self.policy = policy
        self.delivery = LocalGPURelayDeliveryProbe(relay_root, relay_port=relay_port, expected_model=model)
        self.startup_sha256 = expected_relay_startup_sha256
        self.command_sha256 = expected_relay_cmdline_sha256

    def _bind_original_relay(self):
        """The historical actor is closed; authenticate the still-owned relay status."""
        root = self.delivery.root
        startup_path = root / "verified-startup.json"
        _require(_file_pin(startup_path)["sha256"] == self.startup_sha256,
                 "historical relay startup bytes differ")
        startup = read_json(startup_path)
        _require(startup["role"] == "gpu" and startup["port"] == self.delivery.port
            and startup["state_root"] == str(root), "historical relay configuration differs")
        process = Path("/proc") / str(startup["relay_pid"])
        raw = (process / "stat").read_text()
        fields = raw[raw.rfind(")") + 2:].split()
        command = (process / "cmdline").read_bytes()
        _require(process.stat().st_uid == os.geteuid() and fields[0] not in {"Z", "X"}
            and int(fields[19]) == startup["relay_pid_start_ticks"]
            and hashlib.sha256(command).hexdigest() == self.command_sha256,
            "historical relay owner changed or exited")
        args = command.rstrip(b"\0").decode().split("\0")
        _require("rollout_manager.terminal_bench.browser_relay.relay_server" in args
            and args[args.index("--root") + 1] == str(root)
            and args[args.index("--role") + 1] == "gpu"
            and args[args.index("--port") + 1] == str(self.delivery.port),
            "historical relay command differs")
        _require(os.readlink(process / "ns/net") == os.readlink("/proc/self/ns/net"),
                 "historical relay network namespace differs")
        sockets = set()
        for descriptor in (process / "fd").iterdir():
            try:
                target = os.readlink(descriptor)
            except FileNotFoundError:
                continue  # A completed browser poll can close its own descriptor.
            if target.startswith("socket:["):
                sockets.add(target[8:-1])
        listeners = {line.split()[9] for name in ("tcp", "tcp6")
            for line in (process / "net" / name).read_text().splitlines()[1:]
            if line.split()[3] == "0A" and int(line.split()[1].split(":")[-1], 16) == self.delivery.port}
        _require(bool(sockets & listeners), "historical relay does not own its listener")
        status = self.delivery._get("/api/status")
        _require(status.get("role") == "gpu" and status.get("boot_id") == startup["boot_id"],
                 "authenticated historical relay boot differs")
        again = (process / "stat").read_text().rsplit(")", 1)[1].split()
        _require(again[0] not in {"Z", "X"} and int(again[19]) == startup["relay_pid_start_ticks"],
                 "historical relay owner changed during binding")
        self.delivery.boot_id = startup["boot_id"]

    def validate(self, group):
        _require(group.policy == self.policy, "historical group differs from its original sampled policy")
        identities = [turn.request_id for sample in group.samples for turn in sample.turns]
        _require(len(identities) == len(set(identities)), "historical group repeats a request identity")
        records = []
        for sample in group.samples:
            for turn in sample.turns:
                path = self.output / (turn.request_id + ".training.json")
                observed = strict_json(read_private(path, 1024**2))
                expected = turn.to_dict()
                observed.pop("turn_id", None)
                expected.pop("turn_id", None)
                _require(observed == expected, "historical serving evidence differs from original CPU tokens/probabilities")
                state_path = self.output / (turn.request_id + ".http-state.json")
                state = strict_json(read_private(state_path, 1024**2))
                _require(state.get("id") == turn.request_id and state.get("http_response_written") is True
                    and state.get("sampler_drained") is True and state.get("automatic_replay") is False,
                    "original generation completion is unproven")
                records.append({"request_id": turn.request_id, "training_record": _file_pin(path),
                                "original_http_state": _file_pin(state_path)})
        self._bind_original_relay()
        proof = self.delivery.wait(identities)
        return {**proof, "kind": "original-historical-sampled-group-delivery", "policy": self.policy.to_dict(),
            "original_serving_root": str(self.output), "original_serving_records": records,
            "new_actor_claims_original_generation": False, "episode_generation_replayed": False}
