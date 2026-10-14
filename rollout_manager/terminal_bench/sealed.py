"""Manual CPU/A handoff of original Terminal-Bench evidence, never remote RPC.

CPU seals cover owned episode artifacts after actual Coordinator/backend close.
The A-side caller must additionally call its delivery controller's
``validate_group_delivery(group)`` and pause it before invoking the optimizer.
This module proves CPU evidence only; empty CPU spools do not prove GPU drain.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat

from rollout_manager.benchmark_training import GroupAssembler, GroupRegistration, RewardPolicy
from rollout_manager.contracts import EpisodeIdentity, EpisodePlan, EpisodeResult, ResourceRequirements, TaskSpec
from rollout_manager.generation_training import TrainingGenerationContract, evidence_digest
from rollout_manager.training import PolicyRef, TrainingGroup, TrainingTrajectory, TurnTrace, token_digest

TASKS = ("merge-diff-arc-agi-task", "sparql-university")
TASK_PIN = "184faa0638cfd08499f51d53a7bf64a492daf1c14ec4f6aa9d889483678e228e"
PAYLOAD_PIN = "6abd7da26fc66958cd36274edebe7720da9eb24a62ceeb0e6c2ffaed8c9521f5"
HASH = re.compile(r"[a-f0-9]{64}\Z")
MAX_JSON = 512 * 1024**2


def read_json(path, limit=MAX_JSON):
    from rollout_manager.terminal_bench.controller import read_json as checked_read
    return checked_read(Path(path), limit)


def write_new(path, value):
    from rollout_manager.terminal_bench.controller import write_new as checked_write
    checked_write(Path(path), value)


def file_sha256(path):
    digest = hashlib.sha256()
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("evidence must be a regular file")
        for data in iter(lambda: stream.read(1024**2), b""):
            digest.update(data)
        after = os.fstat(stream.fileno())
    signature = lambda x: (x.st_dev, x.st_ino, x.st_size, x.st_mtime_ns, x.st_ctime_ns)
    if signature(before) != signature(after):
        raise ValueError("evidence changed while hashing")
    return {"bytes": before.st_size, "sha256": digest.hexdigest()}


def initial_payload(instruction, model, temperature):
    """Exact public turn-one payload of the inspected FileRelayAgent (30s shell)."""
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("public original task instruction is required")
    return {"model": model, "messages": [{"role": "system", "content": (
        "Solve the task in its Linux environment using the shell tool. "
        "Commands use the environment's working directory; filesystem changes persist, shell state does not. "
        "Each command has a 30-second deadline and bounded output. "
        "Never assume a failed or timed-out command was not executed. "
        "Finish with a brief final answer and no tool calls.")}, {"role": "user", "content": instruction}],
        "tools": [{"type": "function", "function": {"name": "shell",
            "description": "Run a shell command in the task environment. Deadline 30 seconds; bounded output.",
            "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}},
                           "required": ["cmd"], "additionalProperties": False}}}],
        "tool_choice": "auto", "parallel_tool_calls": False,
        "chat_template_kwargs": {"enable_thinking": False}, "stream": False,
        "temperature": temperature, "max_tokens": 2048}


def build_lease(public_capsule, policy, capabilities, tokenizer, *, group_size=2):
    """A: tokenize only CPU-exported original public instructions before admission."""
    capsule = read_json(public_capsule, 1024**2) if isinstance(public_capsule, (str, Path)) else public_capsule
    if (capsule.get("schema_version") != 1 or capsule.get("kind") != "terminalbench-public-payloads"
            or capsule.get("tasks") is None or set(capsule["tasks"]) != set(TASKS)
            or capsule.get("task_provenance_sha256") != TASK_PIN or capsule.get("payload_sha256") != PAYLOAD_PIN
            or not HASH.fullmatch(str(capsule.get("agent_sha256")))):
        raise ValueError("unsupported original public capsule")
    if type(group_size) is not int or not 1 <= group_size <= 64:
        raise ValueError("lease group size must be 1..64; singleton leases are evaluation only")
    policy = policy if isinstance(policy, PolicyRef) else PolicyRef.from_dict(policy)
    contract = TrainingGenerationContract(policy, capabilities)
    tasks = {}
    for task_id in TASKS:
        row = capsule["tasks"][task_id]
        payload = row["payload"]
        contract.check_payload(payload)
        if (payload != initial_payload(payload["messages"][1]["content"], capsule["served_model"],
                                      policy.sampling["temperature"])
                or row["payload_sha256"] != evidence_digest(payload)
                or row["instruction_sha256"] != hashlib.sha256(payload["messages"][1]["content"].encode()).hexdigest()
                or not HASH.fullmatch(str(row.get("reset_sha256")))
                or evidence_digest(row["reset_spec"]) != row["reset_sha256"]):
            raise ValueError("public capsule payload or task provenance differs")
        # The relay serializes mappings with sorted keys. Checkpoint templates
        # can render tool mappings in insertion order, so lease that exact wire
        # representation before any episode is admitted.
        wire_payload = json.loads(json.dumps(payload, sort_keys=True, allow_nan=False))
        ids = tokenizer.encode_conversation(wire_payload["messages"], tools=wire_payload["tools"],
                                            chat_template_kwargs=wire_payload["chat_template_kwargs"])
        tasks[task_id] = {key: row[key] for key in ("instruction_sha256", "payload_sha256", "reset_sha256")}
        tasks[task_id].update(initial_context_sha256=token_digest(ids), context_tokens=len(ids))
    lease = {"schema_version": 1, "kind": "terminalbench-policy-lease", "policy": policy.to_dict(),
        "capabilities": capabilities, "served_model": capsule["served_model"],
        "public_capsule_sha256": evidence_digest(capsule), "agent_sha256": capsule["agent_sha256"],
        "tasks": tasks, "cpu_cleanup_is_gpu_drain": False}
    if group_size != 2:
        lease["group_size"] = group_size
        lease["purpose"] = "policy_evaluation" if group_size == 1 else "training"
    return lease


@dataclass(frozen=True)
class PolicyEvaluation:
    """One verified episode; deliberately cannot masquerade as a GRPO group."""
    group_id: int
    task_id: str
    policy: PolicyRef
    samples: tuple[TrainingTrajectory, ...]

    def to_dict(self):
        return {"schema_version": 1, "kind": "terminalbench-policy-evaluation",
                "group_id": self.group_id, "task_id": self.task_id,
                "policy": self.policy.to_dict(), "samples": [sample.to_dict() for sample in self.samples]}


def evaluation_receipt(value, result):
    """Reconstruct a singleton receipt using the same authoritative reward checks."""
    plans = [decode_plan(plan) for plan in value["plans"]]
    policy = PolicyRef.from_dict(value["policy"])
    if (type(value["group_id"]) is not int or value["group_id"] < 0 or len(plans) != 1
            or plans[0].identity.sample_index != 0 or result.identity != plans[0].identity
            or result.source_sha256 != plans[0].task.source_sha256):
        raise ValueError("evaluation requires one exactly identified fresh episode")
    reward = RewardPolicy("terminal-bench", "terminal-validated-verifier-v1").reward(result)
    turns = tuple(TurnTrace.from_dict(turn) for turn in result.native_outcome["training_turns"])
    if any(turn.policy != policy for turn in turns):
        raise ValueError("evaluation turn differs from its activated policy lease")
    states = result.request_state.get("requests", {})
    if any(states.get(turn.request_id, {}).get("state") != "response_received"
           or states.get(turn.request_id, {}).get("training_evidence") != "validated" for turn in turns):
        raise ValueError("evaluation delivery or exact training evidence is unproven")
    sample = TrainingTrajectory(value["group_id"], plans[0].task.task_id, value["initial_context_sha256"],
        value["reset_sha256"], 0, turns, reward,
        evidence={"episode_identity": asdict(result.identity), "native_artifacts": result.native_artifacts,
                  "native_outcome": dict(result.native_outcome), "reward_mapping": "terminal-validated-verifier-v1"},
        sample_id=result.identity.trial_id or "%s:%d:0" % (result.identity.run_id, value["group_id"]),
        raw_reward=reward, sampling_seed=result.identity.sampling_seed)
    return PolicyEvaluation(value["group_id"], plans[0].task.task_id, policy, (sample,))


def decode_plan(value):
    task = value["task"]
    return EpisodePlan(EpisodeIdentity(**value["identity"]),
        TaskSpec(**{**task, "resources": ResourceRequirements(**task["resources"])}), value.get("payload"))


def decode_result(value):
    value = {key: item for key, item in value.items() if key != "schema_version"}
    return EpisodeResult(**{**value, "identity": EpisodeIdentity(**value["identity"])})


def registration(value):
    return GroupRegistration(value["group_id"], [decode_plan(p) for p in value["plans"]],
        PolicyRef.from_dict(value["policy"]), value["initial_context_sha256"], value["reset_sha256"],
        RewardPolicy("terminal-bench", "terminal-validated-verifier-v1"))


def sealed_inventory(root):
    root = Path(root)
    inventory = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("linked evidence member")
        if path.is_file():
            if not path.name.endswith(".seal.json"):
                inventory[path.relative_to(root).as_posix()] = file_sha256(path)
        elif not path.is_dir():
            raise ValueError("nonregular evidence member")
    if not inventory or len(inventory) > 20000 or sum(row["bytes"] for row in inventory.values()) > 4 * 1024**3:
        raise ValueError("sealed evidence exceeds explicit importer capacity")
    return inventory


def write_seals(root, groups, closure):
    """CPU: called once after streams, jobs, event store and spool are terminal."""
    root = Path(root)
    if closure.get("coordinator_closed") is not True or closure.get("cpu_spool_terminal") is not True:
        raise ValueError("cannot seal unresolved CPU work")
    inventory = sealed_inventory(root)
    paths = []
    for row in groups:
        path = root / ("group-%d.seal.json" % row["group_id"])
        write_new(path, {"schema_version": 1, "kind": "terminalbench-sealed-group", "group_id": row["group_id"],
            "cpu_output_root": str(root), "registration_file": row["registration_file"],
            "episode_files": row["episode_files"], "receipt_file": row["receipt_file"],
            "receipt_kind": row["receipt_kind"], "closure": closure, "inventory": inventory,
            "inventory_sha256": evidence_digest(inventory), "gpu_delivery_proven": False})
        paths.append(path)
    return paths


def _member(root, relative, inventory):
    value = PurePosixPath(relative)
    if not isinstance(relative, str) or value.is_absolute() or ".." in value.parts or value.as_posix() != relative:
        raise ValueError("invalid sealed relative path")
    if relative not in inventory:
        raise ValueError("evidence member is absent from sealed inventory")
    path = root.joinpath(*value.parts)
    if any(item.is_symlink() for item in (path, *path.parents)):
        raise ValueError("linked sealed path")
    return path


def _original_path(root, cpu_root, absolute, inventory):
    value = PurePosixPath(absolute)
    try:
        relative = value.relative_to(PurePosixPath(cpu_root)).as_posix()
    except (ValueError, TypeError):
        raise ValueError("original artifact is outside the owned CPU output") from None
    return _member(root, relative, inventory)


def _check_episode(root, cpu_root, inventory, plan, result, contract, capsule):
    from rollout_manager.terminal_bench.controller import (
        classify_trial, TRAINING_PROFILES, NATIVE_PROFILE, validate_training_controller)
    trial = _original_path(root, cpu_root, result.native_artifacts["result"], inventory).parent
    for name in ("result.json", "pouch-environment.json", "offline-bootstrap.json", "original-verifier-terminal.json",
                 "agent/file-relay/agent-result.json", "verifier/ctrf.json", "verifier/reward.txt",
                 "verifier/test-stdout.txt", "agent/file-relay/trajectory.json"):
        _member(root, (trial / name).relative_to(root).as_posix(), inventory)
    episode = _original_path(root, cpu_root, result.native_artifacts["episode_dir"] + "/config.json", inventory).parent
    job = read_json(episode / "config.json")
    cpu_config = read_json(_member(root, "config.json", inventory))
    max_turns = cpu_config.get("terminal_bench", {}).get("agent_max_turns", 128)
    if type(max_turns) is not int or not 1 <= max_turns <= 1024:
        raise ValueError("sealed agent_max_turns must be an integer in 1..1024")
    agents = job.get("agents", [])
    if len(agents) != 1 or agents[0].get("kwargs", {}).get("max_turns", 128) != max_turns:
        raise ValueError("original Harbor agent turn budget differs from sealed CPU configuration")
    configuration_sha256 = hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()
    if (configuration_sha256 != result.configuration_sha256 or job.get("n_attempts") != 1
            or job.get("n_concurrent_trials") != 1 or job.get("timeout_multiplier") != 1
            or job.get("retry", {}).get("max_retries") != 0 or job.get("verifier", {}).get("disable") is not False
            or len(job.get("tasks", [])) != 1 or job["tasks"][0]["path"] != plan.task.payload["task_path"]
            or job.get("environment", {}).get("delete") is not True
            or job["environment"].get("kwargs", {}).get("allow_public_network") is not False):
        raise ValueError("original Harbor attempt/verifier/cleanup configuration differs")
    admission_id = hashlib.sha256(json.dumps(asdict(plan.identity), sort_keys=True).encode()).hexdigest()
    admission_path = episode.parent / ("admission-" + admission_id + ".json")
    _member(root, admission_path.relative_to(root).as_posix(), inventory)
    admitted = read_json(admission_path)
    required = sum(capsule["tasks"][plan.task.task_id]["reset_spec"]["original_task_budgets"].values())
    ready = admitted["readiness"]
    paired = ready["paired_unix"]
    profile = contract.capabilities.get("serving_profile", NATIVE_PROFILE)
    controller_key = "native_controller" if profile == NATIVE_PROFILE else "training_controller"
    if (admitted.get("required_seconds") != required or admitted.get("episode_ids") != [admission_id]
            or profile not in TRAINING_PROFILES
            or admitted.get("model_guard", {}).get("serving_profile") != profile
            or ready.get("protocol") != 2
            or any(ready[role].get("serving_profile") != profile or ready[role].get("max_inflight") != 1
                   or ready[role].get("model") != result.model_identity or ready[role].get("generation_seconds", 0) < 360
                   or ready[role].get("expires_unix", 0) - paired < required for role in ("cpu", "gpu"))):
        raise ValueError("original native admission profile/capacity/service lifetime differs")
    validate_training_controller(ready["gpu"][controller_key], result.model_identity, serving_profile=profile, now=paired,
        required_seconds=required, expected_policy=contract.policy, require_fresh=False)
    row = classify_trial(trial)
    if (row["status"] != "evaluated" or row["cleanup_verified"] is not True
            or row["task_id"] != plan.task.task_id or row["reward"] != contract_reward(result)
            or row["request_ids"] != result.native_outcome["request_ids"]
            or row["container_id"] != result.native_outcome["container_id"]):
        raise ValueError("original verifier/cleanup classification differs from episode result")
    cleanup = read_json(trial / "pouch-environment.json")
    if (cleanup.get("provider") != "pouch" or cleanup.get("started") is not True
            or cleanup.get("stopped") is not True or cleanup.get("deleted") is not True
            or cleanup.get("cleanup_verified") is not True or cleanup.get("network_mode") != "none"
            or cleanup.get("public_network_explicitly_allowed") is not False
            or cleanup.get("image_id") != plan.task.payload["image_id"]
            or cleanup.get("cpu_limit") != plan.task.resources.cpu
            or cleanup.get("memory_limit_mb") * 1024**2 != plan.task.resources.memory_bytes):
        raise ValueError("fresh owned task environment lacks original isolation or cleanup proof")
    bootstrap = read_json(trial / "offline-bootstrap.json")
    prerequisites_path = trial / "agent-prerequisites.json"
    _member(root, prerequisites_path.relative_to(root).as_posix(), inventory)
    prerequisites = read_json(prerequisites_path)
    terminal = read_json(trial / "original-verifier-terminal.json")
    if (bootstrap.get("manifest_sha256") != PAYLOAD_PIN or bootstrap.get("task_id") != plan.task.task_id
            or prerequisites.get("verified") is not True or prerequisites.get("task_id") != plan.task.task_id
            or type(terminal.get("return_code")) is not int or not HASH.fullmatch(str(terminal.get("command_sha256")))):
        raise ValueError("original bootstrap/prerequisite/verifier terminal identity differs")
    reward_bytes = (trial / "verifier/reward.txt").read_bytes()
    if len(reward_bytes) > 64 or float(reward_bytes) != row["reward"]:
        raise ValueError("original reward artifact differs from authoritative Harbor/CTRF result")
    agent = read_json(trial / "agent/file-relay/agent-result.json")
    expected_budgets = {"generation_seconds": 360, "command_seconds": 30, "max_turns": max_turns,
                        "max_tokens": 2048, "max_tool_output_bytes": 16384,
                        "temperature": contract.policy.sampling["temperature"]}
    if (agent.get("policy") != expected_budgets or agent.get("generation_retries") != 0
            or agent.get("plugin_sha256") != capsule["agent_sha256"]):
        raise ValueError("original model/tool budgets or retry semantics changed")
    raw_turns = result.native_outcome["training_turns"]
    states = result.request_state["requests"]
    if len(raw_turns) != agent["turns"] or agent["request_ids"] != [t["request_id"] for t in raw_turns]:
        raise ValueError("actual agent turn order differs from recorded training turns")
    for turn in raw_turns:
        identity = turn["request_id"]
        base = trial / "agent/file-relay"
        for suffix in ("request", "response", "state", "training"):
            _member(root, (base / (identity + "." + suffix + ".json")).relative_to(root).as_posix(), inventory)
        request, response = read_json(base / (identity + ".request.json")), read_json(base / (identity + ".response.json"))
        state, saved = read_json(base / (identity + ".state.json")), read_json(base / (identity + ".training.json"))
        if (request.get("id") != identity or request.get("operation") != "generate"
                or response.get("id") != identity or response.get("ok") is not True
                or state.get("id") != identity or state.get("state") != "response_received"
                or state.get("published") is not True or state.get("replayed") is not False
                or saved != turn):
            raise ValueError("durable original request/response/training records differ")
        verified = contract.validate(request["payload"], response["response"], identity, turn["turn_id"])
        if verified.to_dict() != turn:
            raise ValueError("raw serving evidence differs from exact recorded tokens/probabilities")
        marked = states[identity]
        if ({key: value for key, value in marked.items() if key not in
             {"training_evidence", "training_record", "training_record_sha256"}} != state
                or marked.get("training_evidence") != "validated"
                or marked.get("training_record_sha256") != evidence_digest(saved)
                or _original_path(root, cpu_root, marked["training_record"], inventory) != base / (identity + ".training.json")):
            raise ValueError("episode delivery claim is not bound to both durable originals")
        if turn["turn_id"] == 1:
            payload = request["payload"]
            source = hashlib.sha256(payload["messages"][1]["content"].encode()).hexdigest()
            if source != plan.task.source_sha256 or payload != initial_payload(
                    payload["messages"][1]["content"], result.model_identity, contract.policy.sampling["temperature"]):
                raise ValueError("first actual request is not the original public task instruction")
    return cleanup["container_id"], cleanup["owner"]


def contract_reward(result):
    return RewardPolicy("terminal-bench", "terminal-validated-verifier-v1").reward(result)


def validate_sealed_group(manifest_path, expected_policy, expected_group_id, *, expected_lease=None,
                          expected_group_size=2, evaluation=False, expected_max_turns=None):
    """A: recheck every CPU boundary for a complete group or explicit evaluation.

    This is not optimizer admission: require the caller's independently owned
    GPU complete-ACK receipts and a real serving-controller pause afterward.
    """
    path = Path(manifest_path)
    root, seal = path.parent, read_json(path, 8 * 1024**2)
    expected = expected_policy if isinstance(expected_policy, PolicyRef) else PolicyRef.from_dict(expected_policy)
    if (type(expected_group_size) is not int or not 1 <= expected_group_size <= 64
            or evaluation != (expected_group_size == 1)):
        raise ValueError("singletons are evaluation only; training groups require G >= 2")
    kind = "policy_evaluation" if evaluation else "training_group"
    if (seal.get("schema_version") != 1 or seal.get("kind") != "terminalbench-sealed-group"
            or seal.get("group_id") != expected_group_id or seal.get("receipt_kind") != kind
            or seal.get("gpu_delivery_proven") is not False):
        raise ValueError("sealed group is excluded, unsupported or differs from expected group")
    closure = seal["closure"]
    if (closure.get("coordinator_closed") is not True or closure.get("cpu_spool_terminal") is not True
            or closure.get("lifecycle_failures") != [] or closure.get("quarantined") != {}
            or closure.get("admission_stopped_reason") is not None
            or any(value != 0 for quantities in closure.get("reserved", {}).values() for value in quantities.values())):
        raise ValueError("CPU group has unresolved execution, resources, cleanup or delivery")
    inventory = seal["inventory"]
    if not isinstance(inventory, dict) or evidence_digest(inventory) != seal.get("inventory_sha256"):
        raise ValueError("invalid sealed inventory digest")
    if not 1 <= len(inventory) <= 20000 or sum(row["bytes"] for row in inventory.values()) > 4 * 1024**3:
        raise ValueError("sealed evidence exceeds explicit importer capacity")
    for relative, info in inventory.items():
        member = _member(root, relative, inventory)
        if file_sha256(member) != info:
            raise ValueError("sealed original file changed: " + relative)
    if expected_max_turns is not None:
        if type(expected_max_turns) is not int or not 1 <= expected_max_turns <= 1024:
            raise ValueError("expected_max_turns must be an integer in 1..1024")
        cpu_config = read_json(_member(root, "config.json", inventory))
        actual_max_turns = cpu_config.get("terminal_bench", {}).get("agent_max_turns", 128)
        if type(actual_max_turns) is not int or actual_max_turns != expected_max_turns:
            raise ValueError("sealed CPU agent turn budget differs from the workflow declaration")
    original_registration = read_json(_member(root, seal["registration_file"], inventory))
    if evaluation:
        from types import SimpleNamespace
        registered = SimpleNamespace(**{**original_registration,
            "plans": [decode_plan(value) for value in original_registration["plans"]],
            "policy": PolicyRef.from_dict(original_registration["policy"])})
    else:
        registered = registration(original_registration)
    if (registered.policy != expected or registered.group_id != expected_group_id
            or len(registered.plans) != expected_group_size
            or registered.plans[0].task.task_id not in TASKS):
        raise ValueError("original registration differs from the exact expected group and policy lease")
    lease = read_json(_member(root, "lease.json", inventory), 1024**2)
    capsule = read_json(_member(root, "public-capsule.json", inventory), 1024**2)
    if (lease["policy"] != expected.to_dict() or lease["public_capsule_sha256"] != evidence_digest(capsule)
            or lease.get("cpu_cleanup_is_gpu_drain") is not False
            or expected_lease is not None and lease != expected_lease
            or lease.get("group_size", 2) != expected_group_size
            or evaluation and lease.get("purpose") != "policy_evaluation"
            or capsule.get("task_provenance_sha256") != TASK_PIN or capsule.get("payload_sha256") != PAYLOAD_PIN):
        raise ValueError("original pre-admission public-context lease differs")
    task_id = registered.plans[0].task.task_id
    leased_task = lease["tasks"][task_id]
    if (registered.initial_context_sha256 != leased_task["initial_context_sha256"]
            or registered.reset_sha256 != leased_task["reset_sha256"]
            or leased_task["instruction_sha256"] != registered.plans[0].task.source_sha256):
        raise ValueError("registered initial context or isolated reset differs from prepared task")
    contract = TrainingGenerationContract(expected, lease["capabilities"])
    from rollout_manager.datasets import TerminalBenchAdapter
    for plan in registered.plans:
        original = TerminalBenchAdapter(executor=plan.task.resources.executor,
            task_root=Path(plan.task.payload["task_path"]).parent, task_ids=[task_id]).enumerate()[0]
        reset = capsule["tasks"][task_id]["reset_spec"]
        if (asdict(plan.task) != asdict(original) or reset["task"] != asdict(original)
                or evidence_digest(reset) != registered.reset_sha256
                or reset["task_provenance_sha256"] != TASK_PIN or reset["payload_sha256"] != PAYLOAD_PIN
                or reset["new_owned_container_per_sample"] is not True or reset["network_mode"] != "none"):
            raise ValueError("sealed task definition/reset differs from the pinned original dataset")
    assembler = None if evaluation else GroupAssembler(max_groups=1, max_episodes=expected_group_size)
    if assembler is not None:
        assembler.register(registered)
    receipt, containers, owners = None, set(), set()
    if len(seal["episode_files"]) != expected_group_size:
        raise ValueError("sealed receipt lacks exactly the expected independent episode results")
    for plan, relative in zip(registered.plans, seal["episode_files"]):
        result = decode_result(read_json(_member(root, relative, inventory)))
        if result.identity != plan.identity or result.source_sha256 != plan.task.source_sha256:
            raise ValueError("original episode identity/source differs from registration")
        container, owner = _check_episode(root, seal["cpu_output_root"], inventory, plan, result, contract, capsule)
        containers.add(container)
        owners.add(owner)
        receipt = evaluation_receipt(original_registration, result) if evaluation else assembler.complete(result)
    saved_value = read_json(_member(root, seal["receipt_file"], inventory))
    saved = receipt if evaluation else TrainingGroup.from_dict(saved_value)
    if (not isinstance(receipt, PolicyEvaluation if evaluation else TrainingGroup)
            or receipt.to_dict() != saved_value or len(containers) != expected_group_size
            or len(owners) != expected_group_size):
        raise ValueError("sealed typed group differs from original verifier reassembly or fresh resets")
    return saved
