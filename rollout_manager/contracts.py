"""Small, benchmark-discriminated contracts for rollout coordination."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

SCHEMA_VERSION = 1
BENCHMARKS = frozenset({"kernelbench", "cuda-agent", "terminal-bench", "prompt-response"})


@dataclass(frozen=True)
class ResourceRequirements:
    executor: str
    cpu: float = 0
    memory_bytes: int = 0
    storage_bytes: int = 0
    gpu_count: int = 0
    accounting: str = "episode"

    def __post_init__(self):
        import math
        if not self.executor or self.accounting not in {"episode", "backend-stage"}:
            raise ValueError("resources require an executor and supported accounting scope")
        values = (self.cpu, self.memory_bytes, self.storage_bytes, self.gpu_count)
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or
               not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("resource quantities must be finite and nonnegative")


@dataclass(frozen=True)
class TaskSpec:
    benchmark: str
    task_id: str
    source_sha256: str
    payload: Mapping[str, Any]
    resources: ResourceRequirements
    budgets: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.benchmark not in BENCHMARKS or not self.task_id or not self.source_sha256:
            raise ValueError("task requires a supported benchmark, pinned identity and source hash")
        kind = ("task_path" if self.benchmark == "terminal-bench" else
                "prompt_response" if self.benchmark == "prompt-response" else "native_case")
        if kind not in self.payload:
            raise ValueError(f"{self.benchmark} payload requires {kind}")
        if self.benchmark == "terminal-bench" and "code" in self.payload:
            raise ValueError("Terminal-Bench does not have a CUDA reference-code contract")


@dataclass(frozen=True)
class EpisodeIdentity:
    run_id: str
    benchmark: str
    task_id: str
    group_id: str
    sample_index: int
    pass_index: int = 0
    trial_id: str | None = None
    sampling_seed: int | None = None

    def __post_init__(self):
        if not all((self.run_id, self.task_id, self.group_id)) or self.benchmark not in BENCHMARKS:
            raise ValueError("episode requires run, benchmark, task and group identities")
        if self.sample_index < 0 or self.pass_index < 0:
            raise ValueError("sample and pass indices must be nonnegative")


@dataclass(frozen=True)
class EpisodePlan:
    identity: EpisodeIdentity
    task: TaskSpec
    payload: Mapping[str, Any] | None = None

    def __post_init__(self):
        if (self.identity.benchmark, self.identity.task_id) != (self.task.benchmark, self.task.task_id):
            raise ValueError("episode and task identities disagree")


@dataclass(frozen=True)
class EpisodeResult:
    identity: EpisodeIdentity
    execution_state: str
    verification_state: str
    score_eligible: bool
    cleanup_state: str
    native_outcome: Mapping[str, Any]
    native_artifacts: Any
    usage: Mapping[str, Any] | None = None
    timings: Mapping[str, Any] | None = None
    failure_category: str | None = None
    model_identity: str | None = None
    request_state: Mapping[str, Any] | None = None
    configuration_sha256: str | None = None
    source_sha256: str | None = None


class CancellationToken:
    """Cooperative cancellation shared by async tasks and native threads."""

    def __init__(self, event: threading.Event | None = None):
        self.event = event if event is not None else threading.Event()
        self.reason: str | None = None
        self._lock = threading.Lock()

    def cancelled(self) -> bool:
        return self.event.is_set()

    def cancel(self, reason: str = "cancelled") -> None:
        with self._lock:
            if self.reason is None:
                self.reason = reason
            self.event.set()


class DatasetAdapter(Protocol):
    def enumerate(self) -> list[TaskSpec]: ...


class EpisodeBackend(Protocol):
    async def execute(self, plan: EpisodePlan, cancellation: CancellationToken) -> EpisodeResult: ...


class ModelTransport(Protocol):
    async def submit(self, request: dict) -> dict: ...


def request_uncertain(state: Mapping[str, Any] | None) -> bool:
    """Local cancellation alone never establishes remote termination."""
    if not state:
        return False
    if state.get("uncertain") is True or state.get("spool_terminal") is False:
        return True
    if state.get("state") == "terminal_failure_received":
        from .generation_failure import failure_is_terminal
        receipt = state.get("failure_receipt")
        try:
            if (state.get("generation_failure") != "validated" or not isinstance(receipt, Mapping)
                    or receipt["request_id"] != state.get("id")
                    or receipt["request_payload_sha256"] != state.get("request_payload_sha256")
                    or not failure_is_terminal(receipt)):
                return True
        except (KeyError, TypeError, ValueError):
            return True
    for key in ("state", "delivery_state"):
        value = str(state.get(key, "")).lower()
        if "unknown" in value or "uncertain" in value:
            return True
    return any(request_uncertain(value) for value in state.values() if isinstance(value, Mapping))
