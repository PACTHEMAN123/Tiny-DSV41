"""Exact CPU training records, independent of benchmark admission groups."""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _digest(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")


def _tokens(values, name):
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} requires exact token IDs")
    values = tuple(values)
    if not values or any(type(value) is not int or value < 0 for value in values):
        raise ValueError(f"{name} requires nonempty nonnegative integer token IDs")
    return values


def _finite(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _fields(value, names):
    if not isinstance(value, dict) or set(value) != set(names):
        raise ValueError("training record has missing or unknown fields")


def token_digest(tokens: Sequence[int]) -> str:
    return hashlib.sha256(json.dumps(_tokens(tokens, "tokens"), separators=(",", ":")).encode()).hexdigest()


class _Record:
    def __reduce__(self):
        # Frozen mapping views are not directly pickleable; CPU IPC uses the
        # same validated representation as durable JSON records.
        return type(self).from_dict, (self.to_dict(),)


@dataclass(frozen=True)
class PolicyRef(_Record):
    version: int
    digest: str
    lineage: str
    identities: Mapping[str, str] = field(default_factory=dict)
    sampling: Mapping[str, Any] = field(default_factory=lambda: {"temperature": 1.0, "top_p": 1.0, "top_k": 0})

    def __post_init__(self):
        _integer(self.version, "policy version")
        _digest(self.digest, "policy digest")
        if not isinstance(self.lineage, str) or not self.lineage:
            raise ValueError("policy requires run lineage")
        identities = dict(self.identities)
        if any(not isinstance(key, str) or not key or not isinstance(value, str) or not value
               for key, value in identities.items()):
            raise ValueError("policy identities require nonempty string names and values")
        sampling = dict(self.sampling)
        _fields(sampling, ("temperature", "top_p", "top_k"))
        temperature = _finite(sampling["temperature"], "temperature")
        if (temperature <= 0 or type(sampling["top_p"]) not in (int, float) or sampling["top_p"] != 1
                or type(sampling["top_k"]) is not int or sampling["top_k"] != 0):
            raise ValueError("training requires positive temperature and full categorical sampling")
        object.__setattr__(self, "identities", MappingProxyType(identities))
        object.__setattr__(self, "sampling", MappingProxyType({"temperature": temperature, "top_p": 1.0, "top_k": 0}))

    def to_dict(self):
        return {"version": self.version, "digest": self.digest, "lineage": self.lineage,
                "identities": dict(self.identities), "sampling": dict(self.sampling)}

    @classmethod
    def from_dict(cls, value):
        _fields(value, ("version", "digest", "lineage", "identities", "sampling"))
        return cls(**value)


@dataclass(frozen=True)
class TurnTrace(_Record):
    turn_id: int
    request_id: str
    context_token_ids: Sequence[int]
    action_token_ids: Sequence[int]
    behavior_log_probs: Sequence[float]
    policy: PolicyRef
    finish_reason: str = "stop"
    stop_token_count: int = 0
    context_truncation: str = "none"

    def __post_init__(self):
        _integer(self.turn_id, "turn_id", 1)
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("turn requires a request identity")
        if not isinstance(self.policy, PolicyRef):
            raise ValueError("turn requires an actual policy receipt")
        context = _tokens(self.context_token_ids, "context_token_ids")
        actions = _tokens(self.action_token_ids, "action_token_ids")
        probabilities = tuple(_finite(value, "behavior log probability") for value in self.behavior_log_probs)
        if len(actions) != len(probabilities) or any(value > 0 for value in probabilities):
            raise ValueError("each action token requires one nonpositive behavior log probability")
        if self.finish_reason not in {"eos", "stop", "length", "tool_calls"}:
            raise ValueError("unsupported sampled finish reason")
        _integer(self.stop_token_count, "stop_token_count")
        if self.stop_token_count > len(actions):
            raise ValueError("stop_token_count exceeds sampled actions")
        if not isinstance(self.context_truncation, str) or not self.context_truncation:
            raise ValueError("context truncation rule must be explicit")
        object.__setattr__(self, "context_token_ids", context)
        object.__setattr__(self, "action_token_ids", actions)
        object.__setattr__(self, "behavior_log_probs", probabilities)

    def to_dict(self):
        return {"turn_id": self.turn_id, "request_id": self.request_id,
                "context_token_ids": list(self.context_token_ids), "action_token_ids": list(self.action_token_ids),
                "behavior_log_probs": list(self.behavior_log_probs), "policy": self.policy.to_dict(),
                "finish_reason": self.finish_reason, "stop_token_count": self.stop_token_count,
                "context_truncation": self.context_truncation}

    @classmethod
    def from_dict(cls, value):
        _fields(value, cls.__dataclass_fields__)
        return cls(**{**value, "policy": PolicyRef.from_dict(value["policy"])})


@dataclass(frozen=True)
class TrainingTrajectory(_Record):
    group_id: int
    task_id: str
    initial_context_sha256: str
    reset_sha256: str
    sample_index: int
    turns: Sequence[TurnTrace]
    reward: float | None
    reward_eligible: bool = True
    exclusion_reason: str | None = None
    evidence: Mapping[str, Any] = field(default_factory=dict)
    sample_id: str | None = None
    raw_reward: float | None = None
    sampling_seed: int | None = None

    def __post_init__(self):
        _integer(self.group_id, "training group_id")
        _integer(self.sample_index, "sample_index")
        if not isinstance(self.task_id, str) or not self.task_id:
            raise ValueError("trajectory requires task identity")
        _digest(self.initial_context_sha256, "initial context digest")
        _digest(self.reset_sha256, "reset digest")
        turns = tuple(self.turns)
        if not turns or any(not isinstance(turn, TurnTrace) for turn in turns):
            raise ValueError("trajectory requires exact turn traces")
        if [turn.turn_id for turn in turns] != list(range(1, len(turns) + 1)):
            raise ValueError("trajectory turn IDs must be consecutive and ordered")
        if len({turn.request_id for turn in turns}) != len(turns):
            raise ValueError("trajectory has duplicate request identities")
        if any(turn.policy != turns[0].policy for turn in turns):
            raise ValueError("all trajectory turns require one policy receipt")
        if token_digest(turns[0].context_token_ids) != self.initial_context_sha256:
            raise ValueError("initial context digest differs from actual tokens")
        if type(self.reward_eligible) is not bool:
            raise ValueError("reward eligibility must be explicit")
        if self.reward is not None:
            object.__setattr__(self, "reward", _finite(self.reward, "reward"))
        if self.raw_reward is not None:
            object.__setattr__(self, "raw_reward", _finite(self.raw_reward, "raw reward"))
        if not self.reward_eligible and (not isinstance(self.exclusion_reason, str) or not self.exclusion_reason):
            raise ValueError("excluded trajectory requires a reason")
        if self.reward_eligible and self.exclusion_reason is not None:
            raise ValueError("eligible trajectory cannot have an exclusion reason")
        if self.sampling_seed is not None:
            _integer(self.sampling_seed, "sampling_seed")
        sample_id = self.sample_id if self.sample_id is not None else f"{self.group_id}:{self.sample_index}"
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("trajectory requires sample identity")
        # Evidence stays JSON data; tensors and NaN cannot enter durable packets.
        evidence = json.loads(json.dumps(dict(self.evidence), allow_nan=False))
        object.__setattr__(self, "evidence", MappingProxyType(evidence))
        object.__setattr__(self, "sample_id", sample_id)
        object.__setattr__(self, "turns", turns)

    @property
    def policy(self):
        return self.turns[0].policy

    @property
    def policy_version(self):
        return self.policy.version

    @property
    def behavior_log_probs(self):
        return [value for turn in self.turns for value in turn.behavior_log_probs]

    @property
    def finish_reason(self):
        return self.turns[-1].finish_reason

    @property
    def stop_token_count(self):
        return self.turns[-1].stop_token_count

    def _single_turn(self):
        if len(self.turns) != 1:
            raise ValueError("multi-turn trajectories require per-turn contexts, not flattened responses")
        return self.turns[0]

    @property
    def prompt_ids(self):
        return list(self._single_turn().context_token_ids)

    @property
    def response_ids(self):
        return list(self._single_turn().action_token_ids)

    @property
    def content_ids(self):
        response = self.response_ids
        return response[:-self.stop_token_count] if self.stop_token_count else response

    def to_dict(self):
        return {"group_id": self.group_id, "task_id": self.task_id,
                "initial_context_sha256": self.initial_context_sha256, "reset_sha256": self.reset_sha256,
                "sample_index": self.sample_index, "sample_id": self.sample_id,
                "turns": [turn.to_dict() for turn in self.turns], "reward": self.reward,
                "reward_eligible": self.reward_eligible, "exclusion_reason": self.exclusion_reason,
                "evidence": dict(self.evidence), "raw_reward": self.raw_reward, "sampling_seed": self.sampling_seed}

    @classmethod
    def from_dict(cls, value):
        _fields(value, cls.__dataclass_fields__)
        return cls(**{**value, "turns": [TurnTrace.from_dict(turn) for turn in value["turns"]]})


@dataclass(frozen=True)
class TrainingGroup(_Record):
    group_id: int
    task_id: str
    initial_context_sha256: str
    reset_sha256: str
    policy: PolicyRef
    expected_size: int
    samples: Sequence[TrainingTrajectory]

    def __post_init__(self):
        _integer(self.group_id, "training group_id")
        _integer(self.expected_size, "expected group size", 2)
        samples = tuple(self.samples)
        if len(samples) != self.expected_size or any(not isinstance(sample, TrainingTrajectory) for sample in samples):
            raise ValueError("training requires a complete group of exactly G trajectories")
        if ({sample.sample_index for sample in samples} != set(range(self.expected_size))
                or len({sample.sample_id for sample in samples}) != self.expected_size):
            raise ValueError("training group requires unique sample identities and indices 0..G-1")
        request_ids = [turn.request_id for sample in samples for turn in sample.turns]
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("training group has duplicate request identities")
        for sample in samples:
            if (sample.group_id, sample.task_id, sample.initial_context_sha256, sample.reset_sha256, sample.policy) != (
                    self.group_id, self.task_id, self.initial_context_sha256, self.reset_sha256, self.policy):
                raise ValueError("training group requires the same task, initial context, reset and policy")
            if not sample.reward_eligible or sample.reward is None:
                raise ValueError("training requires G eligible finite rewards; missing is not zero")
        object.__setattr__(self, "samples", tuple(sorted(samples, key=lambda sample: sample.sample_index)))

    @property
    def policy_version(self):
        return self.policy.version

    def to_dict(self):
        return {"group_id": self.group_id, "task_id": self.task_id,
                "initial_context_sha256": self.initial_context_sha256, "reset_sha256": self.reset_sha256,
                "policy": self.policy.to_dict(), "expected_size": self.expected_size,
                "samples": [sample.to_dict() for sample in self.samples]}

    @classmethod
    def from_dict(cls, value):
        _fields(value, cls.__dataclass_fields__)
        return cls(**{**value, "policy": PolicyRef.from_dict(value["policy"]),
                      "samples": [TrainingTrajectory.from_dict(sample) for sample in value["samples"]]})


_RECORDS = {cls.__name__: cls for cls in (PolicyRef, TurnTrace, TrainingTrajectory, TrainingGroup)}


def dumps(record) -> str:
    if type(record).__name__ not in _RECORDS or type(record) is not _RECORDS[type(record).__name__]:
        raise TypeError("unsupported training record")
    return json.dumps({"schema_version": SCHEMA_VERSION, "kind": type(record).__name__, "record": record.to_dict()},
                      sort_keys=True, separators=(",", ":"), allow_nan=False)


def loads(value: str):
    def unique(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("duplicate JSON field")
            result[key] = item
        return result
    def invalid_constant(value):
        raise ValueError("nonfinite JSON number")
    payload = json.loads(value, object_pairs_hook=unique, parse_constant=invalid_constant)
    _fields(payload, ("schema_version", "kind", "record"))
    if type(payload["schema_version"]) is not int or payload["schema_version"] != SCHEMA_VERSION or payload["kind"] not in _RECORDS:
        raise ValueError("unsupported training record schema or kind")
    return _RECORDS[payload["kind"]].from_dict(payload["record"])


def from_rollout_group(group, policy: PolicyRef, task_id: str, reset_sha256: str, *, request_prefix=None) -> TrainingGroup:
    """Convert exact legacy samples; no Torch import or probability reconstruction."""
    if group.policy_version != policy.version:
        raise ValueError("legacy group differs from acknowledged policy version")
    if len(group.samples) < 2:
        raise ValueError("training requires a complete group of at least two samples")
    context = token_digest(group.samples[0].prompt_ids)
    prefix = request_prefix or hashlib.sha256(policy.lineage.encode()).hexdigest()[:12] + f"-g{group.group_id}"
    samples = []
    for index, sample in enumerate(group.samples):
        if sample.policy_version != policy.version:
            raise ValueError("legacy sample differs from acknowledged policy version")
        turn = TurnTrace(1, f"{prefix}-s{index}", sample.prompt_ids, sample.response_ids,
                         sample.behavior_log_probs, policy, sample.finish_reason, sample.stop_token_count)
        samples.append(TrainingTrajectory(group.group_id, task_id, context, reset_sha256, index,
                       (turn,), sample.reward, raw_reward=sample.raw_reward))
    return TrainingGroup(group.group_id, task_id, context, reset_sha256, policy, len(samples), samples)


def to_rollout_group(group: TrainingGroup):
    """Only single-turn groups fit the legacy Sample contract."""
    from dsv41_train.rl.rollout import RolloutGroup, Sample
    samples = []
    for trajectory in group.samples:
        turn = trajectory._single_turn()
        if turn.finish_reason == "tool_calls":
            raise ValueError("legacy Sample does not represent tool-call finish reasons")
        samples.append(Sample(list(turn.context_token_ids), list(turn.action_token_ids),
                       list(turn.behavior_log_probs), trajectory.reward, trajectory.policy_version,
                       turn.finish_reason, turn.stop_token_count, trajectory.raw_reward))
    return RolloutGroup(samples, group.group_id, group.policy_version)
