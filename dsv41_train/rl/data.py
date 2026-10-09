"""Tokenized JSONL records and small, replaceable outcome rewards."""

from __future__ import annotations

import importlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional


def validate_token_ids(value: Any, name: str, *, allow_empty: bool = False) -> list[int]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list of nonnegative integer token IDs")
    if not value and not allow_empty:
        raise ValueError(f"{name} must not be empty")
    if any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in value):
        raise ValueError(f"{name} must contain only nonnegative integer token IDs")
    return list(value)


@dataclass
class PromptRecord:
    prompt_ids: list[int]
    target_ids: Optional[list[int]] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.prompt_ids = validate_token_ids(self.prompt_ids, "prompt_ids")
        if self.target_ids is not None:
            self.target_ids = validate_token_ids(self.target_ids, "target_ids", allow_empty=True)
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a JSON object")
        self.metadata = dict(self.metadata)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PromptRecord":
        if not isinstance(value, Mapping):
            raise ValueError("each JSONL record must be an object")
        if "prompt_ids" not in value:
            raise ValueError("each JSONL record requires prompt_ids")
        metadata = value.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be a JSON object")
        metadata = dict(metadata)
        metadata.update({k: v for k, v in value.items() if k not in {"prompt_ids", "target_ids", "metadata"}})
        return cls(value["prompt_ids"], value.get("target_ids"), metadata)


RewardFn = Callable[[PromptRecord, list[int]], float]


def load_jsonl(path: str | Path) -> list[PromptRecord]:
    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                records.append(PromptRecord.from_mapping(json.loads(line)))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if not records:
        raise ValueError(f"{path}: no records found")
    return records


def _target(record: PromptRecord) -> list[int]:
    if record.target_ids is None:
        raise ValueError("this reward requires target_ids in every JSONL record")
    return record.target_ids


def token_accuracy(record: PromptRecord, response_ids: list[int]) -> float:
    """Position matches / max(target length, response length), counting EOS."""
    target = _target(record)
    denominator = max(len(target), len(response_ids))
    if denominator == 0:
        return 1.0
    return sum(a == b for a, b in zip(target, response_ids)) / denominator


def exact_match(record: PromptRecord, response_ids: list[int]) -> float:
    """Exact token equality; include a desired terminal EOS in target_ids."""
    return float(response_ids == _target(record))


def toy_token_score(record: PromptRecord, response_ids: list[int]) -> float:
    """Synthetic smoke-test reward encouraging larger token IDs, not task quality."""
    if not response_ids:
        raise ValueError("toy_token_score requires at least one response token")
    return sum(response_ids) / len(response_ids)


def build_reward(spec: str) -> RewardFn:
    """Resolve a builtin or import a module:function(record, response_ids) plugin."""
    builtins = {
        "token_accuracy": token_accuracy,
        "exact_match": exact_match,
        "toy_token_score": toy_token_score,
    }
    if spec in builtins:
        reward_fn = builtins[spec]
    else:
        module_name, separator, function_name = spec.partition(":")
        if not separator or not module_name or not function_name or ":" in function_name:
            raise ValueError(f"unknown reward {spec!r}; use a builtin or module:function")
        reward_fn = getattr(importlib.import_module(module_name), function_name)
        if not callable(reward_fn):
            raise ValueError(f"reward {spec!r} is not callable")

    def checked_reward(record: PromptRecord, response_ids: list[int]) -> float:
        reward = float(reward_fn(record, response_ids))
        if not math.isfinite(reward):
            raise ValueError(f"reward {spec!r} returned a non-finite value")
        return reward

    return checked_reward
