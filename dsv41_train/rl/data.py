"""Tokenized JSONL records and small, replaceable outcome rewards."""

from __future__ import annotations

import importlib
import json
import math
import warnings
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
    def from_mapping(cls, value: Mapping[str, Any], *, tokenizer=None) -> "PromptRecord":
        if not isinstance(value, Mapping):
            raise ValueError("each JSONL record must be an object")
        if sum(name in value for name in ("prompt_ids", "prompt", "messages")) != 1:
            raise ValueError("each JSONL record requires exactly one of prompt_ids, prompt or messages")
        prompt_ids = value.get("prompt_ids")
        if "prompt_ids" not in value:
            if tokenizer is None:
                raise ValueError("text records require tokenizer.json in the model directory")
            prompt_ids = tokenizer.encode_prompt(prompt=value.get("prompt"), messages=value.get("messages"))
        metadata = value.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be a JSON object")
        metadata = dict(metadata)
        metadata.update({k: v for k, v in value.items() if k not in {"prompt_ids", "target_ids", "metadata"}})
        return cls(prompt_ids, value.get("target_ids"), metadata)


RewardFn = Callable[[PromptRecord, list[int]], float]


def load_jsonl(path: str | Path, *, tokenizer=None, max_prompt_tokens=None,
               length_policy="error") -> list[PromptRecord]:
    if length_policy not in {"error", "skip"}:
        raise ValueError("length_policy must be error or skip")
    if max_prompt_tokens is not None and (type(max_prompt_tokens) is not int or max_prompt_tokens < 1):
        raise ValueError("max_prompt_tokens must be a positive integer")
    records = []
    skipped = 0
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = PromptRecord.from_mapping(json.loads(line), tokenizer=tokenizer)
                if max_prompt_tokens is not None and len(record.prompt_ids) > max_prompt_tokens:
                    if length_policy == "skip":
                        skipped += 1
                        continue
                    raise ValueError(f"prompt length {len(record.prompt_ids)} exceeds limit {max_prompt_tokens}")
                records.append(record)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if skipped:
        warnings.warn(f"{path}: skipped {skipped} records exceeding the prompt length limit", stacklevel=2)
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


def build_reward(spec: str, *, tokenizer=None) -> RewardFn:
    """Resolve a builtin or import a module:function(record, response_ids) plugin."""
    builtins = {
        "token_accuracy": token_accuracy,
        "exact_match": exact_match,
        "toy_token_score": toy_token_score,
    }
    from .rewards import TEXT_REWARDS, build_text_reward

    if spec in TEXT_REWARDS:
        if tokenizer is None:
            raise ValueError(f"reward {spec!r} requires tokenizer.json in the model directory")
        reward_fn = build_text_reward(spec, tokenizer)
    elif spec in builtins:
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

    checked_reward.strip_stop_tokens = spec in TEXT_REWARDS
    return checked_reward
