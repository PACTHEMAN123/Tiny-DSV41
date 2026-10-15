"""Versioned append-only events and new immutable result records."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Mapping

from .contracts import EpisodeIdentity, EpisodeResult, SCHEMA_VERSION


def json_value(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {field.name: json_value(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


class EventStore:
    def __init__(self, root: Path | str, run_id: str):
        self.root, self.run_id = Path(root), run_id
        self.root.parent.mkdir(parents=True, exist_ok=True)
        self.root.mkdir(exist_ok=False)
        (self.root / "results").mkdir()
        self._lock = threading.Lock()
        self._events = (self.root / "events.jsonl").open("x", encoding="utf-8")
        self._sequence = 0
        self.append("run_created")

    def append(self, event: str, identity: EpisodeIdentity | Mapping | None = None, *,
               turn_id=None, request_id=None, tool_call_id=None, trial_id=None, **details) -> dict:
        episode = json_value(identity)
        record = {"schema_version": SCHEMA_VERSION, "run_id": self.run_id,
                  "clock_domain": "coordinator_monotonic", "unix_ns": time.time_ns(),
                  "monotonic_ns": time.monotonic_ns(), "event": event,
                  "identity": episode, "turn_id": turn_id, "request_id": request_id,
                  "tool_call_id": tool_call_id,
                  "trial_id": trial_id if trial_id is not None else (episode or {}).get("trial_id"),
                  "details": json_value(details)}
        with self._lock:
            record["sequence"] = self._sequence
            self._sequence += 1
            self._events.write(json.dumps(record, allow_nan=False) + "\n")
            self._events.flush()
        return record

    def _write_new(self, path: Path, value: Any) -> Path:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(json_value(value), stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        return path

    def write_result(self, result: EpisodeResult) -> Path:
        identity = json_value(result.identity)
        key = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return self._write_new(self.root / "results" / (key + ".json"),
                               {"schema_version": SCHEMA_VERSION, **json_value(result)})

    def write_summary(self, summary: Mapping[str, Any]) -> Path:
        return self._write_new(self.root / "summary.json",
                               {"schema_version": SCHEMA_VERSION, "run_id": self.run_id, **json_value(summary)})

    def close(self) -> None:
        with self._lock:
            self._events.close()
