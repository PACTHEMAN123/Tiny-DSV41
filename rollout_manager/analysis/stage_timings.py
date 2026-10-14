"""Ordinary host intervals with explicit clock and completion semantics."""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import socket
import time


class StageTimings:
    """Append one completed/failed interval without adding device barriers."""

    def __init__(self, path, *, clock_domain):
        self.path = Path(path)
        self.identity = {"schema_version": 1, "hostname": socket.gethostname(),
                         "pid": os.getpid(), "clock_domain": clock_domain}

    @contextmanager
    def interval(self, stage, *, completion="host_call_return", **details):
        record = {**self.identity, "stage": stage, "details": details,
                  "completion": completion, "start_unix_ns": time.time_ns(),
                  "start_monotonic_ns": time.monotonic_ns(), "status": "completed"}
        try:
            yield
        except BaseException as error:
            record.update(status="failed", error_type=type(error).__name__)
            raise
        finally:
            record.update(end_monotonic_ns=time.monotonic_ns(), end_unix_ns=time.time_ns())
            record["duration_seconds"] = (record["end_monotonic_ns"] - record["start_monotonic_ns"]) / 1e9
            descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            with os.fdopen(descriptor, "a", encoding="utf-8") as output:
                output.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")

    def call(self, stage, function, *args, details=None, completion="host_call_return", **kwargs):
        with self.interval(stage, completion=completion, **(details or {})):
            return function(*args, **kwargs)
