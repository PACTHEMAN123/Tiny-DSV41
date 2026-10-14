"""Canonical source identities and atomic evidence records."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
MAX_BODY = 262144


def wire(value, limit=None):
    def check(item, depth):
        if depth > 32:
            raise ValueError("JSON nesting exceeds 32")
        if isinstance(item, dict):
            if not all(isinstance(key, str) for key in item):
                raise ValueError("JSON keys must be strings")
            for child in item.values():
                check(child, depth + 1)
        elif isinstance(item, (list, tuple)):
            for child in item:
                check(child, depth + 1)
        elif item is not None and not isinstance(item, (str, int, float, bool)):
            raise ValueError("only JSON data is accepted")
        elif isinstance(item, float) and not math.isfinite(item):
            raise ValueError("non-finite JSON number")

    check(value, 0)
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if limit is not None and len(encoded.encode("utf-8")) > limit:
        raise ValueError("payload exceeds its reserved byte limit")
    return encoded


def digest(value):
    return hashlib.sha256(wire(value).encode()).hexdigest()


def save_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def process_identity(pid):
    root = Path("/proc") / str(pid)
    command = (root / "cmdline").read_bytes()
    return {"pid": pid,
            "start_ticks": (root / "stat").read_text().rsplit(")", 1)[1].split()[19],
            "command_sha256": hashlib.sha256(command).hexdigest(),
            "argv": [part.decode() for part in command.split(b"\0") if part]}
