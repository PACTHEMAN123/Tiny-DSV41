"""Run the installed vLLM CLI while retaining its unmodified timing inputs."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import time


def main():
    from vllm.entrypoints.generate.base import serving

    original = serving.build_per_request_timing_metrics
    destination = Path(os.environ["ROLLOUT_ENGINE_TIMELINE"])
    source = Path(serving.__file__)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()

    def measured(metrics, num_generation_tokens):
        result = original(metrics, num_generation_tokens)
        if metrics is not None:
            # Match to responses by all original metrics and token count, rejecting
            # ambiguous matches during analysis. No generation output is changed.
            before = time.monotonic_ns()
            unix = time.time_ns()
            after = time.monotonic_ns()
            record = {
                "pid": os.getpid(), "unix_ns": unix,
                "monotonic_before_ns": before, "monotonic_after_ns": after,
                "engine_timestamps": {name: getattr(metrics, name) for name in
                    ("queued_ts", "scheduled_ts", "first_token_ts", "last_token_ts")},
                "completion_tokens": num_generation_tokens,
                "response_metrics": result.model_dump(), "source_sha256": source_hash,
            }
            descriptor = os.open(destination, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            try:
                data = (json.dumps(record, separators=(",", ":")) + "\n").encode()
                if os.write(descriptor, data) != len(data):
                    raise OSError("incomplete engine timeline record")
            finally:
                os.close(descriptor)
        return result

    serving.build_per_request_timing_metrics = measured
    sys.argv = sys.argv[1:]
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
