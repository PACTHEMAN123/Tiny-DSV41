"""Additive payload-free timing around the genuine offline Pouch environment."""

import hashlib
import json
import os
import time

from harbor.models.trial.paths import EnvironmentPaths
from harbor.utils.scripts import build_execution_command
from .harbor_bootstrap_ubuntu_merge_sparql import UbuntuMergeSparqlEnvironment


class TimingMixin:
    def _event(self, operation, started, **values):
        row = {"name": operation, "clock_domain": "cpu_monotonic", "ts_ns": started,
               "duration_ns": time.monotonic_ns() - started, "wall_unix": time.time(), **values}
        path = self.trial_paths.trial_dir / "environment-phases.jsonl"
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")

    async def _timed(self, operation, call, **metadata):
        started = time.monotonic_ns()
        result, error = None, None
        try:
            result = await call()
            return result
        except BaseException as exc:
            error = type(exc).__name__
            raise
        finally:
            self._event(operation, started, error_type=error, **metadata,
                        return_code=getattr(result, "return_code", None))

    async def start(self, force_build=False):
        return await self._timed("environment_start", lambda: super(TimingMixin, self).start(force_build))

    async def stop(self, delete=True):
        return await self._timed("environment_stop", lambda: super(TimingMixin, self).stop(delete))

    async def exec(self, command, **kwargs):
        paths = EnvironmentPaths.for_os(self.os)
        original = build_execution_command(str(paths.tests_dir / "test.sh"),
            stdout_path=str(paths.verifier_dir / self.trial_paths.test_stdout_path.name), task_os=self.os)
        is_verifier = command == original
        if is_verifier:
            from .verification import original_verifier_sources
            sources_before = original_verifier_sources(self.environment_dir)
        started = time.time()
        result = await self._timed("original_verifier" if is_verifier else "exec",
            lambda: super(TimingMixin, self).exec(command, **kwargs),
            command_sha256=hashlib.sha256(command.encode()).hexdigest())
        if is_verifier:
            from .bootstrap_payload import write_new
            write_new(self.trial_paths.trial_dir / "original-verifier-terminal.json",
                (json.dumps({"terminal_proven_by_provider": True, "return_code": result.return_code,
                    "command_sha256": hashlib.sha256(command.encode()).hexdigest(),
                    "original_verifier_sources_before": sources_before,
                    "original_verifier_sources_after": original_verifier_sources(self.environment_dir),
                    "started_unix": started, "finished_unix": time.time()}) + "\n").encode())
        return result

    async def upload_dir(self, source_dir, target_dir):
        return await self._timed("upload_dir", lambda: super(TimingMixin, self).upload_dir(source_dir, target_dir))

    async def download_dir(self, source_dir, target_dir):
        return await self._timed("download_dir", lambda: super(TimingMixin, self).download_dir(source_dir, target_dir))


class ProfiledUbuntuMergeSparqlEnvironment(TimingMixin, UbuntuMergeSparqlEnvironment):
    pass
