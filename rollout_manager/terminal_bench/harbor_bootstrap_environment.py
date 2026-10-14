"""Opt-in genuine offline bootstrap for four pinned Harbor/Pouch task images.

This only provisions generic tools before the agent starts. Harbor later uploads
and executes the original verifier. No tests, solutions, rewards or oracle run here.
"""

import json
import re
from pathlib import Path
import shlex
import sys

from .bootstrap_payload import ENV, PROFILE, TARGET, TASKS, verify, write_new

from harbor.models.task.config import NetworkMode
from .harbor_pouch_environment import PouchEnvironment, PouchError


def commands(task):
    packages = " ".join(shlex.quote(TARGET + "/" + name) for name in task["debs"])
    pinned_packages = task["package_versions"]
    check_packages = (
        "import json,subprocess; expected=" + repr(pinned_packages) + "; "
        "actual=dict(line.split('\\t',1) for line in subprocess.check_output("
        "['dpkg-query','-W','-f=${Package}\\t${Version}\\n',*expected],text=True).splitlines()); "
        "assert actual==expected,(actual,expected); print(json.dumps(actual,sort_keys=True))")
    uvx = "/root/.local/bin/uvx -p 3.13 -w pytest==8.4.1 -w pytest-json-ctrf==0.3.5 pytest"
    return [
        ("payload_hashes", "sha256sum -c SHA256SUMS", 30),
        ("native_python", "python3.13 -c " + shlex.quote(
            "import os,sys; assert os.geteuid()==0; assert sys.version_info[:3]==(3,13,7); print(sys.version)"), 30),
        ("install_curl", "apt-get --no-install-recommends -y install " + packages, 120),
        ("curl_package_versions", "python3.13 -c " + shlex.quote(check_packages), 30),
        ("curl_version", "/usr/bin/curl --version", 30),
        ("install_uv", "UV_DOWNLOAD_URL=" + shlex.quote("file://" + TARGET + "/uv-0.9.5") +
         " UV_INSTALL_DIR=/root/.local/bin sh " + shlex.quote(TARGET + "/uv-0.9.5/uv-installer.sh"), 60),
        ("uv_version", "/root/.local/bin/uv --version", 30),
        ("uvx_pytest_version", uvx + " --version", 60),
        ("uvx_pytest_plugin", uvx + " --help", 60),
    ]


class BootstrapPouchEnvironment(PouchEnvironment):
    bootstrap_profile = PROFILE
    bootstrap_tasks = TASKS

    def __init__(self, *args, bootstrap_payload_dir, bootstrap_manifest_sha256,
                 bootstrap_task_id=None, **kwargs):
        self.bootstrap_payload_dir = Path(bootstrap_payload_dir)
        self.bootstrap_manifest_sha256 = bootstrap_manifest_sha256
        self.bootstrap = verify(self.bootstrap_payload_dir, bootstrap_manifest_sha256,
                                profile=self.bootstrap_profile, tasks=self.bootstrap_tasks)
        if bootstrap_task_id is not None and bootstrap_task_id not in self.bootstrap["tasks"]:
            raise ValueError("Task is outside the prepared four-image bootstrap profile")
        self.bootstrap_task_id = bootstrap_task_id
        self.bootstrap_started = False
        super().__init__(*args, **kwargs)

    def _validate_definition(self):
        super()._validate_definition()
        if self.network_policy.network_mode != NetworkMode.NO_NETWORK:
            raise ValueError("Offline bootstrap requires effective no-network task policy")
        matches = [name for name, row in self.bootstrap["tasks"].items()
                   if row["image_reference"] == self.task_env_config.docker_image]
        if len(matches) != 1 or self.bootstrap_task_id not in (None, matches[0]):
            raise ValueError("Task is outside the prepared four-image bootstrap profile")
        self.bootstrap_task_id = matches[0]
        self.bootstrap_task = self.bootstrap["tasks"][self.bootstrap_task_id]
        if self.task_env_config.docker_image != self.bootstrap_task["image_reference"]:
            raise ValueError("Bootstrap task image reference differs")

    def _startup_env(self):
        return {**super()._startup_env(), **ENV}

    def _merge_env(self, env):
        return {**(super()._merge_env(env) or {}), **ENV}

    async def start(self, force_build=False):
        if self.bootstrap_started:
            raise PouchError("Bootstrap start is single-attempt")
        self.bootstrap_started = True
        verify(self.bootstrap_payload_dir, self.bootstrap_manifest_sha256,
               profile=self.bootstrap_profile, tasks=self.bootstrap_tasks)
        evidence = {"profile": self.bootstrap_profile, "task_id": self.bootstrap_task_id,
                    "manifest_sha256": self.bootstrap_manifest_sha256, "steps": [],
                    "original_verifier_executed": False, "task_test_results": None,
                    "runtime_qualified": False, "environment": ENV, "preprovisioned": False}
        try:
            await super().start(force_build=force_build)
            if self._receipt.get("image_id") != self.bootstrap_task["image_id"]:
                raise PouchError("Started image differs from inspected bootstrap image")
            await self.upload_dir(self.bootstrap_payload_dir, TARGET)
            for name, command, deadline in commands(self.bootstrap_task):
                result = await self.exec(command, cwd=TARGET, user="root", timeout_sec=deadline)
                stdout, stderr = result.stdout or "", result.stderr or ""
                if len(stdout.encode()) + len(stderr.encode()) > 1024**2:
                    raise PouchError("Bootstrap observation exceeds bound")
                evidence["steps"].append({"name": name, "command": command,
                    "return_code": result.return_code, "stdout": stdout, "stderr": stderr})
                if result.return_code != 0:
                    raise PouchError("Genuine offline bootstrap command failed: " + name)
                if name == "uv_version" and re.fullmatch(r"uv 0\.9\.5(?: \([^\r\n()]+\))?\r?\n?", stdout) is None:
                    raise PouchError("Installed uv version differs")
                if name == "uvx_pytest_version" and "pytest 8.4.1" not in stdout:
                    raise PouchError("uvx did not report the required pytest version")
                if name == "uvx_pytest_plugin" and "--ctrf" not in stdout:
                    raise PouchError("uvx did not load the real CTRF pytest plugin")
            evidence["preprovisioned"] = True
            self._record(bootstrap_profile=self.bootstrap_profile, bootstrap_preprovisioned=True,
                         bootstrap_manifest_sha256=self.bootstrap_manifest_sha256)
        except BaseException as error:
            evidence["failure_type"] = type(error).__name__
            self._broken = True
            if self._create_attempted:
                await self._cleanup_after_failure()
            raise
        finally:
            write_new(self.trial_paths.trial_dir / "offline-bootstrap.json",
                      (json.dumps(evidence, indent=2) + "\n").encode())
