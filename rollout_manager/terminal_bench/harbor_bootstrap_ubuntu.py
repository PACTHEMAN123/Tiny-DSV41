"""Add genuine offline Python/curl tools to seven pinned Ubuntu task images."""

import json
import re
import shlex

from .bootstrap_payload import TARGET, verify, write_new
from .harbor_bootstrap_environment import BootstrapPouchEnvironment, commands as base_commands
from .harbor_pouch_environment import PouchEnvironment, PouchError
from .ubuntu_profile import PROFILE, TASKS, PYTHON_FILE, PYTHON_SHA256, PYTHON_ROOT

RUNTIME = {"file": "python/" + PYTHON_FILE, "sha256": PYTHON_SHA256,
           "version": "3.13.7", "target_root": PYTHON_ROOT}


def commands(task):
    install = ("set -eu; umask 022; test ! -e " + PYTHON_ROOT + "; test ! -L " + PYTHON_ROOT +
        "; test ! -e /usr/local/bin/python3.13; test ! -L /usr/local/bin/python3.13; "
        "mkdir -m 755 " + PYTHON_ROOT + "; tar --no-same-owner --no-same-permissions -xzf " +
        shlex.quote(TARGET + "/python/" + PYTHON_FILE) + " -C " + PYTHON_ROOT +
        "; test -x " + PYTHON_ROOT + "/python/bin/python3.13; ln -s " +
        PYTHON_ROOT + "/python/bin/python3.13 /usr/local/bin/python3.13")
    imports = "python3.13 -c " + shlex.quote(
        "import os,sys,ssl,hashlib,sqlite3,ctypes,bz2,lzma,zlib,multiprocessing,venv,json; "
        "assert sys.version_info[:3]==(3,13,7); "
        "assert os.path.realpath(sys.executable)=='/opt/harbor-python/python/bin/python3.13'; "
        "print(json.dumps({'python':sys.version.split()[0],'openssl':ssl.OPENSSL_VERSION,'sqlite':sqlite3.sqlite_version}))")
    base = base_commands(task)
    return [base[0], ("install_genuine_python", install, 90), base[1],
            ("native_runtime_imports", imports, 30)] + base[2:]


class UbuntuSevenEnvironment(BootstrapPouchEnvironment):
    bootstrap_profile = PROFILE
    bootstrap_tasks = TASKS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.bootstrap.get("python_runtime") != RUNTIME:
            raise ValueError("Ubuntu Python runtime policy differs")

    async def start(self, force_build=False):
        if self.bootstrap_started:
            raise PouchError("Ubuntu bootstrap start is single-attempt")
        self.bootstrap_started = True
        verify(self.bootstrap_payload_dir, self.bootstrap_manifest_sha256, profile=PROFILE, tasks=TASKS)
        evidence = {"profile": PROFILE, "task_id": self.bootstrap_task_id,
            "manifest_sha256": self.bootstrap_manifest_sha256, "python_runtime": RUNTIME,
            "steps": [], "preprovisioned": False, "original_verifier_executed": False,
            "task_test_results": None, "runtime_qualified": False,
            "existing_python_defaults_preserved": True}
        try:
            await PouchEnvironment.start(self, force_build=force_build)
            if self._receipt.get("image_id") != self.bootstrap_task["image_id"]:
                raise PouchError("Started image differs from inspected Ubuntu image")
            await self.upload_dir(self.bootstrap_payload_dir, TARGET)
            for name, command, deadline in commands(self.bootstrap_task):
                result = await self.exec(command, cwd=TARGET, user="root", timeout_sec=deadline)
                stdout, stderr = result.stdout or "", result.stderr or ""
                if len(stdout.encode()) + len(stderr.encode()) > 1024**2:
                    raise PouchError("Ubuntu bootstrap observation exceeds bound")
                evidence["steps"].append({"name": name, "command": command,
                    "return_code": result.return_code, "stdout": stdout, "stderr": stderr})
                if result.return_code != 0:
                    raise PouchError("Genuine Ubuntu bootstrap command failed: " + name)
                if name == "uv_version" and re.fullmatch(r"uv 0\.9\.5(?: \([^\r\n()]+\))?\r?\n?", stdout) is None:
                    raise PouchError("Ubuntu uv version differs")
                if name == "uvx_pytest_version" and "pytest 8.4.1" not in stdout:
                    raise PouchError("Ubuntu pytest version differs")
                if name == "uvx_pytest_plugin" and "--ctrf" not in stdout:
                    raise PouchError("Ubuntu pytest lacks genuine CTRF plugin")
            evidence["preprovisioned"] = True
            self._record(bootstrap_profile=PROFILE, bootstrap_preprovisioned=True,
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
