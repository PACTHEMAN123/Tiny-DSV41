"""Exact merge/SPARQL bootstrap with genuine git and isolated RDFLib."""

import hashlib
import json
from pathlib import Path
import re
import shlex

from .bootstrap_payload import read, strict_object, write_new
from .harbor_bootstrap_ubuntu import UbuntuSevenEnvironment
from .harbor_pouch_cleanup import CleanupPollingMixin
from .harbor_pouch_environment import PouchError
from .ubuntu_profile import PROFILE, TASKS

PAYLOAD_TASKS = TASKS
PAYLOAD_PIN = "6abd7da26fc66958cd36274edebe7720da9eb24a62ceeb0e6c2ffaed8c9521f5"
TASK_PIN = "184faa0638cfd08499f51d53a7bf64a492daf1c14ec4f6aa9d889483678e228e"
IMAGE_MAP_PIN = "558ff7f5447cc723622eeda85091969aa04351aa71d291db5c0eda56187c8ab7"
OVERLAY_PIN = "c69dd9a1fa5945dc45138508c9bedefb1c3dbd07f91e4f93cab1964f4a0d315f"
GIT_IMAGE = "sha256:1994f7ed8975cf2a616e00daa45fc556e842418f8040398053f064cf0bba3c69"
TARGET = "/opt/harbor-merge-git"


def verify_overlay(directory):
    directory = Path(directory)
    if (not directory.is_absolute() or not directory.is_dir()
            or any(p.is_symlink() for p in (directory, *directory.parents))):
        raise ValueError("Real absolute git overlay required")
    raw = read(directory / "manifest.json", 1024**2)
    if hashlib.sha256(raw).hexdigest() != OVERLAY_PIN:
        raise ValueError("Git overlay manifest differs")
    manifest = json.loads(raw, object_pairs_hook=strict_object)
    if (manifest["profile"] != "ubuntu-merge-git-overlay-v1" or manifest["task"] != TASKS[0]
            or manifest["image_id"] != GIT_IMAGE or manifest["base_payload_sha256"] != PAYLOAD_PIN
            or manifest["roots"] != ["git"] or manifest["original_verifier_modified"] is not False):
        raise ValueError("Git overlay contract differs")
    expected, total = {"manifest.json"}, 0
    if not 1 <= len(manifest["files"]) <= 32:
        raise ValueError("Git member count differs")
    for row in manifest["files"]:
        name = row["path"]
        if (name != "SHA256SUMS" and re.fullmatch(r"packages/[a-f0-9]{64}\.deb", name) is None
                or name in expected or type(row["bytes"]) is not int or not 0 < row["bytes"] <= 8 * 1024**2):
            raise ValueError("Unexpected git overlay member")
        if any(p.is_symlink() for p in (directory / name, *(directory / name).parents)):
            raise ValueError("Linked git overlay member")
        body = read(directory / name, row["bytes"])
        if len(body) != row["bytes"] or hashlib.sha256(body).hexdigest() != row["sha256"]:
            raise ValueError("Git overlay member differs")
        expected.add(name)
        total += len(body)
    if (total != manifest["payload_bytes"] or total > 32 * 1024**2
            or any(p.is_symlink() for p in directory.rglob("*"))
            or {str(p.relative_to(directory)) for p in directory.rglob("*") if not p.is_dir()} != expected):
        raise ValueError("Git overlay aggregate differs")
    return manifest


def git_commands(manifest):
    packages = " ".join(shlex.quote(TARGET + "/packages/" + row["file"]) for row in manifest["packages"].values())
    expected = {**manifest["installed_base_versions"], **{name: row["Version"] for name, row in manifest["packages"].items()}}
    check = ("import subprocess; expected=" + repr(expected) + "; "
        "actual=dict(line.split('\\t',1) for line in subprocess.check_output("
        "['dpkg-query','-W','-f=${Package}\\t${Version}\\n'],text=True).splitlines()); "
        "assert actual==expected,(actual,expected); "
        "assert not subprocess.check_output(['dpkg','--audit'],text=True).strip(); "
        "print('exact installed package map verified')")
    apt = "apt-get --no-install-recommends --no-remove -y install " + packages
    return (("git_hashes", "sha256sum -c SHA256SUMS", 30, None),
        ("git_solver", "DEBIAN_FRONTEND=noninteractive " + apt.replace("apt-get ", "apt-get --simulate ", 1), 60, None),
        ("git_install", "DEBIAN_FRONTEND=noninteractive " + apt, 120, None),
        ("git_package_map", "python3.13 -I -B -c " + shlex.quote(check), 30, "exact installed package map verified"),
        ("git_version", "/usr/bin/git --version", 30, "git version 2.43.0"))


def prerequisite_commands(task):
    if task == "merge-diff-arc-agi-task":
        paths = ("/app/bundle1.bundle", "/app/bundle2.bundle", "/app/examples.json")
        check = ("import os,stat; paths=" + repr(paths) + "; "
                 "assert all(stat.S_ISREG(os.lstat(p).st_mode) and os.lstat(p).st_size>0 for p in paths); "
                 "print('original task inputs present')")
        return (("merge_inputs", "python3.13 -I -B -c " + shlex.quote(check), 30, "original task inputs present"),)
    if task == "sparql-university":
        check = ("import importlib.metadata,os,sys,rdflib,pyparsing; "
                 "assert sys.version_info[:3]==(3,13,7); "
                 "assert importlib.metadata.version('rdflib')==rdflib.__version__=='7.1.4'; "
                 "assert importlib.metadata.version('pyparsing')=='3.2.3'; "
                 "assert sys.prefix!=sys.base_prefix; "
                 "assert os.path.realpath(rdflib.__file__).startswith(os.path.realpath(sys.prefix)+'/'); "
                 "print('isolated RDFLib 7.1.4; PyParsing 3.2.3')")
        fixture = ("import os,stat; s=os.lstat('/app/university_graph.ttl'); "
                   "assert stat.S_ISREG(s.st_mode) and s.st_size>0; print('original graph present')")
        return (("rdflib_runtime", "/root/.local/bin/uvx --no-config --offline -p 3.13 "
                 "--with rdflib==7.1.4 python -I -B -c " + shlex.quote(check), 60,
                 "isolated RDFLib 7.1.4; PyParsing 3.2.3"),
                ("sparql_input", "python3.13 -I -B -c " + shlex.quote(fixture), 30, "original graph present"))
    raise ValueError("Unsupported prerequisite task")


def prerequisite_diagnostics(stdout, stderr):
    # Retain bounded evidence without recording paths, exception messages or task data.
    stdout_bytes, stderr_bytes = stdout.encode(), stderr.encode()
    if len(stdout_bytes) + len(stderr_bytes) > 1024**2:
        raise PouchError("Prerequisite observation exceeds bound")
    known_errors = ("AssertionError", "ModuleNotFoundError", "ImportError", "RuntimeError",
                    "PermissionError", "FileNotFoundError", "SyntaxError", "ValueError")
    return {"stdout_bytes": len(stdout_bytes), "stderr_bytes": len(stderr_bytes),
            "stdout_sha256": hashlib.sha256(stdout_bytes).hexdigest(),
            "stderr_sha256": hashlib.sha256(stderr_bytes).hexdigest(),
            "stderr_python_error_types": [name for name in known_errors
                if re.search(r"(?m)^" + name + r"(?::|$)", stderr)]}


class UbuntuMergeSparqlEnvironment(CleanupPollingMixin, UbuntuSevenEnvironment):
    def __init__(self, *args, bootstrap_manifest_sha256, **kwargs):
        if bootstrap_manifest_sha256 != PAYLOAD_PIN:
            raise ValueError("Exact Ubuntu merge/SPARQL payload required")
        super().__init__(*args, bootstrap_manifest_sha256=bootstrap_manifest_sha256, **kwargs)
        self.git_overlay_dir = self.bootstrap_payload_dir.parent / "git-overlay"
        self.git_overlay = verify_overlay(self.git_overlay_dir)

    def _validate_definition(self):
        super()._validate_definition()
        if self.bootstrap_task_id not in TASKS:
            raise ValueError("Only merge-diff-arc-agi-task and sparql-university are admitted")

    async def start(self, force_build=False):
        await super().start(force_build=force_build)
        evidence = {"task_id": self.bootstrap_task_id, "checks": [], "verified": False,
            "fixture_contents_read": False, "original_verifier_executed": False, "agent_executed": False}
        try:
            steps = []
            if self.bootstrap_task_id == "merge-diff-arc-agi-task":
                if self._receipt.get("image_id") != GIT_IMAGE:
                    raise PouchError("Git base image differs")
                verify_overlay(self.git_overlay_dir)
                await self.upload_dir(self.git_overlay_dir, TARGET)
                steps.extend(git_commands(self.git_overlay))
            steps.extend(prerequisite_commands(self.bootstrap_task_id))
            for name, command, timeout, expected in steps:
                result = await self.exec(command, cwd=TARGET if name.startswith("git_") else "/", user="root", timeout_sec=timeout)
                stdout, stderr = result.stdout or "", result.stderr or ""
                diagnostics = prerequisite_diagnostics(stdout, stderr)
                matches = expected is None or stdout.strip() == expected
                evidence["checks"].append({"name": name, "return_code": result.return_code,
                    "command_sha256": hashlib.sha256(command.encode()).hexdigest(),
                    "expected_observation_matches": matches, "diagnostics": diagnostics})
                if result.return_code or not matches:
                    raise PouchError("Task prerequisite failed: " + name)
            evidence["verified"] = True
            self._record(agent_prerequisites_verified=True)
        except BaseException as error:
            evidence["failure_type"] = type(error).__name__
            self._broken = True
            if self._create_attempted:
                await self._cleanup_after_failure()
            raise
        finally:
            write_new(self.trial_paths.trial_dir / "agent-prerequisites.json", (json.dumps(evidence, indent=2) + "\n").encode())
