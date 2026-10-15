"""Pinned original task-copy validation, without executing a verifier."""
import json
import math
import os
from pathlib import Path
import re
import socket
import stat
import time
from .bootstrap_payload import TASKS, digest, read, strict_object


_REVIEWED_SPARQL_BASELINE = {
    "task_id": "sparql-university",
    "qualification_sha256": "92a20307c315d0c4e22955e638c111da8dff6d151fc73b29e7fc813134bc8caf",
    "ctrf_sha256": "c4e78ed7b3f2f99c70389c92391dd6549ec658521f5a23a779fc6229a7b67e35",
    "stdout_sha256": "9f527b3f5abe7c279719e00873f1dea49fc310478050fe0df600e81aba22c664",
}
_EXPECTED_SPARQL_FAILURES = {
    "test_sparql_file_exists": "AssertionError",
    "test_sparql_runs_without_error": "FileNotFoundError",
    "test_sparql_query_results": "FileNotFoundError",
}
_REVIEWED_MERGE_BASELINE = {
    "task_id": "merge-diff-arc-agi-task",
    "qualification_sha256": "c682dcd8017a3c8615f7ff49410812d0927382ffe4eaaa4abb1d4ce563a8e53f",
    "ctrf_sha256": "193ad0ced8ce3a9dd4f3c5dc6e0bfeae882b1bcfd48515ecd69c27682a9797c2",
    "stdout_sha256": "37fe3b18bb1c83cc09d3986e0736632db70a95fc8480666707817327568137d0",
}
_EXPECTED_MERGE_FAILURES = {
    "test_git_repo_init": "AssertionError",
    "test_both_bundles_fetched": "FileNotFoundError",
    "test_algo_py_exists": "AssertionError",
    "test_algo_py_has_map_function": "FileNotFoundError",
    "test_algo_map_function_with_test_input": "FileNotFoundError",
}
_MERGE_MISSING_ARTIFACTS = ["/app/repo", "/app/repo/.git", "/app/repo/algo.py"]


def _private_review(path):
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Baseline failure review requires an absolute unlinked private file")
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError("Baseline failure review must be a same-owner private regular file")
    raw = read(path, 1024**2)
    value = json.loads(raw, object_pairs_hook=strict_object,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite review JSON")))
    if not isinstance(value, dict):
        raise ValueError("Baseline failure review must be a JSON object")
    return value, digest(raw)


def _review_baseline_failure(path, root, evidence, report, qualification_sha, ctrf_sha):
    """Bind one explicit original-log review; never reinterpret episode rewards."""
    if evidence["task_id"] == "sparql-university":
        pins, failures, label = _REVIEWED_SPARQL_BASELINE, _EXPECTED_SPARQL_FAILURES, "SPARQL"
        missing_key, missing = "missing_artifact", "/app/solution.sparql"
        markers = ("AssertionError", "FileNotFoundError", missing)
    elif evidence["task_id"] == "merge-diff-arc-agi-task":
        pins, failures, label = _REVIEWED_MERGE_BASELINE, _EXPECTED_MERGE_FAILURES, "Merge"
        missing_key, missing = "missing_artifacts", _MERGE_MISSING_ARTIFACTS
        markers = ("AssertionError", "FileNotFoundError", *missing)
    else:
        raise ValueError("No explicit baseline failure review is defined for this task")
    review, review_sha = _private_review(path)
    stdout = read(root / "verifier/test-stdout.txt", 16 * 1024**2)
    hashes = {"task_id": evidence["task_id"], "qualification_sha256": qualification_sha,
              "ctrf_sha256": ctrf_sha, "stdout_sha256": digest(stdout)}
    if hashes != pins or any(review.get(key) != value for key, value in hashes.items()):
        raise ValueError("Explicit failure review differs from the pinned original " + label + " baseline")
    if (type(review.get("schema_version")) is not int or review["schema_version"] != 1 or
            review.get("kind") != "original-bootstrap-failure-review" or review.get("review_only") is not True or
            review.get("model_score", "missing") is not None or review.get("qualification_directory") != str(root) or
            review.get("failure_scope") != "expected-unsolved-task-output" or
            review.get(missing_key) != missing or review.get("expected_failures") != failures or
            review.get("original_verifier_terminal") != evidence["original_verifier_terminal"]):
        raise ValueError("Original " + label + " failure review scope or terminal evidence differs")
    failed = [test for test in report["results"]["tests"] if test["status"] == "failed"]
    names = [str(test.get("name", "")).rsplit("::", 1)[-1] for test in failed]
    text = stdout.decode("utf-8")
    if (len(failed) != len(failures) or set(names) != set(failures) or
            any(test.get("raw_status") != "call_failed" for test in failed) or
            any(name not in text for name in failures) or any(marker not in text for marker in markers)):
        raise ValueError("Original verifier logs do not contain exactly the reviewed unsolved-artifact failures")
    cleanup_raw = read(root / "pouch-environment.json", 1024**2)
    cleanup = json.loads(cleanup_raw, object_pairs_hook=strict_object)
    container = cleanup.get("container_id")
    if (review.get("cleanup_sha256") != digest(cleanup_raw) or cleanup.get("provider") != "pouch" or
            any(cleanup.get(key) is not True for key in ("started", "stopped", "deleted", "cleanup_verified")) or
            cleanup.get("network_mode") != "none" or cleanup.get("public_network_explicitly_allowed") is not False or
            not isinstance(container, str) or not re.fullmatch(r"[a-f0-9]{64}", container)):
        raise ValueError("Original reviewed qualification cleanup is incomplete or changed")
    absence, absence_sha = _private_review(review.get("container_absence_proof", ""))
    if (review.get("container_absence_proof_sha256") != absence_sha or
            type(absence.get("schema_version")) is not int or absence["schema_version"] != 1 or
            absence.get("kind") != "pouch-container-absence-observation" or absence.get("read_only") is not True or
            absence.get("hostname") != socket.gethostname() or absence.get("container_id") != container or
            absence.get("socket_path") != "/var/run/pouchd.sock" or absence.get("method") != "GET" or
            absence.get("path") != "/containers/" + container + "/json" or
            type(absence.get("http_status")) is not int or absence["http_status"] != 404):
        raise ValueError("Fresh independent original-container absence observation differs")
    observed, reviewed, now = absence.get("observed_unix"), review.get("reviewed_unix"), time.time()
    if (any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
            for value in (observed, reviewed)) or not observed <= reviewed <= now + 1 or now - observed > 43200):
        raise ValueError("Original-container absence review is stale or has invalid observation times")
    return {"bootstrap_failure_review_sha256": review_sha,
            "bootstrap_failure_review_kind": "expected-unsolved-task-output",
            "container_absence_proof_sha256": absence_sha}


def qualification_proof(directory, name, task_pin, payload_pin, *, failure_review=None):
    """Require the preserved original no-model baseline, never rerun it."""
    from .evidence_io import read_json
    from .verification import ctrf_counts
    root = Path(directory)
    if not root.is_absolute() or any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError("Real absolute bootstrap qualification directory required")
    evidence = read_json(root / "bootstrap-qualification.json", 1024**2)
    if (evidence.get("task_id") != name or evidence.get("task_provenance_sha256") != task_pin or
            evidence.get("payload_manifest_sha256") != payload_pin):
        raise ValueError("Original bootstrap qualification identity or asset pins differ")
    if any(evidence.get(key) is not False for key in ("model_evaluation", "agent_executed", "oracle_executed")):
        raise ValueError("Qualification must be the original no-model baseline")
    if (type(evidence.get("attempts")) is not int or evidence["attempts"] != 1 or
            type(evidence.get("retries")) is not int or evidence["retries"] != 0 or
            evidence.get("bootstrap_check_status") != "native_tools_and_uvx_plugin_passed" or
            evidence.get("cleanup_verified") is not True or evidence.get("failure_type") is not None):
        raise ValueError("Original bootstrap or cleanup qualification is incomplete")
    terminal = evidence.get("original_verifier_terminal")
    if (not isinstance(terminal, dict) or terminal.get("terminal_proven_by_provider") is not True or
            type(terminal.get("return_code")) is not int):
        raise ValueError("Original qualification verifier termination is unproven")
    raw = read(root / "verifier/ctrf.json", 16 * 1024**2)
    snapshot = evidence.get("ctrf")
    if (not isinstance(snapshot, dict) or snapshot.get("sha256") != digest(raw) or
            snapshot.get("bytes") != len(raw)):
        raise ValueError("Original qualification CTRF provenance differs")
    report = json.loads(raw, object_pairs_hook=strict_object)
    counts = ctrf_counts(report)
    verifier = evidence.get("verifier_result")
    rewards = verifier.get("rewards") if isinstance(verifier, dict) else None
    reward = rewards.get("reward") if isinstance(rewards, dict) else None
    if type(reward) not in (int, float) or reward != (0 if counts["failed"] else 1):
        raise ValueError("Original qualification reward and CTRF outcomes disagree")
    qualification_sha = digest(read(root / "bootstrap-qualification.json", 1024**2))
    reviewed = _review_baseline_failure(failure_review, root, evidence, report,
        qualification_sha, digest(raw)) if failure_review is not None else {}
    if not reviewed and any(test["status"] == "failed" and (test.get("raw_status") != "call_failed" or
            "assertion error" not in str(test.get("message", ""))) for test in report["results"]["tests"]):
        raise ValueError("Baseline verifier failures require original log review")
    return {"task_id": name, "qualification_sha256": qualification_sha,
            "ctrf_sha256": digest(raw), "original_baseline_reward": reward, "model_score": None,
            "terminal_proven_by_provider": True, "cleanup_verified": True, **reviewed}

def checked_task(copy_root, provenance_sha256, name, allowed_tasks=TASKS):
    from harbor.models.task.config import NetworkMode, VerifierEnvironmentMode
    from harbor.models.task.task import Task
    from harbor.models.trial.config import AgentConfig, EnvironmentConfig
    from harbor.trial.network_policy import resolve_trial_network_plan
    root = Path(copy_root)
    if not root.is_absolute() or root.is_symlink() or name not in allowed_tasks:
        raise ValueError("Select an absolute pinned four-task copy")
    raw = read(root / "provenance.json", 1024**2)
    if digest(raw) != provenance_sha256:
        raise ValueError("Task-copy provenance pin differs")
    provenance = json.loads(raw, object_pairs_hook=strict_object)
    rows = [row for row in provenance["files"] if row["path"].startswith(name + "/")]
    task_dir = root / "tasks" / name
    if not rows or task_dir.is_symlink() or any(path.is_symlink() for path in task_dir.rglob("*")):
        raise ValueError("Missing or linked task copy")
    if {str(path.relative_to(root / "tasks")) for path in task_dir.rglob("*") if path.is_file()} != {
            row["path"] for row in rows}:
        raise ValueError("Task copy member set differs")
    for row in rows:
        raw = read(root / "tasks" / row["path"], 16 * 1024**2, allow_empty=True)
        if len(raw) != row["effective_bytes"] or digest(raw) != row["effective_sha256"]:
            raise ValueError("Task copy hash differs")
        if not row["network_config_changed"] and row["source_sha256"] != row["effective_sha256"]:
            raise ValueError("Original non-policy source was changed")
    task = Task(task_dir)
    plan = resolve_trial_network_plan(task.config, AgentConfig(), EnvironmentConfig(), None,
                                     verifier_mode=VerifierEnvironmentMode.SHARED)
    for policy in (plan.agent_env_baseline, plan.agent_phase, plan.verifier_phase):
        if policy.network_mode != NetworkMode.NO_NETWORK or policy.allowed_hosts:
            raise ValueError("Every qualification phase must remain no-network")
    return task, plan
