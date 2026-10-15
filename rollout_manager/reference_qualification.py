"""Bind score eligibility to native reference qualification for the exact cohort."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

COMPATIBILITY = "preserve-integer-input-dtype-v1"
WRAPPER = "from model import Model\nclass ModelNew(Model):\n    pass\n"


def read_qualification(path, cases_sha256, *, corrected):
    settings = json.loads((path / "settings.json").read_text())
    summary = json.loads((path / "summary.json").read_text())
    if (settings["cases_sha256"] != cases_sha256 or settings["candidate_wrapper"] != WRAPPER
            or settings.get("preserve_integer_inputs", False) != corrected
            or settings["seed"] != 42 or settings["correctness_trials"] != 5
            or settings["performance_trials"] != 10 or settings["generated_candidates"]
            or summary["fatal"] or summary["completed"] != summary["expected"]
            or summary["completed"] != len(summary["results"])):
        raise ValueError("incomplete or incompatible native qualification")
    records = {item["task_id"]: item for item in summary["results"]}
    if len(records) != len(summary["results"]):
        raise ValueError("duplicate qualification task")
    return records, {"root": str(path), "summary_sha256": hashlib.sha256((path / "summary.json").read_bytes()).hexdigest(),
                     "settings_sha256": hashlib.sha256((path / "settings.json").read_bytes()).hexdigest()}


def build_catalog(cases_path, baseline, correction):
    cases = json.loads(cases_path.read_text())
    cases_sha = hashlib.sha256(cases_path.read_bytes()).hexdigest()
    original, original_evidence = read_qualification(baseline, cases_sha, corrected=False)
    corrected, corrected_evidence = read_qualification(correction, cases_sha, corrected=True)
    ids = {case["task"]["task_id"] for case in cases}
    failed = {key for key, item in original.items() if not item["passed"]}
    if set(original) != ids or not failed <= set(corrected) <= ids:
        raise ValueError("native qualification does not cover the frozen cohort and its failures")
    tasks = {}
    for case in cases:
        task = case["task"]
        key = task["task_id"]
        item = corrected.get(key, original[key])
        result = item.get("result", {})
        passed = (item["passed"] and result.get("correct") is True
                  and result.get("process_group_stopped") is True
                  and result.get("gpu_contexts_released") is True)
        issue = result.get("native", {}).get("metadata", {}).get("correctness_issue")
        tasks[key] = {"source_sha256": task["source_sha256"], "score_eligible": bool(passed),
            "qualification": "correction" if key in corrected else "baseline",
            "reason": None if passed else issue or item.get("error") or "native reference failed"}
    return {"format": "native-reference-qualification-v1", "cases_sha256": cases_sha,
            "compatibility": COMPATIBILITY, "tasks": tasks,
            "evidence": {"baseline": original_evidence, "correction": corrected_evidence}}


def load_catalog(path, cases_path, cases, *, preserve_integer_inputs):
    if path is None:
        raise ValueError("full-suite evaluation requires --reference-qualification")
    catalog = json.loads(Path(path).read_text())
    if (catalog.get("format") != "native-reference-qualification-v1"
            or catalog.get("cases_sha256") != hashlib.sha256(cases_path.read_bytes()).hexdigest()
            or catalog.get("compatibility") != (COMPATIBILITY if preserve_integer_inputs else None)):
        raise ValueError("reference qualification cohort or harness policy differs")
    tasks = catalog["tasks"]
    if set(tasks) != {case["task"]["task_id"] for case in cases}:
        raise ValueError("reference qualification coverage differs")
    for case in cases:
        task = case["task"]
        item = tasks[task["task_id"]]
        if item["source_sha256"] != task["source_sha256"] or type(item["score_eligible"]) is not bool:
            raise ValueError("reference qualification source or eligibility differs")
    return catalog
