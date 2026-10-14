"""Bind tools-only comparisons to exact, completed real inference responses."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from .adapters.candidates import parse_candidate
from .common import digest
from .workloads import suite_groups


def identity(task, rollout):
    return task, rollout["pass_index"], rollout["sample_index"], rollout["sampling_seed"]


def validate_outcome(item):
    try:
        files = parse_candidate(item["content"])
    except (ValueError, SyntaxError) as exc:
        if item.get("candidate_format_error") != str(exc) or "candidate_sha256" in item:
            raise ValueError("frozen format outcome differs from its response") from exc
    else:
        if item.get("candidate_sha256") != digest(files) or "candidate_format_error" in item:
            raise ValueError("frozen candidate digest differs from its response")


def freeze(source: Path, cases_path: Path, *, allow_partial=False):
    settings = json.loads((source / "settings.json").read_text())
    if settings["mode"] != "inference" or settings["schedule"] != "full-suite" or settings["turns"] != 1:
        raise ValueError("source must be a one-turn full-suite inference collection")
    cases = json.loads(cases_path.read_text())
    groups = suite_groups(cases, settings["rollouts_per_task"], settings["repeats"], settings["sampling_seed"])
    expected = {identity(case["task"]["task_id"], case["rollout"]): case for _, samples in groups for case in samples}
    observed = {}
    for path in sorted(source.glob("jobs/*/result.json")):
        result = json.loads(path.read_text())
        key = identity(result["task"], result["rollout"])
        if key not in expected or key in observed:
            raise ValueError("unexpected or duplicate source rollout: " + str(key))
        # An uncertain RPC must never become a replayed candidate failure.
        if result.get("communication_failure") or result.get("error") or not result.get("completed"):
            raise ValueError("source contains an incomplete or infrastructure-failed request: " + str(path))
        case = expected[key]
        dispatch = json.loads((path.parent / "dispatch.json").read_text())
        if dispatch["request"] != case["request"] or result["rollout"] != case["rollout"]:
            raise ValueError("source request or GRPO metadata differs from the planned workload")
        inference = json.loads((path.parent / "inference.json").read_text())
        content = inference["response"]["choices"][0]["message"]["content"]
        item = {"task": result["task"], "rollout": result["rollout"],
                "request_sha256": digest(case["request"]), "content": content,
                "source": {"run": source.name, "job": result["job"],
                           "result_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                           "inference_sha256": hashlib.sha256((path.parent / "inference.json").read_bytes()).hexdigest(),
                           "stages": result["stages"], "communication": result.get("communication"),
                           "usage": result.get("usage"), "finish_reason": result.get("finish_reason")}}
        try:
            files = parse_candidate(content)
        except (ValueError, SyntaxError) as exc:
            if not result.get("candidate_format_error") or (path.parent / "candidate.json").exists():
                raise ValueError("source candidate-format outcome disagrees with response") from exc
            item["candidate_format_error"] = str(exc)
        else:
            if digest(files) != result.get("candidate_sha256") or files != json.loads((path.parent / "candidate.json").read_text()):
                raise ValueError("source candidate digest/content disagrees with response")
            item["candidate_sha256"] = digest(files)
        observed[key] = item
    complete = [[observed[identity(case["task"]["task_id"], case["rollout"])] for case in samples]
                for _, samples in groups if all(identity(case["task"]["task_id"], case["rollout"]) in observed for case in samples)]
    if not complete or (len(complete) != len(groups) and not allow_partial):
        raise ValueError("source does not cover every planned task group; partial replay is qualification only")
    return {"schema": 1, "measurement": "tools_only_frozen_real_inference",
            "cases_sha256": hashlib.sha256(cases_path.read_bytes()).hexdigest(),
            "settings_sha256": hashlib.sha256((source / "settings.json").read_bytes()).hexdigest(),
            "rollouts_per_task": settings["rollouts_per_task"], "passes": settings["repeats"],
            "sampling_seed": settings["sampling_seed"], "expected_groups": len(groups),
            "expected_episodes": len(expected), "partial_qualification": len(complete) != len(groups),
            "observed_responses": len(observed), "included_groups": len(complete),
            "groups": complete}


def load(path, cases_path, planned_groups, *, allow_partial=False):
    frozen = json.loads(path.read_text())
    if frozen.get("schema") != 1 or frozen.get("measurement") != "tools_only_frozen_real_inference":
        raise ValueError("unknown frozen-rollout manifest")
    if frozen["cases_sha256"] != hashlib.sha256(cases_path.read_bytes()).hexdigest():
        raise ValueError("frozen rollout cohort digest differs")
    declared = suite_groups(json.loads(cases_path.read_text()), frozen["rollouts_per_task"],
                            frozen["passes"], frozen["sampling_seed"])
    if declared != planned_groups:
        raise ValueError("frozen sampling settings differ from the planned workload")
    expected = {group: samples for group, samples in planned_groups}
    loaded = {}
    for samples in frozen["groups"]:
        if not samples:
            raise ValueError("empty frozen group")
        group = samples[0]["rollout"]["group_id"]
        if group not in expected or group in loaded or len(samples) != len(expected[group]):
            raise ValueError("duplicate, unknown or incomplete frozen group")
        bound = []
        for item, planned in zip(samples, expected[group]):
            if (item["task"] != planned["task"]["task_id"] or item["rollout"] != planned["rollout"]
                    or item["request_sha256"] != digest(planned["request"])):
                raise ValueError("frozen sample does not match its task, seed or request")
            validate_outcome(item)
            case = copy.deepcopy(planned)
            case["frozen_rollout"] = item
            bound.append(case)
        loaded[group] = bound
    partial = len(loaded) != len(expected)
    if (not loaded or partial != frozen["partial_qualification"] or frozen["expected_groups"] != len(expected)
            or frozen["expected_episodes"] != sum(len(samples) for samples in expected.values())
            or frozen["included_groups"] != len(loaded)
            or not sum(len(samples) for samples in loaded.values()) <= frozen["observed_responses"] <= frozen["expected_episodes"]):
        raise ValueError("frozen coverage metadata is inconsistent")
    if partial and not allow_partial:
        raise ValueError("partial frozen replay requires explicit qualification mode")
    metadata = {k: v for k, v in frozen.items() if k != "groups"}
    metadata["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return [(group, loaded[group]) for group, _ in planned_groups if group in loaded], metadata
