"""Finite task-group scheduling for reproducible GRPO rollout evaluation."""
from __future__ import annotations

import ast
import copy
import hashlib
import random
import threading
from collections import deque


def model_dependency(source):
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and node.func.attr == "from_pretrained"]
    if not calls:
        return None
    values = [node.value.value for node in tree.body if isinstance(node, ast.Assign)
              and any(isinstance(target, ast.Name) and target.id == "model_name" for target in node.targets)
              and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)]
    if len(values) != 1:
        raise ValueError("unresolved pretrained-model dependency")
    return values[0]


def suite_groups(cases, rollouts, passes, seed):
    ids = [case["task"]["task_id"] for case in cases]
    if not cases or len(set(ids)) != len(ids):
        raise ValueError("full-suite cases must have unique task IDs")
    if min(rollouts, passes) < 1:
        raise ValueError("rollout and pass counts must be positive")
    groups = []
    for repeat in range(passes):
        order = list(range(len(cases)))
        random.Random(seed + repeat).shuffle(order)
        for ordinal in order:
            case = cases[ordinal]
            group = f"p{repeat}-task{ordinal}"
            samples = []
            for sample in range(rollouts):
                item = copy.deepcopy(case)
                # Stable across resource configurations and scheduling order.
                identity = f"{seed}:{repeat}:{case['task']['task_id']}:{sample}"
                sampling_seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:4], "big")
                item["request"]["seed"] = sampling_seed
                item["rollout"] = {"group_id": group, "pass_index": repeat,
                    "sample_index": sample, "group_size": rollouts, "sampling_seed": sampling_seed}
                samples.append(item)
            groups.append((group, samples))
    return groups


def external_schedule_coverage(groups, results):
    """Report externally admitted samples without running another scheduler."""
    expected = {group: samples for group, samples in groups}
    observed = {group: [] for group in expected}
    seen = set()
    for result in results:
        rollout = result["rollout"]
        group = rollout["group_id"]
        identity = (group, rollout["sample_index"])
        if group not in expected or identity in seen:
            raise ValueError("unexpected or duplicate externally scheduled sample")
        case = next((case for case in expected[group]
                     if case["rollout"]["sample_index"] == rollout["sample_index"]), None)
        if case is None or result["task"] != case["task"]["task_id"] or rollout != case["rollout"]:
            raise ValueError("external driver changed pinned sample identity")
        seen.add(identity)
        observed[group].append(result)
    completed, incomplete, pending = [], {}, 0
    for group, samples in groups:
        records = observed[group]
        if not records:
            pending += 1
        elif len(records) < len(samples):
            incomplete[group] = {"remaining": len(samples) - len(records), "finished": len(records)}
        else:
            first = min(record["started_unix_ns"] for record in records)
            last = max(record["finished_unix_ns"] for record in records)
            completed.append({"group_id": group, "task": samples[0]["task"]["task_id"],
                "pass_index": samples[0]["rollout"]["pass_index"], "episodes": len(records),
                "verified": sum(bool(record.get("verified_correct", record.get("final_correct")))
                                for record in records),
                "group_wall_seconds": (last-first)/1e9,
                "completion_spread_seconds": (last-min(record["finished_unix_ns"] for record in records))/1e9,
                "jobs": [record["job"] for record in records]})
    return {"expected_groups": len(groups), "expected_episodes": sum(len(samples) for _, samples in groups),
            "completed_groups": len(completed), "pending_groups": pending,
            "incomplete_groups": incomplete, "groups": completed}


class GroupSchedule:
    """Bound active groups independently from the caller's episode worker pool."""

    def __init__(self, groups, max_groups):
        if max_groups < 1:
            raise ValueError("at least one active group is required")
        self.pending = deque(groups)
        self.ready = deque()
        self.active = {}
        self.finished = []
        self.max_groups = max_groups
        self.condition = threading.Condition()
        self.expected_groups = len(groups)
        self.expected_episodes = sum(len(samples) for _, samples in groups)

    def take(self, cancelled):
        with self.condition:
            while not cancelled():
                while self.pending and len(self.active) < self.max_groups:
                    group, samples = self.pending.popleft()
                    self.active[group] = {"remaining": len(samples), "results": []}
                    self.ready.extend(samples)
                if self.ready:
                    return self.ready.popleft()
                if not self.active:
                    return None
                self.condition.wait(.2)
        return None

    def complete(self, case, result):
        with self.condition:
            group = case["rollout"]["group_id"]
            item = self.active[group]
            item["results"].append(result)
            item["remaining"] -= 1
            if item["remaining"] == 0:
                self.finished.append({"group_id": group, "task": case["task"]["task_id"],
                    "pass_index": case["rollout"]["pass_index"], "results": item["results"]})
                del self.active[group]
            self.condition.notify_all()

    def summary(self):
        with self.condition:
            groups = []
            for item in self.finished:
                results = item["results"]
                first = min(r["started_unix_ns"] for r in results)
                last = max(r["finished_unix_ns"] for r in results)
                groups.append({k: v for k, v in item.items() if k != "results"} | {
                    "episodes": len(results), "verified": sum(bool(r.get("verified_correct", r.get("final_correct"))) for r in results),
                    "group_wall_seconds": (last-first)/1e9,
                    "completion_spread_seconds": (last-min(r["finished_unix_ns"] for r in results))/1e9,
                    "jobs": [r["job"] for r in results]})
            return {"expected_groups": self.expected_groups, "expected_episodes": self.expected_episodes,
                "completed_groups": len(groups), "pending_groups": len(self.pending),
                "incomplete_groups": {k: {"remaining": v["remaining"], "finished": len(v["results"])}
                                      for k, v in self.active.items()}, "groups": groups}
