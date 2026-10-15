"""Offline pinned adapters; task bodies and large assets stay external."""
from __future__ import annotations

import copy
import hashlib
import json
import re
from importlib.resources import files
from pathlib import Path

from .contracts import EpisodeIdentity, EpisodePlan, ResourceRequirements, TaskSpec
from .adapters.datasets import DATASETS

TERMINAL_REVISION = "69671fbaac6d67a7ef0dfec016cc38a64ef7a77c"
TERMINAL_CATALOG_SHA256 = "19c6f2eea6830d217900ac188d09834b28852a1bc108985411eebe6e5d66d4c7"


def sha256_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def quantity_bytes(value):
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([KMGT]?)(?:i?B)?", str(value), re.I)
    if not match:
        raise ValueError("unsupported memory/storage quantity: " + str(value))
    return int(float(match[1]) * 1024 ** ("KMGT".find(match[2].upper()) + 1 if match[2] else 0))


class NativeDatasetAdapter:
    benchmark = ""

    def __init__(self, cases: Path | list[dict], *, executor="native-evaluator", mode="coupled"):
        self.cases = json.loads(Path(cases).read_text()) if not isinstance(cases, list) else copy.deepcopy(cases)
        self.executor, self.mode = executor, mode

    def enumerate(self):
        specs = []
        for case in self.cases:
            task = case["task"]
            if (task["dataset"] != self.benchmark or
                    task["revision"] != DATASETS[self.benchmark]["revision"] or
                    not task["task_id"].startswith(self.benchmark + "/")):
                raise ValueError("prepared native task is outside the pinned dataset")
            source = hashlib.sha256(task["code"].encode()).hexdigest()
            if source != task["source_sha256"]:
                raise ValueError("native reference-source digest differs")
            if not isinstance(case.get("request"), dict) or not isinstance(case.get("files"), dict):
                raise ValueError("prepared native case requires request and candidate files")
            specs.append(TaskSpec(self.benchmark, task["task_id"], source,
                {"kind": "native-cuda", "native_case": copy.deepcopy(case)},
                ResourceRequirements(self.executor, gpu_count=int(self.mode != "inference"),
                                     accounting="backend-stage"),
                {"correctness_trials": 5 if self.benchmark == "kernelbench" else None,
                 "verification_contract": "original-native-harness", "performance_samples": 10,
                 "development_seed": 42, "final_seed": 1042},
                {"repository": DATASETS[self.benchmark]["repository"],
                 "revision": task["revision"], "case_sha256": sha256_json(case)}))
        if not specs or len({s.task_id for s in specs}) != len(specs):
            raise ValueError("prepared cases must be nonempty with unique task identities")
        return specs

    def resolve(self, task_id):
        return next(task for task in self.enumerate() if task.task_id == task_id)

    def plan(self, run_id, *, rollouts=8, passes=1, seed=42):
        from .workloads import suite_groups
        specs = {task.task_id: task for task in self.enumerate()}
        plans = []
        for group_id, cases in suite_groups(self.cases, rollouts, passes, seed):
            for case in cases:
                r = case["rollout"]
                identity = EpisodeIdentity(run_id, self.benchmark, case["task"]["task_id"], group_id,
                    r["sample_index"], r["pass_index"],
                    trial_id=sha256_json([run_id, self.benchmark, case["task"]["task_id"],
                                         group_id, r["pass_index"], r["sample_index"]])[:32],
                    sampling_seed=r["sampling_seed"])
                plans.append(EpisodePlan(identity, specs[identity.task_id], case))
        return plans


class KernelBenchAdapter(NativeDatasetAdapter):
    benchmark = "kernelbench"


class CudaAgentAdapter(NativeDatasetAdapter):
    benchmark = "cuda-agent"


class TerminalBenchAdapter:
    def __init__(self, *, executor, task_root=None, task_ids=None):
        raw = files("rollout_manager").joinpath("data/terminal-bench-2.0.json").read_bytes()
        if hashlib.sha256(raw).hexdigest() != TERMINAL_CATALOG_SHA256:
            raise ValueError("packaged Terminal-Bench catalog digest differs")
        self.catalog = json.loads(raw)
        self.executor = executor
        self.task_root = Path(task_root) if task_root else None
        self.task_ids = list(task_ids) if task_ids is not None else [r["task_id"] for r in self.catalog["tasks"]]
        if len(set(self.task_ids)) != len(self.task_ids) or not self.task_ids:
            raise ValueError("task selection must be nonempty and unique")
        if set(self.task_ids) - {r["task_id"] for r in self.catalog["tasks"]}:
            raise ValueError("task selection is outside the pinned 89-task catalog")

    def enumerate(self):
        rows = {r["task_id"]: r for r in self.catalog["tasks"]}
        return [TaskSpec("terminal-bench", name, rows[name]["instruction_sha256"],
            {"kind": "harbor-task", "task_path": str(self.task_root / name) if self.task_root else None,
             "image": rows[name]["environment"]["docker_image"], "image_id": rows[name]["image_id"]},
            ResourceRequirements(self.executor, cpu=rows[name]["environment"]["cpus"],
                memory_bytes=quantity_bytes(rows[name]["environment"]["memory"]),
                storage_bytes=quantity_bytes(rows[name]["environment"]["storage"])),
            {"agent_seconds": rows[name]["agent"]["timeout_sec"],
             "verifier_seconds": rows[name]["verifier"]["timeout_sec"],
             "build_seconds": rows[name]["environment"]["build_timeout_sec"],
             "attempts": 1, "retries": 0, "max_turns": 128,
             "shell_seconds": 30, "model_seconds": 360, "max_output_tokens": 2048},
            {"revision": TERMINAL_REVISION, "catalog_sha256": TERMINAL_CATALOG_SHA256,
             "source_archive_sha256": self.catalog["source_archive_sha256"],
             "source_files": rows[name]["source_files"], "bootstrap_qualified": False})
            for name in self.task_ids]

    def resolve(self, task_id):
        task = next(task for task in self.enumerate() if task.task_id == task_id)
        if self.task_root is None:
            raise ValueError("resolving original assets requires an explicit external task root")
        root = self.task_root / task_id
        # Only public instruction bytes are returned. Verifier/solution bodies are never model inputs.
        raw = (root / "instruction.md").read_bytes()
        if hashlib.sha256(raw).hexdigest() != task.source_sha256:
            raise ValueError("original instruction digest differs")
        return task, raw.decode()

    def plan(self, run_id, *, max_workers, capacity):
        if type(max_workers) is not int or not 1 <= max_workers <= 64:
            raise ValueError("Terminal-Bench requires 1..64 workers")
        if capacity.executor != self.executor or capacity.gpu_count or any(
                getattr(capacity, key) <= 0 for key in ("cpu", "memory_bytes", "storage_bytes")):
            raise ValueError("Terminal-Bench needs a positive CPU-only capacity for its actual executor")
        tasks = self.enumerate()
        waves, wave, totals = [], [], [0, 0, 0]
        limits = [capacity.cpu, capacity.memory_bytes, capacity.storage_bytes]
        for task in tasks:
            demand = [task.resources.cpu, task.resources.memory_bytes, task.resources.storage_bytes]
            if any(d > limit for d, limit in zip(demand, limits)):
                raise ValueError("task exceeds CPU executor admission budget: " + task.task_id)
            if wave and (len(wave) == max_workers or any(t+d > limit for t,d,limit in zip(totals,demand,limits))):
                waves.append(wave); wave, totals = [], [0, 0, 0]
            wave.append(task); totals = [t+d for t,d in zip(totals,demand)]
        if wave:
            waves.append(wave)
        return [EpisodePlan(EpisodeIdentity(run_id, "terminal-bench", task.task_id,
                f"wave-{index}", 0, trial_id=hashlib.sha256(f"{run_id}:{task.task_id}".encode()).hexdigest()[:32]), task)
                for index, wave in enumerate(waves) for task in wave]
