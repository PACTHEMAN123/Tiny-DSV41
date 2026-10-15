from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

from ..common import save_json
from .process import gpu_processes, run_process
from .native_timing import HarnessInstrumentationError


def prepare_workspace(workspace: Path, upstream: Path, task: dict, files: dict[str, str]) -> None:
    workspace.mkdir(parents=True, exist_ok=False)
    source = upstream / "CUDA-Agent" / "agent_workdir"
    for name in ("binding.cpp", "binding_registry.h"):
        shutil.copy2(source / name, workspace / name)
    shutil.copytree(source / "utils", workspace / "utils", ignore=shutil.ignore_patterns("__pycache__"))
    (workspace / "model.py").write_text(task["code"])
    for name, content in files.items():
        path = workspace / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    save_json(workspace / "source-manifest.json", {str(p.relative_to(workspace)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in workspace.rglob("*") if p.is_file()})


class HarnessRunner:
    def __init__(self, upstream: Path, gpu_uuid: str, architecture: str, heartbeat,
                 *, instrumentation: dict | None = None, cores: list[int] | None = None, gpu_guard=None,
                 cpu_threads: int = 1, preserve_integer_inputs: bool = False, cancelled=None,
                 isolate_device_inputs: bool = False):
        self.upstream = upstream.resolve()
        self.gpu_uuid = gpu_uuid
        self.heartbeat = heartbeat
        self.instrumentation = instrumentation or {}
        self.gpu_guard = gpu_guard
        self.cancelled = cancelled
        self.env = {key: value for key, value in os.environ.items() if key in
                    ("PATH", "HOME", "LANG", "LD_LIBRARY_PATH", "VIRTUAL_ENV",
                     "HF_HOME")}
        self.env.update(CUDA_HOME="/usr/local/cuda", TORCH_CUDA_ARCH_LIST=architecture,
                        MAX_JOBS="4", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                        LITELLM_LOCAL_MODEL_COST_MAP="True", HF_HUB_OFFLINE="1")
        self.env["PATH"] = "/usr/local/cuda/bin:" + self.env.get("PATH", "/usr/bin:/bin")
        self.cores = sorted(os.sched_getaffinity(0))[-4:] if cores is None else cores
        if not self.cores or not set(self.cores) <= os.sched_getaffinity(0):
            raise ValueError("worker CPU affinity is outside the container allocation")
        if not isinstance(cpu_threads, int) or not 1 <= cpu_threads <= len(self.cores):
            raise ValueError("CPU thread count must fit the worker affinity")
        self.cpu_threads = cpu_threads
        self.preserve_integer_inputs = preserve_integer_inputs
        self.isolate_device_inputs = isolate_device_inputs
        self.env.update(OMP_NUM_THREADS=str(cpu_threads), MKL_NUM_THREADS=str(cpu_threads))
        self.env["MAX_JOBS"] = str(len(self.cores))

    def child(self, workspace: Path, name: str, argv: list[str], *, gpu: bool, timeout: float) -> dict:
        ownership_start = time.monotonic()
        if gpu:
            if self.gpu_guard is not None:
                self.gpu_guard()
            elif gpu_processes(self.gpu_uuid):
                raise RuntimeError("selected GPU is busy; refusing to overlap")
        admission_check_seconds = time.monotonic() - ownership_start
        env = dict(self.env, CUDA_VISIBLE_DEVICES=self.gpu_uuid if gpu else "",
                   TORCH_EXTENSIONS_DIR=str(workspace / "build"),
                   TORCHINDUCTOR_CACHE_DIR=str(workspace / "inductor-cache"))
        command = ["taskset", "-c", ",".join(map(str, self.cores)), sys.executable] + argv
        # CUDA-Agent's native timing uses Kineto; two CUPTI collectors cannot share a process.
        native_kineto = "rollout_manager.adapters.harness_child" in argv and "profile" in argv
        selected = not self.instrumentation.get("stages") or name in self.instrumentation["stages"]
        if self.instrumentation.get("kind") == "nsys" and not native_kineto and selected:
            env["ROLLOUT_NATIVE_NVTX"] = "1"
            command = ["nsys", "profile", "--trace=cuda,nvtx,osrt", "--sample=none", "--cpuctxsw=none",
                       "--wait=all", "--output=" + str(workspace / name)] + command
        elif self.instrumentation.get("kind") == "perf" and selected:
            command = [self.instrumentation["executable"], "record", "-F", "99", "-g",
                       "--call-graph", "dwarf,4096", "-o", str(workspace / (name + ".perf.data")),
                       "--"] + command
        result = run_process(command, workspace, env, workspace / (name + ".log"),
                             timeout=timeout, heartbeat=self.heartbeat, cancelled=self.cancelled)
        cleanup_start = time.monotonic()
        quiescent = not gpu_processes(self.gpu_uuid) if gpu else True
        released = self.gpu_guard() if gpu and self.gpu_guard is not None else quiescent
        result.update(gpu_quiescent=quiescent, gpu_contexts_released=released, gpu_uuid=self.gpu_uuid if gpu else None,
                      admission_check_seconds=admission_check_seconds,
                      context_cleanup_seconds=time.monotonic()-cleanup_start,
                      cpu_affinity=self.cores, cpu_threads=self.cpu_threads,
                      cuda_visible_devices=env["CUDA_VISIBLE_DEVICES"],
                      instrumentation=self.instrumentation, profiler_selected=selected,
                      nsys_excluded_for_native_kineto=native_kineto and self.instrumentation.get("kind") == "nsys")
        save_json(workspace / (name + "-process.json"), result)
        return result

    def compile(self, workspace: Path) -> dict:
        result = self.child(workspace, "compile", ["-m", "utils.compile"], gpu=False, timeout=240)
        result["compiled"] = result["returncode"] == 0 and (workspace / "cuda_extension.so").is_file()
        result["feedback"] = (workspace / "compile.log").read_text(errors="replace")[-12000:]
        if result["compiled"]:
            result["binary_sha256"] = hashlib.sha256((workspace / "cuda_extension.so").read_bytes()).hexdigest()
        return result

    def evaluate(self, workspace: Path, dataset: str, *, final: bool = False,
                 observe_inputs: bool = False, input_reuse: dict | None = None) -> dict:
        expected_sources = json.loads((workspace / "source-manifest.json").read_text())
        for name, expected in expected_sources.items():
            if hashlib.sha256((workspace / name).read_bytes()).hexdigest() != expected:
                raise ValueError("reviewed source or upstream harness changed: " + name)
        seed = 1042 if final else 42
        prefix = "final" if final else "evaluation"
        modes = ["kernelbench"] if dataset == "kernelbench" else ["verify"] + ([] if final else ["profile"])
        result = {"correct": False, "harness": dataset, "seed": seed, "processes": [],
                  "evidence_kind": "reviewed-development-pilot"}
        for mode in modes:
            report = workspace / f"{prefix}-{mode}.json"
            extra = []
            if mode == "kernelbench" and self.preserve_integer_inputs:
                extra.append("--preserve-integer-inputs")
            if observe_inputs:
                extra += ["--observe-inputs"]
            if self.isolate_device_inputs:
                extra += ['--isolate-device-inputs']
            consumer = None
            if input_reuse:
                from ..input_cache import CacheClient
                cache = CacheClient(input_reuse["socket"])
                consumer = cache.control("begin", copy_limit=input_reuse["consumer_copy_limit_bytes"])["consumer"]
                config = dict(input_reuse, consumer=consumer,
                              namespace="final-verification" if final else "development")
                config["qualification"] = input_reuse["qualifications"][mode]
                config_path = workspace / f"{prefix}-{mode}-input-reuse.json"
                save_json(config_path, config)
                extra += ["--input-reuse-config", str(config_path)]
            process = self.child(workspace, f"{prefix}-{mode}", ["-m", "rollout_manager.adapters.harness_child",
                mode, "--upstream", str(self.upstream), "--seed", str(seed), "--report", str(report)] + extra,
                gpu=True, timeout=240)
            if consumer is not None and process["process_group_stopped"] and process["gpu_contexts_released"]:
                cache.control("end", consumer=consumer)
            result["processes"].append(process)
            log = (workspace / f"{prefix}-{mode}.log").read_text(errors="replace")
            result["feedback"] = log[-12000:]
            if process["returncode"] == 78:
                raise HarnessInstrumentationError("native harness instrumentation failure: " + log[-2000:])
            if process["returncode"] != 0 or process["stopped_for"]:
                result["failed_mode"] = mode
                break
            native = json.loads(report.read_text())
            if mode == "kernelbench":
                result.update(correct=native.get("correctness") is True, native=native)
            elif mode == "verify":
                result["correct"] = native.get("completed") is True and "[PASS] verify success" in log
            else:
                match = re.search(r"Torch Baseline: ([0-9.]+)us, Torch Compile: ([0-9.]+)us, CUDA Extension: ([0-9.]+)us", log)
                if match:
                    result["timing"] = dict(zip(("torch_eager_us", "torch_compile_us", "candidate_us"), map(float, match.groups())))
                    result["timing_method"] = "upstream CUDA profiler event sum / 10; warmup 5"
                else:
                    raise ValueError("upstream profiler completed without its expected timing record")
            if not result["correct"]:
                break
        result["process_group_stopped"] = all(p["process_group_stopped"] for p in result["processes"])
        result["gpu_quiescent"] = all(p["gpu_quiescent"] for p in result["processes"])
        result["gpu_contexts_released"] = all(p["gpu_contexts_released"] for p in result["processes"])
        save_json(workspace / f"{prefix}-summary.json", result)
        return result
