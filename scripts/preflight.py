"""Read-only host/checkpoint inventory before allocating training memory.

Uses only the standard library. It never loads weights or imports CUDA.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def command(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=20)
        return {"returncode": result.returncode, "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


def checkpoint(path):
    root = Path(path)
    result = {"path": str(root), "exists": root.is_dir(), "issues": []}
    if not root.is_dir():
        result["issues"].append("model directory is missing or inaccessible")
        return result
    for filename in ("config.json", "tokenizer.json", "model.safetensors.index.json"):
        result[filename] = (root / filename).is_file()
    if not result["config.json"]:
        result["issues"].append("config.json missing")
    else:
        try:
            raw = json.loads((root / "config.json").read_text())
            config = raw.get("text_config", raw)
            keys = ("model_type", "architectures", "vocab_size", "hidden_size",
                    "num_hidden_layers", "num_experts", "n_routed_experts",
                    "num_experts_per_tok", "torch_dtype", "dtype", "quantization_config",
                    "engram_layer_ids", "engram_num_embeddings", "engram_head_dim")
            result["config"] = {key: config[key] for key in keys if key in config}
            for key in ("architectures", "dtype", "torch_dtype", "quantization_config"):
                if key not in result["config"] and key in raw:
                    result["config"][key] = raw[key]
            rows = config.get("engram_num_embeddings", [])
            width = config.get("engram_head_dim")
            if rows and width:
                count = sum(rows) * width
                result["engram_parameters"] = count
                result["engram_bf16_gib_global"] = count * 2 / 2**30
        except (ValueError, OSError, TypeError) as exc:
            result["issues"].append(f"cannot read config: {exc}")
    if result["model.safetensors.index.json"]:
        try:
            index = json.loads((root / "model.safetensors.index.json").read_text())
            shards = sorted(set(index["weight_map"].values()))
            missing = [name for name in shards if not (root / name).is_file()]
            result.update(shard_count=len(shards), missing_shards=missing,
                          checkpoint_size_bytes=index.get("metadata", {}).get("total_size"))
            if missing:
                result["issues"].append("checkpoint shards are missing")
        except (ValueError, OSError, KeyError, TypeError) as exc:
            result["issues"].append(f"cannot read shard index: {exc}")
    elif not (root / "model.safetensors").is_file():
        result["issues"].append("no safetensors checkpoint/index found")
    if not result["tokenizer.json"]:
        result["issues"].append("tokenizer.json missing; text SFT needs an explicit local tokenizer")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", action="append", default=[])
    parser.add_argument("--output")
    args = parser.parse_args()
    dependencies = {}
    for name in ("torch", "numpy", "tokenizers", "safetensors"):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
    disk = shutil.disk_usage(Path.cwd())
    memory = {}
    if Path("/proc/meminfo").is_file():
        memory = {line.split(":")[0]: line.split(":", 1)[1].strip()
                  for line in Path("/proc/meminfo").read_text().splitlines()
                  if line.startswith(("MemTotal:", "MemAvailable:", "SwapTotal:", "SwapFree:"))}
    paths = args.model_path or ["/mnt/fuse/Qwen/Qwen3-30B-A3B",
                              "/mnt/fuse/deepseek-ai/DeepSeek-V4.1-Flash"]
    report = {
        "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
        "cwd": str(Path.cwd()), "dependencies": dependencies,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "host_memory": memory, "disk_free_gib": disk.free / 2**30,
        "git_commit": command(["git", "rev-parse", "HEAD"]),
        "git_status": command(["git", "status", "--short"]),
        "gpus": command(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv"]),
        "gpu_processes": command(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory", "--format=csv"]),
        "models": [checkpoint(path) for path in paths],
    }
    output = json.dumps(report, indent=2)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(output + "\n")
    print(output)


if __name__ == "__main__":
    main()
