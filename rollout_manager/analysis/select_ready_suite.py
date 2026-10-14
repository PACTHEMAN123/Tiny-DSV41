"""Stage hash-pinned mounted assets and freeze a reproducible ready-task cohort."""
import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

from rollout_manager.workloads import model_dependency


def file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024**2):
            digest.update(block)
    return digest.hexdigest()


def stage(model, assets, mounts):
    name, revision = model["model"], model["revision"]
    mounted = mounts / ("openai-community/gpt2" if name == "gpt2" else name)
    cache = assets / "hf-home/hub" / ("models--" + name.replace("/", "--"))
    snapshot = cache / "snapshots" / revision
    record = {"model": name, "revision": revision, "mount": str(mounted), "ready": False,
              "files": [], "started_unix": time.time(), "origin": "provider mount or verified uploaded cache"}
    for item in model["files"]:
        target = snapshot / item["name"]
        source = target if target.is_file() else mounted / item["name"]
        if not source.is_file() or source.stat().st_size != item["bytes"]:
            record["reason"] = "missing or incomplete " + item["name"]
            return record
    for item in model["files"]:
        target = snapshot / item["name"]
        source = target if target.is_file() else mounted / item["name"]
        start = time.monotonic()
        target.parent.mkdir(parents=True, exist_ok=True)
        if source == target:
            digest = file_sha(target)
        else:
            digest = hashlib.sha256()
            temporary = None
            try:
                with source.open("rb") as incoming, tempfile.NamedTemporaryFile(
                        dir=target.parent, prefix=target.name + ".mount-copy-", delete=False) as output:
                    temporary = Path(output.name)
                    while block := incoming.read(8 * 1024**2):
                        output.write(block)
                        digest.update(block)
                    output.flush()
                    os.fsync(output.fileno())
                if digest.hexdigest() == item["sha256"]:
                    temporary.replace(target)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
            digest = digest.hexdigest()
        if digest != item["sha256"]:
            record["reason"] = "pinned digest mismatch for " + item["name"]
            return record
        record["files"].append({"name": item["name"], "bytes": item["bytes"], "sha256": digest,
                                "stage_seconds": time.monotonic() - start})
    (cache / "refs").mkdir(exist_ok=True)
    (cache / "refs/main").write_text(revision)
    record.update(ready=True, finished_unix=time.time())
    return record


def select(cases, ready):
    selected, deferred = [], []
    for case in cases:
        dependency = model_dependency(case["task"]["code"])
        if dependency is None or dependency in ready:
            selected.append(case)
        else:
            deferred.append({"task_id": case["task"]["task_id"], "model": dependency,
                             "reason": "model was not ready in this cohort snapshot"})
    return selected, deferred


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--mounts", type=Path, default=Path("/mnt/fuse"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=False)
    records = []
    for model in json.loads((args.assets / "model-assets.json").read_text()):
        try:
            record = stage(model, args.assets, args.mounts)
        except Exception as exc:
            record = {"model": model["model"], "ready": False, "reason": str(exc)}
        records.append(record)
        print(json.dumps(record), flush=True)
        (args.root / "asset-readiness.json").write_text(json.dumps(records, indent=2) + "\n")
    cases = json.loads(args.cases.read_text())
    selected, deferred = select(cases, {r["model"] for r in records if r["ready"]})
    output = args.root / "cases.json"
    output.write_text(json.dumps(selected, indent=2) + "\n")
    manifest = {"source_cases_sha256": file_sha(args.cases), "cases_sha256": file_sha(output),
        "total_tasks": len(cases), "selected_tasks": len(selected), "deferred_tasks": deferred,
        "counts": dict(Counter(c["task"]["split"] for c in selected)), "rollouts_per_task": 8,
        "planned_episodes": len(selected)*8,
        "selection": "all tasks with no model dependency or a hash-verified ready model; freeze across comparison arms",
        "model_loading": "HF_HUB_OFFLINE=1; staged from provider mounts; no remote downloads"}
    (args.root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
