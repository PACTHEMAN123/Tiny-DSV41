from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path


DATASETS = {
    "cuda-agent": {
        "repository": "BytedTsinghua-SIA/CUDA-Agent-Ops-6K",
        "revision": "44a734c78c947bfcba5189cbfd13f57a6d29a698",
        "files": ["README.md", "data.parquet"],
    },
    "kernelbench": {
        "repository": "ScalingIntelligence/KernelBench",
        "revision": "ca1464e5d4723fd14f87b8e68306aaa1fe66b81b",
        "files": ["README.md"] + [f"data/level_{i}-00000-of-00001.parquet" for i in range(1, 5)],
    },
}


@dataclass(frozen=True)
class Task:
    task_id: str
    dataset: str
    revision: str
    split: str
    row: int
    name: str
    code: str
    source_sha256: str
    metadata: dict


def _download(url: str, destination: Path, limit: int = 32 * 1024 * 1024) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        descriptor, temporary = tempfile.mkstemp(dir=destination.parent, prefix=".download-")
        try:
            with os.fdopen(descriptor, "wb") as output, urllib.request.urlopen(url, timeout=60) as response:
                count = 0
                while block := response.read(1024 * 1024):
                    count += len(block)
                    if count > limit:
                        raise ValueError("dataset object exceeds the download bound")
                    output.write(block)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    data = destination.read_bytes()
    if len(data) > limit:
        raise ValueError("dataset object exceeds the download bound")
    return {"url": url, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def download_datasets(root: Path) -> dict:
    manifest = {}
    for key, source in DATASETS.items():
        base = root / key / source["revision"]
        manifest[key] = {"repository": source["repository"], "revision": source["revision"], "files": {}}
        for name in source["files"]:
            url = f"https://huggingface.co/datasets/{source['repository']}/resolve/{source['revision']}/{name}"
            manifest[key]["files"][name] = _download(url, base / name)
        manifest[key]["rows"] = len(load_tasks(root, key))
    output = root / "manifest.json"
    encoded = json.dumps(manifest, indent=2) + "\n"
    if output.exists() and output.read_text() != encoded:
        raise ValueError("existing dataset manifest differs; use a new data directory")
    output.write_text(encoded)
    return manifest


def load_tasks(root: Path, dataset: str) -> list[Task]:
    import pyarrow.parquet as parquet

    source = DATASETS[dataset]
    tasks = []
    for name in source["files"]:
        if not name.endswith(".parquet"):
            continue
        rows = parquet.read_table(root / dataset / source["revision"] / name).to_pylist()
        split = "train" if dataset == "cuda-agent" else Path(name).stem.split("-")[0]
        for index, row in enumerate(rows):
            code = row["code"]
            if not isinstance(code, str) or not code or len(code.encode()) > 262144:
                raise ValueError("invalid reference source")
            task_id = f"{dataset}/{split}/{row.get('problem_id', index)}"
            tasks.append(Task(task_id, dataset, source["revision"], split, index,
                              row.get("name", f"row-{index}"), code, hashlib.sha256(code.encode()).hexdigest(),
                              {k: v for k, v in row.items() if k != "code"}))
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Download immutable public task datasets without executing their code")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--select", help="Export one normalized task ID as JSON to stdout")
    args = parser.parse_args()
    if args.select:
        dataset = args.select.split("/", 1)[0]
        match = next(t for t in load_tasks(args.root, dataset) if t.task_id == args.select)
        print(json.dumps(asdict(match), indent=2))
    else:
        print(json.dumps(download_datasets(args.root), indent=2))


if __name__ == "__main__":
    main()
