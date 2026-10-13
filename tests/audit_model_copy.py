"""Verify an independent checkpoint copy without opening any source file for writing."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8*1024**2), b""):
            digest.update(block)
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--copy", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline")
    args = parser.parse_args()
    source, copy = Path(args.source).resolve(), Path(args.copy).resolve()
    if source == copy or source in copy.parents:
        raise ValueError("copy must be an independent directory")
    index = json.loads((source / "model.safetensors.index.json").read_text())
    names = sorted(set(index["weight_map"].values()) | {
        "config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json"})

    def check(name):
        origin, local = source/name, copy/name
        if local.is_symlink() or local.samefile(origin):
            raise ValueError(f"copy aliases shared file: {name}")
        original, duplicate = fingerprint(origin), fingerprint(local)
        if (original["size"], original["sha256"]) != (duplicate["size"], duplicate["sha256"]):
            raise ValueError(f"copy mismatch: {name}")
        return name, original

    with ThreadPoolExecutor(max_workers=4) as executor:
        files = dict(executor.map(check, names))
    result = {"source": str(source), "copy": str(copy), "files": files, "independent_copy": True}
    if args.baseline:
        baseline = json.loads(Path(args.baseline).read_text())
        if baseline["files"] != files:
            raise ValueError("shared checkpoint changed since the initial audit")
        result["shared_unchanged"] = True
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"checked_files": len(files), "independent_copy": True,
                      "shared_unchanged": result.get("shared_unchanged")}), flush=True)


if __name__ == "__main__":
    main()
