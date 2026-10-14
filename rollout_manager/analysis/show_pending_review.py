"""Show every changed line against an explicitly reviewed bundle; never approve."""
import argparse
import difflib
import json
import sys
from pathlib import Path

from rollout_manager.common import digest


def changes(before, after):
    return "".join(line for name in sorted(set(before) | set(after))
        for line in difflib.unified_diff(before.get(name, "").splitlines(keepends=True),
            after.get(name, "").splitlines(keepends=True),
            fromfile="reviewed/" + name, tofile="candidate/" + name))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--run", required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    if not 1 <= args.limit <= 20 or args.offset < 0 or Path(args.run).name != args.run:
        parser.error("use a single run-directory name and bounded review selection")
    known = {sha for sha, entry in json.loads(args.catalog.read_text()).items()
             if entry.get("scope") == "reviewed-development-pilot"}
    reviewed = {}
    for path in args.root.glob("*/jobs/*/candidate.json"):
        files = json.loads(path.read_text())
        sha = digest(files)
        if sha in known:
            reviewed[sha] = files
    for case in json.loads((args.root / "cases.json").read_text()):
        sha = digest(case["files"])
        if sha in known:
            reviewed[sha] = case["files"]
    pending = {}
    for path in (args.root / args.run).glob("jobs/*/candidate.json"):
        files = json.loads(path.read_text())
        sha = digest(files)
        if sha not in known:
            pending[sha] = (path, files)
    selected = []
    print("UNREVIEWED_BUNDLES", len(pending))
    for sha, (path, files) in sorted(pending.items())[args.offset:args.offset + args.limit]:
        if not reviewed:
            raise ValueError("no reviewed baseline is available; inspect the complete source separately")
        alternatives = [(changes(old, files), old_sha) for old_sha, old in reviewed.items()]
        difference, base_sha = min(alternatives, key=lambda item: len(item[0]))
        record = {"sha256": sha, "path": str(path), "base_sha256": base_sha}
        print("CANDIDATE", json.dumps(record))
        print(difference)
        selected.append(record)
    if args.manifest:
        args.manifest.write_text(json.dumps({"status": "not_approved", "candidates": selected}, indent=2) + "\n")


if __name__ == "__main__":
    main()
