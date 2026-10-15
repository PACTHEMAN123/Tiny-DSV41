"""Assemble explicit exact-source review decisions; never infer an approval."""
import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

from rollout_manager.common import digest


def build_catalog(source_roots, review_paths):
    sources = {digest(json.loads(path.read_text()))
               for root in source_roots for path in root.glob("jobs/*/candidate.json")}
    catalog = {}
    for path in sorted(review_paths):
        review = json.loads(path.read_text())
        if review.get("scope") != "reviewed-development-pilot" or not review.get("review_method"):
            raise ValueError("explicit review evidence is required: " + str(path))
        decisions = [(sha, "approve", None) for sha in review.get("approved_sha256", [])]
        decisions += [(sha, "reject", reason) for sha, reason in review.get("rejected", {}).items()]
        for sha, decision, reason in decisions:
            if sha not in sources:
                raise ValueError("review digest has no matching source bundle: " + sha)
            if decision == "reject" and (not isinstance(reason, str) or not reason.strip()):
                raise ValueError("a rejection requires an explicit reason")
            entry = {"scope": review["scope"], "decision": decision}
            if reason is not None:
                entry["reason"] = reason
            previous = catalog.get(sha)
            if previous is not None and {k: v for k, v in previous.items() if k != "evidence"} != entry:
                raise ValueError("conflicting review decisions: " + sha)
            evidence = {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            entry["evidence"] = (previous or {}).get("evidence", []) + [evidence]
            catalog[sha] = entry
    return catalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, nargs="+", required=True)
    parser.add_argument("--reviews", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    catalog = build_catalog(args.sources, args.reviews)
    if args.output.exists():
        previous = json.loads(args.output.read_text())
        for sha, entry in previous.items():
            if sha not in catalog or any(catalog[sha].get(k) != entry.get(k)
                                        for k in ("scope", "decision", "reason")):
                raise ValueError("catalog update would remove or change a decision: " + sha)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=args.output.parent, delete=False) as temporary:
        json.dump(catalog, temporary, indent=2)
        temporary.write("\n")
    os.replace(temporary.name, args.output)
    print(json.dumps({"reviewed": len(catalog), "approved": sum(
        entry["decision"] == "approve" for entry in catalog.values()),
        "rejected": sum(entry["decision"] == "reject" for entry in catalog.values()),
        "output": str(args.output)}))


if __name__ == "__main__":
    main()
