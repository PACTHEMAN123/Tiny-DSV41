"""Combine complete native qualification and explicit compatibility reruns."""
import argparse
import json
import sys
from pathlib import Path

from rollout_manager.reference_qualification import build_catalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--correction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    catalog = build_catalog(args.cases, args.baseline, args.correction)
    with args.output.open("x") as stream:
        json.dump(catalog, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"tasks": len(catalog["tasks"]), "unscorable":
        {key: item["reason"] for key, item in catalog["tasks"].items() if not item["score_eligible"]}}))


if __name__ == "__main__":
    main()
