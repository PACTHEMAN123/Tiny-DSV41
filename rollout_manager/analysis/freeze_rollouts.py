"""Freeze complete GRPO groups from real responses for controlled B-only evaluation."""
import argparse
import json
import sys
from pathlib import Path

from rollout_manager.frozen_rollouts import freeze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial-qualification", action="store_true")
    args = parser.parse_args()
    result = freeze(args.source, args.cases, allow_partial=args.allow_partial_qualification)
    with args.output.open("x") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "groups"}))


if __name__ == "__main__":
    main()
