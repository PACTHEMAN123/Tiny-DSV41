#!/usr/bin/env python3
"""Compatibility entry point for the canonical rollout-manager workflow."""
from rollout_manager.terminal_bench.cpu_workflow import main

if __name__ == "__main__":
    raise SystemExit(main())
