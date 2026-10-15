"""Invoked only in a reviewed evaluation child, never imported by the manager."""
from __future__ import annotations

import argparse
import importlib
import json
import os
import runpy
import sys
import time
from pathlib import Path
from .native_timing import HarnessInstrumentationError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["verify", "profile", "kernelbench", "trace"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--upstream", type=Path,
                        help="explicit external pinned native harness assets; not the manager source tree")
    parser.add_argument("--observe-inputs", action="store_true")
    parser.add_argument("--preserve-integer-inputs", action="store_true")
    parser.add_argument("--input-reuse-config", type=Path)
    parser.add_argument('--isolate-device-inputs', action='store_true')
    args = parser.parse_args()
    if args.mode == "kernelbench":
        if args.upstream is None:
            parser.error("KernelBench execution requires explicit --upstream assets")
        native_source = args.upstream.resolve() / "KernelBench" / "src"
        if not (native_source / "kernelbench" / "eval.py").is_file():
            parser.error("pinned KernelBench native harness source is missing")
        # The optional native harness remains an externally pinned asset; the
        # installed manager is found normally by this worker's Python runtime.
        sys.path.insert(0, str(native_source))
    sys.path.insert(0, str(Path.cwd()))
    import random
    import numpy as np
    import torch
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    adapter = None
    reuse = None
    if args.input_reuse_config:
        from .input_reuse import InputReuse
        config = json.loads(args.input_reuse_config.read_text())
        try:
            reuse = InputReuse(torch, config["socket"], config["consumer"], Path("model.py").read_text(),
                config["plan"], namespace=config["namespace"], seed=args.seed, mode=args.mode,
                qualification=config["qualification"])
        except Exception as exc:
            raise HarnessInstrumentationError("input reuse initialization failed: " + str(exc)) from exc
    if args.observe_inputs or reuse or args.isolate_device_inputs:
        from .native_timing import NativeHarnessTiming
        try:
            adapter = NativeHarnessTiming(torch, args.mode, args.seed, reuse=reuse,
                                          isolate_device_inputs=args.isolate_device_inputs)
        except Exception as exc:
            raise HarnessInstrumentationError("native timing initialization failed: " + str(exc)) from exc
        adapter.metrics["process_startup_seconds"] = (time.monotonic_ns() - int(
            os.environ.get("ROLLOUT_LAUNCH_MONOTONIC_NS", time.monotonic_ns()))) / 1e9
    if args.mode == "trace":
        from model import Model, get_inputs, get_init_inputs
        from model_new import ModelNew
        init = get_init_inputs()
        reference, candidate = Model(*init).cuda().eval(), ModelNew(*init).cuda().eval()
        candidate.load_state_dict(reference.state_dict())
        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        with torch.no_grad(), torch.profiler.profile(activities=activities, record_shapes=True) as profiler:
            with torch.profiler.record_function("inputs_and_h2d"):
                inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in get_inputs()]
                torch.cuda.synchronize()
            for label, model in (("reference", reference), ("candidate", candidate)):
                with torch.profiler.record_function(label):
                    for _ in range(3):
                        model(*inputs)
                    torch.cuda.synchronize()
        trace = args.report.with_suffix(".trace.json")
        profiler.export_chrome_trace(str(trace))
        result = {"completed": True, "trace": str(trace), "seed": args.seed,
                  "scope": "separate instrumented execution; not benchmark timing or correctness evidence"}
    elif args.mode in ("verify", "profile"):
        module = "utils.verification" if args.mode == "verify" else "utils.profiling"
        sys.argv = [module] if args.mode == "verify" else [module, "--iters", "10"]
        if adapter is None:
            runpy.run_module(module, run_name="__main__")
        else:
            native_module = importlib.import_module(module)
            adapter.install_cuda_agent(native_module)
            native_module.main()
        result = {"completed": True, "seed": args.seed, "upstream_module": module}
    else:
        from kernelbench import timing
        from kernelbench import eval as native_module
        compatibility = None
        if args.preserve_integer_inputs:
            from .kernelbench_compat import preserve_integer_inputs
            compatibility = preserve_integer_inputs(native_module, torch)
        if adapter is not None:
            adapter.install_kernelbench(native_module)
        samples = []
        original = timing.get_timing_function

        def observed(method):
            function = original(method)

            def measure(*a, **kw):
                if adapter is None:
                    values = function(*a, **kw)
                else:
                    with adapter.native_timing(), adapter.timed("native_scoring"):
                        values = function(*a, **kw)
                samples.append({"method": method, "samples_ms": [float(x) for x in values]})
                return values
            return measure

        timing.get_timing_function = observed
        result = native_module.eval_kernel_against_ref(Path("model.py").read_text(), Path("model_new.py").read_text(),
            seed_num=args.seed, num_correct_trials=5, num_perf_trials=10, measure_performance=True,
            timing_method="cuda_event", device=torch.device("cuda:0"), backend="cuda", precision=torch.float32)
        result = result.model_dump()
        result.update(timing_unit="ms", timing_samples=samples, seed=args.seed)
        result["native_compatibility"] = compatibility
    if adapter is not None:
        result["stage_metrics"] = adapter.finish(result.get("correctness", result.get("completed", False)))
    args.report.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps(result, default=str), flush=True)


if __name__ == "__main__":
    try:
        main()
    except HarnessInstrumentationError as exc:
        print(json.dumps({"infrastructure_error": "native_harness_instrumentation", "detail": str(exc)}), flush=True)
        sys.exit(78)
