"""Compare continuous and resumed LoRA SFT runs without reading full-model DCP.

All adapter elements, complete step metrics, and final eval probes are compared.
Initial eval probes describe different starting states and are not compared.
Exact equality is the default; nonzero tolerances are explicit in the report.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dsv41_train.lora import read_adapter
from scripts.verify_lora_adapter import compare_probes, file_sha256, validate_probe


LOSS_KEYS = ("loss", "ce_loss", "aux_loss")
COUNT_KEYS = ("input_tokens", "supervised_tokens")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def normalized(value):
    return json.loads(json.dumps(value, allow_nan=False))


def numeric(value, label):
    require(type(value) in (int, float) and math.isfinite(value), f"{label} must be finite numeric")
    return value


def read_run(folder, *, resumed):
    folder = Path(folder)
    config = json.loads((folder / "run_config.json").read_text())
    summary = json.loads((folder / "summary.json").read_text())
    reference = json.loads((folder / "lora_eval_reference.json").read_text())
    require(config["model_type"] in ("qwen", "dsv4") and config["lora"] is not None,
            "run must use Qwen or DSV4 LoRA")
    require(reference["format_version"] == 1 and reference["context"] == config, "eval reference config differs from run_config")
    require(summary["lora_verification"]["verified"] is True, "training LoRA verification did not pass")
    for key in ("model_type", "source", "world_size", "cp_size", "ep_size", "data"):
        require(summary[key] == config[key], f"summary {key} differs from run_config")
    steps, start = summary["steps"], summary["start_step"]
    require(type(steps) is int and steps > 0 and summary["step"] == steps, "invalid final step")
    require(type(start) is int and 0 <= start < steps, "invalid start_step")
    require(start > 0 if resumed else start == 0, "resumed run must have start_step>0; continuous run must have start_step=0")
    rows = [json.loads(line) for line in (folder / "metrics.jsonl").read_text().splitlines() if line.strip()]
    require(all(type(row["step"]) is int for row in rows)
            and [row["step"] for row in rows] == list(range(1, steps + 1)), "metrics must contain the complete ordered step sequence 1..final without gaps/duplicates")
    for row in rows:
        for key in LOSS_KEYS:
            numeric(row[key], f"step {row['step']} {key}")
        for key in COUNT_KEYS:
            require(type(row[key]) is int and row[key] > 0, f"step {row['step']} {key} must be a positive integer")
        require(row["supervised_tokens"] <= row["input_tokens"], "supervised count exceeds input count")
    for key in LOSS_KEYS + COUNT_KEYS:
        require(summary[key] == rows[-1][key], f"summary {key} disagrees with final metric")
    for key, summary_key in (("input_tokens", "trained_input_tokens"), ("supervised_tokens", "trained_supervised_tokens")):
        require(summary[summary_key] == sum(row[key] for row in rows[start:]), f"summary {summary_key} disagrees with this invocation's steps")
    for phase in ("initial", "trained"):
        validate_probe(reference[phase])
        require(reference[phase]["world_size"] == config["world_size"], "eval probe world_size differs from config")
    adapter_path = folder / "adapter.pt"
    digest = file_sha256(adapter_path)
    require(reference["adapter"] is not None and reference["adapter"]["sha256"] == digest, "adapter SHA256 does not match this run's eval reference")
    adapter = read_adapter(adapter_path)
    require(adapter["layer_ids"] == list(range(config["start_layer"], config["start_layer"] + config["num_layers"])), "adapter layer_ids differ from configured window")
    modules = normalized(adapter["modules"])
    require(modules and all(value == config["lora"] for value in modules.values()), "adapter module configuration differs from LoRA config")
    expected = {f"{module}.lora_{kind}" for module in modules for kind in ("a", "b")}
    require(set(adapter["weights"]) == expected, "adapter tensor names do not match all module A/B pairs")
    for name, tensor in adapter["weights"].items():
        require(isinstance(tensor, torch.Tensor) and tensor.is_floating_point()
                and tensor.ndim == 2 and tensor.numel() > 0, f"{name}: invalid adapter tensor")
        require(bool(torch.isfinite(tensor).all()), f"{name}: non-finite adapter tensor")
    return {"folder": str(folder.resolve()), "config": config, "summary": summary,
            "reference": reference, "metrics": rows, "adapter": adapter, "adapter_sha256": digest}


def compare_runs(continuous, resumed, *, atol=0., rtol=0.):
    require(all(math.isfinite(value) and value >= 0 for value in (atol, rtol)), "tolerances must be finite and nonnegative")
    report = {"validation_passed": False, "atol": atol, "rtol": rtol,
              "exact_equality_required": atol == rtol == 0,
              "continuous": str(Path(continuous).resolve()), "resumed": str(Path(resumed).resolve()),
              "full_adapter_tensors_compared": True, "errors": [],
              "scope": "All exported adapter elements, complete training-step losses/counts, and bounded final eval probes. Initial probes are intentionally not compared. No full-model DCP was read."}
    runs = {}
    for name, folder in (("continuous", continuous), ("resumed", resumed)):
        try:
            runs[name] = read_run(folder, resumed=name == "resumed")
        except Exception as error:
            report["errors"].append(f"{name}: {error}")
    if len(runs) != 2:
        report["full_adapter_tensors_compared"] = False
        return report
    left, right = runs["continuous"], runs["resumed"]
    try:
        require(left["config"] == right["config"], "run configurations differ")
        require(left["summary"]["steps"] == right["summary"]["steps"], "final steps differ")
        require(left["adapter"]["layer_ids"] == right["adapter"]["layer_ids"], "adapter layer_ids differ")
        require(normalized(left["adapter"]["modules"]) == normalized(right["adapter"]["modules"]), "adapter modules differ")
        require(left["adapter"]["weights"].keys() == right["adapter"]["weights"].keys(), "adapter tensor keys differ")
        report.update(final_steps=left["summary"]["steps"], resumed_start_step=right["summary"]["start_step"],
                      adapter_file_sha256={key: value["adapter_sha256"] for key, value in runs.items()},
                      adapter_serialization_note="Each file hash is checked against its own reference; file hashes need not match across separately serialized exports.")
        tensor_results = {}
        for name, expected in left["adapter"]["weights"].items():
            actual = right["adapter"]["weights"][name]
            require(expected.shape == actual.shape and expected.dtype == actual.dtype, f"{name}: adapter shape/dtype differs")
            # FP64 preserves differences of every supported adapter float dtype.
            differences = (actual.double() - expected.double()).abs()
            outside = differences > atol + rtol * expected.double().abs()
            tensor_results[name] = {"shape": list(expected.shape), "dtype": str(expected.dtype),
                                    "elements": expected.numel(), "exact_equal": torch.equal(actual, expected),
                                    "max_abs_difference": differences.max().item(),
                                    "outside_tolerance": int(outside.sum()), "passed": not bool(outside.any())}
        report["adapter_tensors"] = tensor_results
        report["adapter_tensor_count"] = len(tensor_results)
        report["adapter_elements_compared"] = sum(row["elements"] for row in tensor_results.values())
        report["adapter_max_abs_difference"] = max(row["max_abs_difference"] for row in tensor_results.values())
        if not all(row["passed"] for row in tensor_results.values()):
            report["errors"].append("adapter elements differ beyond configured tolerances")
        step_results = []
        for expected, actual in zip(left["metrics"], right["metrics"]):
            require(expected["step"] == actual["step"], "metric steps differ")
            deltas = {key: abs(actual[key] - expected[key]) for key in LOSS_KEYS}
            passed = all(delta <= atol + rtol * abs(expected[key]) for key, delta in deltas.items())
            counts = {key: {"continuous": expected[key], "resumed": actual[key], "delta": actual[key] - expected[key]} for key in COUNT_KEYS}
            passed = passed and all(row["delta"] == 0 for row in counts.values())
            step_results.append({"step": expected["step"], "passed": passed, "loss_abs_differences": deltas, "counts": counts})
        report["steps"] = step_results
        if not all(row["passed"] for row in step_results):
            report["errors"].append("per-step losses or token counts differ")
        # Resume's initial reflects the split checkpoint, never compare it to step 0.
        report["final_eval"] = compare_probes(right["reference"]["trained"], left["reference"]["trained"], atol=atol, rtol=rtol)
        if not report["final_eval"]["passed"]:
            report["errors"].append("final trained eval probes differ")
    except Exception as error:
        report["errors"].append(str(error))
        report["full_adapter_tensors_compared"] = "adapter_tensors" in report
    report["validation_passed"] = not report["errors"]
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continuous", required=True)
    parser.add_argument("--resumed", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=0.)
    parser.add_argument("--rtol", type=float, default=0.)
    args = parser.parse_args(argv)
    try:
        report = compare_runs(args.continuous, args.resumed, atol=args.atol, rtol=args.rtol)
    except ValueError as error:
        report = {"validation_passed": False, "errors": [str(error)]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"validation_passed": report["validation_passed"], "output": str(args.output), "errors": report["errors"]}))
    return 0 if report["validation_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
