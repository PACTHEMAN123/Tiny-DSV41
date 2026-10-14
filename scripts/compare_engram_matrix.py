"""Validate five Qwen + Engram execution cases; this is not a quality benchmark.

Read ROOT/{gpu,cpu_sync,cpu_async,cpu_cache,gpu_cache}/summary.json. The GPU
reference still includes randomly initialized Engram tables. For the L20D smoke
matrix these are reduced tables on the full Qwen backbone, using repeated authored
execution data. Neither quality, representative cache hit rates nor speedups can
be inferred. Numeric tolerances are engineering checks, not convergence criteria.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


CASES = ("gpu", "cpu_sync", "cpu_async", "cpu_cache", "gpu_cache")
WORKLOAD = ("world_size", "ep", "cp", "batch_size", "microbatches", "steps",
            "sequence_length", "global_batch", "data", "engram_config",
            "engram_parameters", "optimizer_foreach")
CACHE_COUNTERS = ("cache_hits", "cache_misses", "cache_fill_rows", "cache_evictions",
                  "cache_invalidated_rows", "cache_metadata_h2d_bytes")
METRICS = ("lookup_rows", "h2d_bytes", "d2h_bytes", "resolve_wait_seconds") + CACHE_COUNTERS


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_tree(value, path="summary"):
    if isinstance(value, dict):
        for key, child in value.items():
            finite_tree(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            finite_tree(child, f"{path}[{index}]")
    elif isinstance(value, (float, int)) and not isinstance(value, bool):
        require(math.isfinite(value), f"non-finite value at {path}")


def number(value, path, *, positive=False):
    require(type(value) in (int, float) and math.isfinite(value), f"{path} must be finite numeric")
    require(value > 0 if positive else value >= 0, f"{path} must be {'positive' if positive else 'nonnegative'}")
    return value


def flatten(value):
    if isinstance(value, list):
        for child in value:
            yield from flatten(child)
    else:
        require(type(value) in (int, float) and math.isfinite(value), "sample must contain finite numbers")
        yield value


def shape(value):
    if not isinstance(value, list):
        return ()
    child_shapes = [shape(child) for child in value]
    require(not child_shapes or all(child == child_shapes[0] for child in child_shapes), "ragged sample")
    return (len(value),) + (child_shapes[0] if child_shapes else ())


def difference(actual, reference, *, atol, rtol):
    require(shape(actual) == shape(reference), "sample shapes differ")
    left, right = list(flatten(actual)), list(flatten(reference))
    errors = [abs(a-b) for a, b in zip(left, right)]
    limits = [atol + rtol * abs(b) for b in right]
    failures = sum(error > limit for error, limit in zip(errors, limits))
    return {"passed": failures == 0, "count": len(left), "outside_tolerance": failures,
            "max_abs_difference": max(errors, default=0),
            "max_tolerance_ratio": max((error / limit if limit else (0 if error == 0 else None)
                                         for error, limit in zip(errors, limits)), default=0)
            if all(limits) else None}


def sample_rows(sample):
    ids, values = sample["ids"], sample["values"]
    require(isinstance(ids, list) and isinstance(values, list) and len(ids) == len(values), "table sample IDs/values differ in length")
    require(all(type(row) is int and row >= 0 for row in ids) and len(set(ids)) == len(ids), "table sample row IDs must be unique nonnegative integers")
    require(not values or (len(shape(values)) == 2 and shape(values)[1] > 0), "table samples must contain nonempty row vectors")
    list(flatten(values))
    return dict(zip(ids, values))


def validate_case(case, summary):
    require(isinstance(summary, dict), "summary must be an object")
    finite_tree(summary)
    require(summary["validation_passed"] is True, "summary validation_passed is not true")
    expected_variant = "cpu" if case in ("cpu_sync", "cpu_async") else case
    require(summary["variant"] == expected_variant, "variant does not match directory case")
    ranks = summary["ranks"]
    require(isinstance(ranks, list) and ranks, "missing ranks")
    world = ranks[0]["world_size"]
    require(type(world) is int and world > 0 and len(ranks) == world, "incomplete rank count")
    require({row["rank"] for row in ranks} == set(range(world)), "rank IDs are missing, duplicated or invalid")
    ranks = {row["rank"]: row for row in ranks}
    first = ranks[0]
    workload = {key: first[key] for key in WORKLOAD}
    workload["warmup_steps"] = len(first["warmup_losses"])
    tables = {str(layer) for layer in first["engram_config"]["layer_ids"]}
    require(tables, "no configured Engram tables")
    for key in ("ep", "cp", "batch_size", "microbatches", "steps", "sequence_length", "global_batch", "engram_parameters"):
        number(first[key], key, positive=True)
    require(len(summary["global_losses"]) == first["steps"], "global_losses length differs from steps")
    for key in ("tokens_per_second", "supervised_tokens_per_second", "peak_allocated_gib", "peak_reserved_gib"):
        number(summary[key], key, positive=True)
    expected_device = "cpu" if case.startswith("cpu") else "cuda"
    cached = case.endswith("cache")
    totals = {name: {key: 0 for key in METRICS} for name in tables}
    allocated = 0
    warnings = []
    for rank, row in ranks.items():
        prefix = f"rank {rank}"
        require(row["validation_passed"] is True, f"{prefix}: validation_passed is not true")
        require(row["variant"] == expected_variant, f"{prefix}: wrong variant")
        require({**{key: row[key] for key in WORKLOAD}, "warmup_steps": len(row["warmup_losses"])} == workload, f"{prefix}: workload differs between ranks")
        require(row["optimizer_foreach"] == summary["optimizer_foreach"], f"{prefix}: optimizer foreach differs")
        require(len(row["losses"]) == first["steps"] and len(row["warmup_losses"]) > 0, f"{prefix}: missing loss steps")
        checks = row["validation_checks"]
        minimum_steps = len(row["losses"]) + len(row["warmup_losses"])
        require(checks["finite_loss_before_backward"] >= minimum_steps * row["microbatches"]
                and checks["finite_gradients_before_step"] >= minimum_steps
                and checks["update_steps_verified"] >= 1, f"{prefix}: insufficient validation checks")
        update = row["update_validation"]
        number(update["dense_global_max_update"], f"{prefix} dense update", positive=True)
        require(set(update["tables"]) == tables and row["table_updates_first_step"] == update["tables"], f"{prefix}: incomplete table update verification")
        require(set(row["engram_metrics"]) == tables and set(row["table_samples"]) == tables, f"{prefix}: missing table metrics/samples")
        require(row["parameter_samples"] and "lm_head.weight" in row["parameter_samples"], f"{prefix}: missing dense samples")
        for samples in row["parameter_samples"].values():
            require(len(shape(samples)) == 1 and len(samples) > 0, f"{prefix}: dense sample must be nonempty vector")
            list(flatten(samples))
        number(row["lm_head_sample_update"], f"{prefix} head update")
        policy = row["policy"]
        require(policy == first["policy"], f"{prefix}: execution policy differs between ranks")
        require(policy["device"] == expected_device, f"{prefix}: wrong table placement")
        expected_prefetch = "async" if case == "cpu_async" else "sync" if case in ("cpu_sync", "gpu", "gpu_cache") else None
        require(policy["prefetch"] in ("sync", "async"), f"{prefix}: invalid prefetch policy")
        require(expected_prefetch is None or policy["prefetch"] == expected_prefetch, f"{prefix}: wrong prefetch policy")
        number(policy["cache_bytes"], f"{prefix} cache budget")
        require((policy["cache_bytes"] > 0) == cached, f"{prefix}: wrong cache budget")
        cache_bytes = number(row["cache_allocated_bytes"], f"{prefix} cache allocation")
        require(cache_bytes <= policy["cache_bytes"] * len(tables), f"{prefix}: cache allocation exceeds budget")
        allocated += cache_bytes
        for name in tables:
            record = update["tables"][name]
            for key in ("global_max_update", "global_nonzero_updates", "global_probed_rows", "global_gradient_nnz"):
                number(record[key], f"{prefix} table {name} {key}", positive=True)
            for key in ("max_update", "nonzero_updates", "gradient_nnz"):
                number(record[key], f"{prefix} table {name} {key}")
            sample_rows(row["table_samples"][name])
            metrics = row["engram_metrics"][name]
            for key in METRICS:
                totals[name][key] += number(metrics[key], f"{prefix} table {name} {key}")
            if cached:
                require(metrics["cache_hits"] + metrics["cache_misses"] == metrics["lookup_rows"], f"{prefix} table {name}: cache requests do not match lookups")
                require(metrics["cache_evictions"] <= metrics["cache_fill_rows"] <= metrics["cache_misses"], f"{prefix} table {name}: inconsistent cache fill/eviction counters")
            else:
                require(all(metrics[key] == 0 for key in CACHE_COUNTERS), f"{prefix} table {name}: cache activity with cache disabled")
    for index, loss in enumerate(summary["global_losses"]):
        mean = sum(row["losses"][index] for row in ranks.values()) / world
        require(math.isclose(loss, mean, rel_tol=1e-8, abs_tol=1e-9), "global loss disagrees with per-rank mean")
    for name, metrics in totals.items():
        require(metrics["lookup_rows"] > 0, f"table {name}: no lookup activity")
        if expected_device == "cpu":
            require(metrics["h2d_bytes"] > 0 and metrics["d2h_bytes"] > 0, f"table {name}: CPU offload transfer path was not exercised")
        else:
            require(metrics["h2d_bytes"] == metrics["d2h_bytes"] == 0, f"table {name}: unexpected CPU row traffic for GPU storage")
        if cached:
            require(allocated > 0 and metrics["cache_invalidated_rows"] > 0 and metrics["cache_metadata_h2d_bytes"] > 0,
                    f"table {name}: cache allocation/invalidation/metadata path was not exercised")
            if metrics["cache_hits"] == 0:
                warnings.append(f"table {name}: no timed cache hits; execution passed but reuse was not demonstrated")
        updates = [row["update_validation"]["tables"][name] for row in ranks.values()]
        for row in updates:
            require(row["global_max_update"] == max(part["max_update"] for part in updates)
                    and row["global_nonzero_updates"] == sum(part["nonzero_updates"] for part in updates)
                    and row["global_gradient_nnz"] == sum(part["gradient_nnz"] for part in updates),
                    f"table {name}: global update records disagree with local rank records")
        require(sum(len(row["table_samples"][name]["ids"]) for row in ranks.values()) > 0,
                f"table {name}: all final touched-row samples are empty")
    if all(row["lm_head_sample_update"] == 0 for row in ranks.values()):
        warnings.append("Head probes did not move; validated dense updates were observed in other tensors")
    return ranks, workload, {"validation_passed": True, "cache_allocated_bytes": allocated,
                             "table_metrics": totals, "warnings": warnings,
                             "policy": first["policy"],
                             "tokens_per_second": summary["tokens_per_second"],
                             "peak_allocated_gib": summary["peak_allocated_gib"]}


def compare_case(actual, reference, *, loss_atol, loss_rtol, sample_atol, sample_rtol):
    metrics = {"global_losses": difference(actual["global_losses"], reference["global_losses"], atol=loss_atol, rtol=loss_rtol), "ranks": {}}
    expected_ranks = {row["rank"]: row for row in reference["ranks"]}
    for row in actual["ranks"]:
        expected = expected_ranks[row["rank"]]
        require(set(row["parameter_samples"]) == set(expected["parameter_samples"]), f"rank {row['rank']}: dense sample keys differ")
        parameters = {name: difference(values, expected["parameter_samples"][name], atol=sample_atol, rtol=sample_rtol)
                      for name, values in row["parameter_samples"].items()}
        tables = {}
        for name, sample in row["table_samples"].items():
            rows, reference_rows = sample_rows(sample), sample_rows(expected["table_samples"][name])
            require(set(rows) == set(reference_rows), f"rank {row['rank']} table {name}: sampled row ID sets differ; values are not comparable")
            tables[name] = difference([rows[key] for key in sorted(rows)], [reference_rows[key] for key in sorted(rows)], atol=sample_atol, rtol=sample_rtol)
        metrics["ranks"][str(row["rank"])] = {"parameters": parameters, "tables": tables}
    metrics["validation_passed"] = metrics["global_losses"]["passed"] and all(
        result["passed"] for rank in metrics["ranks"].values() for group in rank.values() for result in group.values())
    return metrics


def compare_matrix(root, *, loss_atol=.02, loss_rtol=.01, sample_atol=.0002, sample_rtol=.02):
    tolerances = dict(loss_atol=loss_atol, loss_rtol=loss_rtol, sample_atol=sample_atol, sample_rtol=sample_rtol)
    for key, value in tolerances.items():
        number(value, key)
    root = Path(root)
    report = {"validation_passed": False, "reference": "gpu", "tolerances": tolerances,
              "scope": "Full Qwen backbone plus randomly initialized reduced Engram tables; gpu is an Engram placement reference, not bare Qwen.",
              "limitations": ["Repeated authored smoke data validates execution only; no quality or representative performance/cache-hit claim.",
                              "Finite loss/gradients are checked each step; actual updates use bounded warmup probes; cross-variant comparisons use bounded final samples.",
                              "BF16 tolerances are configurable engineering thresholds, not evidence of convergence or full-tensor equivalence.",
                              "Summaries do not fingerprint the backbone weights or record learning rate; identical model mounts and learning rate must be ensured by the launcher.",
                              "Timing is reported without a speedup requirement; it includes finite checks and may include external workload interference."],
              "variants": {}, "comparisons": {}, "errors": []}
    summaries, workloads = {}, {}
    for case in CASES:
        path = root / case / "summary.json"
        try:
            summary = json.loads(path.read_text())
            _, workload, validation = validate_case(case, summary)
            summaries[case], workloads[case] = summary, workload
            report["variants"][case] = {"path": str(path), **validation}
        except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
            report["errors"].append(f"{case}: {error}")
            report["variants"][case] = {"path": str(path), "validation_passed": False, "error": str(error)}
    if "gpu" in summaries:
        report["workload"] = workloads["gpu"]
        for case, summary in summaries.items():
            try:
                require(workloads[case] == workloads["gpu"], "workload/config/data identity differs from gpu reference")
                comparison = compare_case(summary, summaries["gpu"], **tolerances)
                report["comparisons"][case] = comparison
                require(comparison["validation_passed"], "numeric comparison exceeds configured BF16 tolerances")
            except (ValueError, KeyError, TypeError, IndexError) as error:
                report["errors"].append(f"{case}: {error}")
    report["validation_passed"] = len(summaries) == len(CASES) and not report["errors"]
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, help="Defaults to ROOT/comparison.json")
    parser.add_argument("--loss-atol", type=float, default=.02)
    parser.add_argument("--loss-rtol", type=float, default=.01)
    parser.add_argument("--sample-atol", type=float, default=.0002)
    parser.add_argument("--sample-rtol", type=float, default=.02)
    args = parser.parse_args(argv)
    try:
        report = compare_matrix(args.root, loss_atol=args.loss_atol, loss_rtol=args.loss_rtol,
                                sample_atol=args.sample_atol, sample_rtol=args.sample_rtol)
    except ValueError as error:
        parser.error(str(error))
    output = args.output or args.root / "comparison.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"validation_passed": report["validation_passed"], "output": str(output), "errors": report["errors"]}))
    return 0 if report["validation_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
