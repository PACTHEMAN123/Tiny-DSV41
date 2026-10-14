"""Efficiency benchmark for Qwen3 MoE plus synthetic trainable Engram.

Table dimensions/addressing pattern can match V4.1, but weights are random and
the Qwen tokenizer uses an identity map. This does not measure model quality or
claim that Qwen has the pretrained V4.1 Engram semantics.
"""

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import datetime
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
import traceback

import torch
import torch.distributed as dist

from dsv41_train.layout import local_tensor, parameter_memory
from dsv41_train.models.qwen import load_qwen3_moe
from dsv41_train.models.qwen.config import Qwen3MoeConfig
from dsv41_train.models.qwen.engram import (
    QwenEngramConfig, attach_engram, shard_qwen_component, shard_qwen_root,
)
from dsv41_train.models.dsv4.engram import EngramPolicy
from dsv41_train.models.dsv4.parallel import build_parallelism
from dsv41_train.models.dsv4.policies import GradientPolicy
from dsv41_train.optimization import OptimizerPolicy, build_optimizer
from dsv41_train.runtime import Runtime
from train_sft import (_collective_finite, _file_hash, _finite_gradients,
                       _probe_indices, _source_identity, _update_probes)
from scripts.benchmark_dsv4_engram_sft import (
    compare_summaries, final_samples, gradient_summary, hash_layout, metric_delta,
    metric_snapshot, numeric_difference, timing_statistics, validate_storage_metrics,
)
from scripts.engram_update_validation import capture_touched_rows, verify_touched_updates
from scripts.qwen_dummy_init_cache import CachedQwenTableInitializer

TOUCHED_ROW_CHUNK = 8192
DENSE_UPDATE_CHUNK = 1048576


def _configure_determinism(enabled):
    """Opt in before CUDA initialization; report the effective execution mode."""
    if enabled:
        if torch.cuda.is_initialized():
            raise RuntimeError("--deterministic must be configured before CUDA initialization")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise RuntimeError("--deterministic requires launcher CUBLAS_WORKSPACE_CONFIG=:4096:8")
        torch.use_deterministic_algorithms(True, warn_only=False)
    return {"requested": bool(enabled),
            "algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
            "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}


class DenseUpdateValidationError(RuntimeError):
    def __init__(self, report):
        super().__init__("no actual numerical update in ALL selected dense elements; bounded-sample ambiguity eliminated")
        self.diagnostics = report


def _append_step_record(output, rank, record):
    with (Path(output) / f"steps-rank-{rank}.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_error_record(output, rank, context, error):
    diagnostic = {**context, "rank": rank, "error_type": type(error).__name__, "error": str(error),
                  "traceback": traceback.format_exc(), "validation_passed": False,
                  "diagnostics": getattr(error, "diagnostics", None)}
    with (Path(output) / f"error-rank-{rank}.json").open("x", encoding="utf-8") as stream:
        json.dump(diagnostic, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _require_all(condition, runtime, message):
    okay = torch.as_tensor(condition, device=runtime.device, dtype=torch.int32).all().to(torch.int32)
    dist.all_reduce(okay, op=dist.ReduceOp.MIN)
    if not okay.item():
        raise RuntimeError(message)


def _validate_data(data, required_rows, runtime, sequence_length=None):
    ids, labels = data.get("input_ids"), data.get("labels")
    valid = (isinstance(ids, torch.Tensor) and isinstance(labels, torch.Tensor)
             and ids.ndim == 2 and labels.shape == ids.shape and ids.shape[1] >= 2
             and ids.shape[0] >= required_rows and ids.dtype == labels.dtype == torch.int64
             and isinstance(data.get("metadata"), dict))
    _require_all(valid, runtime, "benchmark data needs matching int64 input_ids/labels, metadata and enough rows")
    _require_all(labels[:, 1:].ne(-100).any(dim=1).all(), runtime,
                 "every source row must contain shifted supervised targets")
    _require_all(sequence_length is None or ids.shape[1] == sequence_length, runtime,
                 "packed sequence width differs from --seq-len")
    _require_all(labels[:, 0].eq(-100).all() and (labels.eq(-100) | labels.eq(ids)).all(), runtime,
                 "packed labels must align with input tokens and mask the first position")
    if "attention_mask" in data:
        mask = data["attention_mask"]
        _require_all(isinstance(mask, torch.Tensor) and mask.shape == ids.shape and bool(mask.all()), runtime,
                     "efficiency benchmark requires packed sequences without padding")
    _require_all(data["metadata"].get("packing_padding_tokens", 0) == 0, runtime,
                 "packed metadata reports padding")


def _selected_dense_parameters(model):
    names = {"lm_head.weight", "model.norm.weight"}
    return {name: parameter for name, parameter in model.named_parameters()
            if parameter.requires_grad and (name in names or name.endswith(".engram.proj.weight"))}


def _benchmark_dense_probes(model):
    probes = []
    selected = _selected_dense_parameters(model)
    for name, parameter in selected.items():
        value = local_tensor(parameter).detach().reshape(-1)
        if value.numel():
            indices = _probe_indices(value.numel(), device=value.device)
            probes.append((name, parameter, indices, value[indices].float().clone()))
    if not selected:
        raise RuntimeError("no selected dense parameters for benchmark probes")
    return probes


@torch.no_grad()
def _capture_dense_updates(parameters, runtime, *, chunk_elements=DENSE_UPDATE_CHUNK):
    """Snapshot all elements of the selected dense tensors, not the whole model."""
    if type(chunk_elements) is not int or chunk_elements < 1:
        raise ValueError("chunk_elements must be a positive integer")
    valid = bool(parameters) and all(local_tensor(p).is_contiguous() and local_tensor(p).is_floating_point()
                                     for p in parameters.values())
    _require_all(valid, runtime, "full dense validation requires selected contiguous floating parameters")
    snapshots = {}
    for name, parameter in parameters.items():
        weight = local_tensor(parameter).detach()
        flat = weight.view(-1)
        before = flat.new_empty(flat.numel())
        finite = torch.ones((), dtype=torch.bool, device=weight.device)
        for start in range(0, flat.numel(), chunk_elements):
            stop = min(start + chunk_elements, flat.numel())
            before[start:stop].copy_(flat[start:stop])
            finite &= torch.isfinite(before[start:stop]).all()
        _require_all(finite, runtime, f"selected dense {name}: non-finite weight before optimizer step")
        snapshots[name] = (before, tuple(weight.shape), weight.data_ptr())
    return snapshots


@torch.no_grad()
def _verify_dense_updates(parameters, snapshots, runtime, *, chunk_elements=DENSE_UPDATE_CHUNK):
    if type(chunk_elements) is not int or chunk_elements < 1:
        raise ValueError("chunk_elements must be a positive integer")
    _require_all(bool(parameters) and set(parameters) == set(snapshots), runtime,
                 "selected dense parameter/snapshot names differ")
    records = {}
    for name, parameter in parameters.items():
        weight = local_tensor(parameter).detach()
        before, shape, pointer = snapshots[name]
        _require_all(weight.is_contiguous() and tuple(weight.shape) == shape and weight.data_ptr() == pointer
                     and weight.dtype == before.dtype and weight.device == before.device, runtime,
                     f"selected dense {name}: parameter storage changed across optimizer step")
        flat = weight.view(-1)
        maximum = torch.zeros((), dtype=torch.float32, device=weight.device)
        changed = torch.zeros((), dtype=torch.int64, device=weight.device)
        finite = torch.ones((), dtype=torch.bool, device=weight.device)
        for start in range(0, flat.numel(), chunk_elements):
            stop = min(start + chunk_elements, flat.numel())
            after, old = flat[start:stop], before[start:stop]
            finite &= torch.isfinite(after).all()
            changed += after.ne(old).sum()
            # copy=True is essential: FP32 parameters must never alias this buffer.
            delta = after.to(torch.float32, copy=True).sub_(old.float()).abs_()
            finite &= torch.isfinite(delta).all()
            maximum = torch.maximum(maximum, delta.max())
        _require_all(finite, runtime, f"selected dense {name}: non-finite weight/update after optimizer step")
        local_max, local_changed = maximum.item(), changed.item()
        snapshot_bytes = before.untyped_storage().nbytes()
        counts = torch.tensor([flat.numel(), local_changed, snapshot_bytes], dtype=torch.int64,
                              device=runtime.device)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        global_max = maximum.to(runtime.device)
        dist.all_reduce(global_max, op=dist.ReduceOp.MAX)
        records[name] = {"local_shape": list(shape), "dtype": str(before.dtype),
            "local_elements_checked": flat.numel(), "global_elements_checked": counts[0].item(),
            "local_changed_elements": local_changed, "global_changed_elements": counts[1].item(),
            "local_max_abs_delta": local_max, "global_max_abs_delta": global_max.item(),
            "snapshot_allocated_bytes_local": snapshot_bytes, "snapshot_allocated_bytes_global": counts[2].item()}
    report = {"scope": "all_elements_of_selected_dense_parameters", "all_selected_elements_checked": True,
              "finite_before_after": True, "parameter_names": list(records), "parameters": records,
              "chunk_elements": chunk_elements}
    for field in ("local_elements_checked", "global_elements_checked", "local_changed_elements",
                  "global_changed_elements", "snapshot_allocated_bytes_local", "snapshot_allocated_bytes_global"):
        report[field] = sum(record[field] for record in records.values())
    for field in ("local_max_abs_delta", "global_max_abs_delta"):
        report[field] = max(record[field] for record in records.values())
    if report["global_changed_elements"] <= 0 or report["global_max_abs_delta"] <= 0:
        raise DenseUpdateValidationError(report)
    return report


def _compare_qwen_summaries(actual, expected, **tolerances):
    # Preserve every existing reference test and its original tolerance.
    comparison = compare_summaries(actual, expected, **tolerances)
    records = {}
    reference = {row["rank"]: row for row in expected["ranks"]}
    for row in actual["ranks"]:
        steps = []
        for step, prior in zip(row["step_records"], reference[row["rank"]]["step_records"]):
            own, old = step["full_dense_updates"], prior["full_dense_updates"]
            for proof in (own, old):
                if (proof["scope"] != "all_elements_of_selected_dense_parameters"
                        or proof["all_selected_elements_checked"] is not True
                        or proof["finite_before_after"] is not True
                        or proof["global_changed_elements"] <= 0 or proof["global_max_abs_delta"] <= 0):
                    raise ValueError("reference full selected-dense update proof failed")
            if own["parameter_names"] != old["parameter_names"] or set(own["parameters"]) != set(old["parameters"]):
                raise ValueError("reference selected-dense parameter names differ")
            tensors = {}
            for name, proof in own["parameters"].items():
                before = old["parameters"][name]
                for field in ("local_shape", "dtype", "local_elements_checked", "global_elements_checked",
                              "snapshot_allocated_bytes_local", "snapshot_allocated_bytes_global"):
                    if proof[field] != before[field]:
                        raise ValueError(f"reference selected-dense {name} coverage differs: {field}")
                tensors[name] = numeric_difference([proof["global_max_abs_delta"]], [before["global_max_abs_delta"]],
                                                  tolerances["sample_atol"], tolerances["sample_rtol"])
                tensors[name].update(global_changed_elements=proof["global_changed_elements"],
                    reference_global_changed_elements=before["global_changed_elements"],
                    changed_element_count_difference=proof["global_changed_elements"]-before["global_changed_elements"])
            steps.append({"step": step["step"], "parameters": tensors})
        records[str(row["rank"])] = steps
    comparison["full_dense_update_summaries"] = records
    comparison["validation_passed"] &= all(item["passed"] for steps in records.values()
        for step in steps for item in step["parameters"].values())
    return comparison


def _check_backward_inputs(loss, labels, runtime):
    valid = (isinstance(labels, torch.Tensor) and labels.ndim == 2
             and labels.shape[0] > 0 and labels.shape[1] >= 2)
    targets = labels[:, 1:].ne(-100).sum() if valid else 0
    _require_all(targets > 0, runtime, "each microbatch must contain shifted supervised targets")
    _require_all(isinstance(loss, torch.Tensor) and loss.numel() == 1, runtime,
                 "model must return a scalar supervised loss on every rank")
    _collective_finite(loss, runtime, "SFT loss before backward")


def _table_update_probes(tables, runtime):
    valid = all(table.weight.grad is None or local_tensor(table.weight.grad).is_sparse
                for table in tables.values())
    _require_all(valid, runtime, "Engram update verification requires sparse gradients on every rank")
    probes = {}
    for name, table in tables.items():
        gradient = table.weight.grad
        if gradient is None:
            continue
        gradient = local_tensor(gradient)
        rows = gradient._indices()[0].unique()
        if rows.numel():
            rows = rows[_probe_indices(rows.numel(), device=rows.device)]
            probes[name] = (rows, local_tensor(table.weight).detach()[rows].float().clone(), gradient._nnz())
    return probes


def _verify_updates(dense_probes, tables, table_probes, runtime, *, full_snapshots,
                    dense_parameters, dense_snapshots):
    maximum = torch.zeros((), dtype=torch.float32, device=runtime.device)
    for _, parameter, indices, before in dense_probes:
        after = local_tensor(parameter).detach().reshape(-1)[indices].float()
        maximum = torch.maximum(maximum, (after-before).abs().max().to(runtime.device))
    _collective_finite(maximum, runtime, "dense parameter update")
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    try:
        full_dense = _verify_dense_updates(dense_parameters, dense_snapshots, runtime)
    except DenseUpdateValidationError as error:
        error.diagnostics = {"full_dense_updates": error.diagnostics,
                             "dense_update_global_max": maximum.item()}
        raise
    records = {}
    for name, table in tables.items():
        local_max = torch.zeros((), dtype=torch.float32, device=runtime.device)
        counts = torch.zeros(3, dtype=torch.int64, device=runtime.device)
        if name in table_probes:
            rows, before, nnz = table_probes[name]
            after = local_tensor(table.weight).detach()[rows].float()
            local_max.copy_((after-before).abs().max())
            counts[0] = after.ne(before).sum().to(runtime.device)
            counts[1], counts[2] = rows.numel(), nnz
        _collective_finite(local_max, runtime, f"Engram table {name} update")
        local_value, local_nonzero, local_nnz = local_max.item(), counts[0].item(), counts[2].item()
        dist.all_reduce(local_max, op=dist.ReduceOp.MAX)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        records[name] = {"max_update": local_value, "nonzero_updates": local_nonzero,
                         "gradient_nnz": local_nnz, "global_max_update": local_max.item(),
                         "global_nonzero_updates": counts[0].item(),
                         "global_probed_rows": counts[1].item(), "global_gradient_nnz": counts[2].item()}
    # A bounded sample may legitimately round to no BF16 change while another
    # touched row updates. The complete touched-row proof is the hard check.
    full_updates = verify_touched_updates(tables, full_snapshots, runtime, chunk_rows=TOUCHED_ROW_CHUNK)
    return {"dense_global_max_update": maximum.item(), "tables": records,
            "full_table_updates": full_updates, "full_dense_updates": full_dense}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference", help="GPU summary for the same workload, checked before success")
    parser.add_argument("--variant", choices=("gpu", "gpu_cache", "cpu", "cpu_cache"), required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--microbatches", type=int, default=2)
    parser.add_argument("--steps", type=int, default=20, help="Measured optimizer steps after warmup")
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--deterministic", action="store_true",
                        help="Require launcher CUBLAS_WORKSPACE_CONFIG=:4096:8 and strict deterministic algorithms")
    parser.add_argument("--deduplicate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--ep", type=int, default=2)
    parser.add_argument("--layers", type=int, nargs="+", default=[1])
    parser.add_argument("--vocab-per-head", type=int, default=16000000)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--engram-init-manifest", type=Path,
                        help="Optional exact CPU-generated raw initial tables; every copied byte is SHA256 checked")
    parser.add_argument("--cache-mib", type=int, default=128, help="budget per table per rank")
    parser.add_argument("--staging-mib", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=.01)
    parser.add_argument("--optimizer-foreach", action=argparse.BooleanOptionalAction, default=True,
                        help="Dense SGD foreach kernels; disable to control peak memory")
    parser.add_argument("--prefetch", choices=("sync", "async"), default="async")
    parser.add_argument("--trace-dir")
    for name, default in (("loss-atol", .02), ("loss-rtol", .01), ("sample-atol", .0002), ("sample-rtol", .02)):
        parser.add_argument("--" + name, type=float, default=default)
    args = parser.parse_args(argv)
    if min(args.batch_size, args.microbatches, args.steps, args.warmup, args.ep,
           args.vocab_per_head, args.head_dim, args.staging_mib) < 1 or args.seq_len < 2:
        parser.error("batch, accumulation, warmup and measured steps must be positive")
    if args.seed < 0 or args.cache_mib < 0 or (args.variant.endswith("cache") and args.cache_mib == 0):
        parser.error("seed/cache must be nonnegative; cached variants need a positive budget")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning-rate must be positive finite")
    if any(not math.isfinite(getattr(args, key)) or getattr(args, key) < 0
           for key in ("loss_atol", "loss_rtol", "sample_atol", "sample_rtol")):
        parser.error("comparison tolerances must be finite and nonnegative")
    if not args.layers or len(set(args.layers)) != len(args.layers) or min(args.layers) < 0:
        parser.error("Engram layer IDs must be distinct nonnegative integers")
    return args


def main():
    args = parse_args()
    determinism = _configure_determinism(args.deterministic)
    torch.set_num_threads(1)
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device, timeout=datetime.timedelta(seconds=900))
    rank, world = dist.get_rank(), dist.get_world_size()
    runtime = Runtime(device, rank, int(os.environ["LOCAL_RANK"]), world)
    model = None
    out = Path(args.output)
    output_initialized = False

    def progress(**record):
        if rank == 0:
            print(json.dumps({"time": time.time(), **record}), flush=True)

    try:
        _require_all(not out.exists() or (out.is_dir() and not any(out.iterdir())), runtime,
                     "output directory must be absent or empty")
        dist.barrier()
        out.mkdir(parents=True, exist_ok=True)
        output_initialized = True
        data = torch.load(args.data, map_location="cpu", weights_only=True)
        global_batch = args.batch_size * args.microbatches * world
        updates = args.warmup + args.steps + bool(args.trace_dir)
        _require_all(isinstance(data, dict), runtime, "benchmark data must be a dictionary")
        _validate_data(data, global_batch * updates, runtime, args.seq_len)
        model_config = Qwen3MoeConfig.from_json(Path(args.model_path) / "config.json")
        _require_all(args.seq_len <= model_config.max_position_embeddings
                     and bool((data["input_ids"] >= 0).all() & (data["input_ids"] < model_config.vocab_size).all()),
                     runtime, "sequence/token ID exceeds checkpoint bounds")
        _require_all(max(args.layers) < model_config.num_hidden_layers
                     and model_config.num_experts % args.ep == 0, runtime,
                     "Engram layer outside backbone or EP does not divide expert count")
        sequence_length = data["input_ids"].shape[1]
        config = QwenEngramConfig(tuple(args.layers), args.vocab_per_head, args.head_dim, seed=args.seed)
        table_initializer = (CachedQwenTableInitializer(args.engram_init_manifest, args.model_path, config,
                             world_size=world, ep_size=args.ep) if args.engram_init_manifest is not None else None)
        workload = {"source": _source_identity(args.model_path), "data_sha256": _file_hash(args.data),
                    "data_metadata": data["metadata"], "seed": args.seed, "world_size": world,
                    "ep": args.ep, "cp": 1, "batch_size": args.batch_size, "microbatches": args.microbatches,
                    "seq_len": sequence_length, "warmup": args.warmup, "steps": args.steps,
                    "deduplicate": args.deduplicate, "optimizer_foreach": args.optimizer_foreach,
                    "learning_rate": args.learning_rate, "engram_layers": args.layers,
                    "determinism": determinism,
                    "vocab_per_head": args.vocab_per_head, "head_dim": args.head_dim,
                    "dummy_init": (table_initializer.identity if table_initializer is not None
                                   else {"method": "cpu-normal-8192-v1"}),
                    "update_validation": {"method": "all-unique-touched-rows-v1", "chunk_rows": TOUCHED_ROW_CHUNK,
                        "helper_sha256": _file_hash(Path(__file__).resolve().parents[1] /
                                                   "scripts/engram_update_validation.py")},
                    "dense_update_validation": {"method": "all-selected-dense-elements-v1",
                        "benchmark_sha256": _file_hash(__file__), "chunk_elements": DENSE_UPDATE_CHUNK,
                        "selection": ["lm_head.weight", "model.norm.weight", "*.engram.proj.weight"]}}
        identities = [None] * world
        dist.all_gather_object(identities, workload)
        _require_all(all(identity == workload for identity in identities), runtime,
                     "model/data/config identity differs across ranks")
        _require_all(world % args.ep == 0, runtime, "EP must divide world size")
        schedule = []
        for step in range(updates):
            start = step * global_batch
            denominator = int(data["labels"][start:start+global_batch, 1:].ne(-100).sum())
            batches = []
            for micro in range(args.microbatches):
                offset = start + (micro * world + rank) * args.batch_size
                ids = data["input_ids"][offset:offset+args.batch_size].to(device)
                labels = data["labels"][offset:offset+args.batch_size].to(device)
                count = int(data["labels"][offset:offset+args.batch_size, 1:].ne(-100).sum())
                batches.append((ids, labels, count * world / denominator))
            schedule.append(batches)
        meshes, cp, dispatcher = build_parallelism(1, args.ep)
        policy = EngramPolicy("cpu" if args.variant.startswith("cpu") else "cuda",
            args.prefetch if args.variant.startswith("cpu") else "sync", args.deduplicate,
            args.staging_mib * 1024**2, args.cache_mib * 1024**2 if args.variant.endswith("cache") else 0)
        torch.manual_seed(args.seed)
        torch.cuda.reset_peak_memory_stats(device)
        load_start = time.perf_counter()
        load_spans = []
        previous = load_start

        def setup(loaded):
            attach_engram(loaded, config, policy, meshes.engram, device=device, table_initializer=table_initializer)
            progress(phase="engram_initialized", table_parameters=sum(
                table.global_num_embeddings * table.embedding_dim for table in loaded.model.engram_tables.values()))

        def materialize(name, module):
            nonlocal previous
            shard_qwen_component(name, module, meshes)
            torch.cuda.synchronize(device)
            now = time.perf_counter()
            load_spans.append({"name": name, "seconds_since_previous": now-previous})
            previous = now
            if name.endswith((".0", ".12", ".24", ".36")):
                progress(phase="loading", module=name, seconds=now-load_start)

        progress(phase="load_start", variant=args.variant)
        model = load_qwen3_moe(args.model_path, device=device, context_parallel=cp,
            token_dispatcher=dispatcher, setup_callback=setup, module_callback=materialize)
        shard_qwen_root(model, meshes)
        model.gradient_checkpointing_enable()
        model.train()
        torch.cuda.synchronize(device)
        load_seconds = time.perf_counter() - load_start
        load_peak = torch.cuda.max_memory_allocated(device)
        optimizer = build_optimizer(model, OptimizerPolicy("sgd", args.learning_rate, args.optimizer_foreach))
        gradients = GradientPolicy(model, meshes)
        tables = model.model.engram_tables
        addressing = hash_layout(model.model.engram_hash, model.model.engram_layer_ids, tables)
        dense_parameters = _selected_dense_parameters(model)
        initial_dense = _benchmark_dense_probes(model)
        progress(phase="loaded", seconds=load_seconds, allocated_gib=torch.cuda.memory_allocated(device)/2**30)
        head_indices = _probe_indices(local_tensor(model.lm_head.weight).numel(), device=device)
        initial = local_tensor(model.lm_head.weight).detach().flatten()[head_indices].float().cpu().clone()
        table_updates = {}
        update_validation = None
        validation_checks = {"finite_loss_before_backward": 0, "finite_gradients_before_step": 0,
                             "update_steps_verified": 0, "full_dense_steps_verified": 0}
        step_records = []
        last_table_probes = {}
        failure_context = {"phase": "loaded", "rank": rank}

        def run_step(index, tracing=False, verify=True):
            nonlocal table_updates, update_validation, last_table_probes
            failure_context.clear()
            failure_context.update(rank=rank, step=index, warmup=index < args.warmup, phase="zero_grad")
            region = torch.profiler.record_function if tracing else lambda _: nullcontext()
            phase_ms = {}
            metrics_before = metric_snapshot(tables)
            validation_seconds = 0.
            with region("train/zero_grad"):
                optimizer.zero_grad(set_to_none=True)
            gradients.begin_step()
            losses = []
            for ids, labels, scale in schedule[index]:
                failure_context["phase"] = "forward"
                tick = time.perf_counter()
                with region("train/forward"):
                    output = model(ids, labels=labels, output_router_logits=False)
                    torch.cuda.synchronize(device)
                    validation_tick = time.perf_counter()
                    _check_backward_inputs(output.loss, labels, runtime)
                    loss = output.loss * scale
                    _collective_finite(loss, runtime, "scaled SFT loss before backward")
                    validation_checks["finite_loss_before_backward"] += 1
                    validation_seconds += time.perf_counter() - validation_tick
                phase_ms["forward_host"] = phase_ms.get("forward_host", 0) + (time.perf_counter()-tick)*1000
                tick = time.perf_counter()
                with region("train/backward"):
                    failure_context["phase"] = "backward"
                    loss.backward()
                phase_ms["backward_host"] = phase_ms.get("backward_host", 0) + (time.perf_counter()-tick)*1000
                losses.append(loss.detach())
                del output, loss
            with region("train/normalize"):
                gradients.normalize()
                for table in tables.values():
                    table.synchronize()
            torch.cuda.synchronize(device)
            validation_tick = time.perf_counter()
            failure_context["phase"] = "gradient_and_before_update_validation"
            _finite_gradients(model, runtime)
            validation_checks["finite_gradients_before_step"] += 1
            step_loss = torch.stack(losses).sum()
            _collective_finite(step_loss, runtime, "accumulated SFT loss before optimizer step")
            if verify:
                global_loss = step_loss.detach().double().clone()
                dist.all_reduce(global_loss)
                global_loss /= world
                failure_context.update(local_scaled_loss=step_loss.item(), global_loss=global_loss.item())
                dense_probes = _benchmark_dense_probes(model)
                dense_snapshots = _capture_dense_updates(dense_parameters, runtime)
                table_probes = _table_update_probes(tables, runtime)
                full_snapshots = capture_touched_rows(tables, runtime, chunk_rows=TOUCHED_ROW_CHUNK)
                gradient_stats = gradient_summary(dense_probes, tables)
                failure_context["gradients"] = gradient_stats
            torch.cuda.synchronize(device)
            validation_seconds += time.perf_counter() - validation_tick
            tick = time.perf_counter()
            with region("train/optimizer"):
                failure_context["phase"] = "optimizer_step"
                optimizer.step()
            phase_ms["optimizer_host"] = (time.perf_counter()-tick)*1000
            torch.cuda.synchronize(device)
            if verify:
                validation_tick = time.perf_counter()
                failure_context["phase"] = "after_update_validation"
                checked = _verify_updates(dense_probes, tables, table_probes, runtime,
                                          full_snapshots=full_snapshots,
                                          dense_parameters=dense_parameters, dense_snapshots=dense_snapshots)
                del full_snapshots, dense_snapshots
                if update_validation is None:
                    update_validation = checked
                    table_updates = checked["tables"]
                last_table_probes = table_probes
                validation_checks["update_steps_verified"] += 1
                validation_checks["full_dense_steps_verified"] += 1
                validation_seconds += time.perf_counter() - validation_tick
                offset = index * global_batch
                step_records.append({"rank": rank, "step": index, "warmup": index < args.warmup,
                    "local_scaled_loss": step_loss.item(), "global_loss": global_loss.item(),
                    "global_input_tokens": global_batch * sequence_length,
                    "global_supervised_tokens": int(data["labels"][offset:offset+global_batch, 1:].ne(-100).sum()),
                    "gradients": gradient_stats, "dense_update_global_max": checked["dense_global_max_update"],
                    "table_updates": {name: {**item, "parameter_delta_max": item["global_max_update"]}
                                      for name, item in checked["tables"].items()},
                    "full_table_updates": checked["full_table_updates"],
                    "full_dense_updates": checked["full_dense_updates"],
                    "engram_metrics": metric_delta(metric_snapshot(tables), metrics_before),
                    "validation_seconds": validation_seconds})
            return step_loss.item(), phase_ms

        warmup_losses = []
        for index in range(args.warmup):
            tick = time.perf_counter()
            value, _ = run_step(index)
            step_records[-1]["seconds_including_validation"] = time.perf_counter()-tick
            _append_step_record(out, rank, step_records[-1])
            warmup_losses.append(value)
            progress(phase="warmup", step=index, loss=value, seconds=time.perf_counter()-tick,
                     peak_gib=torch.cuda.max_memory_allocated(device)/2**30)
        torch.cuda.reset_peak_memory_stats(device)
        for table in tables.values():
            for key in table.metrics:
                table.metrics[key] = 0
        dist.barrier()
        start = time.perf_counter()
        losses, step_ms, phases = [], [], []
        for index in range(args.warmup, args.warmup+args.steps):
            tick = time.perf_counter()
            value, phase = run_step(index)
            step_ms.append((time.perf_counter()-tick)*1000)
            step_records[-1]["seconds_including_validation"] = step_ms[-1]/1000
            _append_step_record(out, rank, step_records[-1])
            losses.append(value)
            phases.append(phase)
            progress(phase="timed", step=index-args.warmup, loss=value, ms=step_ms[-1],
                     peak_gib=torch.cuda.max_memory_allocated(device)/2**30)
        elapsed = time.perf_counter() - start
        failure_context["phase"] = "final_summary"
        parameter_samples = {}
        for name in ("lm_head.weight", f"model.layers.{args.layers[0]}.engram.proj.weight"):
            parameter_samples[name] = local_tensor(dict(model.named_parameters())[name]).detach().flatten()[:512].float().cpu().tolist()
        table_samples = {}
        for name, table in tables.items():
            grad = table.weight.grad
            ids = (grad._indices()[0].unique()[:128] if grad is not None
                   else torch.empty(0, dtype=torch.long, device=table.weight.device))
            table_samples[name] = {"ids": ids.cpu().tolist(), "values": table.weight.detach()[ids, :8].float().cpu().tolist()}
        result = {"rank": rank, "variant": args.variant, "world_size": world, "ep": args.ep, "cp": 1,
            "batch_size": args.batch_size, "microbatches": args.microbatches, "steps": args.steps,
            "optimizer_foreach": args.optimizer_foreach,
            "seed": args.seed, "deduplicate": args.deduplicate, "step_records": step_records,
            "final_samples": final_samples(initial_dense, tables, last_table_probes),
            "sequence_length": sequence_length, "global_batch": global_batch,
            "data": data["metadata"], "engram_config": asdict(config), "policy": asdict(policy),
            "initialization_verification": (table_initializer.records if table_initializer is not None else []),
            "engram_parameters": sum(t.global_num_embeddings*t.embedding_dim for t in tables.values()),
            "local_parameter_memory": parameter_memory(model, optimizer),
            "load_seconds": load_seconds, "load_peak_gib": load_peak/2**30,
            "logical_read_gib": model.load_bytes_read/2**30, "load_spans": load_spans,
            "elapsed_seconds": elapsed, "step_ms": step_ms, "losses": losses, "warmup_losses": warmup_losses,
            "phase_host_ms": phases, "peak_allocated_gib": torch.cuda.max_memory_allocated(device)/2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device)/2**30,
            "cpu_max_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == "darwin" else 1024)/2**30,
            "pinned_staging_allocated_bytes": sum(getattr(t.store, "_stager").allocated_bytes for t in tables.values() if hasattr(t.store, "_stager")),
            "engram_metrics": {name: dict(t.metrics) for name, t in tables.items()},
            "cache_allocated_bytes": sum(getattr(t.store, "cache", None).allocated_bytes
                for t in tables.values() if getattr(t.store, "cache", None) is not None),
            "table_updates_first_step": table_updates, "parameter_samples": parameter_samples,
            "table_samples": table_samples,
            "lm_head_sample_update": float((local_tensor(model.lm_head.weight).detach().flatten()[head_indices].float().cpu()-initial).abs().max()),
            "lm_head_sample_count": head_indices.numel()}
        if args.trace_dir:
            dist.barrier()
            trace_path = Path(args.trace_dir)
            trace_path.mkdir(parents=True, exist_ok=True)
            # Profiles are separate from the uninstrumented throughput samples.
            capture = (torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA], record_shapes=False, profile_memory=True,
                experimental_config=torch._C._profiler._ExperimentalConfig(profile_all_threads=True))
                if rank == 0 else nullcontext())
            from dsv41_train.profiling import model_regions
            with model_regions(model), capture as prof:
                run_step(args.warmup+args.steps, tracing=True, verify=False)
            if rank == 0:
                from dsv41_train.profiling import summarize_profiler
                trace = trace_path / "rank-0.json"
                prof.export_chrome_trace(str(trace))
                stats = summarize_profiler(prof, 1, trace)
                (trace_path / "rank-0-summary.json").write_text(json.dumps(stats, indent=2) + "\n")
            dist.barrier()
        _require_all(update_validation is not None and validation_checks["update_steps_verified"] >= args.warmup + args.steps
                     and validation_checks["full_dense_steps_verified"] >= args.warmup + args.steps,
                     runtime, "benchmark update verification did not complete")
        result.update(validation_passed=True, validation_checks=dict(validation_checks),
                      update_validation=update_validation,
                      validation_note="Finite loss/gradients, all elements of selected dense tensors and all unique touched rows of every table checked at every warmup/measured optimizer step; bounded samples are diagnostic only.")
        (out / f"rank-{rank}.json").write_text(json.dumps(result, indent=2) + "\n")
        gathered = [None] * world
        dist.all_gather_object(gathered, result)
        storage_metrics = validate_storage_metrics(gathered, policy, tables)
        timing = timing_statistics([max(row["step_ms"][i] for row in gathered)/1000 for i in range(args.steps)])
        validation_timing = timing_statistics([max(row["step_records"][args.warmup+i]["validation_seconds"] for row in gathered)
                                               for i in range(args.steps)])
        summary = {"variant": args.variant, "validation_passed": all(r["validation_passed"] for r in gathered),
            "workload": workload, "engram_format": addressing,
            "global_losses_all_steps": [row["global_loss"] for row in step_records[:args.warmup+args.steps]],
            "engram_metrics_global_timed_steps": storage_metrics,
            "measured_timing_including_validation": timing,
            "measured_validation_timing": validation_timing,
            "scope": "Qwen pretrained backbone plus random Engram tables: placement/cache efficiency only; no quality inference.",
            "limitations": ["Identity Qwen token mapping and one hidden stream; only the n-gram partition/row-width pattern mimics V4.1.",
                "Timings include synchronization and complete selected-dense/touched-row update checks; final cross-variant tensor values remain bounded samples.",
                "Tracked bytes exclude row IDs, NCCL and other device traffic; CPU RSS includes loading and file mappings.",
                "Caching also deduplicates missing rows; traffic reduction must not be attributed solely to hit rate.",
                "Random table initialization is reproducible for fixed seed and sharding topology; cross-world initial weights need not match."],
            "optimizer_foreach": args.optimizer_foreach,
            "tokens_per_second": global_batch * sequence_length * args.steps/timing["total_seconds"],
            "supervised_tokens_per_second": sum(row["global_supervised_tokens"] for row in step_records[args.warmup:args.warmup+args.steps])/timing["total_seconds"],
            "peak_allocated_gib": max(r["peak_allocated_gib"] for r in gathered),
            "peak_reserved_gib": max(r["peak_reserved_gib"] for r in gathered),
            "global_losses": [sum(r["losses"][i] for r in gathered)/world for i in range(args.steps)], "ranks": gathered}
        if args.reference:
            try:
                summary["comparison"] = _compare_qwen_summaries(summary, json.loads(Path(args.reference).read_text()),
                    **{key: getattr(args, key) for key in ("loss_atol", "loss_rtol", "sample_atol", "sample_rtol")})
                summary["validation_passed"] = summary["comparison"]["validation_passed"]
            except (ValueError, OSError, KeyError, TypeError) as error:
                summary["comparison"] = {"validation_passed": False, "error": str(error)}
                summary["validation_passed"] = False
        if rank == 0:
            (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            progress(phase="summary", **{key: summary[key] for key in
                ("variant", "validation_passed", "tokens_per_second", "supervised_tokens_per_second",
                 "peak_allocated_gib", "peak_reserved_gib", "global_losses")},
                median_step_seconds=timing["median_seconds"], output=str(out))
        if not summary["validation_passed"]:
            raise RuntimeError("benchmark reference comparison failed; inspect summary.json")
    except Exception as error:
        # Preserve completed rank-local records plus the failing-step context;
        # a late validation failure must not erase all trajectory evidence.
        if output_initialized:
            try:
                _write_error_record(out, rank, locals().get("failure_context", {"phase": "loading"}), error)
            except Exception as diagnostic_error:
                print(f"rank {rank}: could not preserve error diagnostic: {diagnostic_error}", file=sys.stderr, flush=True)
        raise
    finally:
        if model is not None:
            for table in model.model.engram_tables.values():
                table.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
