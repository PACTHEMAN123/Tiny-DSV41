"""Real DSV4 prefix/window SFT Engram placement benchmark, without saving DCP.

Default prefix2 includes the original pretrained layer-1 Engram table. Never
replace its dimensions with a synthetic small table. Each invocation loads one
policy in a fresh torchrun process group; the caller must check GPU release after
exit. Repeated authored smoke examples validate execution, not quality or typical
cache reuse. Timings include correctness checks, whose measured cost is reported.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import resource
import statistics
import sys
import time

import torch
import torch.distributed as dist

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dsv41_train.activation import ActivationCheckpointConfig
from dsv41_train.layout import local_tensor, parameter_memory
from dsv41_train.loading import LoadPolicy
from dsv41_train.models.dsv4 import load_dsv41_backbone_window
from dsv41_train.models.dsv4.config import DeepSeekV41Config
from dsv41_train.models.dsv4.engram import EngramPolicy
from dsv41_train.models.dsv4.materialize import FSDPMaterializer
from dsv41_train.models.dsv4.parallel import build_parallelism
from dsv41_train.models.dsv4.policies import GradientPolicy
from dsv41_train.optimization import OptimizerPolicy, build_optimizer
from dsv41_train.runtime import initialize_runtime
from dsv41_train.sft_data import SFTDataset
from train_sft import (_collective_finite, _engram_update_metrics, _engram_update_probes,
                       _finite_gradients, _probe_indices, _source_identity, loss_scale)
from scripts.engram_update_validation import capture_touched_rows, verify_touched_updates


VARIANTS = ("gpu", "cpu_sync", "cpu_async", "cpu_cache")
DENSE_SUFFIXES = ("attention.q_a.weight", "attention.o_b.weight", "engram.proj.weight",
                  "post_attention_norm.weight")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data", required=True, help="Local JSONL accepted by SFTDataset")
    parser.add_argument("--output", required=True, help="Absent or empty output directory")
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--reference", help="Optional previous summary.json for same-workload comparisons")
    parser.add_argument("--start-layer", type=int, default=0)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--cache-quantization", choices=("none", "hf"), default="none",
                        help="Attention KV/index value quantization, independent of Engram row caching")
    parser.add_argument("--ep-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--require-full-sequences", action=argparse.BooleanOptionalAction, default=True,
                        help="Reject padding; use packed pretokenized JSONL for real seq-len workloads")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--microbatches", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=20, help="Measured optimizer steps after warmup")
    parser.add_argument("--learning-rate", type=float, default=.01)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--cache-mib", type=int, default=8, help="Per table/rank; used only by cpu_cache")
    parser.add_argument("--staging-mib", type=int, default=128)
    parser.add_argument("--load-chunk-rows", type=int, default=131072)
    parser.add_argument("--load-pipeline", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deduplicate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--optimizer-foreach", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--loss-atol", type=float, default=.02)
    parser.add_argument("--loss-rtol", type=float, default=.01)
    parser.add_argument("--sample-atol", type=float, default=.0002)
    parser.add_argument("--sample-rtol", type=float, default=.02)
    args = parser.parse_args(argv)
    if min(args.num_layers, args.ep_size, args.batch_size, args.microbatches, args.warmup,
           args.steps, args.staging_mib, args.load_chunk_rows) < 1 or args.seq_len < 2:
        parser.error("layers/EP/batch/accumulation/warmup/steps/staging/chunk rows must be positive; seq-len >= 2")
    if args.start_layer < 0 or args.seed < 0 or args.cache_mib < 0 or (args.variant == "cpu_cache" and args.cache_mib == 0):
        parser.error("start-layer/seed/cache-mib must be nonnegative; cpu_cache needs positive cache-mib")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning-rate must be positive and finite")
    if not all(math.isfinite(getattr(args, key)) and getattr(args, key) >= 0
               for key in ("loss_atol", "loss_rtol", "sample_atol", "sample_rtol")):
        parser.error("comparison tolerances must be finite and nonnegative")
    return args


def selected_tables(config, start, count):
    require(start >= 0 and count >= 1 and start + count <= config.num_hidden_layers, "layer window outside checkpoint")
    require(start == 0 or start in config.kv_source_layer_ids, "non-prefix window must begin at a KV source layer")
    result = {str(layer): rows for layer, rows in zip(config.engram_layer_ids, config.engram_num_embeddings)
              if start <= layer < start + count}
    require(result, "selected real prefix/window contains no Engram; use prefix2 or a window containing a configured table")
    return result


def policies(args):
    cpu = args.variant != "gpu"
    policy = EngramPolicy("cpu" if cpu else "cuda", "sync" if args.variant in ("gpu", "cpu_sync") else "async",
                          args.deduplicate, args.staging_mib * 1024**2,
                          args.cache_mib * 1024**2 if args.variant == "cpu_cache" else 0)
    loading = LoadPolicy("direct_shard", args.load_chunk_rows, args.staging_mib * 1024**2, args.load_pipeline)
    return policy, loading


def dense_probes(model):
    probes = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or not (name == "model.norm.weight" or name.endswith(DENSE_SUFFIXES)):
            continue
        value = local_tensor(parameter).detach().reshape(-1)
        if value.numel():
            indices = _probe_indices(value.numel(), device=value.device)
            probes.append((name, parameter, indices, value[indices].float().clone()))
    require(probes, "no selected non-embedding dense parameters found")
    return probes


def metric_snapshot(tables):
    return {name: dict(table.metrics) for name, table in tables.items()}


def metric_delta(after, before):
    require(set(after) == set(before), "metric table keys changed within a step")
    result = {}
    for name, values in after.items():
        require(set(values) == set(before[name]), "metric counter keys changed within a step")
        result[name] = {key: value - before[name][key] for key, value in values.items()}
        require(all(math.isfinite(value) and value >= 0 for value in result[name].values()),
                "metric counter reset or became non-finite within a step")
    return result


def timing_statistics(seconds):
    require(bool(seconds) and all(math.isfinite(value) and value > 0 for value in seconds),
            "measured step durations must be positive finite values")
    ordered = sorted(seconds)
    position = .9 * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    return {"step_seconds": list(seconds), "steps": len(seconds), "total_seconds": sum(seconds),
            "median_seconds": statistics.median(seconds), "mean_seconds": statistics.mean(seconds),
            "p90_seconds": ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower),
            "min_seconds": ordered[0], "max_seconds": ordered[-1],
            "aggregation": "maximum duration across ranks for each measured optimizer step"}


def hash_layout(hashing, layer_ids, tables):
    """Small exact addressing identity; never read/copy embedding weight contents."""
    def digest(tensor):
        return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    rows = {str(layer): int(primes.sum()) for layer, primes in zip(layer_ids, hashing.primes)}
    require(set(rows) == set(tables), "hash/table layer IDs differ")
    require(all(rows[name] == table.global_num_embeddings for name, table in tables.items()),
            "table row count differs from exact n-gram partition sum")
    return {"layer_ids": list(layer_ids), "partition_rows": hashing.primes.cpu().tolist(),
            "partition_offsets": hashing.offsets.cpu().tolist(), "table_rows": rows,
            "hash_multipliers": hashing.multipliers.cpu().tolist(), "max_ngram": hashing.max_ngram,
            "compressed_vocab_size": hashing.compressed_vocab_size, "compressed_pad_id": hashing.pad_id,
            "token_map_sha256_int64": digest(hashing.token_map),
            "table_width": {name: table.embedding_dim for name, table in tables.items()},
            "parity_scope": "Exact implementation identity across placement variants; does not establish independent HF numerical parity."}


def gradient_summary(probes, tables):
    """Bounded dense gradient samples; all touched sparse row gradients per table."""
    dense = {}
    for name, parameter, indices, _ in probes:
        gradient = parameter.grad
        require(gradient is not None, f"selected parameter lacks gradient: {name}")
        values = local_tensor(gradient).detach().reshape(-1)[indices].float()
        dense[name] = {"sample_l2": values.double().square().sum().sqrt().item(),
                       "sample_max_abs": values.abs().max().item(), "sample_nonzero": int(values.count_nonzero()),
                       "sample_elements": values.numel()}
    sparse = {}
    for name, table in tables.items():
        gradient = table.weight.grad
        if gradient is None:
            sparse[name] = {"raw_nnz": 0, "unique_rows": 0, "l2": 0., "max_abs": 0., "nonzero": 0}
            continue
        gradient = local_tensor(gradient)
        require(gradient.is_sparse, f"table {name} must retain sparse gradients")
        values = gradient.detach().float().coalesce().values()
        sparse[name] = {"raw_nnz": gradient._nnz(), "unique_rows": values.shape[0],
                        "l2": values.double().square().sum().sqrt().item(),
                        "max_abs": values.abs().max().item() if values.numel() else 0.,
                        "nonzero": int(values.count_nonzero())}
    return {"dense_bounded_samples": dense, "tables_full_touched_gradients": sparse}


def dense_updates(probes):
    return {name: (local_tensor(parameter).detach().reshape(-1)[indices].float() - before).abs().max().item()
            for name, parameter, indices, before in probes}


def final_samples(probes, tables, touched):
    dense = {name: {"indices": indices.cpu().tolist(),
                    "values": local_tensor(parameter).detach().reshape(-1)[indices].float().cpu().tolist()}
             for name, parameter, indices, _ in probes}
    sparse = {}
    for name, table in tables.items():
        indices = touched[name][0][:32] if name in touched else torch.empty(0, dtype=torch.int64, device=table.weight.device)
        sparse[name] = {"global_row_ids": (indices + getattr(table, "row_start", 0)).cpu().tolist(),
                        "columns": list(range(min(table.embedding_dim, 8))),
                        "values": local_tensor(table.weight).detach()[indices, :8].float().cpu().tolist()}
    sample = {"dense_nonembedding": dense, "tables": sparse}
    sample["bounded_sample_sha256"] = hashlib.sha256(json.dumps(sample, sort_keys=True).encode()).hexdigest()
    return sample


def validate_storage_metrics(ranks, policy, table_names):
    """Fail if a claimed storage/cache path was not actually exercised."""
    totals = {name: {} for name in table_names}
    for row in ranks:
        require(row["cache_allocated_bytes"] <= policy.cache_bytes * len(table_names), "cache allocation exceeds per-table/rank budget")
        require(row["pinned_staging_allocated_bytes"] <= policy.staging_bytes * len(table_names), "pinned staging allocation exceeds budget")
        require(set(row["engram_metrics"]) == set(table_names), "Engram metric table names differ")
        for name, metrics in row["engram_metrics"].items():
            for key, value in metrics.items():
                require(type(value) in (int, float) and math.isfinite(value) and value >= 0, f"table {name}: invalid {key}")
                totals[name][key] = totals[name].get(key, 0) + value
            if policy.cache_bytes:
                require(metrics["cache_hits"] + metrics["cache_misses"] == metrics["lookup_rows"], f"table {name}: cache requests do not match lookup rows")
                require(metrics["cache_evictions"] <= metrics["cache_fill_rows"] <= metrics["cache_misses"], f"table {name}: cache fill/eviction counts inconsistent")
            else:
                require(row["cache_allocated_bytes"] == 0 and all(value == 0 for key, value in metrics.items() if key.startswith("cache_")), "cache activity recorded with cache disabled")
    for name, metrics in totals.items():
        require(metrics["lookup_rows"] > 0, f"table {name}: no lookup activity")
        if policy.device == "cpu":
            require(metrics["h2d_bytes"] > 0 and metrics["d2h_bytes"] > 0, f"table {name}: CPU value/gradient transfers were not exercised")
        else:
            require(metrics["h2d_bytes"] == metrics["d2h_bytes"] == 0, "GPU source table unexpectedly records CPU row transfers")
        if policy.cache_bytes:
            require(sum(row["cache_allocated_bytes"] for row in ranks) > 0
                    and metrics["cache_invalidated_rows"] > 0 and metrics["cache_metadata_h2d_bytes"] > 0,
                    f"table {name}: cache allocation/invalidation/metadata transfer was not exercised")
        requests = metrics["cache_hits"] + metrics["cache_misses"]
        metrics["cache_hit_fraction"] = metrics["cache_hits"] / requests if requests else 0.
        metrics["row_and_cache_metadata_h2d_bytes"] = metrics["h2d_bytes"] + metrics["cache_metadata_h2d_bytes"]
    return totals


def numeric_difference(actual, expected, atol, rtol):
    left, right = torch.tensor(actual, dtype=torch.float64), torch.tensor(expected, dtype=torch.float64)
    require(left.shape == right.shape, "comparison sample shapes differ")
    require(bool(torch.isfinite(left).all() & torch.isfinite(right).all()), "non-finite comparison value")
    delta = (left - right).abs()
    outside = delta > atol + rtol * right.abs()
    return {"passed": not bool(outside.any()), "max_abs_difference": delta.max().item() if delta.numel() else 0.,
            "outside_tolerance": int(outside.sum()), "elements": delta.numel()}


def compare_summaries(actual, expected, *, loss_atol=.02, loss_rtol=.01, sample_atol=.0002, sample_rtol=.02):
    require(actual["workload"] == expected["workload"], "reference workload/model/data identity differs")
    require(actual.get("engram_format") == expected.get("engram_format"), "reference hash/token-map/table layout differs")
    require(actual["validation_passed"] and expected["validation_passed"], "reference or current execution validation failed")
    reference = {row["rank"]: row for row in expected["ranks"]}
    require(set(reference) == {row["rank"] for row in actual["ranks"]}, "reference rank IDs differ")
    losses = numeric_difference(actual["global_losses_all_steps"], expected["global_losses_all_steps"], loss_atol, loss_rtol)
    ranks = {}
    for row in actual["ranks"]:
        other = reference[row["rank"]]
        records = {}
        for kind, ids_key in (("dense_nonembedding", "indices"), ("tables", "global_row_ids")):
            samples, wanted = row["final_samples"][kind], other["final_samples"][kind]
            require(set(samples) == set(wanted), "reference parameter/table sample keys differ")
            for name, sample in samples.items():
                require(sample[ids_key] == wanted[name][ids_key], "reference sample locations differ")
                require(sample.get("columns") == wanted[name].get("columns"), "reference table sample columns differ")
                records[f"{kind}/{name}"] = numeric_difference(sample["values"], wanted[name]["values"], sample_atol, sample_rtol)
        own_steps, old_steps = row.get("step_records", row["steps"]), other.get("step_records", other["steps"])
        require(len(own_steps) == len(old_steps), "reference step count differs")
        trajectories = []
        for step, baseline in zip(own_steps, old_steps):
            require(step["step"] == baseline["step"] and step["global_supervised_tokens"] == baseline["global_supervised_tokens"]
                    and step["global_input_tokens"] == baseline["global_input_tokens"], "reference token schedule differs")
            gradients = {}
            for group in ("dense_bounded_samples", "tables_full_touched_gradients"):
                own, old = step["gradients"][group], baseline["gradients"][group]
                require(set(own) == set(old), "gradient summary tensor keys differ")
                keys = ("sample_l2", "sample_max_abs") if group == "dense_bounded_samples" else ("l2", "max_abs")
                for name, stats in own.items():
                    gradients[f"{group}/{name}"] = numeric_difference([stats[key] for key in keys], [old[name][key] for key in keys], sample_atol, sample_rtol)
            updates = numeric_difference([step["dense_update_global_max"], *[step["table_updates"][name]["parameter_delta_max"] for name in sorted(step["table_updates"])]],
                                         [baseline["dense_update_global_max"], *[baseline["table_updates"][name]["parameter_delta_max"] for name in sorted(baseline["table_updates"])]], sample_atol, sample_rtol)
            full_updates = {}
            if actual["workload"].get("update_validation", {}).get("method") == "all-unique-touched-rows-v1":
                own_full, old_full = step.get("full_table_updates", {}), baseline.get("full_table_updates", {})
                require(set(own_full) == set(old_full) == set(step["table_updates"]), "reference full touched-update proofs missing/different")
                for name, proof in own_full.items():
                    prior = old_full[name]
                    for item in (proof, prior):
                        require(item["scope"] == "all_unique_sparse_touched_rows_all_columns"
                                and item["all_unique_rows_checked"] is True and item["finite_before_after"] is True
                                and item["global_changed_elements"] > 0 and item["global_max_abs_delta"] > 0,
                                "reference full touched-update proof failed")
                    require(proof["global_rows_checked"] == prior["global_rows_checked"]
                            and proof["global_elements_checked"] == prior["global_elements_checked"],
                            "reference full touched-update coverage differs")
                    full_updates[name] = numeric_difference([proof["global_max_abs_delta"]], [prior["global_max_abs_delta"]], sample_atol, sample_rtol)
                    full_updates[name].update(global_changed_elements=proof["global_changed_elements"],
                        reference_global_changed_elements=prior["global_changed_elements"],
                        changed_element_count_difference=proof["global_changed_elements"]-prior["global_changed_elements"])
            trajectories.append({"step": step["step"], "gradient_summaries": gradients, "update_summaries": updates,
                                 "full_touched_update_summaries": full_updates})
        ranks[str(row["rank"])] = {"samples": records, "steps": trajectories}
    passed = losses["passed"] and all(item["passed"] for row in ranks.values() for item in row["samples"].values())
    gradient_parity = all(item["passed"] for row in ranks.values() for step in row["steps"] for item in step["gradient_summaries"].values())
    update_parity = all(step["update_summaries"]["passed"] for row in ranks.values() for step in row["steps"])
    full_update_parity = all(item["passed"] for row in ranks.values() for step in row["steps"]
                             for item in step["full_touched_update_summaries"].values())
    return {"validation_passed": passed and gradient_parity and update_parity and full_update_parity, "losses": losses, "ranks": ranks,
            "tolerances": dict(loss_atol=loss_atol, loss_rtol=loss_rtol, sample_atol=sample_atol, sample_rtol=sample_rtol)}


def benchmark(args, runtime):
    output = Path(args.output)
    require(not output.exists() or (output.is_dir() and not any(output.iterdir())), "output directory must be absent or empty")
    config = DeepSeekV41Config.from_json(Path(args.model_path) / "config.json")
    sizes = selected_tables(config, args.start_layer, args.num_layers)
    require(runtime.world_size % args.ep_size == 0 and config.n_routed_experts % args.ep_size == 0, "EP must divide world size and expert count")
    require(args.seq_len <= config.max_position_embeddings, "sequence length exceeds checkpoint limit")
    dataset = SFTDataset.from_jsonl(args.data, max_length=args.seq_len,
        tokenizer_path=Path(args.model_path) / "tokenizer.json", pad_token_id=config.pad_token_id,
        bos_token_id=config.bos_token_id, eos_token_id=config.eos_token_id)
    require(not args.require_full_sequences or bool(dataset.attention_mask.all()),
            "require-full-sequences rejects padding: prepare packed input_ids/labels JSONL at the requested seq-len")
    active = dataset.labels[dataset.labels.ne(-100)]
    require(dataset.input_ids.min() >= 0 and dataset.input_ids.max() < config.vocab_size
            and active.min() >= 0 and active.max() < config.vocab_size, "dataset token outside vocabulary")
    policy, loading = policies(args)
    workload = {"source": _source_identity(args.model_path), "data": dataset.identity,
                "world_size": runtime.world_size, "ep_size": args.ep_size, "cp_size": 1,
                "start_layer": args.start_layer, "num_layers": args.num_layers, "table_rows": sizes,
                "table_width": config.engram_head_dim, "seed": args.seed,
                "seq_len": args.seq_len, "batch_size": args.batch_size, "microbatches": args.microbatches,
                "require_full_sequences": args.require_full_sequences,
                "warmup": args.warmup, "steps": args.steps, "learning_rate": args.learning_rate,
                "deduplicate": args.deduplicate, "optimizer_foreach": args.optimizer_foreach,
                "loading": asdict(loading), "activation_checkpoint": "full", "router_aux_loss_coef": 0.}
    workload["cache_quantization"] = args.cache_quantization
    workload["update_validation"] = {
        "method": "all-unique-touched-rows-v1", "chunk_rows": 8192,
        "helper_sha256": hashlib.sha256(Path(__file__).with_name("engram_update_validation.py").read_bytes()).hexdigest(),
    }
    identities = [None] * runtime.world_size
    dist.all_gather_object(identities, workload)
    require(all(row == workload for row in identities), "workload differs across ranks")
    schedule = [[dataset.batch_for_step(step, micro, batch_size=args.batch_size, accumulation_steps=args.microbatches,
                 data_rank=runtime.rank, data_world_size=runtime.world_size, seed=args.seed)
                 for micro in range(args.microbatches)] for step in range(args.warmup + args.steps)]
    counts = torch.tensor([[sum(int(batch["supervised_tokens"]) for batch in batches),
                            sum(int(batch["attention_mask"].sum()) for batch in batches)] for batches in schedule],
                          dtype=torch.int64, device=runtime.device)
    dist.all_reduce(counts)
    require(bool((counts[:, 0] > 0).all()), "schedule contains zero shifted supervised targets")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    meshes, cp, dispatcher = build_parallelism(1, args.ep_size, runtime.device.type)
    model = None
    torch.cuda.reset_peak_memory_stats(runtime.device)
    started = time.perf_counter()
    if runtime.is_main:
        print(json.dumps({"phase": "load_start", "variant": args.variant, "real_table_parameters": sum(sizes.values()) * config.engram_head_dim,
                          "global_bf16_table_gib": sum(sizes.values()) * config.engram_head_dim * 2 / 2**30}), flush=True)
    try:
        model = load_dsv41_backbone_window(args.model_path, start_layer=args.start_layer, num_layers=args.num_layers,
            device=runtime.device, dtype=torch.bfloat16, context_parallel=cp, token_dispatcher=dispatcher,
            engram_mesh=meshes.engram, sparse_engram_gradients=True, engram_policy=policy,
            load_policy=loading, materializer=FSDPMaterializer(meshes), cache_quantization=args.cache_quantization)
        tables = model.model.engram_tables
        addressing = hash_layout(model.model.hash, model.model.engram_layer_ids, tables)
        require({name: table.global_num_embeddings for name, table in tables.items()} == sizes, "loaded table sizes differ from the real checkpoint")
        placement_ok = torch.tensor(all(local_tensor(table.weight).device.type == policy.device and table.sparse_gradients
                                       for table in tables.values()), device=runtime.device, dtype=torch.int32)
        dist.all_reduce(placement_ok, op=dist.ReduceOp.MIN)
        require(placement_ok.item(), "table placement/sparse-gradient policy differs from requested strategy")
        model.config.router_aux_loss_coef = 0.
        model.set_activation_checkpointing(ActivationCheckpointConfig("full"))
        model.train()
        torch.cuda.synchronize(runtime.device)
        load_seconds, load_peak = time.perf_counter() - started, torch.cuda.max_memory_allocated(runtime.device)
        optimizer = build_optimizer(model, OptimizerPolicy("sgd", args.learning_rate, args.optimizer_foreach))
        gradients = GradientPolicy(model, meshes)
        initial_probes = dense_probes(model)
        records, touched = [], {}
        table_maxima = {name: 0. for name in tables}
        dense_maximum = 0.
        if runtime.is_main:
            print(json.dumps({"phase": "loaded", "seconds": load_seconds, "load_peak_gib": load_peak / 2**30}), flush=True)
        for step, batches in enumerate(schedule):
            if step == args.warmup:
                for table in tables.values():
                    for key in table.metrics:
                        table.metrics[key] = 0
                torch.cuda.reset_peak_memory_stats(runtime.device)
            inputs = [{key: batch[key].to(runtime.device) for key in ("input_ids", "labels", "attention_mask")} for batch in batches]
            global_targets, global_tokens = counts[step].tolist()
            optimizer.zero_grad(set_to_none=True)
            gradients.begin_step()
            torch.cuda.synchronize(runtime.device)
            dist.barrier()
            tick = time.perf_counter()
            metrics_before = metric_snapshot(tables)
            validation_seconds = 0.
            probe_tick = time.perf_counter()
            probes = dense_probes(model)
            torch.cuda.synchronize(runtime.device)
            validation_seconds += time.perf_counter() - probe_tick
            total = torch.zeros((), dtype=torch.float64, device=runtime.device)
            for batch, source in zip(inputs, batches):
                result = model(**batch)
                torch.cuda.synchronize(runtime.device)
                probe_tick = time.perf_counter()
                _collective_finite(result.loss, runtime, "DSV4 SFT loss before backward")
                scale = loss_scale(int(source["supervised_tokens"]), global_targets, runtime.world_size)
                loss = result.loss * scale
                _collective_finite(loss, runtime, "scaled DSV4 SFT loss before backward")
                validation_seconds += time.perf_counter() - probe_tick
                loss.backward()
                total += result.loss.detach().double() * int(source["supervised_tokens"])
                del result, loss
            gradients.normalize()
            model.synchronize_engram()
            torch.cuda.synchronize(runtime.device)
            probe_tick = time.perf_counter()
            _finite_gradients(model, runtime)
            grad_stats = gradient_summary(probes, tables)
            touched = _engram_update_probes(tables)
            full_touched = capture_touched_rows(tables, runtime)
            torch.cuda.synchronize(runtime.device)
            validation_seconds += time.perf_counter() - probe_tick
            optimizer.step()
            torch.cuda.synchronize(runtime.device)
            probe_tick = time.perf_counter()
            updates = _engram_update_metrics(tables, touched, runtime)
            full_updates = verify_touched_updates(tables, full_touched, runtime)
            del full_touched
            local_updates = dense_updates(probes)
            maximum = torch.tensor(max(local_updates.values()), device=runtime.device)
            _collective_finite(maximum, runtime, "DSV4 dense update")
            dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
            dense_maximum = max(dense_maximum, maximum.item())
            for name, update in updates.items():
                table_maxima[name] = max(table_maxima[name], update["parameter_delta_max"])
            dist.all_reduce(total)
            total /= global_targets
            _collective_finite(total, runtime, "global DSV4 SFT loss")
            validation_seconds += time.perf_counter() - probe_tick
            torch.cuda.synchronize(runtime.device)
            seconds = time.perf_counter() - tick
            records.append({"step": step, "warmup": step < args.warmup, "global_loss": total.item(),
                            "global_supervised_tokens": global_targets, "global_input_tokens": global_tokens,
                            "seconds_including_validation": seconds, "validation_seconds": validation_seconds,
                            "gradients": grad_stats, "dense_updates": local_updates,
                            "dense_update_global_max": maximum.item(), "table_updates": updates,
                            "full_table_updates": full_updates,
                            "engram_metrics": metric_delta(metric_snapshot(tables), metrics_before)})
            if runtime.is_main:
                print(json.dumps({key: records[-1][key] for key in ("step", "warmup", "global_loss", "seconds_including_validation", "dense_update_global_max", "table_updates", "full_table_updates")}), flush=True)
            del inputs, probes
        require(dense_maximum > 0, "no actual dense update observed")
        cached = [table.store.cache for table in tables.values() if getattr(table.store, "cache", None) is not None]
        local = {"rank": runtime.rank, "validation_passed": True, "steps": records,
                 "load_seconds": load_seconds, "load_peak_gib": load_peak / 2**30,
                 "train_peak_allocated_gib": torch.cuda.max_memory_allocated(runtime.device) / 2**30,
                 "train_peak_reserved_gib": torch.cuda.max_memory_reserved(runtime.device) / 2**30,
                 "cpu_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
                 "parameter_memory": parameter_memory(model, optimizer), "loading": model.loading_metrics,
                 "engram_metrics": {name: dict(table.metrics) for name, table in tables.items()},
                 "cache_allocated_bytes": sum(cache.allocated_bytes for cache in cached),
                 "cache_budget_bytes_per_table": policy.cache_bytes,
                 "pinned_staging_allocated_bytes": sum(getattr(table.store, "_stager").allocated_bytes for table in tables.values() if hasattr(table.store, "_stager")),
                 "final_samples": final_samples(initial_probes, tables, touched),
                 "dense_update_max": dense_maximum, "table_update_maxima": table_maxima}
        gathered = [None] * runtime.world_size
        dist.all_gather_object(gathered, local)
        storage_metrics = validate_storage_metrics(gathered, policy, sizes)
        measured = [index for index, record in enumerate(records) if not record["warmup"]]
        timing = timing_statistics([max(row["steps"][index]["seconds_including_validation"] for row in gathered) for index in measured])
        validation_timing = timing_statistics([max(row["steps"][index]["validation_seconds"] for row in gathered) for index in measured])
        seconds = timing["total_seconds"]
        summary = {"validation_passed": True, "variant": args.variant, "policy": asdict(policy), "workload": workload,
                   "engram_format": addressing, "measured_timing_including_validation": timing,
                   "measured_validation_timing": validation_timing,
                   "engram_metrics_global_timed_steps": storage_metrics,
                   "global_losses_all_steps": [record["global_loss"] for record in records],
                   "measured_seconds_including_validation": seconds,
                   "input_tokens_per_second": sum(records[index]["global_input_tokens"] for index in measured) / seconds,
                   "supervised_tokens_per_second": sum(records[index]["global_supervised_tokens"] for index in measured) / seconds,
                   "peak_allocated_gib": max(row["train_peak_allocated_gib"] for row in gathered),
                   "peak_reserved_gib": max(row["train_peak_reserved_gib"] for row in gathered), "ranks": gathered,
                   "scope": "Real pretrained DSV4 selected layers and original Engram tables; local SFT answer masks. No checkpoint written.",
                   "limitations": ["Timings include finite/gradient/update validation and explicit synchronization, not an uninstrumented performance result.",
                                   "Every table has an actual numerical update verified over all unique sparse touched rows and all columns at every step; original bounded probes remain separate evidence.",
                                   "Dense gradients/weights are bounded non-embedding samples; Engram gradient norms cover touched sparse rows, final values sample up to 32 rows and 8 columns.",
                                   "Small or repeated authored datasets cannot establish model quality or representative cache reuse.",
                                   "CPU peak RSS includes loading and file mappings; per-rank peaks are not simultaneous node memory.",
                                   "Tracked transfers cover table values, sparse backward values and cache metadata, not IDs, NCCL or total PCIe traffic.",
                                   "GPU process exit and post-run memory release must be verified by the external launcher."]}
        if args.reference:
            try:
                summary["comparison"] = compare_summaries(summary, json.loads(Path(args.reference).read_text()),
                    **{key: getattr(args, key) for key in ("loss_atol", "loss_rtol", "sample_atol", "sample_rtol")})
                summary["validation_passed"] = summary["comparison"]["validation_passed"]
            except (ValueError, OSError, KeyError, TypeError) as error:
                summary["comparison"] = {"validation_passed": False, "error": str(error)}
                summary["validation_passed"] = False
        if runtime.is_main:
            output.mkdir(parents=True, exist_ok=True)
            for row in gathered:
                (output / f"rank-{row['rank']}.json").write_text(json.dumps(row, indent=2, allow_nan=False) + "\n")
            (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
            print(json.dumps({"phase": "complete", "validation_passed": summary["validation_passed"], "output": str(output)}), flush=True)
        return summary
    finally:
        if model is not None:
            model.close()


def main():
    args = parse_args()
    runtime = None
    try:
        runtime = initialize_runtime(require_distributed=True)
        torch.set_num_threads(1)
        return 0 if benchmark(args, runtime)["validation_passed"] else 1
    finally:
        if runtime is not None and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
