"""Optional DSV4 trace attribution; never installed during performance timing."""

from collections import defaultdict
from contextlib import contextmanager
from functools import wraps
import json
import threading
import time

from torch.autograd import DeviceType
from torch.profiler import record_function


@contextmanager
def loading_regions(spans):
    """Instrument an isolated loader invocation, including its decode workers."""
    from concurrent.futures import ThreadPoolExecutor
    from .loading import reader, source
    from .models.dsv4 import checkpoint
    from torch import nn

    changes = []

    def install(owner, method, category, domain=None):
        original = getattr(owner, method)

        @wraps(original)
        def instrumented(*args, **kwargs):
            name = str(args[1] if len(args) > 1 else kwargs["name"]) if category == "load_read" else (
                ",".join(args[0].names) if category == "load_fill" else method)
            group = domain or ("engram" if ".engram.embed" in name else
                               "experts" if ".ffn.experts." in name else "dense")
            started = time.perf_counter()
            value = None
            try:
                with record_function(f"dsv41/{category}/{group}"):
                    value = original(*args, **kwargs)
                    return value
            finally:
                spans.append({"region": category, "domain": group, "name": name,
                              "start": started, "stop": time.perf_counter(),
                              "thread": threading.get_ident(),
                              "tensor_bytes": value.numel() * value.element_size()
                              if hasattr(value, "numel") else 0})

        changes.append((owner, method, original))
        setattr(owner, method, instrumented)

    try:
        for method in ("tensor", "tensor_rows"):
            install(reader.ShardedSafeTensorReader, method, "load_read")
        for module in (checkpoint, source):
            for method, domain in (("dequantize_fp4_rows", "experts"),
                                   ("dequantize_fp8_blocks", "dense"),
                                   ("dequantize_fp8_rows", "engram")):
                install(module, method, "load_decode", domain)
        install(source.WeightSource, "fill", "load_fill")
        install(nn.Module, "load_state_dict", "load_assign", "all")
        install(checkpoint, "build_compressed_token_map", "load_tokenizer", "all")
        for method in ("__init__", "submit", "shutdown"):
            install(ThreadPoolExecutor, method, "load_executor", "all")
        yield
    finally:
        for owner, method, original in reversed(changes):
            setattr(owner, method, original)


@contextmanager
def model_regions(model, lookup_spans=None):
    from .models.dsv4.model import (
        CompressedAttention, Compressor, DecoderLayer, Engram, HyperConnection,
        NgramHash, RowShardedEmbedding, SparseIndexer,
    )
    from .models.dsv4.moe import RoutedExperts, SharedExpert, TopKRouter
    from .models.qwen.model import Qwen3MoeAttention, Qwen3MoeExperts, Qwen3MoeMLP, Qwen3MoeDecoderLayer

    specifications = {
        CompressedAttention: (("forward", "attention_other"), ("_attend", "attention_core")),
        Compressor: (("forward", "compressor"),),
        SparseIndexer: (("forward", "indexer"),),
        HyperConnection: (("_compute", "mhc"),),
        RoutedExperts: (("forward", "dispatch_and_route"), ("_compute", "routed_experts")),
        SharedExpert: (("_compute", "shared_expert"),),
        TopKRouter: (("forward", "router"),),
        Engram: (("forward", "engram"),),
        NgramHash: (("forward", "engram_hash"),),
        RowShardedEmbedding: (("prefetch", "engram_prefetch"), ("resolve", "engram_resolve")),
        DecoderLayer: (("forward", "residual_and_norm"),),
        Qwen3MoeAttention: (("forward", "attention_core"),),
        Qwen3MoeExperts: (("forward", "routed_experts"),),
        Qwen3MoeMLP: (("forward", "dense_mlp"),),
        Qwen3MoeDecoderLayer: (("forward", "residual_and_norm"),),
    }
    changes = []

    def install(module, method, category, name):
        original = getattr(module, method)
        previous = module.__dict__.get(method)

        @wraps(original)
        def instrumented(*args, **kwargs):
            started = time.perf_counter() if lookup_spans is not None and category.startswith("engram_") else None
            try:
                with record_function(f"dsv41/{category}/{name}"):
                    return original(*args, **kwargs)
            finally:
                if started is not None:
                    lookup_spans.append({"region": category, "table": name,
                                         "start": started, "stop": time.perf_counter()})

        changes.append((module, method, previous))
        setattr(module, method, instrumented)

    try:
        for name, module in model.named_modules():
            for cls, methods in specifications.items():
                if isinstance(module, cls):
                    for method, category in methods:
                        install(module, method, category, name)
            if isinstance(module, RowShardedEmbedding):
                install(module.store, "lookup", "engram_store", name)
                install(module.router, "prepare", "engram_id_route", name)
                install(module.router, "complete", "engram_row_route", name)
        install(model, "forward", "loss_and_aux", "root")
        install(model.model, "forward", "backbone_other", "model")
        if hasattr(model.model, "_causal_mask"):
            install(model.model, "_causal_mask", "attention_mask", "model")
        embedding = getattr(model.model, "embedding", getattr(model.model, "embed_tokens", None))
        install(embedding, "forward", "embedding", "model.embedding")
        install(model.lm_head, "forward", "lm_head", "lm_head")
        yield
    finally:
        for module, method, previous in reversed(changes):
            if previous is None:
                delattr(module, method)
            else:
                setattr(module, method, previous)


def summarize_profiler(profiler, steps, trace_path=None):
    """Attribute backward kernels using autograd's forward sequence/thread IDs."""
    events = profiler.events()

    def ancestors(event):
        while event is not None:
            yield event
            event = event.cpu_parent

    def region(event):
        return next((parent.name for parent in ancestors(event)
                     if parent.name.startswith("dsv41/")), None)

    forward_regions = {}
    for event in events:
        if event.device_type == DeviceType.CPU and event.sequence_nr >= 0:
            owner = region(event)
            if owner and not any(parent.scope == 1 for parent in ancestors(event)):
                forward_regions.setdefault((event.sequence_nr, event.thread), owner)

    components = defaultdict(lambda: {"self_cpu_ms": 0.0, "cuda_ms": 0.0, "events": 0})
    layers = defaultdict(lambda: {"cuda_ms": 0.0, "self_cpu_ms": 0.0})
    for event in events:
        if event.device_type != DeviceType.CPU:
            continue
        parents = list(ancestors(event))
        stage = next((p.name.removeprefix("train/") for p in parents if p.name.startswith("train/")), "other")
        owner = region(event)
        backward = next((p for p in parents if p.scope == 1), None)
        if backward is not None or event.name.startswith("autograd::engine::evaluate_function:"):
            stage = "backward"
        if owner is None and backward is not None:
            owner = forward_regions.get((backward.sequence_nr, backward.fwd_thread))
        category = owner.split("/")[1] if owner else "unattributed"
        if stage == "optimizer":
            category = "optimizer"
        label = f"{stage}/{category}"
        row = components[label]
        row["self_cpu_ms"] += event.self_cpu_time_total / 1000 / steps
        row["cuda_ms"] += event.self_device_time_total / 1000 / steps
        row["events"] += 1 / steps
        if owner:
            layers[f"{stage}/{owner}"]["cuda_ms"] += event.self_device_time_total / 1000 / steps
            layers[f"{stage}/{owner}"]["self_cpu_ms"] += event.self_cpu_time_total / 1000 / steps

    operators = []
    for event in profiler.key_averages(group_by_input_shape=True):
        if event.device_type != DeviceType.CPU or event.key.startswith(("dsv41/", "train/")):
            continue
        operators.append({"name": event.key, "input_shapes": event.input_shapes,
                          "calls": event.count / steps,
                          "self_cpu_ms": event.self_cpu_time_total / 1000 / steps,
                          "cuda_ms": event.self_device_time_total / 1000 / steps})
    kernels = defaultdict(lambda: {"calls": 0, "cuda_ms": 0.0})
    activity = None
    if trace_path is not None:
        # CUDA FunctionEvents also contain overlapping GPU user annotations.
        # Chrome's kernel category excludes these and memory transfer activities.
        with open(trace_path, encoding="utf-8") as file:
            trace_events = json.load(file)["traceEvents"]
        activity = summarize_cuda_activity(trace_events, steps)
        for event in trace_events:
            if event.get("cat") == "kernel":
                kernels[event["name"]]["calls"] += 1 / steps
                kernels[event["name"]]["cuda_ms"] += event["dur"] / 1000 / steps
    return {
        "sampled_steps": steps,
        "note": "Trace-only attribution; CUDA sums can overlap and are not unprofiled wall times.",
        "components": dict(sorted(components.items(), key=lambda item: -item[1]["cuda_ms"])),
        "regions": dict(sorted(layers.items(), key=lambda item: -item[1]["cuda_ms"])),
        "top_operators_cuda": sorted(operators, key=lambda row: -row["cuda_ms"])[:40],
        "top_operators_cpu": sorted(operators, key=lambda row: -row["self_cpu_ms"])[:40],
        "kernel_total_ms": sum(row["cuda_ms"] for row in kernels.values()),
        "kernel_calls": sum(row["calls"] for row in kernels.values()),
        "cuda_activity": activity,
        "top_kernels": sorted(({"name": name, **values} for name, values in kernels.items()),
                              key=lambda row: -row["cuda_ms"])[:30],
    }


def _union(intervals):
    merged = []
    for start, stop in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(stop, merged[-1][1]))
        else:
            merged.append((start, stop))
    return merged


def _intersection_duration(left, right):
    left, right = _union(left), _union(right)
    first = second = 0
    total = 0.0
    while first < len(left) and second < len(right):
        total += max(0.0, min(left[first][1], right[second][1]) - max(left[first][0], right[second][0]))
        if left[first][1] <= right[second][1]:
            first += 1
        else:
            second += 1
    return total


def summarize_cuda_activity(events, steps):
    """Union device timelines before measuring overlap across CUDA streams."""
    intervals = defaultdict(list)
    totals = defaultdict(lambda: {"calls": 0, "duration_ms": 0.0, "bytes": 0})
    devices = set()
    for event in events:
        category = event.get("cat")
        if category not in {"kernel", "gpu_memcpy"} or "dur" not in event:
            continue
        name = event["name"]
        kind = ("nccl" if "nccl" in name.lower() else "compute") if category == "kernel" else (
            "h2d" if "HtoD" in name else "d2h" if "DtoH" in name else "d2d")
        device = event.get("args", {}).get("device", 0)
        devices.add(device)
        intervals[(device, kind)].append((event["ts"], event["ts"] + event["dur"]))
        totals[kind]["calls"] += 1 / steps
        totals[kind]["duration_ms"] += event["dur"] / 1000 / steps
        totals[kind]["bytes"] += event.get("args", {}).get("bytes", 0) / steps
    busy = sum(stop - start for device in devices for start, stop in _union(
        [interval for (owner, _), values in intervals.items() if owner == device for interval in values]))
    h2d_compute = sum(_intersection_duration(intervals[(device, "h2d")], intervals[(device, "compute")])
                      for device in devices)
    h2d_busy = sum(stop - start for device in devices for start, stop in _union(intervals[(device, "h2d")]))
    return {"activities": dict(totals), "device_busy_ms": busy / 1000 / steps,
            "h2d_compute_overlap_ms": h2d_compute / 1000 / steps,
            "h2d_compute_overlap_fraction": h2d_compute / h2d_busy if h2d_busy else 0.0,
            "note": "Device trace only; sums include concurrent streams and profiler-induced stalls. NCCL is excluded from compute overlap."}
