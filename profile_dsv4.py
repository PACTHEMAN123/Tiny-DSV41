"""Reproducible single-GPU or torchrun/FSDP training microbenchmarks."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import statistics
import subprocess
import time
from contextlib import nullcontext
from dataclasses import asdict, replace
from pathlib import Path

import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity, profile, record_function

from dsv41_train.features import add_training_options, adapter_config, checkpoint_policy
from dsv41_train.lora import inject_lora, load_adapter, save_adapter
from dsv41_train.models.dsv4 import DeepSeekV41Config, DeepSeekV41ForCausalLM, load_dsv41_backbone_window
from dsv41_train.models.dsv4.model import AttentionSinks, HyperConnection
from dsv41_train.models.dsv4.parallel import apply_fsdp2, apply_fsdp2_layer, apply_fsdp2_root, build_parallelism
from dsv41_train.parallel import ParallelMeshes
from dsv41_train.profiling import model_regions, summarize_profiler
from dsv41_train.runtime import initialize_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, default=3, help="layer count for tiny/model-path; config supplies its own")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--optimizer", choices=("adamw", "sgd"), default="adamw")
    parser.add_argument("--adamw-implementation", choices=("loop", "foreach", "fused"), default="loop")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--model-path")
    parser.add_argument("--config", help="JSON configuration for a randomly initialized model")
    parser.add_argument("--start-layer", type=int, default=0)
    parser.add_argument("--ep-size", type=int, default=1)
    parser.add_argument("--output", default="artifacts/profile.json")
    parser.add_argument("--trace-dir")
    parser.add_argument("--trace-steps", type=int, default=1)
    add_training_options(parser)
    args = parser.parse_args()
    if min(args.layers, args.batch_size, args.steps, args.cpu_threads, args.ep_size, args.trace_steps) < 1:
        parser.error("layers, batch-size, steps, cpu-threads, ep-size, and trace-steps must be positive")
    if args.seq_len < 2 or args.warmup < 1 or args.learning_rate <= 0:
        parser.error("seq-len >= 2, warmup >= 1, and learning-rate > 0 are required")
    if not args.model_path and (args.start_layer or args.ep_size != 1):
        parser.error("start-layer and expert parallelism require model-path")
    if args.model_path and args.config:
        parser.error("config and model-path are mutually exclusive")
    if args.optimizer != "adamw" and args.adamw_implementation != "loop":
        parser.error("adamw-implementation requires optimizer=adamw")
    return args


def tiny_profile_config(layers: int, seq_len: int, quantization: str) -> DeepSeekV41Config:
    encoder_layers = layers // 2
    ratios = [0] + [2] * encoder_layers + [1] * (layers - encoder_layers - 1)
    sources = [i for i, ratio in enumerate(ratios) if ratio and ratios[i - 1] != ratio]
    return replace(
        DeepSeekV41Config.tiny(), num_hidden_layers=layers,
        max_position_embeddings=max(seq_len, 128), compress_ratios=ratios,
        kv_source_layer_ids=sources, index_source_layer_ids=sources,
        candidate_source_layer_id=sources[-1] if sources else -1,
        engram_layer_ids=[1] if layers > 1 else [],
        engram_num_embeddings=[4096] if layers > 1 else [],
        cache_quantization=quantization,
    )


def main() -> None:
    args = parse_args()
    policy, lora = checkpoint_policy(args), adapter_config(args)
    torch.set_num_threads(args.cpu_threads)
    runtime = initialize_runtime(args.device)
    device = runtime.device
    cuda = device.type == "cuda"

    def synchronize() -> None:
        if cuda:
            torch.cuda.synchronize(device)

    def memory() -> dict[str, float | None]:
        return {
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20 if cuda else None,
            "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20 if cuda else None,
        }

    try:
        torch.manual_seed(args.seed)
        dtype = getattr(torch, args.dtype)
        if cuda:
            torch.cuda.set_device(device)
            torch.cuda.reset_peak_memory_stats(device)
        synchronize()
        started = time.perf_counter()
        if args.model_path:
            meshes = cp = dispatcher = None
            if runtime.distributed:
                meshes, cp, dispatcher = build_parallelism(1, args.ep_size, device.type)
            elif args.ep_size != 1:
                raise ValueError("expert parallelism requires torchrun")
            model = load_dsv41_backbone_window(
                args.model_path, start_layer=args.start_layer, num_layers=args.layers,
                device=device, dtype=dtype, context_parallel=cp, token_dispatcher=dispatcher,
                engram_mesh=meshes.engram if meshes else None,
                sparse_engram_gradients=args.optimizer == "sgd",
                layer_loaded=(lambda layer: apply_fsdp2_layer(layer, meshes)) if meshes else None,
                lora_config=lora, cache_quantization=args.cache_quantization,
                adapter_path=args.adapter_in,
            )
            if meshes:
                apply_fsdp2_root(model, meshes)
        else:
            config = (replace(DeepSeekV41Config.from_json(args.config),
                              cache_quantization=args.cache_quantization) if args.config else
                      tiny_profile_config(args.layers, args.seq_len, args.cache_quantization))
            model = DeepSeekV41ForCausalLM(config).to(device=device, dtype=dtype)
            for module in model.modules():
                if isinstance(module, (HyperConnection, AttentionSinks)):
                    module.float()
            if lora:
                inject_lora(model, lora)
            if args.adapter_in:
                load_adapter(model, args.adapter_in)
            if runtime.distributed:
                apply_fsdp2(model, ParallelMeshes.build(device_type=device.type))
        model.set_activation_checkpointing(policy)
        model.train()
        synchronize()
        load_seconds = time.perf_counter() - started
        load_memory = memory()
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer_class = torch.optim.AdamW if args.optimizer == "adamw" else torch.optim.SGD
        optimizer_options = {"foreach": args.adamw_implementation == "foreach"}
        if args.adamw_implementation == "fused":
            optimizer_options["fused"] = True
        optimizer = optimizer_class(parameters, lr=args.learning_rate, **optimizer_options)
        generator = torch.Generator(device=device).manual_seed(args.seed + runtime.rank)
        tokens = torch.randint(3, model.config.vocab_size, (args.batch_size, args.seq_len),
                               device=device, generator=generator)
        tokens[:, 0] = model.config.bos_token_id

        def step(regions: bool = False) -> tuple[list[float], float]:
            optimizer.zero_grad(set_to_none=True)
            events = [torch.cuda.Event(enable_timing=True) for _ in range(4)] if cuda else []
            stamps = [time.perf_counter()]
            if cuda:
                events[0].record()
            with record_function("train/forward") if regions else nullcontext():
                output = model(tokens, labels=tokens)
            if cuda:
                events[1].record()
            stamps.append(time.perf_counter())
            with record_function("train/backward") if regions else nullcontext():
                output.loss.backward()
            if cuda:
                events[2].record()
            stamps.append(time.perf_counter())
            with record_function("train/optimizer") if regions else nullcontext():
                optimizer.step()
            if cuda:
                events[3].record()
            stamps.append(time.perf_counter())
            synchronize()
            wall = (time.perf_counter() - stamps[0]) * 1000
            stages = ([events[i].elapsed_time(events[i + 1]) for i in range(3)] if cuda
                      else [(stamps[i + 1] - stamps[i]) * 1000 for i in range(3)])
            if not torch.isfinite(output.loss.detach()).item():
                raise RuntimeError("non-finite loss during profiling")
            return [*stages, wall], output.loss.detach().item()

        for _ in range(args.warmup):
            step()
        optimizer.zero_grad(set_to_none=True)
        gc.collect()
        synchronize()
        resident = torch.cuda.memory_allocated(device) / 2**20 if cuda else None
        if cuda:
            torch.cuda.reset_peak_memory_stats(device)
        durations = []
        for _ in range(args.steps):
            timing, loss = step()
            durations.append(timing)
        train_memory = memory()
        samples = torch.tensor(durations, device=device, dtype=torch.float64)
        if runtime.distributed:
            dist.all_reduce(samples, op=dist.ReduceOp.MAX)
        timings = {key: statistics.median(column) for key, column in zip(
            ("forward_ms", "backward_ms", "optimizer_ms", "step_ms"), zip(*samples.cpu().tolist()),
        )}
        traces = []
        operator_summary = None
        if args.trace_dir:
            trace_dir = Path(args.trace_dir)
            trace_dir.mkdir(parents=True, exist_ok=True)
            activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if cuda else [])
            optimizer.zero_grad(set_to_none=True)
            with model_regions(model), profile(activities=activities, profile_memory=True, record_shapes=True) as profiler:
                for _ in range(args.trace_steps):
                    step(regions=True)
            trace = trace_dir / f"rank-{runtime.rank}.json"
            profiler.export_chrome_trace(str(trace))
            traces.append(str(trace))
            table = profiler.key_averages().table(sort_by="self_cpu_time_total", row_limit=25)
            (trace_dir / f"rank-{runtime.rank}.txt").write_text(table, encoding="utf-8")
            summary = summarize_profiler(profiler, args.trace_steps, trace)
            operator_summary = trace_dir / f"rank-{runtime.rank}-summary.json"
            operator_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        if args.adapter_out:
            save_adapter(model, args.adapter_out)
        def tensor_bytes(tensor):
            local = tensor.to_local() if hasattr(tensor, "to_local") else tensor
            return local.numel() * local.element_size()

        state_bytes = sum(tensor_bytes(value) for state in optimizer.state.values()
                          for value in state.values() if isinstance(value, torch.Tensor))
        rank_metrics = {"rank": runtime.rank, "load_seconds": load_seconds,
                        "load_memory": load_memory, "resident_mib": resident,
                        **train_memory, "loss": loss, "traces": traces,
                        "operator_summary": str(operator_summary) if operator_summary else None,
                        "parameter_mib": sum(tensor_bytes(p) for p in model.parameters()) / 2**20,
                        "optimizer_state_mib": state_bytes / 2**20,
                        "gradient_mib": sum(tensor_bytes(p.grad) for p in parameters if p.grad is not None) / 2**20}
        per_rank = [rank_metrics]
        if runtime.distributed:
            per_rank = [None] * runtime.world_size
            dist.all_gather_object(per_rank, rank_metrics)
        revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
        report = {
            "environment": {"torch": torch.__version__, "cuda": torch.version.cuda,
                            "python": platform.python_version(),
                            "device": torch.cuda.get_device_name(device) if cuda else "cpu",
                            "world_size": runtime.world_size, "git_commit": revision.stdout.strip(),
                            "git_dirty": bool(dirty.stdout.strip()), "cpu_threads": args.cpu_threads},
            "configuration": {**vars(args), "model": model.config.to_dict(),
                              "activation_checkpointing": asdict(policy),
                              "adapter": asdict(lora) if lora else None},
            "parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in parameters),
            "timing": {**timings,
                       "input_tokens_per_second": args.batch_size * args.seq_len * runtime.world_size / (timings["step_ms"] / 1000)},
            "samples_ms": samples.cpu().tolist(), "per_rank": per_rank,
        }
        if runtime.is_main:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({"output": str(output), "timing": report["timing"],
                              "per_rank": per_rank}), flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
