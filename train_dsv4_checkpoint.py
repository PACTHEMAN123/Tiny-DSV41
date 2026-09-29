"""Train a real DeepSeek V4.1 prefix from the released checkpoint."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import resource
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist

from dsv41_train.models.dsv4 import load_dsv41_backbone_window
from dsv41_train.models.dsv4.config import DeepSeekV41Config
from dsv41_train.models.dsv4.materialize import FSDPMaterializer
from dsv41_train.models.dsv4.policies import GradientPolicy, data_parallel_rank
from dsv41_train.memory_options import add_memory_options, memory_policies
from dsv41_train.optimization import OptimizerPolicy, build_optimizer
from dsv41_train.layout import parameter_memory
from dsv41_train.sharded_state import StateAdapter
from dsv41_train.checkpoint import CheckpointManager, TrainingState
from dsv41_train.features import add_training_options, adapter_config, checkpoint_policy
from dsv41_train.lora import save_adapter
from dsv41_train.models.dsv4.parallel import (
    build_parallelism,
)
from dsv41_train.runtime import Runtime, distributed_mean, initialize_runtime, local_tensor


def distributed_trace(runtime: Runtime, event: str) -> None:
    if os.environ.get("DSV41_DISTRIBUTED_TRACE"):
        print(json.dumps({"trace": event, "rank": runtime.rank}), flush=True)


def maximum_parameter_delta(before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
    if before.shape != after.shape:
        raise ValueError("parameter snapshots must have matching shapes")
    if after.numel() == 0:
        return torch.zeros((), dtype=torch.float32, device=after.device)
    return (after.float() - before.float()).abs().max()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--start-layer", type=int, default=0)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--optimizer", choices=("sgd", "adamw"), default="sgd")
    parser.add_argument("--cp-size", type=int, default=1)
    parser.add_argument("--ep-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metrics-file")
    parser.add_argument("--output-dir")
    parser.add_argument("--resume", action="store_true")
    add_training_options(parser)
    add_memory_options(parser)
    return parser.parse_args()


def validate_args(args: argparse.Namespace, world_size: int) -> None:
    model_path = Path(args.model_path)
    if not model_path.is_dir():
        raise FileNotFoundError(f"model path does not exist: {model_path}")
    for filename in ("config.json", "model.safetensors.index.json"):
        if not (model_path / filename).is_file():
            raise FileNotFoundError(f"checkpoint file does not exist: {model_path / filename}")
    if args.steps < 1 or args.batch_size < 1 or args.num_layers < 1 or args.seq_len < 2:
        raise ValueError(
            "steps, batch-size, and num-layers must be positive; seq-len must be at least 2"
        )
    if args.start_layer < 0:
        raise ValueError("start-layer must be non-negative")
    if args.learning_rate <= 0:
        raise ValueError("learning-rate must be positive")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("gradient-accumulation-steps must be positive")
    if args.cp_size < 1 or args.ep_size < 1:
        raise ValueError("cp-size and ep-size must be positive")
    if world_size % args.cp_size or world_size % args.ep_size:
        raise ValueError("cp-size and ep-size must both divide world size")
    if args.seq_len % args.cp_size:
        raise ValueError("seq-len must divide evenly across cp-size")
    if DeepSeekV41Config.from_json(model_path / "config.json").n_routed_experts % args.ep_size:
        raise ValueError("the checkpoint's experts must divide evenly across ep-size")


def train(args: argparse.Namespace, runtime: Runtime) -> dict[str, object]:
    with ExitStack() as cleanup:
        return _train(args, runtime, cleanup)


def _train(args, runtime, cleanup):
    validate_args(args, runtime.world_size)
    if args.resume and not args.output_dir:
        raise ValueError("resume requires output-dir")
    if args.resume and args.adapter_in:
        raise ValueError("adapter-in and resume are mutually exclusive")
    lora = adapter_config(args)
    policy = checkpoint_policy(args)
    loading, engram, fsdp = memory_policies(args)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats(runtime.device)
    meshes, context_parallel, token_dispatcher = build_parallelism(
        args.cp_size, args.ep_size, runtime.device.type
    )

    started = time.perf_counter()
    model = load_dsv41_backbone_window(
        args.model_path,
        start_layer=args.start_layer,
        device=runtime.device,
        dtype=torch.bfloat16,
        context_parallel=context_parallel,
        token_dispatcher=token_dispatcher,
        engram_mesh=meshes.engram,
        num_layers=args.num_layers,
        sparse_engram_gradients=args.optimizer == "sgd",
        materializer=FSDPMaterializer(meshes, fsdp),
        load_policy=loading,
        engram_policy=engram,
        lora_config=lora,
        cache_quantization=args.cache_quantization,
        adapter_path=args.adapter_in,
    )
    cleanup.callback(model.close)
    model.set_activation_checkpointing(policy)
    model.train()
    torch.cuda.synchronize(runtime.device)
    loaded_at = time.perf_counter()

    routed = model.model.layers[0].moe.routed
    local_expert_parameters = sum(
        layer.moe.routed.local_parameter_count
        for layer in model.model.layers
    )
    local_engram_parameters = sum(
        table.weight.numel() for table in model.model.engram_tables.values()
    )
    global_engram_parameters = sum(
        table.global_num_embeddings * model.config.engram_head_dim
        for table in model.model.engram_tables.values()
    )
    local_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    full_parameter_count = (
        local_parameter_count
        - local_expert_parameters
        - local_engram_parameters
        + local_expert_parameters * args.ep_size
        + global_engram_parameters
    )

    sharded_at = time.perf_counter()
    load_peak = torch.cuda.max_memory_allocated(runtime.device)
    torch.cuda.reset_peak_memory_stats(runtime.device)
    tracked = (next(parameter for name, parameter in model.named_parameters() if name.endswith("lora_b"))
               if lora is not None else model.model.layers[0].attention_hc.fn)
    optimizer = build_optimizer(model, OptimizerPolicy(args.optimizer, args.learning_rate, args.optimizer_foreach))
    gradients = GradientPolicy(model, meshes)

    generator = torch.Generator(device=runtime.device)
    generator.manual_seed(args.seed + data_parallel_rank(runtime.rank, args.cp_size))
    checkpointer = None
    if args.output_dir:
        checkpointer = CheckpointManager(
            args.output_dir, TrainingState(
                model, optimizer, model_config=model.config.to_dict(), data_generator=generator,
                training_config={"lora": asdict(lora) if lora else None,
                    "batch_size": args.batch_size, "sequence_length": args.seq_len,
                    "gradient_accumulation_steps": args.gradient_accumulation_steps},
                state_adapter=StateAdapter(model, cp=args.cp_size, ep=args.ep_size),
            ),
        )
    start_step = checkpointer.load() if args.resume else 0
    if start_step >= args.steps:
        raise ValueError("steps must exceed the resumed step")
    before = local_tensor(tracked).detach().float().clone()
    last_loss = last_aux_loss = 0.0
    torch.cuda.synchronize(runtime.device)
    train_started = time.perf_counter()
    for step in range(start_step + 1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        gradients.begin_step()
        step_loss = step_aux_loss = 0
        for _ in range(args.gradient_accumulation_steps):
            input_ids = torch.randint(3, model.config.vocab_size,
                (args.batch_size, args.seq_len), device=runtime.device, generator=generator)
            input_ids[:, 0] = model.config.bos_token_id
            output = model(input_ids, labels=input_ids)
            if output.loss is None or not torch.isfinite(output.loss):
                raise RuntimeError("training produced a non-finite loss")
            (output.loss / args.gradient_accumulation_steps).backward()
            step_loss = step_loss + output.loss.detach() / args.gradient_accumulation_steps
            step_aux_loss = step_aux_loss + output.aux_loss.detach() / args.gradient_accumulation_steps
        gradients.normalize()
        model.synchronize_engram()
        optimizer.step()
        distributed_trace(runtime, "optimizer_step_complete")
        last_loss = distributed_mean(step_loss, runtime)
        last_aux_loss = distributed_mean(step_aux_loss, runtime)
        distributed_trace(runtime, "metrics_reduced")
        if runtime.is_main:
            print(
                json.dumps(
                    {"step": step, "loss": last_loss, "aux_loss": last_aux_loss}
                ),
                flush=True,
            )

    distributed_trace(runtime, "parameter_delta_start")
    after = local_tensor(tracked).detach()
    parameter_delta = maximum_parameter_delta(before, after).to(runtime.device)
    distributed_trace(runtime, "parameter_delta_local_complete")
    dist.all_reduce(parameter_delta, op=dist.ReduceOp.MAX)
    distributed_trace(runtime, "parameter_delta_reduced")
    if parameter_delta.item() == 0:
        raise RuntimeError("optimizer step did not update the tracked parameter")

    torch.cuda.synchronize(runtime.device)
    distributed_trace(runtime, "cuda_synchronize_complete")
    finished = time.perf_counter()
    if checkpointer is not None:
        checkpointer.save(args.steps)
    if args.adapter_out:
        save_adapter(model, args.adapter_out)
    metrics: dict[str, object] = {
        "model": (
            f"DeepSeek-V4.1-Flash {args.num_layers}-layer prefix"
            if args.start_layer == 0
            else f"DeepSeek-V4.1-Flash layers {args.start_layer}-"
            f"{args.start_layer + args.num_layers - 1} window"
        ),
        "checkpoint": str(Path(args.model_path).resolve()),
        "parameters": full_parameter_count,
        "local_parameters_before_fsdp": local_parameter_count,
        "local_parameters": sum(local_tensor(parameter).numel() for parameter in model.parameters()),
        "world_size": runtime.world_size,
        "cp_size": args.cp_size,
        "ep_size": args.ep_size,
        "fsdp_size": meshes.fsdp.size(),
        "experts_per_rank": routed.num_experts,
        "engram_size": meshes.engram.size() if meshes.engram is not None else 1,
        "engram_rows_per_rank": sum(
            table.weight.shape[0] for table in model.model.engram_tables.values()
        ),
        "sparse_engram_gradients": args.optimizer == "sgd",
        "steps": args.steps,
        "batch_size_per_rank": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "sequence_length": args.seq_len,
        "start_layer": args.start_layer,
        "optimizer": args.optimizer,
        "checkpoint_mode": policy.mode,
        "lora": asdict(lora) if lora else None,
        "loss": last_loss,
        "aux_loss": last_aux_loss,
        "parameter_delta_max": parameter_delta.item(),
        "load_seconds": loaded_at - started,
        "shard_seconds": sharded_at - loaded_at,
        "train_seconds": finished - train_started,
        "tokens_per_second": ((args.steps-start_step) * args.gradient_accumulation_steps * args.batch_size
                              * args.seq_len * (runtime.world_size // args.cp_size) / (finished-train_started)),
        "peak_memory_gib": max(load_peak, torch.cuda.max_memory_allocated(runtime.device)) / 1024**3,
        "load_peak_gib": load_peak / 1024**3,
        "train_peak_gib": torch.cuda.max_memory_allocated(runtime.device) / 1024**3,
        "reserved_gib": torch.cuda.max_memory_reserved(runtime.device) / 1024**3,
        "cpu_max_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
        "parameter_bytes": parameter_memory(model, optimizer),
        "loading": model.loading_metrics,
        "engram_policy": asdict(engram),
        "fsdp_policy": asdict(fsdp),
        "engram_metrics": {key: table.metrics for key, table in model.model.engram_tables.items()},
    }
    if runtime.is_main:
        print(json.dumps({"summary": metrics}), flush=True)
        if args.metrics_file:
            path = Path(args.metrics_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def main() -> None:
    args = parse_args()
    runtime: Runtime | None = None
    try:
        runtime = initialize_runtime(require_distributed=True)
        train(args, runtime)
    finally:
        if runtime is not None and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
