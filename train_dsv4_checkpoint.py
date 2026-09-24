"""Train a real DeepSeek V4.1 prefix from the released checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

from dsv41_train.models.dsv4 import load_dsv41_backbone_window
from dsv41_train.models.dsv4.parallel import (
    ParallelParameters,
    apply_fsdp2_layer,
    apply_fsdp2_root,
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


def data_parallel_rank(global_rank: int, cp_size: int) -> int:
    if global_rank < 0 or cp_size < 1:
        raise ValueError("global-rank must be non-negative and cp-size must be positive")
    return global_rank // cp_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--start-layer", type=int, default=0)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--optimizer", choices=("sgd", "adamw"), default="sgd")
    parser.add_argument("--cp-size", type=int, default=1)
    parser.add_argument("--ep-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metrics-file")
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
    if args.cp_size < 1 or args.ep_size < 1:
        raise ValueError("cp-size and ep-size must be positive")
    if world_size % args.cp_size or world_size % args.ep_size:
        raise ValueError("cp-size and ep-size must both divide world size")
    if args.cp_size > 1 and args.cp_size != args.ep_size:
        raise ValueError("DSV4 context parallelism requires cp-size == ep-size")
    if args.seq_len % args.cp_size:
        raise ValueError("seq-len must divide evenly across cp-size")
    if 384 % args.ep_size:
        raise ValueError("the checkpoint's 384 experts must divide evenly across ep-size")


def train(args: argparse.Namespace, runtime: Runtime) -> dict[str, float | int | str]:
    validate_args(args, runtime.world_size)
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
        layer_loaded=lambda layer: apply_fsdp2_layer(layer, meshes),
    )
    model.train()
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

    apply_fsdp2_root(model, meshes)
    sharded_at = time.perf_counter()
    parameters = ParallelParameters.collect(model)
    tracked = model.model.layers[0].attention_hc.fn
    before = local_tensor(tracked).detach().float().clone()
    if args.optimizer == "adamw":
        optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, foreach=False
        )
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, foreach=False)

    generator = torch.Generator(device=runtime.device)
    generator.manual_seed(args.seed + data_parallel_rank(runtime.rank, args.cp_size))
    last_loss = last_aux_loss = 0.0
    for step in range(1, args.steps + 1):
        input_ids = torch.randint(
            3,
            model.config.vocab_size,
            (args.batch_size, args.seq_len),
            device=runtime.device,
            generator=generator,
        )
        input_ids[:, 0] = model.config.bos_token_id
        optimizer.zero_grad(set_to_none=True)
        output = model(input_ids, labels=input_ids)
        if output.loss is None or not torch.isfinite(output.loss):
            raise RuntimeError("training produced a non-finite loss")
        output.loss.backward()
        parameters.synchronize(context_parallel, dense_fully_sharded=True)
        distributed_trace(runtime, "context_parallel_gradients_synchronized")
        optimizer.step()
        distributed_trace(runtime, "optimizer_step_complete")
        last_loss = distributed_mean(output.loss, runtime)
        last_aux_loss = distributed_mean(output.aux_loss, runtime)
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
    parameter_delta = maximum_parameter_delta(before, after)
    distributed_trace(runtime, "parameter_delta_local_complete")
    dist.all_reduce(parameter_delta, op=dist.ReduceOp.MAX)
    distributed_trace(runtime, "parameter_delta_reduced")
    if parameter_delta.item() == 0:
        raise RuntimeError("optimizer step did not update the tracked parameter")

    torch.cuda.synchronize(runtime.device)
    distributed_trace(runtime, "cuda_synchronize_complete")
    finished = time.perf_counter()
    metrics: dict[str, float | int | str] = {
        "model": (
            f"DeepSeek-V4.1-Flash {args.num_layers}-layer prefix"
            if args.start_layer == 0
            else f"DeepSeek-V4.1-Flash layers {args.start_layer}-"
            f"{args.start_layer + args.num_layers - 1} window"
        ),
        "checkpoint": str(Path(args.model_path).resolve()),
        "parameters": full_parameter_count,
        "local_parameters_before_fsdp": local_parameter_count,
        "world_size": runtime.world_size,
        "cp_size": args.cp_size,
        "ep_size": args.ep_size,
        "fsdp_size": meshes.fsdp.size(),
        "expert_fsdp_size": (
            meshes.expert_fsdp.size() if meshes.expert_fsdp is not None else 1
        ),
        "data_parallel_size": runtime.world_size // args.cp_size,
        "experts_per_rank": routed.num_experts,
        "engram_size": meshes.engram.size() if meshes.engram is not None else 1,
        "engram_rows_per_rank": sum(
            table.weight.shape[0] for table in model.model.engram_tables.values()
        ),
        "sparse_engram_gradients": args.optimizer == "sgd",
        "steps": args.steps,
        "batch_size_per_rank": args.batch_size,
        "global_batch_size": args.batch_size * (runtime.world_size // args.cp_size),
        "sequence_length": args.seq_len,
        "start_layer": args.start_layer,
        "optimizer": args.optimizer,
        "loss": last_loss,
        "aux_loss": last_aux_loss,
        "parameter_delta_max": parameter_delta.item(),
        "load_seconds": loaded_at - started,
        "shard_seconds": sharded_at - loaded_at,
        "train_seconds": finished - sharded_at,
        "peak_memory_gib": torch.cuda.max_memory_allocated(runtime.device) / 1024**3,
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
