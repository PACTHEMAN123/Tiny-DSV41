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


def install_backward_traces(
    model: torch.nn.Module,
    runtime: Runtime,
) -> list[torch.utils.hooks.RemovableHandle]:
    """Trace nested backward progress without affecting normal training runs."""

    if not os.environ.get("DSV41_LAYER_TRACE"):
        return []

    handles: list[torch.utils.hooks.RemovableHandle] = []

    def register(module: torch.nn.Module, name: str) -> None:
        def start(_module: torch.nn.Module, _grad_output: tuple[torch.Tensor, ...]) -> None:
            distributed_trace(runtime, f"{name}_backward_start")

        def complete(
            _module: torch.nn.Module,
            _grad_input: tuple[torch.Tensor | None, ...],
            _grad_output: tuple[torch.Tensor | None, ...],
        ) -> None:
            distributed_trace(runtime, f"{name}_backward_complete")

        handles.append(module.register_full_backward_pre_hook(start))
        handles.append(module.register_full_backward_hook(complete))

    decoder = model.model
    for layer in decoder.layers:
        prefix = f"layer_{layer.layer_id}"
        register(layer, prefix)
        register(layer.moe.routed, f"{prefix}_routed_experts")
        register(layer.attention, f"{prefix}_attention")
    return handles


def maximum_parameter_delta(before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
    if before.shape != after.shape:
        raise ValueError("parameter snapshots must have matching shapes")
    if after.numel() == 0:
        return torch.zeros((), dtype=torch.float32, device=after.device)
    return (after.float() - before.float()).abs().max()


def sgd_step_in_backward(
    parameter: torch.nn.Parameter,
    *,
    learning_rate: float,
    gradient_scale: float = 1.0,
) -> None:
    gradient = parameter.grad
    if gradient is None:
        return
    with torch.no_grad():
        local_tensor(parameter).add_(
            local_tensor(gradient),
            alpha=-learning_rate * gradient_scale,
        )
    parameter.grad = None


def register_sgd_in_backward(
    parameters: ParallelParameters,
    *,
    learning_rate: float,
    cp_size: int,
) -> tuple[set[int], list[torch.utils.hooks.RemovableHandle]]:
    expert_ids = {id(parameter) for parameter in parameters.experts}
    fsdp_parameters = [
        parameter
        for parameter in (*parameters.dense, *parameters.experts, *parameters.replicated)
        if hasattr(parameter, "to_local")
    ]
    handles = []
    for parameter in fsdp_parameters:
        gradient_scale = 1.0 / cp_size if id(parameter) in expert_ids else 1.0
        handles.append(
            parameter.register_post_accumulate_grad_hook(
                lambda value, scale=gradient_scale: sgd_step_in_backward(
                    value,
                    learning_rate=learning_rate,
                    gradient_scale=scale,
                )
            )
        )
    return {id(parameter) for parameter in fsdp_parameters}, handles


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
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--offload-engram", action="store_true")
    parser.add_argument("--fsdp-cpu-offload-layers", type=int, default=0)
    parser.add_argument("--optimizer-in-backward", action="store_true")
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
    if args.offload_engram and args.optimizer != "sgd":
        raise ValueError("Engram CPU offload requires the sparse SGD training path")
    if not 0 <= args.fsdp_cpu_offload_layers <= args.num_layers:
        raise ValueError("fsdp-cpu-offload-layers must be between zero and num-layers")
    if args.optimizer_in_backward and args.optimizer != "sgd":
        raise ValueError("optimizer-in-backward requires SGD")
    if args.optimizer_in_backward and args.fsdp_cpu_offload_layers:
        raise ValueError("optimizer-in-backward cannot be combined with FSDP CPU offload")


def train(args: argparse.Namespace, runtime: Runtime) -> dict[str, float | int | str]:
    validate_args(args, runtime.world_size)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats(runtime.device)
    meshes, context_parallel, token_dispatcher = build_parallelism(
        args.cp_size, args.ep_size, runtime.device.type
    )

    started = time.perf_counter()
    distributed_trace(runtime, "checkpoint_load_start")
    offload_layer_start = args.start_layer + args.num_layers - args.fsdp_cpu_offload_layers
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
        offload_engram=args.offload_engram,
        layer_loaded=lambda layer: apply_fsdp2_layer(
            layer,
            meshes,
            cpu_offload=layer.layer_id >= offload_layer_start,
        ),
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.train()
    loaded_at = time.perf_counter()
    distributed_trace(runtime, "checkpoint_load_complete")

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
    distributed_trace(runtime, "root_shard_complete")
    backward_trace_handles = install_backward_traces(model, runtime)
    parameters = ParallelParameters.collect(model)
    tracked = model.model.layers[0].attention_hc.fn
    before = local_tensor(tracked).detach().float().clone()
    optimizer_hook_handles: list[torch.utils.hooks.RemovableHandle] = []
    in_backward_parameter_ids: set[int] = set()
    if args.optimizer_in_backward:
        in_backward_parameter_ids, optimizer_hook_handles = register_sgd_in_backward(
            parameters,
            learning_rate=args.learning_rate,
            cp_size=args.cp_size,
        )
    optimizer_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in in_backward_parameter_ids
    ]
    if args.optimizer == "adamw":
        optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            optimizer_parameters, lr=args.learning_rate, foreach=False
        )
    else:
        optimizer = torch.optim.SGD(
            optimizer_parameters,
            lr=args.learning_rate,
            foreach=False,
        )

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
        model.zero_grad(set_to_none=True)
        distributed_trace(runtime, "forward_start")
        output = model(input_ids, labels=input_ids)
        distributed_trace(runtime, "forward_complete")
        if output.loss is None or not torch.isfinite(output.loss):
            raise RuntimeError("training produced a non-finite loss")
        distributed_trace(runtime, "backward_start")
        output.loss.backward()
        distributed_trace(runtime, "backward_complete")
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
        "gradient_checkpointing": args.gradient_checkpointing,
        "engram_cpu_offload": args.offload_engram,
        "fsdp_cpu_offload_layers": args.fsdp_cpu_offload_layers,
        "optimizer_in_backward": args.optimizer_in_backward,
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
    for handle in backward_trace_handles:
        handle.remove()
    for handle in optimizer_hook_handles:
        handle.remove()
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
