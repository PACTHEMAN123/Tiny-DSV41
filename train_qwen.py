"""Native CP, EP, and FSDP2 training entry point for Qwen3-30B-A3B."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.distributed as dist

from dsv41_train.models.qwen import load_qwen3_moe
from dsv41_train.models.qwen.parallel import ParallelParameters, apply_fsdp2, build_parallelism
from dsv41_train.runtime import Runtime, distributed_mean, initialize_runtime, local_tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Qwen3-30B-A3B with native PyTorch CP, EP, and FSDP2"
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cp-size", type=int, default=1)
    parser.add_argument("--ep-size", type=int, default=1)
    parser.add_argument("--metrics-file")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.steps < 1 or args.batch_size < 1:
        raise ValueError("steps and batch-size must be positive")
    if args.seq_len < 2:
        raise ValueError("seq-len must be at least 2")
    if args.learning_rate <= 0:
        raise ValueError("learning-rate must be positive")
    if args.cp_size < 1 or args.ep_size < 1:
        raise ValueError("cp-size and ep-size must be positive")
    if args.cp_size != args.ep_size:
        raise ValueError("the overlapping CP/EP topology requires cp-size == ep-size")
    if args.seq_len % args.cp_size:
        raise ValueError("seq-len must divide evenly across cp-size")
    if not Path(args.model_path).is_dir():
        raise FileNotFoundError(f"model path does not exist: {args.model_path}")


def train(args: argparse.Namespace, runtime: Runtime) -> dict[str, float | int | str]:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats(runtime.device)
    started = time.perf_counter()
    meshes, context_parallel, token_dispatcher = build_parallelism(
        args.cp_size, args.ep_size, runtime.device.type,
    )
    model = load_qwen3_moe(
        args.model_path,
        device=runtime.device,
        dtype=torch.bfloat16,
        context_parallel=context_parallel,
        token_dispatcher=token_dispatcher,
    )
    loaded_at = time.perf_counter()
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.train()
    full_parameter_count = ParallelParameters.collect(model).numel(args.ep_size)
    apply_fsdp2(model, meshes)
    sharded_at = time.perf_counter()
    parameters = ParallelParameters.collect(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        foreach=False,
    )
    generator = torch.Generator(device=runtime.device)
    generator.manual_seed(args.seed + meshes.fsdp.get_local_rank())
    tracked_parameter = model.model.layers[0].self_attn.q_proj.weight
    before = local_tensor(tracked_parameter).detach().float().clone()

    last_loss = last_aux_loss = last_grad_norm = 0.0
    for step in range(1, args.steps + 1):
        input_ids = torch.randint(
            0,
            model.config.vocab_size,
            (args.batch_size, args.seq_len),
            device=runtime.device,
            generator=generator,
        )
        input_ids[:, 0] = model.config.bos_token_id
        optimizer.zero_grad(set_to_none=True)
        output = model(
            input_ids,
            labels=input_ids,
            output_router_logits=True,
        )
        if output.loss is None or not torch.isfinite(output.loss):
            raise RuntimeError("training produced a non-finite loss")
        output.loss.backward()
        parameters.synchronize(context_parallel)
        grad_norm = parameters.clip_grad_norm(context_parallel)
        optimizer.step()

        last_loss = distributed_mean(output.loss, runtime)
        if isinstance(output.aux_loss, torch.Tensor):
            last_aux_loss = distributed_mean(output.aux_loss, runtime)
        last_grad_norm = grad_norm.item()
        if runtime.is_main:
            print(
                json.dumps(
                    {
                        "step": step,
                        "loss": last_loss,
                        "aux_loss": last_aux_loss,
                        "grad_norm": last_grad_norm,
                    }
                ),
                flush=True,
            )

    after = local_tensor(tracked_parameter).detach().float()
    parameter_delta = (after - before).abs().max()
    dist.all_reduce(parameter_delta, op=dist.ReduceOp.MAX)
    parameter_delta_value = parameter_delta.item()
    if parameter_delta_value == 0:
        raise RuntimeError("optimizer step did not update the tracked parameter")

    torch.cuda.synchronize(runtime.device)
    finished = time.perf_counter()
    metrics: dict[str, float | int | str] = {
        "model": "Qwen3-30B-A3B",
        "parameters": full_parameter_count,
        "world_size": runtime.world_size,
        "cp_size": args.cp_size,
        "ep_size": args.ep_size,
        "fsdp_size": meshes.fsdp.size(),
        "expert_fsdp_size": (
            meshes.expert_fsdp.size() if meshes.expert_fsdp is not None else meshes.fsdp.size()
        ),
        "steps": args.steps,
        "batch_size_per_data_rank": args.batch_size,
        "global_batch_size": args.batch_size * meshes.fsdp.size(),
        "sequence_length": args.seq_len,
        "loss": last_loss,
        "aux_loss": last_aux_loss,
        "grad_norm": last_grad_norm,
        "parameter_delta_max": parameter_delta_value,
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
    validate_args(args)
    runtime: Runtime | None = None
    try:
        runtime = initialize_runtime(require_distributed=True)
        train(args, runtime)
    finally:
        if runtime is not None and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
