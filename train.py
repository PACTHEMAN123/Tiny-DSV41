import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.distributed as dist

from dsv41_train import (
    CheckpointManager,
    DeepSeekV41Config,
    DeepSeekV41ForCausalLM,
    FSDP,
    ParallelMeshes,
    TrainingState,
)


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Runtime:
    device: torch.device
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    @property
    def distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tiny DeepSeek V4.1 causal LM")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--output-dir", default="outputs/final")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--local-rank",
        "--local_rank",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def initialize_runtime(device_name: str, local_rank_arg: int | None = None) -> Runtime:
    """Bind one torchrun process to one GPU and initialize NCCL."""

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 1:
        raise ValueError("WORLD_SIZE must be positive")
    if world_size == 1:
        return Runtime(device=choose_device(device_name))

    if device_name not in {"auto", "cuda"}:
        raise ValueError("distributed training assigns devices automatically; omit --device")
    if not dist.is_available():
        raise RuntimeError("this PyTorch build does not provide torch.distributed")
    if not torch.cuda.is_available():
        raise RuntimeError("multi-process training requires CUDA and the NCCL backend")
    local_rank_value = os.environ.get("LOCAL_RANK")
    if local_rank_value is None and local_rank_arg is None:
        raise RuntimeError("WORLD_SIZE is set but LOCAL_RANK is missing; launch with torchrun")

    local_rank = int(local_rank_value) if local_rank_value is not None else local_rank_arg
    assert local_rank is not None
    device_count = torch.cuda.device_count()
    if not 0 <= local_rank < device_count:
        raise RuntimeError(
            f"LOCAL_RANK {local_rank} is outside the {device_count} visible CUDA devices"
        )

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return Runtime(
        device=torch.device("cuda", local_rank),
        rank=dist.get_rank(),
        local_rank=local_rank,
        world_size=dist.get_world_size(),
    )


def make_batch(
    batch_size: int,
    seq_len: int,
    device: torch.device,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Generate short, learnable token sequences without a dataset dependency."""

    starts = torch.randint(0, 10, (batch_size, 1), device=device, generator=generator)
    offsets = torch.arange(seq_len - 1, device=device).unsqueeze(0)
    body = 3 + (starts + offsets) % 10
    bos = torch.ones((batch_size, 1), dtype=torch.long, device=device)
    return torch.cat((bos, body), dim=1)


def clip_grad_norm(parameters: Iterable[torch.Tensor], max_norm: float) -> torch.Tensor:
    """Clip local or FSDP2 DTensor gradients and return a replicated scalar."""

    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm, foreach=False)
    if hasattr(grad_norm, "full_tensor"):
        grad_norm = grad_norm.full_tensor()
    return grad_norm.detach().float()


def distributed_mean(value: torch.Tensor, runtime: Runtime) -> float:
    value = value.detach().float()
    if runtime.distributed:
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value /= runtime.world_size
    return value.item()


def validate_args(args: argparse.Namespace) -> None:
    if args.steps < 1 or args.batch_size < 1 or args.log_every < 1:
        raise ValueError("steps, batch-size, and log-every must be positive")
    if not 2 <= args.seq_len <= 128:
        raise ValueError("seq-len must be between 2 and 128")


def train(args: argparse.Namespace, runtime: Runtime) -> None:
    # All ranks initialize identical parameters; the per-rank generator below varies the data.
    torch.manual_seed(args.seed)
    config = DeepSeekV41Config.tiny()
    model = DeepSeekV41ForCausalLM(config)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    model = model.to(runtime.device)

    if runtime.distributed:
        meshes = ParallelMeshes.build(device_type=runtime.device.type)
        model = FSDP(meshes).apply(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    batch_generator = torch.Generator(device=runtime.device)
    batch_generator.manual_seed(args.seed + runtime.rank)
    checkpointer = CheckpointManager(
        ROOT / args.output_dir / "checkpoint",
        TrainingState(
            model,
            optimizer,
            model_config=config.to_dict(),
            data_generator=batch_generator,
        ),
    )

    if runtime.is_main:
        print(
            json.dumps(
                {
                    "device": str(runtime.device),
                    "world_size": runtime.world_size,
                    "parallelism": "fsdp" if runtime.distributed else "none",
                    "parameters": parameter_count,
                    "global_batch_size": args.batch_size * runtime.world_size,
                }
            ),
            flush=True,
        )

    for step in range(1, args.steps + 1):
        input_ids = make_batch(
            args.batch_size,
            args.seq_len,
            runtime.device,
            generator=batch_generator,
        )
        attention_mask = torch.ones_like(input_ids)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=runtime.device.type,
            dtype=torch.bfloat16,
            enabled=runtime.device.type == "cuda",
        ):
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
            loss = output.loss

        assert loss is not None
        loss.backward()
        grad_norm = clip_grad_norm(model.parameters(), 1.0)
        optimizer.step()

        should_log = step % args.log_every == 0 or step == args.steps
        if should_log:
            mean_loss = distributed_mean(loss, runtime)
            if runtime.is_main:
                print(
                    json.dumps(
                        {
                            "step": step,
                            "loss": mean_loss,
                            "grad_norm": grad_norm.item(),
                        }
                    ),
                    flush=True,
                )

    checkpoint_path = checkpointer.save(args.steps)
    if runtime.is_main:
        print(f"saved checkpoint to {checkpoint_path}", flush=True)


def main() -> None:
    args = parse_args()
    validate_args(args)
    runtime: Runtime | None = None
    try:
        runtime = initialize_runtime(args.device, args.local_rank)
        train(args, runtime)
    finally:
        if runtime is not None and runtime.distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
