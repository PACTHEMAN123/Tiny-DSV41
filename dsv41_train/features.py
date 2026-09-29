"""Shared CLI options for DSV4 post-training and profiling."""

import argparse

from .activation import ActivationCheckpointConfig
from .lora import LoRAConfig


def add_training_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint-mode", choices=("none", "full", "selective"), default="none")
    parser.add_argument("--checkpoint-components", default="attention,moe")
    parser.add_argument("--cache-quantization", choices=("none", "hf"), default="none")
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lora-targets", default=",".join(LoRAConfig().targets))
    parser.add_argument("--adapter-out")
    parser.add_argument("--adapter-in")


def checkpoint_policy(args: argparse.Namespace) -> ActivationCheckpointConfig:
    return ActivationCheckpointConfig(
        args.checkpoint_mode, tuple(args.checkpoint_components.split(",")),
    )


def adapter_config(args: argparse.Namespace) -> LoRAConfig | None:
    if not args.lora:
        if args.adapter_out or args.adapter_in:
            raise ValueError("adapter-in and adapter-out require --lora")
        return None
    return LoRAConfig(args.lora_rank, args.lora_alpha, args.lora_dropout,
                      tuple(args.lora_targets.split(",")))
