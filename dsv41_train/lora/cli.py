"""Optional argparse integration, independent of other training features."""

import argparse

from .config import LoRAConfig


def add_lora_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("LoRA")
    defaults = LoRAConfig()
    group.add_argument("--lora", action="store_true")
    group.add_argument("--lora-rank", type=int, default=defaults.rank)
    group.add_argument("--lora-alpha", type=float, default=defaults.alpha)
    group.add_argument("--lora-dropout", type=float, default=defaults.dropout)
    group.add_argument("--lora-targets", default=",".join(defaults.targets))
    group.add_argument("--adapter-out")
    group.add_argument("--adapter-in")


def adapter_config(args: argparse.Namespace) -> LoRAConfig | None:
    if not args.lora:
        if args.adapter_out or args.adapter_in:
            raise ValueError("adapter-in and adapter-out require --lora")
        return None
    if args.adapter_in and getattr(args, "resume", False):
        raise ValueError("adapter-in and resume are mutually exclusive")
    return LoRAConfig(
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        targets=tuple(target.strip() for target in args.lora_targets.split(",")),
    )
