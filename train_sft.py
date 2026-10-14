"""Reproducible, token-weighted SFT from local JSONL and pretrained checkpoints."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

import torch
import torch.distributed as dist

from dsv41_train.checkpoint import CheckpointManager, TrainingState
from dsv41_train.features import add_training_options, adapter_config, checkpoint_policy
from dsv41_train.layout import local_tensor, parameter_memory
from dsv41_train.lora import LoRAConfig, LoRALinear, save_adapter
from dsv41_train.memory_options import add_memory_options, memory_policies
from dsv41_train.models.dsv4 import load_dsv41_backbone_window
from dsv41_train.models.dsv4.config import DeepSeekV41Config
from dsv41_train.models.dsv4.engram import EngramPolicy
from dsv41_train.models.dsv4.materialize import FSDPMaterializer
from dsv41_train.models.dsv4.parallel import build_parallelism
from dsv41_train.models.dsv4.policies import GradientPolicy
from dsv41_train.models.qwen import load_qwen3_moe
from dsv41_train.models.qwen.config import Qwen3MoeConfig
from dsv41_train.models.qwen.checkpoint import QWEN_LORA_TARGETS
from dsv41_train.models.qwen.engram import (
    QwenEngramConfig, attach_engram, shard_qwen_component, shard_qwen_root,
)
from dsv41_train.optimization import OptimizerPolicy, build_optimizer
from dsv41_train.runtime import Runtime, initialize_runtime
from dsv41_train.sft_data import SFTDataset
from dsv41_train.sharded_state import StateAdapter


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-type", choices=("qwen", "dsv4"), required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data", required=True, help="Local JSONL; see docs/sft.md for formats")
    parser.add_argument("--tokenizer-path", help="Defaults to MODEL_PATH/tokenizer.json for text rows")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=3, help="Total optimizer steps, including resumed steps")
    parser.add_argument("--resume", action="store_true", help="Resume latest complete checkpoint in output-dir")
    parser.add_argument("--save-every", type=int, default=0, help="0 saves only the final checkpoint")
    parser.add_argument("--checkpoint-state", choices=("full", "lora"), default="full",
                        help="lora saves adapters/optimizer/RNG/buffers and reloads frozen base from model-path")
    parser.add_argument("--batch-size", type=int, default=1, help="Examples per rank per microbatch")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--optimizer", choices=("sgd", "adamw"), default="sgd")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--router-aux-loss-coef", type=float, default=0.0,
                        help="Optional token-weighted microbatch router loss; 0 is pure SFT")
    parser.add_argument("--cp-size", type=int, default=1)
    parser.add_argument("--ep-size", type=int, default=1)
    parser.add_argument("--start-layer", type=int, default=0, help="DSV4 only; 0 selects a prefix")
    parser.add_argument("--num-layers", type=int, help="DSV4 only; defaults to one-layer prefix")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify-lora", action="store_true",
                        help="Verify frozen base, adapter gradients/updates and record fixed eval probes")
    parser.add_argument("--lora-probe-tokens", type=int, default=4096,
                        help="Bound fixed teacher-forced probes; training still uses full seq-len")
    parser.add_argument("--min-active-tokens-per-example", type=int, default=0,
                        help="Reject examples below this effective length; padding does not count")
    parser.add_argument("--identity-eval-prompts", help="Prepared identity heldout.json for base-before/adapter-after generation")
    parser.add_argument("--identity-eval-max-new-tokens", type=int, default=16)
    parser.add_argument("--identity-eval-seed", type=int, default=20261014)
    parser.add_argument("--qwen-engram", action="store_true",
                        help="Add newly initialized trainable Engram tables to Qwen; defaults off")
    parser.add_argument("--qwen-engram-layers", type=int, nargs="+",
                        help="Qwen layers to augment; defaults to 1 24")
    parser.add_argument("--qwen-engram-vocab-per-head", type=int)
    parser.add_argument("--qwen-engram-head-dim", type=int)
    parser.add_argument("--qwen-engram-heads", type=int)
    parser.add_argument("--qwen-engram-max-ngram", type=int)
    parser.add_argument("--qwen-engram-seed", type=int,
                        help="Engram initialization seed; defaults to 20261013")
    add_training_options(parser)
    add_memory_options(parser)
    parser.set_defaults(lora_targets=None)
    args = parser.parse_args(argv)
    if args.lora_targets is None:
        targets = QWEN_LORA_TARGETS if args.model_type == "qwen" else LoRAConfig().targets
        args.lora_targets = ",".join(targets)
    return args


def qwen_engram_options(args, model_config=None):
    """Keep Qwen augmentation identity separate from pretrained model config."""
    keys = {"layer_ids": args.qwen_engram_layers,
            "vocab_per_head": args.qwen_engram_vocab_per_head,
            "head_dim": args.qwen_engram_head_dim, "heads": args.qwen_engram_heads,
            "max_ngram": args.qwen_engram_max_ngram, "seed": args.qwen_engram_seed}
    if args.model_type != "qwen":
        if args.qwen_engram or any(value is not None for value in keys.values()):
            raise ValueError("qwen-engram options require --model-type qwen")
        return None, None
    if not args.qwen_engram:
        if any(value is not None for value in keys.values()):
            raise ValueError("qwen-engram configuration requires --qwen-engram")
        if (args.engram_device != "cuda" or args.engram_prefetch != "sync"
                or args.engram_deduplicate or args.engram_cache_mib):
            raise ValueError("Qwen Engram memory options require --qwen-engram")
        return None, None
    values = {key: value for key, value in keys.items() if value is not None}
    if "layer_ids" in values:
        values["layer_ids"] = tuple(values["layer_ids"])
    config = QwenEngramConfig(**values)
    if config.seed < 0:
        raise ValueError("qwen-engram-seed must be nonnegative")
    if model_config is not None and max(config.layer_ids) >= model_config.num_hidden_layers:
        raise ValueError("Qwen Engram layer lies outside the checkpoint backbone")
    policy = EngramPolicy(args.engram_device, args.engram_prefetch, args.engram_deduplicate,
                          args.staging_mib * 1024**2, args.engram_cache_mib * 1024**2)
    if args.optimizer != "sgd":
        raise ValueError("Qwen Engram uses sparse table gradients and requires --optimizer sgd")
    if args.weight_decay:
        raise ValueError("Qwen Engram sparse SGD requires --weight-decay 0")
    return config, policy


def validate_args(args, world_size):
    if args.cp_size != 1:
        raise ValueError("SFT currently requires --cp-size 1; masked CP loss is not yet validated")
    if args.ep_size < 1 or world_size % args.ep_size:
        raise ValueError("ep-size must be positive and divide WORLD_SIZE")
    if min(args.steps, args.batch_size, args.gradient_accumulation_steps) < 1 or args.seq_len < 2:
        raise ValueError("steps/batch/accumulation must be positive and seq-len >= 2")
    if args.save_every < 0 or args.start_layer < 0 or args.seed < 0:
        raise ValueError("save-every, start-layer and seed must be nonnegative")
    for key in ("learning_rate", "weight_decay", "router_aux_loss_coef"):
        value = getattr(args, key)
        if not math.isfinite(value) or value < 0 or (key == "learning_rate" and value == 0):
            raise ValueError(f"invalid {key.replace('_', '-')}")
    if args.resume and args.adapter_in:
        raise ValueError("resume and adapter-in are mutually exclusive")
    if args.verify_lora and not args.lora:
        raise ValueError("verify-lora requires --lora")
    if args.checkpoint_state == "lora" and not args.lora:
        raise ValueError("checkpoint-state lora requires --lora")
    if not 2 <= args.lora_probe_tokens <= 4096:
        raise ValueError("lora-probe-tokens must be in 2..4096")
    if not 0 <= args.min_active_tokens_per_example <= args.seq_len:
        raise ValueError("min-active-tokens-per-example must be in 0..seq-len")
    if not 1 <= args.identity_eval_max_new_tokens <= 64 or args.identity_eval_seed < 0:
        raise ValueError("invalid identity generation length/seed")
    if args.identity_eval_prompts:
        if not args.lora or not args.verify_lora or not args.adapter_out:
            raise ValueError("identity generation requires --lora --verify-lora --adapter-out")
        if args.steps < 20:
            raise ValueError("identity stability runs require at least 20 total optimizer steps")
        if not Path(args.identity_eval_prompts).is_file():
            raise FileNotFoundError(args.identity_eval_prompts)
    folder = Path(args.model_path)
    for filename in ("config.json", "model.safetensors.index.json"):
        if not (folder / filename).is_file():
            raise FileNotFoundError(folder / filename)
    if not Path(args.data).is_file():
        raise FileNotFoundError(args.data)
    if args.model_type == "qwen":
        if args.num_layers is not None or args.start_layer:
            raise ValueError("Qwen uses its full backbone; layer selection is DSV4-only")
        if args.lora and args.qwen_engram:
            raise ValueError("Qwen LoRA with added Engram is not supported")
        if args.lora and set(adapter_config(args).targets) - set(QWEN_LORA_TARGETS):
            raise ValueError("Qwen LoRA currently supports self_attn q_proj/k_proj/v_proj/o_proj targets only")
        if args.checkpoint_mode == "selective":
            raise ValueError("Qwen supports checkpoint-mode none/full")
        if (args.load_mode != "layerwise" or args.load_pipeline or args.fsdp_offload_layers
                or args.cache_quantization != "none"):
            raise ValueError("Qwen uses layerwise GPU backbone loading; DSV4 loading/offload options are unsupported")
        config = Qwen3MoeConfig.from_json(folder / "config.json")
        expert_count = config.num_experts
    else:
        args.num_layers = 1 if args.num_layers is None else args.num_layers
        config = DeepSeekV41Config.from_json(folder / "config.json")
        if args.num_layers < 1 or args.start_layer + args.num_layers > config.num_hidden_layers:
            raise ValueError("DSV4 layer selection is outside the checkpoint backbone")
        if args.start_layer and args.start_layer not in config.kv_source_layer_ids:
            raise ValueError("a DSV4 window must start at a KV source layer")
        expert_count = config.n_routed_experts
        memory_policies(args)
    qwen_engram_options(args, config)
    if expert_count % args.ep_size:
        raise ValueError("ep-size must divide the checkpoint's expert count")
    if args.seq_len > config.max_position_embeddings:
        raise ValueError("seq-len exceeds the checkpoint's max_position_embeddings")
    adapter_config(args)
    checkpoint_policy(args)
    return config


def _record(runtime, **record):
    if runtime.is_main:
        print(json.dumps(record, ensure_ascii=False), flush=True)


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_identity(folder):
    folder = Path(folder).resolve()
    return {"path": str(folder), "config_sha256": _file_hash(folder / "config.json"),
            "index_sha256": _file_hash(folder / "model.safetensors.index.json")}


def loss_scale(local_supervised_tokens, global_supervised_tokens, world_size):
    """FSDP averages gradients, so each local token sum needs WORLD_SIZE / N."""
    if global_supervised_tokens <= 0 or local_supervised_tokens <= 0:
        raise ValueError("each SFT microbatch must contain shifted supervised targets")
    return world_size * local_supervised_tokens / global_supervised_tokens


def _bounded_lora_probe(batch, token_limit):
    """Keep the final supervised target and preceding context in a small probe.

    This only shortens the fixed verification probe, never training batches.
    Labels still align with token positions and the first cropped token is ignored.
    """
    if batch["input_ids"].shape[0] != 1 or not 2 <= token_limit <= 4096:
        raise ValueError("fixed probe requires one example and 2..4096 token limit")
    labels, mask = batch["labels"], batch["attention_mask"]
    targets = (labels[0].ne(-100) & mask[0]).nonzero().flatten()
    targets = targets[targets > 0]
    if not len(targets):
        raise ValueError("fixed probe has no shifted supervised target")
    length = batch["input_ids"].shape[1]
    end = length if length <= token_limit else int(targets[-1]) + 1
    start = max(0, end - token_limit)
    result = {key: batch[key][:, start:end].clone() for key in ("input_ids", "labels", "attention_mask")}
    result["labels"][:, 0] = -100
    return result


def _cuda_memory_snapshot(runtime):
    values = torch.tensor([torch.cuda.memory_allocated(runtime.device),
                           torch.cuda.max_memory_allocated(runtime.device),
                           torch.cuda.max_memory_reserved(runtime.device)],
                          device=runtime.device, dtype=torch.int64)
    dist.all_reduce(values, op=dist.ReduceOp.MAX)
    return {key: value / 2**30 for key, value in zip(
        ("allocated_gib", "peak_allocated_gib", "peak_reserved_gib"), values.tolist())}


def _collective_finite(value, runtime, label):
    okay = torch.isfinite(value.detach()).all().to(device=runtime.device, dtype=torch.int32)
    dist.all_reduce(okay, op=dist.ReduceOp.MIN)
    if not okay.item():
        raise RuntimeError(f"non-finite {label} on at least one rank")


def _finite_gradients(model, runtime):
    # Keep checks on the parameter's device, including CPU-offloaded Engram.
    flags = {}
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        gradient = local_tensor(parameter.grad)
        values = gradient._values() if gradient.is_sparse else gradient
        flag = torch.isfinite(values).all()
        flags[values.device] = flags.get(values.device, flag.new_tensor(True)) & flag
    okay = torch.ones((), dtype=torch.int32, device=runtime.device)
    for flag in flags.values():
        okay.mul_(flag.to(runtime.device, dtype=torch.int32))
    dist.all_reduce(okay, op=dist.ReduceOp.MIN)
    if not okay.item():
        raise RuntimeError("non-finite gradient on at least one rank; optimizer step cancelled")


def _probe_indices(numel, *, device=None):
    """Choose at most 256 exact in-bounds offsets, without floating-point rounding."""
    if numel < 0:
        raise ValueError("parameter size must be nonnegative")
    size = min(numel, 256)
    positions = torch.arange(size, device=device, dtype=torch.int64)
    if size < 2:
        return positions
    # Splitting the quotient and remainder also avoids overflow in i * (N - 1).
    quotient, remainder = divmod(numel - 1, size - 1)
    return positions * quotient + (positions * remainder) // (size - 1)


def _update_probes(model):
    """Bounded samples from every trainable dense tensor; no model-sized clone."""
    probes = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or "engram_tables." in name or ".moe.routed." in name or ".mlp.experts." in name:
            continue
        tensor = local_tensor(parameter).detach().reshape(-1)
        size = min(tensor.numel(), 256)
        if size:
            indices = _probe_indices(tensor.numel(), device=tensor.device)
            probes.append((name, parameter, indices, tensor[indices].float().clone()))
    if not probes:
        raise RuntimeError("model has no trainable dense parameters for update verification")
    return probes


def _parameter_delta(probes, runtime):
    maximum = torch.zeros((), dtype=torch.float32, device=runtime.device)
    for _, parameter, indices, before in probes:
        after = local_tensor(parameter).detach().reshape(-1)[indices].float()
        maximum = torch.maximum(maximum, (after - before).abs().max().to(runtime.device))
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    return maximum.item()


def _sample_parameter(parameter, indices):
    tensor = local_tensor(parameter).detach()
    if tensor.is_contiguous():
        return tensor.view(-1)[indices].float()
    # Avoid a parameter-sized contiguous copy for a transposed base weight.
    return tensor[torch.unravel_index(indices, tensor.shape)].float()


class LoRAVerifier:
    """Inspect every freeze/gradient flag, with at most 256 values per tensor.

    Snapshots are taken after checkpoint restoration. A/B deltas therefore
    measure this invocation; no full base parameter or model clone is needed.
    """

    def __init__(self, model, optimizer, runtime):
        self.runtime = runtime
        self.optimizer = optimizer
        kinds = {id(getattr(module, f"lora_{kind}")): kind
                 for module in model.modules() if isinstance(module, LoRALinear)
                 for kind in ("a", "b")}
        self.adapters = [(name, parameter, kinds[id(parameter)])
                         for name, parameter in model.named_parameters() if id(parameter) in kinds]
        self.base = [(name, parameter) for name, parameter in model.named_parameters()
                     if id(parameter) not in kinds]
        self._check_structure()
        self.base_probes = self._probes(self.base)
        self.adapter_probes = {kind: self._probes([(name, parameter) for name, parameter, group
                                                 in self.adapters if group == kind])
                               for kind in ("a", "b")}
        self.previous = {name: before.clone() for probes in self.adapter_probes.values()
                         for name, _, _, before in probes}
        self.maxima = {kind: 0.0 for kind in ("a", "b")}
        counts = torch.tensor([
            sum(local_tensor(parameter).numel() for _, parameter, _ in self.adapters),
            sum(local_tensor(parameter).numel() * parameter.element_size()
                for _, parameter, _ in self.adapters),
            sum(indices.numel() for _, _, indices, _ in self.base_probes),
        ], device=runtime.device, dtype=torch.int64)
        local_counts = counts.tolist()
        self._reduce(counts, dist.ReduceOp.SUM)
        self.metadata = {"adapter_parameters_rank0": local_counts[0],
                         "adapter_bytes_rank0": local_counts[1],
                         "adapter_parameters_global": counts[0].item(),
                         "adapter_bytes_global": counts[1].item(),
                         "adapter_parameter_names": [name for name, _, _ in self.adapters],
                         "optimizer_contains_exactly_adapters": True,
                         "frozen_base_tensors_per_rank": len(self.base),
                         "base_sampled_values_global": counts[2].item(),
                         "counting": "physical parameter shards summed across ranks",
                         "base_verification": "all requires_grad=False and grad=None; <=256 values per tensor unchanged"}

    def _reduce(self, value, operation):
        if dist.is_initialized():
            dist.all_reduce(value, op=operation)

    def _check_structure(self):
        optimized = [id(parameter) for group in self.optimizer.param_groups for parameter in group["params"]]
        expected = {id(parameter) for _, parameter, _ in self.adapters}
        flags = torch.tensor([
            bool(self.adapters),
            all(not parameter.requires_grad for _, parameter in self.base),
            all(parameter.grad is None for _, parameter in self.base),
            all(parameter.requires_grad for _, parameter, _ in self.adapters),
            len(optimized) == len(expected) and set(optimized) == expected,
        ], device=self.runtime.device, dtype=torch.int32)
        self._reduce(flags, dist.ReduceOp.MIN)
        for okay, message in zip(flags.tolist(), ("no LoRA adapters found", "base is not frozen",
                                                   "base has gradients", "adapter is frozen",
                                                   "optimizer must contain each adapter exactly once and no base")):
            if not okay:
                raise RuntimeError(f"LoRA verification failed: {message} on at least one rank")

    @staticmethod
    def _probes(parameters):
        probes = []
        for name, parameter in parameters:
            tensor = local_tensor(parameter)
            if tensor.numel():
                indices = _probe_indices(tensor.numel(), device=tensor.device)
                probes.append((name, parameter, indices, _sample_parameter(parameter, indices).clone()))
        return probes

    def gradients(self):
        """Zero A gradients are valid when B is zero, particularly at step one."""
        self._check_structure()
        sums = torch.zeros((2, 3), device=self.runtime.device, dtype=torch.float64)
        okay = torch.ones(2, device=self.runtime.device, dtype=torch.int32)
        for _, parameter, kind in self.adapters:
            if not local_tensor(parameter).numel():
                continue
            if parameter.grad is None:
                okay[0] = 0
                continue
            gradient = local_tensor(parameter.grad)
            gradient = gradient._values() if gradient.is_sparse else gradient
            okay[1].mul_(torch.isfinite(gradient).all().to(okay))
            values = gradient.detach().reshape(-1)
            row = 0 if kind == "a" else 1
            for chunk in values.split(1024**2):
                chunk = chunk.double()
                sums[row, 0] += chunk.square().sum().to(sums.device)
                sums[row, 1] += chunk.count_nonzero().to(sums.device)
                sums[row, 2] += chunk.numel()
        self._reduce(okay, dist.ReduceOp.MIN)
        self._reduce(sums, dist.ReduceOp.SUM)
        if not okay[0].item():
            raise RuntimeError("LoRA verification failed: adapter gradient is missing")
        if not okay[1].item():
            raise RuntimeError("LoRA verification failed: non-finite adapter gradient")
        return {kind: {"gradient_finite": True, "gradient_l2": math.sqrt(sums[row, 0].item()),
                       "gradient_nonzero": int(sums[row, 1].item()),
                       "gradient_elements": int(sums[row, 2].item())}
                for row, kind in enumerate(("a", "b"))}

    def after_step(self, gradient_metrics):
        self._check_structure()
        # A failed local check is reduced before raising so peers cannot continue.
        unchanged = torch.ones((), device=self.runtime.device, dtype=torch.int32)
        for _, parameter, indices, before in self.base_probes:
            unchanged.mul_(torch.eq(_sample_parameter(parameter, indices), before).all().to(unchanged))
        self._reduce(unchanged, dist.ReduceOp.MIN)
        if not unchanged.item():
            raise RuntimeError("LoRA verification failed: frozen base samples changed")
        result = {"base_requires_grad_false": True, "base_grad_none": True,
                  "base_probe_delta_max": 0.0, "optimizer_contains_exactly_adapters": True}
        for kind, probes in self.adapter_probes.items():
            delta = torch.zeros(2, device=self.runtime.device, dtype=torch.float32)
            for name, parameter, indices, before in probes:
                after = _sample_parameter(parameter, indices)
                current = torch.stack(((after - before).abs().max(),
                                       (after - self.previous[name]).abs().max())).to(delta.device)
                delta = torch.maximum(delta, current)
                self.previous[name].copy_(after)
            self._reduce(delta, dist.ReduceOp.MAX)
            if not torch.isfinite(delta).all().item():
                raise RuntimeError("LoRA verification failed: non-finite adapter parameter samples")
            self.maxima[kind] = max(self.maxima[kind], delta[0].item())
            result[kind] = {**gradient_metrics[kind], "parameter_probe_delta_max": delta[0].item(),
                            "step_parameter_probe_delta_max": delta[1].item()}
        return result

    def finish(self):
        if self.maxima["b"] == 0:
            raise RuntimeError("LoRA verification failed: no B parameter update detected")
        return {**self.metadata, "a_parameter_probe_delta_max": self.maxima["a"],
                "b_parameter_probe_delta_max": self.maxima["b"], "verified": True}


def _engram_tables(model):
    return getattr(getattr(model, "model", model), "engram_tables", {})


def _synchronize_engram(model):
    for table in _engram_tables(model).values():
        table.synchronize()


def _trainable_engram_tables(model):
    return {name: table for name, table in _engram_tables(model).items() if table.weight.requires_grad}


def _engram_gradient_rows(gradient, limit=256):
    """Find touched rows without materializing sparse gradients or scanning copies of whole tables."""
    if gradient.is_sparse:
        return gradient._indices()[0].unique()[:limit]
    # Dense gradients already exist for DSV4 AdamW. Inspect a bounded number
    # of elements at once; never make a table-sized boolean/FP32 temporary.
    rows_per_chunk = max(1, 1024**2 // gradient.shape[1])
    selected, count = [], 0
    for start in range(0, gradient.shape[0], rows_per_chunk):
        touched = gradient[start:start + rows_per_chunk].ne(0).any(dim=1).nonzero().flatten()
        touched = touched[:limit - count]
        if touched.numel():
            selected.append(touched + start)
            count += touched.numel()
        if count == limit:
            break
    return torch.cat(selected) if selected else torch.empty(0, dtype=torch.int64, device=gradient.device)


def _engram_update_probes(tables):
    """Observe at most 256 touched rows per trainable table, including CPU rows."""
    probes = {}
    for name, table in tables.items():
        gradient = table.weight.grad
        if gradient is None:
            continue
        gradient = local_tensor(gradient)
        indices = _engram_gradient_rows(gradient)
        if indices.numel():
            before = local_tensor(table.weight).detach()[indices].float().clone()
            probes[name] = (indices, before, gradient._nnz() if gradient.is_sparse else 0)
    return probes


def _engram_update_metrics(tables, probes, runtime):
    records = {}
    for name, table in tables.items():
        maximum = torch.zeros((), dtype=torch.float32, device=runtime.device)
        counts = torch.zeros(2, dtype=torch.int64, device=runtime.device)
        if name in probes:
            indices, before, nnz = probes[name]
            after = local_tensor(table.weight).detach()[indices].float()
            maximum.copy_((after - before).abs().max())
            counts[0], counts[1] = indices.numel(), nnz
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        if not torch.isfinite(maximum).item():
            raise RuntimeError(f"optimizer produced a non-finite Engram table {name}")
        records[name] = {"parameter_delta_max": maximum.item(),
                         "probed_rows": counts[0].item(),
                         "gradient_layout": "sparse" if table.sparse_gradients else "dense"}
        if table.sparse_gradients:
            records[name]["gradient_nnz"] = counts[1].item()
    return records


def _engram_summary(model, update_maxima):
    tables = _engram_tables(model)
    if not tables:
        return {}
    return {"engram_parameter_delta_max": dict(update_maxima),
            "engram_metrics_rank0": {name: dict(table.metrics) for name, table in tables.items()},
            "engram_policies": {name: asdict(table.policy) for name, table in tables.items()},
            "engram_trainable_tables": list(_trainable_engram_tables(model)),
            "engram_table_parameters": sum(table.global_num_embeddings * table.embedding_dim
                                             for table in tables.values())}


def _qwen_engram_metadata(args, model_config=None):
    config, policy = qwen_engram_options(args, model_config)
    if config is None:
        return {}
    return {"qwen_engram": asdict(config), "engram_policy": asdict(policy)}


def _load_model(args, runtime, meshes, cp, dispatcher, cleanup):
    started = time.perf_counter()
    if args.model_type == "qwen":
        engram_config, engram_policy = qwen_engram_options(args)

        def setup(loaded):
            try:
                attach_engram(loaded, engram_config, engram_policy, meshes.engram,
                              device=runtime.device, dtype=torch.bfloat16)
            finally:
                # Install cleanup before subsequent checkpoint reads can fail.
                for table in _engram_tables(loaded).values():
                    cleanup.callback(table.close)
            _record(runtime, phase="engram_initialized", config=asdict(engram_config),
                    policy=asdict(engram_policy),
                    table_parameters=sum(table.global_num_embeddings * table.embedding_dim
                                         for table in _engram_tables(loaded).values()))

        def materialize(name, module):
            shard_qwen_component(name, module, meshes)
            _record(runtime, phase="loading", module=name, seconds=time.perf_counter() - started)

        model = load_qwen3_moe(args.model_path, device=runtime.device, dtype=torch.bfloat16,
                              context_parallel=cp, token_dispatcher=dispatcher,
                              setup_callback=setup if engram_config is not None else None,
                              module_callback=materialize, lora_config=adapter_config(args),
                              adapter_path=args.adapter_in)
        shard_qwen_root(model, meshes)
        if args.checkpoint_mode == "full":
            model.gradient_checkpointing_enable()
    else:
        loading, engram, fsdp = memory_policies(args)

        class ProgressMaterializer(FSDPMaterializer):
            def layer_loaded(self, layer):
                super().layer_loaded(layer)
                _record(runtime, phase="loading", layer=layer.layer_id,
                        seconds=time.perf_counter() - started)

        model = load_dsv41_backbone_window(
            args.model_path, start_layer=args.start_layer, num_layers=args.num_layers,
            device=runtime.device, dtype=torch.bfloat16, context_parallel=cp,
            token_dispatcher=dispatcher, engram_mesh=meshes.engram,
            sparse_engram_gradients=args.optimizer == "sgd",
            materializer=ProgressMaterializer(meshes, fsdp), load_policy=loading,
            engram_policy=engram, lora_config=adapter_config(args),
            cache_quantization=args.cache_quantization, adapter_path=args.adapter_in,
        )
        cleanup.callback(model.close)
        model.set_activation_checkpointing(checkpoint_policy(args))
    model.config.router_aux_loss_coef = args.router_aux_loss_coef
    model.train()
    return model


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _truncate_resumed_metrics(path, completed):
    """Drop uncheckpointed metrics, tolerating a torn final append after a crash."""
    path = Path(path)
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    rows = []
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                break
            raise
        if not isinstance(row, dict) or type(row.get("step")) is not int:
            raise ValueError("invalid step record in metrics.jsonl")
        if row["step"] <= completed:
            rows.append(row)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(path)


def train(args, runtime: Runtime):
    with ExitStack() as cleanup:
        return _train(args, runtime, cleanup)


def _train(args, runtime, cleanup):
    config = validate_args(args, runtime.world_size)
    output = Path(args.output_dir)
    # Every rank preflights the shared directory before any expensive allocation.
    if args.resume:
        if not output.is_dir():
            raise FileNotFoundError(f"resume output directory does not exist: {output}")
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError("output-dir is not empty; choose a new directory or use --resume")
    dist.barrier()
    output.mkdir(parents=True, exist_ok=True)
    tokenizer_path = args.tokenizer_path or str(Path(args.model_path) / "tokenizer.json")
    dataset = SFTDataset.from_jsonl(
        args.data, tokenizer_path=tokenizer_path, max_length=args.seq_len,
        pad_token_id=getattr(config, "pad_token_id", config.eos_token_id),
        bos_token_id=config.bos_token_id, eos_token_id=config.eos_token_id,
    )
    dataset.require_active_tokens(args.min_active_tokens_per_example)
    lengths = dataset.length_statistics()
    encoding = dataset.identity.get("encoding")
    checkpoint_model_type = json.loads((Path(args.model_path) / "config.json").read_text()).get("model_type")
    if encoding and (encoding["config_sha256"] != _file_hash(Path(args.model_path) / "config.json")
                     or encoding.get("model_type") != checkpoint_model_type):
        raise ValueError("prepared chat data model identity differs from mounted checkpoint")
    identity_spec = identity_tokenizer = None
    if args.identity_eval_prompts:
        from scripts.evaluate_identity_qa import load_prompt_spec, capture_identity_generation, compare_identity_generations
        identity_spec, identity_tokenizer = load_prompt_spec(
            args.identity_eval_prompts, tokenizer_path, vocab_size=config.vocab_size,
            max_position_embeddings=config.max_position_embeddings,
            config_sha256=_file_hash(Path(args.model_path) / "config.json"),
            max_new_tokens=args.identity_eval_max_new_tokens)
    if dataset.input_ids.min() < 0 or dataset.input_ids.max() >= config.vocab_size:
        raise ValueError("dataset input token ID lies outside model vocabulary")
    valid_labels = dataset.labels[dataset.labels.ne(-100)]
    if valid_labels.min() < 0 or valid_labels.max() >= config.vocab_size:
        raise ValueError("dataset target token ID lies outside model vocabulary")
    del valid_labels
    # Compare all data/config identities before entering model collectives.
    training_config = {
        "entrypoint": "train_sft_v1", "model_type": args.model_type,
        "source": _source_identity(args.model_path), "data": dataset.identity,
        "batch_size": args.batch_size, "sequence_length": args.seq_len,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "world_size": runtime.world_size, "cp_size": args.cp_size, "ep_size": args.ep_size,
        "seed": args.seed, "optimizer": args.optimizer, "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay, "optimizer_foreach": args.optimizer_foreach,
        "router_aux_loss_coef": args.router_aux_loss_coef,
        "checkpoint_policy": asdict(checkpoint_policy(args)),
        "lora": asdict(adapter_config(args)) if args.lora else None,
        "start_layer": args.start_layer,
        "num_layers": config.num_hidden_layers if args.model_type == "qwen" and args.lora else args.num_layers,
        "cache_quantization": args.cache_quantization,
    }
    if args.model_type == "dsv4":
        training_config["memory_policies"] = [asdict(policy) for policy in memory_policies(args)]
    if args.checkpoint_state == "lora":
        training_config["checkpoint_state"] = "lora"
    if args.min_active_tokens_per_example:
        training_config["min_active_tokens_per_example"] = args.min_active_tokens_per_example
    if args.lora_probe_tokens != 4096:
        training_config["lora_probe_tokens"] = args.lora_probe_tokens
    if identity_spec is not None:
        training_config["identity_generation"] = {
            "prompts_sha256": _file_hash(args.identity_eval_prompts),
            "max_new_tokens": args.identity_eval_max_new_tokens, "seed": args.identity_eval_seed,
            "tokenizer_sha256": identity_spec["tokenizer_sha256"],
            "template_sha256": identity_spec["template_sha256"], "decoding": "greedy-no-cache"}
    training_config.update(_qwen_engram_metadata(args, config))
    # Normalize tuples to JSON lists so DCP and the on-disk manifest agree.
    training_config = json.loads(json.dumps(training_config, sort_keys=True))
    identities = [None] * runtime.world_size
    dist.all_gather_object(identities, training_config)
    if any(identity != training_config for identity in identities):
        raise ValueError("model/data/training configuration differs across ranks")
    manifest_path = output / "run_config.json"
    if args.resume:
        if not manifest_path.is_file() or json.loads(manifest_path.read_text()) != training_config:
            raise ValueError("resume configuration/data differs from run_config.json")
        complete = [int(path.name[5:]) for path in output.glob("step-*")
                    if path.name[5:].isdigit() and (path / ".metadata").is_file()]
        if not complete or args.steps <= max(complete):
            raise ValueError("resume requires a complete checkpoint and --steps greater than its step")
    elif runtime.is_main:
        _write_json(manifest_path, training_config)
    _record(runtime, phase="data_ready", examples=len(dataset), data=dataset.identity, lengths=lengths)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats(runtime.device)
    meshes, cp, dispatcher = build_parallelism(1, args.ep_size, runtime.device.type)
    started = time.perf_counter()
    model = _load_model(args, runtime, meshes, cp, dispatcher, cleanup)
    torch.cuda.synchronize(runtime.device)
    load_seconds = time.perf_counter() - started
    load_peak = torch.cuda.max_memory_allocated(runtime.device)
    optimizer = build_optimizer(model, OptimizerPolicy(args.optimizer, args.learning_rate, args.optimizer_foreach))
    if args.weight_decay and (any(getattr(module, "sparse_gradients", False) for module in model.modules())
                              or (args.optimizer == "sgd" and args.fsdp_offload_layers)):
        raise ValueError("weight decay is unsupported for sparse Engram/CPU-offloaded SGD")
    for group in optimizer.param_groups:
        group["weight_decay"] = args.weight_decay
    gradients = GradientPolicy(model, meshes)
    generator = torch.Generator(device="cpu").manual_seed(args.seed + runtime.rank)
    state = TrainingState(model, optimizer, model_config=model.config.to_dict(),
                          data_generator=generator, training_config=training_config,
                          state_adapter=StateAdapter(model, cp=1, ep=args.ep_size),
                          checkpoint_state=args.checkpoint_state)
    checkpointer = CheckpointManager(output, state)
    completed = checkpointer.load() if args.resume else 0
    probes = _update_probes(model)
    lora_verifier = LoRAVerifier(model, optimizer, runtime) if args.verify_lora else None
    evaluation_memory = {}
    if lora_verifier is not None:
        from scripts.verify_lora_adapter import capture_probe, make_reference

        eval_batch = _bounded_lora_probe(dataset.batch([0]), args.lora_probe_tokens)
        torch.cuda.reset_peak_memory_stats(runtime.device)
        initial_lora_eval = capture_probe(model, eval_batch, runtime)
        evaluation_memory["initial_fixed_probe"] = _cuda_memory_snapshot(runtime)
        _record(runtime, phase="lora_verification_ready", **lora_verifier.metadata)
    if identity_spec is not None:
        torch.cuda.reset_peak_memory_stats(runtime.device)
        base_generation = capture_identity_generation(model, identity_spec, identity_tokenizer, runtime,
            adapters_enabled=False, max_new_tokens=args.identity_eval_max_new_tokens, seed=args.identity_eval_seed)
        evaluation_memory["base_generation_before_training"] = _cuda_memory_snapshot(runtime)
        if runtime.is_main:
            _write_json(output / f"identity-generation-base-step-{completed}.json", {
                "context": training_config, "start_step": completed, "base_before_training": base_generation})
        _record(runtime, phase="base_generation_complete", strict_success_count=base_generation["strict_success_count"])
    engram_tables = _trainable_engram_tables(model)
    engram_update_maxima = {name: 0.0 for name in engram_tables}
    _record(runtime, phase="loaded", load_seconds=load_seconds, start_step=completed,
            allocated_gib=torch.cuda.memory_allocated(runtime.device) / 2**30)
    history_path = output / "metrics.jsonl"
    if runtime.is_main and args.resume and history_path.is_file():
        # A crash may leave logged steps newer than the last complete checkpoint.
        _truncate_resumed_metrics(history_path, completed)
    history = cleanup.enter_context(history_path.open("a", encoding="utf-8")) if runtime.is_main else None
    torch.cuda.reset_peak_memory_stats(runtime.device)
    dist.barrier()
    train_started = time.perf_counter()
    seen_targets = seen_tokens = seen_processed_tokens = 0
    training_peak_allocated = training_peak_reserved = 0.0
    last_metrics = None
    for step_index in range(completed, args.steps):
        torch.cuda.reset_peak_memory_stats(runtime.device)
        step_started = time.perf_counter()
        batches = [dataset.batch_for_step(step_index, micro, batch_size=args.batch_size,
                   accumulation_steps=args.gradient_accumulation_steps,
                   data_rank=runtime.rank, data_world_size=runtime.world_size,
                   seed=args.seed) for micro in range(args.gradient_accumulation_steps)]
        counts = torch.tensor([sum(int(batch["supervised_tokens"]) for batch in batches),
                               sum(int(batch["attention_mask"].sum()) for batch in batches)],
                              dtype=torch.int64, device=runtime.device)
        dist.all_reduce(counts)
        global_targets, global_tokens = counts.tolist()
        processed_tokens = args.batch_size * args.gradient_accumulation_steps * args.seq_len * runtime.world_size
        active_lengths = torch.cat([batch["attention_mask"].sum(dim=1) for batch in batches])
        active_min = active_lengths.min().to(runtime.device)
        active_max = active_lengths.max().to(runtime.device)
        dist.all_reduce(active_min, op=dist.ReduceOp.MIN)
        dist.all_reduce(active_max, op=dist.ReduceOp.MAX)
        optimizer.zero_grad(set_to_none=True)
        gradients.begin_step()
        totals = torch.zeros(3, dtype=torch.float64, device=runtime.device)
        for batch in batches:
            count = int(batch["supervised_tokens"])
            inputs = {key: batch[key].to(runtime.device) for key in ("input_ids", "labels", "attention_mask")}
            kwargs = {"output_router_logits": args.router_aux_loss_coef > 0} if args.model_type == "qwen" else {}
            result = model(**inputs, **kwargs)
            if result.loss is None:
                raise RuntimeError("model did not return a supervised loss")
            _collective_finite(result.loss, runtime, "loss")
            auxiliary = result.aux_loss if isinstance(result.aux_loss, torch.Tensor) else result.loss.new_zeros(())
            ce_loss = result.loss - args.router_aux_loss_coef * auxiliary
            scale = loss_scale(count, global_targets, runtime.world_size)
            (result.loss * scale).backward()
            totals += torch.stack((result.loss.detach(), ce_loss.detach(), auxiliary.detach())).double() * count
            del result, inputs, auxiliary, ce_loss
        gradients.normalize()
        _synchronize_engram(model)
        _finite_gradients(model, runtime)
        lora_gradients = lora_verifier.gradients() if lora_verifier is not None else None
        table_probes = _engram_update_probes(engram_tables)
        optimizer.step()
        lora_updates = lora_verifier.after_step(lora_gradients) if lora_verifier is not None else None
        table_updates = _engram_update_metrics(engram_tables, table_probes, runtime)
        del table_probes
        for name, record in table_updates.items():
            engram_update_maxima[name] = max(engram_update_maxima[name], record["parameter_delta_max"])
        dist.all_reduce(totals)
        totals /= global_targets
        _collective_finite(totals, runtime, "step metrics")
        delta = _parameter_delta(probes, runtime)
        if not math.isfinite(delta):
            raise RuntimeError("optimizer produced a non-finite probed parameter")
        torch.cuda.synchronize(runtime.device)
        step_memory = _cuda_memory_snapshot(runtime)
        training_peak_allocated = max(training_peak_allocated, step_memory["peak_allocated_gib"])
        training_peak_reserved = max(training_peak_reserved, step_memory["peak_reserved_gib"])
        elapsed = torch.tensor(time.perf_counter() - step_started, device=runtime.device)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        last_metrics = {"step": step_index + 1, "loss": totals[0].item(), "ce_loss": totals[1].item(),
                        "aux_loss": totals[2].item(), "supervised_tokens": global_targets,
                        "input_tokens": global_tokens, "seconds": elapsed.item(),
                        "supervised_tokens_per_second": global_targets / elapsed.item(),
                        "parameter_probe_delta_max": delta, "loss_finite": True, "gradients_finite": True,
                        "sequence_length": args.seq_len, "processed_tokens": processed_tokens,
                        "padded_tokens": processed_tokens - global_tokens,
                        "active_tokens_min": int(active_min), "active_tokens_max": int(active_max),
                        "cuda_memory": step_memory}
        if table_updates:
            last_metrics["engram_updates"] = table_updates
        if lora_updates is not None:
            last_metrics["lora_updates"] = lora_updates
        seen_targets += global_targets
        seen_tokens += global_tokens
        seen_processed_tokens += processed_tokens
        _record(runtime, **last_metrics)
        if history is not None:
            history.write(json.dumps(last_metrics) + "\n")
            history.flush()
        if args.save_every and (step_index + 1) % args.save_every == 0 and step_index + 1 < args.steps:
            _synchronize_engram(model)
            checkpointer.save(step_index + 1)
            _record(runtime, phase="saved", step=step_index + 1)
    train_seconds = time.perf_counter() - train_started
    # Capture training separately before any post-training generation or save.
    training_memory = {"peak_allocated_gib": training_peak_allocated,
                       "peak_reserved_gib": training_peak_reserved,
                       "end_allocated_gib": _cuda_memory_snapshot(runtime)["allocated_gib"]}
    _record(runtime, phase="training_finished", steps=args.steps, memory=training_memory)
    if last_metrics["parameter_probe_delta_max"] == 0:
        raise RuntimeError("no parameter update detected in dense probes; check gradients/lr/BF16 rounding")
    unchanged_tables = [name for name, value in engram_update_maxima.items() if value == 0]
    if unchanged_tables:
        raise RuntimeError(f"no Engram updates detected in tables {unchanged_tables}; check gradients/lr/BF16 rounding")
    lora_summary = lora_verifier.finish() if lora_verifier is not None else None
    if lora_verifier is not None:
        torch.cuda.reset_peak_memory_stats(runtime.device)
        trained_lora_eval = capture_probe(model, eval_batch, runtime)
        evaluation_memory["final_fixed_probe"] = _cuda_memory_snapshot(runtime)
        # Validate shared eval evidence on every rank before an expensive DCP
        # save; a failed output-change check must not leave peers in a collective.
        make_reference(training_config, initial_lora_eval, trained_lora_eval)
    if identity_spec is not None:
        torch.cuda.reset_peak_memory_stats(runtime.device)
        adapted_generation = capture_identity_generation(model, identity_spec, identity_tokenizer, runtime,
            adapters_enabled=True, max_new_tokens=args.identity_eval_max_new_tokens, seed=args.identity_eval_seed)
        generation_effect = compare_identity_generations(base_generation, adapted_generation)
        evaluation_memory["adapter_generation_after_training"] = _cuda_memory_snapshot(runtime)
        _record(runtime, phase="adapter_generation_complete", comparison=generation_effect)
    _synchronize_engram(model)
    torch.cuda.reset_peak_memory_stats(runtime.device)
    checkpoint_path = checkpointer.save(args.steps)
    if args.adapter_out:
        save_adapter(model, args.adapter_out)
    if lora_verifier is not None:
        dist.barrier()
        if runtime.is_main:
            reference = make_reference(training_config, initial_lora_eval, trained_lora_eval, args.adapter_out)
            _write_json(output / "lora_eval_reference.json", reference)
    saving_memory = _cuda_memory_snapshot(runtime)
    if identity_spec is not None and runtime.is_main:
        _write_json(output / "identity-generation.json", {
            "format_version": 1, "context": training_config, "start_step": completed, "final_step": args.steps,
            "adapter_sha256": _file_hash(args.adapter_out), "base_before_training": base_generation,
            "adapted_after_training": adapted_generation, "comparison": generation_effect,
            "scope": "The same frozen backbone and prompts before and after SFT; adapters bypassed for baseline, dropout disabled for both. Fixed identity prompts do not establish general model quality."})
    memory = torch.tensor([load_peak, torch.cuda.max_memory_allocated(runtime.device),
                           torch.cuda.max_memory_reserved(runtime.device)], device=runtime.device, dtype=torch.int64)
    dist.all_reduce(memory, op=dist.ReduceOp.MAX)
    elapsed = torch.tensor(train_seconds, device=runtime.device)
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    summary = {**last_metrics, "model_type": args.model_type, "checkpoint": str(checkpoint_path.resolve()),
               "checkpoint_state": args.checkpoint_state,
               "source": training_config["source"], "world_size": runtime.world_size,
               "cp_size": 1, "ep_size": args.ep_size, "start_step": completed,
               "steps": args.steps, "data": dataset.identity, "load_seconds_rank0": load_seconds,
               "train_seconds": elapsed.item(), "trained_supervised_tokens": seen_targets,
               "trained_input_tokens": seen_tokens,
               "supervised_tokens_per_second": seen_targets / elapsed.item(),
               "load_peak_gib": memory[0].item() / 2**30, "train_peak_gib": training_peak_allocated,
               "reserved_peak_gib": training_peak_reserved,
               "memory_reporting": "train_peak_gib/reserved_peak_gib cover optimizer steps only; evaluation and saving are separate phases",
               "training_memory": training_memory, "evaluation_memory": evaluation_memory,
               "checkpoint_and_adapter_save_memory": saving_memory, "data_lengths": lengths,
               "trained_processed_tokens": seen_processed_tokens,
               "parameter_bytes_rank0": parameter_memory(model, optimizer),
               "update_probe_parameters": [name for name, *_ in probes],
               "update_verification": "max absolute change in up to 256 values per trainable dense tensor"}
    if args.qwen_engram:
        summary.update(_qwen_engram_metadata(args, config))
    summary.update(_engram_summary(model, engram_update_maxima))
    if lora_summary is not None:
        summary["lora_verification"] = lora_summary
        summary["lora_eval_reference"] = str((output / "lora_eval_reference.json").resolve())
        summary["fixed_lora_probe"] = {"sequence_length": eval_batch["input_ids"].shape[1],
                                       "training_sequence_length": args.seq_len,
                                       "selection": "unchanged short example or tail window retaining final supervised target; not a long-context quality evaluation"}
    if identity_spec is not None:
        summary["identity_generation"] = {"path": str((output / "identity-generation.json").resolve()), **generation_effect}
    if runtime.is_main:
        _write_json(output / "summary.json", summary)
    _record(runtime, phase="complete", summary=summary)
    return summary


def main():
    args = parse_args()
    runtime = None
    try:
        runtime = initialize_runtime(require_distributed=True)
        train(args, runtime)
    finally:
        if runtime is not None and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
