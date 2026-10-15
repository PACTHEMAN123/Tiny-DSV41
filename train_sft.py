"""Text SFT on released DSV4 or Qwen weights using the native parallel paths.

Launch with torchrun. --steps is the total optimizer-step target; --steps 0
loads (and optionally resumes) a model for fixed teacher-forced evaluation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import torch
import torch.distributed as dist

from dsv41_train.lora import LoRAConfig, adapter_parameters, inject_lora, save_adapter
from dsv41_train.rl.checkpoint import run_identity
from dsv41_train.rl.tokenizer import ModelTokenizer
from dsv41_train.runtime import initialize_runtime, local_tensor
from dsv41_train.sft_data import collate_examples, load_jsonl


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data", required=True, help="JSONL prompt/answer or messages/answer records")
    parser.add_argument("--eval-data", help="Defaults to the training data")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--family", choices=("dsv4", "qwen"), default="dsv4")
    parser.add_argument("--mode", choices=("full", "lora"), default="lora")
    parser.add_argument("--num-layers", type=int, help="DSV4 prefix; default is the complete backbone")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=4096, help="Actual token limit, without truncation")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--optimizer", choices=("sgd", "adamw"))
    parser.add_argument("--cp-size", type=int, default=1)
    parser.add_argument("--ep-size", type=int, default=1)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lora-targets", help="Comma-separated layer-relative linear module suffixes")
    parser.add_argument("--engram-offload", action="store_true", help="Keep DSV4 Engram row shards on CPU")
    parser.add_argument("--engram-cache-mib", type=int, default=0, help="Per-rank row cache budget including tags, split across offloaded tables")
    parser.add_argument("--train-engram", action="store_true", help="Also update Engram tables in LoRA mode (SGD)")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-batches", type=int, default=1, help="Fixed batches per data group")
    parser.add_argument("--generation-eval", action="store_true", help="Also generate from fixed evaluation prompts without answers")
    parser.add_argument("--generation-max-new-tokens", type=int, default=16)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--resume", help="Same-topology SFT checkpoint directory")
    return parser.parse_args(argv)


def validate_args(args, world_size):
    if args.steps < 0 or args.save_every < 0:
        raise ValueError("steps and save-every must be nonnegative")
    if min(args.batch_size, args.gradient_accumulation_steps, args.eval_batches, args.generation_max_new_tokens) < 1 or args.seq_len < 2:
        raise ValueError("batch-size, accumulation and eval-batches must be positive; seq-len must be at least 2")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning-rate must be positive and finite")
    if min(args.cp_size, args.ep_size) < 1 or world_size % args.cp_size or world_size % args.ep_size:
        raise ValueError("cp-size and ep-size must divide world size")
    if (args.family == "qwen" or args.cp_size > 1) and args.cp_size != args.ep_size:
        raise ValueError("this topology requires cp-size == ep-size")
    if args.family != "dsv4" and (args.engram_offload or args.train_engram):
        raise ValueError("Engram options require the DSV4 model family")
    if args.engram_cache_mib < 0 or (args.engram_cache_mib and not args.engram_offload):
        raise ValueError("engram-cache-mib must be nonnegative and requires engram-offload")
    if args.seq_len % args.cp_size:
        raise ValueError("seq-len must divide evenly across cp-size")
    for name in ("config.json", "model.safetensors.index.json"):
        if not (Path(args.model_path) / name).is_file():
            raise FileNotFoundError(Path(args.model_path) / name)
    if args.family == "dsv4":
        from dsv41_train.models.dsv4 import DeepSeekV41Config
        config = DeepSeekV41Config.from_json(Path(args.model_path) / "config.json")
        args.num_layers = config.num_hidden_layers if args.num_layers is None else args.num_layers
        if not 1 <= args.num_layers <= config.num_hidden_layers:
            raise ValueError("num-layers must select a nonempty DSV4 backbone prefix")
        if args.engram_cache_mib and not any(layer < args.num_layers for layer in config.engram_layer_ids):
            raise ValueError("the selected backbone has no Engram table to cache")
        experts = config.n_routed_experts
    else:
        from dsv41_train.models.qwen import Qwen3MoeConfig
        if args.num_layers is not None:
            raise ValueError("num-layers is supported only for DSV4")
        experts = Qwen3MoeConfig.from_json(Path(args.model_path) / "config.json").num_experts
    if experts % args.ep_size:
        raise ValueError("expert count must divide evenly across ep-size")
    args.optimizer = args.optimizer or ("sgd" if args.mode == "full" or args.train_engram else "adamw")
    if (args.train_engram or (args.engram_offload and args.mode == "full")) and args.optimizer != "sgd":
        raise ValueError("trainable offloaded Engram or LoRA with trainable Engram requires SGD")


def lora_config(args):
    defaults = LoRAConfig().targets if args.family == "dsv4" else (
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    )
    targets = tuple(name.strip() for name in args.lora_targets.split(",")) if args.lora_targets else defaults
    return LoRAConfig(args.lora_rank, args.lora_alpha, args.lora_dropout, targets)


def load_model(args, runtime):
    """Use main's loaders and wrapping order; adapters are installed before FSDP."""
    config = lora_config(args) if args.mode == "lora" else None
    if args.family == "dsv4":
        from dsv41_train.models.dsv4 import load_dsv41_backbone_window
        from dsv41_train.models.dsv4.parallel import (
            apply_fsdp2_layer, apply_fsdp2_root, build_parallelism, synchronize_cp_gradients,
        )
        meshes, cp, dispatcher = build_parallelism(args.cp_size, args.ep_size, runtime.device.type)

        def layer_loaded(layer):
            if config is not None:
                inject_lora(layer, config)
            apply_fsdp2_layer(layer, meshes)

        model = load_dsv41_backbone_window(
            args.model_path, num_layers=args.num_layers, device=runtime.device, dtype=torch.bfloat16,
            context_parallel=cp, token_dispatcher=dispatcher, engram_mesh=meshes.engram,
            sparse_engram_gradients=args.mode == "lora" or args.optimizer == "sgd",
            engram_offload=args.engram_offload,
            layer_loaded=layer_loaded,
        )
        if config is not None:
            adapter_ids = {id(parameter) for parameter in adapter_parameters(model)}
            for parameter in model.parameters():
                if id(parameter) not in adapter_ids:
                    parameter.requires_grad_(False)
            model.model.engram_tables.requires_grad_(args.train_engram)
        apply_fsdp2_root(model, meshes)
        if args.engram_cache_mib:
            tables = list(model.model.engram_tables.values())
            budget, remainder = divmod(args.engram_cache_mib * 2**20, len(tables))
            for index, table in enumerate(tables):
                table.configure_cache(budget + (index < remainder), runtime.device)
        synchronize = lambda: synchronize_cp_gradients(model, cp)
    else:
        from dsv41_train.models.qwen import load_qwen3_moe
        from dsv41_train.models.qwen.parallel import ParallelParameters, apply_fsdp2, build_parallelism
        meshes, cp, dispatcher = build_parallelism(args.cp_size, args.ep_size, runtime.device.type)
        model = load_qwen3_moe(args.model_path, device=runtime.device, dtype=torch.bfloat16,
                              context_parallel=cp, token_dispatcher=dispatcher)
        if config is not None:
            inject_lora(model, config)
        apply_fsdp2(model, meshes)
        parameters = ParallelParameters.collect(model)
        synchronize = lambda: parameters.synchronize(cp)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    return model, synchronize


def make_batches(examples, cursor, *, batch_size, count, runtime, cp_size, pad_token_id):
    """Identical data within each CP group; deterministic global cyclic cursor."""
    data_size, data_rank = runtime.world_size // cp_size, runtime.rank // cp_size
    batches = []
    for micro in range(count):
        start = cursor + (micro * data_size + data_rank) * batch_size
        selected = [examples[(start + index) % len(examples)] for index in range(batch_size)]
        batches.append(collate_examples(selected, pad_token_id=pad_token_id, pad_to_multiple_of=cp_size))
    return batches, cursor + count * data_size * batch_size


def sft_loss(model, batch, family):
    options = {"compute_aux_loss": False} if family == "dsv4" else {"output_router_logits": False}
    loss = model(**batch, **options).loss
    if loss is None:
        raise RuntimeError("model did not produce an SFT loss")
    return loss


def _reduce(value, runtime, op=dist.ReduceOp.SUM):
    if runtime.distributed:
        dist.all_reduce(value, op=op)
    return value


def _checked_loss(model, batch, family, runtime):
    loss = sft_loss(model, {name: value.to(runtime.device) for name, value in batch.items()}, family)
    finite = torch.isfinite(loss.detach()).to(torch.int32)
    if not _reduce(finite, runtime, dist.ReduceOp.MIN).item():
        raise RuntimeError("SFT produced a non-finite loss")
    return loss


def _check_gradients(model, runtime):
    finite = torch.ones((), device=runtime.device, dtype=torch.int32)
    for parameter in model.parameters():
        if parameter.requires_grad and parameter.grad is not None:
            gradient = local_tensor(parameter.grad)
            if gradient.is_sparse:
                gradient = gradient.float().coalesce().values().to(gradient.dtype)
            finite = finite & torch.isfinite(gradient).all().to(device=runtime.device, dtype=torch.int32)
    if not _reduce(finite, runtime, dist.ReduceOp.MIN).item():
        raise RuntimeError("SFT produced a non-finite gradient before optimizer.step")


def train_update(model, optimizer, batches, *, family, runtime, cp_size=1, synchronize=lambda: None):
    """Normalize accumulated gradients by global supervised tokens, not batches."""
    counts = [(batch["labels"][:, 1:] != -100).sum().item() for batch in batches]
    totals = torch.tensor([sum(counts), sum(batch["attention_mask"].sum().item() for batch in batches)],
                          device=runtime.device, dtype=torch.float64)
    totals = _reduce(totals, runtime) / cp_size
    if not counts or min(counts) == 0 or totals[0].item() <= 0:
        raise ValueError("each SFT microbatch must contain supervised next-token targets")
    optimizer.zero_grad(set_to_none=True)
    weighted_loss = torch.zeros((), device=runtime.device, dtype=torch.float64)
    model.train()
    for batch, count in zip(batches, counts):
        loss = _checked_loss(model, batch, family, runtime)
        (loss * ((runtime.world_size // cp_size) * count / totals[0])).backward()
        weighted_loss += loss.detach().double() * count
    synchronize()
    _check_gradients(model, runtime)
    optimizer.step()
    loss_value = (_reduce(weighted_loss, runtime) / cp_size / totals[0]).item()
    lengths = [length for batch in batches for length in batch["attention_mask"].sum(dim=1).tolist()]
    bounds = _max_metrics(runtime, input_tokens_per_example_max=max(lengths), negative_min=-min(lengths))
    return {"loss": loss_value, "supervised_tokens": int(totals[0].item()), "input_tokens": int(totals[1].item()),
            "input_tokens_per_example_min": int(-bounds["negative_min"]),
            "input_tokens_per_example_max": int(bounds["input_tokens_per_example_max"])}


@torch.no_grad()
def evaluate(model, batches, *, family, runtime, cp_size=1):
    was_training = model.training
    model.eval()
    totals = torch.zeros(2, device=runtime.device, dtype=torch.float64)
    try:
        for batch in batches:
            count = (batch["labels"][:, 1:] != -100).sum().item()
            loss = _checked_loss(model, batch, family, runtime)
            totals[0] += loss.double() * count
            totals[1] += count
        totals = _reduce(totals, runtime) / cp_size
        if totals[1].item() <= 0:
            raise ValueError("evaluation requires supervised tokens")
        return (totals[0] / totals[1]).item()
    finally:
        model.train(was_training)


def _max_metrics(runtime, **values):
    tensor = torch.tensor(list(values.values()), dtype=torch.float64, device=runtime.device)
    return dict(zip(values, _reduce(tensor, runtime, dist.ReduceOp.MAX).tolist()))


def _cache_counts(model):
    counters = [table._cpu_lookup.cache_metrics for table in model.model.engram_tables.values()
                if table._cpu_lookup is not None]
    return {key: sum(item[key] for item in counters)
            for key in ("hits", "misses", "transferred_rows", "resident_bytes")}


def train(args, runtime):
    from dsv41_train.sft_checkpoint import load_checkpoint, save_checkpoint
    validate_args(args, runtime.world_size)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    tokenizer = ModelTokenizer(args.model_path)
    examples = load_jsonl(args.data, tokenizer, max_length=args.seq_len)
    eval_examples = load_jsonl(args.eval_data, tokenizer, max_length=args.seq_len) if args.eval_data else examples
    identity = run_identity(args.model_path, args.data, False, eval_data_path=args.eval_data)
    identity["training_config"] = {key: getattr(args, key) for key in (
        "family", "mode", "num_layers", "batch_size", "gradient_accumulation_steps", "seq_len",
        "learning_rate", "optimizer", "cp_size", "ep_size", "seed", "gradient_checkpointing",
    )}
    identity["training_config"].update({key: True for key in ("engram_offload", "train_engram")
                                        if getattr(args, key)})
    identity["training_config"].update(world_size=runtime.world_size, enable_thinking=False,
                                       lora=asdict(lora_config(args)) if args.mode == "lora" else None)
    # JSON-normalize tuples for portable identity metadata and strict resume checks.
    identity = json.loads(json.dumps(identity))
    torch.cuda.reset_peak_memory_stats(runtime.device)
    started = time.perf_counter()
    model, synchronize = load_model(args, runtime)
    torch.cuda.synchronize(runtime.device)
    load_metrics = _max_metrics(runtime, load_seconds=time.perf_counter() - started,
                               load_peak_allocated_gib=torch.cuda.max_memory_allocated(runtime.device) / 2**30,
                               load_peak_reserved_gib=torch.cuda.max_memory_reserved(runtime.device) / 2**30)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer_type = torch.optim.SGD if args.optimizer == "sgd" else torch.optim.AdamW
    optimizer = optimizer_type(parameters, lr=args.learning_rate, weight_decay=0, foreach=False)
    if args.train_engram or (args.engram_offload and args.mode == "full"):
        from dsv41_train.engram_offload import install_engram_sgd_hooks
        install_engram_sgd_hooks(model, optimizer)
    step, cursor = load_checkpoint(args.resume, model, optimizer, identity=identity) if args.resume else (0, 0)
    expected_cursor = step * args.gradient_accumulation_steps * args.batch_size * (runtime.world_size // args.cp_size)
    if type(cursor) is not int or cursor != expected_cursor:
        raise ValueError("checkpoint data cursor does not match completed SFT steps")
    if args.steps and step > args.steps:
        raise ValueError("resume step exceeds the requested total steps")
    initial_step = step
    output_dir = Path(args.output_dir)
    if runtime.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)

    def emit(event):
        if runtime.is_main:
            print(json.dumps(event), flush=True)
            with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as destination:
                destination.write(json.dumps(event) + "\n")

    batch_options = dict(batch_size=args.batch_size, runtime=runtime, cp_size=args.cp_size,
                         pad_token_id=tokenizer.eos_token_ids[0])
    eval_batches, _ = make_batches(eval_examples, 0, count=args.eval_batches, **batch_options)

    def generation_eval():
        if not args.generation_eval:
            return None
        from dsv41_train.sft_eval import evaluate_answers
        count = args.eval_batches * args.batch_size * (runtime.world_size // args.cp_size)
        return evaluate_answers(model, eval_examples[:count], tokenizer, family=args.family,
                                context_parallel=model.context_parallel,
                                max_new_tokens=args.generation_max_new_tokens)

    torch.cuda.reset_peak_memory_stats(runtime.device)
    before = evaluate(model, eval_batches, family=args.family, runtime=runtime, cp_size=args.cp_size)
    generation_before = generation_eval()
    eval_before_metrics = _max_metrics(runtime, eval_before_peak_allocated_gib=torch.cuda.max_memory_allocated(runtime.device) / 2**30)
    emit({"event": "loaded", "step": step, "eval_loss_before": before, "generation_before": generation_before,
          **load_metrics, **eval_before_metrics})
    saved_step = None
    training_started = time.perf_counter()
    train_step_seconds_total = 0.0
    for step in range(initial_step + 1, args.steps + 1):
        batches, next_cursor = make_batches(examples, cursor, count=args.gradient_accumulation_steps, **batch_options)
        torch.cuda.synchronize(runtime.device)
        torch.cuda.reset_peak_memory_stats(runtime.device)
        step_started = time.perf_counter()
        cache_before = _cache_counts(model) if args.engram_cache_mib else None
        metrics = train_update(model, optimizer, batches, family=args.family, runtime=runtime,
                               cp_size=args.cp_size, synchronize=synchronize)
        torch.cuda.synchronize(runtime.device)
        metrics.update(_max_metrics(runtime, seconds=time.perf_counter() - step_started,
                                    peak_allocated_gib=torch.cuda.max_memory_allocated(runtime.device) / 2**30,
                                    peak_reserved_gib=torch.cuda.max_memory_reserved(runtime.device) / 2**30))
        if cache_before is not None:
            cache_after = _cache_counts(model)
            keys = ("hits", "misses", "transferred_rows")
            counts = torch.tensor([cache_after[key] - cache_before[key] for key in keys],
                                  device=runtime.device, dtype=torch.int64)
            counters = dict(zip(keys, _reduce(counts, runtime).tolist()))
            counters["hit_rate"] = counters["hits"] / max(1, counters["hits"] + counters["misses"])
            counters.update(_max_metrics(runtime, max_rank_resident_bytes=cache_after["resident_bytes"]))
            metrics["engram_cache"] = counters
        train_step_seconds_total += metrics["seconds"]
        cursor = next_cursor
        emit({"event": "train", "step": step, "data_position": cursor, **metrics})
        if args.save_every and step % args.save_every == 0:
            save_checkpoint(output_dir / f"checkpoint-{step:08d}", model, optimizer,
                            step=step, identity=identity, data_position=cursor)
            saved_step = step
    train_seconds = _max_metrics(runtime, loop_seconds=time.perf_counter() - training_started)
    train_seconds["train_step_seconds_total"] = train_step_seconds_total
    if step > initial_step and step != saved_step:
        save_checkpoint(output_dir / f"checkpoint-{step:08d}", model, optimizer,
                        step=step, identity=identity, data_position=cursor)
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(runtime.device)
    after = evaluate(model, eval_batches, family=args.family, runtime=runtime, cp_size=args.cp_size)
    generation_after = generation_eval()
    eval_after_metrics = _max_metrics(runtime, eval_after_peak_allocated_gib=torch.cuda.max_memory_allocated(runtime.device) / 2**30)
    if args.mode == "lora" and not args.train_engram and step > initial_step:
        save_adapter(model, output_dir / "adapter.pt")
        if runtime.is_main:
            (output_dir / "adapter_metadata.json").write_text(json.dumps({"identity": identity, "step": step}, indent=2) + "\n")
    summary = {"event": "summary", "step": step, "initial_step": initial_step, "data_position": cursor,
               "eval_loss_before": before, "eval_loss_after": after, **load_metrics, **train_seconds,
               "generation_before": generation_before, "generation_after": generation_after,
               **eval_before_metrics, **eval_after_metrics}
    emit(summary)
    if runtime.is_main:
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    runtime = None
    try:
        args = parse_args()
        runtime = initialize_runtime(require_distributed=True)
        train(args, runtime)
    finally:
        if runtime is not None and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
