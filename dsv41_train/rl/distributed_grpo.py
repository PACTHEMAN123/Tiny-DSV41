"""Batched native query-LoRA GRPO with a persistent remote vLLM rollout service.

All EP16/FSDP16 learner ranks score the same response microbatch. One logical
G32 group contributes equal response weights and produces one optimizer step.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid

import torch
import torch.distributed as dist

from rollout_manager.backends.vllm_lora_remote import RemoteLoRAClient
from rollout_manager.backends.vllm_lora_transport import policy_ack
from .algorithm import (decoupled_token_tis_loss, group_advantages, grpo_loss,
                        response_log_probs, response_log_probs_batch)
from .gsm8k import DATASET_REVISION, formatting_correct, prepare_data, wilson_interval
from .learner_batch_planner import parse_learner_batch_shapes, plan_learner_microbatches
from .lora_config import DistributedLoRAConfig, SUPPORTED_LORA_RANKS, parse_lora_layers
from .lora_export import export_q_a_update
from .native_lora import (adapter_parameters_at, adapter_snapshot, default_native_source, gradient_norm,
                          load_distributed_learner, reference_base, verify_optimizer_step)
from .tokenizer import ModelTokenizer


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        os.chmod(path, 0o600)
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")

def append_json(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        os.chmod(path, 0o600)
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()

def rank_zero_call(function):
    """Broadcast errors and five-second progress pulses during bounded host I/O.

    Actor startup, a G32 group, and evaluation may exceed the NCCL watchdog's
    600-second collective deadline. Pulses keep healthy peers out of a single
    long blocked broadcast; the remote operation retains its own finite timeout.
    This helper is for rank-zero host I/O, never FSDP model operations.
    """
    executor = ThreadPoolExecutor(max_workers=1) if dist.get_rank() == 0 else None
    future = executor.submit(function) if executor is not None else None
    try:
        while True:
            envelope = [None]
            if future is not None:
                wait([future], timeout=5)
                if not future.done():
                    envelope[0] = {'heartbeat': True}
                else:
                    try:
                        envelope[0] = {'result': future.result()}
                    except BaseException as error:
                        envelope[0] = {'error_type': type(error).__name__, 'error': str(error)}
            dist.broadcast_object_list(envelope, src=0)
            if envelope[0].get('heartbeat'):
                continue
            if 'error' in envelope[0]:
                raise RuntimeError(f"rank-zero operation failed: {envelope[0]['error_type']}: {envelope[0]['error']}")
            return envelope[0]['result']
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

def semantic_configuration(args):
    paths = {"model_dir", "native_source", "train_jsonl", "test_jsonl", "output_dir"}
    return {key: value for key, value in vars(args).items() if key not in paths}

def request_specs(record, *, count, phase, index, seed, max_tokens, temperature, lineage):
    return [{"prompt_ids": record.prompt_ids, "max_tokens": max_tokens,
        "temperature": temperature, "sampling_seed": seed + index * 32 + sample,
        "request_id": hashlib.sha256(f"{lineage}:{phase}:{index}:{sample}".encode()).hexdigest()[:32]}
        for sample in range(count)]

def validate_results(results, record, expected_ack, expected_count):
    if not isinstance(results, list) or len(results) != expected_count:
        raise ValueError("only complete actual rollout groups may enter the learner")
    for result in results:
        response, old = result["action_token_ids"], result["behavior_log_probs"]
        if (result["context_token_ids"] != record.prompt_ids or not response
                or len(response) != len(old) or any(type(t) is not int or t < 0 for t in response)
                or any(not math.isfinite(p) or p > 1e-6 for p in old)
                or result.get("logprob_convention") != "sampled_distribution"
                or result["generation_metrics"]["policy_ack"] != expected_ack
                or result["finish_reason"] not in {"eos", "stop", "length"}
                or result["stop_token_count"] not in (0, 1)):
            raise ValueError("rollout tokens, sampled probabilities, or immutable policy identity differ")
    return results

def sample_record(client, record, *, count, phase, index, args, lineage, temperature=None):
    expected = dict(client.ack)
    specs = request_specs(record, count=count, phase=phase, index=index, seed=args.seed,
        max_tokens=args.max_new_tokens,
        temperature=args.temperature if temperature is None else temperature, lineage=lineage)
    results = validate_results(client.sample_group(specs), record, expected, count)
    for spec, result in zip(specs, results):
        metrics = result["generation_metrics"]
        if (metrics.get("request_id") != spec["request_id"]
                or metrics.get("sampling_seed") != spec["sampling_seed"]):
            raise ValueError("rollout response order or sampling identity differs")
    return results


def score_rewards(results, record, reward, tokenizer):
    values, details = [], []
    for result in results:
        ids = result["action_token_ids"]
        stop = result["stop_token_count"]
        content = ids[:-stop] if stop else ids
        value = reward(record, content)
        if value not in (0.0, 1.0):
            raise ValueError("GSM8K numeric correctness must be binary")
        values.append(value)
        details.append({"reward": value,
            "format_correct": formatting_correct(tokenizer.decode(content, skip_special_tokens=True)),
            "prompt_tokens": len(record.prompt_ids), "output_tokens": len(ids),
            "finish_reason": result["finish_reason"], "stop_token_count": stop})
    return values, details

def evaluate(client, records, reward, tokenizer, args, lineage, phase):
    """Evaluate a fixed policy in batches, retaining aggregate scores only."""
    correct = format_correct = output_tokens = 0
    batch_sizes = []
    expected = dict(client.ack)
    started = time.monotonic()
    for start in range(0, len(records), 32):
        batch = records[start:start + 32]
        specs = [request_specs(record, count=1, phase=phase, index=start + offset,
                    seed=args.seed, max_tokens=args.max_new_tokens,
                    temperature=args.eval_temperature, lineage=lineage)[0]
                 for offset, record in enumerate(batch)]
        results = client.sample_group(specs)
        if not isinstance(results, list) or len(results) != len(batch) or client.ack != expected:
            raise ValueError("evaluation requires a complete batch under its fixed policy identity")
        for record, spec, result in zip(batch, specs, results):
            validate_results([result], record, expected, 1)
            metrics = result["generation_metrics"]
            if (metrics.get("sampling_seed") != spec["sampling_seed"]
                    or metrics.get("request_id") != spec["request_id"]):
                raise ValueError("evaluation batch response order or request sampling identity differs")
        batch_sizes.append(len(batch))
        for record, result in zip(batch, results):
            _, details = score_rewards([result], record, reward, tokenizer)
            correct += int(details[0]["reward"])
            format_correct += int(details[0]["format_correct"])
            output_tokens += details[0]["output_tokens"]
    total = len(records)
    ordered_ids = json.dumps([record.metadata["id"] for record in records], separators=(",", ":"))
    return {"n": total, "correct": correct, "accuracy": correct / total,
        "wilson_95_interval": wilson_interval(correct, total),
        "format_accuracy": format_correct / total,
        "elapsed_seconds": time.monotonic() - started, "output_tokens": output_tokens,
        "temperature": args.eval_temperature, "request_batch_sizes": batch_sizes,
        "policy_ack": expected, "ordered_ids_sha256": hashlib.sha256(ordered_ids.encode()).hexdigest()}


def validate_correction_configuration(args):
    mode = getattr(args, "rollout_correction", "none")
    if mode not in {"none", "decoupled-token-tis"}:
        raise ValueError("unknown rollout correction mode")
    if mode == "decoupled-token-tis":
        if args.kl_beta != 0:
            raise ValueError("Token-TIS qualification requires kl_beta=0")
        if not math.isfinite(args.rollout_is_cap) or args.rollout_is_cap <= 0:
            raise ValueError("rollout importance cap must be finite and positive")
    return mode

def initial_admission(maximum, bound, mode):
    if not math.isfinite(maximum) or maximum < 0 or not math.isfinite(bound) or bound <= 0:
        raise ValueError("initial admission requires finite measured probabilities and bound")
    within_bound = maximum <= bound
    if mode == "none" and not within_bound:
        raise ValueError("initial learner/actual-vLLM action logprob difference exceeds explicit admission bound")
    if mode not in {"none", "decoupled-token-tis"}:
        raise ValueError("unknown rollout correction mode")
    return {"within_strict_admission_bound": within_bound, "rollout_correction": mode,
        "admission_basis": "strict numerical bound" if mode == "none" else
            "explicit decoupled Token-TIS; finite aligned probabilities; numerical parity is not assumed"}

def aggregate_token_tis_metrics(rows):
    # Weight diagnostics pool actual tokens. The objective still averages response
    # token means equally, then responses equally; ESS does not count IID samples.
    tokens = sum(row["token_count"] for row in rows)
    total = sum(row["is_weight_sum"] for row in rows)
    squared = sum(row["is_weight_square_sum"] for row in rows)
    raw = sum(row["is_raw_weight_sum"] for row in rows)
    clipped = sum(row["is_clipped_tokens"] for row in rows)
    dropped = sum(row["is_dropped_weight_mass"] for row in rows)
    return {"loss": sum(row["loss"] for row in rows) / len(rows),
        "policy_loss": sum(row["policy_loss"] for row in rows) / len(rows),
        "response_count": len(rows), "token_count": tokens, "is_cap": rows[0]["is_cap"],
        "is_weight_sum": total, "is_weight_square_sum": squared, "is_raw_weight_sum": raw,
        "is_weight_mean": total / tokens,
        "is_weight_min": min(row["is_weight_min"] for row in rows),
        "is_weight_max": max(row["is_weight_max"] for row in rows),
        "is_raw_weight_min": min(row["is_raw_weight_min"] for row in rows),
        "is_raw_weight_max": max(row["is_raw_weight_max"] for row in rows),
        "is_clipped_tokens": clipped, "is_clipped_fraction": clipped / tokens,
        "is_dropped_weight_mass": dropped, "is_dropped_weight_mass_fraction": dropped / raw,
        "is_token_weight_ess": total * total / squared,
        "is_token_weight_ess_fraction": total * total / squared / tokens,
        "native_old_minus_behavior_mean": sum(row["native_old_minus_behavior_mean"] * row["token_count"] for row in rows) / tokens,
        "native_old_minus_behavior_max_abs": max(row["native_old_minus_behavior_max_abs"] for row in rows),
        "native_ppo_ratio_mean": sum(row["native_ppo_ratio_mean"] * row["token_count"] for row in rows) / tokens,
        "native_ppo_clip_fraction": sum(row["native_ppo_clip_fraction"] * row["token_count"] for row in rows) / tokens}

def train_group(model, optimizer, gradient_policy, results, advantages, args, update, *,
                expected_ack=None, native_old_snapshot=None):
    """Accumulate response token means / 32, then apply one adapter update."""
    correction = validate_correction_configuration(args)
    microbatch_size = getattr(args, "learner_microbatch_size", 1)
    if type(microbatch_size) is not int or not 1 <= microbatch_size <= 32:
        raise ValueError("learner microbatch size must be an integer from 1 to 32")
    if (len(results) != 32 or len(advantages) != 32
            or not torch.isfinite(torch.as_tensor(advantages)).all()):
        raise ValueError("a logical learner update requires all 32 responses and finite advantages")
    generating_ack, policy_lag = expected_ack, None
    if correction == "decoupled-token-tis":
        generating_ack = (native_old_snapshot["ack"] if native_old_snapshot is not None else expected_ack)
        if (expected_ack is None or expected_ack.get("version") != update - 1
                or generating_ack is None
                or type(generating_ack.get("version")) is not int
                or any(result.get("generation_metrics", {}).get("policy_ack") != generating_ack
                       for result in results)):
            raise ValueError("Token-TIS native-old anchor must match every response's generating policy")
        policy_lag = expected_ack["version"] - generating_ack["version"]
        if (not 0 <= policy_lag <= (0 if getattr(args, "disable_prefetch", False) else 1)
                or generating_ack.get("lineage") != expected_ack.get("lineage")
                or (policy_lag == 0 and generating_ack != expected_ack)):
            raise ValueError("Token-TIS generating policy has an invalid lineage or policy lag")
    elif expected_ack is not None:
        generating_ack = results[0].get("generation_metrics", {}).get("policy_ack")
        if (expected_ack.get("version") != update - 1 or generating_ack is None
                or type(generating_ack.get("version")) is not int
                or generating_ack.get("lineage") != expected_ack.get("lineage")
                or any(result.get("generation_metrics", {}).get("policy_ack") != generating_ack
                       for result in results)):
            raise ValueError("strict policy lag requires consistent learner and generating ACKs")
        policy_lag = expected_ack["version"] - generating_ack["version"]
        if (not 0 <= policy_lag <= (0 if getattr(args, "disable_prefetch", False) else 1)
                or (policy_lag == 0 and generating_ack != expected_ack)):
            raise ValueError("strict generating policy has an invalid policy lag or ACK")
    shapes = parse_learner_batch_shapes(getattr(args, "learner_batch_shapes", None))
    input_lengths = [len(result["context_token_ids"]) + len(result["action_token_ids"]) - 1
                     for result in results]
    plan = plan_learner_microbatches(input_lengths, microbatch_size, shapes)
    before = adapter_snapshot(optimizer)
    optimizer.zero_grad(set_to_none=True)
    gradient_policy.begin_step()
    metrics, physical_batch_sizes = [None] * 32, []
    valid_tokens = padded_tokens = 0
    forward_counts = {"native_old": 0, "current": 0, "reference": 0}
    for planned in plan:
        indices = planned["response_indices"]
        batch = [results[index] for index in indices]
        prompts = [result["context_token_ids"] for result in batch]
        responses = [result["action_token_ids"] for result in batch]
        lengths = [len(prompt) + len(response) - 1 for prompt, response in zip(prompts, responses)]

        def score(phase):
            if len(batch) == 1:
                values = [response_log_probs(model, prompts[0], responses[0], args.temperature)]
                layout = {"input_shape": [1, lengths[0]], "valid_input_tokens": lengths[0],
                    "padded_input_tokens": 0, "response_tokens": len(responses[0]),
                    "model_forward_calls": 1, "padding_side": "right"}
            else:
                values, layout = response_log_probs_batch(model, prompts, responses, args.temperature,
                                                          return_metadata=True)
            if (len(values) != len(batch) or layout.get("input_shape") != [len(batch), max(lengths)]
                    or layout.get("valid_input_tokens") != sum(lengths)
                    or layout.get("padded_input_tokens") != len(batch) * max(lengths) - sum(lengths)
                    or layout.get("response_tokens") != sum(map(len, responses))
                    or layout.get("model_forward_calls") != 1 or layout.get("padding_side") != "right"
                    or any(value.shape != (len(response),) for value, response in zip(values, responses))):
                raise ValueError("learner scorer did not return one physical aligned response microbatch")
            forward_counts[phase] += 1
            return values

        reference, native_old = None, None
        if correction == "decoupled-token-tis":
            if policy_lag:
                with adapter_parameters_at(model, optimizer, native_old_snapshot["parameters"]):
                    native_old = [value.detach().clone() for value in score("native_old")]
            else:
                with torch.no_grad():
                    native_old = [value.detach().clone() for value in score("native_old")]
        if args.kl_beta:
            with reference_base(model):
                reference = score("reference")
        current = score("current")
        losses = []
        for offset, index in enumerate(indices):
            behavior, advantage = batch[offset]["behavior_log_probs"], advantages[index]
            if correction == "decoupled-token-tis":
                loss, row = decoupled_token_tis_loss([current[offset]], [native_old[offset]],
                    [behavior], [advantage], clip_eps=args.clip_eps, is_cap=args.rollout_is_cap)
            else:
                loss, row = grpo_loss([current[offset]], [behavior], [advantage],
                    clip_eps=args.clip_eps, kl_beta=args.kl_beta,
                    ref_log_probs=[reference[offset]] if reference is not None else None)
            losses.append(loss)
            metrics[index] = row
        # Length sorting and ragged batches must not alter response weights.
        (torch.stack(losses).sum() / 32).backward()
        physical_batch_sizes.append(len(batch))
        valid_tokens += sum(lengths)
        padded_tokens += len(batch) * max(lengths) - sum(lengths)
    gradient_policy.normalize()
    norm = gradient_norm(optimizer)
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("a genuine optimizer update requires finite nonzero adapter gradients")
    optimizer.step()
    changed = verify_optimizer_step(model, optimizer, before)
    pooled = (aggregate_token_tis_metrics(metrics) if correction == "decoupled-token-tis" else
              {key: sum(row[key] for row in metrics) / 32 for key in metrics[0]})
    return pooled | {"rollout_correction": correction,
        "native_old_policy_ack": generating_ack if correction == "decoupled-token-tis" else None,
        "learner_policy_ack": expected_ack, "policy_lag": policy_lag,
        "learner_batching": {"physical_batch_sizes": physical_batch_sizes,
            "padding_fraction": padded_tokens / (valid_tokens + padded_tokens),
            "model_forward_calls": forward_counts, "backward_calls": len(physical_batch_sizes)},
        "gradient_norm": norm, "changed_adapter_elements": changed}


def run(args):
    """Train query adapters, synchronize every update, and compare held-out scores."""
    parse_learner_batch_shapes(args.learner_batch_shapes)
    layers = getattr(args, "lora_layers", None)
    if isinstance(layers, str):
        layers = parse_lora_layers(layers)
    lora_config = DistributedLoRAConfig(rank=getattr(args, "lora_rank", 8),
        alpha=getattr(args, "lora_alpha", 8.0), layers=layers)
    # This deployment loads all 40 layers; reject invalid selection before CUDA,
    # distributed initialization, or a remote rollout process can be started.
    lora_config.selected_layers(40)
    correction = validate_correction_configuration(args)
    if (not 1 <= args.updates <= args.max_groups <= 7473 or args.group_size != 32
            or not 1 <= args.eval_questions <= 1319
            or not 1 <= args.max_prompt_tokens or not 1 <= args.max_new_tokens
            or not 2 <= args.max_model_len <= 2048
            or args.max_prompt_tokens + args.max_new_tokens > args.max_model_len
            or not 0 < args.max_initial_logprob_delta < float("inf")
            or not 0 < args.learning_rate < float("inf") or not 0 <= args.kl_beta < float("inf")
            or not 0 < args.temperature <= 10 or not 0 < args.eval_temperature <= 10
            or not 0 <= args.clip_eps < 1 or not 1 <= args.num_threads <= 64
            or not 1 <= args.learner_microbatch_size <= 32
            or not 0 <= args.seed < 2**63 - 1 - max(args.max_groups, args.eval_questions) * 32):
        raise ValueError("invalid GRPO batch, context, evaluation, or optimizer configuration")
    torch.set_num_threads(args.num_threads)
    token = os.environ["DSV41_ROLLOUT_TOKEN"]
    rank = int(os.environ.get("RANK", "0"))
    output = Path(args.output_dir).resolve()
    started = time.monotonic()
    summary = {"status": "started", "optimizer_updates": 0, "acknowledged_updates": 0,
               "adapter_configuration": lora_config.to_dict()}
    client = model = executor = None
    output_created = False
    try:
        tokenizer = ModelTokenizer(args.model_dir)
        train, test, reward, data = prepare_data(args.train_jsonl, args.test_jsonl, tokenizer,
            revision=args.dataset_revision, seed=args.seed, max_prompt_tokens=args.max_prompt_tokens,
            max_train_groups=args.max_groups, eval_questions=args.eval_questions)
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl", timeout=timedelta(seconds=600),
            device_id=torch.device("cuda", int(os.environ["LOCAL_RANK"])))
        rank_zero_call(lambda: output.mkdir(mode=0o700, parents=True, exist_ok=False))
        output_created = rank == 0
        identity = {key: hashlib.sha256((Path(args.model_dir) / name).read_bytes()).hexdigest()
            for key, name in (("config_sha256", "config.json"), ("index_sha256", "model.safetensors.index.json"))}
        identities = [None] * dist.get_world_size()
        dist.all_gather_object(identities, {"base": identity, "data": data,
            "torch": torch.__version__, "configuration": semantic_configuration(args),
            "lora_config": lora_config.to_dict()})
        if any(item != identities[0] for item in identities):
            raise ValueError("learner ranks differ in data, base identity, Torch, or configuration")
        lineage = rank_zero_call(lambda: "gsm8k-grpo-" + uuid.uuid4().hex)
        if rank == 0:
            client = RemoteLoRAClient(args.rollout_host, args.rollout_port, token, timeout=7200)
            executor = ThreadPoolExecutor(max_workers=1)
        startup = rank_zero_call(lambda: client.initialize(lora_config=lora_config.to_dict()))
        if startup["base_identity"] != identity:
            raise ValueError("rollout and learner mounted metadata identities differ")
        if DistributedLoRAConfig.from_dict(startup["lora_config"]) != lora_config:
            raise ValueError("rollout and learner LoRA configurations differ")
        model, optimizer, gradients, runtime, native = load_distributed_learner(
            args.model_dir, args.native_source, seed=args.seed, lora_config=lora_config)
        sources = [None] * dist.get_world_size()
        dist.all_gather_object(sources, native)
        if any(item != sources[0] for item in sources):
            raise ValueError("learner ranks differ in native source")
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate
        summary["trainable_parameter_count"] = lora_config.parameter_count(
            model.config.hidden_size, model.config.q_lora_rank, model.config.num_hidden_layers)

        def activate(version):
            packet = export_q_a_update(model, version=version, lineage=lineage,
                adapter_id=version + 1, base_identity=identity, optimizer=optimizer)
            def send():
                receipt = client.activate(*packet)
                if receipt["ack"] != policy_ack(packet[0]):
                    raise ValueError("worker activation acknowledgement differs from learner packet")
                return receipt["ack"]
            ack = rank_zero_call(send)
            rank_zero_call(lambda: client.resume())
            return ack

        learner_ack = activate(0)
        policy_snapshots = {}
        if correction == "decoupled-token-tis":
            policy_snapshots[0] = {"ack": dict(learner_ack), "parameters": adapter_snapshot(optimizer)}
        validation = rank_zero_call(lambda: sample_record(client, train[0], count=1,
            phase="initial-validation", index=0, args=args, lineage=lineage))
        with torch.no_grad():
            current = response_log_probs(model, train[0].prompt_ids,
                validation[0]["action_token_ids"], args.temperature)
            behavior = current.new_tensor(validation[0]["behavior_log_probs"])
            maximum = (current - behavior).abs().max()
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        summary["initial_logprob_comparison"] = {"max_abs": maximum.item(),
            **initial_admission(maximum.item(), args.max_initial_logprob_delta, correction)}
        summary["baseline"] = rank_zero_call(lambda: evaluate(client, test, reward, tokenizer,
            args, lineage, "baseline"))
        pending = None
        for group_index, record in enumerate(train):
            def obtain_group():
                return pending.result() if pending is not None else sample_record(client, record,
                    count=32, phase="train", index=group_index, args=args, lineage=lineage)
            results = rank_zero_call(obtain_group)
            rewards, _ = score_rewards(results, record, reward, tokenizer)
            advantages = group_advantages(rewards)
            usable = bool(advantages.abs().sum())
            group_row = {"group": group_index, "question_id": record.metadata["id"],
                "behavior_version": results[0]["generation_metrics"]["policy_ack"]["version"],
                "correct": int(sum(rewards)), "responses": 32, "usable": usable}
            rank_zero_call(lambda: append_json(output / "groups.jsonl", group_row))
            if not usable:
                pending = None
                continue
            update = summary["optimizer_updates"] + 1
            pending = None
            if (not args.disable_prefetch and rank == 0 and update > 1
                    and group_index + 1 < len(train) and update < args.updates):
                pending = executor.submit(sample_record, client, train[group_index + 1], count=32,
                    phase="train", index=group_index + 1, args=args, lineage=lineage)
            metrics = train_group(model, optimizer, gradients, results, advantages, args, update,
                expected_ack=learner_ack,
                native_old_snapshot=policy_snapshots.get(group_row["behavior_version"]))
            summary["optimizer_updates"] = update
            def join_prefetch():
                if pending is not None:
                    pending.result()
            rank_zero_call(join_prefetch)
            rank_zero_call(lambda: client.pause())
            learner_ack = activate(update)
            if correction == "decoupled-token-tis":
                policy_snapshots[update] = {"ack": dict(learner_ack), "parameters": adapter_snapshot(optimizer)}
                policy_snapshots = {version: state for version, state in policy_snapshots.items()
                                    if version >= update - 1}
            proof = rank_zero_call(lambda: sample_record(client, record, count=1,
                phase="activation-check", index=update, args=args, lineage=lineage))
            if proof[0]["generation_metrics"]["policy_ack"] != learner_ack:
                raise ValueError("subsequent rollout did not use the acknowledged adapter")
            summary["acknowledged_updates"] = update
            rank_zero_call(lambda: append_json(output / "updates.jsonl",
                {"update": update, "group": group_index, "policy_ack": learner_ack, **metrics}))
            if update >= args.updates:
                break
        if summary["optimizer_updates"] < args.updates:
            raise RuntimeError("available nonconstant GSM8K groups did not complete the requested updates")
        summary["final"] = rank_zero_call(lambda: evaluate(client, test, reward, tokenizer,
            args, lineage, "final"))
        if summary["baseline"]["ordered_ids_sha256"] != summary["final"]["ordered_ids_sha256"]:
            raise ValueError("baseline/final held-out subsets differ")
        summary.update(status="completed", policy_ack=learner_ack)
    except BaseException as error:
        summary.update(status="failed", error_type=type(error).__name__, reason=str(error))
        raise
    finally:
        cleanup_errors = []
        operations = [lambda: executor.shutdown(wait=True, cancel_futures=True)] if executor else []
        if client is not None:
            operations.append(client.close)
        if model is not None:
            operations.append(model.close)
        if dist.is_initialized():
            operations.append(dist.destroy_process_group)
        for close in operations:
            try:
                close()
            except Exception as error:
                cleanup_errors.append(error)
        failed = summary["status"] == "failed"
        if cleanup_errors:
            summary.update(status="failed", cleanup_errors=[
                {"error_type": type(error).__name__, "reason": str(error)} for error in cleanup_errors])
        summary["elapsed_seconds"] = time.monotonic() - started
        if output_created:
            write_json(output / "summary.json", summary)
        if cleanup_errors and not failed:
            raise cleanup_errors[0]
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model-dir", "train-jsonl", "test-jsonl", "output-dir", "rollout-host"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--dataset-revision", default=DATASET_REVISION)
    parser.add_argument("--native-source", default=str(default_native_source()),
                        help="manifest-verified native compatibility runtime")
    parser.add_argument("--rollout-port", type=int, required=True)
    parser.add_argument("--max-initial-logprob-delta", type=float, required=True)
    parser.add_argument("--rollout-correction", choices=("none", "decoupled-token-tis"), default="none")
    parser.add_argument("--rollout-is-cap", type=float, default=2.)
    parser.add_argument("--disable-prefetch", action="store_true",
                        help="sample each group after the preceding learner update and activation")
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--max-groups", type=int, default=64)
    parser.add_argument("--eval-questions", type=int, default=128)
    parser.add_argument("--group-size", type=int, choices=(32,), default=32)
    parser.add_argument("--learner-microbatch-size", type=int, choices=range(1, 33), default=4)
    parser.add_argument("--learner-batch-shapes", default=None,
                        help="explicit capacity-qualified B:T cells, e.g. 4:2047,8:1023; no OOM fallback")
    parser.add_argument("--lora-rank", type=int, choices=SUPPORTED_LORA_RANKS, default=8)
    parser.add_argument("--lora-alpha", type=float, default=8.0)
    parser.add_argument("--lora-layers", type=parse_lora_layers, default=None, metavar="all|CSV",
                        help="zero-based query projection layer indices; default: all layers")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-prompt-tokens", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=.8)
    parser.add_argument("--eval-temperature", type=float, default=.01)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--clip-eps", type=float, default=.2)
    parser.add_argument("--kl-beta", type=float, default=0.)
    parser.add_argument("--num-threads", type=int, default=2)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
