from __future__ import annotations

import fcntl
import json
import os
import queue
import time
import uuid
from dataclasses import replace
from pathlib import Path

import torch

from .algorithm import group_advantages, grpo_loss, response_log_probs
from .checkpoint import (load_training_checkpoint, publish_policy_artifact,
                         run_identity, save_training_checkpoint, load_policy_artifact, validate_identity,
                         add_lora_identity)
from .config import RLConfig
from .data import build_reward, load_jsonl
from .evaluation import append_jsonl, evaluate_policy, sample_record
from .family import build_policy, load_model_config, policy_lora_config
from .parallel import layer_device_map, validate_available_devices
from .rollout import generation_options
from .tokenizer import ModelTokenizer
from .trajectory import trajectory_log_probs


def make_model(model_dir, model_config, config, devices, random_init, *, engram_weights=None):
    qwen = getattr(model_config, "model_type", None) == "qwen3_moe"
    last_position = getattr(config, "qwen_last_position_only", False)
    if config.qwen_kv_cache and not qwen:
        raise ValueError("request KV-cache sampling requires native Qwen")
    if (last_position or getattr(config, "backprop_chunk_tokens", 0)) and not qwen:
        raise ValueError("last-position logits and chunked backward currently require native Qwen")
    torch.manual_seed(config.seed)
    model = build_policy(model_dir, model_config, random_init=random_init,
                              devices=devices, dtype=getattr(torch, config.dtype),
                              engram_weights=engram_weights, engram_offload=config.engram_offload,
                              trainable_scope=config.trainable_scope,
                              lora_config=policy_lora_config(config, model_config),
                              initial_adapter=config.initial_adapter)
    if qwen:
        model.qwen_last_position_only = last_position
        model.qwen_kv_cache = config.qwen_kv_cache
    if config.gradient_checkpointing:
        model.backend.gradient_checkpointing_enable()
    return model


def policy_lag(learner_version: int, rollout_version: int, maximum: int) -> int | None:
    lag = learner_version - rollout_version
    if lag < 0:
        raise ValueError("rollout policy version is newer than learner")
    return lag if lag <= maximum else None


def validate_policy_ack(acknowledgement, artifact):
    expected = {key: artifact[key] for key in ("version", "digest", "lineage")}
    if (not isinstance(acknowledgement, dict) or type(acknowledgement.get("version")) is not int or
            acknowledgement != expected):
        raise ValueError("rollout provider did not acknowledge the exact published policy")


def validate_device_ownership(learner_devices, rollout_devices):
    def cuda_indices(devices):
        return {device.index if device.index is not None else torch.cuda.current_device()
                for device in map(torch.device, devices) if device.type == "cuda"}
    if cuda_indices(learner_devices).intersection(cuda_indices(rollout_devices)):
        raise ValueError("learner and rollout CUDA devices must have separate ownership")


def _training_chunks(model, reference, sample, config):
    """Yield the unchanged per-trajectory objective with bounded replay graphs."""
    from rollout_manager.training import TrainingTrajectory

    if isinstance(sample, TrainingTrajectory):
        if config.temperature != sample.policy.sampling["temperature"]:
            raise ValueError("learner temperature differs from recorded behavior sampling")
        turns = [(list(turn.context_token_ids), list(turn.action_token_ids), list(turn.behavior_log_probs))
                 for turn in sample.turns]
    else:
        turns = [(sample.prompt_ids, sample.response_ids, sample.behavior_log_probs)]
    width = getattr(config, "backprop_chunk_tokens", 0)
    if not width:
        current = trajectory_log_probs(model, sample, config.temperature)
        with torch.no_grad():
            ref = trajectory_log_probs(reference, sample, config.temperature) if reference is not None else None
        yield current, sample.behavior_log_probs, ref, 1.
        return
    total_actions = sum(len(actions) for _, actions, _ in turns)
    for prompt, actions, behavior in turns:
        for start in range(0, len(actions), width):
            stop = min(start + width, len(actions))
            prefix, targets = prompt + actions[:start], actions[start:stop]
            current = response_log_probs(model, prefix, targets, config.temperature)
            with torch.no_grad():
                ref = response_log_probs(reference, prefix, targets, config.temperature) if reference is not None else None
            yield current, behavior[start:stop], ref, len(targets) / total_actions


def _exact_zero_objective(model, groups, config):
    """A constant GRPO objective needs zero gradients, not decoder replay.

    Restrict this optimization to the fully observed, sole trainable language
    head. Supplying a zero gradient preserves the optimizer's momentum and step
    semantics; omitting gradients or skipping optimizer.step would not.
    """
    from rollout_manager.training import TrainingTrajectory
    parameters = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
    return bool(groups and config.kl_beta == 0 and config.weight_decay == 0
                and config.trainable_scope == "lm_head" and len(parameters) == 1
                and parameters[0][0].split(".")[-2:] == ["lm_head", "weight"]
                and all(group.samples and all(isinstance(sample, TrainingTrajectory)
                    and sample.reward_eligible for sample in group.samples)
                    and not bool(group_advantages([sample.reward for sample in group.samples]).count_nonzero())
                    for group in groups))


def _validate_zero_objective_sample(model, sample, config):
    from rollout_manager.training import TrainingTrajectory
    # Revalidate every serialized token/probability/turn-order field. No current
    # probability, importance ratio or backward timing is claimed here.
    checked = TrainingTrajectory.from_dict(sample.to_dict())
    if checked.policy.sampling["temperature"] != config.temperature:
        raise ValueError("learner temperature differs from recorded behavior sampling")
    for turn in checked.turns:
        if (len(turn.context_token_ids) + len(turn.action_token_ids) > model.config.max_position_embeddings
                or any(token >= model.config.vocab_size
                       for token in (*turn.context_token_ids, *turn.action_token_ids))):
            raise ValueError("zero-objective trajectory exceeds mounted vocabulary/context")


def train_groups(model, reference, optimizer, groups, config):
    from rollout_manager.training import TrainingTrajectory
    from .qwen_policy import LAST_POSITION_LOGITS_PATH

    qwen = getattr(getattr(model, "config", None), "model_type", None) == "qwen3_moe"
    last_position = getattr(config, "qwen_last_position_only", False)
    if (last_position or getattr(config, "backprop_chunk_tokens", 0)) and not qwen:
        raise ValueError("last-position logits and chunked backward currently require native Qwen")
    if qwen and getattr(model, "qwen_last_position_only", False) != last_position:
        raise ValueError("learner Qwen logits profile differs from its configuration")
    if qwen and getattr(model, "qwen_kv_cache", False) != config.qwen_kv_cache:
        raise ValueError("learner Qwen sampling profile differs from its configuration")
    if reference is not None and getattr(reference, "qwen_last_position_only", False) != last_position:
        raise ValueError("reference Qwen logits profile differs from its configuration")
    if reference is not None and getattr(reference, "qwen_kv_cache", False) != config.qwen_kv_cache:
        raise ValueError("reference Qwen sampling profile differs from its configuration")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_samples = sum(len(group.samples) for group in groups)
    skip_replay = _exact_zero_objective(model, groups, config)
    if skip_replay and any(group.get("weight_decay", 0) != 0 for group in optimizer.param_groups):
        raise ValueError("zero-objective optimization requires zero actual optimizer weight decay")
    metrics = {}
    rewards = []
    nonzero_advantages = 0
    zero_variance_groups = 0
    truncated = 0
    for group in groups:
        advantages = group_advantages([sample.reward for sample in group.samples])
        nonzero_advantages += int(advantages.count_nonzero())
        zero_variance_groups += int(not bool(advantages.count_nonzero()))
        for sample, advantage in zip(group.samples, advantages):
            if qwen:
                expected = LAST_POSITION_LOGITS_PATH if last_position else None
                if (last_position and not isinstance(sample, TrainingTrajectory) or
                        isinstance(sample, TrainingTrajectory) and sample.policy.identities.get("qwen_logits_path") != expected):
                    raise ValueError("Qwen training receipt differs from the selected logits profile")
                if isinstance(sample, TrainingTrajectory):
                    from rollout_manager.backends.qwen_kv_cache import QWEN_KV_SAMPLING_PATH
                    expected_sampling = QWEN_KV_SAMPLING_PATH if config.qwen_kv_cache else None
                    if getattr(model, "behavior_engine", "native-pytorch") == "vllm":
                        from rollout_manager.backends.vllm_profile import validate_vllm_policy_profile
                        validate_vllm_policy_profile(sample.policy, config, model.config.max_position_embeddings)
                    elif (sample.policy.identities.get("rollout_engine") == "vllm" or
                            sample.policy.identities.get("qwen_sampling_path") != expected_sampling):
                        raise ValueError("Qwen training receipt differs from the selected sampling profile")
            if skip_replay:
                _validate_zero_objective_sample(model, sample, config)
            for log_probs, behavior, ref, fraction in (() if skip_replay else
                                                       _training_chunks(model, reference, sample, config)):
                loss, values = grpo_loss(
                    [log_probs], [behavior], [float(advantage)], clip_eps=config.clip_eps,
                    ref_log_probs=[ref] if ref is not None else None, kl_beta=config.kl_beta,
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite RL loss; optimizer step was not applied")
                scale = fraction / total_samples
                ((loss * fraction) / total_samples).backward()
                for name, value in values.items():
                    metrics[name] = metrics.get(name, 0.0) + value * scale
            rewards.append(sample.reward)
            truncated += int(sample.finish_reason == "length")
    parameters = [p for p in model.parameters() if p.requires_grad]
    if skip_replay:
        for parameter in parameters:
            if not bool(torch.isfinite(parameter).all()):
                raise FloatingPointError("zero-objective head contains non-finite parameters")
            parameter.grad = torch.zeros_like(parameter, memory_format=torch.preserve_format)
        metrics.update(loss=0.0, policy_loss=0.0, kl=0.0,
                       replay_skipped=True, replay_skip_reason="exact_zero_advantages_kl0_weight_decay0",
                       sampled_action_tokens=sum(len(turn.action_token_ids) for group in groups
                           for sample in group.samples for turn in sample.turns),
                       replayed_action_tokens=0, backward_calls=0,
                       replay_diagnostics="not_measured_exact_zero_objective")
    norm = torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm, error_if_nonfinite=True,
                                         foreach=False)
    optimizer.step()
    model.assert_engram_invariants()
    metrics.update(reward_mean=sum(rewards) / len(rewards), grad_norm=float(norm),
                   nonzero_advantages=nonzero_advantages, samples=total_samples,
                   zero_variance_group_fraction=zero_variance_groups / len(groups),
                   truncation_fraction=truncated / total_samples)
    metrics["optimizer_step_applied"] = True
    return metrics


def _check_inputs(records, model_config, config):
    for record in records:
        if len(record.prompt_ids) + config.max_new_tokens > model_config.max_position_embeddings:
            raise ValueError("prompt length + max_new_tokens exceeds model context length")
        if any(token < 0 or token >= model_config.vocab_size
               for token in list(record.prompt_ids) + list(record.target_ids or [])):
            raise ValueError("dataset token IDs must be inside the model vocabulary")


def prepare_records(data_path, model_config, config, tokenizer):
    limit = model_config.max_position_embeddings - config.max_new_tokens
    if config.max_prompt_tokens is not None:
        limit = min(limit, config.max_prompt_tokens)
    if limit < 1:
        raise ValueError("max_new_tokens must leave room for a prompt within the model context")
    records = load_jsonl(data_path, tokenizer=tokenizer, max_prompt_tokens=limit,
                         length_policy=config.length_policy)
    _check_inputs(records, model_config, config)
    return records


def validate_reward_records(records, reward, spec):
    from .rewards import TEXT_REWARDS

    for index, record in enumerate(records):
        if spec in {"token_accuracy", "exact_match"} and record.target_ids is None:
            raise ValueError(f"record {index}: reward {spec} requires target_ids")
        if spec in TEXT_REWARDS:
            try:
                reward(record, [])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"record {index}: {exc}") from exc


def run_training(*, model_dir, data_path, config: RLConfig, random_init=False, resume=None,
                 eval_data_path=None, rollout_factory=None, rollout_identity=None,
                 initial_policy=None, initial_policy_consumer=None):
    config.validate()
    if initial_policy is not None and (resume is not None or random_init or rollout_factory is None):
        raise ValueError("original initial policy recovery requires a pretrained external provider and no checkpoint resume")
    if initial_policy_consumer is not None and initial_policy is None:
        raise ValueError("an original group consumer requires its original initial policy")
    if config.checkpoint_interval != 1:
        raise ValueError("manager training requires checkpoint_interval=1 before policy activation")
    torch.set_num_threads(config.num_threads)
    model_dir, data_path = str(Path(model_dir).resolve()), str(Path(data_path).resolve())
    model_config = load_model_config(model_dir)
    generic_vllm = rollout_factory is None and config.rollout_engine == "vllm"
    if generic_vllm:
        from rollout_manager.backends.vllm_rl import require_vllm_rollout
        require_vllm_rollout(model_config, config, random_init=random_init)
    for devices in (config.learner_devices, config.rollout_devices):
        layer_device_map(model_config.num_hidden_layers, devices)
        validate_available_devices(devices)
    validate_device_ownership(config.learner_devices, config.rollout_devices)
    if generic_vllm:
        from rollout_manager.backends.vllm_rl import validate_vllm_device_ownership
        validate_vllm_device_ownership(config)
    tokenizer = ModelTokenizer(model_dir) if (Path(model_dir) / "tokenizer.json").is_file() else None
    records = prepare_records(data_path, model_config, config, tokenizer)
    eval_records = prepare_records(eval_data_path, model_config, config, tokenizer) if eval_data_path else []
    if {tuple(record.prompt_ids) for record in records}.intersection(
        tuple(record.prompt_ids) for record in eval_records
    ):
        raise ValueError("validation prompts overlap training prompts; use an independent validation set")
    reward_fn = build_reward(config.reward, tokenizer=tokenizer)
    validate_reward_records(records + eval_records, reward_fn, config.reward)
    identity = run_identity(model_dir, data_path, random_init, eval_data_path=eval_data_path)
    add_lora_identity(identity, config, model_config)
    if getattr(model_config, "model_type", None) == "qwen3_moe":
        identity["trainable_scope"] = config.trainable_scope
        if config.qwen_last_position_only:
            from .qwen_policy import LAST_POSITION_LOGITS_PATH
            identity["qwen_logits_path"] = LAST_POSITION_LOGITS_PATH
        if config.qwen_kv_cache:
            from rollout_manager.backends.qwen_kv_cache import QWEN_KV_SAMPLING_PATH
            identity["qwen_sampling_path"] = QWEN_KV_SAMPLING_PATH
    if rollout_factory is not None:
        if not isinstance(rollout_identity, dict) or not rollout_identity:
            raise ValueError("custom rollout provider requires a stable rollout_identity manifest")
        identity["rollout_profile"] = json.loads(json.dumps(rollout_identity, allow_nan=False))
        from rollout_manager.backends.vllm_profile import (
            VLLM_ROLLOUT_PROFILE, vllm_policy_identities, require_vllm_learner)
        if rollout_identity.get("profile") == VLLM_ROLLOUT_PROFILE:
            if getattr(model_config, "model_type", None) != "qwen3_moe":
                raise ValueError("qualified vLLM training currently requires mounted Qwen3 MoE")
            require_vllm_learner(config)
            identity.pop("qwen_sampling_path", None)
            identity.update(vllm_policy_identities(model_config.max_position_embeddings))
    elif rollout_identity is not None:
        raise ValueError("rollout_identity requires a custom rollout provider")
    if generic_vllm:
        from rollout_manager.backends.vllm_profile import vllm_policy_identities
        identity.pop("qwen_sampling_path", None)
        identity.update(vllm_policy_identities(model_config.max_position_embeddings))
        identity["rollout_profile"] = {"profile": "prompt-response-vllm-v1",
                                      "sampling_rng": "vllm_engine_seeded_sampler"}
    elif config.rollout_engine == "vllm" and identity.get("rollout_engine") != "vllm":
        raise ValueError("selected vLLM engine requires a vLLM rollout provider; native fallback is disabled")
    output_dir = Path(config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    lock = output_dir / ".run.lock"
    # Kernel-owned locks are released even after SIGKILL; the file may persist.
    lock_handle = lock.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_handle.close()
        raise RuntimeError(f"another training process owns output_dir: {output_dir}") from None
    manager = None
    from rollout_manager.analysis.stage_timings import StageTimings
    timings = StageTimings(output_dir / "learner-stage-timings.jsonl",
                           clock_domain="learner_process_monotonic")
    try:
        model = timings.call("model_startup", make_model, model_dir, model_config, config,
                             config.learner_devices, random_init,
                             details={"devices": config.learner_devices})
        model.behavior_engine = identity.get("rollout_engine", "native-pytorch")
        engram_weights = model.share_engram_memory()
        reference = None
        if config.kl_beta:
            reference = timings.call("reference_model_startup", make_model, model_dir, model_config,
                                     config, config.learner_devices, random_init, engram_weights=engram_weights)
            reference.eval().requires_grad_(False)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                      lr=config.learning_rate, weight_decay=config.weight_decay,
                                      foreach=False)
        step, next_group_id, rng_state, dropped = 0, 0, None, 0
        recovery_state = None
        lineage = uuid.uuid4().hex
        if initial_policy is not None:
            if initial_policy.get("version") != 0:
                raise ValueError("pre-update recovery requires the exact original policy version zero")
            validate_identity(initial_policy["identity"], identity)
            expected_sampling = {"temperature": config.temperature, "top_p": 1, "top_k": 0}
            if initial_policy["sampling"] != expected_sampling:
                raise ValueError("original initial policy sampling differs")
            timings.call("original_initial_policy_restore", load_policy_artifact, model, initial_policy,
                         details={"policy_version": 0, "policy_digest": initial_policy["digest"]})
            lineage = initial_policy["lineage"]
        if resume:
            state = timings.call("checkpoint_restore", load_training_checkpoint,
                                 resume, model, optimizer, config, identity)
            step, next_group_id = state["step"], state["next_group_id"]
            rng_state, dropped = state["rollout_rng_state"], state["dropped_groups"]
            recovery_state = state.get("manager_state")
            if recovery_state is not None:
                lineage = recovery_state["active_policy"]["lineage"]
            config = replace(config, eos_token_ids=state["config"]["eos_token_ids"])
            del state
        elif any(output_dir.glob("step-*.pt")):
            raise FileExistsError("output_dir already has checkpoints; use --resume or a new output directory")
        generation_kwargs = generation_options(model_dir, model_config, config)
        config = replace(config, eos_token_ids=generation_kwargs["eos_token_id"])
        if step >= config.max_steps:
            raise ValueError("max_steps must exceed the restored checkpoint step")
        if any(int(path.stem.removeprefix("step-")) > step for path in output_dir.glob("step-*.pt")):
            raise FileExistsError("output_dir contains later checkpoints; resume into a new --output-dir")
        if eval_records and not generic_vllm:
            metrics = timings.call("baseline_evaluation", evaluate_policy, model, eval_records, config, reward_fn=reward_fn,
                                      generation_kwargs=generation_kwargs, step=step, tokenizer=tokenizer,
                                      output_dir=output_dir, phase="resume" if resume else "baseline")
            print(json.dumps(metrics, sort_keys=True, allow_nan=False), flush=True)
        from .manager import RolloutManager

        sampling = {"temperature": config.temperature, "top_p": 1, "top_k": 0}
        artifact = dict(initial_policy) if initial_policy is not None else timings.call("initial_policy_publish", publish_policy_artifact,
                                model, step, output_dir / "policies",
                                lineage=lineage, identity=identity, sampling=sampling,
                                details={"policy_version": step})
        with timings.interval("manager_startup", completion="initial_policy_ack_verified", policy_version=step):
            manager = (rollout_factory or RolloutManager)(model_dir, model_config, config, records, random_init,
                                     engram_weights=engram_weights, initial_policy=artifact,
                                     recovery_state=recovery_state, next_group_id=next_group_id,
                                     rng_state=rng_state, event_root=output_dir / f"rollouts-{uuid.uuid4().hex}",
                                     **({"learner_model": model} if rollout_factory is not None and
                                        identity.get("rollout_engine") == "vllm" else {}),
                                     **({"start_paused": bool(eval_records)} if generic_vllm else {}))
            acknowledgement = manager.start()
            validate_policy_ack(acknowledgement, artifact)
        def actor_evaluation(phase):
            result = manager.evaluate(eval_records, step=step, phase=phase)
            append_jsonl(output_dir / "eval-samples.jsonl", result["rows"])
            append_jsonl(output_dir / "eval-metrics.jsonl", [result["metrics"]])
            print(json.dumps(result["metrics"], sort_keys=True, allow_nan=False), flush=True)

        if generic_vllm and eval_records:
            timings.call("baseline_evaluation", actor_evaluation, "resume" if resume else "baseline")
            timings.call("actor_resume_after_baseline", manager.resume)
        print(json.dumps({"event": "started", "learner_pid": os.getpid(), "rollout_pid": manager.pid,
                          "rollout_provider": "rollout_manager", "policy_ack": acknowledgement,
                          "rollout_engine": model.behavior_engine,
                          "engram_device": model.engram_device,
                          "engram_offload": config.engram_offload,
                          "engram_shared": model.engram_shared,
                          "engram_bytes": sum(value.numel() * value.element_size() for value in engram_weights.values()),
                          "trainer_devices": config.learner_devices, "rollout_devices": config.rollout_devices,
                          "policy_version": step,
                          **policy_diagnostics(model)}), flush=True)
        previous_train_interval = None
        while step < config.max_steps:
            groups, packets, lags = [], [], []
            deadline = time.monotonic() + config.worker_timeout_seconds
            with timings.interval("wait_eligible_groups", step=step, required_groups=config.groups_per_step):
                while len(groups) < config.groups_per_step:
                    if time.monotonic() > deadline:
                        raise TimeoutError("timed out waiting for a fresh rollout group")
                    try:
                        packet = manager.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    if packet.get("exclusion") is not None:
                        dropped += 1
                        manager.commit(dropped_group_ids=[packet["group_id"]])
                        append_jsonl(output_dir / "excluded-groups.jsonl", [packet["exclusion"]])
                        continue
                    group = packet.get("training_group", packet["group"])
                    lag = policy_lag(step, group.policy_version, config.max_policy_lag)
                    if lag is None:
                        dropped += 1
                        manager.commit(dropped_group_ids=[group.group_id])
                        continue
                    if len(group.samples) != config.group_size or any(
                        sample.policy_version != group.policy_version for sample in group.samples
                    ):
                        raise ValueError("incomplete group or mixed policy versions")
                    groups.append(group)
                    packets.append(packet)
                    lags.append(lag)
            timings.call("actor_pause_before_training", manager.pause, details={"step": step})
            with timings.interval("train_groups", step=step + 1, completion="gradient_norm_cpu_read_after_optimizer_step"):
                train_start = time.monotonic()
                metrics = train_groups(model, reference, optimizer, groups, config)
                train_end = time.monotonic()
            step += 1
            manager.commit(consumed_group_ids=[group.group_id for group in groups])
            recovery_state = timings.call("actor_pause_after_group_commit", manager.pause, details={"step": step})
            next_group_id = recovery_state["next_group_id"]
            rng_state = recovery_state["rng_state"]
            artifact = timings.call("policy_publish", publish_policy_artifact,
                                    model, step, output_dir / "policies",
                                    lineage=lineage, identity=identity, sampling=sampling,
                                    details={"policy_version": step})
            checkpoint = output_dir / f"step-{step:08d}.pt"
            timings.call("checkpoint_write", save_training_checkpoint,
                checkpoint, model, optimizer, step=step, config=config, identity=identity,
                next_group_id=next_group_id, rollout_rng_state=rng_state, dropped_groups=dropped,
                manager_state=recovery_state,
                policy_publication={**artifact, "state": "staged", "previous_ack": acknowledgement},
                details={"step": step, "checkpoint": str(checkpoint)},
            )
            if initial_policy_consumer is not None:
                initial_policy_consumer.committed(checkpoint, artifact)
            with timings.interval("actor_activation", policy_version=step, completion="published_policy_ack_verified"):
                acknowledgement = manager.activate(artifact)
                validate_policy_ack(acknowledgement, artifact)
            actor_eval_due = generic_vllm and eval_records and (
                step % (config.eval_interval or config.checkpoint_interval) == 0 or step == config.max_steps)
            if not actor_eval_due:
                timings.call("actor_resume", manager.resume, details={"policy_version": step,
                             "includes_any_provider_evaluation_wait": True})
            if config.log_samples_per_step:
                rows = []
                for group in groups:
                    record = records[group.group_id % len(records)]
                    for sample in group.samples:
                        if len(rows) < config.log_samples_per_step:
                            if len(getattr(sample, "turns", ())) > 1:
                                rows.append({"step": step, "group_id": group.group_id,
                                             "trajectory": sample.to_dict()})
                            else:
                                rows.append(sample_record(record, sample, step=step, group_id=group.group_id,
                                                          tokenizer=tokenizer))
                append_jsonl(output_dir / "train-samples.jsonl", rows)
            overlap = 0.0
            if previous_train_interval is not None:
                start, end = previous_train_interval
                overlap = sum(max(0.0, min(p["finished_at"], end) - max(p["started_at"], start))
                              for p in packets)
            metrics.update(event="train", step=step, policy_version=step, policy_lag_max=max(lags),
                           rollout_provider="rollout_manager", policy_digest=artifact["digest"],
                           rollout_engine=model.behavior_engine,
                           policy_ack=acknowledgement,
                           dropped_groups=dropped, next_group_id=next_group_id,
                           rollout_versions=[group.policy_version for group in groups],
                           engram_device=model.engram_device,
                           engram_shared=model.engram_shared,
                           rollout_engram_shared=all(packet["engram_shared"] for packet in packets),
                           train_seconds=train_end - train_start,
                           overlap_previous_train_seconds=overlap,
                           **policy_diagnostics(model))
            line = json.dumps(metrics, sort_keys=True, allow_nan=False)
            print(line, flush=True)
            with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
            previous_train_interval = (train_start, train_end)
            if eval_records and (step % (config.eval_interval or config.checkpoint_interval) == 0
                                 or step == config.max_steps):
                if generic_vllm:
                    timings.call("evaluation", actor_evaluation, "eval")
                    timings.call("actor_resume_after_evaluation", manager.resume, details={"policy_version": step})
                else:
                    metrics = timings.call("evaluation", evaluate_policy, model, eval_records, config, reward_fn=reward_fn,
                                          generation_kwargs=generation_kwargs, step=step, tokenizer=tokenizer,
                                          output_dir=output_dir)
                    print(json.dumps(metrics, sort_keys=True, allow_nan=False), flush=True)
        return output_dir / f"step-{step:08d}.pt"
    finally:
        try:
            if manager is not None:
                timings.call("manager_close", manager.close)
        finally:
            fcntl.flock(lock_handle, fcntl.LOCK_UN)
            lock_handle.close()


def policy_diagnostics(model):
    """Bounded model receipts when a family exposes immutable-base checks."""
    fingerprint = getattr(model, "frozen_parameter_fingerprint", None)
    if fingerprint is None:
        return {}
    devices = sorted({p.device.index for p in model.parameters() if p.device.type == "cuda"})
    return {"trainable_scope": model.trainable_scope,
            "parameter_report": fingerprint(),
            "cuda_peak_bytes": {str(index): {"allocated": torch.cuda.max_memory_allocated(index),
                                             "reserved": torch.cuda.max_memory_reserved(index)}
                                for index in devices}}
