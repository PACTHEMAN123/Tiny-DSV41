from __future__ import annotations

import fcntl
import json
import multiprocessing as mp
import os
import queue
import time
import traceback
from pathlib import Path

import torch

from .algorithm import group_advantages, grpo_loss, response_log_probs
from .checkpoint import (load_policy, load_training_checkpoint, publish_policy,
                         run_identity, save_training_checkpoint)
from .config import RLConfig
from .data import build_reward, load_jsonl
from .policy import DSV41Config, build_dsv41_policy
from .parallel import layer_device_map, validate_available_devices
from .rollout import generate_group


def make_model(model_dir, model_config, config, devices, random_init, *, engram_weights=None):
    torch.manual_seed(config.seed)
    return build_dsv41_policy(model_dir, model_config, random_init=random_init,
                              devices=devices, dtype=getattr(torch, config.dtype),
                              engram_weights=engram_weights, engram_offload=config.engram_offload)


def _rollout_worker(model_dir, model_config, config, data_path, random_init,
                    snapshot_path, published_version, output, errors, stop,
                    next_group_id, rng_state, engram_weights):
    # Only plain Python data crosses this queue; CUDA tensors never enter IPC.
    output.cancel_join_thread()
    try:
        torch.set_num_threads(config.num_threads)
        model = make_model(model_dir, model_config, config, config.rollout_devices, random_init,
                           engram_weights=engram_weights)
        model.eval()
        records = load_jsonl(data_path)
        reward_fn = build_reward(config.reward)
        generator = torch.Generator(device="cpu").manual_seed(config.seed + 1)
        if rng_state is not None:
            generator.set_state(torch.tensor(list(rng_state), dtype=torch.uint8))
        version = -1
        while not stop.is_set():
            if published_version.value > version:
                snapshot = torch.load(snapshot_path, map_location="cpu", weights_only=True)
                load_policy(model, snapshot)
                version = snapshot["version"]
                del snapshot
            started_at = time.monotonic()
            group = generate_group(
                model, records[next_group_id % len(records)],
                group_size=config.group_size, max_new_tokens=config.max_new_tokens,
                temperature=config.temperature, eos_token_id=model_config.eos_token_id,
                policy_version=version, group_id=next_group_id,
                generator=generator, reward_fn=reward_fn,
            )
            packet = {
                "group": group,
                "rng_state": bytes(generator.get_state().tolist()),
                "started_at": started_at,
                "finished_at": time.monotonic(),
                "worker_pid": os.getpid(),
                "engram_shared": model.engram_shared,
            }
            while not stop.is_set():
                try:
                    output.put(packet, timeout=0.2)
                    break
                except queue.Full:
                    continue
            next_group_id += 1
    except BaseException:
        errors.put(traceback.format_exc())


def _worker_error(worker, errors) -> None:
    try:
        error = errors.get_nowait()
    except queue.Empty:
        error = None
    if error is not None:
        raise RuntimeError(f"rollout worker failed:\n{error}")
    if not worker.is_alive():
        try:
            error = errors.get(timeout=0.2)
        except queue.Empty:
            error = f"process exited with code {worker.exitcode}"
        raise RuntimeError(f"rollout worker failed:\n{error}")


def policy_lag(learner_version: int, rollout_version: int, maximum: int) -> int | None:
    lag = learner_version - rollout_version
    if lag < 0:
        raise ValueError("rollout policy version is newer than learner")
    return lag if lag <= maximum else None


def train_groups(model, reference, optimizer, groups, config):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_samples = sum(len(group.samples) for group in groups)
    metrics = {}
    rewards = []
    nonzero_advantages = 0
    for group in groups:
        advantages = group_advantages([sample.reward for sample in group.samples])
        nonzero_advantages += int(advantages.count_nonzero())
        for sample, advantage in zip(group.samples, advantages):
            log_probs = response_log_probs(model, sample.prompt_ids, sample.response_ids, config.temperature)
            reference_probs = None
            if reference is not None:
                with torch.no_grad():
                    reference_probs = [response_log_probs(
                        reference, sample.prompt_ids, sample.response_ids, config.temperature)]
            loss, values = grpo_loss(
                [log_probs], [sample.behavior_log_probs], [float(advantage)],
                clip_eps=config.clip_eps, ref_log_probs=reference_probs, kl_beta=config.kl_beta,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite RL loss; optimizer step was not applied")
            (loss / total_samples).backward()
            metrics["loss"] = metrics.get("loss", 0.0) + float(loss.detach()) / total_samples
            for name, value in values.items():
                if name != "loss":
                    metrics[name] = metrics.get(name, 0.0) + value / total_samples
            rewards.append(sample.reward)
    parameters = [p for p in model.parameters() if p.requires_grad]
    norm = torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm, error_if_nonfinite=True,
                                         foreach=False)
    optimizer.step()
    model.assert_engram_invariants()
    metrics.update(reward_mean=sum(rewards) / len(rewards), grad_norm=float(norm),
                   nonzero_advantages=nonzero_advantages, samples=total_samples)
    return metrics


def _check_inputs(records, model_config, config):
    for record in records:
        if len(record.prompt_ids) + config.max_new_tokens > model_config.max_position_embeddings:
            raise ValueError("prompt length + max_new_tokens exceeds model context length")
        if any(token < 0 or token >= model_config.vocab_size
               for token in list(record.prompt_ids) + list(record.target_ids or [])):
            raise ValueError("dataset token IDs must be inside the model vocabulary")


def run_training(*, model_dir, data_path, config: RLConfig, random_init=False, resume=None):
    config.validate()
    torch.set_num_threads(config.num_threads)
    model_dir, data_path = str(Path(model_dir).resolve()), str(Path(data_path).resolve())
    model_config = DSV41Config.from_model_dir(model_dir)
    for devices in (config.learner_devices, config.rollout_devices):
        layer_device_map(model_config.num_hidden_layers, devices)
        validate_available_devices(devices)
    records = load_jsonl(data_path)
    _check_inputs(records, model_config, config)
    build_reward(config.reward)
    identity = run_identity(model_dir, data_path, random_init)
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
    worker = None
    channels = []
    try:
        model = make_model(model_dir, model_config, config, config.learner_devices, random_init)
        engram_weights = model.share_engram_memory()
        reference = None
        if config.kl_beta:
            reference = make_model(model_dir, model_config, config, config.learner_devices, random_init,
                                   engram_weights=engram_weights)
            reference.eval().requires_grad_(False)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                      lr=config.learning_rate, weight_decay=config.weight_decay,
                                      foreach=False)
        step, next_group_id, rng_state, dropped = 0, 0, None, 0
        if resume:
            state = load_training_checkpoint(resume, model, optimizer, config, identity)
            step, next_group_id = state["step"], state["next_group_id"]
            rng_state, dropped = state["rollout_rng_state"], state["dropped_groups"]
            del state
        elif any(output_dir.glob("step-*.pt")):
            raise FileExistsError("output_dir already has checkpoints; use --resume or a new output directory")
        if step >= config.max_steps:
            raise ValueError("max_steps must exceed the restored checkpoint step")
        if any(int(path.stem.removeprefix("step-")) > step for path in output_dir.glob("step-*.pt")):
            raise FileExistsError("output_dir contains later checkpoints; resume into a new --output-dir")
        snapshot_path = output_dir / "policy-latest.pt"
        publish_policy(model, step, snapshot_path)
        context = mp.get_context("spawn")
        output = context.Queue(maxsize=config.queue_capacity)
        errors = context.Queue(maxsize=1)
        channels = [output, errors]
        stop = context.Event()
        published_version = context.Value("q", step)
        worker = context.Process(
            target=_rollout_worker,
            args=(model_dir, model_config, config, data_path, random_init,
                  str(snapshot_path), published_version, output, errors, stop,
                  next_group_id, rng_state, engram_weights),
            name="dsv41-rollout",
        )
        worker.start()
        print(json.dumps({"event": "started", "learner_pid": os.getpid(), "rollout_pid": worker.pid,
                          "engram_device": model.engram_device,
                          "engram_offload": config.engram_offload,
                          "engram_shared": model.engram_shared,
                          "engram_bytes": sum(value.numel() * value.element_size() for value in engram_weights.values()),
                          "trainer_devices": config.learner_devices, "rollout_devices": config.rollout_devices,
                          "policy_version": step}), flush=True)
        previous_train_interval = None
        while step < config.max_steps:
            groups, packets, lags = [], [], []
            deadline = time.monotonic() + config.worker_timeout_seconds
            while len(groups) < config.groups_per_step:
                _worker_error(worker, errors)
                if time.monotonic() > deadline:
                    raise TimeoutError("timed out waiting for a fresh rollout group")
                try:
                    packet = output.get(timeout=0.2)
                except queue.Empty:
                    continue
                group = packet["group"]
                lag = policy_lag(step, group.policy_version, config.max_policy_lag)
                if lag is None:
                    dropped += 1
                    continue
                if len(group.samples) != config.group_size or any(
                    sample.policy_version != group.policy_version for sample in group.samples
                ):
                    raise ValueError("incomplete group or mixed policy versions")
                groups.append(group)
                packets.append(packet)
                lags.append(lag)
            train_start = time.monotonic()
            metrics = train_groups(model, reference, optimizer, groups, config)
            train_end = time.monotonic()
            step += 1
            publish_policy(model, step, snapshot_path)
            published_version.value = step
            next_group_id = groups[-1].group_id + 1
            rng_state = packets[-1]["rng_state"]
            overlap = 0.0
            if previous_train_interval is not None:
                start, end = previous_train_interval
                overlap = sum(max(0.0, min(p["finished_at"], end) - max(p["started_at"], start))
                              for p in packets)
            metrics.update(event="train", step=step, policy_version=step, policy_lag_max=max(lags),
                           dropped_groups=dropped, next_group_id=next_group_id,
                           rollout_versions=[group.policy_version for group in groups],
                           engram_device=model.engram_device,
                           engram_shared=model.engram_shared,
                           rollout_engram_shared=all(packet["engram_shared"] for packet in packets),
                           train_seconds=train_end - train_start,
                           overlap_previous_train_seconds=overlap)
            line = json.dumps(metrics, sort_keys=True, allow_nan=False)
            print(line, flush=True)
            with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
            previous_train_interval = (train_start, train_end)
            if step % config.checkpoint_interval == 0 or step == config.max_steps:
                save_training_checkpoint(
                    output_dir / f"step-{step:08d}.pt", model, optimizer,
                    step=step, config=config, identity=identity, next_group_id=next_group_id,
                    rollout_rng_state=rng_state, dropped_groups=dropped,
                )
        return output_dir / f"step-{step:08d}.pt"
    finally:
        if worker is not None and worker.pid is not None:
            stop.set()
            worker.join(timeout=5)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5)
            if worker.is_alive():
                worker.kill()
                worker.join()
        for channel in channels:
            channel.close()
            channel.join_thread()
        fcntl.flock(lock_handle, fcntl.LOCK_UN)
        lock_handle.close()
