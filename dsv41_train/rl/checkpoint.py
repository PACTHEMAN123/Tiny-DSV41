from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import torch

from .config import RLConfig


def atomic_save(payload: dict, destination: str | Path) -> None:
    """Readers see a complete old or new file, never a partially written snapshot."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}-", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def policy_state(model) -> dict:
    return {
        "trainable": model.trainable_state_dict(),
        "frozen": model.frozen_state_dict(include_engram=False),
    }


def load_policy(model, state: dict) -> None:
    model.load_trainable_state_dict(state["trainable"])
    model.load_frozen_state_dict(state["frozen"], include_engram=False)
    model.assert_engram_invariants()


def publish_policy(model, version: int, path: str | Path) -> None:
    atomic_save({"version": version, **policy_state(model)}, path)


def run_identity(model_dir: str | Path, data_path: str | Path, random_init: bool,
                 *, eval_data_path=None) -> dict:
    root = Path(model_dir)
    files = []
    if not random_init:
        for path in sorted(root.rglob("*")):
            if path.suffix in (".safetensors", ".bin") or path.name.endswith(".index.json"):
                stat = path.stat()
                files.append([str(path.relative_to(root)), stat.st_size, stat.st_mtime_ns])
    config_bytes = (root / "config.json").read_bytes()
    identity = {
        "identity_version": 2,
        "model_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "data_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
        "base_files": files,
        "random_init": random_init,
    }
    identity["engram_mapping_sha256"] = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ("tokenizer.json", "engram_token_map.pt") if (root / name).is_file()
    }
    assets = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
              for name in ("tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja",
                           "generation_config.json") if (root / name).is_file()}
    if assets:
        identity["text_assets_sha256"] = assets
    if eval_data_path is not None:
        identity["eval_data_sha256"] = hashlib.sha256(Path(eval_data_path).read_bytes()).hexdigest()
    return identity


def validate_identity(stored, current, *, inference=False):
    ignored = {"data_sha256", "eval_data_sha256"} if inference else set()
    if "identity_version" not in stored:
        # Legacy checkpoints did not fingerprint template/generation metadata.
        ignored |= {"identity_version", "text_assets_sha256"}
    for key in (stored.keys() | current.keys()) - ignored:
        if stored.get(key) != current.get(key):
            raise ValueError(f"checkpoint identity mismatch: {key}")


def checkpoint_config(values, model_config):
    values = dict(values)
    if "eos_token_ids" not in values:
        # Old rollout used only model.config.eos_token_id, ignoring generation_config.
        eos = model_config.eos_token_id
        values["eos_token_ids"] = [eos] if isinstance(eos, int) else eos
    return RLConfig.from_dict(values)


def save_training_checkpoint(path, model, optimizer, *, step, config, identity,
                             next_group_id, rollout_rng_state, dropped_groups) -> None:
    if Path(path).exists():
        raise FileExistsError(f"checkpoint already exists: {path}")
    atomic_save({
        "format_version": 1,
        "policy": policy_state(model),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "config": config.to_dict(),
        "identity": identity,
        "next_group_id": next_group_id,
        "rollout_rng_state": rollout_rng_state,
        "learner_rng_state": torch.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "dropped_groups": dropped_groups,
    }, path)


def load_training_checkpoint(path, model, optimizer, config, identity) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["format_version"] != 1:
        raise ValueError("unsupported checkpoint format")
    validate_identity(state["identity"], identity)
    mutable = {"max_steps", "output_dir", "checkpoint_interval", "worker_timeout_seconds",
               "learner_devices", "rollout_devices", "num_threads", "engram_offload",
               "eval_interval", "log_samples_per_step", "gradient_checkpointing"}
    current = json.loads(json.dumps(config.to_dict()))
    previous = json.loads(json.dumps(checkpoint_config(state["config"], model.config).to_dict()))
    if current["eos_token_ids"] is None:
        current["eos_token_ids"] = previous["eos_token_ids"]
    for key in current.keys() - mutable:
        if current[key] != previous[key]:
            raise ValueError(f"resume cannot change algorithm setting {key}")
    state["config"] = previous
    load_policy(model, state["policy"])
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["learner_rng_state"])
    if torch.cuda.is_available() and len(state["cuda_rng_states"]) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(state["cuda_rng_states"])
    return state
