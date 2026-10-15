from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import torch

from .config import RLConfig


def _sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_engram_identity(identity, config):
    if config.train_engram or config.initial_engram_delta is not None:
        identity["engram"] = {"trainable": config.train_engram, "dtype": config.dtype,
            "seed": config.seed, "optimizer": config.optimizer,
            "initial_delta_sha256": file_sha256(config.initial_engram_delta)
                if config.initial_engram_delta is not None else None}
    return identity


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
        _sync_directory(destination.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def policy_state(model) -> dict:
    state = {
        "trainable": model.trainable_state_dict(),
        "frozen": model.frozen_state_dict(include_engram=False),
    }
    if getattr(model, "lora_metadata", None) is not None:
        state.update(adapter_format=1, adapter=model.lora_metadata)
    if getattr(model, "engram_state", None) is not None:
        state["engram_delta"] = model.engram_state.state_dict()
    return state


def load_policy(model, state: dict) -> None:
    engram = getattr(model, "engram_state", None)
    if (engram is not None) != ("engram_delta" in state):
        raise ValueError("policy Engram delta presence differs from the configured baseline")
    if engram is not None:
        engram.validate(state["engram_delta"])
    metadata = getattr(model, "lora_metadata", None)
    if metadata is not None or "adapter_format" in state:
        required = {"trainable", "frozen", "adapter_format", "adapter"}
        if (not required.issubset(state)
                or state.keys() - required - {"version", "lineage", "identity", "sampling", "engram_delta"}
                or state["adapter_format"] != 1 or metadata is None or state["adapter"] != metadata):
            raise ValueError("LoRA policy configuration or initial adapter identity differs")
        if state["frozen"]:
            raise ValueError("LoRA policy artifacts must not include frozen base weights")
    model.load_trainable_state_dict(state["trainable"])
    model.load_frozen_state_dict(state["frozen"], include_engram=False)
    if engram is not None:
        engram.load_state_dict(state["engram_delta"])
    model.assert_engram_invariants()


def publish_policy(model, version: int, path: str | Path) -> None:
    atomic_save({"version": version, **policy_state(model)}, path)


def publish_policy_artifact(model, version: int, directory: str | Path, *,
                            lineage: str, identity: dict, sampling: dict) -> dict:
    """Stage immutable weights; a consumer must explicitly activate this artifact."""
    if type(version) is not int or version < 0 or not isinstance(lineage, str) or not lineage:
        raise ValueError("policy artifact requires a nonnegative version and run lineage")
    if getattr(model, "lora_metadata", None) is not None and identity.get("lora") != model.lora_metadata:
        raise ValueError("policy publication LoRA identity differs from loaded model")
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"version": version, "lineage": lineage, "identity": identity,
               "sampling": sampling, **policy_state(model)}
    with tempfile.TemporaryDirectory(prefix=".policy-stage-", dir=directory) as temporary:
        staged = Path(temporary) / "payload.pt"
        atomic_save(payload, staged)
        digest = file_sha256(staged)
        destination = directory / f"policy-{version:08d}-{digest}.pt"
        try:
            os.link(staged, destination)
        except FileExistsError:
            if file_sha256(destination) != digest:
                raise ValueError("existing policy artifact digest mismatch") from None
        _sync_directory(directory)
    return {"version": version, "digest": digest, "path": str(destination),
            "lineage": lineage, "identity": identity, "sampling": sampling}


def load_policy_artifact(model, artifact: dict) -> dict:
    """Return an activation receipt only after validated weights are loaded."""
    digest = artifact.get("digest")
    if not isinstance(digest, str) or len(digest) != 64 or file_sha256(artifact["path"]) != digest:
        raise ValueError("policy artifact digest mismatch")
    snapshot = torch.load(artifact["path"], map_location="cpu", weights_only=True)
    for key in ("version", "lineage", "identity", "sampling"):
        if snapshot.get(key) != artifact.get(key):
            raise ValueError(f"policy artifact metadata mismatch: {key}")
    load_policy(model, snapshot)
    return {"version": artifact["version"], "digest": digest, "lineage": artifact["lineage"]}


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


def add_lora_identity(identity, config, model_config):
    """Bind the initial SFT adapter and random-base seed to every policy version."""
    if config.trainable_scope == "lora":
        from .family import lora_metadata, policy_lora_config
        identity.update(trainable_scope="lora", initialization_seed=config.seed,
                        lora=lora_metadata(policy_lora_config(config, model_config), config.initial_adapter))
    return identity


def validate_identity(stored, current, *, inference=False):
    ignored = {"data_sha256", "eval_data_sha256"} if inference else set()
    if "identity_version" not in stored:
        # Legacy checkpoints did not fingerprint template/generation metadata.
        ignored |= {"identity_version", "text_assets_sha256"}
    for key in (stored.keys() | current.keys()) - ignored:
        if stored.get(key) != current.get(key):
            raise ValueError(f"checkpoint identity mismatch: {key}")


def checkpoint_config(values, model_config, *, rollout_engine=None):
    values = dict(values)
    # Only legacy checkpoints migrate to native sampling. New API/CLI
    # configurations default to vLLM and never infer a native fallback.
    values.setdefault("rollout_engine", rollout_engine or "native-pytorch")
    if "eos_token_ids" not in values:
        # Old rollout used only model.config.eos_token_id, ignoring generation_config.
        eos = model_config.eos_token_id
        values["eos_token_ids"] = [eos] if isinstance(eos, int) else eos
    return RLConfig.from_dict(values)


def save_training_checkpoint(path, model, optimizer, *, step, config, identity,
                             next_group_id, rollout_rng_state, dropped_groups,
                             manager_state=None, policy_publication=None) -> None:
    if Path(path).exists():
        raise FileExistsError(f"checkpoint already exists: {path}")
    payload = {
        "format_version": 2 if manager_state is not None else 1,
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
    }
    if manager_state is not None:
        if manager_state.get("next_group_id") != next_group_id:
            raise ValueError("manager issued cursor must match checkpoint next_group_id")
        payload.update(manager_state=manager_state, policy_publication=policy_publication)
    atomic_save(payload, path)


def load_training_checkpoint(path, model, optimizer, config, identity) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["format_version"] not in (1, 2):
        raise ValueError("unsupported checkpoint format")
    if state["format_version"] == 2:
        manager = state.get("manager_state")
        if not isinstance(manager, dict) or manager.get("next_group_id") != state["next_group_id"]:
            raise ValueError("checkpoint manager recovery state is invalid")
        publication = state.get("policy_publication")
        if not isinstance(publication, dict) or publication.get("version") != state["step"]:
            raise ValueError("checkpoint policy publication does not match learner step")
        active = manager.get("active_policy", {})
        expected_ack = {key: active.get(key) for key in ("version", "digest", "lineage")}
        if (manager.get("policy_ack") != expected_ack or
                publication.get("previous_ack") != expected_ack or
                publication.get("lineage") != active.get("lineage") or
                publication.get("identity") != state["identity"] or
                publication.get("sampling") != active.get("sampling") or
                publication.get("state") != "staged"):
            raise ValueError("checkpoint policy publication identities or acknowledgement mismatch")
        digest = publication.get("digest")
        if (not isinstance(digest, str) or len(digest) != 64 or
                any(character not in "0123456789abcdef" for character in digest)):
            raise ValueError("checkpoint policy publication digest is invalid")
    validate_identity(state["identity"], identity)
    mutable = {"max_steps", "output_dir", "checkpoint_interval", "worker_timeout_seconds",
               "learner_devices", "rollout_devices", "num_threads", "engram_offload",
               "eval_interval", "log_samples_per_step", "gradient_checkpointing", "engram_cache_mib"}
    current = json.loads(json.dumps(config.to_dict()))
    previous = json.loads(json.dumps(checkpoint_config(state["config"], model.config,
        rollout_engine=state["identity"].get("rollout_engine")).to_dict()))
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
