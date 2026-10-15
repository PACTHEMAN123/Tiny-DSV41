"""Generate token IDs from an RL checkpoint and its original base model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import torch

from .checkpoint import add_engram_identity, add_lora_identity, checkpoint_config, load_policy, run_identity, validate_identity
from .data import PromptRecord
from .evaluation import decoded_fields
from .family import load_model_config
from .rollout import generate_group, generation_options
from .runtime import make_model
from .tokenizer import ModelTokenizer


def load_checkpoint_model(
    *,
    model_dir: str | Path,
    checkpoint: str | Path,
    devices: Sequence[str] = ("cpu",),
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    seed: Optional[int] = None,
) -> tuple:
    """Restore a policy without optimizer or training/validation data.

    The checkpoint's initialization seed is retained for randomly initialized
    frozen Engram tables. An explicit seed overrides only the CPU sampling generator.
    Training data is not required, but base-model identity must still match.
    """
    model_dir = Path(model_dir).resolve()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(state, dict) or state.get("format_version") not in (1, 2):
        raise ValueError("unsupported RL checkpoint format")
    if not all(key in state for key in ("config", "identity", "policy", "step")):
        raise ValueError("RL checkpoint is missing config, identity, policy or step")
    identity = state["identity"]
    if not isinstance(identity, dict) or type(identity.get("random_init")) is not bool:
        raise ValueError("checkpoint identity must contain a boolean random_init flag")
    if identity.get("rollout_engine") == "vllm" or state["config"].get("rollout_engine") == "vllm":
        raise ValueError("this checkpoint requires vLLM inference; standalone native sample/evaluate "
                         "is disabled. Use the owned Rollout Manager evaluation during training/resume.")
    random_init = identity["random_init"]
    model_config = load_model_config(model_dir)
    current_identity = run_identity(model_dir, model_dir / "config.json", random_init)
    initial_config = checkpoint_config(state["config"], model_config,
                                       rollout_engine=identity.get("rollout_engine"))
    from dsv41_train.sft_export import resolve_initial_bundle
    initial_config = resolve_initial_bundle(initial_config)
    add_lora_identity(current_identity, initial_config, model_config)
    add_engram_identity(current_identity, initial_config)
    if getattr(model_config, "model_type", None) == "qwen3_moe":
        current_identity["trainable_scope"] = state["config"].get("trainable_scope", "all")
    validate_identity(identity, current_identity, inference=True)
    version = state["step"]
    if type(version) is not int or version < 0:
        raise ValueError("checkpoint step must be a nonnegative integer")
    policy = state["policy"]
    values = dict(state["config"])
    del state
    values.update(learner_devices=tuple(devices), rollout_devices=tuple(devices))
    if max_new_tokens is not None:
        values["max_new_tokens"] = max_new_tokens
    if temperature is not None:
        values["temperature"] = temperature
    config = resolve_initial_bundle(checkpoint_config(values, model_config))
    sampling_seed = config.seed if seed is None else seed
    if type(sampling_seed) is not int or not 0 <= sampling_seed < 2**63 - 1:
        raise ValueError("sampling seed must be an integer in [0, 2**63 - 1)")
    torch.set_num_threads(config.num_threads)
    model = make_model(model_dir, model_config, config, devices, random_init, role="actor")
    load_policy(model, policy)
    del policy
    model.eval()
    return model, config, version, sampling_seed


def sample_checkpoint(*, model_dir: str | Path, checkpoint: str | Path,
                      prompt_ids: Optional[Sequence[int]] = None, prompt: Optional[str] = None,
                      messages: Optional[list[dict]] = None, devices: Sequence[str] = ("cpu",),
                      max_new_tokens: Optional[int] = None, temperature: Optional[float] = None,
                      seed: Optional[int] = None) -> dict:
    tokenizer = ModelTokenizer(model_dir) if (Path(model_dir) / "tokenizer.json").is_file() else None
    value = {name: item for name, item in (("prompt_ids", prompt_ids), ("prompt", prompt),
                                          ("messages", messages)) if item is not None}
    record = PromptRecord.from_mapping(value, tokenizer=tokenizer)
    model, config, version, sampling_seed = load_checkpoint_model(
        model_dir=model_dir, checkpoint=checkpoint, devices=devices, max_new_tokens=max_new_tokens,
        temperature=temperature, seed=seed)
    limit = model.config.max_position_embeddings - config.max_new_tokens
    if config.max_prompt_tokens is not None:
        limit = min(limit, config.max_prompt_tokens)
    if len(record.prompt_ids) > limit:
        raise ValueError("prompt length + max_new_tokens exceeds model context or prompt limit")
    if any(token >= model.config.vocab_size for token in record.prompt_ids):
        raise ValueError("prompt token IDs must be inside the model vocabulary")
    group = generate_group(
        model, record, group_size=1, max_new_tokens=config.max_new_tokens,
        temperature=config.temperature, **generation_options(model_dir, model.config, config),
        policy_version=version, group_id=0,
        generator=torch.Generator(device="cpu").manual_seed(sampling_seed),
        reward_fn=lambda record, response_ids: 0.0,
    )
    sample = group.samples[0]
    result = {
        "prompt_ids": record.prompt_ids,
        "response_ids": sample.response_ids,
        "finish_reason": sample.finish_reason,
        "stop_token_count": sample.stop_token_count,
        "policy_version": version,
        "seed": sampling_seed,
        "engram_device": model.engram_device,
    }
    if tokenizer is not None:
        result.update(decoded_fields(tokenizer, sample.content_ids, "response"))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample text or token IDs from a native RL policy")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--checkpoint", required=True, help="RL step-XXXXXXXX.pt checkpoint")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--prompt-ids", help="JSON list of token IDs, for example '[1,4,5]'")
    inputs.add_argument("--prompt", help="user text; uses checkpoint chat template when available")
    inputs.add_argument("--messages", help="JSON list of role/content messages ending in a user turn")
    parser.add_argument("--devices", "--device", default="cpu", help="cpu or comma-separated CUDA devices")
    parser.add_argument("--max-new-tokens", type=int, help="defaults to the checkpoint generation length")
    parser.add_argument("--temperature", type=float, help="defaults to the checkpoint temperature")
    parser.add_argument("--seed", type=int, help="sampling seed; defaults to the checkpoint seed")
    args = parser.parse_args()
    try:
        result = sample_checkpoint(
            model_dir=args.model_dir, checkpoint=args.checkpoint,
            prompt_ids=json.loads(args.prompt_ids) if args.prompt_ids is not None else None,
            prompt=args.prompt, messages=json.loads(args.messages) if args.messages is not None else None,
            devices=tuple(device.strip() for device in args.devices.split(",")),
            max_new_tokens=args.max_new_tokens, temperature=args.temperature, seed=args.seed,
        )
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
