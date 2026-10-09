"""Generate token IDs from an RL checkpoint and its original base model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import torch

from .checkpoint import load_policy, run_identity
from .config import RLConfig
from .data import PromptRecord
from .policy import DSV41Config
from .rollout import generate_group
from .runtime import make_model


def sample_checkpoint(
    *,
    model_dir: str | Path,
    checkpoint: str | Path,
    prompt_ids: Sequence[int],
    devices: Sequence[str] = ("cpu",),
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    seed: Optional[int] = None,
) -> dict:
    """Load policy weights without an optimizer and generate one response.

    The checkpoint's initialization seed is always retained for the frozen tiny
    Engram table. An explicit seed overrides only the CPU sampling generator.
    Training data is not required, but base-model identity must still match.
    """
    model_dir = Path(model_dir).resolve()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(state, dict) or state.get("format_version") != 1:
        raise ValueError("unsupported RL checkpoint format")
    if not all(key in state for key in ("config", "identity", "policy", "step")):
        raise ValueError("RL checkpoint is missing config, identity, policy or step")
    identity = state["identity"]
    if not isinstance(identity, dict) or type(identity.get("random_init")) is not bool:
        raise ValueError("checkpoint identity must contain a boolean random_init flag")
    random_init = identity["random_init"]
    current_identity = run_identity(model_dir, model_dir / "config.json", random_init)
    for key in (identity.keys() | current_identity.keys()) - {"data_sha256"}:
        if identity.get(key) != current_identity.get(key):
            raise ValueError(f"checkpoint base-model identity mismatch: {key}")
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
    config = RLConfig.from_dict(values)
    sampling_seed = config.seed if seed is None else seed
    if type(sampling_seed) is not int or not 0 <= sampling_seed < 2**63 - 1:
        raise ValueError("sampling seed must be an integer in [0, 2**63 - 1)")
    record = PromptRecord(prompt_ids)
    model_config = DSV41Config.from_model_dir(model_dir)
    if random_init:
        model_config.validate_random_init()
    if len(record.prompt_ids) + config.max_new_tokens > model_config.max_position_embeddings:
        raise ValueError("prompt length + max_new_tokens exceeds model context length")
    if any(token >= model_config.vocab_size for token in record.prompt_ids):
        raise ValueError("prompt token IDs must be inside the model vocabulary")
    torch.set_num_threads(config.num_threads)
    model = make_model(model_dir, model_config, config, devices, random_init)
    load_policy(model, policy)
    del policy
    model.eval()
    group = generate_group(
        model, record, group_size=1, max_new_tokens=config.max_new_tokens,
        temperature=config.temperature, eos_token_id=model_config.eos_token_id,
        policy_version=version, group_id=0,
        generator=torch.Generator(device="cpu").manual_seed(sampling_seed),
        reward_fn=lambda record, response_ids: 0.0,
    )
    return {
        "prompt_ids": record.prompt_ids,
        "response_ids": group.samples[0].response_ids,
        "policy_version": version,
        "seed": sampling_seed,
        "engram_device": model.engram_device,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample token IDs from a DeepSeek-V4.1 RL policy")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--checkpoint", required=True, help="RL step-XXXXXXXX.pt checkpoint")
    parser.add_argument("--prompt-ids", required=True, help="JSON list of token IDs, for example '[1,4,5]'")
    parser.add_argument("--devices", "--device", default="cpu", help="cpu or comma-separated CUDA devices")
    parser.add_argument("--max-new-tokens", type=int, help="defaults to the checkpoint generation length")
    parser.add_argument("--temperature", type=float, help="defaults to the checkpoint temperature")
    parser.add_argument("--seed", type=int, help="sampling seed; defaults to the checkpoint seed")
    args = parser.parse_args()
    try:
        result = sample_checkpoint(
            model_dir=args.model_dir, checkpoint=args.checkpoint,
            prompt_ids=json.loads(args.prompt_ids),
            devices=tuple(device.strip() for device in args.devices.split(",")),
            max_new_tokens=args.max_new_tokens, temperature=args.temperature, seed=args.seed,
        )
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
