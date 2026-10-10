"""Evaluate a saved policy on text or token-ID JSONL without training."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .data import build_reward
from .evaluation import evaluate_policy
from .rollout import generation_options
from .runtime import prepare_records, validate_reward_records
from .sample import load_checkpoint_model
from .tokenizer import ModelTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--devices", default="cpu", help="cpu or comma-separated CUDA devices")
    parser.add_argument("--reward", help="defaults to the checkpoint reward")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--seed", type=int, help="evaluation seed; defaults to checkpoint eval_seed")
    args = parser.parse_args()
    model, config, step, _ = load_checkpoint_model(
        model_dir=args.model_dir, checkpoint=args.checkpoint,
        devices=tuple(device.strip() for device in args.devices.split(",")), max_new_tokens=args.max_new_tokens)
    config = replace(config, reward=args.reward or config.reward,
                     eval_seed=config.eval_seed if args.seed is None else args.seed)
    config.validate()
    tokenizer = ModelTokenizer(args.model_dir) if (Path(args.model_dir) / "tokenizer.json").is_file() else None
    records = prepare_records(args.data, model.config, config, tokenizer)
    reward = build_reward(config.reward, tokenizer=tokenizer)
    validate_reward_records(records, reward, config.reward)
    metrics = evaluate_policy(model, records, config, tokenizer=tokenizer, reward_fn=reward,
                              generation_kwargs=generation_options(args.model_dir, model.config, config),
                              step=step, output_dir=args.output_dir)
    print(json.dumps(metrics, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
