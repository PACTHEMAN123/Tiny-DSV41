"""Fixed-prompt evaluation and inspectable rollout records."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import torch

from .rollout import generate_group


def append_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def decoded_fields(tokenizer, ids, name, *, skip_special_tokens=True):
    try:
        return {f"{name}_text": tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)}
    except ValueError as exc:
        # Padded model vocabularies can contain IDs with no tokenizer representation.
        return {f"{name}_text": None, f"{name}_decode_error": str(exc)}


def sample_record(record, sample, *, step, group_id, tokenizer=None, phase="train"):
    row = dict(phase=phase, step=step, group_id=group_id, policy_version=sample.policy_version,
               prompt_ids=sample.prompt_ids, response_ids=sample.response_ids,
               reward=sample.reward, raw_reward=sample.raw_reward,
               finish_reason=sample.finish_reason, stop_token_count=sample.stop_token_count,
               metadata=record.metadata)
    if tokenizer is not None:
        row.update(decoded_fields(tokenizer, sample.prompt_ids, "prompt", skip_special_tokens=False))
        row.update(decoded_fields(tokenizer, sample.content_ids, "response"))
    return row


def evaluate_policy(model, records, config, *, reward_fn, generation_kwargs, step,
                    tokenizer=None, output_dir=None, phase="eval"):
    if not records:
        raise ValueError("evaluation requires at least one prompt")
    generator = torch.Generator(device="cpu").manual_seed(config.eval_seed)
    samples, rows = [], []
    for index, record in enumerate(records):
        group = generate_group(
            model, record, group_size=config.eval_samples_per_prompt,
            max_new_tokens=config.max_new_tokens, temperature=config.temperature,
            policy_version=step, group_id=index, generator=generator, reward_fn=reward_fn,
            **generation_kwargs,
        )
        samples.extend(group.samples)
        rows.extend(sample_record(record, sample, step=step, group_id=index,
                                  tokenizer=tokenizer, phase=phase) for sample in group.samples)
    rewards = [sample.reward for sample in samples]
    metrics = dict(event="eval", phase=phase, step=step, policy_version=step,
                   prompts=len(records), samples=len(samples), reward_mean=statistics.mean(rewards),
                   reward_std=statistics.pstdev(rewards),
                   raw_reward_mean=statistics.mean(sample.raw_reward for sample in samples),
                   response_length_mean=statistics.mean(len(sample.response_ids) for sample in samples),
                   truncation_fraction=statistics.mean(sample.finish_reason == "length" for sample in samples),
                   zero_reward_fraction=statistics.mean(reward == 0 for reward in rewards))
    if output_dir is not None:
        append_jsonl(Path(output_dir) / "eval-samples.jsonl", rows)
        append_jsonl(Path(output_dir) / "eval-metrics.jsonl", [metrics])
    return metrics
