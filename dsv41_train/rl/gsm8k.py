"""Official GSM8K JSONL selection and numeric rewards, separate from execution."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random

from .data import PromptRecord, build_reward
from .rewards import _numeric_answer

DATASET_REVISION = "3101c7d5072418e28b9008a6636bde82a006892c"
SPLIT_SHA256 = {"train": "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465",
                "test": "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"}


def load_split(path, split):
    if split not in {"train", "test"}:
        raise ValueError("GSM8K split must be train or test")
    path = Path(path)
    rows = []
    for index, line in enumerate(path.read_text().splitlines()):
        row = json.loads(line)
        if (not isinstance(row, dict) or not isinstance(row.get("question"), str)
                or not row["question"].strip() or not isinstance(row.get("answer"), str)
                or "####" not in row["answer"]):
            raise ValueError(f"{path}:{index + 1}: not an official GSM8K question/answer record")
        answer = row["answer"].rsplit("####", 1)[1].strip()
        _numeric_answer(answer)  # Invalid references abort; they are never zero rewards.
        rows.append({"id": f"gsm8k:{split}:{index}", "question": row["question"], "answer": answer})
    expected = 7473 if split == "train" else 1319
    if len(rows) != expected or len({row["question"] for row in rows}) != len(rows):
        raise ValueError(f"official GSM8K {split} must contain {expected} distinct questions")
    return rows, hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_data(train_path, test_path, tokenizer, *, revision, seed=42,
                 max_prompt_tokens=256, max_train_groups=64, eval_questions=128):
    if revision != DATASET_REVISION:
        raise ValueError("GSM8K revision differs from the pinned official dataset")
    if not 1 <= eval_questions <= 1319 or max_train_groups < 1 or max_prompt_tokens < 1:
        raise ValueError("use a nonempty held-out subset and train groups, with a positive prompt limit")
    train, train_hash = load_split(train_path, "train")
    test, test_hash = load_split(test_path, "test")
    if {"train": train_hash, "test": test_hash} != SPLIT_SHA256:
        raise ValueError("official GSM8K files differ from the pinned revision hashes")
    if {row["question"] for row in train} & {row["question"] for row in test}:
        raise ValueError("GSM8K training/test question overlap")
    rng = random.Random(seed)
    rng.shuffle(train)
    rng.shuffle(test)
    selected, excluded = {}, {}
    for split, rows, count in (("train", train, max_train_groups), ("test", test, eval_questions)):
        selected[split], excluded[split] = [], []
        for row in rows:
            text = row["question"] + "\nSolve the problem. End your answer with #### followed by the final numeric answer."
            record = PromptRecord(tokenizer.encode_prompt(prompt=text),
                metadata={**row, "dataset": "gsm8k", "split": split, "revision": revision})
            if len(record.prompt_ids) > max_prompt_tokens:
                excluded[split].append(row["id"])
                continue
            selected[split].append(record)
            if len(selected[split]) == count:
                break
        if len(selected[split]) != count:
            raise ValueError(f"not enough GSM8K {split} records fit the prompt limit")
    provenance = {"repository": "https://github.com/openai/grade-school-math",
        "revision": revision, "train_sha256": train_hash, "test_sha256": test_hash,
        "seed": seed, "max_prompt_tokens": max_prompt_tokens,
        "selection": "seeded shuffle, exclude overlength prompts without truncation",
        "subset_ids": {key: [record.metadata["id"] for record in values] for key, values in selected.items()},
        "overlength_ids_encountered": excluded,
        "reward": "existing numeric_answer: exact rational scalar, terminal ####/box/answer tag; malformed output scores zero",
        "format_metric": "terminal #### marker followed by a valid numeric scalar; not part of reward"}
    return selected["train"], selected["test"], build_reward("numeric_answer", tokenizer=tokenizer), provenance


def formatting_correct(text):
    if "####" not in text:
        return False
    try:
        _numeric_answer(text.rsplit("####", 1)[1].strip())
        return True
    except (ValueError, ArithmeticError):
        return False


def wilson_interval(correct, total):
    if total < 1 or not 0 <= correct <= total:
        raise ValueError("invalid binomial counts")
    z = 1.959963984540054
    p, z2 = correct / total, z * z
    center = (p + z2 / (2 * total)) / (1 + z2 / total)
    radius = z * ((p * (1 - p) / total + z2 / (4 * total**2))**.5) / (1 + z2 / total)
    return [center - radius, center + radius]
