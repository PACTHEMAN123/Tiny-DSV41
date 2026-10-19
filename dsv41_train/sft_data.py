"""Last-answer text SFT using the checkpoint's chat template and stop IDs."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .rl.tokenizer import ModelTokenizer


def encode_example(record: dict, tokenizer: ModelTokenizer, *, max_length: int,
                   enable_thinking: bool = False) -> dict:
    """Encode one prompt/answer or messages/answer record without truncation.

    Labels remain aligned with input_ids; the model performs the causal shift.
    Only the final answer through its first EOS is supervised. Chat template
    suffixes after EOS (for example Qwen's newline) remain masked.
    """
    if type(max_length) is not int or max_length < 2:
        raise ValueError("max_length must be an integer of at least 2")
    if not isinstance(record, dict) or set(record) not in ({"prompt", "answer"}, {"messages", "answer"}):
        raise ValueError("SFT records require exactly prompt/answer or messages/answer")
    messages = record.get("messages")
    if "prompt" in record:
        messages = [{"role": "user", "content": record["prompt"]}]
    if not isinstance(messages, list) or not messages:
        raise ValueError("SFT messages must be a nonempty list")
    for message in messages:
        if (not isinstance(message, dict) or set(message) != {"role", "content"}
                or message["role"] not in ("system", "user", "assistant")
                or not isinstance(message["content"], str) or not message["content"].strip()):
            raise ValueError("SFT messages support only system/user/assistant roles with nonempty text content")
    if messages[-1]["role"] != "user":
        raise ValueError("SFT messages must end with a user message")
    prompt_ids, input_ids = tokenizer.encode_completion(
        messages, record["answer"], chat_template_kwargs={"enable_thinking": enable_thinking},
    )
    if len(input_ids) > max_length:
        raise ValueError(f"SFT example has {len(input_ids)} tokens, exceeding max_length={max_length}; truncation is disabled")
    prompt_length = len(prompt_ids)
    eos_index = next((index for index in range(prompt_length, len(input_ids))
                      if input_ids[index] in tokenizer.eos_token_ids), None)
    if eos_index is None:
        raise ValueError("completed SFT answer must contain a checkpoint EOS token")
    if eos_index == prompt_length:
        raise ValueError("SFT answer must contain at least one token before EOS")
    labels = [-100] * len(input_ids)
    labels[prompt_length:eos_index + 1] = input_ids[prompt_length:eos_index + 1]
    return {"input_ids": input_ids, "labels": labels,
            "attention_mask": [1] * len(input_ids), "prompt_ids": prompt_ids}


def load_jsonl(path, tokenizer: ModelTokenizer, *, max_length: int,
               enable_thinking: bool = False) -> list[dict]:
    """Read and validate text SFT records, reporting the failing source line."""
    examples = []
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                examples.append(encode_example(json.loads(line), tokenizer, max_length=max_length,
                                               enable_thinking=enable_thinking))
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    if not examples:
        raise ValueError(f"{path}: SFT dataset must contain at least one example")
    return examples


def collate_examples(examples: list[dict], *, pad_token_id: int,
                     pad_to_multiple_of: int = 1) -> dict[str, torch.Tensor]:
    """Right-pad encoded examples; return only model input tensors on CPU.

    Padding is an implementation detail, not additional training context. If a
    sequence-length cap is required, callers must allow for multiple rounding.
    """
    if not examples:
        raise ValueError("cannot collate an empty SFT batch")
    if type(pad_token_id) is not int or pad_token_id < 0:
        raise ValueError("pad_token_id must be a nonnegative integer")
    if type(pad_to_multiple_of) is not int or pad_to_multiple_of <= 0:
        raise ValueError("pad_to_multiple_of must be a positive integer")
    length = max(len(example["input_ids"]) for example in examples)
    length = (length + pad_to_multiple_of - 1) // pad_to_multiple_of * pad_to_multiple_of
    shape = (len(examples), length)
    batch = {"input_ids": torch.full(shape, pad_token_id, dtype=torch.long),
             "labels": torch.full(shape, -100, dtype=torch.long),
             "attention_mask": torch.zeros(shape, dtype=torch.long)}
    for row, example in enumerate(examples):
        size = len(example["input_ids"])
        if not size or len(example["labels"]) != size or len(example["attention_mask"]) != size:
            raise ValueError("SFT input_ids, labels and attention_mask must have the same nonzero length")
        for name in batch:
            batch[name][row, :size] = torch.tensor(example[name], dtype=torch.long)
    return batch
