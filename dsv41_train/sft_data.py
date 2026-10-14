"""Local, answer-masked completion data and resumable distributed batch indexing.

Text records use the explicit plain-completion format ``prompt + "\n\n" +
response``. This is not a model chat template. Alpaca records use instruction
plus optional ``"\n\n" + input`` as their prompt. Tokenizer special-token
insertion is disabled; callers may explicitly supply BOS/EOS token IDs.
"""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import random

import torch


IGNORE_INDEX = -100
FORMAT_VERSION = "plain-completion-v1"


def _integer(value, name, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _integer_list(value, name, *, labels=False):
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a nonempty list")
    for item in value:
        if labels and type(item) is int and item == IGNORE_INDEX:
            continue
        _integer(item, name)
    return list(value)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _plain_text(record):
    if "prompt" in record or "response" in record:
        prompt, response = record.get("prompt"), record.get("response")
    else:
        prompt, response = record.get("instruction"), record.get("output")
        extra = record.get("input", "")
        if not isinstance(extra, str):
            raise ValueError("Alpaca input must be a string")
        if extra and isinstance(prompt, str):
            prompt += "\n\n" + extra
    if not isinstance(prompt, str) or not isinstance(response, str):
        raise ValueError("record needs string prompt/response or instruction/input/output")
    if not response.strip():
        raise ValueError("response must contain text")
    return prompt + "\n\n", response


def _encode_text(record, tokenizer, bos_token_id, eos_token_id):
    prompt, response = _plain_text(record)
    encoding = tokenizer.encode(prompt + response, add_special_tokens=False)
    ids = _integer_list(list(encoding.ids), "encoded input_ids")
    offsets = list(encoding.offsets)
    if len(offsets) != len(ids):
        raise ValueError("tokenizer must return one character offset per token")
    # A token crossing the prompt boundary is not wholly an answer token.
    labels = [token if start >= len(prompt) and stop > start else IGNORE_INDEX
              for token, (start, stop) in zip(ids, offsets)]
    if bos_token_id is not None:
        ids.insert(0, bos_token_id)
        labels.insert(0, IGNORE_INDEX)
    if eos_token_id is not None:
        ids.append(eos_token_id)
        labels.append(eos_token_id)
    return ids, labels, [True] * len(ids)


def _encode_tokens(record):
    ids = _integer_list(record.get("input_ids"), "input_ids")
    labels = _integer_list(record.get("labels"), "labels", labels=True)
    if len(ids) != len(labels):
        raise ValueError("input_ids and labels must have equal lengths")
    mask = record.get("attention_mask", [1] * len(ids))
    if not isinstance(mask, list) or len(mask) != len(ids):
        raise ValueError("attention_mask must match input_ids length")
    if any(type(value) not in (bool, int) or value not in (0, 1) for value in mask):
        raise ValueError("attention_mask values must be 0 or 1")
    # Restrict external records to the same right-padding contract as text data.
    if any(mask[index] > mask[index - 1] for index in range(1, len(mask))):
        raise ValueError("attention_mask must use right padding")
    return ids, labels, [bool(value) for value in mask]


class SFTDataset:
    """Fixed-length CPU examples; labels align with their original token positions.

    The model performs the one-token causal shift. ``supervised_tokens`` counts
    labels after that shift, excluding prompt, padding, and the first position.
    Data access never downloads a tokenizer or dataset.
    """

    def __init__(self, input_ids, labels, attention_mask, identity):
        self.input_ids = torch.tensor(input_ids, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.attention_mask = torch.tensor(attention_mask, dtype=torch.bool)
        self.supervised_tokens = self.labels[:, 1:].ne(IGNORE_INDEX).sum(dim=1)
        self.active_tokens = self.attention_mask.sum(dim=1)
        self.identity = identity
        self._permutations = OrderedDict()

    def __len__(self):
        return self.input_ids.shape[0]

    @classmethod
    def from_jsonl(
        cls, path, *, max_length, tokenizer=None, tokenizer_path=None,
        pad_token_id=0, bos_token_id=None, eos_token_id=None,
    ):
        """Read prompt/response, Alpaca, or input_ids/labels JSONL records.

        Truncation keeps the sequence prefix. A record with no remaining causal
        target is rejected with its source line number, rather than silently
        dropping it and changing the resume/data-rank schedule.
        """
        _integer(max_length, "max_length", minimum=2)
        _integer(pad_token_id, "pad_token_id")
        for name, value in (("bos_token_id", bos_token_id), ("eos_token_id", eos_token_id)):
            if value is not None:
                _integer(value, name)
        if tokenizer is not None and tokenizer_path is not None:
            raise ValueError("provide tokenizer or tokenizer_path, not both")
        source = Path(path).read_bytes()
        records = []
        for line_number, line in enumerate(source.decode("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: record must be an object")
            records.append((line_number, record))
        if not records:
            raise ValueError("SFT dataset has no records")

        needs_tokenizer = any("input_ids" not in record for _, record in records)
        encodings = [record.get("encoding") for _, record in records]
        encoding = next((value for value in encodings if value is not None), None)
        if encoding is not None:
            if (needs_tokenizer or not isinstance(encoding, dict)
                    or any(value != encoding for value in encodings)):
                raise ValueError("encoded chat records must share identical encoding metadata")
            for key in ("tokenizer_sha256", "config_sha256", "template_sha256"):
                value = encoding.get(key)
                if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                    raise ValueError(f"invalid encoded chat {key}")
            if tokenizer_path is None or _sha256(Path(tokenizer_path).read_bytes()) != encoding["tokenizer_sha256"]:
                raise ValueError("encoded chat requires its matching local tokenizer_path")
            if not isinstance(encoding.get("format"), str) or not encoding["format"]:
                raise ValueError("encoded chat format is missing")
        tokenizer_hash = None
        if needs_tokenizer:
            if tokenizer_path is not None:
                from tokenizers import Tokenizer
                tokenizer_bytes = Path(tokenizer_path).read_bytes()
                tokenizer = Tokenizer.from_str(tokenizer_bytes.decode("utf-8"))
                tokenizer_hash = _sha256(tokenizer_bytes)
            elif tokenizer is not None:
                if not callable(getattr(tokenizer, "to_str", None)):
                    raise ValueError("supplied tokenizer must expose to_str() for data identity")
                tokenizer_hash = _sha256(tokenizer.to_str().encode("utf-8"))
            else:
                raise ValueError("text records require a local tokenizer or tokenizer_path")
            if callable(getattr(tokenizer, "no_padding", None)):
                # Saved Tokenizers settings must not silently truncate answers
                # or insert padding before our explicit EOS. Clone a supplied
                # object so preparing data does not mutate the caller's tokenizer.
                tokenizer = type(tokenizer).from_str(tokenizer.to_str())
                tokenizer.no_padding()
                tokenizer.no_truncation()

        all_ids, all_labels, all_masks = [], [], []
        for line_number, record in records:
            try:
                if "input_ids" in record:
                    ids, labels, mask = _encode_tokens(record)
                else:
                    ids, labels, mask = _encode_text(record, tokenizer, bos_token_id, eos_token_id)
                ids, labels, mask = ids[:max_length], labels[:max_length], mask[:max_length]
                labels = [label if index > 0 and mask[index] else IGNORE_INDEX
                          for index, label in enumerate(labels)]
                if not any(label != IGNORE_INDEX for label in labels[1:]):
                    raise ValueError("record has no supervised tokens after causal shift/truncation")
                padding = max_length - len(ids)
                all_ids.append(ids + [pad_token_id] * padding)
                all_labels.append(labels + [IGNORE_INDEX] * padding)
                all_masks.append(mask + [False] * padding)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error

        identity = {
            "format": FORMAT_VERSION,
            "source_sha256": _sha256(source),
            "tokenizer_sha256": tokenizer_hash,
            "max_length": max_length,
            "pad_token_id": pad_token_id,
            "bos_token_id": bos_token_id,
            "eos_token_id": eos_token_id,
            "examples": len(records),
        }
        if encoding is not None:
            identity.update(format=encoding["format"], tokenizer_sha256=encoding["tokenizer_sha256"],
                            encoding=encoding)
        identity["fingerprint"] = _sha256(json.dumps(identity, sort_keys=True).encode("utf-8"))
        return cls(all_ids, all_labels, all_masks, identity)

    def length_statistics(self):
        """Distinguish effective tokens from padded tensor positions."""
        slots = self.input_ids.numel()
        active = int(self.active_tokens.sum())
        return {"examples": len(self), "tensor_sequence_length": self.input_ids.shape[1],
                "active_tokens_min": int(self.active_tokens.min()),
                "active_tokens_max": int(self.active_tokens.max()),
                "active_tokens_mean": float(self.active_tokens.double().mean()),
                "supervised_tokens_min": int(self.supervised_tokens.min()),
                "supervised_tokens_max": int(self.supervised_tokens.max()),
                "active_tokens": active, "processed_positions": slots,
                "padding_positions": slots - active,
                "fully_active_examples": int(self.active_tokens.eq(self.input_ids.shape[1]).sum())}

    def require_active_tokens(self, minimum):
        _integer(minimum, "minimum active tokens")
        if minimum and bool(self.active_tokens.lt(minimum).any()):
            indices = self.active_tokens.lt(minimum).nonzero().flatten().tolist()
            raise ValueError(f"examples {indices[:8]} have fewer than {minimum} effective tokens; padding is not long-context input")

    def batch(self, indices, *, device=None):
        """Return tensors and a scalar shifted supervised-token count."""
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
        if not isinstance(indices, (list, tuple)) or not indices:
            raise ValueError("indices must be a nonempty list")
        for index in indices:
            _integer(index, "example index")
            if index >= len(self):
                raise ValueError("example index is outside the dataset")
        selected = torch.tensor(indices, dtype=torch.long)
        result = {
            "input_ids": self.input_ids.index_select(0, selected),
            "labels": self.labels.index_select(0, selected),
            "attention_mask": self.attention_mask.index_select(0, selected),
            "supervised_tokens": self.supervised_tokens.index_select(0, selected).sum(),
        }
        return {name: value.to(device) if device is not None else value
                for name, value in result.items()}

    def _permutation(self, seed, epoch):
        key = (seed, epoch)
        if key not in self._permutations:
            order = list(range(len(self)))
            random.Random(seed + epoch).shuffle(order)
            self._permutations[key] = order
            if len(self._permutations) > 4:
                self._permutations.popitem(last=False)
        self._permutations.move_to_end(key)
        return self._permutations[key]

    def indices_for_step(
        self, step, micro_step, *, batch_size, accumulation_steps,
        data_rank=0, data_world_size=1, seed=42,
    ):
        """Partition a shuffled infinite example stream; step/micro_step are zero-based.

        A resumed run supplies its completed optimizer-step count as ``step``.
        ``data_world_size`` counts distinct data ranks, excluding CP replicas.
        To reproduce an earlier run keep seed, rank topology, batch size and
        accumulation steps unchanged and validate ``identity`` on resume.
        """
        for name, value in (("step", step), ("micro_step", micro_step), ("data_rank", data_rank),
                            ("seed", seed)):
            _integer(value, name)
        for name, value in (("batch_size", batch_size), ("accumulation_steps", accumulation_steps),
                            ("data_world_size", data_world_size)):
            _integer(value, name, minimum=1)
        if micro_step >= accumulation_steps:
            raise ValueError("micro_step must be less than accumulation_steps")
        if data_rank >= data_world_size:
            raise ValueError("data_rank must be less than data_world_size")
        offset = ((step * accumulation_steps + micro_step) * data_world_size + data_rank) * batch_size
        indices = []
        for position in range(offset, offset + batch_size):
            epoch, index = divmod(position, len(self))
            indices.append(self._permutation(seed, epoch)[index])
        return indices

    def batch_for_step(
        self, step, micro_step, *, batch_size, accumulation_steps,
        data_rank=0, data_world_size=1, seed=42, device=None,
    ):
        indices = self.indices_for_step(
            step, micro_step, batch_size=batch_size, accumulation_steps=accumulation_steps,
            data_rank=data_rank, data_world_size=data_world_size, seed=seed,
        )
        return self.batch(indices, device=device)


__all__ = ["SFTDataset", "IGNORE_INDEX", "FORMAT_VERSION"]
