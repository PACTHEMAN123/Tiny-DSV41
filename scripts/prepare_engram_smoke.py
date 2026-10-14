"""Pack repeated authored examples for execution checks, never quality/performance claims.

Uses SFTDataset's exact prompt/answer mask semantics. Complete examples are
concatenated in file order and repeated until enough nonempty packs exist;
attention may cross example boundaries. No tokens are added as packing padding.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dsv41_train.models.qwen.config import Qwen3MoeConfig
from dsv41_train.sft_data import IGNORE_INDEX, SFTDataset


def _positive_integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def pack_dataset(dataset, *, seq_len=256, packs=64):
    """Return the benchmark's input_ids/labels/metadata payload on CPU.

    A supervised token landing at a pack's first position is masked because the
    model has no preceding logit for it. Empty resulting packs are discarded;
    a full cycle without any usable pack fails instead of looping forever.
    """
    _positive_integer(seq_len, "seq_len", 2)
    _positive_integer(packs, "packs")
    if dataset.input_ids.ndim != 2 or dataset.input_ids.shape != dataset.labels.shape:
        raise ValueError("source input_ids and labels must have matching rank-two shapes")
    if dataset.attention_mask.shape != dataset.input_ids.shape:
        raise ValueError("source attention_mask must match input_ids")
    mask = dataset.attention_mask.bool()
    flat_ids = dataset.input_ids[mask].cpu()
    flat_labels = dataset.labels[mask].cpu()
    if not flat_ids.numel() or not flat_labels.ne(IGNORE_INDEX).any():
        raise ValueError("source contains no supervised tokens")

    source_tokens = flat_ids.numel()
    offset_period = source_tokens // math.gcd(source_tokens, seq_len)
    positions = torch.arange(seq_len, dtype=torch.long)
    packed_ids, packed_labels = [], []
    candidates = skipped = consecutive_empty = boundary_targets_masked = 0
    while len(packed_ids) < packs:
        indices = (positions + candidates * seq_len).remainder(source_tokens)
        ids = flat_ids[indices]
        labels = flat_labels[indices].clone()
        boundary_target = int(labels[0] != IGNORE_INDEX)
        labels[0] = IGNORE_INDEX
        candidates += 1
        if not labels[1:].ne(IGNORE_INDEX).any():
            skipped += 1
            consecutive_empty += 1
            if consecutive_empty >= offset_period:
                raise ValueError("no supervised tokens remain in any pack after masking its first position")
            continue
        consecutive_empty = 0
        boundary_targets_masked += boundary_target
        packed_ids.append(ids)
        packed_labels.append(labels)

    ids = torch.stack(packed_ids).long()
    labels = torch.stack(packed_labels).long()
    consumed_tokens = candidates * seq_len
    metadata = {
        "format_version": "engram-authored-smoke-v1",
        "purpose": "execution-only",
        "data_kind": "authored execution examples repeated by continuous packing",
        "quality_evaluation_data": False,
        "representative_performance_data": False,
        "warning": "Repeated authored smoke data; do not use for model quality, representative throughput, or cache-hit claims.",
        "source_data_identity": dataset.identity,
        "source_examples": len(dataset),
        "source_input_tokens": source_tokens,
        "source_supervised_tokens": int(flat_labels.ne(IGNORE_INDEX).sum()),
        "source_padding_tokens_removed": int((~mask).sum()),
        "source_order": "file order, no shuffle",
        "repetition_policy": "cycle the source token stream until enough nonempty packs exist",
        "source_repeated": consumed_tokens > source_tokens,
        "source_passes_consumed": consumed_tokens / source_tokens,
        "source_stream_tokens_consumed": consumed_tokens,
        "sequence_length": seq_len,
        "packed_sequences": packs,
        "candidate_packs": candidates,
        "empty_packs_discarded": skipped,
        "packing_padding_tokens": 0,
        "cross_example_attention": True,
        "first_position_label": IGNORE_INDEX,
        "accepted_pack_boundary_targets_masked": boundary_targets_masked,
        "input_tokens": ids.numel(),
        "supervised_tokens": int(labels[:, 1:].ne(IGNORE_INDEX).sum()),
        "supervised_tokens_per_pack_min": int(labels[:, 1:].ne(IGNORE_INDEX).sum(dim=1).min()),
    }
    return {"input_ids": ids, "labels": labels, "metadata": metadata}


def prepare(data_path, model_path, *, seq_len=256, packs=64,
            source_max_length=4096, tokenizer_path=None):
    """Encode through SFTDataset, reject source truncation, then pack without padding."""
    _positive_integer(seq_len, "seq_len", 2)
    _positive_integer(packs, "packs")
    _positive_integer(source_max_length, "source_max_length", 2)
    model_path = Path(model_path)
    config_bytes = (model_path / "config.json").read_bytes()
    if json.loads(config_bytes).get("model_type") != "qwen3_moe":
        raise ValueError("Engram smoke preparation requires a Qwen3 MoE config")
    config = Qwen3MoeConfig.from_json(model_path / "config.json")
    if seq_len > config.max_position_embeddings:
        raise ValueError("seq_len exceeds the model's max_position_embeddings")
    dataset = SFTDataset.from_jsonl(
        data_path, max_length=source_max_length,
        tokenizer_path=tokenizer_path or model_path / "tokenizer.json",
        pad_token_id=config.eos_token_id, bos_token_id=config.bos_token_id,
        eos_token_id=config.eos_token_id,
    )
    # SFTDataset intentionally supports truncation. This packing utility needs
    # complete source examples, so an example touching the cap is rejected.
    if dataset.attention_mask.sum(dim=1).max() >= source_max_length:
        raise ValueError("a source example reaches source_max_length; increase it to avoid truncation")
    valid_ids = dataset.input_ids[dataset.attention_mask]
    valid_labels = dataset.labels[dataset.labels.ne(IGNORE_INDEX)]
    if valid_ids.min() < 0 or valid_ids.max() >= config.vocab_size:
        raise ValueError("source input token ID lies outside model vocabulary")
    if valid_labels.min() < 0 or valid_labels.max() >= config.vocab_size:
        raise ValueError("source target token ID lies outside model vocabulary")
    result = pack_dataset(dataset, seq_len=seq_len, packs=packs)
    result["metadata"]["model_config_sha256"] = hashlib.sha256(config_bytes).hexdigest()
    result["metadata"]["source_truncation"] = "rejected when an example reaches the source length cap"
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Local Qwen3 MoE config/tokenizer directory")
    parser.add_argument("--data", default="examples/sft_smoke.jsonl", help="Local authored JSONL execution examples")
    parser.add_argument("--tokenizer-path", help="Optional local tokenizer.json override")
    parser.add_argument("--output", required=True, help="Output benchmark .pt file; a metadata .json is also written")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--packs", type=int, default=64)
    parser.add_argument("--source-max-length", type=int, default=4096,
                        help="Encoding cap per original example; increase if reached, never silently truncate")
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.suffix != ".pt":
        parser.error("--output must end in .pt")
    result = prepare(args.data, args.model_path, seq_len=args.seq_len, packs=args.packs,
                     source_max_length=args.source_max_length, tokenizer_path=args.tokenizer_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    torch.save(result, temporary)
    temporary.replace(output)
    output.with_suffix(".json").write_text(
        json.dumps(result["metadata"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "metadata": result["metadata"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
