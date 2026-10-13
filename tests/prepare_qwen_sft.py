"""Prepare reproducible answer-only Alpaca SFT packs from a local Parquet file."""

import argparse
import hashlib
import json
from pathlib import Path
import random

import pyarrow.parquet as pq
import torch
from tokenizers import Tokenizer


def prepare(data_file, tokenizer_file, seq_len, max_packs, seed):
    records = pq.read_table(data_file, columns=["instruction", "input", "output"]).to_pylist()
    random.Random(seed).shuffle(records)
    tokenizer = Tokenizer.from_file(str(tokenizer_file))
    ids, labels, packed_ids, packed_labels = [], [], [], []
    consumed = 0
    for record in records:
        user = record["instruction"]
        if record["input"]:
            user += "\n\n" + record["input"]
        prompt = f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
        encoding = tokenizer.encode(prompt + record["output"] + "<|im_end|>\n", add_special_tokens=False)
        target = [token if start >= len(prompt) else -100
                  for token, (start, _) in zip(encoding.ids, encoding.offsets)]
        ids.extend(encoding.ids)
        labels.extend(target)
        consumed += 1
        while len(ids) >= seq_len:
            current = labels[:seq_len]
            current[0] = -100
            if any(token != -100 for token in current):
                packed_ids.append(ids[:seq_len])
                packed_labels.append(current)
            del ids[:seq_len]
            del labels[:seq_len]
            if len(packed_ids) == max_packs:
                break
        if len(packed_ids) == max_packs:
            break
    tokens = torch.tensor(packed_ids, dtype=torch.long)
    targets = torch.tensor(packed_labels, dtype=torch.long)
    metadata = {"dataset": "tatsu-lab/alpaca", "source": "https://huggingface.co/datasets/tatsu-lab/alpaca",
                "license": "CC-BY-NC-4.0", "seed": seed, "dataset_records": len(records),
                "consumed_records": consumed, "sequence_length": seq_len,
                "packed_sequences": len(packed_ids), "input_tokens": tokens.numel(),
                "supervised_tokens": int(targets.ne(-100).sum()),
                "format": "Qwen user/assistant chat; prompt masked; continuous packing",
                "dataset_sha256": hashlib.sha256(Path(data_file).read_bytes()).hexdigest(),
                "tokenizer_sha256": hashlib.sha256(Path(tokenizer_file).read_bytes()).hexdigest()}
    return {"input_ids": tokens, "labels": targets, "metadata": metadata}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--max-packs", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = prepare(args.data, args.tokenizer, args.seq_len, args.max_packs, args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output)
    output.with_suffix(".json").write_text(json.dumps(result["metadata"], indent=2) + "\n")
    print(json.dumps(result["metadata"]), flush=True)


if __name__ == "__main__":
    main()
