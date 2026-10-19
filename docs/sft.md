# Supervised fine-tuning

`train_sft.py` uses the native model loaders and FSDP/CP/EP execution paths.
Run `python3 train_sft.py --help` for all options.

## Data and execution

Each JSONL record contains `prompt` and `answer`, or a text `messages` history
ending in a user turn and an `answer`. Only the final answer through EOS has
labels. The actual tokenized input must fit `--seq-len`; inputs are not truncated.
Use separate evaluation data and compare prompt-only generation as well as
teacher-forced loss.

Set `--family dsv4` or `qwen`, and `--mode lora` or `full`. DeepSeek loads the
complete decoder backbone unless `--num-layers` selects a prefix. Qwen requires
`CP == EP`. Gradient accumulation uses the global supervised-token count.
LoRA freezes base parameters, including Engram tables, unless `--train-engram`
is requested. LoRA defaults to AdamW; full training defaults to SGD.

```bash
python3 -m torch.distributed.run --standalone --nproc-per-node=8 \
  train_sft.py --model-path /path/to/DeepSeek-V4.1-Flash \
  --family dsv4 --mode lora --cp-size 1 --ep-size 8 \
  --data /path/to/train.jsonl --eval-data /path/to/eval.jsonl \
  --output-dir runs/sft --steps 20 --seq-len 4096 \
  --batch-size 1 --gradient-accumulation-steps 1 \
  --gradient-checkpointing --generation-eval
```

`--generation-eval` evaluates prompts without reference-answer tokens.
`--eval-batches` limits teacher-forced evaluation batches per data group;
`--generation-max-new-tokens` bounds generation. Authored identity fixtures can
be generated with `scripts/prepare_identity_sft.py`; they are development data,
not a general quality benchmark.

## Optimizer and Engram options

DeepSeek `--optimizer-in-backward` updates reduced FSDP shards during backward
with plain SGD and one accumulation microbatch. Momentum, weight decay and
gradient clipping are unsupported. A non-finite update aborts every rank;
earlier shard updates are not rolled back, so recover from a checkpoint.

With this path, repeated `--sgd-fp32-master-parameter NAME` options keep selected
trainable BF16 DTensor parameters in FP32 master storage. Exact names, such as
`lm_head.weight`, are required. This adds four bytes per selected local element
plus temporary buffers and changes checkpoint identity.

`--engram-offload` keeps each DeepSeek Engram row shard on CPU. Full SFT updates
these tables with SGD; LoRA freezes them unless `--train-engram` is supplied.
Trainable LoRA Engram also requires SGD. `--engram-cache-mib N` optionally caches
queried CPU rows on each rank's GPU; its per-rank budget covers values and tags,
while request outputs and staging are additional. Updates and restores invalidate
the cache. Sparse row gradients are combined in FP32 for resident and offloaded
tables. Frozen CPU tables are compatible with LoRA's AdamW.

DeepSeek SFT uses stable sparse-indexer selection and ordered sparse-attention
KV backward. Those policies are recorded in checkpoint identity; strict resume
rejects checkpoints with a missing or different policy. Qwen is unaffected.
These policies do not make every model operation bitwise deterministic.

## Checkpoints and initialization

Checkpoints include rank-owned trainable weights, optimizer state, data position
and RNG. Frozen base weights reload from the original `--model-path`.
Resume requires matching base identity, training configuration, world size and
CP/EP ownership:

```bash
# Repeat the original launch and training arguments, then add:
--resume runs/sft/checkpoint-00000020 --steps 40
```

`--steps` is the total target; `--steps 0` evaluates a restored model. `--save-every`
adds periodic snapshots. `--no-save` suppresses snapshots and adapter/bundle
exports, preserving metrics and evaluation; it cannot be combined with nonzero
`--save-every`.

Frozen-Engram LoRA exports `adapter.pt` and `adapter_metadata.json`. Base identity
hashes configuration/tokenizer assets and records weight filenames, sizes and
modification times, rather than hashing every base-weight byte. The adapter is
not an optimizer checkpoint. Legacy adapters without identity metadata require
an explicit compatibility warning.

Trainable CPU Engram checkpoints retain cumulative touched rows over the original
base. These snapshots require the same storage schema; they grow with distinct
rows touched. Complete-layer LoRA with trainable CPU Engram and zero dropout
exports an `sft-bundle-STEP` containing adapter and row state. Full SFT can
initialize native RL directly from its rank checkpoint. See [RL initialization](rl.md#sft-initialization).
