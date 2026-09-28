# Native PyTorch LoRA

LoRA adds a trainable low-rank update to selected `nn.Linear` modules. The base
model stays frozen. No Hugging Face or PEFT runtime dependency is required.

## Read the implementation

Read these files in order; each owns one part of the adapter lifecycle:

| File | Responsibility |
| --- | --- |
| [`config.py`](../dsv41_train/lora/config.py) | Rank, alpha, dropout, and target names |
| [`linear.py`](../dsv41_train/lora/linear.py) | Forward computation and one layer's merge/unmerge |
| [`adapters.py`](../dsv41_train/lora/adapters.py) | Validate targets, freeze the base, inject adapters, select parameters |
| [`checkpoint.py`](../dsv41_train/lora/checkpoint.py) | Portable adapter export/import and per-layer loading |
| [`cli.py`](../dsv41_train/lora/cli.py) | Shared command-line options for the training scripts |
| [`__init__.py`](../dsv41_train/lora/__init__.py) | Stable public imports; no additional behavior |

For a base weight `W[out, in]`, the two trainable parameters are `A[rank, in]`
and `B[out, rank]`. The forward pass is:

```text
y = linear(x, W, bias) + (alpha / rank) * linear(linear(dropout(x), A), B)
```

`A` uses Kaiming initialization; `B` starts at zero. The initial adapter update
is therefore zero. On the first backward pass, A's gradient is zero; after B
updates, both factors can learn. Dropout is applied only on the adapter path.

`LoRALinear` reuses the original weight and bias objects. Their checkpoint names
stay unchanged, and `lora_a` / `lora_b` are added beside them. The default DSV4
targets are `attention.q_a`, `attention.q_b`, `attention.kv_proj`, and
`attention.o_b`, with rank 8, alpha 16, and dropout 0. Architectural
`q_lora_rank` / `o_lora_rank` are independent of the adapter rank.

## Use the Python API

```python
import torch
from dsv41_train.lora import (
    LoRAConfig, adapter_parameters, inject_lora, load_adapter, save_adapter,
)
from dsv41_train.models.dsv4 import DeepSeekV41Config, DeepSeekV41ForCausalLM

model = DeepSeekV41ForCausalLM(DeepSeekV41Config.tiny()).cuda()
matched = inject_lora(model, LoRAConfig(rank=8, alpha=16))
# Optional: load_adapter(model, "outputs/adapter.pt")
# Optional: apply_fsdp2(model, meshes)
optimizer = torch.optim.AdamW(adapter_parameters(model), lr=1e-3)

tokens = torch.randint(3, model.config.vocab_size, (1, 16), device="cuda")
optimizer.zero_grad(set_to_none=True)
model(tokens, labels=tokens).loss.backward()
optimizer.step()
save_adapter(model, "outputs/adapter.pt")
```

The required order is **load base weights -> inject -> load adapter (optional)
-> FSDP wrap (optional) -> create optimizer**. The real checkpoint loader accepts
`lora_config` and `adapter_path`; it injects and restores each materialized layer
before invoking the existing `layer_loaded` FSDP callback. Root embeddings and
the output head remain frozen as well.

Decoder layers return their shared KV/index state explicitly: FSDP may copy
input dictionaries, so in-place updates alone would be lost between layers.

Targets match exact module paths or dotted suffixes. Every target must match;
all matched modules must be materialized, unsharded `nn.Linear` modules. The
whole selection is validated before freezing. Grouped projections, expert
parameter tensors, meta tensors, and already FSDP-wrapped targets are rejected.
Loading a checkpoint layer by layer requires targets relative to each layer,
such as `attention.q_a`; the general `inject_lora` API also accepts full paths.

## Train, export, and resume

```bash
python3 train.py --device cuda --steps 10 --seq-len 32 \
  --lora --lora-rank 8 --lora-alpha 16 --lora-dropout 0.1 \
  --output-dir outputs/lora --adapter-out outputs/lora/adapter.pt

python3 train.py --device cuda --steps 20 --seq-len 32 \
  --lora --lora-rank 8 --lora-alpha 16 --lora-dropout 0.1 \
  --output-dir outputs/lora --resume
```

Both `train.py` and `train_dsv4_checkpoint.py` accept the same LoRA flags.
For distributed training, launch with the repository's normal `torchrun`
configuration and include `--lora`. All ranks must call `save_adapter`, because
it gathers FSDP adapter tensors; only rank zero writes the file.

There are two different checkpoint purposes:

| Operation | Contents and use |
| --- | --- |
| `--adapter-out` / `--adapter-in` | Version-1 adapter metadata and A/B tensors only; requires the same base model |
| `--output-dir` / `--resume` | Full DCP model, optimizer, step, data generator, and LoRA dropout RNG state |

Keep the same model, adapter configuration, and world size when resuming.
`--steps` is the final total step, not the number of extra steps. `--adapter-in`
starts from adapter weights with a fresh optimizer; it cannot be combined with
`--resume`. The adapter file records layer IDs and hyperparameters, but does not
fingerprint the base weights: the caller must choose the correct base checkpoint.

## Merge for inference

```python
from dsv41_train.lora import merge_adapters, unmerge_adapters

model.eval()
merge_adapters(model)
# Inference now uses W + (alpha / rank) * B @ A.
unmerge_adapters(model)
```

Merge is supported only on unsharded models in eval mode. The implementation
retains an exact copy of each base weight so BF16 unmerge does not accumulate
rounding error. Returning to `train()` automatically unmerges. Unmerge before
saving a full training checkpoint; use `save_adapter` for portable adapters.

## Verification

```bash
OMP_NUM_THREADS=2 python3 -m unittest discover -s tests -v
PYTHONPATH=.:tests CUDA_VISIBLE_DEVICES=0 python3 tests/dsv4_lora_cuda.py
PYTHONPATH=.:tests CUDA_VISIBLE_DEVICES=0,1 \
  python3 -m torch.distributed.run --standalone --nproc-per-node=2 \
  tests/dsv4_lora_fsdp.py
```

Tests cover dense-update output/gradient equivalence, base freezing, adapter
updates, file round trips, merge/unmerge, exact dropout/optimizer resume, and
injection/restore before each loader callback. GPU checks use small random
models and do not establish full released-model precision parity.

Validation on 2026-09-28 used PyTorch 2.10.0+cu129 and NVIDIA H20 GPUs:

- All 56 unit tests passed, with no skips.
- Single-GPU checks used a five-layer, hidden-size-32 model with 20 adapters
  and 10,240 trainable parameters. In 20 fixed-batch updates, evaluation loss
  changed from 3.4785 to 2.9734 in FP32 and from 3.4786 to 2.9945 with BF16
  autocast / FP32 weights. Base weights were unchanged; adapter reload was exact.
- Two-GPU FSDP2 matched reference gradients within a maximum absolute error of
  `7.45e-9`, across two AdamW steps with two microbatches per step. Adapter export
  passed, and continuing after DCP restore exactly matched uninterrupted updates.
- The existing post-training worktree was also tested with this package loaded
  in place of its original `lora.py`: 57 tests passed, including AC combinations;
  five optional HF precision tests remained skipped due to the unavailable
  reference implementation. AC itself is not included in this feature branch.
