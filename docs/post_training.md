# Native DSV4 post-training

Validated with Python 3.10.20 and PyTorch 2.10.0+cu129 on NVIDIA H20.
The default execution path remains native PyTorch. Transformers is only an
optional numerical reference; PEFT and custom kernels are not required.

## Activation checkpointing

All DSV4 training/profiling entry points accept:

| Option | Default | Behavior |
| --- | --- | --- |
| `--checkpoint-mode` | `none` | `none`, `full`, or `selective` |
| `--checkpoint-components` | `attention,moe` | Comma-separated `attention`, `moe`, `hc`; used by `selective` |
| `--cache-quantization` | `none` | `none` or explicit `hf` forward quantization with an identity STE |

`full` recomputes each decoder layer. Compressed KV, index keys, candidate masks,
and visibility lengths cross layer boundaries as explicit `AttentionState`
inputs/outputs; recomputation does not capture a mutable shared dictionary.

`selective` recomputes the attention softmax/value core and local routed/shared
expert computation. KV gathers, expert dispatch/combine, the used-expert
collective, and attention-sink materialization stay outside these regions.
Adding `hc` also recomputes HyperConnection normalization and Sinkhorn iterations.
Both policies use non-reentrant checkpointing and preserve dropout RNG state.
Checkpointing is inactive in evaluation or under `no_grad()`.

```python
from dsv41_train.activation import ActivationCheckpointConfig

model.set_activation_checkpointing(
    ActivationCheckpointConfig("selective", ("attention", "moe", "hc"))
)
```

`gradient_checkpointing_enable()` selects full-layer recomputation;
`gradient_checkpointing_disable()` restores the normal path.

## LoRA

`--lora` freezes base weights and trains only adapters. Defaults are rank 8,
alpha 16, dropout 0, and targets
`attention.q_a,attention.q_b,attention.kv_proj,attention.o_b`.
Use `--lora-rank`, `--lora-alpha`, `--lora-dropout`, and `--lora-targets` to override.
Targets match module names or qualified suffixes. Only `nn.Linear` is supported;
unmatched targets, grouped `o_a`, routers, and raw expert parameter lists are
rejected. Shared expert linears may be selected explicitly.

The adapters preserve base `weight`/`bias` names, initialize B to zero, and add
`lora_a`/`lora_b` parameters. Existing architectural `q_lora_rank` and
`o_lora_rank` are independent of these trainable adapters.

```bash
python3 train.py --steps 10 --seq-len 64 --lora \
  --checkpoint-mode selective --output-dir outputs/lora \
  --adapter-out outputs/lora/adapter.pt

python3 train.py --steps 20 --seq-len 64 --lora \
  --checkpoint-mode selective --output-dir outputs/lora --resume
```

`--steps` is the total target step when resuming. The distributed checkpoint
restores model, optimizer, step, data generator, and per-rank CPU/CUDA RNG state.
Use the same architecture, adapter settings, optimizer, and world size.
The post-training metadata/RNG extension is opt-in through `TrainingState`'s
`training_config` argument; existing callers using its default retain their
original checkpoint layout.

`--adapter-out` exports only adapter tensors and configuration. All distributed
ranks must participate; rank zero writes the file. `--adapter-in` restores a
matching adapter before FSDP wrapping and is mutually exclusive with `--resume`.
Adapter files validate the selected layer window as well as names, shapes, rank,
alpha, dropout, and targets. They require the matching base weights.

```python
import torch

from dsv41_train.lora import (
    LoRAConfig, inject_lora, adapter_parameters, load_adapter,
    save_adapter, merge_adapters, unmerge_adapters,
)

inject_lora(model, LoRAConfig())  # after materialization, before FSDP
optimizer = torch.optim.AdamW(adapter_parameters(model), lr=1e-4)
save_adapter(model, "adapter.pt")
load_adapter(model, "adapter.pt")  # before FSDP
model.eval()
merge_adapters(model)  # unsharded models only
unmerge_adapters(model)
```

Merge retains an unmerged weight copy so unmerge restores it exactly even in
BF16. This costs additional memory. Returning to `train()` automatically
unmerges. Save a portable adapter with `save_adapter`; a merged model's ordinary
state dict is not an adapter export.

## Released checkpoint loading

```bash
python3 -m torch.distributed.run --standalone --nproc-per-node=2 \
  train_dsv4_checkpoint.py --model-path /path/to/DeepSeek-V4.1-Flash \
  --start-layer 0 --num-layers 3 --ep-size 2 --cp-size 1 \
  --optimizer adamw --steps 2 --seq-len 128 --lora \
  --checkpoint-mode selective --adapter-out outputs/adapter.pt
```

This command requires the actual checkpoint and sufficient memory; the measured
benchmarks use reduced random/synthetic models instead. CP optimizations are not
part of this change. The existing `main` CP training path has not been promoted
to a validated long-sequence recipe.

`load_dsv41_backbone_window` and its prefix wrapper accept `lora_config`,
`adapter_path`, and `cache_quantization`. The loader freezes the base on meta,
materializes one layer, injects/restores its adapters, and invokes `layer_loaded`
for FSDP wrapping. Adapter targets are relative to each decoder layer.
Temporary state dicts and dequantization tensors leave scope before that callback.
Expert selection and chunked Engram row loading are preserved.

## Precision boundaries

`cache_quantization=none` retains the original training arithmetic. `hf` uses
the precision branch's FP8 window KV and FP4 compressed KV/index quantizers.
Forward values match those helper functions exactly; backward explicitly passes
the incoming gradient through the quantizer unchanged. This STE is a training
choice, not a claim that quantization is differentiable or that all gradients
match the inference reference. Discrete top-k selection remains non-differentiable.

HF parity tests are present but require the same `deepseek_v41` reference used
by the upstream precision branch. Neither the installed Transformers nor the
official snapshot checked for this work contains that module. Consequently,
external HF numerical parity is **not verified**. See the strict reference test
command in [testing.md](testing.md).
