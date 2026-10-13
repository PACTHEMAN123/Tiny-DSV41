# H20 profiling and framework analysis

For the newer 625M-parameter LoRA/full-training comparison and module-level
bottleneck analysis, see [PEFT versus full training](profiling_peft_full.md).

Measured on 2026-09-28, based on `main@fc133842`, with the local post-training
changes. Hardware: 8 available NVIDIA H20 GPUs, 97,871 MiB per GPU, NV18 links;
these experiments use one or two GPUs. Software: Python 3.10.20, PyTorch
2.10.0+cu129, CUDA 12.9 runtime, NVIDIA driver 580.105.08.

## Method

Each single-GPU run starts in a fresh process with seed 42, 3 warmup steps and
10 measured steps. Batch size is 1. The three-layer random model has hidden size
256, 8 attention heads of width 64, 8 experts/top-2, and compression ratios
`[0, 2, 1]`. Engram is enabled in layer 1. Parameters use BF16 except the existing
FP32 HyperConnection/sink domains. AdamW uses `foreach=False`, learning rate 1e-4.
The base model has 12,227,322 parameters; LoRA adds 50,688 trainable parameters.

Step time is synchronized wall-clock forward/backward/optimizer time, excluding
batch generation, zeroing gradients, logging, and profiling. Individual stages
use CUDA events. Peak allocated memory includes parameters, optimizer state,
gradients, and activations; it is not an activation-only measurement. Reserved
memory and raw samples are also recorded. Profiler traces are collected in a
separate step after timing and memory measurements.

## Single-GPU results

| Tokens | Policy | LoRA | Step median, ms | Peak allocated, MiB |
| ---: | --- | --- | ---: | ---: |
| 128 | none | no | 69.34 | 177.43 |
| 128 | full | no | 115.43 | 165.07 |
| 128 | selective | no | 90.29 | 166.09 |
| 128 | selective | yes | 76.79 | 107.09 |
| 512 | none | no | 74.65 | 332.79 |
| 512 | full | no | 115.34 | 222.20 |
| 512 | selective | no | 96.89 | 254.42 |
| 512 | selective | yes | 78.80 | 172.16 |
| 1024 | none | no | 83.36 | 607.52 |
| 1024 | full | no | 128.36 | 334.78 |
| 1024 | selective | no | 103.72 | 430.39 |
| 1024 | selective | yes | 80.64 | 319.75 |

Selective AC uses `attention,moe`. At 1024 tokens it reduces total peak allocation
by 29.2% at a 24.4% step-time cost; full AC reduces it by 44.9% at a 54.0% cost.
The full-training loss matches across all three AC policies at every length.
LoRA freezes the base and changes the optimization problem: its time/loss must
not be presented as equivalent full-model training performance.

## Two GPUs and loading

The BF16 FSDP2 + LoRA + selective run at 512 tokens per rank uses 168.13 MiB peak
per GPU and takes 123.75 ms per step, or 8,275 input tokens/s globally. This is a
small, launch/communication-heavy model, not evidence of large-model scaling.

The six-case two-GPU FP32 test covers all AC policies with and without LoRA,
accumulating two distinct microbatches before each update. Outputs, averaged
parameter gradients, and SGD updates match the unsharded per-rank references;
maximum gradient difference was 7.45e-9 for these fixtures.
Adapter export/reload and fresh-process DCP resume were also exercised.

A separate six-layer synthetic quantized checkpoint comparison uses the pinned
upstream loader and identical FSDP2 callbacks. Per-rank peak allocation decreases
from 30.86 to 30.17 MiB. Allocation immediately after each layer's shard callback
drops by about 8 MiB; the final such point drops from 30.86 to 22.82 MiB.
This confirms removal of retained temporary full weights. It is a bounded
lifetime optimization, not pipelined I/O or loading directly into final shards.

## Bottlenecks and next work

- The sampled 512-token baseline launches about 6,657 `cudaLaunchKernel` calls
  plus 331 `cuLaunchKernelEx` calls. Many are small elementwise/reduction kernels;
  per-expert Python iteration and the 20 Sinkhorn iterations matter at this size.
  Profiler overhead is substantial, which is why trace timings are not used as
  throughput results.
- Attention still constructs dense local-query/global-key logits despite a
  sliding-window mask. The indexer also materializes dense scores. AC reduces
  tensors retained for backward, but does not remove quadratic forward work.
- CP currently gathers window KV and, at compression source layers, full hidden
  states before compression. CSA1/2 could instead exchange compressed latents
  and boundary information, but padding/group alignment and backward scaling
  require separate parity tests. No CP algorithm change is claimed here.
- Engram rows are fetched synchronously before the layer loop. The upstream CP
  branch has CPU offload, but asynchronous prefetch and sparse optimizer semantics
  need their own measured design.
- Keep future attention kernels behind the attention core, expert compute behind
  local routed computation, and EP libraries behind `TokenDispatcher`. Standard
  dense FlashAttention is not automatically equivalent to the current combined
  window/compressed attention with trainable sinks.

The upstream CP branch's staged router auxiliary-gradient hook fails external
loss scaling (`grad(0.5 * loss)` is not half the original gradient). This work
keeps the ordinary differentiable auxiliary loss and tests scaling/accumulation.

## Reproduce

```bash
CUDA_VISIBLE_DEVICES=0 python3 profile_dsv4.py \
  --layers 3 --seq-len 1024 --warmup 3 --steps 10 \
  --checkpoint-mode selective --output artifacts/selective.json

CUDA_VISIBLE_DEVICES=0 python3 profile_dsv4.py \
  --seq-len 512 --trace-dir artifacts/traces/baseline \
  --output artifacts/baseline.json

CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.run \
  --standalone --nproc-per-node=2 profile_dsv4.py \
  --seq-len 512 --steps 5 --checkpoint-mode selective --lora \
  --output artifacts/fsdp2.json
```

On the measured machine, the existing virtualenv's interpreter symlink is
broken. These runs use `/opt/python/3.10.20/bin/python3.10` with
`PYTHONPATH=.:tests:/data0/guorunfan.grf/envs/roll-vllm-017-py310-20260910-r5/lib/python3.10/site-packages`.
The GPU is exposed outside the coding sandbox. No existing environment was
modified and no full DeepSeek checkpoint was downloaded.

[Versioned raw results](results/post-training-h20.json) include configurations,
samples and per-rank measurements. Local detailed traces are under
`artifacts/traces/` and individual reports under `artifacts/profiles/`; generated
artifacts are gitignored. These reduced-shape results do not establish full
DeepSeek-V4.1 quality or throughput. HF reference parity remains unverified.
