# PEFT versus full-parameter post-training profile

Measured on 2026-09-28 on H20. LoRA removes most optimizer/parameter-gradient
storage but leaves the expensive activation path: dense masked attention,
compressed-KV gather/scatter, MoE input gradients, and mHC. Short sequences
expose eager dispatch overhead; long sequences expose quadratic attention
memory traffic. FSDP also exposes an optimizer dispatch bottleneck.

## Scope and method

- One or two NVIDIA H20 GPUs, 97,871 MiB each, NV18 links. Python 3.10.20,
  PyTorch 2.10.0+cu129, CUDA runtime 12.9.
- Random five-layer model: hidden 1024, 16 heads x 128, 32 routed experts/top-4,
  intermediate 1024, vocabulary 32768, mHC multiplicity 4, 20 Sinkhorn iterations.
  Ratios `[0, 2, 2, 1, 1]` exercise sliding attention, CSA2, CSA1, shared KV,
  and Engram. This is a reduced architecture, not full pretrained weights.
- Full mode: 625,243,742 parameters marked trainable. LoRA: 271,360 trainable
  parameters, about 0.0434% of the base; rank 8, alpha 16, dropout 0,
  targets `q_a/q_b/kv_proj/o_b`. PEFT here means native LoRA.
- BF16 parameter storage and AdamW moments, except existing FP32 mHC/sink
  domains. This is not FP32 master-weight mixed-precision training or QLoRA.
  Cache simulation is off. AdamW LR 1e-4; baseline `foreach=False`; batch 1/GPU.
  All-token shifted causal-LM loss uses ordinary autograd auxiliary loss.
- Fresh process per case, seed 42, 3 warmup and 10 measured updates. A generated
  batch is reused within a run; matched cases use identical initialization and
  input. These are performance fixtures, not convergence or quality evaluations.
- Step time is synchronized forward/backward/optimizer wall time, excluding
  data generation, gradient clearing and logging. Phase times use CUDA events;
  separate phase medians need not sum to step median. FSDP takes the maximum
  rank per sample and reports global input-token throughput.
- Peak allocated/reserved memory is reset after warmup and collected before
  tracing. A separate extra update samples operators and shapes. Module scopes
  are installed only for that update; backward attribution uses autograd's
  forward sequence/thread IDs. CUDA user annotations are excluded from kernel
  totals. Component CUDA sums include memory operations, can overlap between
  streams, and are not additional wall time or SM utilization. About 8% of
  4096-token single-GPU CUDA activity remains explicitly unattributed.

There are 20 timed runs: 18 distinct configurations and two repeats, plus ten
per-rank Chrome traces. See [configuration](../configs/profile_dsv41_medium.json)
and [versioned measurements](results/peft-full-h20.json).

## Single-GPU comparison

No AC; medians in milliseconds, peak allocated memory in GiB:

| Tokens | Training | Forward | Backward | Optimizer | Step | Peak GiB |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 512 | Full | 62.73 | 109.14 | 20.97 | 192.94 | 4.92 |
| 512 | LoRA | 62.39 | 88.83 | 2.29 | 153.52 | 2.27 |
| 2048 | Full, first | 83.06 | 117.30 | 22.14 | 222.89 | 9.98 |
| 2048 | Full, repeat | 84.42 | 116.12 | 22.97 | 224.33 | 9.98 |
| 2048 | LoRA, first | 116.00 | 92.55 | 3.25 | 211.46 | 6.80 |
| 2048 | LoRA, repeat | 78.94 | 90.71 | 1.11 | 171.29 | 6.80 |
| 4096 | Full | 138.08 | 169.86 | 15.53 | 323.49 | 20.40 |
| 4096 | LoRA | 137.93 | 149.58 | 0.62 | 288.24 | 16.28 |

The first 2048-token LoRA forward had substantial timing variability: step IQR
205.15-215.97 ms, versus 169.55-171.78 ms on repeat. The cause was not isolated.
Both runs are retained; no precise 2048-token LoRA speedup is claimed. At 4096
the IQRs were 322.55-324.99 ms (full) and 286.55-290.76 ms (LoRA).

LoRA saves 53.8% of peak memory at 512 tokens but only 20.2% at 4096. Base weights
still occupy about 1194 MiB. Full AdamW state is about 2387 MiB, versus 1.04 MiB
for LoRA. Frozen weights do not eliminate attention matrices or input gradients
through frozen experts needed to reach earlier adapters.

## Bottlenecks

### Dense masked attention dominates long sequences

At 4096 tokens, attention core uses 118.10 ms of sampled CUDA activity in full
training and 117.93 ms in LoRA, respectively 43.0% and 47.5% of the component
activity total including unattributed work. At 2048 it uses 32.61/32.51 ms:
doubling sequence length costs about 3.6x in this region.

`CompressedAttention._attend` computes `[B,H,L,L]` logits, then masks them to a
window. It concatenates compressed candidates and a sink, casts/normalizes in
FP32, applies softmax, and multiplies by KV. One FP32 `[1,16,4096,4096]` tensor
is 1 GiB, despite a window of only 128. The full trace spends 24.24 ms in copies
shaped `[1,16,4096,4160]`, another 10.14 ms in `[1,16,4096,4096]` copies, and
15.64 ms in `cat` across the model. Matmul throughput alone is not the issue.

Priority: fused attention that visits the window and selected compressed keys
and preserves their shared softmax/sink semantics. Ordinary dense FlashAttention
is not a drop-in replacement. See [model.py](../dsv41_train/models/dsv4/model.py).

### MoE has eager-loop and synchronization overhead

At 512 tokens, full training launches 20,954 kernels, versus 16,445 with LoRA.
Routed expert kernels consume 20.35/16.83 ms; the CPU path contains many launches,
small matrix operations, allocations and dynamic-index operations. The code
converts used expert IDs to a Python list, then runs `where`, gathers, three
linears and `index_add_` per expert. At 2048, `nonzero([8192])` occurs 160 times,
once per expert per layer; dynamic output sizes require CUDA synchronization.

Keeping top-4 and width but reducing expert count 32->8 lowers full step time
from 223-224 to 136.88 ms, and LoRA from 171-211 to 124.90 ms. This changes
parameter count, routing, optimizer work and GEMM shapes, so it is not an
equivalent-model optimization. It supports prioritizing grouped expert
computation and sorted/batched dispatch over the current Python loop.

Priority: group token assignments once, remove per-expert host synchronization,
and add grouped-GEMM/fused-expert computation behind the existing boundary.
See [moe.py](../dsv41_train/models/dsv4/moe.py).

### Compressed-KV selection has an expensive backward scatter

Advanced-index selection materializes KV as `[B,L,K,D]`. Backward scatters into
shared compressed rows with repeated destinations. At 2048, four selected-KV
`_index_put_impl_` calls total 13.75 ms in full training. At 4096, the CSA2 calls
alone take 9.57 ms full / 9.64 ms LoRA. Freezing base weights leaves this cost
because earlier adapters still require input gradients.

Priority: integrate selected-KV access/reduction into attention, or use a
dedicated grouped gradient reduction instead of per-query KV copies and generic
atomic scatter. Atomic contention is an inference from repeated destinations
and indexed-backward operators; hardware contention counters were not collected.

### mHC carries many tiny operations

At 2048, full forward/backward contains 790 sums over `[1,2048,4,4]` and 1560
divisions with row/column broadcast shapes. Changing Sinkhorn iterations 20->1
reduces full step time to 189.57 ms and LoRA to 149.76 ms. This changes the
model and is only a sensitivity control, not a recommended training setting.

Priority: fuse/compile the existing iterations and mixing while preserving
their math and iteration count, then check gradients. CUDA time fractions alone
understate their launch/autograd overhead.

### Indexer and Engram are secondary at the tested scale

Indexer activity grows from about 2.50 ms at 2048 to 8.41 ms at 4096. Dense
query-key scores are computed before candidate masking/top-k, so this is also
a long-sequence concern. Hard top-k indices do not provide learning gradients
to all indexer projections; this run does not establish a full indexer-training
objective, even though full mode marks the parameters trainable.

Engram integration uses 3.40 ms full / 1.94 ms LoRA at 4096, with small
hash/lookup costs. The table here is just 131072 x 128, resident on GPU.
Full-scale table capacity, CPU offload and asynchronous prefetch are unmeasured.

## FSDP and optimizer controls

At 2048 tokens per rank, global batch grows from 1 to 2 on two GPUs. This is
fixed-per-rank-batch comparison, not fixed-global-batch strong scaling.

| GPUs | Training | AdamW | Forward ms | Backward ms | Optimizer ms | Step ms | Peak GiB/GPU | Global tokens/s |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Full | loop | 83.06 | 117.30 | 22.14 | 222.89 | 9.98 | 9188 |
| 2 | Full | loop | 103.23 | 155.35 | 105.64 | 362.29 | 8.38 | 11306 |
| 1 | Full | foreach | 80.15 | 114.89 | 14.22 | 212.84 | 9.98 | 9622 |
| 2 | Full | foreach | 109.63 | 157.67 | 15.68 | 276.82 | 8.38 | 14797 |
| 2 | LoRA | loop | 106.09 | 120.08 | 16.99 | 242.51 | 6.38 | 16890 |

Two-GPU full loop AdamW has 142.76 ms traced CPU self time but only 6.66 ms
associated CUDA work. Top CPU operators include DTensor `PythonSubclass`
dispatch. Explicit `foreach=True` lowers optimizer-stage wall time by 85% and
whole-step time by 23.6%. This is the most direct measured implementation
improvement. Training entrypoint defaults were not changed in this task.

LoRA still all-gathers frozen base weights: prominent rank-0 all-gather kernels
occur 40 times, compared with 41 for full training. Their cumulative durations
are 15.65/54.60 ms but include waits and overlap; do not add them to step time.
Investigate wrapping/prefetch after optimizer dispatch. CP/EP, MoonEP and UltraEP
communication were not profiled.

Single-GPU fused AdamW measured 218.08 ms full and 180.21 ms LoRA, within or
close to loop-baseline variability. No substantial fused-optimizer whole-step
gain is established by this experiment.

## Activation checkpoint tradeoff

At 4096 tokens, selective `attention,moe` checkpointing gives:

| Training | No AC: ms / GiB | Selective: ms / GiB |
| --- | ---: | ---: |
| Full | 323.49 / 20.40 | 496.26 / 12.11 |
| LoRA | 288.24 / 16.28 | 446.03 / 8.31 |

AC saves 40.6%/48.9% of peak allocation but adds 53.4%/54.7% step time. It helps
fit sequences; it does not solve dense forward work or expert launch overhead.

## Reproduce and validation

```bash
python3 profile_post_training.py --fsdp-gpus 0,1

python3 profile_post_training.py --fsdp-gpus 0,1 \
  --output-dir artifacts/architecture-profile-followup --no-traces \
  --only 'base-s2048-(full|lora)-none-loop-gpu1|base-s2048-full-none-foreach-gpu[12]'

CUDA_VISIBLE_DEVICES=0 python3 profile_dsv4.py \
  --config configs/profile_dsv41_medium.json --seq-len 4096 --lora \
  --trace-dir artifacts/lora-trace --output artifacts/lora-profile.json
```

The sweep now also offers single/two-GPU foreach LoRA controls for future runs;
those two additional cases are not in this report. The measured machine's
interpreter/PYTHONPATH workaround is in [the earlier report](profiling.md#reproduce).
Logs, full summaries and Chrome traces live in `artifacts/architecture-profile/`;
repeat/foreach results live in `artifacts/architecture-profile-followup/`.

Profiler scopes pass output/gradient-equivalence and cleanup tests. The full
suite passes 57 tests, with 5 explicit HF-reference skips. Regression on the
GPU-visible host exposed a CPU-checkpoint RNG shape bug when CUDA initialization
changed between save/load; capture now follows the model device. CPU regression
and an exact GPU LoRA optimizer/dropout-resume check pass after that fix.
No model math, kernel backend or training defaults changed. These reduced-shape
results do not establish full 40-layer, 384-expert or million-token performance.
