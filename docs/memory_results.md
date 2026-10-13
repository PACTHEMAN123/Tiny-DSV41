# DSV4 memory-policy acceptance results

The [eight-rank follow-up](memory_eight_results.md) passed the complete FP32
matrix but exposed BF16 multi-step parity failures. Those failures remain open;
the results below describe the earlier two/six-rank runs. Eight-GPU repeated
performance and offload profiling have not yet run.

Measured on 2026-09-29 with Python 3.10.20, PyTorch 2.10.0+cu129 and NVIDIA H20.
The host exposes eight GPUs outside the coding sandbox; GPU acceptance was
therefore performed with actual FSDP2/NCCL, including six-rank intersecting groups.
The complete portable [JSON report](results/memory-offload-h20.json) contains the
configuration, per-rank memory/transfer counters, losses and correctness results.

## Correctness

| Check | Result |
| --- | --- |
| Existing workspace unit suite | 78 tests, 73 passed and 5 HF-reference skips. |
| Six-rank CPU explicit reduction | All 16 combinations of CP/EP in `{1,2,3,6}` passed. Maximum gradient error `1.975e-6`. |
| Six-GPU FP32 FSDP2/NCCL | The same 16 topologies passed. Maximum gradient error `3.960e-6` against the unsharded reference. |
| Two-GPU BF16 | All 5 placement/loading/AC combinations passed against same-topology GPU-resident references. |
| Six-GPU BF16 | CP2/EP3 with combined offload and CP3/EP2 passed. |
| CUDA Engram lifecycle | 8 dtype/prefetch/deduplication combinations, 24 updates each. Outputs and weights matched the GPU baseline exactly; post-warmup allocated-byte range was zero. |
| Fresh-process SGD resume | CPU tables/FSDP offload to GPU placement; next-update maximum parameter error `7.276e-12`, exact CPU/CUDA/data RNG. |
| Fresh-process AdamW resume | Real sharded optimizer moments, including unused parameters; next-update maximum error `7.451e-9`, exact RNG. |
| Training CLI | BF16 CP2/EP2, two microbatches, full AC, direct loading, async CPU tables and dense/expert offload saved and resumed with GPU placement. |
| LoRA regression | Six FSDP full/LoRA AC combinations passed; maximum gradient errors `2.981e-8` / `8.382e-9`. LoRA CLI save/resume also passed. |
| Qwen regression | Two-GPU logits/loss/aux errors zero; maximum gradient error `4.471e-8` after the shared dispatch change. |

Matrices use a five-layer quantized fixture with six experts, padding and two
uneven Engram tables. They compare initial loaded weights, scaled losses,
gradients, multiple accumulated SGD updates and fresh-model checkpoint restore.
The FP32 matrix supplies the independent mathematical reference. BF16 uses the
same distributed topology to avoid confusing distributed reduction rounding with
placement differences. Deduplicated BF16 row gradients can differ before their
final dtype conversion; the lifecycle test's maximum such difference was
`0.00390625`, while final updates remained bitwise equal.

The implementation fixes exposed by these tests include root-output pytree
registration for FSDP backward, expert usage agreement across replica groups,
fixed-order paired token/weight exchanges on empty routes, compute-device router
buffers under CPU offload, explicit sparse-row coalescing, and DCP schema-driven
optimizer restoration without dummy updates or full-table gradient allocation.

## Performance workload

Each variant runs in a new two-GPU process, with CP1/EP1, a five-layer model,
hidden size 128, six experts, BF16 parameter storage, and two tables of
262147/262151 rows by 128 columns. Sequence length is 96, local batch size is 1,
with two microbatches per update. Two warmup steps precede five timed steps. The
same quantized fixture seed and input batches are used for every variant. The
final benchmark pass ran separately from correctness jobs.

Allocated peaks below are the maximum across ranks. Throughput counts unique
global input tokens. FSDP offload selects global layers 1 and 3. The combined
variant enables direct pipelined loading, asynchronous deduplicated CPU tables,
both FSDP parameter domains and dense foreach. Other rows enable only their
named feature relative to baseline; CPU variants use the same CPU storage mode.

| Variant | Load peak, MiB | Train peak, MiB | Tokens/s |
| --- | ---: | ---: | ---: |
| Baseline | 70.85 | 157.74 | 1379.91 |
| Direct shard loading | 68.06 | 157.74 | 1357.15 |
| CPU Engram, synchronous | 5.31 | 93.54 | 1301.61 |
| CPU Engram, asynchronous | 5.31 | 93.54 | 1373.92 |
| CPU Engram, deduplicated | 5.31 | 93.55 | 1357.05 |
| GPU Engram, deduplicated | 70.85 | 157.73 | 1319.52 |
| Dense FSDP offload | 70.59 | 156.73 | 1290.53 |
| Expert FSDP offload | 70.29 | 155.49 | 1350.89 |
| Dense foreach | 70.85 | 157.74 | 1270.50 |
| Combined | 2.42 | 90.28 | 1289.94 |

CPU tables remove approximately 64 MiB of local table storage from each GPU.
Combined training peak allocation is 42.8% below baseline and load peak is 96.6%
lower. Direct loading alone reduces encoded logical reads across both ranks
from 79.625 to 75.833 MiB, and load peak by 3.9%. Its load time was 1.121 seconds
versus 0.904 seconds for layerwise loading on this small fixture. Direct loading
is a capacity feature here, not a measured load-time improvement.

Across five measured updates, CPU lookup transfers 1.875 MiB each way over both
ranks; deduplication reduces each direction to 1.792 MiB. Asynchronous CPU lookup
was 5.6% faster than synchronous CPU lookup in this pass. Foreach and GPU
deduplication did not improve throughput. These short, launch-heavy measurements
have no confidence interval and do not establish stable speedups; all options
remain disabled by default.

CPU maximum RSS ranged from roughly 2128 to 2202 MiB and includes PyTorch/CUDA
initialization and synthetic checkpoint construction. It cannot isolate the CPU
table or loader footprint. Reserved/allocated bytes, local domain byte counts,
row-transfer totals and host resolve time remain available in the raw report.

## Delivery coverage and limits

| Stage | Acceptance evidence |
| --- | --- |
| Module extraction / baseline | Existing model tests and parameter keys preserved; local DTensor byte accounting in every GPU report. |
| Parallel correctness | Complete CPU/GPU topology matrices, independent CP/EP seeding and normalization, empty-route ordering regression. |
| Sharded state | Uneven table offsets, global expert/layer names, schema/configuration rejection, fresh-process SGD/AdamW next-update checks. |
| Synchronous CPU tables | Sparse gradients, CPU ownership, no full GPU table, consecutive updates and relocation rejection. |
| Direct loading | Full-tensor reads forbidden in unit tests; quantized fusions, grouped projections, shard tails, layer windows and final GPU shard parity. |
| Deduplication / prefetch | CPU/GPU variants, empty/repeated requests, AC, multiple microbatches, invalid/failed tickets, update boundary guards and stable CUDA allocations. |
| Selective FSDP / integration | Independent dense/expert measurements, combined mode, foreach, actual CLI save/resume and LoRA/Qwen regressions. |

No full released DeepSeek checkpoint is installed here; this work does not claim
full-model loading parity, peak capacity or end-to-end production throughput.
No multi-node offload run or HF-reference parity result is claimed. GPU testing
closes the small-model FSDP/NCCL acceptance gap; full-scale measurements remain
workload-specific validation. The previous profiling documents describe earlier
code and retain their original measurements.

See [reproduction commands](memory_validation.md). On this host the existing
virtualenv interpreter link is broken, so commands used
`/opt/python/3.10.20/bin/python3.10` with
`PYTHONPATH=.:tests:/data0/guorunfan.grf/envs/roll-vllm-017-py310-20260910-r5/lib/python3.10/site-packages`.
No environment or model download was required.
