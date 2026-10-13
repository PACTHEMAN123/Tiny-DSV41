# Loading review and optimization

Measured on 2026-09-29 using the reduced synthetic model in
[profile_memory_small.json](../configs/profile_memory_small.json). This review
changes loading execution and CPU FP8 decoding; it does not change training
math, Engram gradient routing, optimizer algorithms or feature defaults.

## Review findings

- **Open, correctness:** eight-rank BF16 deduplication parity is still not
  accepted. Earlier CP1/EP8, CP8/EP1 and CP8/EP8 cases failed after updates;
  CP2/EP4 also needs its NCCL failure investigated. The matched-dedup reference
  diagnostic points to intermediate BF16 rounding, but is not an acceptance
  substitute. See [the recorded failures](memory_eight_results.md). This review
  does not claim to fix them.
- **Fixed, loading latency:** `WeightSource.fill` created a pool for every
  parameter, including single-chunk parameters. `_load_direct` now owns one
  executor for the entire load. Single/empty chunks bypass task dispatch;
  multi-chunk sources retain one-chunk lookahead. A failed copy drains pending
  reads before the caller can close the reader's memory mapping.
- **Fixed, loading capacity:** an eager list contained every chunk range before
  the first read. Ranges are now generated lazily, so metadata does not grow
  with the full table's chunk count.
- **Fixed, decode latency:** CPU E4M3 conversion now uses a 256-entry FP32 table
  derived from PyTorch's native decoder. All encodings, NaNs, signed zeros,
  noncontiguous inputs, scales and FP32/BF16 outputs are checked against native
  conversion. CUDA decoding and other dtypes retain their existing path.

## Method and CPU comparison

The before implementation is commit `e3e808c`, extracted to an isolated source
directory. Both versions use the same current measurement scripts and profiler
instrumentation. Reports record `loading_implementation` to verify the imported
source. The workload has 14,398,338 parameters, five layers, hidden size 128,
eight experts, two 65539/65543-row Engram tables with 64 columns, BF16 storage,
SGD at 0.0001, two microbatches and sequence length 128.

Each cell is the median of three fresh processes, with two warmup updates and
five timed updates. Warm loading is a later reload within the same process,
without the profiler. The CPU materializer holds full local tensors and does
not measure FSDP shard savings. Filesystem pages may be cached; these are not
cold-disk timings. CPU tests use one intra-op thread on a Xeon Platinum 8469C.

| CPU warm load | Before ms [range] | After ms [range] | Reduction |
| --- | ---: | ---: | ---: |
| Layerwise | 116.25 [114.11, 118.11] | 90.78 [85.87, 106.14] | 21.9% |
| Direct synchronous | 131.43 [129.21, 141.20] | 93.40 [92.57, 96.93] | 28.9% |
| Direct pipelined | 185.77 [183.86, 190.01] | 97.03 [95.96, 98.33] | 47.8% |

Fresh-process load medians were 501.52 -> 472.18 ms, 539.14 -> 487.95 ms and
585.61 -> 494.41 ms respectively; these include framework initialization.
Every before/after measured loss array matched exactly across all nine pairs.
No training-throughput improvement is attributed to these loading changes.

In the separate CPU direct-pipeline load traces, pool constructions fell from
213 to 1, init/submit/shutdown calls from 673 to 40, and their inclusive time
from 51.38 to 2.10 ms. There are now 38 read submissions; 209 single-chunk
parameters avoid dispatch. Engram decode time fell from 58.86 to 28.11 ms in
those traces. In synchronous direct traces it fell from 46.16 to 19.58 ms.
Parent/child spans and spans from different threads overlap and cannot be
added as wall-time shares.

Raw CPU reports and traces are in `artifacts/loading-review-cpu-before/` and
`artifacts/loading-review-cpu-after/`, with a `portable.json` summary in each.

## Two-GPU timings and capacity

For the subsequent CP2/EP4 comparison on eight ranks, see the
[eight-rank memory baseline](memory_eight_baseline.md). Its smaller local table
shards give different savings; the two-rank measurements below remain separate.

NVIDIA H20, PyTorch 2.10.0+cu129, world2/CP1/EP2. Each variant ran in three
fresh process groups, with two warmup and five timed updates. The first trial
then captured a separate training trace. All 15 jobs completed; every recorded
loss matched the current resident baseline exactly for this workload and
learning rate. This is not evidence of eight-rank BF16 parity at learning rate
0.01, where the earlier failures were observed.

The direct-pipeline GPU load median improved **991.29 -> 943.79 ms (4.8%)**
against the old implementation, with ranges [982.56, 1058.85] and
[940.56, 955.78] ms. All six before/after runs had identical loss arrays on both
ranks. These fresh-process times include FSDP initialization and allocation.
Training throughput ranges overlap, so no training improvement is attributed
to the loader patch.

| Current policy | Tokens/s median [range] | First load ms | Load peak MiB | Train peak MiB |
| --- | ---: | ---: | ---: | ---: |
| GPU resident, layerwise | 1502.55 [1464.80, 1553.39] | 924.51 | 17.94 | 122.72 |
| Direct, pipelined | 1488.23 [1412.32, 1521.36] | 943.79 | 13.89 | 122.72 |
| CPU Engram, synchronous | 1441.82 [1386.46, 1458.94] | 971.10 | 9.94 | 114.31 |
| CPU Engram, asynchronous | 1418.69 [1357.43, 1485.96] | 935.15 | 9.94 | 114.30 |
| Combined | 1398.92 [1388.82, 1439.58] | 972.02 | 3.01 | 108.59 |

Peaks are maximum allocated bytes over ranks. Combined enables direct loading,
async CPU Engram, deduplication, dense foreach and dense/expert FSDP CPU offload
on layers 1 and 3. Its load peak is 83.2% lower and training peak 11.5% lower
than the current resident baseline; these are **policy capacity benefits**,
not new memory reductions caused by this loader patch. Direct loading requests
14.15 MiB of encoded data across both ranks instead of 16.91 MiB, a 16.3%
reduction. These are logical reads, not physical disk traffic.

Async throughput is 1.6% below synchronous CPU Engram in the medians and the
ranges overlap: this small workload shows no stable async speedup. CPU Engram
moves 1.50 MiB per step in each direction across both ranks. Combined dedup
reduces that to 1.484 MiB, only 1.09%, on these low-repetition random inputs.
CPU tables occupy zero GPU parameter bytes; their per-rank 8.00 MiB table
storage is on the CPU. Activations and communication still contribute to the
reported GPU training peak.

[Portable GPU measurements](results/loading-review-gpu.json) include all
samples and rank-level activity summaries. Old direct-loader measurements are
in `artifacts/loading-review-gpu-before/portable.json`.

## What the traces show

One training step per rank is profiled after timing. In the synchronous CPU
variant, rank 0 has 18,444 compute kernels totaling 31.28 ms, 152 H2D copies
totaling 0.424 ms and 152 D2H copies totaling 0.381 ms. Host CUDA launch calls
alone account for 68.15 ms of instrumented self time. The many short kernels
and host/control synchronization are more promising targets on this workload
than row-payload bandwidth.

Async has the same 18,444 compute kernels and 152 copies in each direction on
rank 0. Its all-H2D/compute overlap is only 0.40% on rank 0 and 0% on rank 1;
NCCL kernels are excluded from compute overlap. These are all model copies,
not an Engram-only overlap metric. In the async trace, local lookup/staging
takes 3.08-5.01 ms per rank, while ID routing takes 6.69-14.13 ms and resolve
2.08-2.51 ms. Those inclusive spans overlap across threads. Unprofiled resolve
host time is 1.32 ms per step on the slowest rank, including row routing and
reconstruction, not just waiting for I/O.

Trace NCCL duration varies greatly by rank: 298.44 versus 15.45 ms for sync CPU
and 259.94 versus 22.60 ms for async CPU. Profiler-induced rank skew and waiting
are included. These sums cannot establish link saturation or be divided by
unprofiled step time to estimate utilization. The traces support investigating
launch/synchronization latency, not claiming a measured NCCL bandwidth limit.

Loading traces are collected in separate invocations. Shape capture is disabled
because it retains tensor references that prevent FSDP's meta-storage swap.
The diagnostic load times include profiler overhead and are excluded from the
table above. On rank 0:

- Layerwise transfers 8,876,808 H2D bytes in 400 copies, and runs 1,023 compute
  kernels for decoding and initialization. Direct transfers 14,526,408 bytes
  in 196 copies with CPU decoding. Direct loading reduces source reads and
  storage peaks but currently sends decoded BF16 data rather than encoded
  weights; it does **not** minimize PCIe traffic for GPU-resident parameters.
- Combined keeps CPU-targeted shards on CPU during loading: rank 0 transfers
  3,135,128 H2D bytes in 103 copies, taking 0.249 ms of DMA in its trace.
- Direct Engram CPU decode takes 14.27 ms, expert decode 18.28 ms and dense
  decode 6.38 ms in instrumented host regions. Total H2D DMA is only 0.80 ms.
  Loader/framework CPU work remains more visible than payload DMA here.
- Direct-pipeline loading still synchronizes each copied chunk before buffer
  reuse. The executor change removes thread churn but is not a new event-driven
  DMA pipeline. `CPURowStore` also still allocates two staging buffers per lookup
  and synchronously returns gradients to CPU.

Raw load traces are under `artifacts/loading-review-load-traces/`; raw training
traces are under `artifacts/loading-review-gpu-after/traces/`. Further work
should prioritize reducing scalar/count synchronization, reusable bounded
staging buffers and host launch overhead. Any change to event ownership or
collective order needs its own lifecycle and distributed correctness tests.

## Verification

- Full unit suite: 91 tests, 86 passed and five skipped because HF reference
  files are absent. Log: `artifacts/loading-review-unit-final.log`.
- Two-GPU FSDP2/NCCL correctness: all five FP32 and all five BF16 policy cases
  passed, including multiple SGD updates, accumulation and checkpoint restore.
  Maximum reported gradient errors were 1.48e-6 and 6.10e-5 respectively.
  Results: `artifacts/loading-review-gpu-fp32/` and
  `artifacts/loading-review-gpu-bf16/`.
- New loading tests exercise bounded chunk metadata, borrowed executor reuse,
  single/empty chunks and read/copy failure cleanup. Existing fusion, grouped
  projection, partial quantization blocks and tail-shard checks remain active.
- Separate two-GPU loading captures for layerwise, direct and combined passed
  after disabling loading shape capture. Three summary-tool tests passed,
  including refusal to mix profiled and unprofiled loading timings.

The GPU driver became accessible again during this review. GPU 5 was occupied
by another workload, so this review uses only idle GPUs 0 and 1. Two-rank
results do not replace the outstanding eight-rank BF16 acceptance. Released
full-size checkpoint weights are not available; all current loads are synthetic.

## Reproduction

Use the same PyTorch environment for before and after; the baseline source
directory must precede the checkout in `PYTHONPATH`. The current scripts emit
the imported source path in every report. Run timing jobs in isolation.

```bash
PYTHONPATH=.:tests python3 tests/profile_memory_components.py \
  --config configs/profile_memory_small.json --seq-len 128 --threads 1 \
  --warmup 2 --steps 5 --repeats 3 \
  --variants layerwise,direct,direct_pipeline \
  --output-dir artifacts/loading-review-cpu-after

CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=.:tests \
python3 tests/run_memory_validation.py --suite benchmark --ranks 2 \
  --cases baseline,direct,cpu,cpu_async,combined --repeats 3 \
  --steps 5 --warmup 2 --seq-len 128 --cp 1 --ep 2 --learning-rate .0001 \
  --config configs/profile_memory_small.json \
  --trace-variants baseline,direct,cpu,cpu_async,combined --timeout 600 \
  --output-dir artifacts/loading-review-gpu-after

PYTHONPATH=tests python3 tests/summarize_memory_benchmark.py \
  artifacts/loading-review-gpu-after --output docs/results/loading-review-gpu.json

CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=.:tests \
python3 -m torch.distributed.run --standalone --nproc-per-node=2 \
  tests/dsv4_memory_benchmark.py --variant direct \
  --model-path artifacts/loading-review-gpu-after/fixture \
  --steps 1 --warmup 1 --seq-len 128 --cp 1 --ep 2 --learning-rate .0001 \
  --load-trace-dir artifacts/loading-review-load-traces/direct \
  --output artifacts/loading-review-load-traces/direct.json
```
