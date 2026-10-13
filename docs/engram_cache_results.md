# Engram offload profiling and GPU row cache

Measured on 2026-10-12, using eight H20 GPUs, PyTorch 2.10.0+cu129,
CP2/EP4, and the 423,209,462-parameter synthetic model in
`configs/memory_eight_profile.json`. This is a five-layer performance fixture,
not a full DSV4.1 model or a representative text corpus. Another process retains
about 86 GiB on GPU 2; the trials fit in its remaining memory. The host is shared.

## Changes

- Batch index bounds and send/receive counts into one GPU-to-CPU metadata read.
  Fixed-size scatter counts replace `bincount`'s dynamic size synchronization.
- Reuse bounded pinned row buffers after DMA completion.
- In asynchronous mode, enqueue pinned ID D2H on the caller's CUDA stream and
  wait for its event in the lookup worker. Collectives still run on the caller.
- Add an optional owner-local GPU row cache, configured by
  `--engram-cache-mib`, default `0`. Budgets apply separately to every local table.
- Reject checkpoint restore while a prefetch ticket is unconsumed, before DCP
  writes parameter storage. Loading over an existing cache clears it.

The cache uses direct-mapped slots with bounded CPU tags. CPU parameters remain
authoritative; cached values are not registered parameters or checkpoint state.
Hits skip CPU row reads and value H2D. Misses fill slots, with deterministic
collision eviction. Output rows never alias cache slots. Backward sends every
original gradient occurrence to the CPU sparse parameter, preserving accumulation
order. The existing optimizer coalesces gradients and performs SGD exactly once.

After sparse SGD, touched cached rows are invalidated. Untouched rows may survive
across steps. Unknown weight version changes invalidate the whole cache;
parameter replacement, state loading, dtype/placement changes and close also
clear it. Unlike inference KV caching, frequently trained rows cannot simply
remain valid across updates. The main training reuse opportunity is within
gradient accumulation. Direct `.data` writes require an explicit `clear_cache()`.

This owner-local cache does not bypass ID routing, row-return NCCL, or gradient
D2H. A requester-local cache that also avoids remote row responses would need a
different gradient-routing and distributed invalidation protocol.

## Measurement Method

Each variant starts in a fresh eight-process group. There are three repeats per
workload, deterministically shuffled variant order, four warmup updates and
twenty timed updates. BF16 sparse SGD uses learning rate `1e-4`, zero momentum
and weight decay. Loading is layerwise; router deduplication, dense foreach and
FSDP CPU offload are disabled during throughput trials. Cache size is 8 MiB per
table, or 16 MiB per rank across the two tables.

`fresh` uses new random inputs for every microbatch and update, with two
microbatches per update. `repeated` uses four identical microbatches within an
update but new inputs on the next update. Input generation is outside timing.
CP peers receive the same full input. This avoids treating an endlessly repeated
fixed synthetic batch as evidence of realistic cache reuse.

Memory is the maximum process-local CUDA peak over ranks. Transfer volumes are
global sums over eight ranks per optimizer update. "Total cache H2D" includes
row values and cache indexing metadata; it excludes unchanged unrelated traffic.
Throughput ratios are paired by repeat. Ranges are observed ranges, not confidence
intervals. Profiler and readiness diagnostics run after the throughput interval.

## Fresh Inputs

| Variant | Tokens/s median [min, max] | Training peak MiB/rank | Loading peak MiB/rank |
| --- | ---: | ---: | ---: |
| GPU-resident baseline | 11073.18 [11052.55, 11115.31] | 579.98 | 187.73 |
| CPU async, optimized | 10429.18 [10271.05, 10446.24] | 512.01 | 123.61 |
| CPU async plus cache | 10352.14 [10294.02, 10451.92] | 528.01 | 123.61 |

Cache hits were **2.19%**. Row-value H2D decreased from 48 to 45.86 MiB/update,
but cache metadata added 2.84 MiB, producing **48.69 MiB total**, more than the
uncached path. The paired cache/uncached throughput change was **+0.05%**, with
a range of **-1.30% to +0.79%**. This provides no evidence of a stable speedup.
The median of paired ratios is different from the ratio of throughput medians.

Without caching, the final offload penalty relative to the matching GPU baseline
was 5.66% at the paired median, ranging from 5.64% to 7.60%. Before these changes
the paired median penalty was 6.13%, ranging from 3.30% to 6.26%. The small
apparent improvement is within run-to-run variation. The intermediate
counts/staging-only implementation also had no stable cache benefit. These
ablations do not establish a reliable end-to-end improvement from the scheduling
changes alone.

Final median maximum-rank load times were 1.08 s for GPU, 1.13 s for CPU async,
and 1.13 s with cache. Cache allocation is lazy and does not optimize checkpoint
reading or decoding. Loading peaks remain unchanged by enabling the cache.

## Repeated Microbatches

| Variant | Tokens/s median [min, max] | Training peak MiB/rank | Total cache H2D MiB/update |
| --- | ---: | ---: | ---: |
| GPU-resident baseline | 11400.36 [11290.69, 11403.93] | 585.66 | 0 |
| CPU async | 10691.73 [10647.58, 10870.28] | 513.57 | 96.00 |
| CPU async plus cache | 10849.61 [10744.17, 11354.34] | 529.57 | 31.95 |

Cache hits were **70.00%**. Row H2D was 28.12 MiB and cache metadata was 3.83 MiB
per update, a **66.72% total reduction** relative to uncached CPU lookup. The
cache/uncached paired throughput change was **+0.91%**, ranging from **-0.19% to
+6.20%**. One faster repetition dominates the upper end; this is not evidence of
a reliable 6% gain. The bandwidth reduction is repeatable, but training speedup
remains small and variable. All measured loss values matched the GPU baseline
exactly in both final workloads, across every rank and measured update.

All 768 diagnostic lookups per CPU variant were ready before consumption in
this workload too. With cache, the minimum completion margin was 8.42 ms for
layer 1 and 36.79 ms for layer 3. Without cache, minima were 3.63 and 25.20 ms.

Keep the cache disabled for low-reuse workloads. Enabling it is an explicit
memory/bandwidth tradeoff, most useful to evaluate when accumulation or corpus
statistics produce substantial reuse. In the fresh workload, caching gives back
16 MiB/rank of the roughly 68 MiB/rank saved by CPU offload, leaving about
52 MiB/rank of measured savings relative to GPU residence. These fixture memory
numbers must not be extrapolated into full-model capacity claims.

## What The Timeline Shows

The old asynchronous trace had, per rank/update, 12 scalar reads and 24 explicit
stream synchronizations inside four Engram ID-routing calls. The final trace
has zero scalar reads and four explicit stream synchronizations there. However,
the required split-count D2H copy still waits for GPU/collective progress.
Some waiting moved into `cudaMemcpyAsync`; the count reduction cannot be treated
as equivalent wall-time savings. Old and new traces use different sampled
updates, so operation counts are the stronger comparison.

Means across eight ranks for a separately profiled fresh-input update:

| Region/activity | CPU async, ms | CPU async plus cache, ms |
| --- | ---: | ---: |
| Forward host scope | 259.19 | 265.99 |
| Backward host scope | 329.73 | 331.76 |
| Optimizer host scope | 17.84 | 20.29 |
| Engram ID routing host scope | 27.95 | 30.20 |
| Local store host scope, including H2D completion | 6.77 | 15.92 |
| Engram resolve host scope, including row routing | 2.00 | 2.04 |
| GPU compute kernel duration sum | 37.70 | 37.76 |
| All H2D DMA duration sum | 0.577 | 0.542 |
| All D2H DMA duration sum | 0.457 | 0.450 |
| NCCL kernel duration sum, including waiting | 350.04 | 352.78 |

Scopes overlap and are inclusive; do not sum this table. Profiling substantially
increases host time, so these are not throughput measurements. NCCL durations
include synchronization and rank skew, not just data movement on the network.
There are roughly 18,000 compute kernels per update, many small operations.
The evidence points toward host dispatch and communication synchronization as
more promising targets than H2D bandwidth in this fixture. It does not prove
that all remaining offload penalty comes from any single component.

In the separate unprofiled readiness probe, all 384 sampled table lookups per
CPU variant completed before consumption. Minimum completion margins were
8.85/37.45 ms for Engram layers 1/3 without cache and 8.44/35.73 ms with cache.
Maximum-rank Future waits averaged about 0.020 ms/update in both variants.
The cache increased mean local lookup time from about 1.60 to 3.53 ms, but that
work still completed early. Saving already-hidden H2D time therefore has little
opportunity to improve the critical path. These readiness margins exclude the
row-return collective performed at consumption.

## Correctness And Lifecycle

- Full unit discovery passed 100 tests, with 8 GPU/external-weight skips in the
  sandbox. GPU cache tests were then run separately with host GPU access.
- Eight GPU cache unit tests passed, covering FP32/BF16 gradients and updates,
  duplicate and empty requests, eviction before backward, external optimizer
  updates, state reload, pinned-memory bounds and live-cache DCP restore.
- Eight lifecycle combinations (dtype, sync/async, dedup on/off) each passed 24
  updates. Outputs, gradients and weights matched the uncached CPU control
  bit-for-bit. GPU allocated-memory range after warmup was zero in every case.
- With repeated microbatches, the FP32 GPU comparison had maximum output/weight
  differences of `2.24e-8` and gradient differences of `1.67e-6`, within the
  existing numeric tolerances. BF16 without dedup matched exactly; BF16 with
  dedup had gradient differences up to `0.0078125` within relative tolerance,
  with identical updated weights. These cross-device/reduction differences
  are not cache-induced and are not described as bitwise GPU parity.
- Eight-rank full-model tests passed with activation checkpointing, selective
  FSDP CPU offload, cache and checkpoint restore. CP2/EP4 FP32 versus an unsharded
  reference had maximum gradient error `6.76e-6`; CP4/EP2 BF16 versus the GPU
  reference at the same topology had maximum gradient error `3.05e-5`.
- Six-process CPU CP2/EP3 and CP3/EP2 output, gradient and restore checks passed
  after the routing change. They exercise crossed groups; they do not substitute
  for the eight-GPU FSDP tests.
- Fresh eight-process CP2/EP4 resume passed after saving CPU tables with cache
  and selective FSDP offload, then restarting with GPU tables and no FSDP CPU
  offload. CPU/CUDA RNG and the input generator restored exactly; the next
  parameter update passed the FP32 comparison.

The separate historical BF16 full-topology deduplication limitations documented
in `memory_eight_results.md` are not resolved by this cache. Defaults remain
dedup off and cache off.

## Reproduce

Set `PYTHONPATH=.:tests` and use the same PyTorch environment for all trials.

```bash
python tests/run_memory_validation.py --suite benchmark --ranks 8 --cp 2 --ep 4 \
  --cases baseline,cpu_async,cpu_cache --repeats 3 --steps 20 --warmup 4 \
  --seq-len 512 --learning-rate .0001 --input-pattern fresh --prefetch-steps 4 \
  --trace-variants cpu_async,cpu_cache --config configs/memory_eight_profile.json \
  --output-dir artifacts/cache-fresh

python tests/run_memory_validation.py --suite benchmark --ranks 8 --cp 2 --ep 4 \
  --cases baseline,cpu_async,cpu_cache --repeats 3 --steps 20 --warmup 4 \
  --seq-len 512 --microbatches 4 --learning-rate .0001 --input-pattern repeated \
  --prefetch-steps 4 --config configs/memory_eight_profile.json \
  --output-dir artifacts/cache-repeated

python tests/summarize_engram_cache.py artifacts/cache-fresh artifacts/cache-repeated \
  --route-traces artifacts/cache-fresh/traces/cpu_async \
  --route-traces artifacts/cache-fresh/traces/cpu_cache --output artifacts/cache-analysis.json

python tests/summarize_memory_timeline.py artifacts/cache-fresh/cpu_cache-r0.json \
  --phases training --output-dir artifacts/cache-timeline

python -m unittest discover -s tests -p 'test_*.py'
python -m unittest test_engram_cache -v
python tests/dsv4_engram_offload_cuda.py --cache-bytes 4096 \
  --output artifacts/cache-lifecycle.json

python -m torch.distributed.run --standalone --nproc-per-node=8 \
  tests/dsv4_memory_parallel.py --case 2 --cache-bytes 4096 \
  --disable-dedup --activation-checkpoint full --output artifacts/cache-parity-fp32.json
python -m torch.distributed.run --standalone --nproc-per-node=8 \
  tests/dsv4_memory_parallel.py --case 3 --cache-bytes 4096 --bf16 \
  --disable-dedup --activation-checkpoint full --output artifacts/cache-parity-bf16.json

python -m torch.distributed.run --standalone --nproc-per-node=8 \
  tests/dsv4_memory_resume.py --phase save --cp 2 --ep 4 --cache-bytes 2048 \
  --folder artifacts/cache-resume
python -m torch.distributed.run --standalone --nproc-per-node=8 \
  tests/dsv4_memory_resume.py --phase resume --cp 2 --ep 4 --folder artifacts/cache-resume
```

Rendering requires Plotly. Timelines are standalone HTML and work offline.
Raw measurements and traces for this run are in
`artifacts/engram-cache-20261012/`. The `before-fresh`, `after-fresh` and
`final-fresh` directories distinguish original code, counts/staging changes,
and the final asynchronous-ID implementation. They are not repetitions of the
same implementation. The final high-reuse workload is in `final-repeated`.

The aggregate machine-readable report is
[`results/engram-cache-h20.json`](results/engram-cache-h20.json). Offline GPU
timelines and detailed accounting are in
[`timeline-async`](../artifacts/engram-cache-20261012/timeline-async/timeline.html)
and [`timeline-cache`](../artifacts/engram-cache-20261012/timeline-cache/timeline.html).
Each directory also contains `training-rank-0.html`, `time-breakdown.csv` and
`time-breakdown.json`. These are GPU traces; the readiness probe uses host
timestamps and is reported separately.

Validation source records are `lifecycle-cache.json`, `parity-cp2ep4-fp32.json`,
`parity-cp4ep2-bf16.json`, `cpu-cross/summary.json` and
`resume-placement/resume-result.json` under the same artifact root. There were
33 isolated throughput trials in total, plus two eight-rank GPU trace captures.
The HTML overview and detailed timelines were checked with Playwright for
nonempty plots, successful zoom and absence of browser errors.
