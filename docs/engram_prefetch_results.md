# Engram prefetch deadlines and throughput

Measured on 2026-10-12 on eight H20 GPUs, world8/CP2/EP4, PyTorch
2.10.0+cu129. CPU lookup and H2D finished before consumption for every sampled
lookup on layers 1 and 3, including the 423,209,462-parameter workload.
This did not translate into a throughput gain over synchronous CPU Engram.
An immediate-consumption layer-0 control missed every prefetch deadline.

## Method

- Compare GPU-resident Engram, synchronous CPU Engram and asynchronous CPU
  Engram with identical weights, input seeds, BF16 storage and sparse SGD
  (learning rate 0.0001, no momentum or weight decay).
- Keep deduplication and FSDP CPU offload disabled; all cases use layerwise
  loading. Thus the sync/async pair changes only Engram scheduling.
- Four warmup updates, 15 uninstrumented timed updates, two microbatches per
  update. The normal workloads have three fresh-process repetitions per
  policy, in reproducibly shuffled order. The layer-0 control has two pairs.
- After timing, sample eight additional updates with
  `tests/engram_prefetch_probe.py`. The probe is never active during throughput
  measurement and adds no CUDA synchronization.
- Record prefetch entry/return, store start/return, resolve entry/return,
  `Future.result()` duration and row-return routing separately. In the current
  `CPURowStore`, returning from lookup follows completion of staged H2D DMA.
  Therefore `resolve_start - store_ready` is a conservative readiness margin.
- Readiness is local to the owner GPU, before row-return all-to-all. It does
  not mean the requesting GPU has already received the rows. Host arrival at
  resolve is also not a measurement of the eventual GPU consumer kernel.
- Separately capture one training update per rank with CPU/CUDA torch.profiler
  for GPU residency and async CPU placement on the larger workload. These
  captures are diagnostic and are excluded from the repeated throughput data.

The probe restores patched methods on exit. Tests check on-time and delayed
lookups, unchanged outputs/gradients, exception cleanup, and separation of
Future waiting from routing. Ten focused probe, timeline and summary tests
passed. Production scheduling and offload implementations were not changed.

## Readiness

Layer IDs are zero-based. A lookup here is one table request batch, not one row.
Margins are measured when the main thread reaches resolve; positive means the
CPU lookup and H2D have already completed.

| Workload | Layer | Async lookups | Ready before resolve | Minimum margin ms | Median margin ms | Future call p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 14.4M, sequence128, row width64 | 1 | 384 | 100% | 10.902 | 12.260 | 0.0045 |
| 14.4M, sequence128, row width64 | 3 | 384 | 100% | 35.995 | 37.604 | 0.0041 |
| 423.2M, sequence512, row width256 | 1 | 384 | 100% | 8.754 | 13.384 | 0.0070 |
| 423.2M, sequence512, row width256 | 3 | 384 | 100% | 36.861 | 40.573 | 0.0042 |
| Immediate layer0, sequence512, row width256 | 0 | 256 | 0% | -1.727 | -0.770 | 1.0001 |

The two normal workloads transfer 3 MiB and 48 MiB of Engram row values H2D
per optimizer update, respectively, summed over all ranks. The larger test
increases transfer volume by 16x and global resident table size by 32x.

On the normal workloads, Future calls average 0.018/0.020 ms per update after
taking the maximum rank each update. Those calls include Python overhead;
every Future was already done, so this is not evidence of residual I/O waiting.
Layer-0 waiting averages 0.795 ms per lookup (maximum 1.751 ms), or 1.931 ms per
update after taking the maximum rank. It has no preceding decoder layer and
only one Engram table, so there is no other table's request dispatch to hide
its lookup either.

Most small-workload layer-1 lookups (99.48%) are already ready when dispatching
all table requests finishes, before entering the decoder loop. That fraction
is 46.61% for the larger workload. Layer-3 lookups finish later, but still well
before their layer consumes them. This explains why a low H2D/compute-kernel
overlap percentage alone cannot establish that prefetch is late: CPU dispatch
and communication also occupy the interval before consumption.

## Throughput

Tokens/s are medians over repetitions. The last column is the median of the
within-repetition async/sync ratios, which need not equal the ratio of medians.

| Workload | GPU resident | CPU sync | CPU async | Paired async vs sync |
| --- | ---: | ---: | ---: | ---: |
| 14.4M, sequence128 | 2978.69 | 2959.18 | 2957.77 | -0.34% |
| 423.2M, sequence512 | 11296.21 | 10554.98 | 10517.28 | -0.51% |
| Immediate layer0, sequence512 | not sampled | 11535.96 | 11692.47 | +1.36% |

Normal-workload paired async/sync differences range from -0.49% to +0.30%
and from -1.80% to -0.36%. There is no demonstrated async speedup for the
current layers 1/3 placement. On the larger workload async CPU placement is
5.78%-9.32% slower than the corresponding GPU-resident run; the paired median
loss is 5.99%. Completing forward prefetch on time does not eliminate all
offload costs.

The layer-0 control demonstrates an actual missed deadline, not an observed
throughput regression against sync. Sync also pays for lookup and transfer;
the two sampled async runs were 1.24%-1.47% faster. This small, shared-host
sample cannot establish a general layer-0 speedup.

All 22 repeated runs completed. Every recorded loss across ranks and timed
updates matched the paired comparison exactly; the normal workloads also
matched GPU residency. This is loss agreement on this low-learning-rate
fixture, not a substitute for full parameter/gradient parity or resolution
of the previously recorded eight-rank BF16 deduplication failures.

## GPU timelines

The following are means across eight ranks for one separately profiled update.
Host scopes are inclusive, and GPU event sums overlap across streams.

| Diagnostic scope, ms/update | GPU resident | CPU async |
| --- | ---: | ---: |
| Forward host scope | 246.684 | 257.380 |
| Backward host scope | 331.698 | 329.201 |
| Optimizer host scope | 21.813 | 18.003 |
| Engram ID-route host scopes | 16.898 | 30.389 |
| Engram store host scopes | 0.331 | 7.359 |
| Engram resolve host scopes | 1.808 | 2.362 |
| GPU compute kernels | 37.919 | 37.647 |
| H2D DMA | 0.299 | 0.573 |
| D2H DMA | 0.333 | 0.506 |

The trace points to request routing/synchronization as a useful next target,
with about 18K GPU kernels per update and substantial CPU launch overhead.
It does not support attributing the measured throughput loss solely to DMA
bandwidth or CPU SGD. The optimizer scope is actually smaller in this one
CPU-offload capture. Profiler overhead and rank skew are substantial, so this
single sample cannot causally partition the unprofiled 6%-9% loss.

Offline plots and raw traces:

- [GPU-resident overview](../artifacts/engram-prefetch-20261012/timeline-baseline/timeline.html)
- [Async CPU overview](../artifacts/engram-prefetch-20261012/timeline-cpu_async/timeline.html)
- [Async CPU rank-0 detail](../artifacts/engram-prefetch-20261012/timeline-cpu_async/training-rank-0.html)
- [Async CPU time accounting](../artifacts/engram-prefetch-20261012/timeline-cpu_async/time-breakdown.json)
- [Larger-workload readiness trace](../artifacts/engram-prefetch-20261012/medium-seq512/prefetch-host-timeline.json)
- [Layer-0 readiness trace](../artifacts/engram-prefetch-20261012/first-layer-seq512/prefetch-host-timeline.json)

The readiness traces contain host spans only; the GPU timelines use Kineto
CPU/CUDA events. Original GPU traces are under
`artifacts/engram-prefetch-20261012/profiles/{baseline,cpu_async}/`.

## Limits and reproduction

GPU 2 retained an existing VLLM process using 88,214 MiB throughout the
recorded occupancy checks. These are shared-host measurements, not exclusive
performance acceptance. All benchmark workers exited; the existing service
was left running. The results cover synthetic five-layer workloads and do
not guarantee readiness for a full released model, faster fused backbones,
other CP/EP configurations, or combined FSDP CPU offload.

The current default of disabled async prefetch should remain. CPU Engram can
serve the capacity goal, but async should be enabled based on paired workload
measurements, not assumed to recover the cost of offload. Layer-0 placement
needs an earlier input-stage prefetch opportunity if its waiting is to be
hidden.

[Portable results](results/engram-prefetch-h20.json) contain every repetition,
loss comparison, deadline distribution and per-rank timing aggregation. Raw
reports and execution manifests are under `artifacts/engram-prefetch-20261012/`.

```bash
PYTHONPATH=.:tests python3 tests/run_memory_validation.py \
  --suite benchmark --ranks 8 --cp 2 --ep 4 \
  --cases baseline,cpu,cpu_async --repeats 3 --steps 15 --warmup 4 \
  --seq-len 512 --learning-rate 0.0001 --prefetch-steps 8 \
  --config configs/memory_eight_profile.json \
  --output-dir artifacts/prefetch-medium --timeout 300

PYTHONPATH=.:tests python3 -m unittest \
  test_engram_prefetch_probe test_memory_timeline test_memory_benchmark_summary -v

python3 tests/summarize_engram_prefetch.py artifacts/prefetch-medium \
  --output artifacts/prefetch-medium-summary.json
```

For the small test, use `configs/profile_memory_small.json` and sequence128.
For the boundary control, use `configs/profile_engram_first_layer.json`,
`--cases cpu,cpu_async --repeats 2`, and sequence512. Add `--trace-variants`
to capture GPU diagnostics after timing. Rendering existing traces supports
`tests/summarize_memory_timeline.py REPORT --phases training --output-dir DIR`
with Plotly installed; rendering does not rerun training.
