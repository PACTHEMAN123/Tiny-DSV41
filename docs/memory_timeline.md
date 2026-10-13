# GPU timelines and component timing

Generated on 2026-09-30 from the successful 2026-09-29 eight-H20 CP2/EP4 capture.
The reduced model uses the combined policy: direct loading, asynchronous CPU
Engram, deduplication, selected FSDP CPU offload and dense foreach. No new GPU
measurement was needed to produce these views.

## Artifacts

- [Eight-rank training timeline](../artifacts/gpu-timeline-20260930/timeline.html):
  CPU stages and per-rank compute/NCCL/H2D/D2H/D2D activity on a shared clock.
- [Rank 0 training detail](../artifacts/gpu-timeline-20260930/training-rank-0.html):
  exact GPU kernel/copy intervals per CUDA stream and CPU Engram scopes per
  thread, with hover details and zoom.
- [Rank 0 loading detail](../artifacts/gpu-timeline-20260930/loading-rank-0.html):
  reading, decoding, filling, executor activity and CUDA copies.
- [Timing report](../artifacts/gpu-timeline-20260930/README.md),
  [CSV](../artifacts/gpu-timeline-20260930/time-breakdown.csv),
  [JSON](../artifacts/gpu-timeline-20260930/time-breakdown.json): all eight ranks,
  mean/range, per-component forward/backward attribution and loading domains.

HTML files embed Plotly and data, open offline and require no development
server. The detailed training chart is approximately 15 MiB. Original Chrome
JSON traces remain under `artifacts/gpu-profile-recheck-20260929-194457/` and
can be opened in Perfetto for complete operator nesting and flow arrows.

## Collection and accounting

`torch.profiler.profile` records CPU and CUDA activity through Kineto/CUPTI.
`record_function` labels training stages and model boundaries. All-thread
collection includes the Engram and weight-loading workers. The separate
`loading_regions` instrumentation records reads, FP8/FP4 decode, parameter fill,
assignment, tokenizer setup and executor activity. Loading disables shape
capture because retained tensor references conflict with FSDP meta-storage
swaps; training keeps shape recording.

`summarize_profiler` attributes operators to their nearest model annotation.
Backward attribution uses autograd forward sequence/thread IDs. Unmatched
operations remain explicitly unattributed. The generated timing report retains
both self CPU time and linked CUDA time; the latter includes associated
communication/copies and is not pure arithmetic time.

The timeline exporter reads the already recorded Chrome events. It selects
CPU `user_annotation` scopes only, excluding same-named `gpu_user_annotation`
events to avoid counting a host stage twice. Device `kernel` events are split
into compute and NCCL; `gpu_memcpy` events are split by direction. Clock bases
are accounted for before combining ranks on this single host. Loading and
training are separate captures, with separate origins.
Each chart starts at its first selected event; uninstrumented framework setup
before that event is not drawn as a model region.

The overview unions concurrent intervals within each activity category before
binning at 0.25 ms. Each cell shows the fraction of that time bin occupied by
the activity. The detail charts retain exact endpoints. Different categories
can overlap, so their busy fractions cannot be summed into utilization.

## Observed timings

Each profiled training step contains two microbatches. Values are means across
eight ranks; the downloadable report includes the range and individual ranks.

| Host stage | Inclusive ms per step |
| --- | ---: |
| Forward | 236.760 |
| Backward | 337.852 |
| Gradient normalization | 0.991 |
| Optimizer | 4.148 |
| Zero gradients | 2.259 |

The profiled CPU stage window is 581.09-582.59 ms per rank. The separate
unprofiled updates took 416.79-430.65 ms; trace collection adds substantial
overhead. Traced durations must not replace the unprofiled throughput samples.

| GPU activity | Summed event ms per step |
| --- | ---: |
| Compute kernels | 27.091 |
| NCCL kernels, including waits | 172.868 |
| H2D | 0.601 |
| D2H | 0.632 |
| D2D | 0.280 |

NCCL totals vary from 61.59 to 242.89 ms by rank. They include rank skew and
waiting and do not demonstrate bandwidth saturation. These event sums also
overlap across streams and are not an additive step-time decomposition.

Engram CPU scopes include ID routing at 6.420 ms, local lookup/staging at
2.526 ms, resolve at 2.174 ms, hash at 1.613 ms and projection/gating forward at
3.423 ms. Prefetch/resolve contain child scopes, and workers run concurrently;
these figures must not be summed as independent wall time.

The loading trace shows dense/Engram/expert decode means of
4.201/5.400/5.904 ms, encoded reads totaling 8.591 ms of region time, and GPU H2D
of 0.117 ms per rank. Framework initialization and FSDP setup are not covered
by those load subregions; the JSON retains their unattributed profiler time.

This analysis does not resolve the previously recorded eight-rank BF16 parity
failures or establish a throughput improvement over a baseline.

## Reproduction

Plotly is needed only for offline rendering; the training environment and
production dependencies are unchanged. The renderer used an isolated install
under `/tmp/dsv41-timeline-plotly`.

```bash
PYTHONPATH=/tmp/dsv41-timeline-plotly python3 \
  tests/summarize_memory_timeline.py \
  artifacts/gpu-profile-recheck-20260929-194457/report.json \
  --output-dir artifacts/gpu-timeline-20260930 --bin-ms .25

PYTHONPATH=tests python3 -m unittest test_memory_timeline -v
```

See [the GPU recheck](gpu_profile_recheck.md) for the eight-rank collection
command. Three accounting tests cover duplicate CPU/GPU annotations, concurrent
stream union and partial bins, and separation of loading domains.
All six timeline/benchmark-summary tests passed. A headless browser verified
offline rendering and zoom for all three HTML files without JavaScript errors;
PNG screenshots are saved next to the HTML artifacts.
