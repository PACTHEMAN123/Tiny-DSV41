# Eight-GPU profiling availability recheck

[Offline timelines and component timing](memory_timeline.md) are now available
for this capture, with eight-rank overview, exact rank-0 events and CSV/JSON.

2026-10-10 follow-up: the [matched eight-rank baseline](memory_eight_baseline.md)
is complete. Three runs per policy measured 89.62 MiB for the resident baseline
and 85.96 MiB for combined, a 4.08% reduction in training allocated peak. That
comparison uses new unprofiled runs and documents the shared GPU environment.

2026-09-29, approximately 19:45 Asia/Shanghai. Host GPU execution is available:
all eight NVIDIA H20s were idle at the preflight check. The sandbox still cannot
communicate with the driver; the external execution path worked. No production
code was changed for this recheck.

## Completed run

- Eight ranks, CP2/EP4, PyTorch 2.10.0+cu129, BF16, SGD at 0.0001.
- Reduced synthetic model: five layers, hidden128, eight experts, two Engram
  tables; the same `configs/profile_memory_small.json` fixture as the review.
- Combined policy: direct pipelined loading, asynchronous CPU Engram,
  deduplication, selected dense/expert FSDP CPU offload and dense foreach.
- Two warmup updates, three timed updates, two microbatches, sequence128;
  one additional training update captured by the profiler.
- Process exit code 0. All eight rank reports are present and losses are finite.
  Each rank has a loading trace, training trace and both activity summaries.
  All training traces contain compute, NCCL, H2D and D2H events; all loading
  traces contain H2D and Engram decode events.

The single run recorded 2416.94 tokens/s and a maximum training allocation of
85.96 MiB across ranks. This is an availability check without repeated trials
or a paired baseline, so it does not establish a speedup. Loading was profiled
and `load_profiled=true`; its duration must not be compared with normal loads.

## Diagnostic sample

Ranges below are across eight ranks for one separately profiled training step:

| Activity | Per-rank measurement |
| --- | ---: |
| Compute kernels | 17,052-17,115 calls, 26.62-27.37 ms |
| H2D DMA | 0.557-0.628 ms |
| D2H DMA | 0.607-0.662 ms |
| NCCL kernel durations | 61.59-242.89 ms |
| All-H2D overlap with compute | 0.30%-3.60% |

Engram row counters from the timed updates report 3,126,016 bytes per step in
each direction across all ranks. The slowest rank's resolve host time averages
1.50 ms per step, including row routing and reconstruction. CUDA activity sums
include overlapping streams and profiler-induced rank skew. NCCL time is not
a bandwidth measurement, and overlap excludes NCCL kernels from compute.

This successful profiling run does not resolve the separately recorded
[eight-rank BF16 parity failures](memory_eight_results.md), nor does it run the
larger 423M-parameter workload proposed in that document.

## Artifacts and reproduction

All outputs are under `artifacts/gpu-profile-recheck-20260929-194457/`:
`report.json`, `loading/rank-{0..7}.json`, `training/rank-{0..7}.json`, and
corresponding `rank-{0..7}-summary.json` files. The sibling `.log` records the
process output. The Chrome traces can be opened in Perfetto.

With the project's CUDA-enabled Python environment:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=.:tests python3 -m torch.distributed.run \
  --standalone --nproc-per-node=8 tests/dsv4_memory_benchmark.py \
  --variant combined --model-path artifacts/loading-review-cpu-after/fixture \
  --steps 3 --warmup 2 --seq-len 128 --cp 2 --ep 4 --learning-rate .0001 \
  --load-trace-dir artifacts/gpu-profile-recheck-20260929-194457/loading \
  --trace-dir artifacts/gpu-profile-recheck-20260929-194457/training \
  --output artifacts/gpu-profile-recheck-20260929-194457/report.json
```
