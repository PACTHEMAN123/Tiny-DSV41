# Eight-rank memory baseline

The later [Engram prefetch measurements](engram_prefetch_results.md) add
per-lookup readiness, a 423M-parameter workload, a layer-0 boundary control,
and paired GPU timelines.

Measured on 2026-10-10. This supplies the missing GPU-resident baseline for the
small eight-H20 CP2/EP4 workload and remeasures offload policies in the same
session. All 12 fresh-process runs completed successfully. No production code
was changed.

## Workload and method

- NVIDIA H20, PyTorch 2.10.0+cu129, world8/CP2/EP4, BF16 storage, sparse Engram
  SGD and learning rate 0.0001. The SGD optimizer has no momentum state.
- Same [14.4M-parameter synthetic model](../configs/profile_memory_small.json)
  as the September profiling run: five layers, hidden128, eight experts and
  two 65539/65543-row Engram tables with width64.
- Same weights and input seeds across policies, sequence128, two microbatches,
  two warmup updates and five measured updates. Four policies, three fresh
  process groups per policy, reproducibly shuffled order.
- All loading and training measurements have profiling disabled. The older
  `load_profiled=true` diagnostic run is not mixed into these samples.
- Table entries are medians over three runs. For memory, each run first takes
  the maximum over its eight ranks. Every policy's loading and training peak
  was identical across all three repetitions.

The checkpoint file SHA-256 is
`b78aecce8037d7ce0230b1d7e28f42d0b19388dbb282f739bf614f7115e859c4`, identical
to `artifacts/loading-review-cpu-after/fixture/model.safetensors` used for the
earlier profiling run. Model configurations also match exactly. This comparison
uses five timed updates instead of the earlier diagnostic run's three.

## Memory results

| Policy | Train allocated peak MiB | Saved MiB | Reduction | Load allocated peak MiB | Train reserved peak MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| GPU resident baseline | 89.622 | 0 | 0% | 9.257 | 132 |
| CPU Engram, synchronous | 87.414 | 2.208 | 2.46% | 6.976 | 136 |
| CPU Engram, asynchronous | 87.407 | 2.215 | 2.47% | 6.976 | 136 |
| Combined | 85.963 | 3.659 | 4.08% | 0.782 | 134 |

The baseline uses layerwise loading, GPU Engram without deduplication and GPU
FSDP placement. CPU-only policies change Engram placement and, for async, its
lookup scheduling. Combined also uses direct pipelined loading, deduplication,
dense foreach and FSDP CPU offload for dense/expert domains on layers 1 and 3.

Combined reduces loading peak by **91.56%**, from 9.257 to 0.782 MiB. This is
distinct from its **4.08% training-peak reduction**. The combined training peak
reproduces the earlier single-run 85.963 MiB result.

Each rank already holds only about **2 MiB of Engram weights** because the
roughly 16 MiB global tables are row-sharded across eight ranks. CPU Engram
moves those local table parameters completely off GPU. The earlier two-rank
run had about 8 MiB of table weights per rank, so its 6.9% training reduction
does not carry over to this topology. Activations, other weights, gradients and
communication buffers still contribute to the training peak.

Async and sync memory are effectively the same: their peaks differ by only
6.5 KiB in this run. Placement provides the capacity saving; async lookup aims
to hide latency rather than remove additional table storage.

These figures are PyTorch allocator measurements within the benchmark worker,
not the whole device or the process total reported by `nvidia-smi`. Reserved
memory did **not** decrease: the allocator kept 132 MiB for the baseline,
136 MiB for the CPU-table policies and 134 MiB for combined. The allocated
reduction must not be reported as the same reduction in driver-level usage.

## Timing and validation

| Policy | Tokens/s median [range] | First load median ms |
| --- | ---: | ---: |
| GPU resident baseline | 2817.63 [2787.29, 2969.98] | 978.81 |
| CPU Engram, synchronous | 2904.02 [2868.19, 2924.90] | 1025.65 |
| CPU Engram, asynchronous | 2805.05 [2784.28, 2868.20] | 973.79 |
| Combined | 2799.43 [2757.89, 2980.30] | 991.36 |

These timings are descriptive only. GPU 2 and GPU 6 held existing inference
services at preflight, with about 9 GiB and 16 GiB free respectively and zero
GPU utilization throughout the preflight samples. The small benchmark used
the available capacity without modifying those services. Postflight showed
only the existing GPU 2 allocation; other workload occupancy changed during
the session. The run therefore is not exclusive-GPU performance acceptance,
and no throughput improvement is established from these overlapping ranges.
No benchmark workers remained in the postflight GPU process listing.

All reports contain eight ranks and five finite measured losses per rank.
Every policy's recorded loss array matches its corresponding baseline exactly
in all three repetitions. That is a check of this low-learning-rate workload,
not a full gradient/parameter parity test and not resolution of the earlier
[eight-rank BF16 failures](memory_eight_results.md). Full-size released weights
and the separate 423M-parameter benchmark have not been tested here.

## Artifacts and reproduction

[Portable results](results/memory-eight-baseline-20261010.json) contain sample
distributions, loss comparisons, all policies' trials and environment caveats.
Raw per-rank reports, logs, the execution manifest and shared fixture are in
`artifacts/memory-eight-baseline-20261010/`.

Using the CUDA-enabled project environment, and a new directory for another
independent measurement:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=.:tests python3 tests/run_memory_validation.py \
  --suite benchmark --ranks 8 --cases baseline,cpu,cpu_async,combined \
  --repeats 3 --steps 5 --warmup 2 --seq-len 128 --cp 2 --ep 4 \
  --learning-rate .0001 --config configs/profile_memory_small.json \
  --timeout 300 --output-dir artifacts/memory-eight-baseline-20261010

PYTHONPATH=tests python3 tests/summarize_memory_benchmark.py \
  artifacts/memory-eight-baseline-20261010 \
  --output artifacts/memory-eight-baseline-20261010/summary-portable.json
```

The versioned JSON additionally records the measured environment under
`measurement_context`; the generic summarizer does not infer hardware sharing.
