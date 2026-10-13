# Simplified loading and Engram profile

Follow-up: [loading review and optimization](loading_review.md) records the
implemented fixes, fresh CPU before/after runs and actual two-H20 profiling
after GPU access recovered. The measurements below describe the earlier run.

Measured on 2026-09-29. Current results are **CPU-only**: the sandbox cannot
access the NVIDIA driver, and the external GPU execution approval service again
failed with `stream disconnected before completion`. No current H2D/D2H, NCCL,
FSDP or GPU overlap measurement is claimed. The historical CUDA section below
is explicitly separate.

## Workload and method

- Intel Xeon Platinum 8469C, PyTorch 2.10.0+cu129, one intra-op CPU thread.
- 14,398,338 parameters, five layers, hidden size 128, eight routed experts,
  top-2 routing, MHC with 20 Sinkhorn iterations and compressed attention.
- Two Engram tables, 65539/65543 rows by 64 columns; BF16 parameter storage,
  sparse Engram SGD, learning rate 0.0001, sequence length 128, two microbatches.
- Same quantized synthetic checkpoint and input batches for every variant.
  Three warmup updates, ten timed updates, three fresh processes per variant,
  randomized order: 18 completed runs. All recorded losses matched the baseline
  exactly on this CPU workload. This does not resolve the eight-rank BF16 issue.
- Timing is collected without the profiler. The first repetition subsequently
  collects a separate training trace and an additional load trace. First-load
  time includes fresh-process framework initialization. Warm-load time is a
  later unprofiled reload in the same process. Filesystem pages may be cached;
  neither number is a cold-disk measurement.

[Configuration](../configs/profile_memory_small.json),
[portable measurements](results/memory-components-cpu.json), and raw traces in
`artifacts/memory-components-cpu-final/traces/<variant>/`.
The direct loader uses a local CPU materializer without FSDP, so every logical
tensor is local. This diagnoses loading overhead, not distributed shard savings.

## Unprofiled timings

Values are medians over three fresh processes; brackets give their range.
Training-stage values are medians of each process's ten measured updates.

| Variant | First load ms | Warm load ms | Step ms | Tokens/s |
| --- | ---: | ---: | ---: | ---: |
| Layerwise, synchronous | 540.65 | 116.02 [110.42, 116.75] | 274.87 [269.36, 300.34] | 922.04 |
| Direct, synchronous | 514.90 | 126.77 [122.49, 136.58] | 277.14 [269.39, 332.11] | 887.97 |
| Direct, pipelined loading | 575.88 | 185.24 [181.16, 187.75] | 277.04 [276.96, 278.13] | 918.08 |
| Async CPU lookup | 493.88 | 118.63 [115.88, 122.38] | 282.36 [277.86, 296.61] | 900.14 |
| Deduplicated CPU lookup | 493.98 | 116.00 [114.98, 117.13] | 300.05 [285.77, 304.87] | 855.34 |
| Dense foreach | 488.62 | 113.45 [113.17, 119.00] | 286.27 [280.71, 287.41] | 890.88 |

The baseline spends approximately 106.94 ms in forward, 158.21 ms in backward,
and 9.44 ms in the optimizer: about 39%, 58% and 3% of the step. Separate medians
need not sum exactly. Throughput uses total measured elapsed time, so it is not
the reciprocal of median step time. The direct variant's first repetition had
a longer step-time tail; its throughput range was 684.30-940.80 tokens/s.

## Loading findings

The baseline load trace lasts 138.06 ms, compared with its 110.42 ms unprofiled
warm reload in that process. The following are instrumented region wall times,
not the unprofiled loader's exact time distribution:

| Region | Time ms | Share of profiled load | Calls |
| --- | ---: | ---: | ---: |
| Engram FP8 row decode | 42.38 | 30.7% | 34 |
| Expert FP4 decode | 36.76 | 26.6% | 120 |
| Dense FP8 block decode | 12.11 | 8.8% | 46 |
| All tensor reads/clones | 21.27 | 15.4% | 485 |
| Parameter assignment | 4.13 | 3.0% | 10 |
| Tokenizer mapping | 1.70 | 1.2% | 1 |

The decode regions total 91.26 ms, about 66% of this trace. Logical encoded
reads are 14.13 MiB; Engram accounts for 9.00 MiB. Engram decode emits 16.00 MiB
of BF16 rows. `aten::copy_` on the `[4096,2,32]` conversion path alone accounts
for 35.16 ms of CPU self time. These copies include dtype conversions and
temporary CPU tensors; they are not H2D transfers. This points first to decode
and intermediate-memory costs under cached-input conditions, not disk bandwidth.

Pipelined loading is 46.1% slower than synchronous direct loading in the warm
median. Its trace creates **213 thread pools**, one per populated
`WeightSource.fill`, with 673 init/submit/shutdown calls taking 49.91 ms in total.
There are only 247 submitted read tasks. Most weights are too small to amortize
that setup. Fill, worker-decode and executor spans overlap and must not be added
together as separate wall-time contributions.

The first loader improvement to test is a loader-lifetime executor and a minimum
work threshold for pipelining. Next, benchmark fused/vectorized FP8/FP4 decode
and fewer intermediate copies, especially for CPU-resident Engram. The current
capacity benefit of direct loading remains a separate distributed property.

## Engram findings

Baseline forward/backward Engram regions, hash, lookup and reconstruction total
17.91 ms of CPU self time, 4.6% of the 388.85 ms instrumented CPU event total.
Projection/gating and their backward account for 15.84 ms of that; the actual
four local table lookups total only 0.39 ms of inclusive trace time. Hashing is
0.88 ms. Parent/child inclusive spans are not additive.

The larger attributed training costs are routed-expert compute at 77.54 ms
(19.9%) and MHC at 66.87 ms (17.2%). Another 81.07 ms (20.8%) remains under
unattributed backward/autograd. It is not assigned to a model operator here.
These CPU trace shares are not GPU utilization estimates.

Deduplication reduces requests from 6144 to 6068 rows per step, only **1.24%**.
The equivalent row payload is 0.75 MiB versus 0.741 MiB per direction at BF16,
but no such device transfer occurs in this CPU run. Unprofiled resolve host time
rises from 0.055 to 0.724 ms per step because it includes inverse reconstruction;
that metric is not pure I/O wait. Deduplication shows no stable throughput benefit
on these low-repetition random inputs. Natural-text hit rates need separate data.

Async lookup also has no stable improvement here: there is no H2D transfer to
hide, and the synchronous local lookup is already small. The background lookup
is captured correctly, including workers created during warmup. CPU BF16 dense
SGD uses the explicit opmath path before native SGD, so the foreach comparison
does not exercise the same dense-update path as CUDA and cannot predict its gain.

## Transfer paths to measure on GPU

Code inspection identifies several plausible latency costs. Their current CUDA
cost and overlap have not been measured in this run:

| Location | Current behavior | Measurement/optimization target |
| --- | --- | --- |
| `loading/source.py:76` | Executor created inside each parameter fill | Reuse executor; avoid tiny pipeline tasks |
| `loading/source.py:72` | Each decoded block is pinned before copy | Reuse bounded pinned staging buffers |
| `loading/source.py:84` | Current CUDA stream synchronized after every chunk | Event-governed buffer reuse; measure actual decode/copy overlap |
| `engram/storage.py:21` | Two pinned buffers allocated per lookup | Persistent bounded buffer pool with ticket ownership |
| `engram/storage.py:36` | Worker waits for the final DMA events | Measure host stalls; preserve buffer lifetime when changing this |
| `engram/storage.py:42` | Gradient copied synchronously back to CPU | Attribute backward D2H wait separately |
| `engram/routing.py:63` | GPU min/max used in Python bounds check | Measure scalar synchronization overhead |
| `engram/routing.py:80` | Send/receive counts converted to Python lists | Measure collective launch latency plus count synchronization |
| `engram/table.py:70` | Query IDs copied to CPU before worker submission | Separate ID wait from row lookup/H2D |

These waits currently protect correctness and buffer lifetime. Simply deleting
them is not a valid optimization. The first GPU trace should separate the small
ID/control transfers, row payload DMA, forward wait and synchronous backward.

## Historical CUDA context

The portable JSON also re-analyzes an earlier single-H20 resident-model trace:
five layers, hidden1024, 32 experts, sequence512, AdamW, earlier code revision.
It is a different workload and is not a current offload measurement.

That run had an unprofiled median step of 192.94 ms. Its separate trace contains
20,954 compute kernels totaling 64.42 ms. Routed-expert forward/backward kernels
sum to 20.35 ms and MHC to 6.23 ms; Engram forward/backward/hash/lookup sum to
approximately 0.54 ms, excluding its optimizer share. This supports investigating
many small expert/MHC operations as a GPU bottleneck candidate, not treating
local Engram lookup as the only cost.

The same trace has 172 D2H copies totaling only 1945 bytes and 0.415 ms of DMA
time, consistent with many scalar/control transfers. This suggests examining
synchronization latency. It does not measure CPU-table row transfers. Kernel
sums divided by unprofiled step time would not be a valid GPU utilization metric.

## Reproduction and verification

```bash
PYTHONPATH=.:tests python3 tests/profile_memory_components.py \
  --output-dir artifacts/memory-components-cpu-final \
  --config configs/profile_memory_small.json --seq-len 128 \
  --threads 1 --warmup 3 --steps 10 --repeats 3

PYTHONPATH=.:tests python3 tests/summarize_memory_components.py \
  artifacts/memory-components-cpu-final \
  --output docs/results/memory-components-cpu.json \
  --historical-cuda artifacts/architecture-profile/base-s512-full-none-loop-gpu1.json

PYTHONPATH=.:tests python3 -m unittest test_profiling -v
```

All 18 profiling runs completed and all five profiling regression tests passed,
including identical loaded weights, exact logical-read accounting across worker
threads, unchanged gradients, and hook restoration on exceptions. Production
loading, transfer and optimizer behavior was not changed by this profiling work.
