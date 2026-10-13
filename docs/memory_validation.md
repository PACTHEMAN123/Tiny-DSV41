# Loading and offload validation

Run from the repository root with Python 3.10+ and PyTorch 2.10+. Set
`PYTHONPATH=.:tests`. CUDA tests require visible GPUs and working NCCL. The
validation runner uses a fresh process group per case and terminates timed-out
workers, avoiding stale device-mesh caches between topologies.

## Unit tests

```bash
python3 -m unittest discover -s tests -v
```

The memory tests cover quantized fusion and grouped shapes, partial source
blocks, tail shards, global layer windows, bounded decode refusal, sparse SGD,
duplicate and empty requests, ticket reuse/cancellation/errors, implicit device
relocation rejection, checkpoint schema/configuration validation and next-update
restoration. They also check that stateless SGD checkpointing creates neither
gradients nor optimizer updates.

## CP/EP matrices

All sixteen legal combinations for a six-rank, six-expert fixture:

```bash
python3 tests/run_memory_validation.py --suite cpu --ranks 6 \
  --all-topologies --output-dir artifacts/memory-cpu

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
python3 tests/run_memory_validation.py --suite cuda --ranks 6 \
  --all-topologies --output-dir artifacts/memory-gpu
```

Each case checks initial loaded weights, padded outputs, externally scaled
losses, gradients, three SGD updates with two microbatches each, activation
recomputation and restoration into a fresh model. The CPU cases use explicit
reductions; the CUDA cases use actual FSDP2 and NCCL.

Two-GPU policy combinations and six-GPU intersecting groups in BF16:

```bash
CUDA_VISIBLE_DEVICES=0,1 python3 tests/run_memory_validation.py \
  --suite cuda --bf16 --output-dir artifacts/memory-bf16

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
python3 tests/run_memory_validation.py --suite cuda --ranks 6 --bf16 \
  --output-dir artifacts/memory-cross-bf16
```

Omit `--bf16` for an FP32 unsharded mathematical reference. BF16 compares with a
GPU-resident, same-topology, layerwise-loaded baseline to isolate placement and
loading differences from distributed reduction rounding. The two-GPU cases
cover baseline, GPU deduplication, synchronous CPU/FSDP offload, asynchronous CPU
deduplication, and their combination with activation checkpointing. Six-GPU
cases specifically cover CP2/EP3 and CP3/EP2.

Eight-rank fixtures use eight experts and length 16, so the complete matrix
includes every CP/EP pair in `{1,2,4,8}`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python3 tests/run_memory_validation.py --suite cuda --ranks 8 \
  --all-topologies --output-dir artifacts/memory-eight/fp32

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python3 tests/run_memory_validation.py --suite cuda --ranks 8 --bf16 \
  --keep-going --output-dir artifacts/memory-eight/bf16
```

The eight-rank BF16 suite has six selected placement/topology cases. Its current
failures and diagnostics are documented in [memory_eight_results.md](memory_eight_results.md).
`--match-reference-dedup` on `dsv4_memory_parallel.py` is an isolation diagnostic;
it does not replace the default non-deduplicated reference.

## Lifecycle and fresh-process resume

```bash
CUDA_VISIBLE_DEVICES=0 python3 tests/dsv4_engram_offload_cuda.py

CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.run \
  --standalone --nproc-per-node=2 tests/dsv4_memory_resume.py \
  --phase save --folder artifacts/memory-resume

CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.run \
  --standalone --nproc-per-node=2 tests/dsv4_memory_resume.py \
  --phase resume --folder artifacts/memory-resume
```

Use `--optimizer adamw` on both resume commands with a separate folder to test
sharded Adam moments. Save runs with selected FSDP CPU placement; resume runs with
GPU placement. SGD also changes CPU asynchronous/deduplicated tables to GPU
synchronous tables. The next update, dropout, CPU/CUDA random draws, and data
generator state must agree. The fixture compares only corresponding local
shards and never gathers a complete model.

The lifecycle test runs 24 updates per dtype/prefetch/deduplication combination,
with repeated IDs, empty requests, small pinned buffers and no retained optimizer
state. It checks output/weight equality and post-warmup CUDA allocation stability.

## Performance

```bash
CUDA_VISIBLE_DEVICES=0,1 python3 tests/run_memory_validation.py \
  --suite benchmark --output-dir artifacts/memory-benchmark
```

Ten variants independently measure baseline, direct loading, CPU tables,
asynchronous lookup, CPU/GPU deduplication, dense/expert FSDP offload, dense
foreach and the combined policy. Each uses identical quantized weights, inputs,
two warmup steps, five measured updates and two microbatches. Run benchmarks
separately from correctness tests to avoid host contention. Reports contain all
rank measurements and policies. CPU RSS includes fixture construction; it is a
process high-water mark, not isolated loader allocation. Logical reads measure
requested encoded bytes, not physical disk I/O after filesystem caching.

For a larger shared fixture, repeated trials and CUDA/worker-thread profiling,
see the [eight-rank measurement commands](memory_eight_results.md#resume-commands).
`--config` constructs the shared checkpoint outside benchmark workers;
`--repeats` shuffles variant order reproducibly. `--trace-variants` collects
separate post-timing traces on the first repetition. Summarize the folder with
`tests/summarize_memory_benchmark.py`; retain both the manifest and run summary
to identify unfinished trials.

To diagnose weight loading, run `tests/dsv4_memory_benchmark.py` under the same
`torch.distributed.run` command with `--load-trace-dir <directory>`. It records
CPU decode/read/fill regions, background workers and CUDA copies in per-rank
traces. This must be a separate diagnostic invocation: its `load_seconds`
includes profiler overhead, and the report sets `load_profiled=true`. The
summarizer rejects mixing those loads with unprofiled timing runs. Training
`--trace-dir` continues to capture a separate sample after timed updates.
Loading shape capture is disabled because the profiler's tensor references
interfere with FSDP's meta-to-local storage swap; region and copy metrics remain
available.

When GPU execution is unavailable, the [simplified CPU component profile](memory_component_profile.md)
measures loader read/decode/fill work and local Engram execution in fresh
processes. Its CPU results and any historical CUDA context are explicitly
separated; it does not substitute for offload DMA or distributed acceptance.

The [loading review](loading_review.md) records the subsequent loader fixes,
CPU before/after measurements and actual two-GPU timing and transfer traces.

Run the LoRA/full-training FSDP regression with:

```bash
CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.run \
  --standalone --nproc-per-node=2 tests/dsv4_post_training_fsdp.py
```

Qwen regression commands remain in [testing.md](testing.md). The retained HF
tests explicitly skip when their reference is absent; this is not a successful
HF parity result.
