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

Run the LoRA/full-training FSDP regression with:

```bash
CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.run \
  --standalone --nproc-per-node=2 tests/dsv4_post_training_fsdp.py
```

Qwen regression commands remain in [testing.md](testing.md). The retained HF
tests explicitly skip when their reference is absent; this is not a successful
HF parity result.
