# DSV4 loading, offload, and sharded training state

The training entry point is `train_dsv4_checkpoint.py`. Runtime policies are
separate from `DeepSeekV41Config`; all performance switches default to disabled.
This implementation requires PyTorch 2.10 or later for the FSDP2 APIs used here.
The measured version is 2.10.0+cu129 on NVIDIA H20.
For a feature overview, cache lifecycle and the separate Qwen SFT workflow, see
the [Engram feature guide](engram_features.md).

## Policies and compatibility

| Option | Default | Effect |
| --- | --- | --- |
| `--load-mode` | `layerwise` | `direct_shard` fills final local shards from bounded source slices. |
| `--load-chunk-rows` | `131072` | Upper bound on rows per read; quantization alignment and staging size can reduce it. |
| `--load-pipeline` | off | Overlaps the next CPU read/decode with copying the current chunk. |
| `--staging-mib` | `256` | Loading budget and per-table pinned staging budget. |
| `--engram-device` | `cuda` | `cpu` keeps each owned table shard on the host. |
| `--engram-prefetch` | `sync` | `async` overlaps CPU lookup/H2D with backbone computation. Requires CPU tables. |
| `--engram-deduplicate` | off | Removes duplicate requests and restores their original order. Works with CPU or GPU tables. |
| `--engram-cache-mib` | `0` | Experimental GPU row cache budget per local table, with CPU or GPU source weights; independent of request deduplication. |
| `--fsdp-offload-layers` | empty | Comma-separated global layer IDs and half-open ranges, such as `1,3:6`. |
| `--fsdp-offload-targets` | `dense,experts` | Intersection of selected layers and parameter domains. Excludes Engram tables. |
| `--fsdp-offload-pin-memory` | off | Pins the CPU shards managed by FSDP. |
| `--optimizer-foreach` | off | Enables foreach for dense groups; sparse groups use the compatible path. |
| `--gradient-accumulation-steps` | `1` | Normalizes once after all microbatches, before the update. |

CPU Engram and cached tables support trainable sparse SGD with zero momentum and
weight decay. GPU dense AdamW remains available with caching disabled.
Parameter dtype is preserved and no persistent
FP32 master weights are created. Duplicate sparse rows are coalesced before SGD.
Low-precision coalescing and deduplication use temporary FP32 arithmetic. CPU
dense BF16/FP16 SGD also uses temporary FP32 arithmetic to match the CUDA dense
kernel; sparse CPU/CUDA updates use native SGD after coalescing.

LoRA remains supported on the baseline loading/GPU placement path. Combining
LoRA with direct loading, CPU Engram, or selective FSDP offload raises an explicit
error. Never call whole-model `.to(cuda)` after configuring CPU tables: the table
module rejects implicit relocation. Load into the intended placement instead.

CP and EP independently divide world size; EP must divide the expert count.
Sequence length must satisfy the model's CP constraints. CP need not equal EP.
Each CP group receives the same full input, using `rank // cp_size` for its data
generator seed. Ordinary parameters use world-wide FSDP; experts use replica
groups and their averaged gradients are divided by EP once per optimizer step.
Engram owner routing averages request gradients over the global row mesh.

## Examples

Baseline, on eight GPUs:

```bash
python3 -m torch.distributed.run --standalone --nproc-per-node=8 \
  train_dsv4_checkpoint.py --model-path /path/to/DeepSeek-V4.1-Flash \
  --num-layers 5 --cp-size 2 --ep-size 4 --seq-len 32 \
  --optimizer sgd --steps 3 --gradient-accumulation-steps 2
```

Enable each option independently to measure its effect. For example, adding
`--engram-device cpu` selects synchronous CPU lookup. Add
`--engram-prefetch async` to compare prefetch on the same workload. Use
`--fsdp-offload-layers 1,3 --fsdp-offload-targets experts` to offload only the
experts in global layers 1 and 3.

For cache experiments, add `--engram-cache-mib 8` to CPU Engram. Two local
tables then reserve at most 16 MiB of persistent GPU cache per rank, in addition
to transient query outputs. The budget is per table, not per model. The cache
can reduce transfers when requests repeat within gradient accumulation; it does
not guarantee higher throughput. See [cache measurements](engram_cache_results.md).
GPU-resident tables also accept a cache budget with `--optimizer sgd`; this
retains the complete local GPU table shard and adds cache storage. It is an
experimental comparison mode, not a way to reduce GPU table residency. The
[Qwen four-way comparison](qwen30b_engram_sft_results.md) measures both placements.

Combined configuration and checkpoint save:

```bash
python3 -m torch.distributed.run --standalone --nproc-per-node=8 \
  train_dsv4_checkpoint.py --model-path /path/to/DeepSeek-V4.1-Flash \
  --num-layers 5 --cp-size 2 --ep-size 4 --seq-len 32 \
  --optimizer sgd --steps 3 --gradient-accumulation-steps 2 \
  --load-mode direct_shard --load-pipeline --staging-mib 64 \
  --engram-device cpu --engram-prefetch async --engram-deduplicate \
  --fsdp-offload-layers 1,3 --fsdp-offload-targets dense,experts \
  --optimizer-foreach --output-dir outputs/offload \
  --metrics-file artifacts/offload.json
```

Resume with the same model window, world size, CP, EP, dtype, optimizer algorithm,
batch size, sequence length, and accumulation count. Set `--resume` and increase
`--steps`, which is the total target optimizer-step count. CPU/GPU placement,
loading strategy and prefetch policy may change. Optimizer hyperparameters and
state are restored from the checkpoint.

## Module boundaries

| Module | Contract |
| --- | --- |
| `dsv41_train/layout.py` | Logical names, global shapes, local offsets, expert/table IDs, and local byte accounting. No communication or updates. |
| `dsv41_train/loading/` | Indexed reader, row/block decoders, source slices and materializer protocol. No DSV4 weight names. |
| `models/dsv4/weights.py` | DSV4 names, gate/up concatenation, grouped projection mapping. |
| `models/dsv4/materialize.py` | Meta FSDP wrapping, final local allocation, per-domain placement, and wrapping lifecycle. |
| `models/dsv4/engram/storage.py` | Differentiable local row reads; CPU staging has no process-group knowledge. |
| `models/dsv4/engram/cache.py` | Bounded owner-local GPU rows, CPU tags, and update invalidation; no collectives or optimizer updates. |
| `models/dsv4/engram/routing.py` | Owner requests, response exchange, inverse mapping and global gradient averaging. |
| `models/dsv4/engram/table.py` | Sole parameter registration and single-use lookup tickets. |
| `models/dsv4/policies.py` | Selective FSDP policy and once-per-step gradient normalization. |
| `dsv41_train/optimization.py` | Dense/sparse parameter groups, placement-compatible SGD and ordinary AdamW. |
| `dsv41_train/sharded_state.py` | DCP adapters for logical parameter and optimizer state shards. |

Existing model forward signatures, parameter keys and loading imports remain
available. The legacy `layer_loaded` callback remains supported and is mutually
exclusive with `materializer`. Direct loading requires a materializer; duplicate
FSDP wrapping raises an error.

Direct loading wraps meta parameters first and reads only the source regions
needed by the final local tensor, allowing quantization-block alignment reads.
The embedding and LM head are independently sharded early. Tables destined for
CPU are decoded directly into their local CPU storage. Staging must hold at
least one aligned decode block; an undersized budget fails explicitly. The
pipeline holds at most the current chunk and one future read. It never constructs
a complete decoded layer.

CPU lookup retains autograd through the device copy. With caching disabled,
only returned rows live on the GPU; backward sends their gradients to the sparse
CPU table. Prefetch exchanges IDs and enqueues their pinned D2H copy on the main
thread. Workers wait for the ID event, then perform local lookup and H2D on
independent copy streams. Response collectives run on the main thread in layer
order; implicit cross-layer FSDP backward prefetch is disabled. There is at most
one outstanding ticket per table. Pinned buffers are bounded per active table,
so the aggregate budget scales with the number of tables being prefetched.
Pinned row buffers are reused after their DMA completes. By default, resolved
values are not cached across optimizer steps. `close()` drains workers and
invalidates pending tickets, including exception cleanup.

The optional row cache has fixed GPU slots and bounded CPU tag metadata. It
caches the owner's rows, so request and response NCCL collectives still run.
Cache hits skip CPU row gathering and value H2D; misses fill slots. Collisions
evict previous rows. Each lookup returns its own row tensor, so later eviction
cannot change values retained for backward or activation recomputation.
Repeated misses share transferred values but preserve every gradient occurrence
and its order; this is separate from router deduplication.

The source table parameter, on CPU or GPU, remains authoritative and is the only
registered table parameter. For GPU source tables, cache misses gather rows from
the GPU shard; CPU tag management and cache metadata transfers still add overhead.
The optimizer invalidates touched cached rows after sparse SGD, while unmodified
rows can survive the step. Unrecognized parameter version changes invalidate
all rows. Parameter reload, placement/dtype changes and `close()` clear the cache.
Caches are omitted from checkpoints. Direct untracked writes through `.data`
must be followed by `table.clear_cache()`; ordinary `copy_` and supported state
loading invalidate automatically. Cache use does not change the supported
optimizer algorithms, gradient averaging, or accumulation rules.

## Checkpoint format

Format version 2 stores global layer/expert names, each table's true global row
offset, the topology/layout signature, configuration, parameter shards, optimizer
state schema and tensors, CPU/CUDA RNG, per-rank data-generator state and step.
Uneven Engram row partitions use custom DCP chunk metadata. No full model is
gathered for saving or loading. The optimizer schema restores only state that
exists, including parameters unused by AdamW; stateless SGD restore creates no
dummy gradients or optimizer updates.

The training entry saves only after an optimizer step. Both optimizer and save
boundaries reject unconsumed lookup tickets. Checkpoint restore validates format,
topology, layout, model/training configuration and optimizer type before writing
model storage. Missing metadata or unsafe legacy layouts are rejected. Topology
resharding is deliberately unsupported in this version.

## Metrics and acceptance

Metrics include load/train allocated peaks, reserved CUDA memory, process maximum
RSS, encoded logical bytes read, local weight/gradient/optimizer bytes by domain,
Engram row H2D/D2H bytes, resolve wait and global unique-input tokens/s. DTensor
global shapes are never used for physical byte accounting. Engram transfer
counters describe row payloads; they exclude routing IDs and NCCL/FSDP traffic.
`resolve_wait_seconds` includes the host time spent issuing response routing, so
it is not a pure CUDA event measurement of waiting for the copy stream.

See [validation commands](memory_validation.md) and
[measured results](memory_results.md). The GPU tests use real FSDP2/NCCL. The
performance workload is a synthetic quantized checkpoint, not a downloaded full
DeepSeek checkpoint. Its capacity/throughput does not establish full-model
capacity, quality, or multi-node scaling. HF-reference checks remain conditional
on the unavailable reference implementation.
