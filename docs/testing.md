# Testing

Run all commands from the repository root.

Loading/offload policy tests, the complete six-rank CP/EP matrix, fresh-process
resume and performance commands are in [memory validation](memory_validation.md).
Engram prefetch deadline and throughput comparisons are in
[prefetch measurements](engram_prefetch_results.md).
GPU row cache coherence, transfer accounting and throughput measurements are in
[cache measurements](engram_cache_results.md).

## Unit tests

```bash
python3 -m unittest discover -s tests -v
```

The Qwen checkpoint tests are skipped when the optional `safetensors` package
is not installed.

## DSV4 post-training checks

The unit suite covers AC output/gradient parity (including dropout, padding,
CSA1/2 shared state), loss scaling, gradient accumulation, frozen LoRA weights,
adapter merge/export/reload, layerwise loading, and checkpoint/RNG resume.

```bash
PYTHONPATH=.:tests python3 -m unittest test_post_training test_dsv4_loading -v

PYTHONPATH=.:tests CUDA_VISIBLE_DEVICES=0 python3 tests/dsv4_lora_cuda.py

PYTHONPATH=.:tests CUDA_VISIBLE_DEVICES=0,1 \
python3 -m torch.distributed.run --standalone --nproc-per-node=2 \
  tests/dsv4_post_training_fsdp.py

PYTHONPATH=.:tests CUDA_VISIBLE_DEVICES=0,1 \
python3 -m torch.distributed.run --standalone --nproc-per-node=2 \
  tests/dsv4_loading_memory.py
```

The loading comparison needs the pinned `fc133842` commit in local Git history.
The CUDA scripts write small JSON reports to `artifacts/`. The LoRA CUDA check
runs 20 updates for each AC policy in FP32 and BF16 autocast; the FSDP check
runs two optimizer updates with two microbatches each, including nonzero A
gradients after the first B update. See [LoRA verification](lora_validation.md).

### Optional HF reference

`tests/test_dsv4_hf_parity.py` originates from `codex/dsv41-hf-precision-parity`
and additionally covers CSA1/2, padding, and downstream query/output gradients.
It requires a Transformers build exposing `transformers.models.deepseek_v41`,
which was absent both locally and in the official `07338b6c` snapshot checked
for this work. The five tests are explicitly skipped in the ordinary suite
when unavailable; a skip is not a parity result.

Once the correct reference is on `PYTHONPATH`, require it with:

```bash
DSV41_REQUIRE_HF_REFERENCE=1 python3 -m unittest discover \
  -s tests -p test_dsv4_hf_parity.py -v
```

The strict command fails instead of skipping when the reference cannot import.
The native quantization tests always run and verify exact helper forward values
and the explicitly chosen identity straight-through gradient independently.

## CPU distributed parity

```bash
PYTHONPATH=. python3 -m torch.distributed.run \
  --nnodes=1 --nproc-per-node=2 \
  --master-addr=127.0.0.1 --master-port=29501 \
  tests/dsv4_engram_parallel_parity.py
```

```bash
PYTHONPATH=.:tests python3 -m torch.distributed.run \
  --nnodes=1 --nproc-per-node=2 \
  --master-addr=127.0.0.1 --master-port=29502 \
  tests/qwen_parallel_parity.py --cpu --padded
```

## GPU distributed checks

Two-GPU Qwen output and gradient parity:

```bash
PYTHONPATH=.:tests CUDA_VISIBLE_DEVICES=0,1 \
python3 -m torch.distributed.run --standalone --nproc-per-node=2 \
  tests/qwen_parallel_parity.py
```

Eight-GPU Qwen FSDP smoke test:

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python3 -m torch.distributed.run --standalone --nproc-per-node=8 \
  tests/qwen_fsdp_smoke.py
```

## Two-node NCCL check

Run this on both nodes. Change `NODE_RANK` to `0` or `1`, and replace
`10.0.0.1` with the first container's IP:

```bash
NODE_RANK=0
MASTER_ADDR=10.0.0.1

PYTHONPATH=. NCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python3 -m torch.distributed.run \
  --nnodes=2 --nproc-per-node=8 --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" --master-port=29500 \
  tests/distributed_nccl_smoke.py
```
