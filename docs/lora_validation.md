# LoRA verification

Revalidated on 2026-09-28 using NVIDIA H20 and PyTorch 2.10.0+cu129.
No LoRA implementation changes were needed. This pass strengthens the tests
so they exercise both adapter matrices after the zero-initialized B matrix
has received its first update.

## Results

- All 13 focused post-training/loading regression tests pass, with no skips.
- An independent dense-weight calculation checks forward values and input/A/B
  gradients in FP64 at `rtol=atol=1e-12`, including non-default rank and alpha.
- Every default adapter's A and B receives a finite, nonzero gradient on the
  second training step. Both change with optimizer weight decay disabled.
  Every base parameter remains frozen, gradient-free, and bitwise unchanged.
- Nonzero adapters with LoRA dropout 0.2 and attention dropout 0.15 give matching
  outputs and gradients with full/selective AC, including padding and shared KV.
- Adapter export/reload preserves outputs exactly. FP32 merged-inference maximum
  absolute error on CUDA is 2.09e-7; unmerge restores outputs exactly. A separate
  test verifies exact restoration of BF16 base weights after merge/unmerge.
- The DCP resume test restores optimizer and dropout RNG state: the next loss
  and updated adapters exactly match uninterrupted CPU training.
- All six two-GPU FSDP combinations pass: three AC policies, with and without
  LoRA. Each performs two updates, each accumulating two distinct microbatches.
  LoRA's maximum gradient difference from the averaged unsharded reference is
  8.38e-9; the full-training control's maximum is 2.98e-8. Parameter updates and
  adapter export/reload also match.

## Single-GPU learning check

The random five-layer model has hidden size 32, vocabulary 32, compression
ratios `[0, 2, 2, 1, 1]`, shared KV/index state, and Engram. A fixed batch has
two 12-token sequences with padding. The 20 default attention adapters contain
10,240 trainable parameters: rank 8, alpha 16, dropout 0.2. Attention dropout
is 0.1. AdamW uses learning rate 0.01 and zero weight decay for 20 updates.
Losses below are evaluated with dropout disabled before and after training.

| Precision | AC modes | Initial loss | Final loss | Base parameter change |
| --- | --- | ---: | ---: | ---: |
| FP32 | none/full/selective | 3.478513 | 2.973384 | 0 |
| BF16 autocast, FP32 weights | none/full/selective | 3.478569 | 2.994485 | 0 |

Within each precision, all three AC modes produced identical training loss
sequences. Every run checks finite gradients, updates to every adapter matrix,
unchanged base parameters, zero-init identity, export/reload, and merge/unmerge.
This is a fixed-batch functional learning check, not a downstream quality
benchmark or a test of a full pretrained DeepSeek checkpoint.

## Reproduce

```bash
PYTHONPATH=.:tests python3 -m unittest test_post_training test_dsv4_loading -v

PYTHONPATH=.:tests CUDA_VISIBLE_DEVICES=0 python3 tests/dsv4_lora_cuda.py

PYTHONPATH=.:tests CUDA_VISIBLE_DEVICES=0,1 \
python3 -m torch.distributed.run --standalone --nproc-per-node=2 \
  tests/dsv4_post_training_fsdp.py
```

The measured machine's interpreter/PYTHONPATH workaround is documented in
[profiling](profiling.md#reproduce). [Versioned raw results](results/lora-validation-h20.json)
contain all learning curves and FSDP measurements. Fresh script outputs are
written to `artifacts/lora-cuda-verification.json` and `artifacts/fsdp-parity.json`.

This pass covers the default attention targets on small models and a custom
linear layer. CP/EP combinations, large pretrained weights, and downstream
fine-tuning quality remain outside these checks. HF reference parity is a
separate unverified item; it is not counted as passing here.
