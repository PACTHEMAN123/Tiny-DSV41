# Testing

Run all commands from the repository root.

## Unit tests

```bash
python3 -m unittest discover -s tests -v
```

The Qwen checkpoint tests are skipped when the optional `safetensors` package
is not installed.

Default RL selection, learner/actor contracts and legacy checkpoint compatibility
can be checked without starting a vLLM engine:

```bash
PYTHONPATH=.:tests python3 -m unittest -v \
  test_generic_vllm_rl.GenericSelectionTests \
  test_generic_vllm_rl.GenericVLLMBackendTests \
  test_vllm_learner_contract.VLLMLearnerContractTests \
  test_rl_family.ModelFamilyTests \
  test_rl_evaluation.EvaluationTests.test_legacy_checkpoint_keeps_original_eos_and_checks_recorded_identity
```

These 18 CPU fixture tests passed on the integrated branch. Full SFT source
reconstruction and native SFT-to-RL initialization also passed six CPU tests on
Linux (`python3 -m unittest discover -s tests -p 'test_*sft_source.py' -v`).
Neither suite validates a released-model vLLM GPU engine.

SFT save controls have seven focused CPU regressions, also verified on Linux:

```bash
python3 -m unittest discover -s tests -p 'test_train_sft.py' -v
```

They cover default/periodic saving, `--no-save` for full and LoRA training,
suppression of adapter/Engram bundle exports, and resume followed by an update
without creating another checkpoint. Tokenizer/model loading and CUDA telemetry
are fixtures; actual tiny training and checkpoint save/load still run.

The optional short/long identity fixture, answer labels and native greedy
generation have 12 focused CPU checks, verified locally and on Linux:

```bash
PYTHONPATH=.:tests python3 -m unittest -v \
  test_prepare_identity_sft test_sft_data test_sft_eval
```

To reproduce the Qwen 20-step mixed-context identity experiment:

```bash
PYTHONPATH=. python3 scripts/prepare_identity_sft.py \
  --model-path /path/to/Qwen3-30B-A3B --output-dir runs/qwen-identity-mixed \
  --include-short-training
PYTHONPATH=. python3 -m torch.distributed.run --standalone --nproc-per-node=8 \
  train_sft.py --model-path /path/to/Qwen3-30B-A3B --family qwen --mode full \
  --data runs/qwen-identity-mixed/train.jsonl --eval-data runs/qwen-identity-mixed/eval.jsonl \
  --output-dir runs/qwen-identity-mixed20 --steps 20 --seq-len 4608 \
  --batch-size 1 --gradient-accumulation-steps 1 --cp-size 1 --ep-size 1 \
  --optimizer adamw --learning-rate 1e-5 --seed 42 --gradient-checkpointing \
  --generation-eval --generation-max-new-tokens 16 --no-save
```

The eight examples give each global step four real long and four short inputs;
short inputs are not padded to 4096 for this claim. The unchanged five-prompt
evaluation includes one seen short prompt, three unseen short questions and one
unseen long prompt. The observed 3/5 result must retain that split. Native CE
evaluation assigns those five examples cyclically across eight ranks, repeating
the first three; it is not the equal-example five-question CE mean. Generation
evaluates each of the five prompts once. This fixture checks identity tuning,
not broad model quality, and `--no-save` does not retain the trained weights.

A separate original-step20 diagnostic compared teacher-forced, prompt-only and
same-length masked-EOS inputs for two short questions on all eight ranks. For
`你是谁？`, the first observed difference was a layer0 output delta of 0.000977;
later MoE routing diverged and the final maximum logit delta was 1.742188.
The masked-EOS and teacher-forced traces matched exactly; the second question
matched across all three forms. This supports sensitivity to input shape and
numerical paths, without identifying a particular kernel defect. No padding or
routing change was made to production inference to force that comparison to pass.

## CPU distributed parity

Shared RL token log probabilities and checkpointed gradients, including a CP
rank with no response tokens (model-level parity, not a distributed RL runtime):

```bash
PYTHONPATH=. python3 -m torch.distributed.run \
  --nnodes=1 --nproc-per-node=2 \
  --master-addr=127.0.0.1 --master-port=29503 \
  tests/dsv4_rl_cp_parity.py
```

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

Four-rank SFT checks cover full training and LoRA with trainable CPU Engram,
CP2/EP2, two-rank expert FSDP groups, and unequal supervised token counts:

```bash
PYTHONPATH=.:tests CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 -m torch.distributed.run --standalone --nproc-per-node=4 \
  tests/dsv4_sft_dp_parity.py --device cuda --output runs/sft-dp4-cuda.json
```

CUDA uses deterministic Torch reference kernels with BF16/FP32 parameters.
Each step starts from bit-exact matching state and checks gradients and one
SGD update. This does not assert equality of independently rounded BF16
training trajectories. Use `--device cpu` for two independent FP32 steps.
The output path must be new; reports include the precision and error bounds.

The integration corrects expert-gradient normalization for CP1/EP>1 and disables
FSDP backward prefetch whenever EP is enabled. These shared changes also affect
the original training entry points: old expert updates can differ in magnitude,
and prefetch scheduling can change performance. The CP/EP gradient checks verify
normalization against an unsharded reference; they do not promise identical
numerics or throughput to the earlier implementation.

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
