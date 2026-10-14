# Rollout Manager usage guide

New RL runs use vLLM for rollout inference and native PyTorch for learning.
The current vLLM training adapter supports flat, unquantized Qwen3 MoE in
BF16 with `lm_head` training. DeepSeek vLLM activation is not supported and
fails before learner allocation; `--rollout-engine native-pytorch` explicitly
selects the legacy compatibility route.

## Environment and ownership

Use the existing verified Python environments and invoke their executables
directly. Keep mounted model loading offline. Install this same source revision
on the learner machine and the CPU executor; a wheel is preferable to divergent
copies. The existing vLLM environment must contain **vLLM 0.31.0**. It is a
serving-environment prerequisite, not installed by the base package.

```sh
# Run in the selected existing environment, from this repository.
python -m pip install --no-deps -e .
python -m dsv41_train.rl.train --help
python -m rollout_manager.terminal_bench.cpu_supervisor --help
```

Base dependencies are in `pyproject.toml`, including Transformers
`>=4.57.6,<6` and `tomli` on Python 3.10. The `rollout-vllm` extra pins
vLLM 0.31.0, whose Transformers requirement narrows that range to
`>=5.10.4,<5.18`; inspect the actual serving environment before installation.
The measured actor used Transformers 5.17.0. The Harbor executor uses Python 3.12
or newer and the pinned `rollout-terminal` extra. Provision those dependencies
in the existing executor environment; a containerized GPU machine does not
imply host Docker or Pouch privileges.

Before loading models, inspect hostname, current directory, CPU affinity,
accessible cgroup quotas, available RAM/shared memory/disk, GPU UUIDs and
ownership, running processes, ports and actual GPU topology. Accessible host
memory is not necessarily the container's allocation entitlement. Select
disjoint learner/actor GPU UUIDs explicitly; the backend checks physical
identity before model allocation. Preserve unrelated processes. Use new private
output directories, and do not start a second actor or relay over a live run.

For the qualified 30B Qwen profile, one dedicated actor GPU needs room for the
model, the fixed 32 GiB KV cache and graph allocations. The learner needs its
own GPU allocation for the frozen backbone, 311,164,928 head parameters and
optimizer state. Full mounted context is preserved; fitting a short prompt
alone does not establish full-context capacity.

## Generic prompt-response RL

`examples/rl/qwen-head.json` explicitly selects the supported head-only vLLM
profile and one learner/one actor GPU. Its `toy_token_score` reward is a
diagnostic for training plumbing, not task correctness. Set `LEARNER_GPU_UUID`
and `ACTOR_GPU_UUID` to the inspected free devices. Choose a new `RUN_DIR`.

```sh
CUDA_VISIBLE_DEVICES="$LEARNER_GPU_UUID,$ACTOR_GPU_UUID" \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
python -m dsv41_train.rl.train \
  --model-dir /mnt/fuse/Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --data examples/rl/qwen-diagnostic-train.jsonl \
  --config examples/rl/qwen-head.json \
  --max-steps 1 --output-dir "$RUN_DIR/training"
```

For task training, supply a disjoint train/evaluation dataset and a reward
function through the existing RL data interface. Token JSONL uses
`prompt_ids`; text JSONL uses `messages` or `prompt`, with the model's mounted
tokenizer/chat template. See the existing [RL text-data description](../../README.md#text-data-task-rewards-and-evaluation)
for examples. Do not replace the mounted tokenizer or drop prior tool messages
to fit a context limit. Length violations follow the configured error/skip
policy, never silent history truncation.

For a new configuration, the required serving fields are:

```json
{
  "rollout_engine": "vllm",
  "trainer_num_gpus": 1,
  "rollout_num_gpus": 1,
  "dtype": "bfloat16",
  "trainable_scope": "lm_head",
  "qwen_last_position_only": true,
  "qwen_kv_cache": true,
  "backprop_chunk_tokens": 0,
  "checkpoint_interval": 1
}
```

Merge these fields into a complete RL configuration, rather than treating this
fragment as a complete experiment. The engine rejects incompatible trainable
scope, missing cache settings, overlapping devices, random initialization and
token stop sequences. `RLConfig()` also defaults to vLLM; CPU fixtures and
native model experiments must choose native compatibility explicitly.

Add `--eval-data PATH` for baseline, periodic and final evaluation through the
same actor. Evaluation uses independent engine seeds and normal EOS behavior.
Resume with the original model/data/config identities:

```sh
CUDA_VISIBLE_DEVICES="$LEARNER_GPU_UUID,$ACTOR_GPU_UUID" \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m dsv41_train.rl.train \
  --model-dir /mnt/fuse/Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --data examples/rl/qwen-diagnostic-train.jsonl \
  --config examples/rl/qwen-head.json \
  --resume "$RUN_DIR/training/step-00000001.pt" \
  --max-steps 2 --output-dir "$RUN_DIR/training"
```

Keep the original `--eval-data` if it was present. Standalone native `sample`
and `evaluate` commands reject vLLM checkpoints; use the training/resume actor
evaluation route. A checkpoint's version change is not proof of changed
weights: inspect the actual loaded-head digest and optimizer receipts.

## Terminal-Bench with an existing CPU executor

The GPU machine owns model serving/learning. The separate existing CPU
executor owns Harbor, Pouch sandboxes, shell execution, the original verifier
and cleanup. The current executable bootstrap covers only
`sparql-university` and `merge-diff-arc-agi-task`; a catalog entry alone does
not qualify another task. The CPU configuration must already identify checked
task copies, payload/image provenance, bootstrap qualification, network
isolation, inspected capacity and original budgets.

Prepare a versioned CPU workflow config with these relationships:

| Field | Value to supply |
| --- | --- |
| `source_root`, `expected_hostname` | Actual installed source and inspected CPU hostname |
| `executor`, `capacity` | Existing executor identity and usable CPU/memory/storage capacity |
| `public_capsule`, `lease_capsule`, `output_dir` | Absolute private paths; episode output absent under an existing parent |
| `task_ids`, `group_size`, `first_group_id` | One selected prepared task, G=8, group 0 |
| `purpose` | `training`; the later singleton uses `policy_evaluation` |
| `terminal_bench.serving_profile` | `vllm-training-v1` |
| `terminal_bench.execution_profile` | `training` |
| `terminal_bench.relay_root` | This run's existing CPU relay root |
| `terminal_bench.agent_max_turns` | Explicit optional 1..1024; default 128; quick experiment used 16 |

Keep the original 360-second generation deadline, 30-second shell deadline
and task/verifier budgets. Tool calls per turn are uncapped. Multi-turn is
enabled; the turn cap counts model responses, not executed shell calls.
The supervisor supplies each phase's immutable lease path and fresh model-guard
proof. Do not manufacture bootstrap/review evidence or loosen pinned source
checks to get past admission.

First export the checked public first-turn capsule on the CPU executor:

```sh
python -m rollout_manager.terminal_bench.cpu_workflow prepare \
  --config "$CPU_TRAIN_CONFIG" --output "$CPU_RUN/cpu-public-capsule.json"
```

Copy this public capsule to A once during setup and verify identical bytes and
SHA256. Credentials, task-private test data and model files are not capsule
contents. Subsequent leases, G8 evidence and v1 evaluation evidence use the
automatic artifact channel.

Start the A workflow in a dedicated terminal with two inspected GPUs. Create
the private `$A_RUN`, `$A_RUN/automatic` and `$GPU_RELAY_ROOT` parents first;
leave `training`, `group-0` and `evaluation-1` absent.

```sh
CUDA_VISIBLE_DEVICES="$LEARNER_GPU_UUID,$ACTOR_GPU_UUID" \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
python -m examples.rl.terminalbench_vllm_a \
  --model-dir /mnt/fuse/Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --capsule "$A_RUN/cpu-public-capsule.json" \
  --task-ids sparql-university --group-size 8 \
  --agent-max-turns 16 \
  --group-manifests "$A_RUN/automatic/group-0/group-0.seal.json" \
  --post-update-manifest "$A_RUN/automatic/evaluation-1/group-1.seal.json" \
  --gpu-relay-root "$GPU_RELAY_ROOT" --gpu-relay-port 18095 \
  --shim-port 8002 --learner-devices cuda:0 --rollout-devices cuda:1 \
  --output-dir "$A_RUN/training" --backprop-chunk-tokens 0
```

The vLLM wrapper enables mounted model qualification and automatic artifacts
by default. `--manual-artifacts` is legacy compatibility. Wait for
`A_VLLM_READY`, then use its actual controller PID, startup receipt and private
token-file path to start the GPU relay. Inspect that the selected ports are
free; change both workflow/relay configuration consistently if they are not.
The explicit 16-turn A flag must match both CPU phase configs. Omit it on both
sides for the normal 128-turn scope. Sealed imports check the inventory-bound
CPU configuration; an old serialized recovery identity is preserved rather
than rewritten to change historical metadata.
`GPU_PUBLIC_BASE`, `CPU_PUBLIC_BASE` and the peer origins must come from the
actual private provider gateway, not guessed URLs.

```sh
# A: private serving token stays in its file, never on the command line.
python -m rollout_manager.terminal_bench.browser_relay.relay_server \
  --role gpu --root "$GPU_RELAY_ROOT" --port 18095 \
  --public-base "$GPU_PUBLIC_BASE" --peer-origin "$CPU_ORIGIN" \
  --upstream http://127.0.0.1:8002 \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507 --model-pid "$CONTROLLER_PID" \
  --serving-profile vllm-training-v1 \
  --training-controller-receipt "$CONTROLLER_RECEIPT" \
  --training-controller-token-file "$CONTROLLER_TOKEN_FILE" \
  --max-inflight 1 --generation-seconds 360 --max-seconds 30000

# CPU: use the actual executor UID required by its qualified Pouch setup.
python -m rollout_manager.terminal_bench.browser_relay.relay_server \
  --role cpu --root "$CPU_RELAY_ROOT" --port 18096 \
  --public-base "$CPU_PUBLIC_BASE" --peer-origin "$GPU_ORIGIN" \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --serving-profile vllm-training-v1 \
  --max-inflight 1 --generation-seconds 360 --max-seconds 30000
```

Use `--proxy-origin` only for the exact origin emitted by a verified gateway.
Open the two private relay pages, pair them and keep their loop running.
Inspect a successful authenticated echo, actual engine/loaded-policy proof and
fresh readiness heartbeat before CPU admission. A transient readiness 503 is
retried by the browser; authentication and identity mismatches remain terminal.
Do not expose token fragments or private launch URLs in logs or Git.

The CPU supervisor receives leases and launches declared phases once. Its JSON
embeds two complete copies of the qualified CPU workflow config:

```json
{
  "schema_version": 1,
  "source_root": "/absolute/installed/source",
  "expected_hostname": "inspected-cpu-hostname",
  "expected_uid": 0,
  "output_root": "/absolute/private/new-cpu-run",
  "relay_root": "/absolute/private/new-cpu-run/relay-cpu",
  "wait_seconds": 30000,
  "phases": [
    {
      "lease_key": "lease-v0-training", "seal_key": "group-0",
      "policy_version": 0,
      "config": {"REPLACE_WITH_COMPLETE_G8_WORKFLOW_CONFIG": true}
    },
    {
      "lease_key": "lease-v1-evaluation", "seal_key": "evaluation-1",
      "policy_version": 1,
      "config": {"REPLACE_WITH_COMPLETE_SINGLETON_EVALUATION_CONFIG": true}
    }
  ]
}
```

This is a schema illustration, not an executable config. Use the inspected UID,
not the illustrative zero. Phase 0 has `first_group_id: 0`, `group_size: 8`
and `purpose: training`; phase 1 has `first_group_id: 1`, `group_size: 1`
and `purpose: policy_evaluation`. Both select the same single task and have
distinct absent output directories directly under `output_root`. Their
source/hostname/relay paths must equal the supervisor fields. The output root
must exist with private permissions. Then run:

```sh
python -m rollout_manager.terminal_bench.cpu_supervisor \
  --config "$CPU_RUN/supervisor.json"
```

No manual G8 archive transfer, lease copy or second worker launch is needed.
A pending phase claim after a crash is never automatically relaunched. A
terminal exclusion remains evidence and stops admission without inventing
reward zero. The fresh singleton evaluation is not another training update.

## Inspect completion and recover safely

Retain generation `.sampled.json` receipts, exact raw relay request/response
records, `.training.json` turn evidence, original verifier/cleanup results,
group seals, import receipts, learner metrics, policy artifacts, activation
readback, stage timings and final owned-process shutdown receipts. Confirm:

1. Every group member has original eligible verifier reward, complete action
   evidence and cleanup; all turns bind the same behavior policy.
2. CPU and GPU journals bind the same request/payload/policy and show actual
   acknowledged terminal delivery, with no unresolved requests.
3. The optimizer/checkpoint record consumes each group once, and the worker
   acknowledges the exact published head artifact.
4. A fresh post-update episode binds v1, verifies normally, returns durably,
   and all owned workers close with actual drain evidence.

Malformed complete shell output is recorded as `invalid_tool`; it executes no
call and still reaches the original verifier. A real transport/actor failure
requires typed terminal/drain proof. Never reinterpret uncertain delivery as
completed, replay an unknown request or clear quarantine using HTTP status
alone. Stop only exact owned processes, retain their ownership receipts and
inspect outputs after commands.

Generic recovery uses `--resume` and the format-2 checkpoint. Specialized
Terminal-Bench pre-update recovery additionally requires
`--recover-pre-update-from`, original workflow/policy/group-seal/unforced actor
exit pins and the historical relay ownership pins shown by `--help`. It is
limited to a proven failure before any optimizer step. It restores the original
v0 and consumes the already eligible sealed group exclusively; it does not
resample or relabel it. After a committed step, do not use that recovery mode
to repeat optimization.

## Local checks and measurement scope

Run focused contracts before hardware tests:

```sh
python -m unittest discover -s tests -p 'test_generic_vllm_rl.py'
python -m unittest discover -s tests -p 'test_vllm_controller.py'
python -m unittest discover -s tests/rollout_terminal -p 'test_automatic_artifacts.py'
node --test tests/rollout_terminal/test_relay_artifacts.mjs
```

The Terminal-Bench A wrapper's mounted qualification checks actual graph
capture/replay, actor/learner head identity, 128 real sampled tokens and a
finite native replay gradient without optimizing. It reports cross-engine
probability differences honestly; it does not gate them on false equality.
Fixture tests are not CUDA or live benchmark qualification. A short public
decode rate is not task throughput, and a zero-gradient step is not a learner
throughput estimate for nonzero objectives. See the
[implementation report](implementation.md) for the completed measured cycle
and its limitations.
