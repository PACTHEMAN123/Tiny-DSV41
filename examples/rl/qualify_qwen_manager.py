"""Bounded mounted-Qwen manager experiment; no random initialization or downloads.

The GRPO reward deliberately scores token IDs to exercise weight updates. Four
independent arithmetic prompts assess generation before and after three updates;
this small diagnostic does not establish a change in model quality.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time


HEAD_KEY = "backend.lm_head.weight"


def require(value, message):
    if not value:
        raise RuntimeError(message)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def device_receipt(devices):
    import torch

    result = []
    for name in devices:
        device = torch.device(name)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            result.append(dict(device=name, name=torch.cuda.get_device_name(device),
                               peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                               peak_reserved_bytes=torch.cuda.max_memory_reserved(device)))
    return result


def probe(args):
    import torch
    from dsv41_train.rl.algorithm import response_log_probs
    from dsv41_train.rl.checkpoint import load_policy_artifact, run_identity
    from dsv41_train.rl.data import build_reward, load_jsonl
    from dsv41_train.rl.evaluation import append_jsonl, sample_record
    from dsv41_train.rl.family import build_policy, load_model_config
    from dsv41_train.rl.rollout import generate_group
    from dsv41_train.rl.tokenizer import ModelTokenizer

    start = time.monotonic()
    torch.set_num_threads(args.num_threads)
    torch.manual_seed(42)
    config = load_model_config(args.model_dir)
    require(config.model_type == "qwen3_moe", "experiment requires a mounted Qwen3 MoE checkpoint")
    tokenizer = ModelTokenizer(args.model_dir)
    records = load_jsonl(args.eval_data, tokenizer=tokenizer, max_prompt_tokens=256)
    descriptor = None
    if args.policy_artifact:
        descriptor = json.loads(args.policy_artifact.read_text())
        require(isinstance(descriptor.get("identity"), dict) and
                descriptor["identity"].get("trainable_scope") == "lm_head",
                "probe artifact must declare trainable_scope=lm_head")
        current = run_identity(args.model_dir, args.train_data, False)
        for key, value in current.items():
            if key not in {"data_sha256", "eval_data_sha256"}:
                require(descriptor["identity"].get(key) == value, "mounted base identity changed: " + key)
    model = build_policy(args.model_dir, config, random_init=False,
                         devices=(args.rollout_device,), dtype=torch.bfloat16, trainable_scope="lm_head")
    require({name for name, value in model.named_parameters() if value.requires_grad} == {HEAD_KEY},
            "probe must train only the output head")
    frozen = model.frozen_parameter_fingerprint()
    activation = None
    if descriptor is not None:
        activation = load_policy_artifact(model, descriptor)
    model.eval()
    reward = build_reward("numeric_answer", tokenizer=tokenizer)
    generator = torch.Generator(device="cpu").manual_seed(1234)
    rows, differences = [], []
    with torch.no_grad():
        for index, record in enumerate(records):
            require(len(record.prompt_ids) + args.eval_tokens <= config.max_position_embeddings,
                    "evaluation exceeds model context")
            group = generate_group(model, record, group_size=1, max_new_tokens=args.eval_tokens,
                                   temperature=.7, eos_token_id=tokenizer.eos_token_ids,
                                   policy_version=0 if activation is None else activation["version"],
                                   group_id=index, generator=generator, reward_fn=reward)
            sample = group.samples[0]
            actual = response_log_probs(model, sample.prompt_ids, sample.response_ids, .7).float().cpu()
            observed = torch.tensor(sample.behavior_log_probs)
            difference = float((actual - observed).abs().max())
            require(difference <= args.parity_tolerance,
                    f"sampled/teacher-forced BF16 log-probability mismatch: {difference}")
            differences.append(difference)
            row = sample_record(record, sample, step=group.policy_version, group_id=index,
                                tokenizer=tokenizer, phase=args.probe_phase)
            row.update(behavior_log_probs=sample.behavior_log_probs,
                       teacher_forced_log_probs=actual.tolist(), max_log_prob_difference=difference)
            rows.append(row)
    model.assert_engram_invariants()
    require(frozen == model.frozen_parameter_fingerprint(), "frozen backbone diagnostic changed")
    append_jsonl(args.output_dir / (args.probe_phase + "-samples.jsonl"), rows)
    receipt = dict(status="passed", phase=args.probe_phase, scope="four fixed arithmetic prompts",
                   model_dir=str(args.model_dir), random_init=False, trainable_scope="lm_head",
                   accuracy=sum(row["reward"] for row in rows) / len(rows), samples=len(rows),
                   truncation_fraction=sum(row["finish_reason"] == "length" for row in rows) / len(rows),
                   max_log_prob_difference=max(differences), bf16_parity_tolerance=args.parity_tolerance,
                   frozen_backbone=frozen, policy_ack=activation,
                   wall_seconds=time.monotonic() - start,
                   process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss *
                   (1 if sys.platform == "darwin" else 1024), gpu_memory=device_receipt((args.rollout_device,)))
    write_json(args.output_dir / (args.probe_phase + "-receipt.json"), receipt)
    print(json.dumps(receipt, allow_nan=False), flush=True)


def verify_training(output, group_size):
    import torch
    from dsv41_train.rl.checkpoint import file_sha256
    from dsv41_train.rl.manager import _decode_packet

    training = output / "training"
    metrics = [json.loads(line) for line in (training / "metrics.jsonl").read_text().splitlines()]
    require([row["step"] for row in metrics] == [1, 2, 3], "training did not execute steps 1, 2 and resumed 3")
    require(any(row["nonzero_advantages"] > 0 for row in metrics), "all diagnostic groups had zero advantages")
    require(metrics[0]["rollout_versions"] == [0] and abs(metrics[0]["importance_ratio"] - 1) <= 1e-5 and
            metrics[0]["clip_fraction"] == 0 and abs(metrics[0]["approx_kl"]) <= 1e-5,
            "initial learner/rollout action probabilities are inconsistent")
    report = metrics[0]["parameter_report"]
    require(report["trainable_parameter_count"] > 0 and
            report["frozen_parameter_count"] > report["trainable_parameter_count"] and
            all(row["trainable_scope"] == "lm_head" and row["parameter_report"] == report for row in metrics),
            "learner frozen-backbone diagnostic or trainable scope changed")
    initial_path = next((training / "policies").glob("policy-00000000-*.pt"))
    initial_snapshot = torch.load(initial_path, map_location="cpu", weights_only=True)
    require(initial_snapshot["version"] == 0 and initial_snapshot["identity"].get("trainable_scope") == "lm_head" and
            initial_snapshot["identity"].get("random_init") is False and
            set(initial_snapshot["trainable"]) == {HEAD_KEY} and initial_snapshot["frozen"] == {} and
            file_sha256(initial_path) == initial_path.stem.split("-")[-1],
            "initial mounted-head publication has invalid scope, identity or digest")
    initial = initial_snapshot["trainable"][HEAD_KEY]
    del initial_snapshot
    cursors, changed, adam_steps = [], [], []
    for step in (2, 3):
        state = torch.load(training / f"step-{step:08d}.pt", map_location="cpu", weights_only=True)
        require(state["format_version"] == 2, "manager recovery checkpoint must use format 2")
        require(set(state["policy"]["trainable"]) == {HEAD_KEY} and state["policy"]["frozen"] == {},
                "checkpoint unexpectedly contains frozen backbone weights")
        require(state["config"]["trainable_scope"] == state["identity"].get("trainable_scope") == "lm_head",
                "checkpoint scope is not bound to its identity")
        count = int((initial != state["policy"]["trainable"][HEAD_KEY]).count_nonzero())
        require(count > 0, "real mounted output-head weights did not change")
        changed.append(count)
        optimizer = state["optimizer"]["state"]
        require(len(optimizer) == 1, "optimizer must contain exactly the trainable output head")
        adam = next(iter(optimizer.values()))
        require(bool(adam["exp_avg"].count_nonzero()) and bool(adam["exp_avg_sq"].count_nonzero()),
                "Adam moments did not update")
        require(int(adam["step"]) == step, "optimizer step did not resume")
        adam_steps.append(int(adam["step"]))
        recovery = state["manager_state"]
        pending = [_decode_packet(packet)["group"].group_id for packet in recovery["pending_packets"]]
        require(pending == list(range(recovery["committed_highwater"], recovery["next_group_id"])),
                "manager pending cursor range is incomplete")
        cursors.append(state["next_group_id"])
        if step == 3:
            descriptor_path = output / "final-policy.json"
            if descriptor_path.exists():
                require(json.loads(descriptor_path.read_text()) == state["policy_publication"],
                        "existing final-policy descriptor differs from the checkpoint")
            else:
                write_json(descriptor_path, state["policy_publication"])
        del state, optimizer, adam
    for row in metrics:
        require(row["rollout_provider"] == "rollout_manager", "unexpected rollout producer")
        ack = row["policy_ack"]
        artifact = training / f"policies/policy-{ack['version']:08d}-{ack['digest']}.pt"
        require(ack["version"] == row["step"] and row["policy_digest"] == ack["digest"] and
                file_sha256(artifact) == ack["digest"], "weight activation digest/version mismatch")
        snapshot = torch.load(artifact, map_location="cpu", weights_only=True)
        require(snapshot["version"] == ack["version"] and snapshot["lineage"] == ack["lineage"] and
                set(snapshot["trainable"]) == {HEAD_KEY} and snapshot["frozen"] == {},
                "immutable artifact metadata or trainable scope mismatch")
        del snapshot
    samples = [json.loads(line) for line in (training / "train-samples.jsonl").read_text().splitlines()]
    consumed = []
    for step in (1, 2, 3):
        selected = [row for row in samples if row["step"] == step]
        ids = {row["group_id"] for row in selected}
        require(len(selected) == group_size and len(ids) == 1, "incomplete consumed group evidence")
        require(all(row["response_text"] is not None for row in selected), "missing decoded training output")
        consumed.append(next(iter(ids)))
    require(len(set(consumed)) == 3 and cursors[1] >= cursors[0], "resume reused a consumed group or issued cursor")
    return dict(steps=[1, 2, 3], policy_acks=[row["policy_ack"] for row in metrics],
                consumed_group_ids=consumed, issued_cursors=cursors, adam_steps=adam_steps,
                changed_output_head_elements=changed, learner_parameter_report=report, training_metrics=metrics)


def main():
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("/mnt/fuse/Qwen/Qwen3-30B-A3B-Instruct-2507"))
    parser.add_argument("--learner-device", required=True)
    parser.add_argument("--rollout-device", required=True)
    parser.add_argument("--train-data", type=Path, default=repo / "examples/rl/qwen-diagnostic-train.jsonl")
    parser.add_argument("--eval-data", type=Path, default=repo / "examples/rl/qwen-math-eval.jsonl")
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--train-tokens", type=int, default=8)
    parser.add_argument("--eval-tokens", type=int, default=8)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--parity-tolerance", type=float, default=1e-5)
    parser.add_argument("--probe-phase", choices=["baseline", "final"], help=argparse.SUPPRESS)
    parser.add_argument("--policy-artifact", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output_dir, args.model_dir = args.output_dir.resolve(), args.model_dir.resolve()
    require(args.model_dir.is_dir(), "mounted model directory is missing")
    if args.probe_phase:
        probe(args)
        return
    require(not args.output_dir.exists(), "experiment output directory must be new")
    from dsv41_train.rl.config import RLConfig

    values = dict(max_steps=2, group_size=args.group_size, groups_per_step=1, max_new_tokens=args.train_tokens,
                  seed=42, queue_capacity=1, checkpoint_interval=1, kl_beta=0, reward="toy_token_score",
                  temperature=1.3, learning_rate=1e-4, dtype="bfloat16", trainable_scope="lm_head",
                  learner_devices=[args.learner_device], rollout_devices=[args.rollout_device],
                  output_dir=str(args.output_dir / "training"), worker_timeout_seconds=args.timeout_seconds,
                  log_samples_per_step=args.group_size, num_threads=args.num_threads, max_prompt_tokens=256)
    RLConfig.from_dict(values)
    require((args.learner_device, args.rollout_device) == ("cpu", "cpu") or
            args.learner_device != args.rollout_device, "learner and rollout need separate CUDA ownership")
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / "receipt.json", dict(status="incomplete", scope="mounted Qwen output-head GRPO"))
    config_path = args.output_dir / "config.json"
    write_json(config_path, values)
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
           "PYTHONDONTWRITEBYTECODE": "1"}
    env.pop("PYTHONPATH", None)
    common = [sys.executable, "-B", "-m", "examples.rl.qualify_qwen_manager", "--output-dir", str(args.output_dir),
              "--model-dir", str(args.model_dir), "--learner-device", args.learner_device,
              "--rollout-device", args.rollout_device, "--train-data", str(args.train_data),
              "--eval-data", str(args.eval_data), "--eval-tokens", str(args.eval_tokens),
              "--num-threads", str(args.num_threads), "--parity-tolerance", str(args.parity_tolerance)]
    command = [sys.executable, "-B", "-m", "dsv41_train.rl.train", "--model-dir", str(args.model_dir),
               "--data", str(args.train_data), "--config", str(config_path),
               "--rollout-engine", "native-pytorch"]
    start = time.monotonic()

    def execute(name, argv):
        with (args.output_dir / (name + ".log")).open("x") as log:
            process = subprocess.Popen(argv, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            try:
                status = process.wait(timeout=args.timeout_seconds)
            except subprocess.TimeoutExpired:
                # This process group contains only this invocation and its manager.
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=30)
                raise RuntimeError("experiment timed out; owned process group stopped: " + name) from None
        require(status == 0, "experiment failed; inspect " + str(args.output_dir / (name + ".log")))

    execute("baseline", common + ["--probe-phase", "baseline"])
    execute("initial", command)
    execute("resume", command + ["--resume", str(args.output_dir / "training/step-00000002.pt"), "--max-steps", "3"])
    verified = verify_training(args.output_dir, args.group_size)
    execute("final", common + ["--probe-phase", "final", "--policy-artifact", str(args.output_dir / "final-policy.json")])
    baseline, final = [json.loads((args.output_dir / (name + "-receipt.json")).read_text()) for name in ("baseline", "final")]
    require(baseline["frozen_backbone"] == final["frozen_backbone"], "mounted frozen-base diagnostic differs after resume")
    require(verified["learner_parameter_report"] == baseline["frozen_backbone"],
            "learner and evaluation did not reconstruct the same frozen-base diagnostic")
    receipt = dict(schema_version=1, status="passed", model_dir=str(args.model_dir), random_init=False,
                   trainable_scope="lm_head", dtype="bfloat16", learner_device=args.learner_device,
                   rollout_device=args.rollout_device, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                   diagnostic_reward="toy_token_score: plumbing only, not answer quality",
                   quality_scope="four fixed arithmetic prompts; no claim of learning quality",
                   baseline=baseline, final=final, wall_seconds=time.monotonic() - start, **verified)
    write_json(args.output_dir / "receipt.json", receipt)
    print(json.dumps(dict(status="passed", steps=[1, 2, 3], baseline_accuracy=baseline["accuracy"],
                          final_accuracy=final["accuracy"], receipt=str(args.output_dir / "receipt.json"))), flush=True)


if __name__ == "__main__":
    main()
