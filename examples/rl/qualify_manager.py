"""Tiny native manager qualification: python -m examples.rl.qualify_manager ..."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def require(value, message):
    if not value:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--learner-device", required=True)
    parser.add_argument("--rollout-device", required=True)
    parser.add_argument("--dtype", choices=["bfloat16"], default="bfloat16")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    require(not output.exists(), "qualification output directory must be new")
    require((args.learner_device, args.rollout_device) == ("cpu", "cpu") or
            (args.learner_device.startswith("cuda:") and args.rollout_device.startswith("cuda:") and
             args.learner_device != args.rollout_device), "use CPU/CPU or two distinct explicit CUDA devices")
    repo = Path(__file__).resolve().parents[2]
    model_dir = repo / "examples/rl/tiny-dsv41"
    values = dict(max_steps=2, group_size=4, groups_per_step=1, max_new_tokens=4, seed=42,
                  queue_capacity=1, checkpoint_interval=1, kl_beta=.02, reward="toy_token_score",
                  dtype=args.dtype, learner_devices=[args.learner_device], rollout_devices=[args.rollout_device],
                  output_dir=str(output / "training"), worker_timeout_seconds=120, log_samples_per_step=4)
    from dsv41_train.rl.config import RLConfig
    RLConfig.from_dict(values)
    require(args.learner_device == "cpu" or int(args.learner_device.split(":")[1]) !=
            int(args.rollout_device.split(":")[1]), "learner and rollout must use distinct CUDA indices")
    output.mkdir(parents=True)
    (output / "receipt.json").write_text(json.dumps({"status": "incomplete", "scope": "tiny native manager"}) + "\n")
    config_path = output / "config.json"
    config_path.write_text(json.dumps(values, indent=2) + "\n")
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
           "PYTHONDONTWRITEBYTECODE": "1"}
    env.pop("PYTHONPATH", None)
    command = [sys.executable, "-B", "-m", "dsv41_train.rl.train", "--model-dir", str(model_dir),
               "--data", str(repo / "examples/rl/prompts.jsonl"), "--config", str(config_path), "--random-init",
               "--rollout-engine", "native-pytorch"]
    checkpoint_two = output / "training/step-00000002.pt"
    rows = []
    for name, extra in (("initial", []), ("resume", ["--resume", str(checkpoint_two), "--max-steps", "3"])):
        log_path = output / (name + ".log")
        with log_path.open("x") as log:
            status = subprocess.run(command + extra, cwd=repo, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=300).returncode
        require(status == 0, "training failed; inspect " + str(log_path))
        rows.extend(json.loads(line) for line in log_path.read_text().splitlines() if line.startswith("{"))
    import torch
    from dsv41_train.rl.checkpoint import file_sha256, load_policy
    from dsv41_train.rl.manager import _decode_packet
    from dsv41_train.rl.policy import DSV41Config, build_dsv41_policy
    torch.set_num_threads(1)
    torch.manual_seed(42)
    initial = build_dsv41_policy(model_dir, DSV41Config.from_model_dir(model_dir), random_init=True,
                                devices=("cpu",), dtype=torch.bfloat16)
    initial_weights = initial.trainable_state_dict()
    engram = {name: value.clone() for name, value in initial.cpu_engram.items()}
    checkpoints = [torch.load(output / f"training/step-{step:08d}.pt", map_location="cpu", weights_only=True)
                   for step in (2, 3)]
    def cpu_plain(value):
        return (all(cpu_plain(v) for v in value.values()) if isinstance(value, dict) else
                all(cpu_plain(v) for v in value) if isinstance(value, (list, tuple)) else
                isinstance(value, (str, int, float, bool, bytes, type(None))))
    for state in checkpoints:
        require(state["format_version"] == 2 and cpu_plain(state["manager_state"]), "invalid CPU recovery codec")
        require(any(not torch.equal(initial_weights[k], v) for k, v in state["policy"]["trainable"].items()),
                "trainable weights did not change")
        optimizer = state["optimizer"]["state"]
        require(optimizer and any(v["exp_avg"].count_nonzero().item() for v in optimizer.values()), "Adam did not update")
        require(not any("engram_tables" in k for part in state["policy"].values() for k in part), "Engram leaked into policy")
        recovery = state["manager_state"]
        pending = [_decode_packet(packet)["group"].group_id for packet in recovery["pending_packets"]]
        require(pending == list(range(recovery["committed_highwater"], recovery["next_group_id"])), "recovery cursor gap")
        load_policy(initial, state["policy"])
        initial.assert_engram_invariants()
        require(all(torch.equal(engram[k], v) for k, v in initial.cpu_engram.items()), "frozen Engram changed on load")
    metrics = [row for row in rows if row.get("event") == "train"]
    require([row["step"] for row in metrics] == [1, 2, 3], "missing activation steps")
    require(any(row["nonzero_advantages"] for row in metrics), "all groups had zero advantages")
    for row in rows:
        require(row.get("rollout_provider") == "rollout_manager", "unexpected rollout producer")
        ack = row["policy_ack"]
        artifact = output / f"training/policies/policy-{ack['version']:08d}-{ack['digest']}.pt"
        require(ack["version"] == row["policy_version"] and file_sha256(artifact) == ack["digest"], "policy ACK mismatch")
        published = torch.load(artifact, map_location="cpu", weights_only=True)
        require(all(ack[key] == published[key] for key in ("version", "lineage")), "policy ACK metadata mismatch")
        require(row.get("policy_digest", ack["digest"]) == ack["digest"], "metric digest differs from ACK")
        require(row["engram_shared"] and row.get("rollout_engram_shared", True) and row["engram_device"] == "cpu",
                "Engram sharing/offload invariant failed")
    samples = [json.loads(line) for line in (output / "training/train-samples.jsonl").read_text().splitlines()]
    consumed = []
    for step in (1, 2, 3):
        selected = [row for row in samples if row["step"] == step]
        ids = {row["group_id"] for row in selected}
        require(len(selected) == 4 and len(ids) == 1, "consumed group/sample evidence is incomplete")
        consumed.append(next(iter(ids)))
    require(len(set(consumed)) == 3 and checkpoints[1]["next_group_id"] >= checkpoints[0]["next_group_id"], "resume reused cursor")
    receipt = dict(schema_version=1, status="passed", scope="tiny random native PyTorch manager", dtype=args.dtype,
                   learner_device=args.learner_device, rollout_device=args.rollout_device,
                   cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), policy_acks=[r["policy_ack"] for r in metrics],
                   consumed_group_ids=consumed, issued_cursors=[s["next_group_id"] for s in checkpoints])
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": "passed", "steps": [1, 2, 3], "receipt": str(output / "receipt.json")}))


if __name__ == "__main__":
    main()
