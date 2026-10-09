from __future__ import annotations

import argparse

from .config import RLConfig
from .runtime import run_training


def main():
    parser = argparse.ArgumentParser(description="Minimal asynchronous DeepSeek-V4.1 GRPO")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--data", required=True, help="pretokenized prompt/target JSONL")
    parser.add_argument("--config", required=True, help="RL runtime JSON")
    parser.add_argument("--allow-random-init", action="store_true", help="tiny smoke tests only")
    parser.add_argument("--resume", help="step-XXXXXXXX.pt checkpoint; retains base-model requirement")
    parser.add_argument("--max-steps", type=int, help="override total optimizer steps, including restored steps")
    parser.add_argument("--output-dir", help="override checkpoint and log directory")
    parser.add_argument("--trainer-num-gpus", type=int, help="trainer GPU count; zero selects CPU")
    parser.add_argument("--rollout-num-gpus", type=int, help="rollout GPU count; allocated after trainer GPUs")
    parser.add_argument("--engram-offload", action=argparse.BooleanOptionalAction, default=None,
                        help="keep frozen Engram in shared CPU memory (default: enabled)")
    args = parser.parse_args()
    values = RLConfig.from_json(args.config).to_dict()
    if args.max_steps is not None:
        values["max_steps"] = args.max_steps
    if args.output_dir is not None:
        values["output_dir"] = args.output_dir
    if args.trainer_num_gpus is not None or args.rollout_num_gpus is not None:
        if args.trainer_num_gpus is None or args.rollout_num_gpus is None:
            parser.error("specify both --trainer-num-gpus and --rollout-num-gpus")
        values.pop("learner_devices")
        values.pop("rollout_devices")
        values.update(trainer_num_gpus=args.trainer_num_gpus, rollout_num_gpus=args.rollout_num_gpus)
    if args.engram_offload is not None:
        values["engram_offload"] = args.engram_offload
    checkpoint = run_training(model_dir=args.model_dir, data_path=args.data,
                              config=RLConfig.from_dict(values),
                              random_init=args.allow_random_init, resume=args.resume)
    print(f"checkpoint: {checkpoint}", flush=True)


if __name__ == "__main__":
    main()
