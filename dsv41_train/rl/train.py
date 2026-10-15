from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import RLConfig
from .runtime import run_training


def main():
    parser = argparse.ArgumentParser(description="Rollout Manager GRPO with vLLM inference and a native learner")
    parser.add_argument("--model-dir", required=True, help="local checkpoint directory; model family comes from config.json")
    parser.add_argument("--data", required=True, help="JSONL containing prompt, messages or prompt_ids")
    parser.add_argument("--eval-data", help="independent validation JSONL; enables baseline and periodic evaluation")
    parser.add_argument("--config", required=True, help="RL runtime JSON")
    parser.add_argument("--random-init", "--allow-random-init", action="store_true",
                        help="initialize from config.json without loading base weights")
    parser.add_argument("--resume", help="step-XXXXXXXX.pt checkpoint; retains base-model requirement")
    parser.add_argument("--max-steps", type=int, help="override total optimizer steps, including restored steps")
    parser.add_argument("--output-dir", help="override checkpoint and log directory")
    parser.add_argument("--trainer-num-gpus", type=int, help="trainer GPU count; zero selects CPU")
    parser.add_argument("--rollout-num-gpus", type=int, help="rollout GPU count; allocated after trainer GPUs")
    parser.add_argument("--rollout-engine", choices=("vllm", "native-pytorch"),
                        help="vllm by default; native-pytorch explicitly selects legacy sampling")
    parser.add_argument("--trainable-scope", choices=("all", "lm_head", "lora"),
                        help="LoRA requires the native-pytorch rollout engine")
    parser.add_argument("--initial-adapter", help="portable SFT adapter; requires trainable-scope=lora")
    parser.add_argument("--engram-offload", action=argparse.BooleanOptionalAction, default=None,
                        help="keep frozen Engram in shared CPU memory (default: enabled)")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=None,
                        help="recompute decoder activations during learner backward (default: disabled)")
    args = parser.parse_args()
    values = json.loads(Path(args.config).read_text(encoding="utf-8"))
    values["rollout_engine"] = args.rollout_engine or values.get("rollout_engine", "vllm")
    values = RLConfig.from_dict(values).to_dict()
    if args.max_steps is not None:
        values["max_steps"] = args.max_steps
    if args.output_dir is not None:
        values["output_dir"] = args.output_dir
    if args.trainable_scope is not None:
        values["trainable_scope"] = args.trainable_scope
    if args.initial_adapter is not None:
        values["initial_adapter"] = args.initial_adapter
    if args.trainer_num_gpus is not None or args.rollout_num_gpus is not None:
        if args.trainer_num_gpus is None or args.rollout_num_gpus is None:
            parser.error("specify both --trainer-num-gpus and --rollout-num-gpus")
        values.pop("learner_devices")
        values.pop("rollout_devices")
        values.update(trainer_num_gpus=args.trainer_num_gpus, rollout_num_gpus=args.rollout_num_gpus)
    if args.engram_offload is not None:
        values["engram_offload"] = args.engram_offload
    if args.gradient_checkpointing is not None:
        values["gradient_checkpointing"] = args.gradient_checkpointing
    checkpoint = run_training(model_dir=args.model_dir, data_path=args.data,
                              config=RLConfig.from_dict(values),
                              random_init=args.random_init, resume=args.resume, eval_data_path=args.eval_data)
    print(f"checkpoint: {checkpoint}", flush=True)


if __name__ == "__main__":
    main()
