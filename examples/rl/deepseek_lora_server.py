"""Serve the owned DeepSeek rollout actor on an explicit private IPv4 address."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from rollout_manager.backends.vllm_lora_controller import LoRARolloutConfig, VLLMLoRAController
from rollout_manager.backends.vllm_lora_remote import RemoteLoRAServer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--token-env", default="DSV41_ROLLOUT_TOKEN")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--work-dir", required=True,
                        help="new directory for owned actor sockets, process state and logs")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--operation-timeout-seconds", type=int, default=7200)
    args = parser.parse_args()
    token = os.environ.pop(args.token_env, None)
    if token is None:
        parser.error("the configured RPC token environment variable is missing")
    config = LoRARolloutConfig(operation_timeout_seconds=args.operation_timeout_seconds,
        rollout_devices=tuple("cuda:" + item for item in args.gpus.split(",")),
        max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs, seed=args.seed)
    controller = VLLMLoRAController(args.model_dir, config, work_dir=args.work_dir)
    with RemoteLoRAServer((args.bind, args.port), token, controller,
                         io_timeout=args.operation_timeout_seconds) as server:
        try:
            server.serve_forever(poll_interval=.2)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
