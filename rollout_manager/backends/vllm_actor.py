"""Private, bounded actor process owning one vLLM engine and its workers."""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import socket
import struct
import time

from .vllm_worker import process_identity, VLLM_KERNEL_CONFIG, VLLM_WORKER_EXTENSION, VLLM_WORKER_RPC


MAX_RPC_BYTES = 2 * 1024**2


def read_exact(stream, length):
    parts = []
    remaining = length
    while remaining:
        part = stream.recv(remaining)
        if not part:
            raise EOFError("incomplete owned actor RPC")
        parts.append(part)
        remaining -= len(part)
    return b"".join(parts)


def receive(stream):
    length = struct.unpack("!I", read_exact(stream, 4))[0]
    if not 0 < length <= MAX_RPC_BYTES:
        raise ValueError("owned actor RPC exceeds byte budget")
    return json.loads(read_exact(stream, length))


def send(stream, value):
    data = json.dumps(value, allow_nan=False, separators=(",", ":")).encode()
    if not 0 < len(data) <= MAX_RPC_BYTES:
        raise ValueError("owned actor response exceeds byte budget")
    stream.sendall(struct.pack("!I", len(data)) + data)


def output_tokens(output, prompt_ids, eos_ids):
    """Read exact raw token IDs and processed sampled probabilities, including EOS."""
    if not output.finished or output.prompt_token_ids != prompt_ids or len(output.outputs) != 1:
        raise ValueError("vLLM request output is unfinished or its actual prompt differs")
    choice = output.outputs[0]
    ids = list(choice.token_ids)
    probabilities = choice.logprobs
    if not ids or probabilities is None or len(probabilities) != len(ids):
        raise ValueError("vLLM did not return aligned actual sampled token probabilities")
    values = []
    for token, row in zip(ids, probabilities):
        if type(token) is not int or token < 0 or row is None or token not in row:
            raise ValueError("vLLM sampled token is absent from its raw probability row")
        value = float(row[token].logprob)
        if not math.isfinite(value) or value > 1e-6:
            raise ValueError("vLLM sampled probability is invalid")
        values.append(value)
    if choice.finish_reason not in {"stop", "length"}:
        raise ValueError("vLLM returned an unsupported sampling termination")
    stop_count = int(choice.finish_reason == "stop" and ids[-1] in eos_ids)
    return {"context_token_ids": list(output.prompt_token_ids), "action_token_ids": ids,
            "behavior_log_probs": values, "finish_reason": "eos" if stop_count else choice.finish_reason,
            "stop_token_count": stop_count, "logprob_convention": "sampled_distribution"}


class ActorEngine:
    def __init__(self, bootstrap):
        import vllm
        import vllm.envs
        from vllm import LLM
        from vllm.config.compilation import CompilationConfig, CUDAGraphMode
        if vllm.__version__ != "0.31.0" or vllm.envs.VLLM_ALLOW_INSECURE_SERIALIZATION:
            raise RuntimeError("owned actor requires the qualified vLLM 0.31.0 runtime")
        self.bootstrap = bootstrap
        self.ack = self.activation = None
        started = time.monotonic()
        self.llm = LLM(model=bootstrap["model_dir"], tokenizer=bootstrap["model_dir"],
            skip_tokenizer_init=True, tensor_parallel_size=1, dtype=bootstrap["dtype"],
            quantization=None, seed=bootstrap["seed"], trust_remote_code=False,
            gpu_memory_utilization=.70, kv_cache_memory_bytes=bootstrap["kv_cache_memory_bytes"],
            max_model_len=bootstrap["max_model_len"], max_num_seqs=1,
            enable_prefix_caching=False, enforce_eager=False, logprobs_mode="processed_logprobs",
            kernel_config=dict(VLLM_KERNEL_CONFIG),
            worker_extension_cls=VLLM_WORKER_EXTENSION,
            compilation_config=CompilationConfig(mode=0, cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
                                                  cudagraph_capture_sizes=[1]),
            disable_log_stats=True, distributed_executor_backend="mp")
        self.startup_seconds = time.monotonic() - started
        self.workers = self.llm.collective_rpc(VLLM_WORKER_RPC, args=("observe",), timeout=120)
        if len(self.workers) != 1 or self.workers[0]["graph"]["captured_graph_count"] < 1:
            raise RuntimeError("owned vLLM engine did not establish actual CUDA Graph capture")
        self.source_hashes = {}
        for name in ("vllm", "vllm.v1.sample.sampler", "vllm.compilation.cuda_graph"):
            import importlib
            module = importlib.import_module(name)
            self.source_hashes[name] = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()

    def receipt(self):
        self.workers = self.llm.collective_rpc(VLLM_WORKER_RPC, args=("stats",), timeout=120)
        if len(self.workers) != 1 or self.workers[0]["kernel_config"] != VLLM_KERNEL_CONFIG:
            raise RuntimeError("actual worker kernel configuration changed")
        return {"schema_version": 1, "serving_profile": "vllm-training-v1", "version": "0.31.0",
            "tensor_parallel_size": 1, "logprobs_mode": "processed_logprobs", "prefix_caching": False,
            "enforce_eager": False, "cudagraph_capture_sizes": [1], "workers": self.workers,
            "compilation_mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY",
            "kernel_config": self.workers[0]["kernel_config"],
            "startup_adapter": self.workers[0]["startup_adapter"],
            "worker_rpc_method": self.workers[0]["worker_rpc_method"],
            "insecure_serialization": self.workers[0]["insecure_serialization"],
            "actor_process": process_identity(), "engine_startup_seconds": self.startup_seconds,
            "engine_startup_includes_compile_warmup_capture": True,
            "kv_cache_memory_bytes": self.bootstrap["kv_cache_memory_bytes"],
            "kv_bytes_per_token": self.bootstrap["kv_bytes_per_token"],
            "full_context_raw_kv_bytes": self.bootstrap["kv_bytes_per_token"] * self.bootstrap["max_model_len"],
            "gpu_memory_utilization": .70, "max_model_len": self.bootstrap["max_model_len"],
            "source_sha256": self.source_hashes, "loaded_policy_ack": self.ack,
            "loaded_trainable_tensors": ({**self.activation["loaded_trainable_tensors"],
                "shape": self.activation["shape"], "dtype": self.activation["dtype"]} if self.activation else None)}

    def load(self, artifact):
        receipts = self.llm.collective_rpc(VLLM_WORKER_RPC, args=("load_head", artifact), timeout=300)
        expected = {key: artifact[key] for key in ("version", "digest", "lineage")}
        if len(receipts) != 1 or any(item["policy_ack"] != expected for item in receipts):
            raise RuntimeError("vLLM workers did not acknowledge the exact actual head artifact")
        if self.llm.reset_prefix_cache() is not True:
            raise RuntimeError("vLLM did not acknowledge prefix cache invalidation")
        version = json.dumps(expected, sort_keys=True, separators=(",", ":"))
        self.llm.update_weight_version(version)
        if self.llm.get_weight_version() != version:
            raise RuntimeError("vLLM weight version differs from the actual loaded artifact")
        self.ack, self.activation = expected, receipts[0]
        return {"ack": expected, "activation_receipt": receipts[0], "engine": self.receipt()}

    def generate(self, request):
        from vllm import SamplingParams
        if self.ack != request["policy_ack"]:
            raise RuntimeError("vLLM request does not match its actually loaded head")
        prompt = request["prompt_ids"]
        budget = request["max_tokens"]
        if (not prompt or any(type(token) is not int or token < 0 for token in prompt)
                or type(budget) is not int or budget < 1
                or len(prompt) + budget > self.bootstrap["max_model_len"]):
            raise ValueError("invalid or truncated vLLM context/action budget")
        before = self.receipt()
        params = SamplingParams(n=1, temperature=request["temperature"], top_p=1., top_k=-1,
            min_p=0., presence_penalty=0., frequency_penalty=0., repetition_penalty=1.,
            seed=request["seed"], max_tokens=budget, logprobs=0,
            stop_token_ids=[] if request.get("ignore_eos", False) else self.bootstrap["eos_token_ids"],
            # Stop only on the resolved configured IDs. vLLM otherwise unions
            # model/generation EOS defaults into an explicit override or [].
            ignore_eos=True,
            detokenize=False)
        started = time.monotonic()
        outputs = self.llm.generate({"prompt_token_ids": prompt}, params, use_tqdm=False)
        elapsed = time.monotonic() - started
        if len(outputs) != 1:
            raise RuntimeError("vLLM must return one actual request output")
        actual = output_tokens(outputs[0], prompt, self.bootstrap["eos_token_ids"])
        after = self.receipt()
        old, new = before["workers"][0], after["workers"][0]
        if any(old[key] != new[key] for key in ("pid", "start_ticks", "cmdline_sha256", "device_uuid")):
            raise RuntimeError("vLLM worker identity changed during a request")
        raw_metrics = getattr(outputs[0], "metrics", None)
        timestamps = {name: getattr(raw_metrics, name, None) for name in
            ("queued_ts", "scheduled_ts", "first_token_ts", "last_token_ts")}
        measured = all(isinstance(value, (int, float)) and math.isfinite(value) and value > 0
                       for value in timestamps.values())
        actual["generation_metrics"] = {"elapsed_seconds": elapsed, "context_tokens": len(prompt),
            "action_tokens": len(actual["action_token_ids"]), "clock_domain": "vllm_actor_monotonic",
            "completion_barrier": "finished_vllm_RequestOutput_tokens_and_logprobs_on_cpu",
            "prefill_decode_separation": "engine_core_monotonic" if measured else "not_measured",
            "engine_core_timestamps": timestamps if measured else None,
            "engine_prefill_seconds": timestamps["first_token_ts"] - timestamps["scheduled_ts"] if measured else None,
            "engine_decode_seconds": timestamps["last_token_ts"] - timestamps["first_token_ts"] if measured else None,
            "engine_queue_seconds": timestamps["scheduled_ts"] - timestamps["queued_ts"] if measured else None,
            "sampling_seed": request["seed"],
            "sampling_rng": "vllm_engine_seeded_sampler", "graph_captured_count": new["graph"]["captured_graph_count"],
            "graph_replay_calls_before": old["graph"]["replay_calls"],
            "graph_replay_calls_after": new["graph"]["replay_calls"],
            "graph_replay_calls_delta": new["graph"]["replay_calls"] - old["graph"]["replay_calls"],
            "worker_pid": new["pid"], "device_uuid": new["device_uuid"], "engine_receipt": after}
        return actual

    def close(self):
        self.llm.llm_engine.engine_core.shutdown(timeout=60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    args = parser.parse_args()
    token = args.token_file.read_text().strip()
    bootstrap = json.loads(args.bootstrap.read_text())
    actor = ActorEngine(bootstrap)
    deadline = time.monotonic() + min(30000, bootstrap["ttl_seconds"])
    closed = False
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(str(args.socket))
        os.chmod(args.socket, 0o600)
        server.listen(1)
        server.settimeout(1)
        try:
            while time.monotonic() < deadline and not closed:
                try:
                    connection, _ = server.accept()
                except socket.timeout:
                    continue
                with connection:
                    connection.settimeout(600)
                    operation = None
                    try:
                        request = receive(connection)
                        if not hmac.compare_digest(str(request.pop("token", "")), token):
                            raise ValueError("unauthorized owned actor request")
                        operation = request.pop("operation")
                        if operation == "stats":
                            result = actor.receipt()
                        elif operation == "load":
                            result = actor.load(request["artifact"])
                        elif operation == "generate":
                            result = actor.generate(request)
                        elif operation == "close":
                            actor.close()
                            closed = True
                            result = {"closed": True, "drain_complete": True}
                        else:
                            raise ValueError("unsupported actor operation")
                        send(connection, {"ok": True, "result": result})
                    except BaseException as error:
                        # No traceback, arguments or credentials enter RPC evidence.
                        send(connection, {"ok": False, "error_type": type(error).__name__,
                                          "error": str(error)[:1000]})
                        if operation in {"load", "close"}:
                            raise
        finally:
            if not closed:
                actor.close()


if __name__ == "__main__":
    main()
