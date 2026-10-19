"""Atomic group admission for the supported vLLM synchronous engine."""
from __future__ import annotations

import hashlib
from pathlib import Path

# Private FIFO and request-state APIs must match the inspected vLLM runtime.
ADMISSION_SOURCE_SHA256 = {
    "entrypoints/llm.py": "daa6aeb5e971a66ccf4cba24390a3ba6c450e7bd9048942ac1c39f0dd54f237f",
    "entrypoints/offline_utils.py": "582149546697619665c5cf93a0e54ab367cac07830c7efb1f4412cb91f6bd65e",
    "v1/engine/core.py": "f116962c35d9d312b20b0b76c34a8cdaa6d3f1d31f2036db4e123957d2d1555b",
    "v1/engine/core_client.py": "24eecc78e8fc4849d4eac948c7586ee349ebfd503d07e80f05ec6c859da06fdd",
    "v1/engine/llm_engine.py": "fd838313336aff1734a435f606bbdc705ee7efac94c479e26c2cc87eee0f82a8",
    "v1/engine/input_processor.py": "d79601e4a053a7bacb0d6a23e273348435e4ab70c6a315dcccb2d428ed57d96b",
    "v1/engine/output_processor.py": "c72d0149e18d6da71d53e23b6550d09310af6917523fef6e282b0907d36b8d54",
    "v1/core/sched/scheduler.py": "f820bee4f42b878b262a30f3728a97d7413477b2a277748e20f2528ac2ec2f2a",
    "v1/request.py": "dd7c30bf8b0894dc123a739c299c0133566f8cc075b081e45905f1c29506e69d",
}


def qualify_admission_runtime(vllm, llm_class):
    root = Path(vllm.__file__).resolve().parent
    if (vllm.__version__ != "0.31.0"
            or any(hashlib.sha256((root / name).read_bytes()).hexdigest() != digest
                   for name, digest in ADMISSION_SOURCE_SHA256.items())
            or any(not callable(getattr(llm_class, name, None))
                   for name in ("enqueue", "wait_for_completion"))):
        raise RuntimeError("atomic rollout admission requires the supported vLLM scheduling runtime")


def enqueue_before_schedule(llm, prompts, params, lora_request):
    """Pause scheduling, enqueue the whole group through one FIFO, then resume.

    A utility reply after enqueue acknowledges consumption of all preceding
    ADD messages while still paused. Request state binds internal IDs to the
    public output order. Always attempt resume after a possibly successful
    pause, including unknown RPC outcomes; callers latch failures, never retry.
    """
    engine = llm.llm_engine
    utility = engine.engine_core.call_utility
    if engine.has_unfinished_requests():
        raise RuntimeError("atomic rollout admission requires an empty owned engine")
    if utility("is_scheduler_paused") is not False:
        raise RuntimeError("atomic rollout admission found a pre-existing scheduler pause")
    try:
        utility("pause_scheduler", "keep", False)
        if utility("is_scheduler_paused") is not True:
            raise RuntimeError("rollout scheduler did not acknowledge keep pause")
        ids = llm.enqueue(prompts, params, use_tqdm=False, lora_request=lora_request)
        if (not isinstance(ids, list) or len(ids) != len(prompts)
                or any(not isinstance(value, str) or not value for value in ids)
                or len(set(ids)) != len(ids)):
            raise RuntimeError("vLLM enqueue request identity/order differs")
        states = engine.output_processor.request_states
        if set(states) != set(ids):
            raise RuntimeError("vLLM enqueued state membership differs")
        output_ids = [states[value].external_req_id for value in ids]
        if (any(not isinstance(value, str) or not value.isdecimal() for value in output_ids)
                or len(set(output_ids)) != len(ids)
                or list(map(int, output_ids)) != sorted(map(int, output_ids))
                or any(states[value].request_id != value
                       or states[value].prompt_token_ids != prompt["prompt_token_ids"]
                       or states[value].lora_request != lora_request
                       for value, prompt in zip(ids, prompts))):
            raise RuntimeError("vLLM admitted internal/external request mapping differs")
        if utility("is_scheduler_paused") is not True:
            raise RuntimeError("rollout scheduler resumed before every request was admitted")
    finally:
        utility("resume_scheduler")
        if utility("is_scheduler_paused") is not False:
            raise RuntimeError("rollout scheduler did not acknowledge resume")
    outputs = llm.wait_for_completion(use_tqdm=False)
    if [getattr(output, "request_id", None) for output in outputs] != output_ids:
        raise RuntimeError("vLLM completed request IDs differ from admitted request IDs")
    return outputs
