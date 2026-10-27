"""Two-rank CUDA check of stale Token-TIS shards against separate full models.

Run: torchrun --standalone --nproc-per-node=2 -m tests.distributed_grpo_async_fsdp
"""
import json
import os
import types

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

from dsv41_train.rl.algorithm import decoupled_token_tis_loss, response_log_probs_batch
from dsv41_train.rl.distributed_grpo import train_group
from dsv41_train.rl.native_lora import adapter_snapshot, scoring_model_class
from tests.test_distributed_grpo import NativeBase


def check(dtype, device, mesh):
    torch.manual_seed(41)
    model = scoring_model_class(NativeBase)().to(device=device, dtype=dtype)
    old = scoring_model_class(NativeBase)().to(device=device, dtype=dtype)
    old.load_state_dict(model.state_dict())
    oracle = scoring_model_class(NativeBase)().to(device=device, dtype=dtype)
    oracle.load_state_dict(model.state_dict())
    fully_shard(model.model.adapter, mesh=mesh, reshard_after_forward=True)
    fully_shard(model, mesh=mesh, reshard_after_forward=True)
    parameters = [p for p in model.parameters() if p.requires_grad]
    reference_parameters = [p for p in oracle.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=.03, weight_decay=0, foreach=False)
    reference_optimizer = torch.optim.AdamW(reference_parameters, lr=.03, weight_decay=0, foreach=False)
    ack = {"version": 0, "digest": "old", "lineage": "fsdp-test", "adapter_id": 1}
    current_ack = {**ack, "version": 1, "digest": "current", "adapter_id": 2}
    snapshot = {"ack": ack, "parameters": adapter_snapshot(optimizer)}
    for current_optimizer in (optimizer, reference_optimizer):
        for group in current_optimizer.param_groups:
            for parameter in group["params"]:
                parameter.grad = torch.full_like(parameter, .05)
        current_optimizer.step()
        current_optimizer.zero_grad(set_to_none=True)
    prompts = [[0, 2]] * 32
    responses = [[3, 4] if index % 2 == 0 else [5, 6] for index in range(32)]
    advantages = [1., -1.] * 16
    results, expected_loss = [], 0.
    for start in range(0, 32, 4):
        with torch.no_grad():
            old_values = response_log_probs_batch(old, prompts[start:start + 4], responses[start:start + 4], .8)
        current_values = response_log_probs_batch(oracle, prompts[start:start + 4], responses[start:start + 4], .8)
        terms = []
        for offset, (native_old, current) in enumerate(zip(old_values, current_values)):
            index = start + offset
            behavior = (native_old - .5).tolist()
            results.append({"context_token_ids": prompts[index], "action_token_ids": responses[index],
                "behavior_log_probs": behavior, "generation_metrics": {"policy_ack": ack}})
            loss, _ = decoupled_token_tis_loss([current], [native_old], [behavior], [advantages[index]])
            terms.append(loss)
        loss = torch.stack(terms).sum() / 32
        expected_loss += loss.item()
        loss.backward()
    reference_optimizer.step()
    args = types.SimpleNamespace(temperature=.8, kl_beta=0., clip_eps=.2, rollout_is_cap=2.,
        rollout_correction="decoupled-token-tis", disable_prefetch=False, learner_microbatch_size=4)
    policy = types.SimpleNamespace(begin_step=lambda: None, normalize=lambda: None)
    metrics = train_group(model, optimizer, policy, results, advantages, args, 2,
                          expected_ack=current_ack, native_old_snapshot=snapshot)
    tolerance = {"rtol": 2e-5, "atol": 2e-7} if dtype == torch.float32 else {"rtol": .02, "atol": 2e-4}
    max_gradient_difference = max_parameter_difference = 0.
    for actual, expected in zip(parameters, reference_parameters):
        full_gradient = actual.grad.full_tensor()
        full_parameter = actual.full_tensor()
        torch.testing.assert_close(full_gradient, expected.grad, **tolerance)
        torch.testing.assert_close(full_parameter, expected, **tolerance)
        max_gradient_difference = max(max_gradient_difference, (full_gradient - expected.grad).abs().max().item())
        max_parameter_difference = max(max_parameter_difference, (full_parameter - expected).abs().max().item())
        for name, value in optimizer.state[actual].items():
            full = value.full_tensor() if hasattr(value, "full_tensor") else value
            torch.testing.assert_close(full, reference_optimizer.state[expected][name], **tolerance)
    torch.testing.assert_close(torch.tensor(metrics["loss"]), torch.tensor(expected_loss), **tolerance)
    if metrics["policy_lag"] != 1 or metrics["native_old_policy_ack"] != ack:
        raise AssertionError("stale generating identity was not retained")
    return {"dtype": str(dtype), "world_size": dist.get_world_size(), "policy_lag": 1,
        "gradient_max_abs": max_gradient_difference, "parameter_max_abs": max_parameter_difference,
        "loss": metrics["loss"], "oracle_loss": expected_loss, "passed": True}


def main():
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    dist.init_process_group("nccl", device_id=device)
    try:
        if dist.get_world_size() != 2:
            raise ValueError("this regression requires exactly two CUDA ranks")
        mesh = init_device_mesh("cuda", (2,))
        results = [check(dtype, device, mesh) for dtype in (torch.float32, torch.bfloat16)]
        if dist.get_rank() == 0:
            print(json.dumps({"tests": results, "passed": True}), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
