"""Two-rank Qwen EP parity when every token is routed to one owner.

Run with torchrun --nproc-per-node=2; CPU Gloo only, no FSDP or CUDA required.
"""

from datetime import timedelta
import json

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from dsv41_train.dispatch import AllToAllTokenDispatcher, TokenDispatcher
from dsv41_train.models.qwen.model import Qwen3MoeSparseMoeBlock
from qwen_helpers import tiny_config


def check_case(mesh, *, owner, frozen):
    torch.manual_seed(2031)
    config = tiny_config()
    reference = Qwen3MoeSparseMoeBlock(config, TokenDispatcher())
    parallel = Qwen3MoeSparseMoeBlock(config, AllToAllTokenDispatcher(mesh))
    experts = parallel.experts
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.normal_(std=.1)
        reference.gate.weight[:, 0] = -5
        reference.gate.weight[owner * 2:owner * 2 + 2, 0] = 5
        parallel.gate.weight.copy_(reference.gate.weight)
        for name, parameter in experts.named_parameters():
            parameter.copy_(getattr(reference.experts, name)[experts.expert_start:experts.expert_start + experts.num_experts])
    reference.requires_grad_(not frozen)
    parallel.requires_grad_(not frozen)
    full_input = torch.randn(1, 5, config.hidden_size) * .2
    full_input[:, :, 0] = 1
    full_input.requires_grad_()
    start, stop = ((0, 2), (2, 5))[dist.get_rank()]
    local_input = full_input[:, start:stop].detach().clone().requires_grad_()
    reference_output, reference_logits = reference(full_input)
    output, router_logits = parallel(local_input)
    reference_logits.retain_grad()
    router_logits.retain_grad()
    assert torch.all(router_logits.topk(2, dim=-1).indices // 2 == owner)
    torch.testing.assert_close(output, reference_output[:, start:stop], rtol=1e-5, atol=1e-7)
    reference_output.square().sum().backward()
    output.square().sum().backward()
    torch.testing.assert_close(local_input.grad, full_input.grad[:, start:stop], rtol=2e-5, atol=2e-7)
    torch.testing.assert_close(router_logits.grad, reference_logits.grad[start:stop], rtol=2e-5, atol=2e-7)
    assert local_input.grad.abs().sum() > 0
    assert router_logits.grad.abs().sum() > 0
    expert_error = 0.0
    for name, parameter in experts.named_parameters():
        if frozen:
            assert parameter.grad is None
        else:
            expected = getattr(reference.experts, name).grad[experts.expert_start:experts.expert_start + experts.num_experts]
            assert parameter.grad is not None
            torch.testing.assert_close(parameter.grad, expected, rtol=2e-5, atol=2e-7)
            expert_error = max(expert_error, (parameter.grad - expected).abs().max().item())
            if dist.get_rank() != owner:
                assert parameter.grad.count_nonzero() == 0
    if frozen:
        assert parallel.gate.weight.grad is None
    else:
        dist.all_reduce(parallel.gate.weight.grad)
        torch.testing.assert_close(parallel.gate.weight.grad, reference.gate.weight.grad, rtol=2e-5, atol=2e-7)
    return dict(owner=owner, frozen=frozen,
                input_gradient_max_abs=(local_input.grad - full_input.grad[:, start:stop]).abs().max().item(),
                router_gradient_max_abs=(router_logits.grad - reference_logits.grad[start:stop]).abs().max().item(),
                expert_gradient_max_abs=expert_error)


def main():
    torch.set_num_threads(1)
    dist.init_process_group("gloo", timeout=timedelta(seconds=60))
    try:
        if dist.get_world_size() != 2:
            raise RuntimeError("this parity check requires exactly two ranks")
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("ep",))
        results = [check_case(mesh, owner=owner, frozen=frozen)
                   for frozen in (False, True) for owner in (0, 1)]
        gathered = [None, None]
        dist.all_gather_object(gathered, results)
        if dist.get_rank() == 0:
            print(json.dumps({"qwen_empty_expert_parity": "passed", "ranks": gathered}), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
