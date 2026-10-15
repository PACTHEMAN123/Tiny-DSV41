"""Two-rank CPU parity for token-weighted accumulation with CP1/EP2 and CP2/EP2.

Uses real expert routing and uneven Engram row ownership; dense FSDP averaging
is performed explicitly so this regression needs only Gloo, not CUDA.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard

from dsv41_train.cp import ContextParallel
from dsv41_train.models.dsv4.attention import AttentionSinks
from dsv41_train.models.dsv4.config import DeepSeekV41Config
from dsv41_train.models.dsv4.model import RowShardedEmbedding
from dsv41_train.models.dsv4.moe import AllToAllTokenDispatcher, RoutedExperts, TokenDispatcher
from dsv41_train.models.dsv4.parallel import synchronize_cp_gradients
from train_dsv4_checkpoint import install_sgd_in_backward


def parity(mesh, cp_size):
    rank, world = dist.get_rank(), dist.get_world_size()
    cp = ContextParallel(mesh if cp_size > 1 else None)
    config = replace(DeepSeekV41Config.tiny(), hidden_size=4,
                     moe_intermediate_size=3, n_routed_experts=2, num_experts_per_tok=1)
    generator = torch.Generator().manual_seed(117)
    reference, actual = nn.Module(), nn.Module()
    reference.experts = RoutedExperts(config, TokenDispatcher())
    actual.experts = RoutedExperts(config, AllToAllTokenDispatcher(mesh))
    reference.table = nn.Embedding(11, 4)
    actual.table = RowShardedEmbedding(11, 4, mesh, sparse_gradients=True)
    for model in (reference, actual):
        model.dense = nn.Parameter(torch.tensor(1.7))
        model.sinks = AttentionSinks(1)
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.2)
        reference.dense.fill_(1.7)
        reference.sinks.weight.fill_(0.8)
        for name, parameter in actual.named_parameters():
            source = dict(reference.named_parameters())[name]
            if name.startswith("experts."):
                source = source[actual.experts.expert_start:actual.experts.expert_start + 1]
            elif name == "table.weight":
                source = source[actual.table.row_start:actual.table.row_stop]
            parameter.copy_(source)

    # Unequal answer counts across ranks and microbatches, including a rank with
    # zero valid answers that must still participate in all backward collectives.
    masks = ((torch.tensor([1, 0]), torch.tensor([1, 1, 1])),
             (torch.tensor([0, 0]), torch.tensor([0, 1, 1])))
    total = sum(int(mask.sum()) for microbatch in masks for mask in microbatch)
    for microbatch in masks:
        inputs = [torch.randn(len(mask), 4, generator=generator) for mask in microbatch]
        indices = [torch.tensor([0, 7]), torch.tensor([0, 3, 10])]
        experts = [torch.tensor([[0], [1]]), torch.tensor([[1], [0], [1]])]
        weights = [torch.rand(len(mask), 1, generator=generator) for mask in microbatch]

        def forward(model, x, rows, expert_ids, router_weights):
            hidden = (x + model.table(rows)) * model.dense * model.sinks(x)
            return model.experts(hidden, expert_ids, router_weights)

        expected = forward(reference, torch.cat(inputs), torch.cat(indices),
                           torch.cat(experts), torch.cat(weights))
        (expected.square().sum(-1) * torch.cat(microbatch)).sum().div(total).backward()
        output = forward(actual, inputs[rank], indices[rank], experts[rank], weights[rank])
        offset = 0 if rank == 0 else len(microbatch[0])
        torch.testing.assert_close(output, expected.detach()[offset:offset + len(microbatch[rank])])
        count = microbatch[rank].sum()
        loss_sum = (output.square().sum(-1) * microbatch[rank]).sum()
        loss = cp.mean_loss(loss_sum, count)
        group_count = cp.sum(count)
        # Model backward is CP * local_sum / group_count. Global dense FSDP
        # averages over world, hence this multiplier gives sum(all losses)/total.
        (loss * (world / cp_size) * group_count / total).backward()

    reference_parameters = dict(reference.named_parameters())
    for name, parameter in actual.experts.named_parameters():
        expected = reference_parameters["experts." + name].grad[rank:rank + 1]
        torch.testing.assert_close(parameter.grad, expected * world)
    # CP1 FP32 islands are also globally FSDP sharded; CP2 leaves them replicated
    # and synchronize_cp_gradients must reduce them separately.
    dense_parameters = [actual.dense] + ([actual.sinks.weight] if cp_size == 1 else [])
    for parameter in dense_parameters:
        dist.all_reduce(parameter.grad)
        parameter.grad.div_(world)
    synchronize_cp_gradients(actual, cp)
    errors = {}
    for name, parameter in actual.named_parameters():
        expected = reference_parameters[name].grad
        if name.startswith("experts."):
            expected = expected[rank:rank + 1]
        elif name == "table.weight":
            expected = expected[actual.table.row_start:actual.table.row_stop]
        gradient = parameter.grad.to_dense() if parameter.grad.is_sparse else parameter.grad
        torch.testing.assert_close(gradient, expected, rtol=2e-4, atol=2e-7,
                                   msg=lambda message, name=name: f"{name}: {message}")
        error = (gradient - expected).abs().max()
        dist.all_reduce(error, op=dist.ReduceOp.MAX)
        errors[name] = error.item()
    return dict(cp=cp_size, ep=world, accumulation_steps=2,
                global_answer_tokens=total, maximum_gradient_errors=errors)


def sgd_in_backward_parity(mesh):
    # EP2 gives each rank different experts; their FSDP subgroup has size one.
    subgroup = init_device_mesh("cpu", (2, 1), mesh_dim_names=("ep", "expert_fsdp"))["expert_fsdp"]
    config = replace(DeepSeekV41Config.tiny(), hidden_size=4,
                     moe_intermediate_size=3, n_routed_experts=2)
    model = nn.Module()
    model.experts = RoutedExperts(config, AllToAllTokenDispatcher(mesh))
    model.dense = nn.Parameter(torch.ones(2))
    initial, gradients = {}, {}
    for name, parameter in list(model.named_parameters()):
        initial[name] = torch.ones_like(parameter)
        tensor = DTensor.from_local(initial[name].clone(), subgroup, [Shard(0)])
        parent, _, leaf = name.rpartition(".")
        setattr(model.get_submodule(parent), leaf, nn.Parameter(tensor))
        gradients[name] = DTensor.from_local(torch.full_like(parameter, 1.0 + dist.get_rank()),
                                             subgroup, [Shard(0)])
    installed = install_sgd_in_backward(model, 0.1)
    assert installed == {id(parameter) for parameter in model.parameters()}
    sum((parameter * gradients[name]).sum() for name, parameter in model.named_parameters()).backward()
    for name, parameter in model.named_parameters():
        divisor = 2 if name.startswith("experts.") else 1
        expected = initial[name] - 0.1 * gradients[name].to_local() / divisor
        torch.testing.assert_close(parameter.to_local(), expected)
        assert parameter.grad is None
    return dict(optimizer_in_backward="sgd", ep=2, expert_divisor=2, gradients_cleared=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    args = parser.parse_args()
    torch.set_num_threads(1)
    dist.init_process_group("gloo")
    try:
        if dist.get_world_size() != 2:
            raise RuntimeError("this parity test requires exactly two ranks")
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("ep",))
        results = [parity(mesh, size) for size in (1, 2)]
        results.append(sgd_in_backward_parity(mesh))
        if dist.get_rank() == 0:
            report = json.dumps(results, indent=2) + "\n"
            if args.output:
                path = Path(args.output)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(report)
            print(report, flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
