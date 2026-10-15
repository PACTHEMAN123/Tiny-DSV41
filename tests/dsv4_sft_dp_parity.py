"""Four-rank FSDP2/CP2/EP2 SFT parity against an unsharded data oracle.

Unlike EP=world tests, each expert FSDP subgroup contains two ranks. Two
accumulated microbatches have unequal answer counts across both data groups.
CPU uses FP32. CUDA uses deterministic Torch references with native BF16/FP32
islands on both sides; gradient gates distinguish rounding from wrong scaling.
"""

import argparse
from dataclasses import replace
from datetime import timedelta
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from dsv41_train.engram_offload import install_engram_sgd_hooks
from dsv41_train.lora import LoRAConfig, inject_lora
from dsv41_train.models.dsv4 import DeepSeekV41ForCausalLM
from dsv41_train.models.dsv4.attention import AttentionSinks
from dsv41_train.models.dsv4.mhc import HyperConnection
from dsv41_train.models.dsv4.model import RowShardedEmbedding
from dsv41_train.models.dsv4.moe import RoutedExperts
from dsv41_train.models.dsv4.parallel import apply_fsdp2, build_parallelism, synchronize_cp_gradients
from dsv41_train.runtime import Runtime
from dsv41_train.sft_data import collate_examples
from dsv4_sft_in_backward import config_for
from engram_offload_cuda_parity import reference_torch
from train_sft import train_update


def owner_slice(model, name, value):
    module = model.get_submodule(name.rpartition(".")[0])
    if isinstance(module, RoutedExperts):
        return value[module.expert_start:module.expert_start + module.num_experts]
    if isinstance(module, RowShardedEmbedding):
        return value[module.row_start:module.row_stop]
    return value


def category(name):
    if "engram_tables" in name:
        return "engram"
    if ".lora_" in name:
        return name.rsplit(".", 1)[-1]
    return "experts" if ".moe.routed." in name else "dense_and_fp32"


def native(config, lora, meshes=None, cp=None, dispatcher=None, source=None,
           *, device=torch.device("cpu"), bf16=False):
    torch.manual_seed(47)
    model = DeepSeekV41ForCausalLM(config, cp, dispatcher, None if meshes is None else meshes.engram,
                                   sparse_engram_gradients=True, engram_offload=True)
    if lora:
        inject_lora(model, LoRAConfig(rank=4, alpha=8))
        model.model.engram_tables.requires_grad_(True)
    if bf16:
        fp32 = {id(p) for module in model.modules() if isinstance(module, (HyperConnection, AttentionSinks))
                for p in module.parameters(recurse=False)}
        for name, parameter in model.named_parameters():
            parameter.data = parameter.data.to(device="cpu" if ".engram_tables." in name else device,
                dtype=torch.float32 if id(parameter) in fp32 else torch.bfloat16)
        for module in model.modules():
            for name, buffer in module.named_buffers(recurse=False):
                setattr(module, name, buffer.to(device))
    if source is not None:
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                parameter.copy_(owner_slice(model, name, source.get_parameter(name)))
            for name, buffer in model.named_buffers():
                buffer.copy_(source.get_buffer(name))
    model.gradient_checkpointing_enable()
    if meshes is not None:
        apply_fsdp2(model, meshes)
        for module in model.modules():
            if isinstance(module, RoutedExperts):
                assert isinstance(module.gate_up, DTensor)
                assert module.gate_up.device_mesh.size() == 2
                assert module.num_experts * 2 == module.global_num_experts
        for table in model.model.engram_tables.values():
            assert table.size == 4 and table.weight.device.type == "cpu"
            assert not isinstance(table.weight, DTensor)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.05, foreach=False)
    install_engram_sgd_hooks(model, optimizer)
    return model, optimizer


def materialized(value):
    if isinstance(value, DTensor):
        value = value.full_tensor()
    return value.float().coalesce().to_dense() if value.is_sparse else value.float()


def compare(model, reference, *, gradients=False, device=torch.device("cpu"), before=None):
    errors, magnitudes, squares, budget_fractions = {}, {}, {}, {}
    for name, parameter in model.named_parameters():
        expected = reference.get_parameter(name)
        actual = parameter.grad if gradients else parameter.detach()
        expected = expected.grad if gradients else expected.detach()
        if actual is None or expected is None:
            assert actual is expected is None, name
            continue
        actual = materialized(actual)
        expected = owner_slice(model, name, materialized(expected))
        key = category(name)
        difference = (actual - expected).abs()
        if device.type == "cpu":
            torch.testing.assert_close(actual, expected, rtol=4e-4, atol=2e-7, msg=name)
        elif not gradients:
            if parameter.grad is None:
                torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=name)
            else:
                old_actual, old_expected = before[name]
                gradient = materialized(parameter.grad)
                target = owner_slice(model, name, materialized(reference.get_parameter(name).grad))
                # SGD error is bounded by prior trajectory error, LR times
                # gradient error, and rounding of multiply/add on either side.
                # This uses each parameter's BF16 or FP32 epsilon, not a loose
                # relative tolerance against the much larger base weight.
                rounding = 2 * torch.finfo(parameter.dtype).eps * (
                    old_actual.abs() + old_expected.abs() + .05 * (gradient.abs() + target.abs()))
                budget = (old_actual - old_expected).abs() + .05 * (gradient - target).abs() + rounding
                assert bool((difference <= budget).all()), f"{name}: SGD update exceeds rounding budget"
                budget_fractions[key] = max(budget_fractions.get(key, 0.),
                    (difference / budget.clamp_min(torch.finfo(torch.float32).tiny)).max().item())
        pair = squares.setdefault(key, [0., 0.])
        pair[0] += difference.double().square().sum().item()
        pair[1] += expected.double().square().sum().item()
        errors[key] = max(errors.get(key, 0.), (actual - expected).abs().max().item())
        magnitudes[key] = max(magnitudes.get(key, 0.), expected.abs().max().item())
    for values in (errors, magnitudes, budget_fractions):
        if not values:
            continue
        tensor = torch.tensor(list(values.values()), device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        values.update(zip(values, tensor.tolist()))
    norms = torch.tensor(list(squares.values()), dtype=torch.float64, device=device)
    dist.all_reduce(norms)
    relative_l2 = dict(zip(squares, (norms[:, 0] / norms[:, 1].clamp_min(1e-300)).sqrt().tolist()))
    relative_max = {key: errors[key] / max(magnitudes[key], 1e-300) for key in errors}
    if gradients and device.type == "cuda":
        # BF16 epsilon is 1/128. Accumulation and FSDP reduction reorder several
        # roundings: permit 4 eps in group L2 and 8 eps in group max norm. A 2x
        # expert/DP scaling error is 100%, well beyond either explicit budget.
        eps = torch.finfo(torch.bfloat16).eps
        assert all(value <= 4 * eps for value in relative_l2.values()), relative_l2
        assert all(value <= 8 * eps for value in relative_max.values()), relative_max
    return dict(maximum_errors=errors, oracle_maximum_magnitudes=magnitudes,
                relative_l2_errors=relative_l2, relative_max_errors=relative_max,
                maximum_update_budget_fraction=budget_fractions)


def parity(lora, meshes, cp, dispatcher, runtime):
    cuda = runtime.device.type == "cuda"
    config = replace(config_for(runtime.device.type), num_hidden_layers=2, compress_ratios=[0, 2],
                     kv_source_layer_ids=[1], index_source_layer_ids=[1], candidate_source_layer_id=1,
                     engram_num_embeddings=[4097])
    options = dict(device=runtime.device, bf16=cuda)
    reference, reference_optimizer = native(config, lora, **options)
    model, optimizer = native(config, lora, meshes, cp, dispatcher, reference, **options)
    counts = ((1, 3), (2, 5))
    oracle_batches, local_batches = [], []
    for micro, group_counts in enumerate(counts):
        examples = []
        for group, count in enumerate(group_counts):
            tokens = [1, 4 + group, 7 + micro, 8, 9, 3, 6, 2]
            if cuda:
                tokens = [1, 4, 7, 8, 9, 3, 6, 5] + tokens
            examples.append(dict(input_ids=tokens, labels=[-100] * (len(tokens) - count) + tokens[-count:],
                                 attention_mask=[1] * len(tokens)))
        oracle_batches.append(collate_examples(examples, pad_token_id=0))
        local_batches.append(collate_examples([examples[runtime.rank // 2]], pad_token_id=0))
    reports = []
    # Capture after native normalization/coalescing and before SGD consumes the
    # gradient. LoRA step two exercises A as well as initially-zero B.
    gradient_report = {}
    optimizer.register_step_pre_hook(lambda *_: gradient_report.update(
        compare(model, reference, gradients=True, device=runtime.device)))
    for step in (1, 2):
        before = {name: (materialized(parameter.detach()).clone(),
                        owner_slice(model, name, reference.get_parameter(name).detach()).float().clone())
                  for name, parameter in model.named_parameters()} if cuda else None
        expected = train_update(reference, reference_optimizer, oracle_batches,
                                family="dsv4", runtime=Runtime(runtime.device))
        actual = train_update(model, optimizer, local_batches, family="dsv4", runtime=runtime,
                              cp_size=2, synchronize=lambda: synchronize_cp_gradients(model, cp))
        assert actual["supervised_tokens"] == expected["supervised_tokens"] == 11
        loss_tolerance = torch.finfo(torch.bfloat16).eps * max(1., abs(expected["loss"])) if cuda else 1e-6
        assert abs(actual["loss"] - expected["loss"]) < loss_tolerance, (actual, expected)
        updates = compare(model, reference, device=runtime.device, before=before)
        required = ("engram", "lora_a", "lora_b") if lora else ("engram", "experts", "dense_and_fp32")
        if step == 2:
            assert all(gradient_report["oracle_maximum_magnitudes"][name] > 1e-10 for name in required)
        reports.append(dict(step=step, loss=actual["loss"], loss_error=abs(actual["loss"] - expected["loss"]),
                            gradients=dict(gradient_report), parameters=updates))
    return dict(mode="lora_and_trainable_engram" if lora else "full", cp=2, ep=2, world=4,
                device=runtime.device.type, precision="bf16_with_fp32_islands" if cuda else "fp32",
                backend="torch_reference_nccl" if cuda else "torch_gloo", oracle="same_precision_unsharded",
                tolerances=(dict(gradient_relative_l2=4 / 128, gradient_relative_max=8 / 128,
                                 update="prior_error + lr*gradient_error + 2*dtype_eps*(old_weights + lr*gradients)",
                                 loss="bf16_eps * max(1, oracle_loss)") if cuda else dict(rtol=4e-4, atol=2e-7, loss=1e-6)),
                expert_fsdp_size=2, actual_fsdp=True, accumulation_steps=2,
                supervised_tokens_by_microbatch_and_data_group=counts, steps=reports)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="New JSON report path")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--check-setup", action="store_true", help="Check the selected fixture precision without distributed startup")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.check_setup:
        for lora in (False, True):
            model, _ = native(config_for(args.device), lora, bf16=args.device == "cuda")
            counts = {}
            for parameter in model.parameters():
                assert bool(torch.isfinite(parameter).all())
                counts[str(parameter.dtype)] = counts.get(str(parameter.dtype), 0) + parameter.numel()
            if args.device == "cuda":
                assert counts.keys() == {"torch.float32", "torch.bfloat16"}
            print(json.dumps(dict(event="setup_passed", lora=lora, parameter_count_by_dtype=counts)), flush=True)
        return
    if not args.output:
        parser.error("--output is required unless --check-setup is used")
    device = torch.device("cpu")
    if args.device == "cuda":
        reference_torch()
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        torch.cuda.set_device(device)
    dist.init_process_group("nccl" if args.device == "cuda" else "gloo", timeout=timedelta(seconds=180))
    try:
        assert dist.get_world_size() == 4, "launch with torchrun --nproc-per-node=4"
        meshes, cp, dispatcher = build_parallelism(2, 2, args.device)
        runtime = Runtime(device, rank=dist.get_rank(), world_size=4)
        reports = [parity(lora, meshes, cp, dispatcher, runtime) for lora in (False, True)]
        if runtime.is_main:
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x") as destination:
                json.dump(reports, destination, indent=2)
            print(json.dumps(reports, indent=2), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
