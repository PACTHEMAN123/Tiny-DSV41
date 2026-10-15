"""Four-rank CPU FSDP2/CP2/EP2 SFT parity against an unsharded data oracle.

Unlike EP=world tests, each expert FSDP subgroup contains two ranks. Two
accumulated microbatches have unequal answer counts across both data groups.
"""

import argparse
from dataclasses import replace
from datetime import timedelta
import json
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from dsv41_train.engram_offload import install_engram_sgd_hooks
from dsv41_train.lora import LoRAConfig, inject_lora
from dsv41_train.models.dsv4 import DeepSeekV41ForCausalLM
from dsv41_train.models.dsv4.model import RowShardedEmbedding
from dsv41_train.models.dsv4.moe import RoutedExperts
from dsv41_train.models.dsv4.parallel import apply_fsdp2, build_parallelism, synchronize_cp_gradients
from dsv41_train.runtime import Runtime
from dsv41_train.sft_data import collate_examples
from dsv4_sft_in_backward import config_for
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


def native(config, lora, meshes=None, cp=None, dispatcher=None, source=None):
    torch.manual_seed(47)
    model = DeepSeekV41ForCausalLM(config, cp, dispatcher, None if meshes is None else meshes.engram,
                                   sparse_engram_gradients=True, engram_offload=True)
    if lora:
        inject_lora(model, LoRAConfig(rank=4, alpha=8))
        model.model.engram_tables.requires_grad_(True)
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


def compare(model, reference, *, gradients=False):
    errors, magnitudes = {}, {}
    for name, parameter in model.named_parameters():
        expected = reference.get_parameter(name)
        actual = parameter.grad if gradients else parameter.detach()
        expected = expected.grad if gradients else expected.detach()
        if actual is None or expected is None:
            assert actual is expected is None, name
            continue
        if isinstance(actual, DTensor):
            actual = actual.full_tensor()
        if actual.is_sparse:
            actual = actual.coalesce().to_dense()
        if expected.is_sparse:
            expected = expected.coalesce().to_dense()
        expected = owner_slice(model, name, expected)
        torch.testing.assert_close(actual, expected, rtol=4e-4, atol=2e-7, msg=name)
        key = category(name)
        errors[key] = max(errors.get(key, 0.), (actual - expected).abs().max().item())
        magnitudes[key] = max(magnitudes.get(key, 0.), expected.abs().max().item())
    for values in (errors, magnitudes):
        tensor = torch.tensor(list(values.values()))
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        values.update(zip(values, tensor.tolist()))
    return dict(maximum_errors=errors, oracle_maximum_magnitudes=magnitudes)


def parity(lora, meshes, cp, dispatcher, runtime):
    config = replace(config_for("cpu"), num_hidden_layers=2, compress_ratios=[0, 2],
                     kv_source_layer_ids=[1], index_source_layer_ids=[1], candidate_source_layer_id=1,
                     engram_num_embeddings=[4097])
    reference, reference_optimizer = native(config, lora)
    model, optimizer = native(config, lora, meshes, cp, dispatcher, reference)
    counts = ((1, 3), (2, 5))
    oracle_batches, local_batches = [], []
    for micro, group_counts in enumerate(counts):
        examples = []
        for group, count in enumerate(group_counts):
            tokens = [1, 4 + group, 7 + micro, 8, 9, 3, 6, 2]
            examples.append(dict(input_ids=tokens, labels=[-100] * (len(tokens) - count) + tokens[-count:],
                                 attention_mask=[1] * len(tokens)))
        oracle_batches.append(collate_examples(examples, pad_token_id=0))
        local_batches.append(collate_examples([examples[runtime.rank // 2]], pad_token_id=0))
    reports = []
    # Capture after native normalization/coalescing and before SGD consumes the
    # gradient. LoRA step two exercises A as well as initially-zero B.
    gradient_report = {}
    optimizer.register_step_pre_hook(lambda *_: gradient_report.update(compare(model, reference, gradients=True)))
    for step in (1, 2):
        expected = train_update(reference, reference_optimizer, oracle_batches,
                                family="dsv4", runtime=Runtime(torch.device("cpu")))
        actual = train_update(model, optimizer, local_batches, family="dsv4", runtime=runtime,
                              cp_size=2, synchronize=lambda: synchronize_cp_gradients(model, cp))
        assert actual["supervised_tokens"] == expected["supervised_tokens"] == 11
        assert abs(actual["loss"] - expected["loss"]) < 1e-6, (actual, expected)
        updates = compare(model, reference)
        required = ("engram", "lora_a", "lora_b") if lora else ("engram", "experts", "dense_and_fp32")
        if step == 2:
            assert all(gradient_report["oracle_maximum_magnitudes"][name] > 1e-10 for name in required)
        reports.append(dict(step=step, loss=actual["loss"], loss_error=abs(actual["loss"] - expected["loss"]),
                            gradients=dict(gradient_report), parameters=updates))
    return dict(mode="lora_and_trainable_engram" if lora else "full", cp=2, ep=2, world=4,
                expert_fsdp_size=2, actual_fsdp=True, accumulation_steps=2,
                supervised_tokens_by_microbatch_and_data_group=counts, steps=reports)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="New JSON report path")
    args = parser.parse_args()
    torch.set_num_threads(1)
    dist.init_process_group("gloo", timeout=timedelta(seconds=180))
    try:
        assert dist.get_world_size() == 4, "launch with torchrun --nproc-per-node=4"
        meshes, cp, dispatcher = build_parallelism(2, 2, "cpu")
        runtime = Runtime(torch.device("cpu"), rank=dist.get_rank(), world_size=4)
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
