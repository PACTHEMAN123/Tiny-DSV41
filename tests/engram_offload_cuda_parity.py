"""Two-GPU NCCL gate for synchronous CPU Engram; random tiny weights only.

Run once with --output DIR, then independently with the same DIR --resume-check.
--check-setup validates tiny configuration/model construction on CPU only.
"""

import argparse
import gc
import json
import os
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor import DTensor

from dsv41_train.engram_offload import install_engram_sgd_hooks
from dsv41_train.lora import LoRAConfig, inject_lora
from dsv41_train.models.dsv4 import DeepSeekV41Config, DeepSeekV41ForCausalLM
from dsv41_train.models.dsv4.attention import AttentionSinks
from dsv41_train.models.dsv4.mhc import HyperConnection
from dsv41_train.models.dsv4.model import RowShardedEmbedding
from dsv41_train.models.dsv4.parallel import apply_fsdp2, build_parallelism, synchronize_cp_gradients
from dsv41_train.runtime import Runtime, local_tensor
from dsv41_train.sft_checkpoint import load_checkpoint, save_checkpoint
from train_sft import _check_gradients, train_update


def configuration(length):
    return replace(DeepSeekV41Config.tiny(), num_hidden_layers=2,
                   compress_ratios=[0, 2], kv_source_layer_ids=[1], index_source_layer_ids=[1],
                   candidate_source_layer_id=1, moe_intermediate_size=128, n_routed_experts=4,
                   engram_num_embeddings=[4097], max_position_embeddings=max(128, length))


def native(config, offload, device, meshes=None, cp=None, dispatcher=None):
    torch.manual_seed(613)
    model = DeepSeekV41ForCausalLM(config, context_parallel=cp, token_dispatcher=dispatcher,
                                 engram_mesh=None if meshes is None else meshes.engram,
                                 sparse_engram_gradients=True, engram_offload=offload)
    inject_lora(model, LoRAConfig(rank=4, alpha=8))
    model.model.engram_tables.requires_grad_(True)
    # Match native loader precision: backbone/adapters/tables BF16, mHC/sinks
    # FP32, rotary and router-bias buffers retain their initialized precision.
    fp32 = {id(p) for module in model.modules() if isinstance(module, (HyperConnection, AttentionSinks))
            for p in module.parameters(recurse=False)}
    for name, parameter in model.named_parameters():
        target = "cpu" if offload and ".engram_tables." in name else device
        parameter.data = parameter.data.to(device=target, dtype=torch.float32 if id(parameter) in fp32 else torch.bfloat16)
    for module in model.modules():
        for name, buffer in module.named_buffers(recurse=False):
            setattr(module, name, buffer.to(device))
    model.gradient_checkpointing_enable()
    if meshes is not None:
        apply_fsdp2(model, meshes)
        table = model.model.engram_tables["1"]
        assert not isinstance(table.weight, DTensor), "FSDP must ignore sparse row parameters"
        assert table.weight.device.type == ("cpu" if offload else "cuda")
        assert any(isinstance(p, DTensor) for p in model.parameters()), "FSDP was not applied"
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.125, foreach=False)
    install_engram_sgd_hooks(model, optimizer)
    return model, optimizer


def values(model, gradients=False):
    result = {}
    for name, parameter in model.named_parameters():
        value = parameter.grad if gradients else parameter
        if value is not None:
            value = local_tensor(value).detach()
            # Only these deliberately tiny validation tables are densified.
            value = value.float().coalesce().to_dense() if value.is_sparse else value
            result[name] = value.cpu().clone()
    return result


def compare(actual, expected, *, exact=False):
    assert actual.keys() == expected.keys(), "state/gradient keys differ"
    error = 0.
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name], rtol=0 if exact else .02,
                                   atol=0 if exact else 2e-5, msg=lambda message: f"{name}: {message}")
        if actual[name].numel():
            error = max(error, (actual[name].float() - expected[name].float()).abs().max().item())
    return error


def row_probe(meshes, runtime):
    reports = []
    for frozen in (False, True):
        models, optimizers = [], []
        for offload in (True, False):
            model = nn.Module()
            model.table = RowShardedEmbedding(11, 32, meshes.engram, sparse_gradients=True,
                                             engram_offload=offload).bfloat16()
            if not offload:
                model.table.to(runtime.device)
            with torch.no_grad():
                initial = torch.arange(11 * 32).reshape(11, 32).float() / 1024
                model.table.weight.copy_(initial[model.table.row_start:model.table.row_stop])
            model.table.requires_grad_(not frozen)
            model.dense = nn.Parameter(torch.tensor(.5, device=runtime.device))
            optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.125, foreach=False)
            install_engram_sgd_hooks(model, optimizer)
            models.append(model)
            optimizers.append(optimizer)
        for empty in (False, True):
            ids = torch.tensor([0, 0, 0] if empty else [0, 6, 10 - runtime.rank], device=runtime.device)
            for model, optimizer in zip(models, optimizers):
                optimizer.zero_grad(set_to_none=True)
                for coefficients in ([256., 1., 1.], [1., -256., 1.]):
                    output = model.table(ids).float() * model.dense
                    (output * output.new_tensor(coefficients)[:, None]).sum().div(64).backward()
                _check_gradients(model, runtime)
            grad_error = compare(values(models[0], True), values(models[1], True))
            if empty and runtime.rank == 1 and not frozen:
                assert models[0].table.weight.grad._nnz() == 0
            for optimizer in optimizers:
                optimizer.step()
            weight_error = compare(values(models[0]), values(models[1]))
            reports.append(dict(frozen=frozen, empty_owner=empty, gradient_error=grad_error,
                                weight_error=weight_error))
    return reports


def batches(runtime, cp_size, length):
    generator = torch.Generator().manual_seed(184 + runtime.rank // cp_size)
    result = []
    for _ in range(2):
        ids = torch.randint(3, 32, (1, length), generator=generator)
        labels = ids.clone()
        labels[:, :length // 3] = -100
        result.append(dict(input_ids=ids, labels=labels, attention_mask=torch.ones_like(ids)))
    return result


def native_probe(args, runtime, cp_size):
    meshes, cp, dispatcher = build_parallelism(cp_size, 2)
    config = configuration(args.seq_len)
    actual, optimizer = native(config, True, runtime.device, meshes, cp, dispatcher)
    data = batches(runtime, cp_size, args.seq_len)
    table = actual.model.engram_tables["1"]
    hashes = actual.model.hash(data[0]["input_ids"].to(runtime.device), data[0]["attention_mask"].bool().to(runtime.device))
    assert hashes.max().item() < (4097 + 1) // 2, "fixture must leave owner rank 1 empty"
    directory = args.output / f"cp{cp_size}"
    identity = dict(probe="native-lora-trainable-engram", cp=cp_size, ep=2, seq_len=args.seq_len)
    options = dict(family="dsv4", runtime=runtime, cp_size=cp_size)
    update = lambda model, opt: train_update(model, opt, data, synchronize=lambda: synchronize_cp_gradients(model, cp), **options)
    if args.resume_check:
        step, cursor = load_checkpoint(directory / "step1", actual, optimizer, identity=identity)
        assert (step, cursor) == (1, 2)
        metrics = update(actual, optimizer)
        expected = torch.load(directory / f"expected-rank{runtime.rank}.pt", map_location="cpu", weights_only=True)
        compare(values(actual), expected["parameters"], exact=True)
        assert metrics["loss"] == expected["loss"], (metrics["loss"], expected["loss"])
        report = dict(cp=cp_size, independent_resume_exact=True, loss=metrics["loss"])
    else:
        resident, control_optimizer = native(config, False, runtime.device, meshes, cp, dispatcher)
        initial = table.weight.detach().clone()
        errors = []
        for step in (1, 2):
            if runtime.is_main:
                print(json.dumps(dict(event="native_update", cp=cp_size, step=step)), flush=True)
            metrics, control = update(actual, optimizer), update(resident, control_optimizer)
            assert abs(metrics["loss"] - control["loss"]) < 5e-4
            errors.append(dict(parameters=compare(values(actual), values(resident)),
                               gradients=compare(values(actual, True), values(resident, True))))
            assert table.weight.device.type == "cpu" and table.weight.grad.device.type == "cpu"
            if runtime.rank == 1:
                assert table.weight.grad._nnz() == 0
            if step == 1:
                save_checkpoint(directory / "step1", actual, optimizer, step=1, identity=identity, data_position=2)
        changed = torch.tensor(int(not torch.equal(table.weight, initial)), device=runtime.device)
        dist.all_reduce(changed)
        assert changed.item() > 0, "the CPU table did not update"
        torch.save(dict(parameters=values(actual), loss=metrics["loss"]), directory / f"expected-rank{runtime.rank}.pt")
        report = dict(cp=cp_size, mode="lora+trainable_table", empty_owner_rank=1,
                      gradient_checkpointing=True, accumulated_microbatches=2, steps=2, errors=errors)
        del resident, control_optimizer
    del actual, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--resume-check", action="store_true")
    parser.add_argument("--check-setup", action="store_true")
    args = parser.parse_args()
    if args.seq_len < 8 or args.seq_len % 2:
        parser.error("seq-len must be even and at least 8")
    torch.set_num_threads(1)
    if args.check_setup:
        model, _ = native(configuration(args.seq_len), True, torch.device("cpu"))
        assert model.model.engram_tables["1"].weight.device.type == "cpu"
        print("CPU setup passed; CUDA/NCCL/FSDP execution has not been tested.")
        return
    if args.output is None or not torch.cuda.is_available():
        parser.error("--output and two CUDA GPUs are required")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=180))
    try:
        assert dist.get_world_size() == 2, "launch with exactly two ranks"
        runtime = Runtime(torch.device("cuda", local_rank), dist.get_rank(), local_rank, 2)
        if runtime.is_main and not args.resume_check:
            args.output.mkdir(parents=True, exist_ok=False)
        dist.barrier()
        reports = {}
        if not args.resume_check:
            meshes, _, _ = build_parallelism(1, 2)
            reports["row_probes"] = row_probe(meshes, runtime)
        reports["native"] = [native_probe(args, runtime, cp) for cp in (1, 2)]
        gathered = [None] * 2
        dist.all_gather_object(gathered, reports)
        if runtime.is_main:
            path = args.output / ("resume-results.json" if args.resume_check else "results.json")
            path.write_text(json.dumps(gathered, indent=2) + "\n")
            print(json.dumps(dict(event="passed", report=str(path))), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
