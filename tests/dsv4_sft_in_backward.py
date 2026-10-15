"""Two-rank full-parameter FSDP2 SFT update/resume and finite-abort parity.

CPU preserves the small FP32 fixture. CUDA uses native BF16/FP32 islands and
CPU sparse Engram; --reference-torch removes known Triton atomic nondeterminism.
"""

import argparse
from dataclasses import replace
from datetime import timedelta
import json
import os
from pathlib import Path
from unittest.mock import Mock

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from dsv41_train.engram_offload import install_engram_sgd_hooks
from dsv41_train.models.dsv4 import DeepSeekV41Config, DeepSeekV41ForCausalLM
from dsv41_train.models.dsv4.attention import AttentionSinks
from dsv41_train.models.dsv4.mhc import HyperConnection
from dsv41_train.models.dsv4.parallel import apply_fsdp2, build_parallelism, synchronize_cp_gradients
from dsv41_train.optim import install_sgd_in_backward
from dsv41_train.runtime import Runtime, local_tensor
from dsv41_train.sft_checkpoint import load_checkpoint, save_checkpoint
from dsv41_train.sft_engram import EngramCheckpoint
from dsv41_train.sft_data import collate_examples
from train_sft import train_update
from engram_offload_cuda_parity import assert_exact, configuration, reference_torch


def config_for(device_type):
    if device_type == "cuda":
        return configuration(16)
    return replace(DeepSeekV41Config.tiny(), hidden_size=16, num_attention_heads=2,
                     head_dim=8, q_lora_rank=8, qk_rope_head_dim=4, index_n_heads=2,
                     index_head_dim=8, o_groups=1, o_lora_rank=8, moe_intermediate_size=16,
                     n_routed_experts=4, hc_mult=2, hc_sinkhorn_iters=5, engram_head_dim=4)


def native(config, device, *, bf16=False, meshes=None, cp=None, dispatcher=None):
    torch.manual_seed(47)
    model = DeepSeekV41ForCausalLM(config, cp, dispatcher, None if meshes is None else meshes.engram,
                                   sparse_engram_gradients=True, engram_offload=True)
    if bf16:
        islands = {id(p) for module in model.modules() if isinstance(module, (HyperConnection, AttentionSinks))
                   for p in module.parameters(recurse=False)}
        for name, parameter in model.named_parameters():
            parameter.data = parameter.data.to(
                device="cpu" if ".engram_tables." in name else device,
                dtype=torch.float32 if id(parameter) in islands else torch.bfloat16)
        for module in model.modules():
            for name, buffer in module.named_buffers(recurse=False):
                setattr(module, name, buffer.to(device))
    assert all(p.requires_grad for p in model.parameters()), "full-parameter fixture must not freeze weights"
    assert not any("lora_A" in name or "lora_B" in name for name, _ in model.named_parameters())
    model.gradient_checkpointing_enable()
    if meshes is not None:
        apply_fsdp2(model, meshes)
        assert any(isinstance(p, DTensor) for p in model.parameters()), "FSDP was not applied"
    for table in model.model.engram_tables.values():
        assert table.weight.device.type == "cpu" and not isinstance(table.weight, DTensor)
    optimizer = torch.optim.SGD(model.parameters(), lr=.05, foreach=False)
    install_engram_sgd_hooks(model, optimizer)
    # This tiny random fixture has no mounted base checkpoint. Keep its small
    # owner shards only for the same row-reader contract used by released SFT.
    originals = {f"model.engram_tables.{name}.weight": table.weight.detach().clone()
                 for name, table in model.model.engram_tables.items()}
    starts = {f"model.engram_tables.{name}.weight": table.row_start
              for name, table in model.model.engram_tables.items()}
    model._sft_engram_checkpoint = EngramCheckpoint(model, optimizer,
        lambda name, ids: originals[name].index_select(0, ids - starts[name]))
    return model, optimizer


def snapshot(model, optimizer, device):
    return dict(parameters={name: local_tensor(p).detach().cpu().clone() for name, p in model.named_parameters()},
                buffers={name: local_tensor(b).detach().cpu().clone() for name, b in model.named_buffers()},
                optimizer=optimizer.state_dict(), cpu_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state(device) if device.type == "cuda" else None)


def parity(cp_size, output, runtime, exact=False):
    meshes, cp, dispatcher = build_parallelism(cp_size, 2, runtime.device.type)
    config = config_for(runtime.device.type)
    models, optimizers = [], []
    for _ in range(2):
        model, optimizer = native(config, runtime.device, bf16=runtime.device.type == "cuda",
                                  meshes=meshes, cp=cp, dispatcher=dispatcher)
        models.append(model)
        optimizers.append(optimizer)
    state = install_sgd_in_backward(models[1], optimizers[1])
    # Different data/answer counts at CP1; CP2 peers share one complete example.
    examples = [dict(input_ids=[1, 4, 7, 8, 9, 2], labels=[-100, -100, -100, -100, 9, 2], attention_mask=[1]*6),
                dict(input_ids=[1, 5, 3, 6, 9, 2], labels=[-100, -100, 3, 6, 9, 2], attention_mask=[1]*6)]
    if runtime.device.type == "cuda":
        for example in examples:
            example["input_ids"] = [1, 4, 7, 8, 9] * 2 + example["input_ids"]
            example["labels"] = [-100] * 10 + example["labels"]
            example["attention_mask"] = [1] * 16
    batch = collate_examples([examples[runtime.rank if cp_size == 1 else 1]], pad_token_id=0)
    options = dict(family="dsv4", runtime=runtime, cp_size=cp_size)
    initial = {name: local_tensor(value).detach().clone() for name, value in models[1].named_parameters()}
    checkpoint = output / f"cp{cp_size}-step1"
    identity = {"test": "sgd-in-backward", "cp": cp_size, "optimizer_in_backward": True,
                "device_type": runtime.device.type, "reference_torch": exact}
    max_error = 0.0
    for step in (1, 2):
        metrics = []
        for index, (model, optimizer) in enumerate(zip(models, optimizers)):
            metrics.append(train_update(model, optimizer, [batch], **options,
                           synchronize=lambda model=model: synchronize_cp_gradients(model, cp),
                           in_backward=state if index else None))
        tolerance = 1e-6 if runtime.device.type == "cpu" else 5e-4
        loss_matches = (metrics[0]["loss"] == metrics[1]["loss"] if exact else
                        abs(metrics[0]["loss"] - metrics[1]["loss"]) < tolerance)
        assert loss_matches, metrics
        for (name, expected), (_, actual) in zip(models[0].named_parameters(), models[1].named_parameters()):
            expected, actual = local_tensor(expected), local_tensor(actual)
            torch.testing.assert_close(actual, expected, rtol=0 if exact else 1e-5, atol=0 if exact else 1e-7, msg=name)
            if actual.numel():
                max_error = max(max_error, (actual.float() - expected.float()).abs().max().item())
        if step == 1:
            save_checkpoint(checkpoint, models[1], optimizers[1], step=1, identity=identity, data_position=2)
            payload = torch.load(checkpoint / f"rank-{runtime.rank}.pt", map_location="cpu", weights_only=True, mmap=True)
            assert not models[1]._sft_engram_checkpoint.parameter_names.intersection(payload["parameters"])
            assert payload["engram_rows"]["schema"] == models[1]._sft_engram_checkpoint.schema
            del payload
            saved = snapshot(models[1], optimizers[1], runtime.device)
    final = {name: local_tensor(value).detach().clone() for name, value in models[1].named_parameters()}
    assert load_checkpoint(checkpoint, models[1], optimizers[1], identity=identity) == (1, 2)
    assert_exact(snapshot(models[1], optimizers[1], runtime.device), saved)
    # Native load_state_dict replaces groups; hooks must follow their new LR.
    old_group = optimizers[1].param_groups[0]
    optimizers[1].load_state_dict(optimizers[1].state_dict())
    old_group["lr"] = .9
    train_update(models[1], optimizers[1], [batch], **options,
                 synchronize=lambda: synchronize_cp_gradients(models[1], cp), in_backward=state)
    for name, parameter in models[1].named_parameters():
        torch.testing.assert_close(local_tensor(parameter), final[name], rtol=0, atol=0, msg=name)
        if isinstance(parameter, DTensor):
            assert parameter.grad is None, name
        else:
            assert parameter.grad is not None, name
    changed = [name for name, value in final.items() if not torch.equal(value, initial[name])]
    updated = torch.tensor([any(part in name for name in changed)
                            for part in ("engram_tables", "attention_hc", "moe.routed")],
                           dtype=torch.int32, device=runtime.device)
    dist.all_reduce(updated, op=dist.ReduceOp.MAX)
    assert updated.all(), updated
    maximum_error = torch.tensor(max_error, device=runtime.device)
    dist.all_reduce(maximum_error, op=dist.ReduceOp.MAX)
    if cp_size == 2:
        assert any(not isinstance(parameter, DTensor) and "attention_hc" in name
                   for name, parameter in models[1].named_parameters())

    # Only rank 1 observes a bad local reduced gradient; both must abort before
    # the native final step. Earlier hook updates are intentionally not atomic.
    parameter = next(parameter for parameter in models[1].parameters() if isinstance(parameter, DTensor))
    state.reset()
    original_step = state._step
    def bad_step(value, optimizer, group_index, divisor):
        if value is parameter and runtime.rank == 1:
            value.grad.to_local().fill_(float("inf"))
        original_step(value, optimizer, group_index, divisor)
    state._step = bad_step
    optimizers[1].step = Mock(wraps=optimizers[1].step)
    try:
        train_update(models[1], optimizers[1], [batch], **options,
                     synchronize=lambda: synchronize_cp_gradients(models[1], cp), in_backward=state)
    except RuntimeError as error:
        assert "non-finite gradient or in-backward update" in str(error), error
    else:
        raise AssertionError("non-finite in-backward gradient was accepted")
    optimizers[1].step.assert_not_called()
    return dict(cp=cp_size, ep=2, device=runtime.device.type, actual_fsdp=True,
                actual_cpu_fsdp=runtime.device.type == "cpu", full_parameter=True,
                reference_torch=exact, steps=2, restored_exact=True, resume_exact=True,
                maximum_update_difference=maximum_error.item(), hooked_parameters=len(state.parameter_ids),
                global_nonfinite_abort=True, sparse_engram_and_fp32_updated=True, sparse_engram_checkpoint=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="New directory for temporary checkpoint evidence")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--reference-torch", action="store_true")
    parser.add_argument("--check-setup", action="store_true", help="Validate fixture precision on CPU without distributed startup")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.reference_torch:
        reference_torch()
    if args.check_setup:
        model, _ = native(config_for(args.device), torch.device("cpu"), bf16=args.device == "cuda")
        assert model._sft_engram_checkpoint.parameter_names
        counts = {}
        for parameter in model.parameters():
            assert bool(torch.isfinite(parameter).all())
            counts[str(parameter.dtype)] = counts.get(str(parameter.dtype), 0) + parameter.numel()
        if args.device == "cuda":
            assert counts.keys() == {"torch.bfloat16", "torch.float32"}
        print(json.dumps(dict(event="setup_passed", device_fixture=args.device, full_parameter=True,
                              parameter_count_by_dtype=counts, sparse_engram_checkpoint=True)), flush=True)
        return
    if not args.output:
        parser.error("--output is required unless --check-setup is used")
    device = torch.device("cpu")
    if args.device == "cuda":
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        torch.cuda.set_device(device)
    dist.init_process_group("nccl" if args.device == "cuda" else "gloo", timeout=timedelta(seconds=180))
    try:
        assert dist.get_world_size() == 2
        output = Path(args.output)
        if dist.get_rank() == 0:
            output.mkdir(parents=True, exist_ok=False)
        dist.barrier()
        runtime = Runtime(device, rank=dist.get_rank(), world_size=2)
        results = [parity(size, output, runtime, args.reference_torch) for size in (1, 2)]
        if dist.get_rank() == 0:
            text = json.dumps(results, indent=2) + "\n"
            (output / "results.json").write_text(text)
            print(text, flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
