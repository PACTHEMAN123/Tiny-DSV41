"""Two-GPU NCCL gate for synchronous CPU Engram; random tiny weights only.

Run once with --output DIR, then independently with the same DIR --resume-check.
--check-setup validates tiny configuration/model construction on CPU only.
--repeat-baseline diagnoses U,C,U,C,U,C restores at CP1/EP2; --reference-torch
uses strict deterministic Torch references and requires exact repeatability.
"""

import argparse
import gc
import json
import os
from dataclasses import replace
from datetime import timedelta
from importlib import import_module
from pathlib import Path
from types import MethodType

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor import DTensor
from torch.nn import functional as F

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


def oracle_lookup(table, indices):
    # Quantize before owner-gradient scaling and NCCL, matching the BF16 table
    # boundary while retaining FP32 sparse accumulation in this test reference.
    return F.embedding(indices, table.weight, sparse=True).bfloat16()


def native(config, offload, device, meshes=None, cp=None, dispatcher=None, diagnostics=None):
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
    if not offload:
        for table in model.model.engram_tables.values():
            table.weight.data = table.weight.data.float()  # Already BF16-rounded.
            table._lookup = MethodType(oracle_lookup, table)
    model.gradient_checkpointing_enable()
    if meshes is not None:
        apply_fsdp2(model, meshes)
        table = model.model.engram_tables["1"]
        assert not isinstance(table.weight, DTensor), "FSDP must ignore sparse row parameters"
        assert table.weight.device.type == ("cpu" if offload else "cuda")
        assert any(isinstance(p, DTensor) for p in model.parameters()), "FSDP was not applied"
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.125, foreach=False)
    if diagnostics is not None:
        def capture_before_coalescing(_optimizer, _args, _kwargs):
            diagnostics.clear()
            diagnostics.update(gradients={}, nnz={})
            for name, parameter in model.named_parameters():
                if ".engram_tables." in name and parameter.grad is not None:
                    gradient = parameter.grad.detach()
                    diagnostics["nnz"][name] = gradient._nnz()
                    # Use the same CPU FP32 reduction for both raw accumulators.
                    diagnostics["gradients"][name] = gradient.cpu().float().coalesce().to_dense()
        optimizer.register_step_pre_hook(capture_before_coalescing)
    install_engram_sgd_hooks(model, optimizer)
    if not offload:
        def quantize_gradient(_optimizer, _args, _kwargs):
            for table in model.model.engram_tables.values():
                if table.weight.grad is not None:
                    table.weight.grad = table.weight.grad.bfloat16().float()
        def quantize_weight(_optimizer, _args, _kwargs):
            with torch.no_grad():
                for table in model.model.engram_tables.values():
                    table.weight.copy_(table.weight.bfloat16().float())
        # Installed after the native FP32 coalescing hook: one BF16 gradient and
        # one BF16 weight rounding per step, retaining no extra weight precision.
        optimizer.register_step_pre_hook(quantize_gradient)
        optimizer.register_step_post_hook(quantize_weight)
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
        left, right = actual[name], expected[name]
        if ".engram_tables." in name:
            left, right = left.float(), right.float()
        torch.testing.assert_close(left, right, rtol=0 if exact else .02,
                                   atol=0 if exact else 2e-5, msg=lambda message: f"{name}: {message}")
        if actual[name].numel():
            error = max(error, (actual[name].float() - expected[name].float()).abs().max().item())
    return error


def grouped_errors(actual, expected):
    assert actual.keys() == expected.keys(), "diagnostic keys differ"
    result = dict(engram=0., lora=0., other=0.)
    for name in actual:
        group = "engram" if ".engram_tables." in name else "lora" if ".lora_" in name else "other"
        if actual[name].numel():
            result[group] = max(result[group], (actual[name].float() - expected[name].float()).abs().max().item())
    return result


def resume_state(model, optimizer, device):
    return dict(parameters=values(model),
                buffers={name: local_tensor(value).detach().cpu().clone() for name, value in model.named_buffers()},
                optimizer=optimizer.state_dict(), cpu_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state(device))


def assert_exact(actual, expected, name="restored"):
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=name)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys(), name
        for key in actual:
            assert_exact(actual[key], expected[key], f"{name}.{key}")
    elif isinstance(actual, (list, tuple)):
        assert type(actual) is type(expected) and len(actual) == len(expected), name
        for index, (left, right) in enumerate(zip(actual, expected)):
            assert_exact(left, right, f"{name}[{index}]")
    else:
        assert actual == expected, (name, actual, expected)


def row_probe(meshes, runtime, cache_bytes=0):
    reports = []
    for frozen in (False, True):
        models, optimizers = [], []
        for offload in (True, False):
            model = nn.Module()
            model.table = RowShardedEmbedding(11, 32, meshes.engram, sparse_gradients=True,
                                             engram_offload=offload).bfloat16()
            if not offload:
                model.table.to(runtime.device)
            elif cache_bytes:
                model.table.configure_cache(cache_bytes, runtime.device)
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
    actual_raw, oracle_raw = {}, {}
    actual, optimizer = native(config, True, runtime.device, meshes, cp, dispatcher, actual_raw)
    data = batches(runtime, cp_size, args.seq_len)
    table = actual.model.engram_tables["1"]
    if args.cache_bytes:
        table.configure_cache(args.cache_bytes, runtime.device)
    hashes = actual.model.hash(data[0]["input_ids"].to(runtime.device), data[0]["attention_mask"].bool().to(runtime.device))
    assert hashes.max().item() < (4097 + 1) // 2, "fixture must leave owner rank 1 empty"
    directory = args.output / f"cp{cp_size}"
    identity = dict(probe="native-lora-trainable-engram", cp=cp_size, ep=2, seq_len=args.seq_len)
    options = dict(family="dsv4", runtime=runtime, cp_size=cp_size)
    update = lambda model, opt: train_update(model, opt, data, synchronize=lambda: synchronize_cp_gradients(model, cp), **options)
    if args.resume_check:
        step, cursor = load_checkpoint(directory / "step1", actual, optimizer, identity=identity)
        assert (step, cursor) == (1, 2)
        restored = torch.load(directory / f"resume-state-rank{runtime.rank}.pt", map_location="cpu", weights_only=True)
        assert_exact(resume_state(actual, optimizer, runtime.device), restored)
        print(json.dumps(dict(event="restored_exact", cp=cp_size, rank=runtime.rank,
                              parameters=True, buffers=True, optimizer=True, rng=True)), flush=True)
        metrics = update(actual, optimizer)
        expected = torch.load(directory / f"expected-rank{runtime.rank}.pt", map_location="cpu", weights_only=True)
        compare(values(actual), expected["parameters"], exact=True)
        assert metrics["loss"] == expected["loss"], (metrics["loss"], expected["loss"])
        report = dict(cp=cp_size, independent_resume_exact=True, loss=metrics["loss"])
    else:
        resident, control_optimizer = native(config, False, runtime.device, meshes, cp, dispatcher, oracle_raw)
        compare(values(actual), values(resident), exact=True)
        initial = table.weight.detach().clone()
        errors = []
        for step in (1, 2):
            if runtime.is_main:
                print(json.dumps(dict(event="native_update", cp=cp_size, step=step)), flush=True)
            metrics, control = update(actual, optimizer), update(resident, control_optimizer)
            actual_values, oracle_values = values(actual), values(resident)
            actual_gradients, oracle_gradients = values(actual, True), values(resident, True)
            diagnostic = dict(weights=grouped_errors(actual_values, oracle_values),
                              gradients=grouped_errors(actual_gradients, oracle_gradients),
                              raw_table_gradient=grouped_errors(actual_raw["gradients"], oracle_raw["gradients"])["engram"],
                              offload_nnz=actual_raw["nnz"], oracle_nnz=oracle_raw["nnz"])
            print(json.dumps(dict(event="native_comparison", cp=cp_size, step=step, rank=runtime.rank,
                                  offload_loss=metrics["loss"], oracle_loss=control["loss"], **diagnostic)), flush=True)
            assert abs(metrics["loss"] - control["loss"]) < 5e-4, dict(
                cp=cp_size, step=step, offload_loss=metrics["loss"], oracle_loss=control["loss"],
                diagnostic=diagnostic, previous_errors=errors)
            errors.append(dict(parameter_max=compare(actual_values, oracle_values),
                               gradient_max=compare(actual_gradients, oracle_gradients), **diagnostic))
            assert table.weight.device.type == "cpu" and table.weight.grad.device.type == "cpu"
            if runtime.rank == 1:
                assert table.weight.grad._nnz() == 0
            if step == 1:
                save_checkpoint(directory / "step1", actual, optimizer, step=1, identity=identity, data_position=2)
                torch.save(resume_state(actual, optimizer, runtime.device), directory / f"resume-state-rank{runtime.rank}.pt")
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


def reference_torch():
    # Test-only switches; keep NCCL, FSDP, dispatch and activation checkpointing.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    prefix = "dsv41_train.models.dsv4."
    import_module(prefix + "attention.module").triton_attention_available = lambda tensor: False
    for name in ("mhc", "moe.kernels", "attention.sparse_indexer"):
        import_module(prefix + name).triton = None


def repeat_baseline(args, runtime):
    meshes, cp, dispatcher = build_parallelism(1, 2)
    raw = {}
    model, optimizer = native(configuration(args.seq_len), True, runtime.device, meshes, cp, dispatcher, raw)
    data, tables = batches(runtime, 1, args.seq_len), list(model.model.engram_tables.values())
    initial = resume_state(model, optimizer, runtime.device)
    identity = dict(probe="repeat-baseline", cp=1, ep=2, seq_len=args.seq_len)
    save_checkpoint(args.output / "s0", model, optimizer, step=0, identity=identity, data_position=0)
    trials, reports = [], []

    def difference(left, right):
        return dict(loss=abs(left["loss"] - right["loss"]),
                    raw_table_gradient=grouped_errors(left["raw_table_gradients"], right["raw_table_gradients"])["engram"],
                    **{key: grouped_errors(left[key], right[key]) for key in ("weights", "gradients")})

    for trial, mode in enumerate("UCUCUC"):
        assert load_checkpoint(args.output / "s0", model, optimizer, identity=identity) == (0, 0)
        for table in tables:
            table.configure_cache(args.cache_bytes if mode == "C" else 0, runtime.device)
        assert_exact(resume_state(model, optimizer, runtime.device), initial)
        snapshots, steps = [], []
        for step in (1, 2):
            metrics = train_update(model, optimizer, data, family="dsv4", runtime=runtime, cp_size=1,
                                   synchronize=lambda: synchronize_cp_gradients(model, cp))
            snapshot = dict(loss=metrics["loss"], weights=values(model), gradients=values(model, True),
                            raw_table_gradients=raw["gradients"])
            snapshots.append(snapshot)
            counts = [table._cpu_lookup.cache_metrics for table in tables]
            report = dict(event="repeat_baseline_step", trial=trial, mode=mode, step=step, rank=runtime.rank,
                          loss=metrics["loss"], relative_to_first_u=difference(snapshot, trials[0][step - 1] if trials else snapshot),
                          cache={key: sum(count[key] for count in counts) for key in counts[0]})
            print(json.dumps(report), flush=True)
            steps.append(report)
        trials.append(snapshots)
        reports.append(dict(trial=trial, mode=mode, restored_exact=True, steps=steps))
    summaries = {}
    for category in ("UU", "CC", "UC"):
        pairs = [(a, b) for a in range(6) for b in range(a + 1, 6)
                 if sorted("UCUCUC"[a] + "UCUCUC"[b]) == sorted(category)]
        errors = [difference(left, right) for a, b in pairs for left, right in zip(trials[a], trials[b])]
        summaries["/".join(category)] = dict(pairs=len(pairs), loss=max(error["loss"] for error in errors),
            raw_table_gradient=max(error["raw_table_gradient"] for error in errors),
            **{key: {group: max(error[key][group] for error in errors) for group in ("engram", "lora", "other")}
               for key in ("weights", "gradients")})
    print(json.dumps(dict(event="repeat_baseline_summary", rank=runtime.rank, summaries=summaries)), flush=True)
    if args.reference_torch:
        error = None
        try:
            for snapshots in trials[1:]:
                assert_exact(snapshots, trials[0], "deterministic repeat")
        except AssertionError as exc:
            error = str(exc)
        valid = torch.tensor(int(error is None), device=runtime.device)
        dist.all_reduce(valid, op=dist.ReduceOp.MIN)
        assert valid.item(), error or "deterministic repeat differs on another rank"
    return dict(cp=1, ep=2, backend="torch" if args.reference_torch else "native",
                exact_gate=args.reference_torch, trials=reports, summaries=summaries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--cache-bytes", type=int, default=0)
    parser.add_argument("--resume-check", action="store_true")
    parser.add_argument("--check-setup", action="store_true")
    parser.add_argument("--repeat-baseline", action="store_true")
    parser.add_argument("--reference-torch", action="store_true")
    args = parser.parse_args()
    if args.seq_len < 8 or args.seq_len % 2:
        parser.error("seq-len must be even and at least 8")
    if args.cache_bytes < 0:
        parser.error("cache-bytes must be nonnegative")
    if args.repeat_baseline and (args.resume_check or args.cache_bytes <= 0):
        parser.error("repeat-baseline requires positive cache-bytes and cannot use resume-check")
    if args.reference_torch and not args.repeat_baseline:
        parser.error("reference-torch requires repeat-baseline")
    torch.set_num_threads(1)
    if args.reference_torch:
        reference_torch()
    if args.check_setup:
        model, _ = native(configuration(args.seq_len), True, torch.device("cpu"))
        reference, _ = native(configuration(args.seq_len), False, torch.device("cpu"))
        compare(values(model), values(reference), exact=True)
        assert model.model.engram_tables["1"].weight.device.type == "cpu"
        table = reference.model.engram_tables["1"]
        assert table.weight.dtype == torch.float32 and table(torch.tensor([0])).dtype == torch.bfloat16
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
        if args.repeat_baseline:
            reports["baseline"] = repeat_baseline(args, runtime)
        elif not args.resume_check:
            meshes, _, _ = build_parallelism(1, 2)
            reports["row_probes"] = row_probe(meshes, runtime, args.cache_bytes)
        if not args.repeat_baseline:
            reports["native"] = [native_probe(args, runtime, cp) for cp in (1, 2)]
        gathered = [None] * 2
        dist.all_gather_object(gathered, reports)
        if runtime.is_main:
            path = args.output / ("baseline-results.json" if args.repeat_baseline else
                                  "resume-results.json" if args.resume_check else "results.json")
            path.write_text(json.dumps(gathered, indent=2) + "\n")
            event = ("baseline_exact_passed" if args.reference_torch else "baseline_diagnostic") if args.repeat_baseline else "passed"
            print(json.dumps(dict(event=event, report=str(path))), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
