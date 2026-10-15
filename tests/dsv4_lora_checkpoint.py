"""Exercise native LoRA on released DSV4 weights, including a fresh-process reload.

Run with torchrun and PYTHONPATH=.; --steps 0 --load-adapter verifies an export.
This is a numerical/update probe, not an SFT quality benchmark.
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.distributed as dist

from dsv41_train.lora import (
    LoRAConfig, adapter_parameters, inject_lora, load_adapter_state,
    read_adapter, save_adapter, validate_adapter_layout,
)
from dsv41_train.models.dsv4 import load_dsv41_backbone_window
from dsv41_train.models.dsv4.parallel import (
    apply_fsdp2_layer, apply_fsdp2_root, build_parallelism, synchronize_cp_gradients,
)
from dsv41_train.runtime import distributed_mean, initialize_runtime, local_tensor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--ep-size", type=int, default=1)
    parser.add_argument("--cp-size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--load-adapter")
    parser.add_argument("--expected-loss", type=float)
    args = parser.parse_args()
    if args.steps < 0 or args.seq_len < 2 or args.num_layers < 1:
        parser.error("steps must be nonnegative; seq-len >= 2 and num-layers >= 1")
    runtime = initialize_runtime(require_distributed=True)
    torch.manual_seed(42)
    torch.cuda.reset_peak_memory_stats(runtime.device)
    started = time.perf_counter()
    meshes, cp, dispatcher = build_parallelism(args.cp_size, args.ep_size)
    config = LoRAConfig(rank=args.rank, alpha=2.0 * args.rank)
    payload = read_adapter(args.load_adapter) if args.load_adapter else None
    loaded = 0

    def layer_loaded(layer):
        nonlocal loaded
        inject_lora(layer, config)
        if payload is not None:
            load_adapter_state(layer, payload, prefix=f"model.layers.{loaded}")
        apply_fsdp2_layer(layer, meshes)
        loaded += 1
        if runtime.is_main:
            print(json.dumps({"loaded_layers": loaded}), flush=True)

    model = load_dsv41_backbone_window(
        args.model_path, num_layers=args.num_layers, device=runtime.device,
        dtype=torch.bfloat16, context_parallel=cp, token_dispatcher=dispatcher,
        engram_mesh=meshes.engram, sparse_engram_gradients=True,
        layer_loaded=layer_loaded,
    )
    for module in (model.model.embedding, model.model.norm, model.lm_head,
                   model.model.engram_tables):
        module.requires_grad_(False)
    if payload is not None:
        validate_adapter_layout(model, payload)
        expected = {name for name, _ in model.named_parameters()
                    if name.endswith((".lora_a", ".lora_b"))}
        if set(payload["weights"]) != expected:
            raise ValueError("adapter keys differ from the selected model")
    model.gradient_checkpointing_enable()
    apply_fsdp2_root(model, meshes)
    adapters = list(adapter_parameters(model))
    assert {id(p) for p in model.parameters() if p.requires_grad} == {id(p) for p in adapters}
    before = [local_tensor(p).detach().clone() for p in adapters]
    frozen = [(p, local_tensor(p).detach().reshape(-1)[:32].clone())
              for p in model.parameters() if not p.requires_grad]
    generator = torch.Generator(device=runtime.device).manual_seed(20261016)
    ids = torch.randint(3, model.config.vocab_size, (1, args.seq_len),
                        generator=generator, device=runtime.device)
    ids[:, 0] = model.config.bos_token_id
    optimizer = torch.optim.AdamW(adapters, lr=1e-3, weight_decay=0, foreach=False)
    losses = []
    model.train()
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        loss = model(ids, labels=ids, compute_aux_loss=False).loss
        assert torch.isfinite(loss), "non-finite loss"
        loss.backward()
        synchronize_cp_gradients(model, cp)
        for parameter in adapters:
            assert parameter.grad is not None, "missing adapter gradient"
            assert torch.isfinite(local_tensor(parameter.grad)).all(), "non-finite gradient"
        optimizer.step()
        losses.append(distributed_mean(loss, runtime))
        if runtime.is_main:
            print(json.dumps({"step": step + 1, "loss": losses[-1]}), flush=True)
    assert all(p.grad is None for p, _ in frozen)
    assert all(torch.equal(local_tensor(p).reshape(-1)[:32], value) for p, value in frozen)
    delta = torch.zeros((), device=runtime.device)
    for parameter, initial in zip(adapters, before):
        local = local_tensor(parameter).detach()
        if local.numel():
            delta = torch.maximum(delta, (local.float() - initial.float()).abs().max())
    dist.all_reduce(delta, op=dist.ReduceOp.MAX)
    if args.steps:
        assert delta.item() > 0, "no adapter update"
    model.eval()
    with torch.no_grad():
        eval_loss = distributed_mean(model(ids, labels=ids, compute_aux_loss=False).loss, runtime)
    if args.expected_loss is not None:
        assert abs(eval_loss - args.expected_loss) < 1e-5, (eval_loss, args.expected_loss)
    output = Path(args.output)
    if runtime.is_main:
        output.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    save_adapter(model, output / "adapter.pt")
    torch.cuda.synchronize(runtime.device)
    peak = torch.tensor(torch.cuda.max_memory_allocated(runtime.device), device=runtime.device)
    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    result = dict(layers=args.num_layers, seq_len=args.seq_len, steps=args.steps,
                  ep=args.ep_size, cp=args.cp_size, losses=losses, eval_loss=eval_loss,
                  adapter_delta=delta.item(), frozen_samples_unchanged=True,
                  peak_allocated_gib=peak.item() / 2**30,
                  elapsed_seconds=time.perf_counter() - started,
                  checkpoint=args.model_path, adapter_reload=args.load_adapter)
    if runtime.is_main:
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
