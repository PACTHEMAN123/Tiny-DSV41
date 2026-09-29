"""Two-GPU output, gradient, update, and adapter-export parity with FSDP2."""

import copy
import json
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist

from dsv41_train.activation import ActivationCheckpointConfig
from dsv41_train.lora import LoRAConfig, inject_lora, load_adapter, save_adapter
from dsv41_train.models.dsv4 import DeepSeekV41ForCausalLM
from dsv41_train.models.dsv4.parallel import apply_fsdp2
from dsv41_train.parallel import ParallelMeshes
from dsv41_train.runtime import initialize_runtime
from post_training_helpers import small_config


def main():
    runtime = initialize_runtime(require_distributed=True)
    torch.set_num_threads(2)
    results = []
    temporary = tempfile.TemporaryDirectory() if runtime.is_main else None
    paths = [temporary.name if temporary else None]
    dist.broadcast_object_list(paths)
    try:
        if runtime.world_size != 2:
            raise ValueError("run this test with exactly two GPUs")
        meshes = ParallelMeshes.build()
        for mode in ("none", "full", "selective"):
            for use_lora in (False, True):
                torch.manual_seed(42)
                reference = DeepSeekV41ForCausalLM(small_config()).to(runtime.device)
                if use_lora:
                    inject_lora(reference, LoRAConfig())
                model = copy.deepcopy(reference)
                model.set_activation_checkpointing(
                    ActivationCheckpointConfig(mode, ("attention", "moe", "hc")),
                )
                apply_fsdp2(model, meshes)
                generator = torch.Generator().manual_seed(123 + runtime.rank)
                tokens = torch.randint(3, 32, (2, 12), generator=generator).to(runtime.device)
                mask = torch.ones_like(tokens)
                mask[1, -3:] = 0
                reference_optimizer = torch.optim.SGD(
                    [p for p in reference.parameters() if p.requires_grad], lr=0.01,
                )
                optimizer = torch.optim.SGD(
                    [p for p in model.parameters() if p.requires_grad], lr=0.01,
                )
                reference_parameters = dict(reference.named_parameters())
                maximum_error = 0.0
                for step in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    reference_optimizer.zero_grad(set_to_none=True)
                    for microbatch in (tokens, tokens.flip(1)):
                        expected = reference(microbatch, mask, labels=microbatch)
                        actual = model(microbatch, mask, labels=microbatch)
                        torch.testing.assert_close(actual.logits, expected.logits, rtol=2e-5, atol=2e-6)
                        torch.testing.assert_close(actual.loss, expected.loss, rtol=2e-5, atol=2e-6)
                        (expected.loss * 0.5).backward()
                        (actual.loss * 0.5).backward()
                    for name, parameter in model.named_parameters():
                        target = reference_parameters[name]
                        present = torch.tensor(int(target.grad is not None), device=runtime.device)
                        dist.all_reduce(present)
                        if not present.item():
                            if parameter.grad is not None:
                                # FSDP may materialize zeros for unused members of a group.
                                gradient = parameter.grad.full_tensor()
                                torch.testing.assert_close(gradient, torch.zeros_like(gradient), rtol=0, atol=0)
                            continue
                        if target.grad is None:
                            target.grad = torch.zeros_like(target)
                        dist.all_reduce(target.grad)
                        target.grad.div_(runtime.world_size)
                        if parameter.grad is None:
                            raise AssertionError(f"missing gradient: {name}")
                        gradient = parameter.grad.full_tensor()
                        if use_lora and step and "lora_" in name and not torch.count_nonzero(gradient).item():
                            raise AssertionError(f"adapter has no gradient after its first update: {name}")
                        maximum_error = max(maximum_error, (gradient - target.grad).abs().max().item())
                        torch.testing.assert_close(gradient, target.grad, rtol=3e-4, atol=3e-6, msg=name)
                    optimizer.step()
                    reference_optimizer.step()
                    for name, parameter in model.named_parameters():
                        torch.testing.assert_close(parameter.full_tensor(), reference_parameters[name],
                                                   rtol=3e-4, atol=3e-6, msg=name)
                if use_lora:
                    path = Path(paths[0]) / f"{mode}.pt"
                    save_adapter(model, path)
                    dist.barrier()
                    restored = copy.deepcopy(reference)
                    with torch.no_grad():
                        for name, parameter in restored.named_parameters():
                            if "lora_" in name:
                                parameter.zero_()
                    load_adapter(restored, path)
                    torch.testing.assert_close(restored(tokens).logits, reference(tokens).logits,
                                               rtol=3e-4, atol=3e-6)
                result = {"mode": mode, "lora": use_lora, "gradient_max_abs": maximum_error,
                          "loss": actual.loss.item(), "optimizer_steps": 2, "microbatches_per_step": 2}
                results.append(result)
                if runtime.is_main:
                    print(json.dumps(result), flush=True)
                del model, reference, optimizer, reference_optimizer, actual, expected
        if runtime.is_main:
            path = Path("artifacts/fsdp-parity.json")
            path.parent.mkdir(exist_ok=True)
            path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    finally:
        dist.destroy_process_group()
        if temporary:
            temporary.cleanup()


if __name__ == "__main__":
    main()
