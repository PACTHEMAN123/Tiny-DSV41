"""Two-GPU LoRA gradient/update parity, adapter export, and DCP resume."""

import copy
import json
import tempfile
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist

from dsv41_train.checkpoint import CheckpointManager, TrainingState
from dsv41_train.lora import (
    LoRAConfig, adapter_parameters, inject_lora, load_adapter, save_adapter,
)
from dsv41_train.models.dsv4 import DeepSeekV41ForCausalLM
from dsv41_train.models.dsv4.parallel import apply_fsdp2
from dsv41_train.parallel import ParallelMeshes
from dsv41_train.runtime import initialize_runtime
from lora_helpers import small_config


def main():
    runtime = initialize_runtime(require_distributed=True)
    torch.set_num_threads(2)
    temporary = tempfile.TemporaryDirectory() if runtime.is_main else None
    paths = [temporary.name if temporary else None]
    dist.broadcast_object_list(paths)
    try:
        if runtime.world_size != 2:
            raise ValueError("run this test with exactly two GPUs")
        torch.manual_seed(42)
        config = LoRAConfig()
        reference = DeepSeekV41ForCausalLM(small_config()).to(runtime.device)
        inject_lora(reference, config)
        model = copy.deepcopy(reference)
        apply_fsdp2(model, ParallelMeshes.build())
        generator = torch.Generator().manual_seed(123 + runtime.rank)
        tokens = torch.randint(3, 32, (2, 12), generator=generator).to(runtime.device)
        mask = torch.ones_like(tokens)
        mask[1, -3:] = 0
        optimizer = torch.optim.AdamW(adapter_parameters(model), lr=0.001, foreach=False)
        reference_optimizer = torch.optim.AdamW(adapter_parameters(reference), lr=0.001, foreach=False)
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
                if not parameter.requires_grad:
                    if parameter.grad is not None:
                        raise AssertionError(f"base parameter has a gradient: {name}")
                    continue
                if parameter.grad is None or target.grad is None:
                    raise AssertionError(f"missing adapter gradient: {name}")
                dist.all_reduce(target.grad)
                target.grad.div_(runtime.world_size)
                gradient = parameter.grad.full_tensor()
                if step and not torch.count_nonzero(gradient).item():
                    raise AssertionError(f"adapter has no gradient after its first update: {name}")
                maximum_error = max(maximum_error, (gradient - target.grad).abs().max().item())
                torch.testing.assert_close(gradient, target.grad, rtol=3e-4, atol=3e-6, msg=name)
            optimizer.step()
            reference_optimizer.step()
            for name, parameter in model.named_parameters():
                torch.testing.assert_close(parameter.full_tensor(), reference_parameters[name],
                                           rtol=3e-4, atol=3e-6, msg=name)

        path = Path(paths[0]) / "adapter.pt"
        save_adapter(model, path)
        dist.barrier()
        restored = copy.deepcopy(reference)
        with torch.no_grad():
            for parameter in adapter_parameters(restored):
                parameter.zero_()
        load_adapter(restored, path)
        torch.testing.assert_close(restored(tokens).logits, reference(tokens).logits,
                                   rtol=3e-4, atol=3e-6)

        state = TrainingState(model, optimizer, model_config=model.config.to_dict(),
                              data_generator=generator, training_config={"lora": asdict(config)},
                              checkpoint_mode="trainable", base_model_identity={"seed": 42})
        manager = CheckpointManager(Path(paths[0]) / "checkpoint", state)
        manager.save(2)

        def update():
            optimizer.zero_grad(set_to_none=True)
            loss = model(tokens, labels=tokens).loss
            loss.backward()
            optimizer.step()
            return loss.detach().clone()

        expected_loss = update()
        expected_parameters = [p.full_tensor().detach().clone() for p in adapter_parameters(model)]
        if manager.load() != 2:
            raise AssertionError("incorrect resumed step")
        torch.testing.assert_close(update(), expected_loss, rtol=0, atol=0)
        for parameter, expected in zip(adapter_parameters(model), expected_parameters):
            torch.testing.assert_close(parameter.full_tensor(), expected, rtol=0, atol=0)
        if runtime.is_main:
            result = {"world_size": runtime.world_size, "gradient_max_abs": maximum_error,
                      "optimizer_steps": 2, "microbatches_per_step": 2,
                      "adapter_export": "passed", "exact_resume": "passed"}
            print(json.dumps(result), flush=True)
            output = Path("outputs/lora-fsdp-verification.json")
            output.parent.mkdir(exist_ok=True)
            output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        dist.barrier()
    finally:
        dist.destroy_process_group()
        if temporary:
            temporary.cleanup()


if __name__ == "__main__":
    main()
