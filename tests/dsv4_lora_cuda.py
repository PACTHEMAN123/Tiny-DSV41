"""CUDA LoRA learning, freezing, export, and merge checks on a small DSV4 model."""

import copy
import json
import tempfile
from pathlib import Path

import torch

from dsv41_train.lora import (
    LoRAConfig, adapter_parameters, inject_lora, load_adapter,
    merge_adapters, save_adapter, unmerge_adapters,
)
from dsv41_train.models.dsv4 import DeepSeekV41ForCausalLM
from lora_helpers import small_config


def verify(mixed_precision, directory):
    torch.manual_seed(42)
    model = DeepSeekV41ForCausalLM(small_config(attention_dropout=0.1)).cuda()
    pristine = copy.deepcopy(model)
    inputs = torch.arange(24, device="cuda").reshape(2, 12) % 10 + 3
    inputs[:, 0] = 1
    mask = torch.ones_like(inputs)
    mask[1, -2:] = 0

    def output(candidate):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=mixed_precision):
            return candidate(inputs, mask, labels=inputs)

    model.eval()
    with torch.no_grad():
        initial_logits = output(model).logits.clone()
    config = LoRAConfig(dropout=0.2)
    names = inject_lora(model, config)
    with torch.no_grad():
        torch.testing.assert_close(output(model).logits, initial_logits, rtol=0, atol=0)
        initial_loss = output(model).loss.item()
    initial_adapters = {name: parameter.detach().clone() for name, parameter in model.named_parameters()
                        if "lora_" in name}
    optimizer = torch.optim.AdamW(adapter_parameters(model), lr=0.01, weight_decay=0)
    model.train()
    losses = []
    for step in range(20):
        optimizer.zero_grad(set_to_none=True)
        loss = output(model).loss
        if not torch.isfinite(loss).item():
            raise AssertionError("non-finite loss")
        loss.backward()
        for name, parameter in model.named_parameters():
            if "lora_" in name:
                if parameter.grad is None or not torch.isfinite(parameter.grad).all().item():
                    raise AssertionError(f"missing or non-finite adapter gradient: {name}")
                if step == 1 and not torch.count_nonzero(parameter.grad).item():
                    raise AssertionError(f"adapter gradient is zero after the first update: {name}")
            elif parameter.requires_grad or parameter.grad is not None:
                raise AssertionError(f"base parameter is not frozen: {name}")
        optimizer.step()
        losses.append(loss.item())
    base_parameters = dict(pristine.named_parameters())
    for name, parameter in model.named_parameters():
        if name in base_parameters:
            torch.testing.assert_close(parameter, base_parameters[name], rtol=0, atol=0, msg=name)
        elif torch.equal(parameter, initial_adapters[name]):
            raise AssertionError(f"adapter did not update: {name}")
    model.eval()
    with torch.no_grad():
        expected = output(model)
        final_loss = expected.loss.item()
    if final_loss >= initial_loss - 0.01:
        raise AssertionError(f"fixed-batch loss did not decrease: {initial_loss} -> {final_loss}")

    path = directory / f"{'bf16' if mixed_precision else 'fp32'}.pt"
    save_adapter(model, path)
    pristine.eval()
    inject_lora(pristine, config)
    load_adapter(pristine, path)
    with torch.no_grad():
        restored = output(pristine).logits
        torch.testing.assert_close(restored, expected.logits, rtol=0, atol=0)
        # Check merged inference in FP32 to isolate algebra from autocast rounding.
        before_merge = pristine(inputs, mask).logits
        merge_adapters(pristine)
        merged = pristine(inputs, mask).logits
        torch.testing.assert_close(merged, before_merge, rtol=3e-4, atol=3e-6)
        unmerge_adapters(pristine)
        torch.testing.assert_close(pristine(inputs, mask).logits, before_merge, rtol=0, atol=0)
    return {
        "precision": "bf16_autocast_fp32_weights" if mixed_precision else "fp32",
        "adapter_modules": len(names), "trainable_parameters": sum(p.numel() for p in adapter_parameters(model)),
        "steps": len(losses), "initial_eval_loss": initial_loss, "final_eval_loss": final_loss,
        "training_losses": losses, "base_max_abs_change": 0.0, "reload_max_abs": 0.0,
        "merge_max_abs": (merged - before_merge).abs().max().item(),
        "adapter_bytes": path.stat().st_size,
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("this verification requires a CUDA GPU")
    torch.set_num_threads(2)
    results = []
    with tempfile.TemporaryDirectory() as directory:
        for mixed_precision in (False, True):
            result = verify(mixed_precision, Path(directory))
            results.append(result)
            print(json.dumps(result), flush=True)
    path = Path("outputs/lora-cuda-verification.json")
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({"torch": torch.__version__, "device": torch.cuda.get_device_name(),
                                "results": results}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
