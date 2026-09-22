"""Compare CP2/EP2 Qwen outputs and gradients with an unsharded reference."""

from __future__ import annotations

import argparse
import json
import os

import torch
import torch.distributed as dist

from dsv41_train.models.qwen import Qwen3MoeForCausalLM
from dsv41_train.models.qwen.parallel import ParallelParameters, build_parallelism
from dsv41_train.runtime import Runtime, initialize_runtime
from qwen_helpers import smoke_config


def copy_reference_weights(
    reference: Qwen3MoeForCausalLM,
    parallel: Qwen3MoeForCausalLM,
) -> None:
    source = reference.state_dict()
    selected = {}
    for name, target in parallel.state_dict().items():
        value = source[name]
        if ".experts." in name:
            layer_id = int(name.split(".")[2])
            experts = parallel.model.layers[layer_id].mlp.experts
            value = value[experts.expert_start : experts.expert_start + experts.num_experts]
        selected[name] = value.clone()
        if selected[name].shape != target.shape:
            raise RuntimeError(f"shape mismatch while copying {name}")
    parallel.load_state_dict(selected, strict=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", action="store_true", help="use Gloo instead of CUDA/NCCL")
    parser.add_argument("--padded", action="store_true", help="include padding and ignored labels")
    args = parser.parse_args()
    if args.cpu:
        dist.init_process_group("gloo")
        runtime = Runtime(
            device=torch.device("cpu"),
            rank=dist.get_rank(),
            local_rank=int(os.environ["LOCAL_RANK"]),
            world_size=dist.get_world_size(),
        )
    else:
        runtime = initialize_runtime(require_distributed=True)
    try:
        if runtime.world_size != 2:
            raise RuntimeError("this parity check requires exactly two ranks")
        torch.manual_seed(2026)
        torch.cuda.manual_seed(2026)
        config = smoke_config()
        reference = Qwen3MoeForCausalLM(config).to(runtime.device)
        _, context_parallel, token_dispatcher = build_parallelism(2, 2, runtime.device.type)
        parallel = Qwen3MoeForCausalLM(
            config,
            context_parallel,
            token_dispatcher,
        ).to(runtime.device)
        copy_reference_weights(reference, parallel)

        generator = torch.Generator(device=runtime.device).manual_seed(777)
        input_ids = torch.randint(
            0,
            config.vocab_size,
            (2, 12),
            generator=generator,
            device=runtime.device,
        )
        attention_mask = torch.ones_like(input_ids) if args.padded else None
        labels = input_ids.clone()
        if attention_mask is not None:
            attention_mask[1, -4:] = 0
            labels[1, -4:] = -100
        reference_output = reference(
            input_ids,
            attention_mask,
            labels=labels,
            output_router_logits=True,
        )
        parallel_output = parallel(
            input_ids,
            attention_mask,
            labels=labels,
            output_router_logits=True,
        )
        gathered_logits = context_parallel.gather(parallel_output.logits.detach(), dim=1)

        assert reference_output.loss is not None and parallel_output.loss is not None
        assert isinstance(reference_output.aux_loss, torch.Tensor)
        assert isinstance(parallel_output.aux_loss, torch.Tensor)
        torch.testing.assert_close(gathered_logits, reference_output.logits, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(
            parallel_output.loss,
            reference_output.loss,
            rtol=1e-5,
            atol=1e-6,
        )
        torch.testing.assert_close(
            parallel_output.aux_loss,
            reference_output.aux_loss,
            rtol=1e-5,
            atol=1e-6,
        )

        reference_output.loss.backward()
        parallel_output.loss.backward()
        parameters = ParallelParameters.collect(parallel)
        assert parameters.numel(2) == sum(parameter.numel() for parameter in reference.parameters())
        parameters.synchronize(context_parallel)

        reference_parameters = dict(reference.named_parameters())
        max_gradient_difference = 0.0
        for name, parameter in parallel.named_parameters():
            assert parameter.grad is not None
            expected = reference_parameters[name].grad
            assert expected is not None
            if ".experts." in name:
                layer_id = int(name.split(".")[2])
                experts = parallel.model.layers[layer_id].mlp.experts
                expected = expected[
                    experts.expert_start : experts.expert_start + experts.num_experts
                ]
            difference = (parameter.grad - expected).abs().max().item()
            max_gradient_difference = max(max_gradient_difference, difference)
            torch.testing.assert_close(parameter.grad, expected, rtol=2e-4, atol=2e-6)

        expected_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.5, foreach=False)
        actual_norm = parameters.clip_grad_norm(context_parallel, 0.5)
        torch.testing.assert_close(actual_norm, expected_norm, rtol=2e-5, atol=1e-6)
        if runtime.is_main:
            print(
                json.dumps(
                    {
                        "logits_max_abs": (
                            gathered_logits - reference_output.logits
                        ).abs().max().item(),
                        "loss_abs": (
                            parallel_output.loss - reference_output.loss
                        ).abs().item(),
                        "aux_loss_abs": (
                            parallel_output.aux_loss - reference_output.aux_loss
                        ).abs().item(),
                        "gradient_max_abs": max_gradient_difference,
                        "grad_norm_abs": (actual_norm - expected_norm).abs().item(),
                    }
                )
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
