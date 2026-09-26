"""Two-GPU parity for regular SGD and FSDP2 SGD-in-backward."""

from __future__ import annotations

import json

import torch
import torch.distributed as dist

from dsv41_train.models.dsv4 import DeepSeekV41Config, DeepSeekV41ForCausalLM
from dsv41_train.models.dsv4.parallel import (
    ParallelParameters,
    apply_fsdp2,
    build_parallelism,
)
from dsv41_train.runtime import initialize_runtime, local_tensor
from train_dsv4_checkpoint import register_sgd_in_backward


def smoke_config() -> DeepSeekV41Config:
    return DeepSeekV41Config(
        vocab_size=32,
        hidden_size=32,
        num_hidden_layers=3,
        num_attention_heads=2,
        head_dim=16,
        q_lora_rank=16,
        qk_rope_head_dim=8,
        max_position_embeddings=32,
        sliding_window=16,
        compress_ratios=[0, 2, 2],
        kv_source_layer_ids=[1],
        index_source_layer_ids=[1, 2],
        candidate_source_layer_id=1,
        candidate_topk_blocks=2,
        candidate_block_size=2,
        index_n_heads=2,
        index_head_dim=16,
        index_topk=4,
        o_groups=2,
        o_lora_rank=16,
        moe_intermediate_size=32,
        n_routed_experts=4,
        num_experts_per_tok=2,
        hc_mult=2,
        hc_sinkhorn_iters=2,
        engram_layer_ids=[1],
        engram_num_embeddings=[128],
        engram_vocab_size=31,
        engram_max_ngram_size=3,
        engram_n_heads=1,
        engram_head_dim=8,
        engram_compressed_vocab_size=32,
    )


def local_parameters(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: local_tensor(parameter).detach().float().clone()
        for name, parameter in model.named_parameters()
    }


def main() -> None:
    runtime = initialize_runtime(require_distributed=True)
    hook_handles: list[torch.utils.hooks.RemovableHandle] = []
    try:
        if runtime.world_size != 2:
            raise RuntimeError("this parity test requires exactly two ranks")

        meshes, context_parallel, token_dispatcher = build_parallelism(2, 2, "cuda")
        config = smoke_config()
        model_seed = 20260926

        torch.manual_seed(model_seed)
        regular = DeepSeekV41ForCausalLM(
            config,
            context_parallel,
            token_dispatcher,
            meshes.engram,
            sparse_engram_gradients=True,
        ).to(runtime.device)
        torch.manual_seed(model_seed)
        in_backward = DeepSeekV41ForCausalLM(
            config,
            context_parallel,
            token_dispatcher,
            meshes.engram,
            sparse_engram_gradients=True,
        ).to(runtime.device)
        regular.gradient_checkpointing_enable()
        in_backward.gradient_checkpointing_enable()
        regular.train()
        in_backward.train()

        apply_fsdp2(regular, meshes)
        apply_fsdp2(in_backward, meshes)
        regular_parameters = ParallelParameters.collect(regular)
        in_backward_parameters = ParallelParameters.collect(in_backward)
        initial = local_parameters(in_backward)

        learning_rate = 3.0e-3
        regular_optimizer = torch.optim.SGD(
            regular.parameters(), lr=learning_rate, foreach=False
        )
        in_backward_ids, hook_handles, statistics = register_sgd_in_backward(
            in_backward_parameters,
            learning_rate=learning_rate,
            cp_size=context_parallel.size,
        )
        remaining_parameters = [
            parameter
            for parameter in in_backward.parameters()
            if id(parameter) not in in_backward_ids
        ]
        remaining_optimizer = torch.optim.SGD(
            remaining_parameters, lr=learning_rate, foreach=False
        )

        generator = torch.Generator(device=runtime.device).manual_seed(777)
        input_ids = torch.randint(
            0,
            config.vocab_size,
            (2, 16),
            device=runtime.device,
            generator=generator,
        )
        attention_mask = torch.ones_like(input_ids)
        attention_mask[1, -4:] = 0
        labels = input_ids.masked_fill(~attention_mask.bool(), -100)

        regular.zero_grad(set_to_none=True)
        regular_output = regular(input_ids, attention_mask, labels=labels)
        assert regular_output.loss is not None
        regular_output.loss.backward()
        regular_parameters.synchronize(context_parallel, dense_fully_sharded=True)
        regular_optimizer.step()

        in_backward.zero_grad(set_to_none=True)
        in_backward_output = in_backward(input_ids, attention_mask, labels=labels)
        assert in_backward_output.loss is not None
        in_backward_output.loss.backward()
        in_backward_parameters.synchronize(
            context_parallel,
            dense_fully_sharded=True,
        )
        remaining_optimizer.step()

        torch.testing.assert_close(
            in_backward_output.loss,
            regular_output.loss,
            rtol=1.0e-6,
            atol=1.0e-7,
        )
        if not statistics["updates"]:
            raise AssertionError("optimizer-in-backward hooks did not run")

        maximum_difference = torch.zeros((), device=runtime.device)
        maximum_update = torch.zeros((), device=runtime.device)
        regular_named = dict(regular.named_parameters())
        for name, candidate in in_backward.named_parameters():
            expected = local_tensor(regular_named[name]).detach().float()
            actual = local_tensor(candidate).detach().float()
            before = initial[name]
            if actual.numel():
                difference = (actual - expected).abs().max()
                update = (actual - before).abs().max()
                maximum_difference = torch.maximum(maximum_difference, difference)
                maximum_update = torch.maximum(maximum_update, update)
            torch.testing.assert_close(
                actual,
                expected,
                rtol=2.0e-5,
                atol=2.0e-6,
                msg=lambda message, parameter_name=name: (
                    f"{parameter_name}: {message}"
                ),
            )

        dist.all_reduce(maximum_difference, op=dist.ReduceOp.MAX)
        dist.all_reduce(maximum_update, op=dist.ReduceOp.MAX)
        if maximum_update.item() == 0:
            raise AssertionError("neither optimizer path updated a parameter")
        if runtime.is_main:
            print(
                json.dumps(
                    {
                        "loss_abs": (
                            in_backward_output.loss - regular_output.loss
                        ).abs().item(),
                        "parameter_max_abs": maximum_difference.item(),
                        "parameter_update_max": maximum_update.item(),
                        "registered_parameters": len(in_backward_ids),
                        "in_backward_updates": statistics["updates"],
                    }
                ),
                flush=True,
            )
    finally:
        for handle in hook_handles:
            handle.remove()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
