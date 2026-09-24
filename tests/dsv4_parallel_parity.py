"""Compare CP2/EP2 DSV4 outputs and gradients with an unsharded reference."""

from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist

from dsv41_train.models.dsv4 import DeepSeekV41Config, DeepSeekV41ForCausalLM
from dsv41_train.models.dsv4.parallel import ParallelParameters, build_parallelism
from dsv41_train.runtime import Runtime


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


def reference_parameter(
    name: str,
    reference: DeepSeekV41ForCausalLM,
    parallel: DeepSeekV41ForCausalLM,
) -> tuple[torch.nn.Parameter, slice | None]:
    source_name = name
    selection = None
    for kind in ("gate_up", "down"):
        marker = f".moe.routed.{kind}."
        if marker in name:
            prefix, local_id = name.rsplit(".", 1)
            layer_id = int(name.split(".")[2])
            routed = parallel.model.layers[layer_id].moe.routed
            source_name = f"{prefix}.{routed.expert_start + int(local_id)}"
            break
    if ".engram_tables." in name and name.endswith(".weight"):
        table_id = name.split(".")[2]
        table = parallel.model.engram_tables[table_id]
        selection = slice(table.row_start, table.row_stop)
    return dict(reference.named_parameters())[source_name], selection


def copy_reference_weights(
    reference: DeepSeekV41ForCausalLM,
    parallel: DeepSeekV41ForCausalLM,
) -> None:
    with torch.no_grad():
        for name, target in parallel.named_parameters():
            source, selection = reference_parameter(name, reference, parallel)
            value = source if selection is None else source[selection]
            if value.shape != target.shape:
                raise RuntimeError(f"shape mismatch while copying {name}")
            target.copy_(value)


def main() -> None:
    dist.init_process_group("gloo")
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if world_size != 2:
            raise RuntimeError("this parity check requires exactly two ranks")
        runtime = Runtime(
            device=torch.device("cpu"),
            rank=rank,
            local_rank=int(os.environ["LOCAL_RANK"]),
            world_size=world_size,
        )
        torch.manual_seed(2026)
        config = smoke_config()
        reference = DeepSeekV41ForCausalLM(config)
        meshes, context_parallel, token_dispatcher = build_parallelism(2, 2, "cpu")
        if meshes.fsdp.size() != world_size:
            raise AssertionError("dense FSDP must span the CP ranks")
        parallel = DeepSeekV41ForCausalLM(
            config,
            context_parallel,
            token_dispatcher,
            meshes.engram,
        )
        copy_reference_weights(reference, parallel)

        generator = torch.Generator().manual_seed(777)
        input_ids = torch.randint(0, config.vocab_size, (2, 16), generator=generator)
        attention_mask = torch.ones_like(input_ids)
        attention_mask[1, -4:] = 0
        labels = input_ids.masked_fill(~attention_mask.bool(), -100)

        reference_output = reference(input_ids, attention_mask, labels=labels)
        parallel_output = parallel(input_ids, attention_mask, labels=labels)
        gathered_logits = context_parallel.gather(parallel_output.logits.detach(), dim=1)

        assert reference_output.loss is not None and parallel_output.loss is not None
        assert reference_output.aux_loss is not None and parallel_output.aux_loss is not None
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
        parameters.synchronize(context_parallel)

        max_gradient_difference = 0.0
        for name, parameter in parallel.named_parameters():
            source, selection = reference_parameter(name, reference, parallel)
            expected = source.grad
            if expected is not None and selection is not None:
                expected = expected[selection]
            if parameter.grad is None or expected is None:
                if parameter.grad is not None or expected is not None:
                    raise AssertionError(f"gradient presence mismatch for {name}")
                continue
            difference = (parameter.grad - expected).abs().max().item()
            max_gradient_difference = max(max_gradient_difference, difference)
            torch.testing.assert_close(parameter.grad, expected, rtol=2e-4, atol=2e-6)

        if runtime.is_main:
            print(
                json.dumps(
                    {
                        "logits_max_abs": (gathered_logits - reference_output.logits)
                        .abs()
                        .max()
                        .item(),
                        "loss_abs": (parallel_output.loss - reference_output.loss).abs().item(),
                        "aux_loss_abs": (
                            parallel_output.aux_loss - reference_output.aux_loss
                        )
                        .abs()
                        .item(),
                        "gradient_max_abs": max_gradient_difference,
                    }
                ),
                flush=True,
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
