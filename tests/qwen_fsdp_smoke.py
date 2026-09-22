"""Eight-rank CP2/EP2/FSDP4 smoke test for Qwen3 MoE training."""

from __future__ import annotations

import json

import torch
import torch.distributed as dist

from dsv41_train.models.qwen import Qwen3MoeForCausalLM
from dsv41_train.models.qwen.parallel import ParallelParameters, apply_fsdp2, build_parallelism
from dsv41_train.runtime import initialize_runtime, local_tensor
from qwen_helpers import smoke_config


def main() -> None:
    runtime = initialize_runtime(require_distributed=True)
    try:
        if runtime.world_size != 8:
            raise RuntimeError("this smoke test requires exactly eight ranks")
        torch.manual_seed(1234)
        torch.cuda.manual_seed(1234)
        meshes, context_parallel, token_dispatcher = build_parallelism(cp=2, ep=2)
        config = smoke_config()
        model = Qwen3MoeForCausalLM(
            config,
            context_parallel,
            token_dispatcher,
        ).to(runtime.device, dtype=torch.bfloat16)
        generator = torch.Generator(device=runtime.device).manual_seed(
            9000 + meshes.fsdp.get_local_rank()
        )
        input_ids = torch.randint(
            0,
            config.vocab_size,
            (2, 12),
            generator=generator,
            device=runtime.device,
        )

        starts: list[int | None] = [None] * runtime.world_size
        local_start = model.model.layers[0].mlp.experts.expert_start
        dist.all_gather_object(starts, local_start)
        if set(starts) != {0, config.num_experts // 2}:
            raise RuntimeError(f"unexpected expert ownership: {starts}")

        apply_fsdp2(model, meshes)
        parameters = ParallelParameters.collect(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, foreach=False)
        tracked = model.model.layers[0].self_attn.q_proj.weight
        before = local_tensor(tracked).detach().float().clone()

        optimizer.zero_grad(set_to_none=True)
        output = model(input_ids, labels=input_ids, output_router_logits=True)
        assert output.loss is not None and torch.isfinite(output.loss)
        output.loss.backward()
        parameters.synchronize(context_parallel)
        grad_norm = parameters.clip_grad_norm(context_parallel)
        optimizer.step()

        parameter_delta = (local_tensor(tracked).detach().float() - before).abs().max()
        dist.all_reduce(parameter_delta, op=dist.ReduceOp.MAX)
        if parameter_delta.item() == 0:
            raise RuntimeError("optimizer did not update the tracked parameter")
        if runtime.is_main:
            print(
                json.dumps(
                    {
                        "world_size": runtime.world_size,
                        "cp_size": context_parallel.size,
                        "ep_size": meshes.ep.size() if meshes.ep is not None else 1,
                        "fsdp_size": meshes.fsdp.size(),
                        "expert_fsdp_size": meshes.expert_fsdp.size(),
                        "expert_starts": starts,
                        "loss": output.loss.detach().float().item(),
                        "grad_norm": grad_norm.detach().float().item(),
                        "parameter_delta_max": parameter_delta.item(),
                    }
                )
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
