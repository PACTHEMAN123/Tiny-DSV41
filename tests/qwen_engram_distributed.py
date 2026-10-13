"""Compare eight-rank FSDP/EP plus Engram against an unsharded SGD reference."""

from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile

import torch
import torch.distributed as dist

from dsv41_train.models.qwen import Qwen3MoeForCausalLM, load_qwen3_moe
from dsv41_train.models.qwen.engram import QwenEngramConfig, attach_engram, shard_qwen_component, shard_qwen_root
from dsv41_train.models.dsv4.engram import EngramPolicy
from dsv41_train.models.dsv4.parallel import build_parallelism
from dsv41_train.models.dsv4.policies import GradientPolicy
from dsv41_train.optimization import OptimizerPolicy, build_optimizer
from qwen_helpers import smoke_config, write_hf_checkpoint


def main():
    torch.set_num_threads(1)
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.manual_seed(19)
    source = Qwen3MoeForCausalLM(smoke_config())
    temporary = tempfile.TemporaryDirectory() if rank == 0 else None
    folder = [temporary.name if temporary else None]
    dist.broadcast_object_list(folder)
    if rank == 0:
        write_hf_checkpoint(Path(folder[0]), source)
    dist.barrier()
    meshes, cp, dispatcher = build_parallelism(1, world)
    options = QwenEngramConfig((0, 1), 17, 4, 2, 3)
    try:
        for variant, policy in [
            ("gpu", EngramPolicy()), ("gpu_cache", EngramPolicy(cache_bytes=4096)),
            ("cpu", EngramPolicy("cpu", "async")),
            ("cpu_cache", EngramPolicy("cpu", "async", cache_bytes=4096)),
        ]:
            model = load_qwen3_moe(folder[0], device=device, dtype=torch.float32,
                context_parallel=cp, token_dispatcher=dispatcher,
                setup_callback=lambda m: attach_engram(m, options, policy, meshes.engram, device=device, dtype=torch.float32),
                module_callback=lambda name, module: shard_qwen_component(name, module, meshes))
            shard_qwen_root(model, meshes)
            model.gradient_checkpointing_enable()
            reference = deepcopy(source).to(device)
            attach_engram(reference, options, device=device, dtype=torch.float32)
            for name, table in model.model.engram_tables.items():
                shards = [None]*world
                dist.all_gather_object(shards, table.weight.detach().cpu())
                with torch.no_grad():
                    reference.model.engram_tables[name].weight.copy_(torch.cat(shards).to(device))
            optimizer = build_optimizer(model, OptimizerPolicy("sgd", .125))
            expected_optimizer = build_optimizer(reference, OptimizerPolicy("sgd", .125))
            gradients = GradientPolicy(model, meshes)
            generator = torch.Generator().manual_seed(63)
            for step in range(3):
                optimizer.zero_grad(set_to_none=True)
                expected_optimizer.zero_grad(set_to_none=True)
                gradients.begin_step()
                for _ in range(2):
                    ids = torch.randint(3, source.config.vocab_size, (2*world, 8), generator=generator).to(device)
                    labels = ids.clone()
                    labels[:, :3] = -100
                    expected = reference(ids, labels=labels)
                    actual = model(ids[rank*2:rank*2+2], labels=labels[rank*2:rank*2+2])
                    torch.testing.assert_close(actual.logits, expected.logits[rank*2:rank*2+2], rtol=5e-4, atol=3e-5)
                    (actual.loss*.37/2).backward()
                    (expected.loss*.37/2).backward()
                gradients.normalize()
                expected_params = dict(reference.named_parameters())
                for name, parameter in model.named_parameters():
                    gradient = parameter.grad
                    gradient = gradient.full_tensor() if hasattr(gradient, "full_tensor") else gradient
                    wanted = expected_params[name].grad
                    if "engram_tables" in name:
                        table = model.model.engram_tables[name.split("engram_tables.")[1].split(".")[0]]
                        wanted = wanted.to_dense()[table.row_start:table.row_stop]
                        gradient = gradient.to_dense()
                    elif ".mlp.experts." in name:
                        expert = dict(model.named_modules())[name.rsplit(".", 1)[0]]
                        wanted = wanted[expert.expert_start:expert.expert_start+expert.num_experts]
                    torch.testing.assert_close(gradient.cpu(), wanted.cpu(), rtol=8e-4, atol=3e-5)
                optimizer.step()
                expected_optimizer.step()
                for name, parameter in model.named_parameters():
                    actual = parameter.full_tensor() if hasattr(parameter, "full_tensor") else parameter
                    wanted = expected_params[name]
                    if "engram_tables" in name:
                        table = model.model.engram_tables[name.split("engram_tables.")[1].split(".")[0]]
                        wanted = wanted[table.row_start:table.row_stop]
                    elif ".mlp.experts." in name:
                        expert = dict(model.named_modules())[name.rsplit(".", 1)[0]]
                        wanted = wanted[expert.expert_start:expert.expert_start+expert.num_experts]
                    torch.testing.assert_close(actual.cpu(), wanted.cpu(), rtol=8e-4, atol=3e-5)
            for table in model.model.engram_tables.values():
                table.close()
            for table in reference.model.engram_tables.values():
                table.close()
            del model, reference, optimizer, expected_optimizer
            if rank == 0:
                print(json.dumps({"variant": variant, "world": world, "steps": 3,
                                  "accumulation": 2, "reference_parity": "passed"}), flush=True)
    finally:
        dist.destroy_process_group()
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    main()
