"""CPU EP checkpoint regression, including fresh-process AdamW resume.

Run with torchrun twice, --phase save then --phase resume, sharing --output.
Use two ranks normally, or four with --expert-fsdp-size 2 to cover DTensor
sharding within the expert subgroup (storage only, not GPU FSDP execution).
No CUDA, large weights, or external data are needed.
"""
import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.checkpoint import FileSystemReader

from dsv41_train.checkpoint import CheckpointManager, TrainingState
from dsv41_train.models.dsv4.parallel import build_parallelism
from dsv41_train.models.qwen import Qwen3MoeForCausalLM
from dsv41_train.layout import local_tensor
from dsv41_train.sharded_state import StateAdapter
from qwen_helpers import tiny_config


def update(model, optimizer, rank):
    optimizer.zero_grad(set_to_none=True)
    for name, parameter in model.named_parameters():
        # Every EP rank must retain distinct experts and distinct Adam moments.
        value = rank + 1. if ".mlp.experts." in name else .25
        parameter.grad = torch.full_like(parameter, value)
    optimizer.step()


def snapshot(model, optimizer):
    return {
        "model": {name: local_tensor(value).detach().clone() for name, value in model.state_dict().items()},
        "optimizer": {name: {key: local_tensor(value).clone() if torch.is_tensor(value) else value
                             for key, value in optimizer.state[parameter].items()}
                      for name, parameter in model.named_parameters()},
    }


def compare(model, optimizer, expected):
    for name, value in model.state_dict().items():
        torch.testing.assert_close(local_tensor(value), expected["model"][name], rtol=0, atol=0)
    for name, parameter in model.named_parameters():
        for key, value in optimizer.state[parameter].items():
            torch.testing.assert_close(local_tensor(value), expected["optimizer"][name][key], rtol=0, atol=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("save", "resume"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expert-fsdp-size", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    torch.set_num_threads(1)
    dist.init_process_group("gloo")
    try:
        rank = dist.get_rank()
        if dist.get_world_size() != 2 * args.expert_fsdp_size:
            raise ValueError("run this test with 2 * expert-fsdp-size CPU ranks")
        meshes, cp, dispatcher = build_parallelism(1, 2, "cpu")
        torch.manual_seed(71 if args.phase == "save" else 999)
        config = tiny_config()
        model = Qwen3MoeForCausalLM(config, cp, dispatcher)
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if ".mlp.experts." in name:
                    parameter.add_(0.1 * (rank % 2))
        if args.expert_fsdp_size > 1:
            from torch.distributed.tensor import distribute_tensor, Shard
            for layer in model.model.layers:
                experts = layer.mlp.experts
                for name, parameter in list(experts.named_parameters()):
                    setattr(experts, name, torch.nn.Parameter(distribute_tensor(
                        parameter.detach(), meshes.expert_fsdp, [Shard(0)])))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        folder = Path(args.output)
        folder.mkdir(parents=True, exist_ok=True)
        manager = CheckpointManager(folder / "checkpoint", TrainingState(model, optimizer,
            model_config=config.to_dict(), data_generator=torch.Generator().manual_seed(rank + 12),
            training_config={"objective": "checkpoint_regression"},
            state_adapter=StateAdapter(model, ep=2)))
        oracle_path = folder / f"oracle-rank-{rank}.pt"
        if args.phase == "save":
            update(model, optimizer, rank % 2)
            checkpoint_path = manager.save(1)
            before = snapshot(model, optimizer)
            update(model, optimizer, rank % 2)
            torch.save({"before": before, "after": snapshot(model, optimizer)}, oracle_path)
            metadata = FileSystemReader(str(checkpoint_path)).read_metadata()
            key = "train.model.model.layers.0.mlp.experts.gate_up_proj"
            tensor = metadata.state_dict_metadata[key]
            assert tensor.size[0] == config.num_experts
            expected_offsets = [0, 2] if args.expert_fsdp_size == 1 else [0, 1, 2, 3]
            assert sorted(chunk.offsets[0] for chunk in tensor.chunks) == expected_offsets
        else:
            expected = torch.load(oracle_path, weights_only=True)
            assert manager.load() == 1
            compare(model, optimizer, expected["before"])
            update(model, optimizer, rank % 2)
            compare(model, optimizer, expected["after"])
        dist.barrier()
        if rank == 0:
            print(json.dumps({"phase": args.phase, "result": "passed", "world_size": dist.get_world_size(),
                              "ep_size": 2, "expert_fsdp_size": args.expert_fsdp_size,
                              "optimizer": "adamw"}), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
