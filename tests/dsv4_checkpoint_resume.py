"""Manual Engram/EP shards plus optional CUDA FSDP: full and LoRA DCP resume."""

import argparse
import json
import os
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh

from dsv41_train.checkpoint import CheckpointManager, TrainingState
from dsv41_train.checkpoint_layout import checkpoint_parameter_names
from dsv41_train.dispatch import AllToAllTokenDispatcher
from dsv41_train.lora import LoRAConfig, inject_lora
from dsv41_train.models.dsv4 import DeepSeekV41ForCausalLM
from dsv41_train.models.dsv4.model import RowShardedEmbedding
from dsv41_train.models.dsv4.moe import RoutedExperts
from dsv41_train.models.dsv4.parallel import apply_fsdp2, build_parallelism
from dsv41_train.runtime import local_tensor
from lora_helpers import small_config


class ShardedFixture(nn.Module):
    def __init__(self, mesh, sparse):
        super().__init__()
        self.table = RowShardedEmbedding(11, 3, mesh, sparse_gradients=sparse)
        self.experts = RoutedExperts(small_config(), AllToAllTokenDispatcher(mesh))
        self.head = nn.Linear(3, 2)
        with torch.no_grad():
            self.table.weight.fill_(mesh.get_local_rank() + 1)
            for parameter in self.experts.parameters():
                parameter.fill_(0.01 * (mesh.get_local_rank() + 1))

    def forward(self, indices):
        result = self.head(self.table(indices)).square().mean()
        return result + sum(p.square().mean() for p in self.experts.parameters())


def verify(device, folder, mesh, meshes, context, dispatcher, *, lora, sparse):
    torch.manual_seed(42)
    if device.type == "cuda":
        model = DeepSeekV41ForCausalLM(
            small_config(attention_dropout=0.1), context, dispatcher,
            meshes.engram, sparse_engram_gradients=sparse,
        ).to(device)
        config = LoRAConfig(dropout=0.2)
    else:
        model = ShardedFixture(mesh, sparse)
        config = LoRAConfig(rank=2, alpha=4, dropout=0.2, targets=("head",))
    if lora:
        inject_lora(model, config)
    if device.type == "cuda":
        apply_fsdp2(model, meshes)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = (torch.optim.SGD(trainable, lr=0.001) if sparse else
                 torch.optim.AdamW(trainable, lr=0.001, foreach=False))
    state = TrainingState(
        model, optimizer, model_config={"fixture": device.type},
        data_generator=torch.Generator(device=device).manual_seed(10 + dist.get_rank()),
        checkpoint_mode="trainable" if lora else "full",
        base_model_identity={"seed": 42}, training_config={"sparse": sparse},
    )
    names = checkpoint_parameter_names(model)

    def update():
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            tokens = torch.randint(3, 32, (2, 12), device=device, generator=state.data_generator)
            loss = model(tokens, labels=tokens).loss
        else:
            indices = torch.randint(0, 11, (2, 6), generator=state.data_generator)
            loss = model(indices)
        loss.backward()
        if device.type == "cpu":
            for name, parameter in model.named_parameters():
                if name not in names and parameter.grad is not None:
                    dist.all_reduce(parameter.grad)
                    parameter.grad.div_(dist.get_world_size())
        optimizer.step()
        return loss.detach().clone()

    update()
    label = f"{'lora' if lora else 'full'}-{'sparse-sgd' if sparse else 'dense-adamw'}"
    manager = CheckpointManager(folder / label, state)
    path = manager.save(1)
    saved_parameters = {name: local_tensor(p).detach().clone() for name, p in model.named_parameters()}
    expected_loss = update()
    expected_parameters = {name: local_tensor(p).detach().clone() for name, p in model.named_parameters()}
    with torch.no_grad():
        for parameter in model.parameters():
            if not lora or parameter.requires_grad:
                local_tensor(parameter).fill_(-99)
    if manager.load() != 1:
        raise AssertionError("wrong restored step")
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(local_tensor(parameter), saved_parameters[name], rtol=0, atol=0, msg=name)
    torch.testing.assert_close(update(), expected_loss, rtol=0, atol=0)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(local_tensor(parameter), expected_parameters[name], rtol=0, atol=0, msg=name)

    # Repeat with an empty optimizer, as in a newly launched training process.
    optimizer.state.clear()
    optimizer.zero_grad(set_to_none=True)
    manager.load()
    torch.testing.assert_close(update(), expected_loss, rtol=0, atol=0)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(local_tensor(parameter), expected_parameters[name], rtol=0, atol=0, msg=name)

    # A mismatch on one rank must make all ranks fail before any tensor IO.
    if dist.get_rank() == 1:
        state.training_config = {"sparse": not sparse}
    try:
        manager.load()
    except ValueError as exc:
        if "training config" not in str(exc):
            raise
    else:
        raise AssertionError("rank-local metadata mismatch was accepted")
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(local_tensor(parameter), expected_parameters[name], rtol=0, atol=0, msg=name)
    result = {"case": label, "exact_resume": True, "collective_rejection": True,
              "manual_shard_parameters": len(names),
              "checkpoint_bytes": sum(p.stat().st_size for p in path.iterdir())}
    if dist.get_rank() == 0:
        print(json.dumps(result), flush=True)
    dist.barrier()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        torch.cuda.set_device(device)
    torch.set_num_threads(2)
    dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    temporary = None
    try:
        if dist.get_world_size() not in (2, 4):
            raise ValueError("run with two or four processes")
        mesh = meshes = context = dispatcher = None
        if device.type == "cuda":
            meshes, context, dispatcher = build_parallelism(ep=2)
        else:
            mesh = init_device_mesh("cpu", (dist.get_world_size(),), mesh_dim_names=("engram",))
        if dist.get_rank() == 0:
            temporary = tempfile.TemporaryDirectory()
        paths = [temporary.name if temporary else None]
        dist.broadcast_object_list(paths)
        results = []
        for lora in (False, True):
            for sparse in (True, False):
                results.append(verify(device, Path(paths[0]), mesh, meshes, context, dispatcher,
                                      lora=lora, sparse=sparse))
        if dist.get_rank() == 0:
            output = Path(f"outputs/checkpoint-resume-{args.device}-{dist.get_world_size()}.json")
            output.parent.mkdir(exist_ok=True)
            output.write_text(json.dumps({"torch": str(torch.__version__), "world_size": dist.get_world_size(),
                                          "results": results}, indent=2) + "\n")
        dist.barrier()
    finally:
        dist.destroy_process_group()
        if temporary:
            temporary.cleanup()


if __name__ == "__main__":
    main()
