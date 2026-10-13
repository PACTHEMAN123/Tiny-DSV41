"""Fresh-process DCP resume with changed CPU/GPU parameter placement."""

import argparse
import datetime
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from dsv41_train.checkpoint import CheckpointManager, TrainingState
from dsv41_train.layout import local_tensor
from dsv41_train.loading import LoadPolicy
from dsv41_train.models.dsv4 import load_dsv41_backbone_window
from dsv41_train.models.dsv4.engram import EngramPolicy
from dsv41_train.models.dsv4.materialize import FSDPMaterializer
from dsv41_train.models.dsv4.parallel import build_parallelism
from dsv41_train.models.dsv4.policies import FSDPPolicy, GradientPolicy
from dsv41_train.optimization import OptimizerPolicy, build_optimizer
from dsv41_train.sharded_state import StateAdapter
from test_dsv4_loading import synthetic_checkpoint
from test_memory_modules import memory_config
from memory_validation_cases import fixture_dimensions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("save", "resume"), required=True)
    parser.add_argument("--folder", required=True)
    parser.add_argument("--cp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=2)
    parser.add_argument("--optimizer", choices=("sgd", "adamw"), default="sgd")
    parser.add_argument("--cache-bytes", type=int, default=0)
    args = parser.parse_args()
    if args.cache_bytes and args.optimizer != "sgd":
        parser.error("CPU Engram cache requires sparse SGD")
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    dist.init_process_group("nccl", device_id=device, timeout=datetime.timedelta(seconds=60))
    rank = dist.get_rank()
    experts, length = fixture_dimensions(dist.get_world_size())
    folder = Path(args.folder)
    model = None
    try:
        if args.phase == "save" and rank == 0:
            folder.mkdir(parents=True, exist_ok=True)
            torch.manual_seed(42)
            synthetic_checkpoint(folder, memory_config(n_routed_experts=experts,
                engram_layer_ids=[1, 3], engram_num_embeddings=[127, 131]))
        dist.barrier()
        meshes, cp, dispatcher = build_parallelism(args.cp, args.ep)
        saving = args.phase == "save"
        fsdp = FSDPPolicy((1, 3) if saving else ())
        model = load_dsv41_backbone_window(folder, num_layers=5, device=device, dtype=torch.float32,
            context_parallel=cp, token_dispatcher=dispatcher, engram_mesh=meshes.engram,
            sparse_engram_gradients=args.optimizer == "sgd", materializer=FSDPMaterializer(meshes, fsdp),
            load_policy=LoadPolicy("direct_shard"),
            engram_policy=EngramPolicy("cpu", "async", True, cache_bytes=args.cache_bytes)
                          if saving and args.optimizer == "sgd" else EngramPolicy())
        optimizer = build_optimizer(model, OptimizerPolicy(args.optimizer, .01 if args.optimizer == "sgd" else .001))
        gradients = GradientPolicy(model, meshes)
        generator = torch.Generator(device=device).manual_seed(123 + rank // args.cp)
        manager = CheckpointManager(folder / "state", TrainingState(model, optimizer,
            model_config=model.config.to_dict(), data_generator=generator,
            state_adapter=StateAdapter(model, cp=args.cp, ep=args.ep)))

        def step():
            optimizer.zero_grad(set_to_none=True)
            gradients.begin_step()
            tokens = torch.randint(3, 32, (1, length), generator=generator, device=device)
            for _ in range(2):
                output = model(tokens, labels=tokens)
                loss = output.loss + F.dropout(output.logits, p=.2, training=True).square().mean() * .01
                (loss * .5).backward()
            gradients.normalize()
            model.synchronize_engram()
            optimizer.step()

        torch.manual_seed(321 + rank)
        if saving:
            step()
            manager.save(1)
        else:
            assert manager.load() == 1
        cpu_random, cuda_random = torch.rand(8), torch.rand(8, device=device)
        step()
        result = {"parameters": {name: local_tensor(value).detach().cpu()
                                 for name, value in model.named_parameters()},
                  "cpu_random": cpu_random, "cuda_random": cuda_random.cpu(),
                  "generator": generator.get_state()}
        path = folder / f"expected-{rank}.pt"
        maximum_error = 0.0
        if saving:
            torch.save(result, path)
        else:
            expected = torch.load(path, weights_only=True)
            for name, value in result["parameters"].items():
                torch.testing.assert_close(value, expected["parameters"][name], rtol=3e-5, atol=3e-6, msg=name)
                if value.numel():
                    maximum_error = max(maximum_error, (value - expected["parameters"][name]).abs().max().item())
            for name in ("cpu_random", "cuda_random", "generator"):
                torch.testing.assert_close(result[name], expected[name], rtol=0, atol=0)
        errors = [None] * dist.get_world_size()
        dist.all_gather_object(errors, maximum_error)
        if rank == 0:
            report = {"phase": args.phase, "optimizer": args.optimizer, "cp": args.cp, "ep": args.ep,
                      "world": dist.get_world_size(), "format": StateAdapter.VERSION,
                      "cache_bytes": args.cache_bytes if saving else 0,
                      "max_parameter_error": max(errors), "rng_exact": True, "passed": True}
            (folder / f"{args.phase}-result.json").write_text(json.dumps(report, indent=2) + "\n")
            print(f"fresh-process {args.phase} passed (CP{args.cp}/EP{args.ep}, {args.optimizer})", flush=True)
    finally:
        if model is not None:
            model.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
