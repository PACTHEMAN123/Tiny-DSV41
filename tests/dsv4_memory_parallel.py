"""Distributed loading, offload, gradient, and fresh-model resume acceptance."""

import argparse
from dataclasses import asdict
import datetime
import gc
from functools import wraps
import json
import os
from pathlib import Path
import tempfile
import time
import traceback

import torch
import torch.distributed as dist

from dsv41_train.activation import ActivationCheckpointConfig
from dsv41_train.checkpoint import CheckpointManager, TrainingState
from dsv41_train.layout import parameter_layouts, local_tensor, parameter_memory
from dsv41_train.loading import LoadPolicy
from dsv41_train.models.dsv4 import load_dsv41_backbone_window, DeepSeekV41ForCausalLM
from dsv41_train.models.dsv4.engram import EngramPolicy
from dsv41_train.models.dsv4.materialize import FSDPMaterializer
from dsv41_train.models.dsv4.parallel import build_parallelism
from dsv41_train.models.dsv4.policies import FSDPPolicy, GradientPolicy
from dsv41_train.optimization import OptimizerPolicy, build_optimizer
from dsv41_train.sharded_state import StateAdapter
from test_dsv4_loading import synthetic_checkpoint
from test_memory_modules import memory_config, LocalMaterializer
from memory_validation_cases import fixture_dimensions, validation_cases


def collective_check(function):
    @wraps(function)
    def checked(*args, **kwargs):
        error, result = None, None
        try:
            result = function(*args, **kwargs)
        except AssertionError as failure:
            error = failure
        device = torch.device("cuda", torch.cuda.current_device()) if dist.get_backend() == "nccl" else "cpu"
        failed = torch.tensor(int(error is not None), device=device)
        dist.all_reduce(failed)
        if failed.item():
            raise error or AssertionError("comparison failed on another rank")
        return result
    return checked


@collective_check
def compare(model, reference, *, gradients=False, tolerance=3e-5, sharded_reference=False):
    reference_layouts = parameter_layouts(reference)
    expected = {reference_layouts[name].logical_name: parameter for name, parameter in reference.named_parameters()}
    layouts = parameter_layouts(model)
    maximum = 0.0
    differences = []
    for name, parameter in model.named_parameters():
        layout = layouts[name]
        source = expected[layout.logical_name]
        value = parameter.grad if gradients else parameter
        target = source.grad if gradients else source
        if value is None and target is None:
            continue
        value = torch.zeros_like(local_tensor(parameter)) if value is None else local_tensor(value)
        if value.is_sparse:
            value = value.float().to_dense()
        target = torch.zeros_like(source) if target is None else target
        if sharded_reference:
            target = local_tensor(target)
            if target.is_sparse:
                target = target.float().to_dense()
            target = target.detach().cpu()
        else:
            slices = tuple(slice(start, start+length) for start, length in zip(layout.offsets, layout.local_shape))
            target = target[slices].detach().cpu()
        value = value.detach().cpu()
        if value.numel():
            error = (value.float()-target.float()).abs().max().item()
            maximum = max(maximum, error)
            if error:
                differences.append((error, name))
        torch.testing.assert_close(value.float(), target.float(), rtol=tolerance*10, atol=tolerance, msg=name)
    if os.environ.get("DSV41_DISTRIBUTED_TRACE"):
        print(json.dumps({"rank":dist.get_rank(), "gradients":gradients,
                          "differences":sorted(differences, reverse=True)[:5]}), flush=True)
    return maximum


@collective_check
def compare_logits(actual, expected, tolerance):
    torch.testing.assert_close(actual, expected, rtol=tolerance*10, atol=tolerance)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--all-topologies", action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--case", type=int)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--detect-anomaly", action="store_true")
    parser.add_argument("--cache-bytes", type=int, default=0)
    parser.add_argument("--disable-dedup", action="store_true")
    parser.add_argument("--activation-checkpoint", choices=("none", "full"))
    parser.add_argument("--match-reference-dedup", action="store_true",
                        help="isolate placement from low-precision duplicate-reduction rounding")
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    device = torch.device("cpu" if args.cpu else f"cuda:{os.environ['LOCAL_RANK']}")
    if not args.cpu:
        torch.cuda.set_device(device)
    torch.set_num_threads(1)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    dist.init_process_group("gloo" if args.cpu else "nccl", timeout=datetime.timedelta(seconds=60),
                            device_id=None if args.cpu else device)
    temporary = tempfile.TemporaryDirectory(prefix="dsv41-memory-") if rank == 0 else None
    paths = [temporary.name if temporary else None]
    dist.broadcast_object_list(paths)
    folder = Path(paths[0])
    experts, length = fixture_dimensions(world)
    config = memory_config(n_routed_experts=experts, engram_layer_ids=[1, 3], engram_num_embeddings=[127, 131])
    if rank == 0:
        torch.manual_seed(42)
        synthetic_checkpoint(folder, config)
    dist.barrier()
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    tolerance = 2e-3 if args.bf16 else 3e-5
    cases = validation_cases(world, args.all_topologies)
    if args.case is not None and not 0 <= args.case < len(cases):
        raise ValueError("case is outside the legal topology list")
    results = []
    try:
        for case, (cp_size, ep_size, mode, storage, prefetch, dedup, offload) in enumerate(cases):
            if args.case is not None and args.case != case:
                continue
            if args.disable_dedup:
                dedup = False
            if rank == 0:
                print(json.dumps({"case_start":case,"cp":cp_size,"ep":ep_size,"mode":mode,
                                  "storage":storage,"prefetch":prefetch,"offload":offload}), flush=True)
            meshes, cp, dispatcher = build_parallelism(cp_size, ep_size, device.type)
            engram = EngramPolicy(storage, prefetch, dedup, 8192, args.cache_bytes)
            fsdp = FSDPPolicy((1,3) if offload and not args.cpu else ())
            def load(placement, *, baseline=False):
                load_mode = "layerwise" if baseline else mode
                materializer = LocalMaterializer() if args.cpu and load_mode == "direct_shard" else None
                if not args.cpu:
                    materializer = FSDPMaterializer(meshes, FSDPPolicy() if baseline else fsdp)
                model = load_dsv41_backbone_window(folder, num_layers=config.num_hidden_layers, device=device,
                    dtype=dtype, context_parallel=cp, token_dispatcher=dispatcher, engram_mesh=meshes.engram,
                    sparse_engram_gradients=True, engram_policy=placement,
                    load_policy=LoadPolicy(load_mode, 19, 131072, True), materializer=materializer)
                checkpoint_mode = args.activation_checkpoint or ("full" if dedup and not baseline else "none")
                model.set_activation_checkpointing(ActivationCheckpointConfig(checkpoint_mode))
                return model
            if not args.cpu:
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            model = load(engram)
            if not args.cpu:
                torch.cuda.synchronize(device)
            load_seconds = time.perf_counter()-started
            # BF16 updates are compared at identical reduction topology. FP32
            # independently validates the distributed math against a full model.
            reference = (load(EngramPolicy(deduplicate=dedup and args.match_reference_dedup), baseline=True) if args.bf16 else
                load_dsv41_backbone_window(folder, num_layers=config.num_hidden_layers, device=device, dtype=dtype))
            optimizer = build_optimizer(model, OptimizerPolicy("sgd", .01, True))
            ref_optimizer = (build_optimizer(reference, OptimizerPolicy("sgd", .01)) if args.bf16
                             else torch.optim.SGD(reference.parameters(), lr=.01))
            ref_gradients = GradientPolicy(reference, meshes, fsdp=not args.cpu) if args.bf16 else None
            gradients = GradientPolicy(model, meshes, fsdp=not args.cpu)
            def check(*, gradients=False):
                return compare(model, reference, gradients=gradients, tolerance=tolerance,
                               sharded_reference=args.bf16)
            check()
            generator = torch.Generator().manual_seed(7 + rank//cp_size)
            def manager(target, opt):
                return CheckpointManager(folder / f"resume-{case}", TrainingState(target, opt,
                    model_config=config.to_dict(), data_generator=generator,
                    state_adapter=StateAdapter(target, cp=cp_size, ep=ep_size)))
            batches = []
            for dp in range(world//cp_size):
                tokens = torch.randint(3,32,(1,length),generator=torch.Generator().manual_seed(123+dp)).to(device)
                mask = torch.ones_like(tokens)
                mask[:,-(dp%3+1):] = 0
                batches.append((tokens,mask))
            max_error = 0
            for step in range(3):
                if os.environ.get("DSV41_DISTRIBUTED_TRACE"):
                    print(json.dumps({"rank": rank, "case": case, "step": step, "phase": "start"}), flush=True)
                optimizer.zero_grad(set_to_none=True); ref_optimizer.zero_grad(set_to_none=True)
                gradients.begin_step()
                if ref_gradients is not None:
                    ref_gradients.begin_step()
                for micro in range(2):
                    expected_loss = 0
                    for dp, (tokens, mask) in enumerate(batches):
                        if args.bf16 and dp != rank // cp_size:
                            continue
                        tokens = tokens if micro == 0 else tokens.flip(1)
                        expected = reference(tokens, mask, labels=tokens)
                        expected_loss = expected_loss + expected.loss / (2 if args.bf16 else 2 * len(batches))
                        if dp == rank//cp_size:
                            expected_logits = expected.logits.detach()
                    expected_loss.backward()
                    tokens, mask = batches[rank//cp_size]
                    tokens = tokens if micro == 0 else tokens.flip(1)
                    actual = model(tokens, mask, labels=tokens)
                    logits = actual.logits.detach() if args.bf16 else cp.gather(actual.logits.detach())
                    compare_logits(logits, expected_logits, tolerance)
                    (actual.loss * .5).backward()
                    if os.environ.get("DSV41_DISTRIBUTED_TRACE"):
                        print(json.dumps({"rank": rank, "step": step, "micro": micro, "phase": "backward"}), flush=True)
                gradients.normalize()
                if ref_gradients is not None:
                    ref_gradients.normalize()
                max_error = max(max_error, check(gradients=True))
                optimizer.step(); ref_optimizer.step()
                check()
                if step == 0:
                    model.synchronize_engram()
                    manager(model, optimizer).save(1)
                    model.close()
                    restored = load(EngramPolicy("cpu", "sync", False, 8192, args.cache_bytes))
                    restored_optimizer = build_optimizer(restored, OptimizerPolicy("sgd", .01, True))
                    manager(restored, restored_optimizer).load()
                    compare(restored, reference, tolerance=tolerance, sharded_reference=args.bf16)
                    model, optimizer = restored, restored_optimizer
                    gradients = GradientPolicy(model, meshes, fsdp=not args.cpu)
            result = {"case":case,"rank":rank,"cp":cp_size,"ep":ep_size,"mode":mode,
                      "experts":experts,"sequence_length":length,
                      "reference_deduplicate":dedup and args.match_reference_dedup,
                      "reference":"same_topology" if args.bf16 else "unsharded", "dtype":str(dtype),
                      "cache_bytes":args.cache_bytes, "activation_checkpoint":args.activation_checkpoint,
                      "storage":storage,"prefetch":prefetch,"offload":offload,"max_gradient_error":max_error,
                      "load_seconds":load_seconds,"parameter_bytes":parameter_memory(model,optimizer),
                      "peak_gib":0 if args.cpu else torch.cuda.max_memory_allocated(device)/1024**3}
            gathered = [None]*world
            dist.all_gather_object(gathered,result)
            results.extend(gathered)
            if rank == 0:
                print(json.dumps({"case_pass":case,"max_gradient_error":max(r["max_gradient_error"] for r in gathered)}),flush=True)
                if args.output:
                    path = Path(args.output)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(results, indent=2) + "\n")
            model.close(); reference.close()
            del model, reference, optimizer, ref_optimizer, gradients, actual, expected, expected_loss
            gc.collect()
            if not args.cpu:
                torch.cuda.empty_cache()
            dist.barrier()
            # Each case creates independent meshes; release their Gloo threads
            # and NCCL communicators before the next topology.
            if args.cpu:
                for group in reversed(list(dist.distributed_c10d._world.pg_map)):
                    if group is not dist.group.WORLD:
                        dist.destroy_process_group(group)
        if rank == 0 and args.output:
            path=Path(args.output)
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(results,indent=2)+"\n")
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        dist.destroy_process_group()
        if temporary:
            temporary.cleanup()


if __name__ == "__main__":
    main()
