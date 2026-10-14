"""Replay a training process's small eval reference after loading its exported adapter.

torchrun --nproc-per-node=2 scripts/verify_lora_adapter.py \
  --reference RUN/lora_eval_reference.json --adapter-in RUN/adapter.pt \
  --output RUN/adapter_reload.json

Loads the real Qwen backbone or DSV4 prefix/window once, with the existing loaders.
Writes no full-model checkpoint. This verifies bounded eval outputs and artifact
identity, not model quality or full-model equivalence. Trainer callers can import
capture_probe and make_reference without executing the CLI.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch
import torch.distributed as dist

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dsv41_train.loading import LoadPolicy
from dsv41_train.lora import LoRAConfig, read_adapter
from dsv41_train.models.dsv4 import load_dsv41_backbone_window
from dsv41_train.models.dsv4.engram import EngramPolicy
from dsv41_train.models.dsv4.materialize import FSDPMaterializer
from dsv41_train.models.dsv4.parallel import build_parallelism
from dsv41_train.models.dsv4.policies import FSDPPolicy
from dsv41_train.models.qwen import load_qwen3_moe
from dsv41_train.models.qwen.engram import shard_qwen_component, shard_qwen_root
from dsv41_train.runtime import initialize_runtime


INPUT_KEYS = ("input_ids", "labels", "attention_mask")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def input_sha256(inputs):
    return hashlib.sha256(json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def require_all(condition, runtime, message):
    flag = torch.as_tensor(condition, dtype=torch.int32, device=runtime.device).all().int()
    if dist.is_initialized():
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    require(bool(flag.item()), message)


def probe_indices(numel, count=256, *, device=None):
    require(type(count) is int and 1 <= count <= 256, "sample_count must be in 1..256")
    require(type(numel) is int and numel > 0, "logits must be nonempty")
    count = min(numel, count)
    positions = torch.arange(count, dtype=torch.int64, device=device)
    if count == 1:
        return positions
    quotient, remainder = divmod(numel - 1, count - 1)
    return positions * quotient + positions * remainder // (count - 1)


def capture_probe(model, batch, runtime, sample_count=256):
    """Collect rank-local eval results without perturbing training flags or RNG.

All distributed ranks must call. The complete small JSON object is returned on
every rank. Preserve caller-owned batches; only the three forward inputs are used.
"""
    inputs = {key: batch[key].detach().to(runtime.device) for key in INPUT_KEYS}
    inputs["attention_mask"] = inputs["attention_mask"].bool()
    ids, labels, mask = (inputs[key] for key in INPUT_KEYS)
    valid = (ids.ndim == 2 and ids.shape == labels.shape == mask.shape and ids.shape[1] >= 2
             and ids.numel() <= 4096 and ids.dtype == labels.dtype == torch.int64)
    require_all(valid, runtime, "eval probe needs matching int64 [batch,sequence] IDs/labels and mask, 2+ positions, <=4096 tokens")
    count = (labels[:, 1:].ne(-100) & mask[:, 1:].bool()).sum()
    require_all(count > 0, runtime, "eval probe needs shifted supervised targets on every rank")
    require_all(type(sample_count) is int and 1 <= sample_count <= 256, runtime, "sample_count must be in 1..256")
    states = [(module, module.training) for module in model.modules()]
    cuda_devices = [runtime.device.index if runtime.device.index is not None else torch.cuda.current_device()] if runtime.device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=cuda_devices), torch.no_grad():
            model.eval()
            output = model(**inputs)
            require_all(isinstance(output.loss, torch.Tensor) and output.loss.numel() == 1,
                        runtime, "model did not return a scalar eval loss")
            require_all(torch.isfinite(output.loss), runtime, "non-finite eval loss")
            logits = output.logits.detach()
            require_all(logits.ndim == 3 and logits.shape[:2] == ids.shape,
                        runtime, "eval logits must have [batch,sequence,vocabulary] shape (CP=1)")
            require_all(torch.isfinite(logits).all(), runtime, "non-finite eval logits")
            indices = probe_indices(logits.numel(), sample_count, device=logits.device)
            serialized = {key: value.cpu().tolist() for key, value in inputs.items()}
            local = {"rank": runtime.rank, "inputs": serialized, "input_sha256": input_sha256(serialized),
                     "loss": output.loss.item(), "shifted_targets": int(count.item()),
                     "logits_shape": list(logits.shape), "logit_indices": indices.cpu().tolist(),
                     "logit_values": logits.reshape(-1)[indices].float().cpu().tolist()}
    finally:
        # Restore even intentionally mixed train/eval submodules exactly.
        for module, training in states:
            module.training = training
    ranks = [None] * runtime.world_size
    if dist.is_initialized():
        dist.all_gather_object(ranks, local)
    else:
        require(runtime.world_size == 1, "distributed eval capture requires an initialized process group")
        ranks[0] = local
    result = {"format_version": 1, "world_size": runtime.world_size, "ranks": ranks}
    validate_probe(result)
    return result


def validate_probe(probe):
    require(probe["format_version"] == 1, "unsupported eval probe version")
    world = probe["world_size"]
    require(type(world) is int and world >= 1 and len(probe["ranks"]) == world, "invalid eval probe rank count")
    require({row["rank"] for row in probe["ranks"]} == set(range(world)), "eval probe rank IDs differ")
    for row in probe["ranks"]:
        inputs = row["inputs"]
        require(set(inputs) == set(INPUT_KEYS) and input_sha256(inputs) == row["input_sha256"], "eval probe token identity mismatch")
        ids = torch.tensor(inputs["input_ids"], dtype=torch.int64)
        labels = torch.tensor(inputs["labels"], dtype=torch.int64)
        mask = torch.tensor(inputs["attention_mask"], dtype=torch.bool)
        require(ids.ndim == 2 and ids.shape == labels.shape == mask.shape and ids.shape[1] >= 2
                and ids.numel() <= 4096, "invalid eval probe input shapes")
        count = int((labels[:, 1:].ne(-100) & mask[:, 1:]).sum())
        require(count > 0 and row["shifted_targets"] == count, "invalid shifted target count")
        shape = row["logits_shape"]
        require(len(shape) == 3 and shape[:2] == list(ids.shape) and type(shape[2]) is int and shape[2] > 0, "invalid logits shape")
        require(ids.min() >= 0 and ids.max() < shape[2], "probe token outside vocabulary")
        active = labels[labels.ne(-100)]
        require(active.min() >= 0 and active.max() < shape[2], "probe label outside vocabulary")
        indices, values = row["logit_indices"], row["logit_values"]
        require(0 < len(indices) == len(values) <= 256 and len(set(indices)) == len(indices)
                and all(type(index) is int and 0 <= index < math.prod(shape) for index in indices), "invalid logit sample indices")
        require(all(type(value) in (int, float) and math.isfinite(value) for value in values + [row["loss"]]), "non-finite eval probe values")


def compare_probes(actual, expected, *, atol=1e-3, rtol=1e-2):
    """Compare by rank with explicit input/sampling identity and bounded tolerances."""
    require(all(math.isfinite(value) and value >= 0 for value in (atol, rtol)), "tolerances must be finite and nonnegative")
    validate_probe(actual)
    validate_probe(expected)
    require(actual["world_size"] == expected["world_size"], "probe world sizes differ")
    reference = {row["rank"]: row for row in expected["ranks"]}
    records = []
    for row in actual["ranks"]:
        other = reference[row["rank"]]
        require(all(row[key] == other[key] for key in ("input_sha256", "inputs", "logits_shape", "logit_indices", "shifted_targets")), "probe inputs or logit sample positions differ")
        differences = [abs(left-right) for left, right in zip(row["logit_values"], other["logit_values"])]
        mismatch = sum(delta > atol + rtol * abs(value) for delta, value in zip(differences, other["logit_values"]))
        loss_delta = abs(row["loss"] - other["loss"])
        records.append({"rank": row["rank"], "logit_max_abs_difference": max(differences),
                        "logit_outside_tolerance": mismatch, "loss_abs_difference": loss_delta,
                        "passed": mismatch == 0 and loss_delta <= atol + rtol * abs(other["loss"]),
                        "output_changed": max(differences) > 0 or loss_delta > 0})
    return {"passed": all(row["passed"] for row in records),
            "output_changed": any(row["output_changed"] for row in records),
            "atol": atol, "rtol": rtol, "ranks": records}


def make_reference(training_config, initial, trained, adapter_path=None):
    """Call on rank 0 after save_adapter and its completion barrier."""
    require(training_config["model_type"] in ("qwen", "dsv4") and training_config["lora"] is not None,
            "reference requires Qwen or DSV4 LoRA training")
    require(training_config["world_size"] == initial["world_size"] == trained["world_size"], "training/probe world sizes differ")
    change = compare_probes(trained, initial, atol=0, rtol=0)
    require(change["output_changed"], "LoRA training did not change the fixed eval loss/logit probes")
    return {"format_version": 1, "context": json.loads(json.dumps(training_config)),
            "adapter": ({"path": str(Path(adapter_path).resolve()), "sha256": file_sha256(adapter_path)}
                        if adapter_path is not None else None),
            "initial": initial, "trained": trained, "training_output_change": change,
            "scope": "Fixed SFT-token eval outputs of the selected Qwen backbone or DSV4 prefix/window, bounded logits; initial is this invocation's starting state (possibly resumed), not necessarily zero-B LoRA. Not a model-quality evaluation."}


def validate_reference(reference, adapter_path, model_path=None):
    require(reference["format_version"] == 1, "unsupported reference version")
    context = reference["context"]
    require(context["model_type"] in ("qwen", "dsv4") and context["lora"] is not None and context["cp_size"] == 1,
            "reference must describe Qwen or DSV4 LoRA with CP=1")
    require(context["model_type"] != "qwen" or not context.get("qwen_engram"),
            "Qwen LoRA with added Engram is not supported")
    require(compare_probes(reference["trained"], reference["initial"], atol=0, rtol=0)["output_changed"],
            "reference does not demonstrate training output change")
    require(context["world_size"] == reference["trained"]["world_size"], "reference topology differs")
    require(reference.get("adapter") is not None, "reference has no adapter export identity; rerun with --adapter-out")
    folder = Path(model_path or context["source"]["path"])
    for filename, key in (("config.json", "config_sha256"), ("model.safetensors.index.json", "index_sha256")):
        require(file_sha256(folder / filename) == context["source"][key], f"base {filename} identity differs from training")
    require(file_sha256(adapter_path) == reference["adapter"]["sha256"], "adapter file SHA256 differs from the training export")
    payload = read_adapter(adapter_path)
    expected = dict(context["lora"])
    expected["targets"] = tuple(expected["targets"])
    lora = LoRAConfig(**expected)
    require(payload["modules"] and all(config == expected for config in payload["modules"].values()), "exported adapter LoRA configuration differs")
    require(payload["layer_ids"] == list(range(context["start_layer"], context["start_layer"] + context["num_layers"])), "exported adapter layer window differs")
    require(payload["weights"] and all(torch.isfinite(tensor).all() for tensor in payload["weights"].values()), "exported adapter contains non-finite weights")
    return folder, lora


def verify(args, runtime):
    reference = json.loads(Path(args.reference).read_text())
    folder, lora = validate_reference(reference, args.adapter_in, args.model_path)
    context = reference["context"]
    require(context["world_size"] == runtime.world_size, "launch with the reference WORLD_SIZE")
    identities = [None] * runtime.world_size
    identity = {"reference_sha256": file_sha256(args.reference), "adapter_sha256": file_sha256(args.adapter_in), "model_path": str(folder.resolve())}
    dist.all_gather_object(identities, identity)
    require(all(row == identity for row in identities), "reference/adapter/source differ across ranks")
    output = Path(args.output)
    require(not output.exists(), "output exists; choose a new verification JSON path")
    meshes, cp, dispatcher = build_parallelism(1, context["ep_size"], runtime.device.type)
    started = time.perf_counter()
    with ExitStack() as cleanup:
        if context["model_type"] == "qwen":
            model = load_qwen3_moe(
                folder, device=runtime.device, dtype=torch.bfloat16,
                context_parallel=cp, token_dispatcher=dispatcher,
                lora_config=lora, adapter_path=args.adapter_in,
                module_callback=lambda name, module: shard_qwen_component(name, module, meshes),
            )
            shard_qwen_root(model, meshes)
        else:
            load_values, engram_values, fsdp_values = context["memory_policies"]
            loading, engram, fsdp = LoadPolicy(**load_values), EngramPolicy(**engram_values), FSDPPolicy(**fsdp_values)
            require(loading.mode == "layerwise" and engram.device == "cuda" and not fsdp.cpu_offload_layers,
                    "LoRA verification requires layerwise loading without CPU offload")
            model = load_dsv41_backbone_window(
                folder, start_layer=context["start_layer"], num_layers=context["num_layers"],
                device=runtime.device, dtype=torch.bfloat16, context_parallel=cp,
                token_dispatcher=dispatcher, engram_mesh=meshes.engram,
                sparse_engram_gradients=context["optimizer"] == "sgd",
                materializer=FSDPMaterializer(meshes, fsdp), load_policy=loading,
                engram_policy=engram, lora_config=lora, adapter_path=args.adapter_in,
                cache_quantization=context["cache_quantization"],
            )
            cleanup.callback(model.close)
        model.config.router_aux_loss_coef = context["router_aux_loss_coef"]
        row = next(row for row in reference["trained"]["ranks"] if row["rank"] == runtime.rank)
        batch = {key: torch.tensor(row["inputs"][key], dtype=torch.bool if key == "attention_mask" else torch.int64) for key in INPUT_KEYS}
        probe = capture_probe(model, batch, runtime, sample_count=len(row["logit_indices"]))
        replay = compare_probes(probe, reference["trained"], atol=args.atol, rtol=args.rtol)
        initial = compare_probes(probe, reference["initial"], atol=0, rtol=0)
        result = {"validation_passed": replay["passed"] and initial["output_changed"],
                  **identity, "context": context, "adapter_reload_vs_trained": replay,
                  "adapter_reload_vs_initial": initial, "probe": probe,
                  "load_and_eval_seconds_rank0": time.perf_counter() - started,
                  "scope": reference["scope"],
                  "limitations": "Bounded logits plus scalar loss; base identity covers config/index, not a hash of all checkpoint shards. Freeze/gradient/update checks are separate training evidence."}
        if runtime.is_main:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            print(json.dumps({"validation_passed": result["validation_passed"], "output": str(output)}), flush=True)
        return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--adapter-in", required=True)
    parser.add_argument("--model-path", help="Optional relocation of the identical base checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-2)
    args = parser.parse_args(argv)
    if not all(math.isfinite(value) and value >= 0 for value in (args.atol, args.rtol)):
        parser.error("tolerances must be finite and nonnegative")
    return args


def main():
    args = parse_args()
    runtime = None
    try:
        runtime = initialize_runtime(require_distributed=True)
        torch.set_num_threads(1)
        return 0 if verify(args, runtime)["validation_passed"] else 1
    finally:
        if runtime is not None and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
