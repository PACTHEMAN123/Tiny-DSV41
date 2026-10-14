"""Two-rank Qwen LoRA EP training, DCP continuation and portable adapter reload.

Use torchrun with --cpu for Gloo EP (replicated dense parameters), or without
--cpu for CUDA FSDP2+EP. Creates only a tiny temporary safetensors checkpoint.
"""

import argparse
import json
from pathlib import Path
import tempfile

import torch
import torch.distributed as dist

from dsv41_train.checkpoint import CheckpointManager, TrainingState
from dsv41_train.layout import local_tensor
from dsv41_train.lora import LoRAConfig, save_adapter
from dsv41_train.models.dsv4.parallel import build_parallelism
from dsv41_train.models.dsv4.policies import GradientPolicy
from dsv41_train.models.qwen import Qwen3MoeForCausalLM, load_qwen3_moe
from dsv41_train.models.qwen.checkpoint import QWEN_LORA_TARGETS
from dsv41_train.models.qwen.engram import shard_qwen_component, shard_qwen_root
from dsv41_train.optimization import OptimizerPolicy, build_optimizer
from dsv41_train.runtime import Runtime, initialize_runtime
from dsv41_train.sharded_state import StateAdapter
from qwen_helpers import tiny_config, write_hf_checkpoint
from scripts.verify_lora_adapter import capture_probe, compare_probes
from train_sft import LoRAVerifier, _source_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--checkpoint-state", choices=("full", "lora"), default="full")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.cpu:
        dist.init_process_group("gloo")
        runtime = Runtime(torch.device("cpu"), rank=dist.get_rank(), world_size=dist.get_world_size())
    else:
        runtime = initialize_runtime(require_distributed=True)
    temporary = None
    try:
        if runtime.world_size != 2:
            raise ValueError("this check requires exactly two ranks")
        temporary = tempfile.TemporaryDirectory() if runtime.is_main else None
        folders = [temporary.name if temporary is not None else None]
        dist.broadcast_object_list(folders)
        folder = Path(folders[0])
        base = folder / "base"
        config = tiny_config(mlp_only_layers=[0])
        if runtime.is_main:
            base.mkdir()
            torch.manual_seed(91)
            write_hf_checkpoint(base, Qwen3MoeForCausalLM(config))
        dist.barrier()
        meshes, cp, dispatcher = build_parallelism(1, 2, runtime.device.type)
        lora = LoRAConfig(rank=2, alpha=4, dropout=.1, targets=QWEN_LORA_TARGETS)
        def load(adapter=None):
            torch.manual_seed(317)
            callback = None if args.cpu else lambda name, module: shard_qwen_component(name, module, meshes)
            model = load_qwen3_moe(base, device=runtime.device, dtype=torch.bfloat16,
                                   context_parallel=cp, token_dispatcher=dispatcher,
                                   module_callback=callback, lora_config=lora, adapter_path=adapter)
            if not args.cpu:
                shard_qwen_root(model, meshes)
            model.gradient_checkpointing_enable()
            return model
        ids = torch.tensor([[1, 3 + runtime.rank, 6, 8, 2, 0]], device=runtime.device)
        labels = ids.masked_fill(ids.eq(0), -100)
        labels[:, :2] = -100
        batch = {"input_ids": ids, "labels": labels, "attention_mask": ids.ne(0)}
        def setup():
            model = load()
            optimizer = build_optimizer(model, OptimizerPolicy("adamw", .01))
            policy = GradientPolicy(model, meshes, fsdp=not args.cpu)
            manager = CheckpointManager(folder / "checkpoint", TrainingState(
                model, optimizer, model_config=model.config.to_dict(),
                data_generator=torch.Generator().manual_seed(runtime.rank),
                training_config={"case": "qwen_lora", "ep": 2, "source": _source_identity(base)},
                state_adapter=StateAdapter(model, ep=2), checkpoint_state=args.checkpoint_state))
            return model, optimizer, policy, manager
        def update(model, optimizer, policy, verifier):
            optimizer.zero_grad(set_to_none=True)
            policy.begin_step()
            loss = model(**batch).loss
            loss.backward()
            policy.normalize()
            gradients = verifier.gradients()
            optimizer.step()
            metrics = verifier.after_step(gradients)
            return loss.detach(), metrics
        model, optimizer, policy, manager = setup()
        verifier = LoRAVerifier(model, optimizer, runtime)
        _, first = update(model, optimizer, policy, verifier)
        assert first["a"]["gradient_nonzero"] == 0
        manager.save(1)
        expected_loss, second = update(model, optimizer, policy, verifier)
        assert second["a"]["gradient_nonzero"] > 0
        verifier.finish()
        expected_probe = capture_probe(model, batch, runtime)
        restored, restored_optimizer, restored_policy, restored_manager = setup()
        assert restored_manager.load() == 1
        restored_verifier = LoRAVerifier(restored, restored_optimizer, runtime)
        actual_loss, _ = update(restored, restored_optimizer, restored_policy, restored_verifier)
        torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
        for (_, expected), (_, actual) in zip(model.named_parameters(), restored.named_parameters()):
            torch.testing.assert_close(local_tensor(actual), local_tensor(expected), rtol=0, atol=0)
            for key, value in restored_optimizer.state.get(actual, {}).items():
                torch.testing.assert_close(local_tensor(value), local_tensor(optimizer.state[expected][key]), rtol=0, atol=0)
        restored_verifier.finish()
        path = folder / "adapter.pt"
        save_adapter(restored, path)
        dist.barrier()
        fresh = load(path)
        replay = compare_probes(capture_probe(fresh, batch, runtime), expected_probe, atol=0, rtol=0)
        assert replay["passed"], replay
        dist.barrier()
        if runtime.is_main:
            print(json.dumps({"passed": True, "world_size": 2, "ep": 2,
                              "fsdp": not args.cpu, "dtype": "bfloat16",
                              "checkpoint_state": args.checkpoint_state,
                              "dropout": .1, "resume_exact": True, "adapter_reload_exact": True}), flush=True)
    finally:
        dist.destroy_process_group()
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    main()
