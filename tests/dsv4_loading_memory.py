"""Compare synthetic checkpoint loading with the pinned upstream loader on two GPUs."""

import gc
import json
import subprocess
import sys
import tempfile
import types
from dataclasses import replace
from pathlib import Path

import torch
import torch.distributed as dist

from dsv41_train.models.dsv4 import DeepSeekV41Config, load_dsv41_backbone_window
from dsv41_train.models.dsv4.parallel import apply_fsdp2_layer, apply_fsdp2_root, build_parallelism
from dsv41_train.runtime import initialize_runtime
from test_dsv4_loading import synthetic_checkpoint


def main():
    runtime = initialize_runtime(require_distributed=True)
    torch.set_num_threads(2)
    temporary = tempfile.TemporaryDirectory() if runtime.is_main else None
    paths = [temporary.name if temporary else None]
    dist.broadcast_object_list(paths)
    try:
        if runtime.world_size != 2:
            raise ValueError("this loading comparison requires two GPUs")
        folder = Path(paths[0])
        if runtime.is_main:
            config = replace(DeepSeekV41Config.tiny(), engram_layer_ids=[], engram_num_embeddings=[])
            synthetic_checkpoint(folder, config)
        dist.barrier()
        baseline = types.ModuleType("dsv41_train.models.dsv4._baseline_checkpoint")
        baseline.__package__ = "dsv41_train.models.dsv4"
        sys.modules[baseline.__name__] = baseline
        source = subprocess.check_output([
            "git", "show", "fc1338428930f960c06f0e5f42c3f9c208cbc028:dsv41_train/models/dsv4/checkpoint.py",
        ], text=True)
        exec(compile(source, "upstream_checkpoint.py", "exec"), baseline.__dict__)
        meshes, cp, dispatcher = build_parallelism(1, 1)
        results = []
        for name, loader in (("upstream", baseline.load_dsv41_backbone_window),
                             ("bounded_temporaries", load_dsv41_backbone_window)):
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            after_layer = []

            def loaded(layer):
                apply_fsdp2_layer(layer, meshes)
                torch.cuda.synchronize()
                after_layer.append(torch.cuda.memory_allocated() / 2**20)

            model = loader(folder, num_layers=6, device=runtime.device, dtype=torch.bfloat16,
                           context_parallel=cp, token_dispatcher=dispatcher,
                           engram_mesh=meshes.engram, layer_loaded=loaded)
            apply_fsdp2_root(model, meshes)
            torch.cuda.synchronize()
            result = {"loader": name, "rank": runtime.rank,
                      "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                      "after_layer_mib": after_layer}
            gathered = [None] * runtime.world_size
            dist.all_gather_object(gathered, result)
            results.extend(gathered)
            if runtime.is_main:
                print(json.dumps(gathered), flush=True)
            del model
        if runtime.is_main:
            path = Path("artifacts/loading-memory.json")
            path.parent.mkdir(exist_ok=True)
            path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
            for rank in range(2):
                before = results[rank]["peak_allocated_mib"]
                after = results[rank + 2]["peak_allocated_mib"]
                if after >= before:
                    raise AssertionError(f"loading peak did not decrease on rank {rank}")
    finally:
        dist.destroy_process_group()
        if temporary:
            temporary.cleanup()


if __name__ == "__main__":
    main()
