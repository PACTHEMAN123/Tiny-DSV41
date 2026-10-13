"""CLI assembly for independently selectable loading and placement policies."""

from .loading import LoadPolicy
from .models.dsv4.engram import EngramPolicy
from .models.dsv4.policies import FSDPPolicy


def layer_ids(value):
    selected = set()
    for part in value.split(","):
        if not part:
            continue
        if ":" in part:
            start, stop = map(int, part.split(":"))
            if stop <= start:
                raise ValueError("offload ranges must have start < stop")
            selected.update(range(start, stop))
        else:
            selected.add(int(part))
    return tuple(sorted(selected))


def add_memory_options(parser):
    parser.add_argument("--load-mode", choices=("layerwise", "direct_shard"), default="layerwise")
    parser.add_argument("--load-chunk-rows", type=int, default=131072)
    parser.add_argument("--load-pipeline", action="store_true")
    parser.add_argument("--staging-mib", type=int, default=256)
    parser.add_argument("--engram-device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--engram-prefetch", choices=("sync", "async"), default="sync")
    parser.add_argument("--engram-deduplicate", action="store_true")
    parser.add_argument("--engram-cache-mib", type=int, default=0,
                        help="GPU row cache budget per local CPU Engram table; 0 disables caching")
    parser.add_argument("--fsdp-offload-layers", default="")
    parser.add_argument("--fsdp-offload-targets", default="dense,experts")
    parser.add_argument("--fsdp-offload-pin-memory", action="store_true")
    parser.add_argument("--optimizer-foreach", action="store_true")


def memory_policies(args):
    staging = args.staging_mib * 1024**2
    loading = LoadPolicy(args.load_mode, args.load_chunk_rows, staging, args.load_pipeline)
    engram = EngramPolicy(args.engram_device, args.engram_prefetch, args.engram_deduplicate, staging,
                          args.engram_cache_mib * 1024**2)
    fsdp = FSDPPolicy(layer_ids(args.fsdp_offload_layers), tuple(args.fsdp_offload_targets.split(",")),
                      pin_memory=args.fsdp_offload_pin_memory)
    selected = set(range(args.start_layer, args.start_layer + args.num_layers))
    if set(fsdp.cpu_offload_layers) - selected:
        raise ValueError("FSDP offload layers must be inside the selected backbone window")
    if engram.device == "cpu" and args.optimizer != "sgd":
        raise ValueError("CPU Engram requires the sparse SGD training path")
    if args.lora and (loading.mode == "direct_shard" or engram.device == "cpu" or fsdp.cpu_offload_layers):
        raise ValueError("LoRA with direct loading or CPU offload is not supported")
    return loading, engram, fsdp
