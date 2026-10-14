"""Optional, newly initialized Engram augmentation for pretrained Qwen3 MoE."""

from dataclasses import dataclass
from types import SimpleNamespace

import torch
from torch import nn

from ..dsv4.engram import EngramPolicy, RowShardedEmbedding
from ..dsv4.engram.hash import NgramHash
from ..dsv4.model import Engram


@dataclass(frozen=True)
class QwenEngramConfig:
    layer_ids: tuple[int, ...] = (1, 24)
    vocab_per_head: int = 900_000
    head_dim: int = 256
    heads: int = 8
    max_ngram: int = 4
    seed: int = 20261013

    def __post_init__(self):
        if not self.layer_ids or len(set(self.layer_ids)) != len(self.layer_ids) or min(self.layer_ids) < 0:
            raise ValueError("Engram requires distinct nonnegative layer IDs")
        if min(self.vocab_per_head, self.head_dim, self.heads) < 1 or self.max_ngram < 2:
            raise ValueError("Engram dimensions must be positive and n-grams start at two")


def attach_engram(model, config, policy=None, mesh=None, *, device="cpu", dtype=torch.bfloat16,
                  table_initializer=None):
    """Add tables before FSDP; initialize CPU/GPU variants from identical CPU RNG."""
    if hasattr(model.model, "engram_tables"):
        raise ValueError("Qwen already has Engram tables")
    if max(config.layer_ids) >= len(model.model.layers):
        raise ValueError("Engram layer lies outside the Qwen backbone")
    policy = policy or EngramPolicy()
    columns = (config.max_ngram - 1) * config.heads
    options = SimpleNamespace(
        vocab_size=model.config.vocab_size,
        hidden_size=model.config.hidden_size,
        rms_norm_eps=model.config.rms_norm_eps,
        hc_mult=1,
        engram_max_ngram_size=config.max_ngram,
        engram_n_heads=config.heads,
        engram_head_dim=config.head_dim,
        engram_vocab_size=config.vocab_per_head,
        engram_compressed_vocab_size=model.config.vocab_size,
        engram_pad_id=0,
        engram_layer_ids=list(config.layer_ids),
        engram_num_embeddings=[columns * (config.vocab_per_head + 10000)] * len(config.layer_ids),
    )
    with torch.device("cpu"):
        hashing = NgramHash(options)
    sizes = (hashing.offsets[:, -1] + hashing.primes.flatten(1)[:, -1]).tolist()
    model.model.engram_hash = hashing.to(device)
    model.model.engram_layer_ids = list(config.layer_ids)
    model.model.engram_tables = nn.ModuleDict()
    model.engram_config = config
    for layer_id, count in zip(config.layer_ids, sizes):
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            # manual_seed also seeds every CUDA generator, which a CPU-only
            # fork cannot restore. Seed only the CPU stream used by nn.Linear.
            torch.random.set_rng_state(
                torch.Generator(device="cpu").manual_seed(config.seed + layer_id).get_state()
            )
            block = Engram(options)
            nn.init.normal_(block.proj.weight, std=model.config.initializer_range)
        model.model.layers[layer_id].engram = block.to(device=device, dtype=dtype)
        with torch.device("meta"):
            table = RowShardedEmbedding(count, config.head_dim, mesh, sparse_gradients=True, policy=policy)
        target = torch.device("cpu") if policy.device == "cpu" else torch.device(device)
        table.weight = nn.Parameter(torch.empty(table.row_stop-table.row_start, config.head_dim,
                                                dtype=dtype, device=target))
        if table_initializer is None:
            generator = torch.Generator(device="cpu").manual_seed(config.seed + layer_id * 100003 + table.row_start)
            # Bounded initialization gives identical values regardless of table placement.
            with torch.no_grad():
                for start in range(0, table.weight.shape[0], 8192):
                    destination = table.weight[start:start+8192]
                    rows = torch.empty(destination.shape, dtype=dtype, device="cpu").normal_(
                        std=model.config.initializer_range, generator=generator)
                    destination.copy_(rows)
        else:
            table_initializer(table, layer_id=layer_id)
        model.model.engram_tables[str(layer_id)] = table


def shard_qwen_component(name, module, meshes):
    """Layerwise FSDP materialization using the same topology as DSV4."""
    from torch.distributed.fsdp import fully_shard

    if name.startswith("model.layers."):
        expert_mesh = meshes.expert_fsdp if meshes.ep is not None else meshes.fsdp
        if hasattr(module.mlp, "experts"):
            fully_shard(module.mlp.experts, mesh=expert_mesh, reshard_after_forward=True)
    if name != "model.norm":
        fully_shard(module, mesh=meshes.fsdp, reshard_after_forward=True)


def shard_qwen_root(model, meshes):
    from torch.distributed.fsdp import FSDPModule, fully_shard

    ignored = {table.weight for table in getattr(model.model, "engram_tables", {}).values()}
    fully_shard(model, mesh=meshes.fsdp, ignored_params=ignored, reshard_after_forward=True)
    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.set_modules_to_backward_prefetch([module])
