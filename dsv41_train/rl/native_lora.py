"""Use a pinned native distributed loader alongside the branch's RL package.

The compatibility tree preserves the reviewed EP/FSDP materializer and HF cache
math. It is source code, never model weights or an adapter checkpoint. Its
manifest is verified before importing any of its code or constructing a model.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys

import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

from .lora_config import DistributedLoRAConfig


PINNED_NATIVE_MANIFEST_SHA256 = "bc9e596346214196d1210606f93224532567ac2c12a373d23982ecfb85004324"


def default_native_source():
    return Path(__file__).parent / "_native_runtime" / "c3868011979ca"


def import_native(source_root=None):
    root = Path(default_native_source() if source_root is None else source_root).resolve()
    manifest_path = root / "source-manifest.json"
    manifest_data = manifest_path.read_bytes()
    manifest_digest = hashlib.sha256(manifest_data).hexdigest()
    bundled = root == default_native_source().resolve()
    if bundled and manifest_digest != PINNED_NATIVE_MANIFEST_SHA256:
        raise ValueError("pinned native runtime manifest digest differs")
    manifest = json.loads(manifest_data)
    files = manifest["files"]
    expected = {name: digest for name, digest in files.items()
                if name.startswith("dsv41_train/") and name.endswith(".py")}
    present = {str(path.relative_to(root)) for path in (root / "dsv41_train").rglob("*.py")}
    if not expected or present != set(expected):
        raise ValueError("native runtime Python inventory differs from its source manifest")
    for name, digest in expected.items():
        path = (root / name).resolve()
        if root not in path.parents or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"native runtime source digest differs: {name}")
    package = "_dsv41_grpo_native"
    if package in sys.modules:
        if Path(sys.modules[package].__file__).resolve() != root / "dsv41_train/__init__.py":
            raise RuntimeError("a different native runtime is already imported")
    else:
        spec = importlib.util.spec_from_file_location(package, root / "dsv41_train/__init__.py",
            submodule_search_locations=[str(root / "dsv41_train")])
        module = importlib.util.module_from_spec(spec)
        sys.modules[package] = module
        spec.loader.exec_module(module)
    def load(name):
        return importlib.import_module(package + "." + name)
    return load, {"head": manifest.get("head"),
                  "manifest_sha256": manifest_digest, "python_files": len(expected),
                  "source_kind": "bundled_pinned_native_runtime" if bundled else "external_native_runtime"}


def scoring_model_class(base_class):
    """Score aligned response targets through the native backbone and LM head.

    CP1 keeps the sequence whole. The LM-head checkpoint avoids retaining its
    vocabulary-sized logits during backward. Each batch row has its own causal
    attention/positions; explicit right-padding masks add no prompt/padding loss.
    """
    class ResponseScoringModel(base_class):
        supports_token_log_probs = True
        scoring_head_chunk_tokens = 2048

        def _score_targets(self, hidden, targets, temperature):
            logits = self.lm_head(hidden).float() / temperature
            return logits.log_softmax(-1).gather(-1, targets[:, None]).squeeze(-1)

        def forward(self, input_ids, attention_mask=None, *, target_ids=None,
                    target_mask=None, temperature=1.0):
            if (target_ids is None or target_mask is None or input_ids.ndim != 2
                    or target_ids.shape != input_ids.shape or target_mask.shape != input_ids.shape
                    or input_ids.dtype != torch.long or target_ids.dtype != torch.long
                    or not 0 < temperature < float("inf")):
                raise ValueError("native GRPO requires aligned int64 targets and a response mask")
            if self.context_parallel.size != 1:
                raise ValueError("native GRPO scoring currently requires CP1")
            token_mask = torch.ones_like(input_ids, dtype=torch.bool) if attention_mask is None else attention_mask.bool()
            if token_mask.shape != input_ids.shape:
                raise ValueError("attention mask must match input IDs")
            if ((token_mask[:, 1:] & ~token_mask[:, :-1]).any()
                    or not token_mask.any(dim=1).all()):
                raise ValueError("native response scoring requires nonempty right-padded rows")
            mask = target_mask.bool()
            if (mask & ~token_mask).any() or not mask.any(dim=1).all():
                raise ValueError("each response mask must select live tokens in its own row")
            selected = target_ids[mask]
            if not selected.numel() or ((selected < 0) | (selected >= self.config.vocab_size)).any():
                raise ValueError("response targets must be nonempty and inside the vocabulary")
            hidden, _ = self.model(input_ids, token_mask)
            chunk_tokens = self.scoring_head_chunk_tokens
            if type(chunk_tokens) is not int or chunk_tokens < 1:
                raise ValueError("scoring head chunk size must be a positive integer")
            # Only the vocabulary head is chunked: every row already shared one
            # backbone forward. Bound FP32 logits instead of retaining [B*T,V].
            chunks = []
            for current, targets in zip(hidden[mask].split(chunk_tokens), selected.split(chunk_tokens)):
                if torch.is_grad_enabled():
                    scores = checkpoint(self._score_targets, current, targets, temperature,
                                        use_reentrant=False)
                else:
                    scores = self._score_targets(current, targets, temperature)
                chunks.append(scores)
            scores = torch.cat(chunks)
            return scores.new_zeros(mask.shape).masked_scatter(mask, scores)
    return ResponseScoringModel


def query_lora_materializer(materializer_class, meshes, native_lora, lora_config, num_layers):
    """Install chosen adapters after loading each layer, before its FSDP wrap.

    The pinned checkpoint reader calls this boundary for every complete layer.
    Keeping selection here preserves that reader and its base-weight math.
    """
    selected = set(lora_config.selected_layers(num_layers))
    config = native_lora.LoRAConfig(lora_config.rank, lora_config.alpha, 0, ("attention.q_a",))

    class QueryLoRAMaterializer(materializer_class):
        def prepare(self, model, *, dtype, device, direct):
            if direct:
                raise ValueError("query LoRA requires layerwise model loading")
            model.requires_grad_(False)
            super().prepare(model, dtype=dtype, device=device, direct=direct)

        def layer_loaded(self, layer):
            if layer.layer_id in selected:
                native_lora.inject_lora(layer, config)
            super().layer_loaded(layer)

    return QueryLoRAMaterializer(meshes)


def load_distributed_learner(model_dir, source_root=None, *, seed=42, ep_size=16,
                             lora_config=None):
    lora_config = DistributedLoRAConfig() if lora_config is None else lora_config
    if not isinstance(lora_config, DistributedLoRAConfig):
        raise ValueError("learner LoRA configuration must be DistributedLoRAConfig")
    lora_config.selected_layers(40)
    native, provenance = import_native(source_root)
    runtime = native("runtime").initialize_runtime("cuda", require_distributed=True)
    if runtime.world_size != 16 or ep_size != 16:
        raise ValueError("this reviewed deployment requires CP1/EP16/FSDP16 on exactly 16 ranks")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    config = native("models.dsv4.config").DeepSeekV41Config.from_json(Path(model_dir) / "config.json")
    if (config.num_hidden_layers, config.hidden_size, config.q_lora_rank) != (40, 5120, 1280):
        raise ValueError("the learner must load the full mounted DeepSeek V4.1 backbone")
    if config.attention_dropout:
        raise ValueError("GRPO requires attention dropout zero")
    selected_layers = lora_config.selected_layers(config.num_hidden_layers)
    meshes, cp, dispatcher = native("models.dsv4.parallel").build_parallelism(cp=1, ep=ep_size)
    materializer_class = native("models.dsv4.materialize").FSDPMaterializer
    loader = native("models.dsv4.checkpoint")
    original_class = loader.DeepSeekV41ForCausalLM
    loader.DeepSeekV41ForCausalLM = scoring_model_class(original_class)
    try:
        model = loader.load_dsv41_backbone_window(model_dir, num_layers=40,
            device=runtime.device, dtype=torch.bfloat16, context_parallel=cp,
            token_dispatcher=dispatcher, engram_mesh=meshes.engram,
            cache_quantization="hf",
            load_policy=native("loading").LoadPolicy("layerwise", staging_bytes=128 * 1024**2),
            materializer=query_lora_materializer(materializer_class, meshes, native("lora"),
                lora_config, config.num_hidden_layers))
    finally:
        loader.DeepSeekV41ForCausalLM = original_class
    model.gradient_checkpointing_enable()
    model.train()
    model.distributed_lora_config = lora_config
    adapters = list(native("lora").adapter_parameters(model))
    expected_count = lora_config.parameter_count(config.hidden_size, config.q_lora_rank,
                                                config.num_hidden_layers)
    if len(adapters) != 2 * len(selected_layers) or sum(p.numel() for p in adapters) != expected_count:
        raise ValueError("native query adapter inventory differs from its configured profile")
    optimizer = torch.optim.AdamW(adapters, lr=1e-4, weight_decay=0, foreach=False)
    gradient_policy = native("models.dsv4.policies").GradientPolicy(model, meshes, fsdp=True)
    return model, optimizer, gradient_policy, runtime, provenance


@contextmanager
def reference_base(model):
    """The frozen BF16 native base is the reference; no second model is loaded."""
    modules = [module for module in model.modules() if hasattr(module, "adapters_enabled")]
    states = [module.adapters_enabled for module in modules]
    if not modules or not all(states):
        raise ValueError("reference scoring requires enabled unmerged live adapters")
    try:
        for module in modules:
            module.adapters_enabled = False
        with torch.no_grad():
            yield
    finally:
        for module, enabled in zip(modules, states):
            module.adapters_enabled = enabled


def local_tensor(value):
    return value.to_local() if hasattr(value, "to_local") else value


def optimizer_parameters(optimizer):
    return [parameter for group in optimizer.param_groups for parameter in group["params"]]


def adapter_snapshot(optimizer):
    return [local_tensor(p.detach()).clone() for p in optimizer_parameters(optimizer)]


@contextmanager
def adapter_parameters_at(model, optimizer, values):
    """Score an immutable generating policy, then restore live adapter shards.

    Call only between completed forward/backward pairs. Resharding invalidates
    any FSDP all-gather cache before changing local shards; parameter objects,
    accumulated gradients and optimizer state remain untouched.
    """
    try:
        from torch.distributed.fsdp import FSDPModule
    except ImportError:
        from torch.distributed._composable.fsdp import FSDPModule
    modules = [module for module in model.modules() if isinstance(module, FSDPModule)]
    parameters = optimizer_parameters(optimizer)
    if len(values) != len(parameters) or any(
            value.shape != local_tensor(parameter).shape
            or value.dtype != parameter.dtype or value.device != parameter.device
            for value, parameter in zip(values, parameters)):
        raise ValueError("generating-policy adapter shards differ from the learner layout")

    def restore(shards):
        for module in reversed(modules):
            module.reshard()
        with torch.no_grad():
            for parameter, value in zip(parameters, shards):
                local_tensor(parameter).copy_(value)

    current = adapter_snapshot(optimizer)
    try:
        restore(values)
        with torch.no_grad():
            yield
    finally:
        restore(current)


def gradient_norm(optimizer):
    total = torch.zeros((), device=optimizer_parameters(optimizer)[0].device, dtype=torch.float64)
    invalid = torch.zeros((), device=total.device, dtype=torch.int32)
    for parameter in optimizer_parameters(optimizer):
        if parameter.grad is None:
            invalid.fill_(1)
            continue
        gradient = local_tensor(parameter.grad)
        if not torch.isfinite(gradient).all():
            invalid.fill_(1)
        total += gradient.double().square().sum()
    if dist.is_initialized():
        dist.all_reduce(invalid, op=dist.ReduceOp.MAX)
        dist.all_reduce(total)
    if invalid.item():
        raise ValueError("a query adapter has no gradient or a nonfinite gradient on a world rank")
    return total.sqrt().item()


def verify_optimizer_step(model, optimizer, before):
    parameters = optimizer_parameters(optimizer)
    ids = {id(p) for p in parameters}
    error = "base parameters must remain frozen and gradient free" if any(
        p.requires_grad or p.grad is not None for p in model.parameters() if id(p) not in ids) else None
    changed = torch.zeros((), device=parameters[0].device, dtype=torch.int64)
    for previous, parameter in zip(before, parameters):
        value = local_tensor(parameter.detach())
        if not torch.isfinite(value).all():
            error = "adapter parameter is nonfinite after the optimizer step"
        changed += (value != previous).sum()
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor) and not torch.isfinite(local_tensor(value)).all():
                error = "optimizer state is nonfinite"
    if dist.is_initialized():
        errors = [None] * dist.get_world_size()
        dist.all_gather_object(errors, error)
        error = next((value for value in errors if value), None)
        dist.all_reduce(changed)
    if error:
        raise ValueError(error)
    if not changed.item():
        raise ValueError("optimizer step changed no trainable adapter element")
    return changed.item()
