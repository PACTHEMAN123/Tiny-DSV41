"""Checkpoint state and storage management for training."""

from __future__ import annotations

import re
import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch import nn
from torch.distributed.checkpoint.state_dict import (
    get_state_dict, set_state_dict, get_model_state_dict, set_model_state_dict, StateDictOptions,
)
from torch.distributed.checkpoint.stateful import Stateful

from .lora import LoRALinear


class TrainingState(Stateful):
    """The mutable state required to resume a training run."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        model_config: dict[str, Any],
        data_generator: torch.Generator,
        training_config: dict[str, Any] | None = None,
        state_adapter=None,
        checkpoint_state: str = "full",
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.model_config = model_config
        self.data_generator = data_generator
        self.training_config = training_config
        self.state_adapter = state_adapter
        if checkpoint_state not in ("full", "lora"):
            raise ValueError("checkpoint_state must be full or lora")
        self.checkpoint_state = checkpoint_state
        self._optimizer_schema = None
        self.step = 0
        if self.checkpoint_state == "lora":
            self._lora_state_keys()

    def _lora_state_keys(self, *, logical=False):
        """Validate adapter-only ownership and include every persistent buffer."""
        if self.state_adapter is None:
            raise ValueError("LoRA checkpoints require a StateAdapter")
        source = (self.training_config or {}).get("source", {})
        if not isinstance(source, dict) or any(not isinstance(source.get(key), str) or not source[key]
               for key in ("path", "config_sha256", "index_sha256")):
            raise ValueError("LoRA checkpoints require the frozen base source identity")
        adapters = {id(getattr(module, name)) for module in self.model.modules()
                    if isinstance(module, LoRALinear) for name in ("lora_a", "lora_b")}
        parameters = dict(self.model.named_parameters())
        if not adapters or {id(parameter) for parameter in parameters.values() if parameter.requires_grad} != adapters:
            raise ValueError("LoRA checkpoints require all base parameters frozen and only A/B trainable")
        optimized = [id(parameter) for group in self.optimizer.param_groups for parameter in group["params"]]
        if len(optimized) != len(adapters) or set(optimized) != adapters:
            raise ValueError("LoRA checkpoint optimizer must contain each adapter exactly once and no base")
        if any(id(parameter) not in adapters for parameter in self.optimizer.state):
            raise ValueError("LoRA checkpoint optimizer state contains a non-adapter parameter")
        keys = {name for name, parameter in parameters.items() if id(parameter) in adapters}
        for prefix, module in self.model.named_modules():
            keys.update((prefix + "." if prefix else "") + name
                        for name, buffer in module._buffers.items()
                        if buffer is not None and name not in module._non_persistent_buffers_set)
        if logical:
            keys = {self.state_adapter.names.get(name, name) for name in keys}
        return keys

    def checkpoint_configuration(self):
        optimizer = self.optimizer
        name = ("sgd" if isinstance(optimizer, torch.optim.SGD) else
                "adamw" if isinstance(optimizer, torch.optim.AdamW) else
                type(optimizer).__module__ + "." + type(optimizer).__qualname__)
        configuration = {"model": self.model_config, "training": self.training_config, "optimizer": name}
        if self.checkpoint_state == "lora":
            adapters = {}
            for name, module in self.model.named_modules():
                if isinstance(module, LoRALinear):
                    parameter_name = (name + "." if name else "") + "lora_a"
                    logical_name = self.state_adapter.names[parameter_name].rpartition(".")[0]
                    adapters[logical_name] = asdict(module.config)
            configuration["checkpoint_state"] = {"kind": "lora", "version": 1,
                                                   "model_keys": sorted(self._lora_state_keys(logical=True)),
                                                   "adapters": adapters}
        return configuration

    def state_dict(self) -> dict[str, Any]:
        if self.state_adapter is not None:
            if self.checkpoint_state == "lora":
                expected = self._lora_state_keys()
                model_state = get_model_state_dict(self.model, options=StateDictOptions(ignore_frozen_params=True))
                if set(model_state) != expected:
                    raise ValueError("LoRA checkpoint model keys differ from adapters and persistent buffers")
            else:
                model_state = get_model_state_dict(self.model)
            optimizer_state = self.state_adapter.optimizer_state(self.optimizer, self._optimizer_schema)
            optimizer_schema = self.state_adapter.optimizer_schema(optimizer_state)
            model_state, optimizer_state = self.state_adapter.encode(model_state, optimizer_state)
        else:
            model_state, optimizer_state = get_state_dict(self.model, self.optimizer)
        rank, world_size = self._distributed_position()
        state = {
            "model": model_state,
            "optimizer": optimizer_state,
            "model_config": deepcopy(self.model_config),
            "data_generator": {
                "world_size": world_size,
                f"rank-{rank}": self.data_generator.get_state(),
            },
            "step": self.step,
        }
        if self.state_adapter is not None:
            state["layout"] = self.state_adapter.signature()
            state["optimizer_schema"] = optimizer_schema
            state["configuration"] = json.dumps(self.checkpoint_configuration(), sort_keys=True)
        if self.training_config is not None:
            state["training_config"] = deepcopy(self.training_config)
        if self.training_config is not None or self.state_adapter is not None:
            cuda_device = next((p.device for p in self.model.parameters() if p.is_cuda), None)
            state["rng"] = {f"rank-{rank}": {
                "cpu": torch.get_rng_state(),
                "cuda": (torch.cuda.get_rng_state(cuda_device) if cuda_device is not None
                         else torch.empty(0, dtype=torch.uint8)),
            }}
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        checkpoint_config = state_dict["model_config"]
        if checkpoint_config != self.model_config:
            raise ValueError("checkpoint model config does not match the current model")
        if state_dict.get("training_config") != self.training_config:
            raise ValueError("checkpoint training config does not match the current model")
        if (self.checkpoint_state == "lora"
                and state_dict.get("configuration") != json.dumps(self.checkpoint_configuration(), sort_keys=True)):
            raise ValueError("LoRA checkpoint adapter/base configuration is incompatible")

        rank, world_size = self._distributed_position()
        generator_state = state_dict["data_generator"]
        if generator_state["world_size"] != world_size:
            raise ValueError(
                "cannot restore the data generator after changing world size"
            )

        model_state, optimizer_state = state_dict["model"], state_dict["optimizer"]
        if self.state_adapter is not None:
            self.state_adapter.validate(state_dict.get("layout"))
            if self.checkpoint_state == "lora" and set(model_state) != self._lora_state_keys(logical=True):
                raise ValueError("LoRA checkpoint model key set is incompatible")
            model_state, optimizer_state = self.state_adapter.decode(model_state, optimizer_state)
            if self.checkpoint_state == "lora":
                result = set_model_state_dict(self.model, model_state, options=StateDictOptions(strict=False))
                expected_missing = {name for name, parameter in self.model.named_parameters()
                                    if not parameter.requires_grad}
                if set(result.missing_keys) != expected_missing or result.unexpected_keys:
                    raise ValueError("LoRA checkpoint partial load omitted more than the frozen base")
            else:
                set_model_state_dict(self.model, model_state)
            self.optimizer.load_state_dict(optimizer_state)
        else:
            set_state_dict(self.model, self.optimizer,
                           model_state_dict=model_state, optim_state_dict=optimizer_state)
        self.data_generator.set_state(generator_state[f"rank-{rank}"])
        rng = state_dict.get("rng", {}).get(f"rank-{rank}")
        if rng is not None:
            torch.set_rng_state(rng["cpu"].cpu())
            if rng["cuda"].numel():
                cuda_device = next((p.device for p in self.model.parameters() if p.is_cuda), None)
                if cuda_device is not None:
                    torch.cuda.set_rng_state(rng["cuda"].cpu(), cuda_device)
        self.step = int(state_dict["step"])

    @staticmethod
    def _distributed_position() -> tuple[int, int]:
        if not dist.is_initialized():
            return 0, 1
        return dist.get_rank(), dist.get_world_size()


class CheckpointManager:
    """Save and restore step-addressed distributed checkpoints."""

    _STEP_PATTERN = re.compile(r"step-(0|[1-9]\d*)")

    def __init__(self, folder: str | Path, state: TrainingState) -> None:
        self.folder = Path(folder)
        self.state = state

    @torch.no_grad()
    def save(self, step: int) -> Path:
        if step < 0:
            raise ValueError("checkpoint step must be non-negative")

        self.state.step = step
        checkpoint_path = self._path_for_step(step)
        options = {}
        if getattr(self.state, "state_adapter", None) is not None:
            from .sharded_state import ShardSavePlanner
            options["planner"] = ShardSavePlanner()
            synchronize = getattr(self.state.model, "synchronize_engram", None)
            if synchronize is not None:
                synchronize()
        dcp.save({"train": self.state}, checkpoint_id=str(checkpoint_path), **options)
        return checkpoint_path

    @torch.no_grad()
    def load(self, step: int | None = None) -> int:
        load_step = self.latest_step() if step is None else step
        if load_step is None:
            raise FileNotFoundError(f"no checkpoint found in {self.folder}")
        if load_step < 0:
            raise ValueError("checkpoint step must be non-negative")

        checkpoint_path = self._path_for_step(load_step)
        if not (checkpoint_path / ".metadata").is_file():
            raise FileNotFoundError(
                f"checkpoint is incomplete or missing: {checkpoint_path}"
            )

        options = {}
        if getattr(self.state, "state_adapter", None) is not None:
            from .sharded_state import ShardLoadPlanner
            synchronize = getattr(self.state.model, "synchronize_engram", None)
            if synchronize is not None:
                synchronize()
            metadata = dcp.FileSystemReader(str(checkpoint_path)).read_metadata()
            if "train.layout.version" not in metadata.state_dict_metadata:
                raise ValueError("checkpoint lacks global parameter layout metadata; unsafe legacy shards cannot be restored")
            rank = f"rank-{self.state.state_adapter.rank}"
            if f"train.optimizer_schema.{rank}" not in metadata.state_dict_metadata:
                raise ValueError("checkpoint lacks optimizer state schema; unsafe legacy state cannot be restored")
            if "train.configuration" not in metadata.state_dict_metadata:
                raise ValueError("checkpoint lacks configuration metadata; unsafe legacy state cannot be restored")
            preflight = {"train": {"layout": self.state.state_adapter.signature(),
                                    "optimizer_schema": {rank: ""}, "configuration": ""}}
            dcp.load(preflight, checkpoint_id=str(checkpoint_path))
            self.state.state_adapter.validate(preflight["train"]["layout"])
            if preflight["train"]["configuration"] != json.dumps(self.state.checkpoint_configuration(), sort_keys=True):
                raise ValueError("checkpoint model/training/optimizer configuration is incompatible")
            if self.state.checkpoint_state == "lora":
                expected = {"train.model." + name for name in self.state._lora_state_keys(logical=True)}
                actual = {name for name in metadata.state_dict_metadata if name.startswith("train.model.")}
                if actual != expected:
                    raise ValueError("LoRA checkpoint model key set is incompatible")
            self.state._optimizer_schema = json.loads(preflight["train"]["optimizer_schema"][rank])
            if self.state.checkpoint_state == "lora":
                adapters = {self.state.state_adapter.names[name] for name, parameter
                            in self.state.model.named_parameters() if parameter.requires_grad}
                if set(self.state._optimizer_schema) - adapters:
                    self.state._optimizer_schema = None
                    raise ValueError("LoRA checkpoint optimizer schema contains non-adapter state")
            options["planner"] = ShardLoadPlanner()
        try:
            dcp.load({"train": self.state}, checkpoint_id=str(checkpoint_path), **options)
        finally:
            self.state._optimizer_schema = None
        return self.state.step

    def latest_step(self) -> int | None:
        if not self.folder.is_dir():
            return None

        steps = []
        for path in self.folder.iterdir():
            match = self._STEP_PATTERN.fullmatch(path.name)
            if match is not None and (path / ".metadata").is_file():
                steps.append(int(match.group(1)))
        return max(steps, default=None)

    def _path_for_step(self, step: int) -> Path:
        return self.folder / f"step-{step}"


__all__ = ["CheckpointManager", "TrainingState"]
