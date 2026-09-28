"""Checkpoint state and storage management for training."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from copy import copy, deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch import nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions, get_model_state_dict, get_optimizer_state_dict,
    set_model_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful

from .checkpoint_layout import checkpoint_parameter_names, parameter_layout


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
        checkpoint_mode: str = "full",
        base_model_identity: dict[str, Any] | None = None,
    ) -> None:
        if checkpoint_mode not in ("full", "trainable"):
            raise ValueError("checkpoint_mode must be full or trainable")
        if checkpoint_mode == "trainable" and not base_model_identity:
            raise ValueError("trainable checkpoints require a base_model_identity")
        self.model = model
        self.optimizer = optimizer
        self.model_config = deepcopy(model_config)
        self.data_generator = data_generator
        self.training_config = deepcopy(training_config)
        self.checkpoint_mode = checkpoint_mode
        self.base_model_identity = deepcopy(base_model_identity)
        self.parameter_names = checkpoint_parameter_names(model)
        self.options = StateDictOptions(
            ignore_frozen_params=checkpoint_mode == "trainable",
            strict=checkpoint_mode == "full",
        )
        self.step = 0

    def metadata(self) -> dict[str, Any]:
        rank, world_size = self._distributed_position()
        decoder = getattr(self.model, "model", self.model)
        return {
            "format_version": 2, "rank": rank, "world_size": world_size,
            "model_config": deepcopy(self.model_config),
            "training_config": deepcopy(self.training_config),
            "checkpoint_mode": self.checkpoint_mode,
            "base_model_identity": deepcopy(self.base_model_identity),
            "layer_ids": getattr(decoder, "layer_ids", None),
            "parameter_names": self.parameter_names,
            "parameter_layout": parameter_layout(self.model),
            "module_metadata": {name: module.checkpoint_metadata()
                                for name, module in self.model.named_modules()
                                if hasattr(module, "checkpoint_metadata")},
            "optimizer_type": type(self.optimizer).__module__ + "." + type(self.optimizer).__qualname__,
            "optimizer_parameters": self._optimizer_parameters(),
            "optimizer_options": [{key: value for key, value in group.items()
                                   if key not in ("params", "lr", "initial_lr")}
                                  for group in self.optimizer.param_groups],
        }

    def _optimizer_parameters(self) -> list[list[str]]:
        names = {id(parameter): name for name, parameter in self.model.named_parameters()}
        return [[names[id(parameter)] for parameter in group["params"]]
                for group in self.optimizer.param_groups]

    def validate_metadata(self, serialized: str) -> None:
        expected = json.loads(json.dumps(self.metadata(), sort_keys=True))
        actual = json.loads(serialized)
        for key in list(expected) + [key for key in actual if key not in expected]:
            if actual.get(key) != expected.get(key):
                raise ValueError(f"checkpoint {key.replace('_', ' ')} does not match the current model")

    def state_dict(self) -> dict[str, Any]:
        model_state = get_model_state_dict(self.model, options=self.options)
        # Native FSDP2 optimizer tensors already carry DTensor placements. Exporting
        # directly also avoids creating Adam moments for parameters never updated.
        optimizer_state = self.optimizer.state_dict()
        groups = self._optimizer_parameters()
        names = {index: name for group, group_names in zip(optimizer_state["param_groups"], groups)
                 for index, name in zip(group["params"], group_names)}
        optimizer_state = {
            "state": {names[index]: value for index, value in optimizer_state["state"].items()},
            "param_groups": [{**group, "params": group_names}
                             for group, group_names in zip(optimizer_state["param_groups"], groups)],
        }
        return self._pack_state(model_state, optimizer_state)

    def load_template(self, saved_optimizer_names: list[str]) -> dict[str, Any]:
        """Allocate moments only for parameters that had state in the checkpoint."""
        template = copy(self.optimizer)
        template.state = defaultdict(dict)
        names = {id(p): self.parameter_names.get(name, name)
                 for name, p in self.model.named_parameters()}
        selected = set(saved_optimizer_names)
        template.param_groups = [
            {**group, "params": [p for p in group["params"] if names[id(p)] in selected]}
            for group in self.optimizer.param_groups
        ]
        parameters = [p for group in template.param_groups for p in group["params"]]
        gradients = [p.grad for p in parameters]
        try:
            for parameter in parameters:
                parameter.grad = None
            optimizer_state = get_optimizer_state_dict(self.model, template, options=self.options)
        finally:
            for parameter, gradient in zip(parameters, gradients):
                parameter.grad = gradient
        optimizer_state["param_groups"] = [
            {**group, "params": names}
            for group, names in zip(self.optimizer.param_groups, self._optimizer_parameters())
        ]
        state = self._pack_state(get_model_state_dict(self.model, options=self.options), optimizer_state)
        if set(state["optimizer"]["state"]) != selected:
            raise ValueError("checkpoint optimizer state does not match the current optimizer")
        return state

    def _pack_state(self, model_state: dict, optimizer_state: dict) -> dict[str, Any]:
        rank, world_size = self._distributed_position()
        rename = lambda state: {self.parameter_names.get(name, name): value
                                for name, value in state.items()}
        state = {
            "metadata": {f"rank-{rank}": json.dumps(self.metadata(), sort_keys=True)},
            "optimizer_state_names": {f"rank-{rank}": json.dumps(
                [self.parameter_names.get(name, name) for name in optimizer_state["state"]]
            )},
            "model": rename(model_state),
            "optimizer": {
                "state": rename(optimizer_state["state"]),
                # Parameter groups refer to rank-local expert names.
                "param_groups": {f"rank-{rank}": deepcopy(optimizer_state["param_groups"])},
            },
            "data_generator": {
                "world_size": world_size,
                f"rank-{rank}": self.data_generator.get_state(),
            },
            "step": self.step,
        }
        cuda_device = next((p.device for p in self.model.parameters() if p.is_cuda), None)
        state["rng"] = {f"rank-{rank}": {
            "cpu": torch.get_rng_state(),
            "cuda": (torch.cuda.get_rng_state(cuda_device) if cuda_device is not None
                     else torch.empty(0, dtype=torch.uint8)),
        }}
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        rank, world_size = self._distributed_position()
        self.validate_metadata(state_dict["metadata"][f"rank-{rank}"])
        generator_state = state_dict["data_generator"]
        if generator_state["world_size"] != world_size:
            raise ValueError(
                "cannot restore the data generator after changing world size"
            )

        local_names = {stored: local for local, stored in self.parameter_names.items()}
        restore = lambda state: {local_names.get(name, name): value for name, value in state.items()}
        optimizer_state = state_dict["optimizer"]
        set_model_state_dict(self.model, restore(state_dict["model"]), options=self.options)
        # Group order is checked in metadata; PyTorch maps these saved FQNs to
        # live parameters. Unused parameters legitimately have no optimizer state.
        self.optimizer.load_state_dict({
            "state": restore(optimizer_state["state"]),
            "param_groups": optimizer_state["param_groups"][f"rank-{rank}"],
        })
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
        dcp.save({"train": self.state}, checkpoint_id=str(checkpoint_path))
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

        optimizer_names = self._validate_checkpoint(checkpoint_path)
        state = self.state.load_template(optimizer_names)
        dcp.load({"train": state}, checkpoint_id=str(checkpoint_path))
        self.state.load_state_dict(state)
        return self.state.step

    @staticmethod
    def _raise_collectively(error: str | None) -> None:
        errors = [error]
        if dist.is_initialized():
            errors = [None] * dist.get_world_size()
            dist.all_gather_object(errors, error)
        if any(errors):
            raise ValueError(next(message for message in errors if message))

    def _validate_checkpoint(self, path: Path) -> list[str]:
        # DCP loads tensors in place. Validate metadata before exposing live tensors.
        rank, _ = self.state._distributed_position()
        key = f"rank-{rank}"
        metadata = dcp.FileSystemReader(path).read_metadata()
        error = None
        if f"train.metadata.{key}" not in metadata.state_dict_metadata:
            error = "checkpoint lacks version-2 rank metadata; legacy training checkpoints require migration"
        self._raise_collectively(error)
        probe = {"train": {"metadata": {key: ""}, "optimizer_state_names": {key: ""}}}
        dcp.load(probe, checkpoint_id=str(path))
        try:
            self.state.validate_metadata(probe["train"]["metadata"][key])
        except (ValueError, TypeError, KeyError) as exc:
            error = str(exc)
        self._raise_collectively(error)
        return json.loads(probe["train"]["optimizer_state_names"][key])

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
