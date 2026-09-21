"""Checkpoint state and storage management for training."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch import nn
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.stateful import Stateful


class TrainingState(Stateful):
    """The mutable state required to resume a training run."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        model_config: dict[str, Any],
        data_generator: torch.Generator,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.model_config = model_config
        self.data_generator = data_generator
        self.step = 0

    def state_dict(self) -> dict[str, Any]:
        model_state, optimizer_state = get_state_dict(self.model, self.optimizer)
        rank, world_size = self._distributed_position()
        return {
            "model": model_state,
            "optimizer": optimizer_state,
            "model_config": self.model_config,
            "data_generator": {
                "world_size": world_size,
                f"rank-{rank}": self.data_generator.get_state(),
            },
            "step": self.step,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        checkpoint_config = state_dict["model_config"]
        if checkpoint_config != self.model_config:
            raise ValueError("checkpoint model config does not match the current model")

        rank, world_size = self._distributed_position()
        generator_state = state_dict["data_generator"]
        if generator_state["world_size"] != world_size:
            raise ValueError(
                "cannot restore the data generator after changing world size"
            )

        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optimizer"],
        )
        self.data_generator.set_state(generator_state[f"rank-{rank}"])
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

        dcp.load({"train": self.state}, checkpoint_id=str(checkpoint_path))
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
