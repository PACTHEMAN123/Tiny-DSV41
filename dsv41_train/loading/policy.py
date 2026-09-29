"""Loading policies and the model-specific materialization contract."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class LoadPolicy:
    mode: str = "layerwise"
    chunk_rows: int = 131072
    staging_bytes: int = 256 * 1024**2
    pipeline: bool = False

    def __post_init__(self):
        if self.mode not in {"layerwise", "direct_shard"}:
            raise ValueError("load mode must be layerwise or direct_shard")
        if self.chunk_rows < 1 or self.staging_bytes < 4096:
            raise ValueError("loading requires positive chunk_rows and at least 4096 staging bytes")


class ParameterMaterializer(Protocol):
    def prepare(self, model, *, dtype, device, direct): ...
    def roots_loaded(self, model): ...
    def layer_loaded(self, layer): ...
    def finish(self, model): ...
