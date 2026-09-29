"""Execution policy, separate from the pretrained model configuration."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EngramPolicy:
    device: str = "cuda"
    prefetch: str = "sync"
    deduplicate: bool = False
    staging_bytes: int = 256 * 1024**2

    def __post_init__(self):
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("Engram device must be cpu or cuda")
        if self.prefetch not in {"sync", "async"}:
            raise ValueError("Engram prefetch must be sync or async")
        if self.prefetch == "async" and self.device != "cpu":
            raise ValueError("asynchronous Engram prefetch requires CPU storage")
        if self.staging_bytes < 4096:
            raise ValueError("Engram staging_bytes must be at least 4096")
