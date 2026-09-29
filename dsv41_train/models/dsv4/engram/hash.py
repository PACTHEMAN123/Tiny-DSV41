"""Token-only addressing for Engram tables."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from ..config import DeepSeekV41Config


def _is_prime(value: int) -> bool:
    if value < 2 or value % 2 == 0:
        return value == 2
    divisor = 3
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 2
    return True


def _next_prime(value: int, used: set[int]) -> int:
    value += 1
    while value in used or not _is_prime(value):
        value += 1
    return value


def build_compressed_token_map(tokenizer_file: str | Path) -> tuple[torch.Tensor, int]:
    try:
        from tokenizers import Regex, Tokenizer, normalizers
    except ImportError as error:
        raise RuntimeError(
            "loading Engram checkpoint weights requires the tokenizers package"
        ) from error

    tokenizer = Tokenizer.from_file(str(tokenizer_file))
    sentinel = "\ue000"
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    compressed: dict[str, int] = {}
    mapping = []
    for token_id in range(tokenizer.get_vocab_size(with_added_tokens=True)):
        text = tokenizer.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            key = tokenizer.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized or text
        new_id = compressed.get(key)
        if new_id is None:
            new_id = len(compressed)
            compressed[key] = new_id
        mapping.append(new_id)
    return torch.tensor(mapping, dtype=torch.long), len(compressed)


def _build_hash_multipliers(
    layer_ids: list[int], max_ngram: int, compressed_vocab_size: int
) -> torch.Tensor:
    """Rebuild the fixed hash constants used to address pretrained Engram rows."""
    bound = max(1, (np.iinfo(np.int64).max // compressed_vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        generator = np.random.default_rng(10007 * layer_id)
        values = generator.integers(0, bound, size=max_ngram, dtype=np.int64)
        rows.append(torch.from_numpy(values * 2 + 1))
    return torch.stack(rows)


class NgramHash(nn.Module):
    def __init__(
        self,
        config: DeepSeekV41Config,
        layer_ids: list[int] | None = None,
        token_map: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.max_ngram = config.engram_max_ngram_size
        if token_map is None:
            token_map = torch.arange(config.vocab_size).remainder(
                config.engram_compressed_vocab_size
            )
        if token_map.ndim != 1 or token_map.shape[0] != config.vocab_size:
            raise ValueError("Engram token map must contain one entry per vocabulary id")
        if not token_map.is_meta and token_map.numel() and (
            token_map.min() < 0
            or token_map.max() >= config.engram_compressed_vocab_size
        ):
            raise ValueError("Engram token map contains an out-of-range compressed id")
        self.pad_id = (
            config.engram_pad_id % config.engram_compressed_vocab_size
            if token_map.is_meta
            else int(token_map[config.engram_pad_id])
        )
        layer_ids = list(config.engram_layer_ids if layer_ids is None else layer_ids)
        if not set(layer_ids).issubset(config.engram_layer_ids):
            raise ValueError("Engram hash layers must be configured Engram layers")
        selected = set(layer_ids)
        layer_primes, layer_offsets, used = [], [], set()
        for layer_id, table_size in zip(
            config.engram_layer_ids, config.engram_num_embeddings
        ):
            primes, offsets, offset = [], [], 0
            current = config.engram_vocab_size - 1
            for _ in range((self.max_ngram - 1) * config.engram_n_heads):
                current = _next_prime(current, used)
                used.add(current)
                primes.append(current)
                offsets.append(offset)
                offset += current
            if offset > table_size:
                raise ValueError(f"engram table needs {offset} rows, got {table_size}")
            if layer_id in selected:
                layer_primes.append(primes)
                layer_offsets.append(offsets)

        primes = torch.tensor(layer_primes).view(
            len(layer_ids), self.max_ngram - 1, config.engram_n_heads
        )
        offsets = torch.tensor(layer_offsets)
        multipliers = _build_hash_multipliers(
            layer_ids, self.max_ngram, config.engram_compressed_vocab_size
        )
        self.compressed_vocab_size = config.engram_compressed_vocab_size
        self.register_buffer("token_map", token_map.to(torch.long), persistent=False)
        self.register_buffer("primes", primes, persistent=False)
        self.register_buffer("offsets", offsets, persistent=False)
        self.register_buffer("multipliers", multipliers, persistent=False)

    def forward(self, input_ids: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        tokens = self.token_map[input_ids]
        dead = -1
        tokens = tokens.masked_fill(~token_mask, dead)
        context = self.max_ngram - 1
        history = F.pad(tokens, (context, 0), value=dead)
        shifted, blocked = [], torch.zeros_like(tokens, dtype=torch.bool)
        for distance in range(self.max_ngram):
            source = history[:, context - distance : context - distance + tokens.shape[1]]
            blocked |= source == dead
            shifted.append(torch.where(blocked, self.pad_id, source))
        windows = torch.stack(shifted, dim=-1)

        products = windows.unsqueeze(2) * self.multipliers
        rolling, hashes = products[..., 0], []
        for index in range(1, self.max_ngram):
            rolling = torch.bitwise_xor(rolling, products[..., index])
            hashes.append(rolling.unsqueeze(-1) % self.primes[:, index - 1])
        return torch.cat(hashes, dim=-1) + self.offsets
