"""Exact raw cache of Qwen's synthetic Engram initialization; no model weights/GPU.

The cache preserves the original CPU BF16 normal_ calls, including their 8192-row
boundaries and per-owner seeds. It is optional and never stores trained weights.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import shutil
import sys
from types import SimpleNamespace

import torch

from dsv41_train.models.qwen.config import Qwen3MoeConfig
from dsv41_train.models.qwen.engram import QwenEngramConfig
from dsv41_train.models.dsv4.engram.hash import NgramHash

ROOT = Path(__file__).resolve().parents[1]
CHUNK_ROWS = 8192
FORMAT = 'qwen-dummy-initial-bf16-raw-v1'


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_spec(config_path, config, world_size, ep_size):
    _require(sys.byteorder == 'little', 'raw BF16 initialization requires little endian')
    _require(type(world_size) is int and type(ep_size) is int and world_size > 0
             and ep_size > 0 and world_size % ep_size == 0, 'invalid world/EP topology')
    _require(config.seed >= 0, 'seed must be nonnegative')
    model = Qwen3MoeConfig.from_json(config_path)
    _require(max(config.layer_ids) < model.num_hidden_layers and model.num_experts % ep_size == 0,
             'Engram layer/EP outside backbone')
    _require(math.isfinite(model.initializer_range) and model.initializer_range > 0, 'invalid initializer std')
    columns = (config.max_ngram - 1) * config.heads
    options = SimpleNamespace(vocab_size=model.vocab_size, engram_compressed_vocab_size=model.vocab_size,
        engram_pad_id=0, engram_max_ngram_size=config.max_ngram, engram_n_heads=config.heads,
        engram_vocab_size=config.vocab_per_head, engram_layer_ids=list(config.layer_ids),
        engram_num_embeddings=[columns * (config.vocab_per_head + 10000)] * len(config.layer_ids))
    with torch.device('cpu'):
        hashing = NgramHash(options)
    counts = (hashing.offsets[:, -1] + hashing.primes.flatten(1)[:, -1]).tolist()
    tables = []
    for layer_id, count in zip(config.layer_ids, counts):
        _require(count >= world_size, 'fewer table rows than owners')
        quotient, remainder = divmod(count, world_size)
        for rank in range(world_size):
            start = rank * quotient + min(rank, remainder)
            stop = start + quotient + int(rank < remainder)
            seed = config.seed + layer_id * 100003 + start
            _require(seed < 2**64, 'derived owner seed outside torch uint64 range')
            tables.append(dict(layer_id=layer_id, rank=rank, global_rows=count,
                row_start=start, row_stop=stop, shape=[stop-start, config.head_dim], seed=seed,
                file=f'layer-{layer_id}-rank-{rank}.bf16.raw', bytes=(stop-start)*config.head_dim*2))
    sources = ['dsv41_train/models/qwen/engram.py', 'dsv41_train/models/dsv4/engram/hash.py',
               'dsv41_train/models/dsv4/engram/routing.py', 'scripts/qwen_dummy_init_cache.py']
    metadata = dict(format=FORMAT, dtype='torch.bfloat16', byteorder='little', chunk_rows=CHUNK_ROWS,
        world_size=world_size, ep_size=ep_size, engram_config=json.loads(json.dumps(asdict(config))),
        initializer_range=model.initializer_range, config_sha256=_sha(config_path),
        torch_version=str(torch.__version__), cpu_architecture=platform.machine(), cpu_threads=1,
        source_sha256={name: _sha(ROOT / name) for name in sources})
    return dict(metadata=metadata, tables=tables)


def _generate_owner(output, spec, rank):
    _require(not torch.cuda.is_initialized(), 'CPU preparation must never initialize CUDA')
    torch.set_num_threads(1)
    records = []
    for item in spec['tables']:
        if item['rank'] != rank:
            continue
        path = Path(output) / item['file']
        temporary = path.with_suffix(path.suffix + '.incomplete')
        _require(not path.exists(), 'refusing existing raw table')
        generator = torch.Generator(device='cpu').manual_seed(item['seed'])
        digest = hashlib.sha256()
        with temporary.open('xb') as stream:
            for start in range(0, item['shape'][0], CHUNK_ROWS):
                rows = min(CHUNK_ROWS, item['shape'][0]-start)
                values = torch.empty((rows, item['shape'][1]), dtype=torch.bfloat16, device='cpu').normal_(
                    std=spec['metadata']['initializer_range'], generator=generator)
                raw = values.view(torch.uint8).numpy().tobytes()
                digest.update(raw)
                stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        _require(temporary.stat().st_size == item['bytes'], 'generated table byte count differs')
        # Both names remain exclusive; interrupted generations are never reused.
        os.link(temporary, path)
        temporary.unlink()
        records.append({**item, 'sha256': digest.hexdigest()})
    _require(not torch.cuda.is_initialized(), 'CPU preparation initialized CUDA')
    return records


def prepare_cache(model_path, config, output_dir, *, world_size=2, ep_size=2, processes=2,
                  reserve_bytes=2**30):
    _require(not torch.cuda.is_initialized(), 'prepare in a fresh CPU-only process')
    _require(type(processes) is int and processes > 0 and type(reserve_bytes) is int and reserve_bytes >= 0,
             'invalid process count or disk reserve')
    spec = build_spec(Path(model_path) / 'config.json', config, world_size, ep_size)
    output = Path(output_dir).absolute()
    _require(not output.exists(), 'refusing existing initialization directory, including incomplete caches')
    output.parent.mkdir(parents=True, exist_ok=True)
    total = sum(item['bytes'] for item in spec['tables'])
    _require(shutil.disk_usage(output.parent).free >= total + reserve_bytes, 'insufficient free disk for raw initialization')
    output.mkdir(exist_ok=False)
    if processes == 1:
        records = [record for rank in range(world_size) for record in _generate_owner(output, spec, rank)]
    else:
        with ProcessPoolExecutor(max_workers=min(processes, world_size),
                                 mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = [pool.submit(_generate_owner, str(output), spec, rank) for rank in range(world_size)]
            records = [record for future in futures for record in future.result()]
    records.sort(key=lambda item: (config.layer_ids.index(item['layer_id']), item['rank']))
    manifest = dict(schema=FORMAT, complete=True, metadata=spec['metadata'], tables=records,
                    total_bytes=total, cuda_initialized=False)
    temporary = output / 'manifest.json.incomplete'
    with temporary.open('x') as stream:
        json.dump(manifest, stream, indent=2); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.link(temporary, output / 'manifest.json')
    temporary.unlink()
    return output / 'manifest.json'


class CachedQwenTableInitializer:
    """Optional attach_engram callback. Every byte copied is hashed before return."""
    def __init__(self, manifest_path, model_path, config, *, world_size, ep_size):
        self.path = Path(manifest_path).absolute()
        _require(self.path.is_file() and not self.path.is_symlink(), 'complete regular manifest required')
        raw = self.path.read_bytes()
        self.manifest_sha256 = hashlib.sha256(raw).hexdigest()
        self.manifest = json.loads(raw)
        spec = build_spec(Path(model_path) / 'config.json', config, world_size, ep_size)
        _require(self.manifest.get('schema') == FORMAT and self.manifest.get('complete') is True
                 and self.manifest.get('cuda_initialized') is False, 'incomplete/wrong initialization manifest')
        _require(self.manifest['metadata'] == spec['metadata'], 'initialization source/config/topology identity differs')
        records = self.manifest['tables']
        _require([{k: value for k, value in item.items() if k != 'sha256'} for item in records] == spec['tables'],
                 'initialization table shape/owner/seed layout differs')
        _require(self.manifest['total_bytes'] == sum(item['bytes'] for item in records), 'raw total byte count differs')
        for item in records:
            _require(isinstance(item.get('sha256'), str) and len(item['sha256']) == 64
                     and all(c in '0123456789abcdef' for c in item['sha256']), 'invalid raw SHA256')
            path = self.path.parent / item['file']
            _require(path.is_file() and not path.is_symlink() and path.stat().st_size == item['bytes'],
                     'raw table missing, incomplete or wrong size')
        self.loaded = []
        self.records = []

    @property
    def identity(self):
        return dict(method='raw-cache-v1', manifest_sha256=self.manifest_sha256)

    @torch.no_grad()
    def __call__(self, table, *, layer_id):
        records = [item for item in self.manifest['tables'] if item['layer_id'] == layer_id and item['rank'] == table.rank]
        _require(len(records) == 1, 'missing/duplicate raw owner record')
        item = records[0]
        _require((layer_id, table.rank) not in self.loaded, 'table initializer called twice for one owner')
        _require(table.size == self.manifest['metadata']['world_size'] and table.global_num_embeddings == item['global_rows']
                 and table.row_start == item['row_start'] and table.row_stop == item['row_stop']
                 and list(table.weight.shape) == item['shape'] and table.weight.dtype == torch.bfloat16,
                 'destination table does not match cached owner layout/dtype')
        digest = hashlib.sha256()
        read_bytes = 0
        path = self.path.parent / item['file']
        _require(not path.is_symlink(), 'raw table changed into symlink')
        with path.open('rb') as stream:
            for start in range(0, item['shape'][0], CHUNK_ROWS):
                rows = min(CHUNK_ROWS, item['shape'][0]-start)
                raw = bytearray(stream.read(rows * item['shape'][1] * 2))
                _require(len(raw) == rows * item['shape'][1] * 2, 'truncated raw initialization')
                digest.update(raw)
                read_bytes += len(raw)
                values = torch.frombuffer(raw, dtype=torch.bfloat16).view(rows, item['shape'][1])
                table.weight[start:start+rows].copy_(values)
            _require(stream.read(1) == b'', 'raw initialization has trailing bytes')
        _require(digest.hexdigest() == item['sha256'], 'raw initialization SHA256 mismatch; model must not run')
        self.loaded.append((layer_id, table.rank))
        self.records.append(dict(layer_id=layer_id, rank=table.rank, row_start=table.row_start,
            row_stop=table.row_stop, shape=list(table.weight.shape), dtype=str(table.weight.dtype),
            target_device=str(table.weight.device), raw_file=item['file'], actual_bytes=read_bytes,
            expected_bytes=item['bytes'], actual_sha256=digest.hexdigest(), expected_sha256=item['sha256'],
            sha256_checked=True, manifest_sha256=self.manifest_sha256,
            preparation_cuda_initialized=self.manifest['cuda_initialized']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--world-size', type=int, default=2)
    parser.add_argument('--ep-size', type=int, default=2)
    parser.add_argument('--processes', type=int, default=2)
    parser.add_argument('--layers', type=int, nargs='+', default=[1])
    parser.add_argument('--vocab-per-head', type=int, default=16000000)
    parser.add_argument('--head-dim', type=int, default=256)
    parser.add_argument('--seed', type=int, default=20261013)
    args = parser.parse_args()
    config = QwenEngramConfig(tuple(args.layers), args.vocab_per_head, args.head_dim, seed=args.seed)
    path = prepare_cache(args.model_path, config, args.output_dir, world_size=args.world_size,
                         ep_size=args.ep_size, processes=args.processes)
    print(json.dumps(dict(passed=True, manifest=str(path), sha256=_sha(path), cuda_initialized=torch.cuda.is_initialized())), flush=True)


if __name__ == '__main__':
    main()
