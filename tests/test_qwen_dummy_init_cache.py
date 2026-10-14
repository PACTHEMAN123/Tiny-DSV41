"""Exact CPU initializer replay, corruption and interrupted-cache regressions."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from dsv41_train.models.qwen import Qwen3MoeForCausalLM
from dsv41_train.models.qwen.engram import QwenEngramConfig, attach_engram
from dsv41_train.models.dsv4.engram import EngramPolicy
from scripts.qwen_dummy_init_cache import CachedQwenTableInitializer, prepare_cache
from qwen_helpers import tiny_config


class MeshFixture:
    ndim = 1
    def __init__(self, rank, size): self._rank, self._size = rank, size
    def size(self): return self._size
    def get_local_rank(self): return self._rank
    def get_group(self): return None


class ExactInitCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.model_config = tiny_config()
        self.model_path = self.root / 'model'; self.model_path.mkdir()
        (self.model_path / 'config.json').write_text(json.dumps(self.model_config.to_dict()))
        # Every owner crosses the original 8192-row chunk boundary and a tail.
        self.config = QwenEngramConfig((0, 1), 5003, 4, heads=2, max_ngram=3, seed=217)

    def prepare(self, *, name='cache', config=None, world=2, processes=1):
        return prepare_cache(self.model_path, config or self.config, self.root/name,
                             world_size=world, ep_size=world, processes=processes, reserve_bytes=0)

    def loader(self, path, *, config=None, world=2):
        return CachedQwenTableInitializer(path, self.model_path, config or self.config,
                                         world_size=world, ep_size=world)

    def test_full_original_attach_equals_cached_every_parameter_for_both_owners(self):
        before = torch.get_rng_state().clone()
        with patch('torch.cuda._lazy_init', side_effect=AssertionError('CUDA initialized')):
            path = self.prepare()
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        initializer = self.loader(path)
        self.assertTrue(all(item['shape'][0] > 8192 and item['shape'][0] % 8192 for item in initializer.manifest['tables']))
        base = Qwen3MoeForCausalLM(self.model_config)
        for rank in range(2):
            original, cached = deepcopy(base), deepcopy(base)
            before = torch.get_rng_state().clone()
            with patch('torch.cuda.manual_seed_all', side_effect=AssertionError('CUDA seeded')):
                attach_engram(original, self.config, EngramPolicy('cpu'), MeshFixture(rank, 2))
                attach_engram(cached, self.config, EngramPolicy('cpu'), MeshFixture(rank, 2),
                              table_initializer=initializer)
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            for name, weight in original.named_parameters():
                self.assertTrue(torch.equal(weight.view(torch.uint8),
                                            dict(cached.named_parameters())[name].view(torch.uint8)), name)
            for model in (original, cached):
                for table in model.model.engram_tables.values(): table.close()
        self.assertEqual(len(initializer.loaded), 4)
        self.assertEqual(len(initializer.records), 4)
        for proof, raw in zip(initializer.records, [initializer.manifest['tables'][i] for i in (0, 2, 1, 3)]):
            self.assertEqual(proof['actual_bytes'], raw['bytes'])
            self.assertEqual(proof['actual_sha256'], raw['sha256'])
            self.assertTrue(proof['sha256_checked'])
            self.assertFalse(proof['preparation_cuda_initialized'])
            self.assertEqual(proof['manifest_sha256'], initializer.manifest_sha256)

    def test_parallel_two_cpu_workers_match_serial_raw_bytes(self):
        serial = self.prepare()
        parallel = self.prepare(name='parallel', processes=2)
        left, right = json.loads(serial.read_text()), json.loads(parallel.read_text())
        self.assertEqual(left, right)
        for item in left['tables']:
            self.assertEqual((serial.parent/item['file']).read_bytes(), (parallel.parent/item['file']).read_bytes())
        self.assertFalse(torch.cuda.is_initialized())

    def test_single_owner_and_small_tail_match_original_loop(self):
        config = replace(self.config, layer_ids=(0,), vocab_per_head=17)
        path = self.prepare(config=config, world=1)
        original, cached = [Qwen3MoeForCausalLM(self.model_config) for _ in range(2)]
        attach_engram(original, config, EngramPolicy('cpu'))
        attach_engram(cached, config, EngramPolicy('cpu'), table_initializer=self.loader(path, config=config, world=1))
        self.assertTrue(torch.equal(original.model.engram_tables['0'].weight, cached.model.engram_tables['0'].weight))
        for model in (original, cached):
            for table in model.model.engram_tables.values(): table.close()

    def test_corrupted_stream_is_rejected_after_copy_before_model_can_run(self):
        path = self.prepare(); loader = self.loader(path)
        item = loader.manifest['tables'][0]
        raw = path.parent/item['file']
        with raw.open('r+b') as stream:
            value = stream.read(1); stream.seek(0); stream.write(bytes([value[0] ^ 1]))
        table = SimpleNamespace(rank=item['rank'], size=2, global_num_embeddings=item['global_rows'],
             row_start=item['row_start'], row_stop=item['row_stop'], weight=torch.empty(item['shape'], dtype=torch.bfloat16))
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'): loader(table, layer_id=item['layer_id'])
        self.assertEqual(loader.loaded, [])

    def test_wrong_source_seed_layout_or_raw_size_fails_closed(self):
        path = self.prepare()
        with self.assertRaisesRegex(ValueError, 'identity differs'):
            self.loader(path, config=replace(self.config, seed=self.config.seed+1))
        with self.assertRaisesRegex(ValueError, 'identity differs'): self.loader(path, world=1)
        raw = json.loads(path.read_text())
        raw['tables'][0]['row_start'] += 1; path.write_text(json.dumps(raw))
        with self.assertRaisesRegex(ValueError, 'layout differs'): self.loader(path)
        raw['tables'][0]['row_start'] -= 1; path.write_text(json.dumps(raw))
        (path.parent/raw['tables'][0]['file']).write_bytes(b'x')
        with self.assertRaisesRegex(ValueError, 'wrong size'): self.loader(path)

    def test_refuse_incomplete_or_existing_directory_and_never_overwrite(self):
        path = self.prepare(); original = path.read_bytes()
        with self.assertRaisesRegex(ValueError, 'existing initialization'): self.prepare()
        self.assertEqual(path.read_bytes(), original)
        with patch('scripts.qwen_dummy_init_cache._generate_owner', side_effect=RuntimeError('interrupted')):
            with self.assertRaisesRegex(RuntimeError, 'interrupted'): self.prepare(name='interrupted')
        self.assertFalse((self.root/'interrupted/manifest.json').exists())
        with self.assertRaisesRegex(ValueError, 'existing initialization'): self.prepare(name='interrupted')
        with self.assertRaisesRegex(ValueError, 'manifest required'): self.loader(self.root/'interrupted/manifest.json')

    def test_gpu_initialized_preparation_and_insufficient_disk_are_rejected(self):
        with patch('torch.cuda.is_initialized', return_value=True), self.assertRaisesRegex(ValueError, 'CPU-only'):
            self.prepare()
        with patch('scripts.qwen_dummy_init_cache.shutil.disk_usage', return_value=SimpleNamespace(free=0)), \
             self.assertRaisesRegex(ValueError, 'insufficient free disk'):
            self.prepare()
        self.assertFalse((self.root/'cache').exists())

    def test_destination_dtype_and_duplicate_load_rejected(self):
        loader = self.loader(self.prepare()); item = loader.manifest['tables'][0]
        table = SimpleNamespace(rank=0, size=2, global_num_embeddings=item['global_rows'], row_start=item['row_start'],
                                row_stop=item['row_stop'], weight=torch.empty(item['shape'], dtype=torch.float32))
        with self.assertRaisesRegex(ValueError, 'layout/dtype'): loader(table, layer_id=0)
        table.weight = table.weight.bfloat16(); loader(table, layer_id=0)
        with self.assertRaisesRegex(ValueError, 'twice'): loader(table, layer_id=0)


if __name__ == '__main__': unittest.main()
