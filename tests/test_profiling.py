import copy
from pathlib import Path
import tempfile
import unittest

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from dsv41_train.models.dsv4 import DeepSeekV41ForCausalLM, load_dsv41_backbone_window
from dsv41_train.loading import LoadPolicy
from dsv41_train.loading.reader import ShardedSafeTensorReader
from dsv41_train.models.dsv4.engram import EngramPolicy
from dsv41_train.profiling import loading_regions, model_regions, summarize_profiler, summarize_cuda_activity
from post_training_helpers import small_config
from test_dsv4_loading import synthetic_checkpoint
from test_memory_modules import LocalMaterializer, memory_config


class ProfilingTest(unittest.TestCase):
    def test_loading_regions_preserve_weights_and_count_worker_reads(self):
        original = ShardedSafeTensorReader.tensor_rows
        with tempfile.TemporaryDirectory() as directory:
            config = memory_config()
            synthetic_checkpoint(Path(directory), config)
            baseline = load_dsv41_backbone_window(directory, num_layers=config.num_hidden_layers,
                                                   dtype=torch.float32)
            try:
                for mode in ("layerwise", "direct_shard"):
                    spans = []
                    with loading_regions(spans):
                        candidate = load_dsv41_backbone_window(directory, num_layers=config.num_hidden_layers,
                            dtype=torch.float32, load_policy=LoadPolicy(mode, chunk_rows=7, pipeline=True),
                            materializer=LocalMaterializer() if mode == "direct_shard" else None)
                    try:
                        for name, value in candidate.state_dict().items():
                            torch.testing.assert_close(value, baseline.state_dict()[name], rtol=0, atol=0)
                        self.assertEqual(sum(span["tensor_bytes"] for span in spans if span["region"] == "load_read"),
                                         candidate.loading_metrics["logical_bytes_read"])
                        self.assertTrue(any(span["domain"] == "engram" and span["region"] == "load_decode"
                                            for span in spans))
                        if mode == "direct_shard":
                            self.assertGreater(len({span["thread"] for span in spans}), 1)
                    finally:
                        candidate.close()
            finally:
                baseline.close()
        self.assertIs(ShardedSafeTensorReader.tensor_rows, original)
        with self.assertRaisesRegex(RuntimeError, "loading failure"):
            with loading_regions([]):
                raise RuntimeError("loading failure")
        self.assertIs(ShardedSafeTensorReader.tensor_rows, original)

    def test_async_lookup_spans_include_worker_and_restore_methods(self):
        model = DeepSeekV41ForCausalLM(small_config(), sparse_engram_gradients=True,
                                      engram_policy=EngramPolicy("cpu", "async", True))
        spans = []
        try:
            tokens = torch.randint(3, 32, (1, 12))
            with torch.no_grad():
                model(tokens)
            with model_regions(model, spans), profile(activities=[ProfilerActivity.CPU],
                    experimental_config=torch._C._profiler._ExperimentalConfig(profile_all_threads=True)) as profiler:
                output = model(tokens)
                output.logits.sum().backward()
            categories = {span["region"] for span in spans}
            self.assertTrue({"engram_store", "engram_prefetch", "engram_resolve",
                             "engram_id_route", "engram_row_route"} <= categories)
            self.assertTrue(any(event.name.startswith("dsv41/engram_store/") for event in profiler.events()))
            for table in model.model.engram_tables.values():
                self.assertNotIn("lookup", table.store.__dict__)
                self.assertNotIn("prepare", table.router.__dict__)
                self.assertNotIn("resolve", table.__dict__)
        finally:
            model.close()

    def test_cuda_overlap_unions_stream_intervals_and_excludes_nccl(self):
        events = [dict(cat="kernel", name="gemm", ts=0, dur=20),
                  dict(cat="kernel", name="gemm", ts=10, dur=20),
                  dict(cat="kernel", name="ncclAllReduce", ts=30, dur=20),
                  dict(cat="gpu_memcpy", name="Memcpy HtoD", ts=15, dur=25, args={"bytes": 100})]
        result = summarize_cuda_activity(events, 1)
        self.assertEqual(result["h2d_compute_overlap_ms"], .015)
        self.assertEqual(result["h2d_compute_overlap_fraction"], .6)
        self.assertEqual(result["activities"]["h2d"]["bytes"], 100)

    def test_regions_preserve_gradients_and_attribute_backward(self):
        torch.manual_seed(42)
        reference = DeepSeekV41ForCausalLM(small_config())
        candidate = copy.deepcopy(reference)
        tokens = torch.randint(3, 32, (1, 12))
        expected = reference(tokens, labels=tokens)
        expected.loss.backward()
        with model_regions(candidate), profile(activities=[ProfilerActivity.CPU]) as profiler:
            with record_function("train/forward"):
                actual = candidate(tokens, labels=tokens)
            with record_function("train/backward"):
                actual.loss.backward()
        torch.testing.assert_close(actual.logits, expected.logits, rtol=0, atol=0)
        for (name, parameter), (_, target) in zip(candidate.named_parameters(), reference.named_parameters()):
            if target.grad is None:
                self.assertIsNone(parameter.grad, name)
            else:
                torch.testing.assert_close(parameter.grad, target.grad, rtol=0, atol=0, msg=name)
        self.assertNotIn("forward", candidate.__dict__)
        summary = summarize_profiler(profiler, 1)
        for component in ("mhc", "attention_core", "routed_experts", "engram"):
            self.assertGreater(summary["components"][f"backward/{component}"]["events"], 0)

    def test_regions_restore_methods_on_failure(self):
        model = DeepSeekV41ForCausalLM(small_config())
        original = model.forward
        with self.assertRaisesRegex(RuntimeError, "test failure"):
            with model_regions(model):
                raise RuntimeError("test failure")
        self.assertEqual(model.forward, original)
        for module in model.modules():
            self.assertNotIn("forward", module.__dict__)


if __name__ == "__main__":
    unittest.main()
