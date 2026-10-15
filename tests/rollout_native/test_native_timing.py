import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock
from unittest.mock import patch

from rollout_manager.adapters.native_timing import HarnessInstrumentationError, NativeHarnessTiming
from rollout_manager.adapters.input_memory import make_memory_plan
from rollout_manager import runner

try:
    import torch
except ImportError:
    torch = None


class NativeTimingTests(unittest.TestCase):
    def test_native_generator_is_called_each_time_without_replacing_values(self):
        timing = NativeHarnessTiming(None, "kernelbench", 42)
        values = [[object()] for _ in range(7)]
        iterator = iter(values)
        generate = timing.wrap_generator(lambda: next(iterator))
        for expected in values:
            self.assertIs(generate(), expected)
        report = timing.finish(True)
        self.assertEqual(report["input_calls"], 7)
        self.assertEqual(report["expected_input_calls"], 7)
        self.assertEqual(report["input_source"], "native_generator")
        self.assertEqual([event["stage"] for event in report["events"]], ["input_generation"] * 7)

    def test_trial_counts_and_early_correctness_failures(self):
        for mode, count in (("verify", 5), ("profile", 1), ("kernelbench", 7)):
            timing = NativeHarnessTiming(None, mode, 1042)
            generate = timing.wrap_generator(lambda: [])
            for _ in range(count - 1):
                generate()
            timing.finish(False)
            with self.assertRaises(HarnessInstrumentationError):
                timing.finish(True)
            generate()
            timing.finish(True)
            with self.assertRaises(HarnessInstrumentationError):
                generate()

    def test_native_scoring_does_not_add_device_timing(self):
        timing = NativeHarnessTiming(None, "kernelbench", 42)
        class Model:
            def forward(self, value):
                return value
        timing.wrap_forward(Model, "candidate_execution")
        with timing.native_timing():
            self.assertEqual(Model().forward(17), 17)
        self.assertFalse(timing.suppress_device_timing)
        self.assertEqual(timing.metrics["events"], [])

    def test_removed_cli_options_fail_before_execution(self):
        base = ["rollout-manager", "--mode", "tools", "--root", "run", "--assets", "assets",
                "--credentials", "credentials", "--cases", "cases.json", "--catalog", "catalog.json",
                "--url", "https://example.invalid"]
        for option in ("--preparation-workers", "--preparation-gib"):
            with patch("sys.argv", base + [option, "8"]), patch.object(runner, "run") as run, \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as exc:
                    runner.main()
                self.assertEqual(exc.exception.code, 2)
                run.assert_not_called()

    def test_failed_correctness_keeps_reference_scoring_logical_index(self):
        timing = NativeHarnessTiming(MagicMock(), "kernelbench", 42)
        module = SimpleNamespace(load_original_model_and_inputs=lambda: None,
            load_custom_model=lambda: None, _process_input_tensor=lambda x: x,
            graceful_eval_cleanup=lambda: None,
            run_and_check_correctness=lambda: SimpleNamespace(correctness=False))
        timing.install_kernelbench(module)
        generate = timing.wrap_generator(lambda: [])
        generate()
        module.run_and_check_correctness()
        generate()
        self.assertEqual(timing.finish(False)["logical_input_indices"], [0, 6])

    @unittest.skipIf(torch is None, "Torch is required for CPU generator checks")
    def test_timing_preserves_native_tensors_and_rng_at_both_seeds(self):
        for seed in (42, 1042):
            def generate():
                return [torch.rand(17, 3), torch.randint(0, 5, (17,)), 3.14]
            torch.manual_seed(seed)
            expected = [generate() for _ in range(7)]
            expected_state = torch.get_rng_state().clone()
            torch.manual_seed(seed)
            timing = NativeHarnessTiming(torch, "kernelbench", seed)
            wrapped = timing.wrap_generator(generate)
            for row in expected:
                actual = wrapped()
                self.assertTrue(torch.equal(row[0], actual[0]))
                self.assertTrue(torch.equal(row[1], actual[1]))
                self.assertEqual(row[2], actual[2])
            self.assertTrue(torch.equal(expected_state, torch.get_rng_state()))
            timing.finish(True)


@unittest.skipIf(torch is None, "Torch is required for meta-memory estimates")
class InputMemoryTests(unittest.TestCase):
    def test_meta_estimate_does_not_allocate_inputs_initialize_model_or_advance_rng(self):
        source = """import torch
N = 17
class Model:
    def __init__(self):
        raise RuntimeError('model construction must not run')
def get_inputs():
    x = torch.rand(N, 3)
    return [x, x.T, torch.randint(0, 5, (N,)), 3.14]
"""
        before = torch.get_rng_state().clone()
        plan = make_memory_plan(source)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(plan["format"], "native-input-memory-v1")
        self.assertEqual(plan["allocation_bytes"], 448)
        self.assertEqual(plan["outputs"][1]["stride"], [1, 3])
        self.assertEqual(plan["outputs"][2]["dtype"], "int64")

    def test_unsupported_generator_requires_an_explicit_memory_estimate(self):
        with self.assertRaisesRegex(ValueError, "unreviewed"):
            make_memory_plan("import torch\ndef get_inputs(): return [torch.randn(3)]")
