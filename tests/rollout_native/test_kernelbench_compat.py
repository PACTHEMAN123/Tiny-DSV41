import unittest
from types import SimpleNamespace

from rollout_manager.adapters.kernelbench_compat import preserve_integer_inputs


class KernelBenchCompatibilityTests(unittest.TestCase):
    def test_only_integer_and_boolean_activation_casting_changes(self):
        class Tensor:
            def __init__(self, dtype):
                self.dtype = dtype
        fp32 = SimpleNamespace(is_floating_point=True, is_complex=False)
        half = SimpleNamespace(is_floating_point=True, is_complex=False)
        integer = SimpleNamespace(is_floating_point=False, is_complex=False)
        boolean = SimpleNamespace(is_floating_point=False, is_complex=False)
        complex_dtype = SimpleNamespace(is_floating_point=False, is_complex=True)
        torch = SimpleNamespace(Tensor=Tensor, float32=fp32)
        module = SimpleNamespace(_process_input_tensor=lambda *args: args)
        original = module._process_input_tensor
        metadata = preserve_integer_inputs(module, torch)
        for dtype in (integer, boolean):
            tensor = Tensor(dtype)
            result = module._process_input_tensor(tensor, "cuda:0", "cuda", half)
            self.assertEqual(result[:3], (tensor, "cuda:0", "cuda"))
            self.assertIs(result[3], dtype)
        for value in (Tensor(half), Tensor(complex_dtype), 3, None):
            self.assertEqual(module._process_input_tensor(value, "cuda:0", "cuda", half),
                             original(value, "cuda:0", "cuda", half))
        self.assertFalse(metadata["native_file_modified"])
