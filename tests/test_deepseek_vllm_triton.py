"""Guard the observed decorated-source truncation without importing Triton/CUDA."""
import ast
import hashlib
import inspect
from pathlib import Path
import re
import tempfile
import types
import unittest
from unittest.mock import patch

from rollout_manager.backends import vllm_lora_triton as compat

SOURCE = '''def identity(*args):
    return lambda function: function

@identity(
    {
        "a": lambda args: int(args["x"]),
        "b": lambda args: int(args["y"]),
    }
)
@identity()
def _topk_index_merge_kernel(value):
    return value + 1
'''


class SourceInspectionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'kernel.py'
        self.path.write_text(SOURCE)
        namespace = {'__name__': compat.TARGET_MODULE}
        exec(compile(SOURCE, str(self.path), 'exec'), namespace)
        self.function = namespace[compat.TARGET_NAME]
        self.first = self.function.__code__.co_firstlineno
        self.digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        function, first = self.function, self.first
        class TruncatedInspection:
            def __getattr__(self, name): return getattr(inspect, name)
            def getsourcelines(self, candidate):
                if candidate is function: return SOURCE.splitlines(keepends=True)[first-1:first+6], first
                return inspect.getsourcelines(candidate)
        self.original = TruncatedInspection()
        self.sources = {compat.TARGET_MODULE: (self.path, self.digest)}
        self.proxy = compat._SourceInspection(self.original, self.sources)

    def test_repairs_only_source_and_preserves_decorators_body_line_and_callable(self):
        raw, _ = self.original.getsourcelines(self.function)
        self.assertIsNone(re.search(r'^def\s+', ''.join(raw), re.MULTILINE))
        lines, first = self.proxy.getsourcelines(self.function)
        expected = ast.parse(SOURCE).body[-1]
        recovered = ast.parse(''.join(lines)).body[0]
        self.assertEqual(ast.dump(expected, include_attributes=False), ast.dump(recovered, include_attributes=False))
        self.assertEqual(first, self.first)
        self.assertEqual(self.function(4), 5)
        self.assertEqual(self.proxy.signature(self.function), inspect.signature(self.function))
        self.assertIs(compat.inspect, inspect)

    def test_ordinary_and_already_complete_inspection_remain_unchanged(self):
        self.assertEqual(self.proxy.getsourcelines(self.test_ordinary_and_already_complete_inspection_remain_unchanged), inspect.getsourcelines(self.test_ordinary_and_already_complete_inspection_remain_unchanged))
        complete = compat._SourceInspection(inspect, self.sources)
        self.assertEqual(complete.getsourcelines(self.function), inspect.getsourcelines(self.function))

    def test_rejects_source_and_filename_drift_without_recovery(self):
        self.path.write_text(SOURCE + '\n# changed\n')
        with self.assertRaisesRegex(ValueError, 'digest'): self.proxy.getsourcelines(self.function)
        other = Path(self.directory.name) / 'other.py'; other.write_text(SOURCE)
        proxy = compat._SourceInspection(inspect, {compat.TARGET_MODULE: (other, self.digest)})
        with self.assertRaisesRegex(ValueError, 'filename'): proxy.getsourcelines(self.function)

    def test_two_pinned_files_recover_multiple_functions_but_not_module_prefix_or_nested(self):
        second = SOURCE.replace(compat.TARGET_NAME, '_gqa_sparse_fwd_kernel')
        second += '\n' + SOURCE[SOURCE.index('@identity('):].replace(compat.TARGET_NAME, '_gqa_sparse_decode_kernel')
        path = Path(self.directory.name) / 'sparse_attn.py'; path.write_text(second)
        module = 'vllm.models.minimax_m3.common.ops.sparse_attn'
        namespace = {'__name__': module}; exec(compile(second, str(path), 'exec'), namespace)
        class TruncatedInspection:
            def getsourcelines(self, function):
                first = function.__code__.co_firstlineno
                return path.read_text().splitlines(keepends=True)[first-1:first+6], first
        proxy = compat._SourceInspection(TruncatedInspection(), {
            **self.sources, module: (path, hashlib.sha256(path.read_bytes()).hexdigest())})
        for name in ('_gqa_sparse_fwd_kernel', '_gqa_sparse_decode_kernel'):
            lines, first = proxy.getsourcelines(namespace[name])
            self.assertEqual(ast.parse(''.join(lines)).body[0].name, name)
            self.assertEqual(namespace[name](5), 6)
        for attribute, value in [('__module__', module + '.unlisted'),
                                 ('__qualname__', 'outer.<locals>._gqa_sparse_fwd_kernel')]:
            function = namespace['_gqa_sparse_fwd_kernel']
            original = getattr(function, attribute)
            try:
                setattr(function, attribute, value)
                self.assertEqual(proxy.getsourcelines(function), TruncatedInspection().getsourcelines(function))
            finally:
                setattr(function, attribute, original)

    def test_rejects_nonprefix_and_source_position_changes(self):
        class InvalidInspection:
            def getsourcelines(inner, function): return ['# unrelated\n'], self.first
        proxy = compat._SourceInspection(InvalidInspection(), self.sources)
        with self.assertRaisesRegex(ValueError, 'prefix'): proxy.getsourcelines(self.function)
        function = self.function
        original = function.__code__
        try:
            function.__code__ = original.replace(co_firstlineno=self.first+1)
            with self.assertRaisesRegex(ValueError, 'position'): self.proxy.getsourcelines(function)
        finally:
            function.__code__ = original

    def test_runtime_guard_is_idempotent_and_rejects_unknown_binding_or_source_drift(self):
        package = Path(self.directory.name) / 'vllm'
        target = package / 'models/minimax_m3/common/ops/index_topk.py'
        target.parent.mkdir(parents=True); target.write_text(SOURCE)
        target.with_name('sparse_attn.py').write_text(SOURCE)
        jit_path = package / 'jit.py'; jit_path.write_text('# inspected JIT fixture\n')
        jit = types.SimpleNamespace(inspect=inspect, __file__=str(jit_path))
        vllm = types.SimpleNamespace(__version__='0.31.0', __file__=str(package/'__init__.py'))
        hashes = {'index_topk': self.digest,
                  'sparse_attn': self.digest,
                  'triton_jit': hashlib.sha256(jit_path.read_bytes()).hexdigest(),
                  'inspect': hashlib.sha256(Path(inspect.__file__).read_bytes()).hexdigest()}
        with patch.object(compat.sys, 'version_info', (3,10,9)), patch.dict(compat.sys.modules, {'vllm':vllm}), \
                patch.object(compat.importlib, 'import_module', return_value=jit), \
                patch.object(compat, '_INSPECTION', None), patch.dict(compat.SOURCE_SHA256, hashes):
            compat.configure_lora_triton_source(); selected = jit.inspect
            compat.configure_lora_triton_source()
            self.assertIs(jit.inspect, selected)
            compat._INSPECTION = None; jit.inspect = object()
            with self.assertRaisesRegex(RuntimeError, 'unknown override'): compat.configure_lora_triton_source()
            jit.inspect = inspect; jit_path.write_text('# changed\n')
            with self.assertRaisesRegex(ValueError, 'source digest'): compat.configure_lora_triton_source()


if __name__ == '__main__': unittest.main()
