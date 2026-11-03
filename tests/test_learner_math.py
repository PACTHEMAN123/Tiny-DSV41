"""Configuration regression only; real layout equivalence requires GPU tests."""
from contextlib import ExitStack
import types
import unittest
from unittest.mock import patch

from dsv41_train.rl import learner_math as policy
from dsv41_train.rl.distributed_grpo import run


class Matmul:
    def __init__(self):
        self.flags = (True, True)

    @property
    def allow_bf16_reduced_precision_reduction(self):
        return self.flags[0]

    @allow_bf16_reduced_precision_reduction.setter
    def allow_bf16_reduced_precision_reduction(self, value):
        self.flags = value

    @property
    def allow_bf16_reduced_precision_reduction_split_k(self):
        return self.flags[1]


class LearnerMathTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.api, self.backend, self.precision = Matmul(), 'cublas', 'high'
        self.calls = []
        def backend(value=None):
            if value is not None:
                self.calls.append(value)
                self.backend = value
            return types.SimpleNamespace(name=self.backend)
        def precision(value):
            self.precision = value
        self.stack.enter_context(patch.object(policy.torch.backends.cuda, 'matmul', self.api))
        self.stack.enter_context(patch.object(policy.torch.backends.cuda, 'preferred_blas_library', side_effect=backend))
        self.stack.enter_context(patch.object(policy.torch, 'get_float32_matmul_precision', side_effect=lambda: self.precision))
        self.setter = self.stack.enter_context(patch.object(policy.torch, 'set_float32_matmul_precision', side_effect=precision))

    def test_default_preserves_existing_settings(self):
        before = policy.learner_math_configuration()
        with policy.configure_learner_math('default') as current:
            self.assertEqual(current, before)
        self.assertEqual(policy.learner_math_configuration(), before)
        self.assertEqual(self.calls, [])
        self.setter.assert_not_called()

    def test_explicit_mode_and_exception_restore_both_flags_backend_and_precision(self):
        self.api.flags = (False, True)
        before = policy.learner_math_configuration()
        with self.assertRaisesRegex(RuntimeError, 'body failure'):
            with policy.configure_learner_math('bf16-no-split-k') as current:
                self.assertEqual(current, policy.require_bf16_no_split_k())
                self.assertEqual(self.api.flags, (False, False))
                self.assertEqual((self.backend, self.precision), ('cublaslt', 'highest'))
                raise RuntimeError('body failure')
        self.assertEqual(policy.learner_math_configuration(), before)

    def test_unsupported_split_k_api_fails_before_changing_settings(self):
        old_api = types.SimpleNamespace(allow_bf16_reduced_precision_reduction=True)
        with patch.object(policy.torch.backends.cuda, 'matmul', old_api):
            with self.assertRaisesRegex(ValueError, 'split-K control'):
                with policy.configure_learner_math('bf16-no-split-k'):
                    self.fail('unsupported math policy entered')
        self.assertEqual(self.calls, [])
        self.setter.assert_not_called()

    def test_sharding_requires_explicit_mode_and_actual_settings(self):
        with self.assertRaisesRegex(ValueError, 'explicit bf16-no-split-k'):
            run(types.SimpleNamespace(learner_response_ownership='sharded', learner_batch_shapes=None))
        with self.assertRaisesRegex(ValueError, 'math policy'):
            policy.require_bf16_no_split_k()


if __name__ == '__main__':
    unittest.main()
