"""CPU regressions for atomic request admission, independent of vLLM."""
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from rollout_manager.backends.vllm_batch_admission import (
    enqueue_before_schedule, qualify_admission_runtime,
)


class RuntimeQualificationTests(unittest.TestCase):
    def test_runtime_source_or_public_api_change_fails_before_engine_start(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'inspected.py'
            source.write_bytes(b'inspected-runtime')
            pins = {'inspected.py': hashlib.sha256(source.read_bytes()).hexdigest()}
            module = NS(__file__=str(root / '__init__.py'), __version__='0.31.0')
            api = NS(enqueue=lambda: None, wait_for_completion=lambda: None)
            with patch('rollout_manager.backends.vllm_batch_admission.ADMISSION_SOURCE_SHA256', pins):
                self.assertIsNone(qualify_admission_runtime(module, api))
                with self.assertRaisesRegex(RuntimeError, 'supported vLLM'):
                    qualify_admission_runtime(module, NS(enqueue=lambda: None))
                source.write_bytes(b'changed-runtime')
                with self.assertRaisesRegex(RuntimeError, 'supported vLLM'):
                    qualify_admission_runtime(module, api)



class AdmissionBarrierTests(unittest.TestCase):
    def llm(self):
        class LLM:
            def __init__(self):
                self.paused, self.pending = False, []
                self.calls = []
                self.llm_engine = NS(engine_core=NS(call_utility=self.utility),
                                     output_processor=NS(request_states={}),
                                     has_unfinished_requests=lambda: bool(self.pending))

            def utility(self, method, *args):
                self.calls.append((method, args, len(self.pending)))
                if method == 'is_scheduler_paused':
                    return self.paused
                if method == 'pause_scheduler':
                    assert args == ('keep', False)
                    self.paused = True
                elif method == 'resume_scheduler':
                    self.paused = False
                else:
                    raise AssertionError(method)

            def enqueue(self, prompts, params, *, use_tqdm, lora_request):
                assert self.paused and not use_tqdm
                self.calls.append(('enqueue', (), len(prompts)))
                self.pending = list(prompts)
                ids = [f'{100 + i}-abcdef{i:02x}' for i in range(len(prompts))]
                self.llm_engine.output_processor.request_states = {
                    value: NS(request_id=value, external_req_id=str(100 + i),
                              prompt_token_ids=prompt['prompt_token_ids'], lora_request=lora_request)
                    for i, (value, prompt) in enumerate(zip(ids, prompts))}
                return ids

            def wait_for_completion(self, *, use_tqdm):
                assert not self.paused and not use_tqdm
                self.calls.append(('wait', (), len(self.pending)))
                return [NS(request_id=str(100+i), prompt_token_ids=prompt['prompt_token_ids'])
                        for i,prompt in enumerate(self.pending)]
        return LLM()

    def run_batch(self, llm):
        return enqueue_before_schedule(llm, [{'prompt_token_ids': [i, 4]} for i in range(32)],
                                       list(range(32)), 'adapter')

    def test_pause_fifo_enqueue_all_resume_wait_order(self):
        llm = self.llm()
        outputs = self.run_batch(llm)
        self.assertEqual([r.prompt_token_ids for r in outputs], [[i, 4] for i in range(32)])
        self.assertEqual([row[0] for row in llm.calls], [
            'is_scheduler_paused', 'pause_scheduler', 'is_scheduler_paused',
            'enqueue', 'is_scheduler_paused', 'resume_scheduler', 'is_scheduler_paused', 'wait'])
        self.assertEqual(llm.calls[4][2], 32)  # all ADDs precede the paused-status barrier

    def test_enqueue_or_unknown_pause_failure_always_attempts_resume(self):
        for failure in ('enqueue', 'pause_scheduler'):
            llm = self.llm()
            if failure == 'enqueue':
                llm.enqueue = lambda *a, **kw: (_ for _ in ()).throw(ValueError('enqueue failed'))
            else:
                original = llm.utility
                def fail_after_pause(method, *args):
                    value = original(method, *args)
                    if method == 'pause_scheduler':
                        raise ValueError('pause reply lost')
                    return value
                llm.llm_engine.engine_core.call_utility = fail_after_pause
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                self.run_batch(llm)
            self.assertFalse(llm.paused)
            self.assertIn('resume_scheduler', [row[0] for row in llm.calls])
            self.assertNotIn('wait', [row[0] for row in llm.calls])

    def test_foreign_pause_or_pending_requests_are_not_disturbed(self):
        for state in ('paused', 'pending'):
            llm = self.llm()
            if state == 'paused':
                llm.paused = True
            else:
                llm.pending = ['existing']
            with self.subTest(state=state), self.assertRaises(RuntimeError):
                self.run_batch(llm)
            self.assertNotIn('resume_scheduler', [row[0] for row in llm.calls])
            self.assertNotIn('enqueue', [row[0] for row in llm.calls])

    def test_wrong_enqueue_identity_fails_and_resumes_without_wait(self):
        llm = self.llm()
        llm.enqueue = lambda *a, **kw: ['duplicate'] * 32
        with self.assertRaisesRegex(RuntimeError, 'identity/order'):
            self.run_batch(llm)
        self.assertFalse(llm.paused)
        self.assertNotIn('wait', [row[0] for row in llm.calls])

    def test_actual_request_state_mapping_is_checked_before_resume(self):
        for mutation in ('prompt', 'adapter', 'external', 'membership'):
            llm = self.llm()
            original = llm.enqueue
            def changed(*args, **kwargs):
                ids = original(*args, **kwargs)
                states = llm.llm_engine.output_processor.request_states
                if mutation == 'prompt': states[ids[0]].prompt_token_ids = [999]
                if mutation == 'adapter': states[ids[0]].lora_request = 'wrong'
                if mutation == 'external': states[ids[0]].external_req_id = '101'
                if mutation == 'membership': states.pop(ids[-1])
                return ids
            llm.enqueue = changed
            with self.subTest(mutation=mutation), self.assertRaises(RuntimeError):
                self.run_batch(llm)
            self.assertFalse(llm.paused)
            self.assertNotIn('wait', [row[0] for row in llm.calls])




if __name__ == '__main__':
    unittest.main()
