"""Stage timing around native input generation and correctness evaluation."""
from __future__ import annotations

import os
import time
from contextlib import contextmanager


class HarnessInstrumentationError(RuntimeError):
    """Harness/instrumentation failure, not an incorrect candidate reward."""


class NativeHarnessTiming:
    def __init__(self, torch, mode, seed, reuse=None, isolate_device_inputs=False):
        self.torch, self.mode, self.seed = torch, mode, seed
        self.reuse = reuse
        self.logical_index = 0
        self.expected_calls = {"kernelbench": 7, "verify": 5, "profile": 1}[mode]
        self.metrics = {"events": [], "input_calls": 0,
                        "input_source": "native_generator_with_reuse" if reuse else "native_generator"}
        self.suppress_device_timing = False
        self.error = None
        self.isolate_device_inputs = isolate_device_inputs
        if isolate_device_inputs:
            if mode != 'kernelbench':
                raise HarnessInstrumentationError('isolated example supports KernelBench only')
            generator = torch.Generator(device='cpu').manual_seed(seed)
            self.trial_seeds = [torch.randint(0, 2**32-1, (1,), generator=generator).item() for _ in range(5)] + [seed, seed]
            self.metrics['input_isolation'] = 'separate reference/candidate clones in correctness; native separate scoring inputs'
            self.metrics['trial_seeds'] = self.trial_seeds
        self.nvtx = os.environ.get("ROLLOUT_NATIVE_NVTX") == "1"

    @contextmanager
    def timed(self, stage, *, cuda=False):
        start = time.monotonic_ns()
        if self.nvtx:
            self.torch.cuda.nvtx.range_push(stage)
        events = None
        if cuda and not self.suppress_device_timing:
            events = (self.torch.cuda.Event(enable_timing=True), self.torch.cuda.Event(enable_timing=True))
            events[0].record()
        try:
            yield
        finally:
            gpu_ms = None
            if events:
                events[1].record()
                events[1].synchronize()
                gpu_ms = events[0].elapsed_time(events[1])
            self.metrics["events"].append({"stage": stage, "start_monotonic_ns": start,
                "wall_seconds": (time.monotonic_ns()-start)/1e9, "cuda_event_ms": gpu_ms})
            if self.nvtx:
                self.torch.cuda.nvtx.range_pop()

    def wrap_generator(self, original):
        def inputs():
            if self.metrics["input_calls"] >= self.expected_calls:
                self.error = "native harness requested extra input generation"
                raise HarnessInstrumentationError(self.error)
            with self.timed("input_generation"):
                value = (self.reuse.get_inputs(original, self.logical_index)
                         if self.reuse else original())
            self.metrics.setdefault("logical_input_indices", []).append(self.logical_index)
            self.logical_index += 1
            self.metrics["input_calls"] += 1
            return value
        return inputs

    def wrap_forward(self, cls, stage):
        original = cls.forward
        adapter = self

        def forward(instance, *args, **kwargs):
            if adapter.suppress_device_timing:
                return original(instance, *args, **kwargs)
            if adapter.isolate_device_inputs:
                from .isolated_inputs import clone_call_inputs
                try:
                    with adapter.timed(stage + '_input_isolation', cuda=True):
                        args, kwargs = clone_call_inputs(adapter.torch, args, kwargs)
                except Exception as exc:
                    adapter.error = 'correctness input isolation failed: ' + str(exc)
                    raise HarnessInstrumentationError(adapter.error) from exc
            with adapter.timed(stage, cuda=True):
                return original(instance, *args, **kwargs)
        cls.forward = forward
        return cls

    def wrap_call(self, original, stage, *, cuda=False):
        def call(*args, **kwargs):
            with self.timed(stage, cuda=cuda):
                return original(*args, **kwargs)
        return call

    @contextmanager
    def native_timing(self):
        previous = self.suppress_device_timing
        self.suppress_device_timing = True
        try:
            yield
        finally:
            self.suppress_device_timing = previous

    def install_cuda_agent(self, module):
        module.get_inputs = self.wrap_generator(module.get_inputs)
        self.wrap_forward(module.Model, "reference_execution")
        self.wrap_forward(module.ModelNew, "candidate_execution")
        original_transform = module.transform_tensors

        def transform(tensors, fn):
            # The upstream recursive transform revisits this wrapper per tensor.
            values = [tensors] if isinstance(tensors, self.torch.Tensor) else tensors
            host = any(isinstance(x, self.torch.Tensor) and x.device.type == "cpu" for x in values)
            if isinstance(tensors, self.torch.Tensor):
                with self.timed("h2d" if host else "device_clone", cuda=True):
                    return fn(tensors)
            return original_transform(tensors, fn)
        module.transform_tensors = transform
        if self.mode == "verify":
            module.check_equal = self.wrap_call(module.check_equal, "validation", cuda=True)
        else:
            benchmark = module.benchmark_model

            def measured(*args, **kwargs):
                with self.native_timing(), self.timed("native_scoring"):
                    return benchmark(*args, **kwargs)
            module.benchmark_model = measured

    def install_kernelbench(self, module):
        load_ref = module.load_original_model_and_inputs
        load_new = module.load_custom_model

        def reference(*args, **kwargs):
            cls, init, inputs = load_ref(*args, **kwargs)
            return self.wrap_forward(cls, "reference_execution"), init, self.wrap_generator(inputs)

        def candidate(*args, **kwargs):
            cls = load_new(*args, **kwargs)
            return self.wrap_forward(cls, "candidate_execution") if cls is not None else cls

        module.load_original_model_and_inputs = reference
        module.load_custom_model = candidate
        correctness = module.run_and_check_correctness

        def checked(*args, **kwargs):
            self.logical_index = 0
            try:
                result = correctness(*args, **kwargs)
            except BaseException:
                self.logical_index = 6
                raise
            # Upstream skips candidate scoring after a failed correctness trial,
            # but still generates the independent reference scoring input.
            self.logical_index = 5 if result.correctness else 6
            return result
        module.run_and_check_correctness = checked
        transfer = module._process_input_tensor

        def process(x, *args, **kwargs):
            if isinstance(x, self.torch.Tensor) and x.device.type == "cpu":
                with self.timed("h2d", cuda=True):
                    return transfer(x, *args, **kwargs)
            return transfer(x, *args, **kwargs)
        module._process_input_tensor = process
        self.torch.allclose = self.wrap_call(self.torch.allclose, "validation", cuda=True)
        module.graceful_eval_cleanup = self.wrap_call(module.graceful_eval_cleanup, "native_cleanup")

    def finish(self, correct):
        if self.reuse:
            self.metrics["input_reuse"] = self.reuse.metrics()
        if self.error:
            raise HarnessInstrumentationError(self.error)
        if correct and self.metrics["input_calls"] != self.expected_calls:
            raise HarnessInstrumentationError("successful native harness did not perform all expected input calls")
        self.metrics["expected_input_calls"] = self.expected_calls
        self.metrics["finished_monotonic_ns"] = time.monotonic_ns()
        return self.metrics
