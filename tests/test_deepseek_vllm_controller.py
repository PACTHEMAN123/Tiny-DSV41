import asyncio
import copy
import json
import multiprocessing
import os
from pathlib import Path
import socket
import sys
import tempfile
import threading
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
from dsv41_train.rl.lora_config import DistributedLoRAConfig

from rollout_manager.backends.vllm_actor import LoRAActorEngine
from rollout_manager.backends.vllm_controller import OwnedVLLMActor
from rollout_manager.backends.vllm_lora_controller import LoRARolloutConfig, VLLMLoRAController
from rollout_manager.backends.vllm_lora_transport import pack_update, policy_ack
from test_deepseek_vllm_transport import update


def _spawn_lora_worker_ipc_receipt(output):
    """Fresh spawned interpreter exercises the actual worker import hook."""
    vllm = ModuleType("vllm")
    vllm.__version__ = "0.31.0"
    vllm.ModelRegistry = SimpleNamespace(register_model=lambda *args: None)
    sys.modules["vllm"] = vllm
    with patch("rollout_manager.backends.vllm_lora_triton.configure_lora_triton_source"):
        import rollout_manager.backends.vllm_lora_worker
    path = Path(tempfile.gettempdir()) / "spawn-worker.sock"
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
        listener.bind(str(path))
        record = {"tempdir": tempfile.gettempdir(), "TMPDIR": os.environ["TMPDIR"],
                  "rpc_base": os.environ["VLLM_RPC_BASE_PATH"], "socket": str(path),
                  "compiler_real": str(Path(os.environ["TMPDIR"]).resolve(strict=True))}
    path.unlink()
    Path(output).write_text(json.dumps(record))


class LoRAIPCPathTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform == "linux", "long native IPC paths require Linux proc aliases")
    def test_long_owned_cwd_uses_short_sockets_and_spawn_keeps_compiler_tmp_absolute(self):
        from rollout_manager.backends.vllm_lora_ipc import create_lora_socket_directory
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / ("long-owned-working-directory-" * 5)
            root.mkdir()
            (root / "tmp").mkdir(mode=0o700)
            compiler_tmp = root / "compiler-tmp"
            compiler_tmp.mkdir(mode=0o700)
            old_cwd = Path.cwd()
            try:
                os.chdir(root)
                with patch.dict(os.environ, {"DSV41_LORA_IPC_DIR": "tmp",
                                             "TMPDIR": str(compiler_tmp)}):
                    directory, base = create_lora_socket_directory()
                    self.assertEqual(base, "tmp")
                    self.assertFalse(directory.is_absolute())
                    path = directory / "actor.sock"
                    self.assertGreater(len(os.fsencode(path.resolve())), 104)
                    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
                        listener.bind(str(path))
                    path.unlink()
                    directory.rmdir()
                    output = root / "spawn-receipt.json"
                    child = multiprocessing.get_context("spawn").Process(
                        target=_spawn_lora_worker_ipc_receipt, args=(str(output),))
                    child.start()
                    child.join(timeout=20)
                    if child.is_alive():
                        child.terminate()
                        child.join(timeout=5)
                    self.assertEqual(child.exitcode, 0)
                    record = json.loads(output.read_text())
                    self.assertEqual(record["tempdir"], "tmp")
                    self.assertEqual(record["rpc_base"], "tmp")
                    self.assertEqual(record["socket"], "tmp/spawn-worker.sock")
                    self.assertEqual(record["compiler_real"], str(compiler_tmp))
                    self.assertTrue(record["TMPDIR"].startswith("/proc/"))
                    self.assertEqual(os.environ["TMPDIR"], str(compiler_tmp))
            finally:
                os.chdir(old_cwd)

    def test_unset_is_noop_and_external_symlink_or_writable_directory_is_rejected(self):
        from rollout_manager.backends.vllm_lora_ipc import configure_lora_ipc, lora_ipc_directory
        with patch.dict(os.environ, clear=False), patch("tempfile.tempdir", "unchanged"):
            os.environ.pop("DSV41_LORA_IPC_DIR", None)
            before = dict(os.environ)
            self.assertIsNone(configure_lora_ipc())
            self.assertEqual(tempfile.tempdir, "unchanged")
            self.assertEqual(dict(os.environ), before)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "private").mkdir(mode=0o700)
            (root / "writable").mkdir(mode=0o700)
            (root / "writable").chmod(0o777)
            (root / "symlink").symlink_to(root / "private", target_is_directory=True)
            old_cwd = Path.cwd()
            try:
                os.chdir(root)
                for invalid in ("", ".", "../escape", str(root / "private"), "symlink", "writable"):
                    with self.subTest(path=invalid), patch.dict(os.environ, {"DSV41_LORA_IPC_DIR": invalid}):
                        with self.assertRaises(ValueError):
                            lora_ipc_directory()
            finally:
                os.chdir(old_cwd)


def workers(base, manifest=None, lora_config=None):
    return [{"pid": index + 1, "start_ticks": 1, "cmdline_sha256": "fixture", "device_uuid": f"GPU-{index}",
             "base_identity": base,
             "lora_config": (lora_config or DistributedLoRAConfig().to_dict()),
             "policy_ack": policy_ack(manifest) if manifest else None,
             "adapter_id": manifest["adapter_id"] if manifest else None,
             "readback_digest": manifest["digest"] if manifest else None} for index in range(2)]


class FakeActor:
    def __init__(self, model_dir, config, directory, *, max_model_len, bootstrap_override):
        self.bootstrap = bootstrap_override
        self.manifest = None
        self.bad_worker = False
        self.started, self.release = threading.Event(), threading.Event()
        self.release.set()
        self.receipt = self.stats()
        self.closed = False

    def stats(self):
        return {"serving_profile": "deepseek-lora-v1", "base_identity": self.bootstrap["base_identity"],
                "lora_config": self.bootstrap["lora_config"],
                "tensor_parallel_size": 2, "workers": workers(
                    self.bootstrap["base_identity"], self.manifest, self.bootstrap["lora_config"]),
                "loaded_policy_ack": policy_ack(self.manifest) if self.manifest else None,
                "adapter_id": self.manifest["adapter_id"] if self.manifest else None}

    def call(self, operation, *, tensor_update=None, **values):
        if operation == "load_lora":
            self.manifest = copy.deepcopy(tensor_update[0])
            result = {"ack": policy_ack(self.manifest), "adapter_id": self.manifest["adapter_id"],
                      "activation_receipts": workers(self.bootstrap["base_identity"], self.manifest,
                                                     self.bootstrap["lora_config"]), "engine": self.stats()}
            if self.bad_worker:
                result["activation_receipts"][1]["readback_digest"] = "wrong"
            return result
        if operation == "generate_batch":
            self.started.set()
            if not self.release.wait(timeout=5):
                raise TimeoutError("unit fixture blocked")
            specs = values["specs"]
            results = []
            for index, spec in enumerate(specs):
                results.append({"context_token_ids": spec["prompt_ids"], "action_token_ids": [2],
                    "behavior_log_probs": [-.5], "finish_reason": "length", "stop_token_count": 0,
                    "logprob_convention": "sampled_distribution", "generation_metrics": {
                        "policy_ack": policy_ack(self.manifest), "adapter_request_id": values["adapter_id"],
                        "request_id": spec["request_id"], "sampling_seed": spec["sampling_seed"],
                        "engine_request_id": str(index), "batch_request_index": index}})
            return {"results": results, "engine_receipt": self.stats(), "batch_id": values["batch_id"]}
        raise AssertionError(operation)

    def close(self):
        self.closed = True


class LoRAControllerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "config.json").write_text(json.dumps({"model_type": "deepseek_v41", "eos_token_id": 1,
            "hidden_size": 4, "q_lora_rank": 2, "head_dim": 4, "num_hidden_layers": 1}))
        (self.root / "model.safetensors.index.json").write_text("{}")
        self.config = LoRARolloutConfig(deadline_unix=time.time() + 600, rollout_devices=("cuda:0", "cuda:1"))
        self.mock = patch("rollout_manager.backends.vllm_lora_controller.OwnedVLLMActor", FakeActor)
        self.mock.start()
        self.addCleanup(self.mock.stop)
        self.controller = VLLMLoRAController(self.root, self.config, work_dir=self.root / "work")
        await self.controller.start()
        self.addAsyncCleanup(self.controller.close)

    async def test_direct_update_versioning_and_pinned_sampling(self):
        manifest, payload, _ = update(base_identity=self.controller.base_identity)
        await self.controller.activate(manifest, payload)
        await self.controller.resume()
        with self.assertRaisesRegex(RuntimeError, "paused"):
            await self.controller.activate(*update(1, 2, self.controller.base_identity)[:2])
        result = self.controller.sample_tokens([1], 2, .8, "a" * 32, sampling_seed=123)
        self.assertEqual(result["generation_metrics"]["adapter_request_id"], 1)
        with self.assertRaisesRegex(RuntimeError, "replay"):
            self.controller.sample_tokens([1], 2, .8, "a" * 32, sampling_seed=123)
        await self.controller.pause()
        with self.assertRaisesRegex(ValueError, "next version"):
            await self.controller.activate(*update(1, 1, self.controller.base_identity)[:2])
        await self.controller.activate(*update(1, 2, self.controller.base_identity)[:2])
        self.assertEqual(self.controller.ack["version"], 1)

    async def test_bad_rank_readback_permanently_fails_controller(self):
        self.controller.actor.bad_worker = True
        with self.assertRaisesRegex(RuntimeError, "readback"):
            await self.controller.activate(*update(base_identity=self.controller.base_identity)[:2])
        self.assertEqual(self.controller._state, "failed")
        with self.assertRaises(RuntimeError):
            await self.controller.resume()

    async def test_custom_profile_requires_matching_peft_and_exact_layer_inventory(self):
        import torch
        profile = DistributedLoRAConfig(rank=16, alpha=24, layers=(0,))
        config = LoRARolloutConfig(rollout_devices=self.config.rollout_devices, lora_config=profile)
        controller = VLLMLoRAController(self.root, config, work_dir=self.root / "custom")
        self.addAsyncCleanup(controller.close)
        receipt = await controller.start()
        self.assertEqual(receipt["lora_config"], profile.to_dict())
        tensors = {"language_model.model.layers.0.attn.wq_a.lora_A.weight":
                   torch.ones(16, 4, dtype=torch.bfloat16),
                   "language_model.model.layers.0.attn.wq_a.lora_B.weight":
                   torch.ones(2, 16, dtype=torch.bfloat16)}
        def packed(values, alpha=24):
            return pack_update(values, version=0, lineage="configured", adapter_id=1,
                peft={"r": 16, "lora_alpha": alpha, "target_modules": ["wq_a"]},
                base_identity=controller.base_identity)
        with self.assertRaisesRegex(ValueError, "PEFT"):
            await controller.activate(*packed(tensors, alpha=16))
        with self.assertRaisesRegex(ValueError, "configured query layers"):
            await controller.activate(*packed({next(iter(tensors)): next(iter(tensors.values()))}))
        self.assertIsNone(controller.actor.manifest)
        self.assertEqual(controller._state, "paused")
        await controller.activate(*packed(tensors))
        self.assertEqual(controller.ack["version"], 0)

    async def test_pause_waits_for_inference_before_next_activation(self):
        await self.controller.activate(*update(base_identity=self.controller.base_identity)[:2])
        await self.controller.resume()
        self.controller.actor.release.clear()
        sample = asyncio.create_task(asyncio.to_thread(self.controller.sample_tokens,
            [1], 2, .8, "b" * 32, sampling_seed=1))
        await asyncio.to_thread(self.controller.actor.started.wait, 2)
        pause = asyncio.create_task(self.controller.pause())
        await asyncio.sleep(.02)
        self.assertFalse(pause.done())
        self.controller.actor.release.set()
        await sample
        self.assertTrue((await pause)["drain_complete"])

    async def test_queued_request_cannot_dispatch_after_unknown_inference_failure(self):
        await self.controller.activate(*update(base_identity=self.controller.base_identity)[:2])
        await self.controller.resume()
        entered, release, calls = threading.Event(), threading.Event(), []
        def failed(operation, **values):
            calls.append(operation)
            entered.set()
            release.wait(timeout=2)
            raise TimeoutError("unknown first remote outcome")
        self.controller.actor.call = failed
        first = asyncio.create_task(asyncio.to_thread(self.controller.sample_tokens,
            [1], 2, .8, "c" * 32, sampling_seed=1))
        await asyncio.to_thread(entered.wait, 2)
        second = asyncio.create_task(asyncio.to_thread(self.controller.sample_tokens,
            [1], 2, .8, "d" * 32, sampling_seed=2))
        for _ in range(100):
            if self.controller._active == 2:
                break
            await asyncio.sleep(.005)
        self.assertEqual(self.controller._active, 2)
        release.set()
        outcomes = await asyncio.gather(first, second, return_exceptions=True)
        self.assertIsInstance(outcomes[0], TimeoutError)
        self.assertIsInstance(outcomes[1], RuntimeError)
        self.assertEqual(calls, ["generate_batch"])

    async def test_closed_controller_cannot_publish_cached_live_engine_receipt(self):
        await self.controller.close()
        with self.assertRaisesRegex(RuntimeError, "healthy"):
            self.controller.engine_receipt()


class LoRAActorBindingTests(unittest.TestCase):
    def test_constructor_uses_exact_deepseek_profile_and_requires_all_eight_workers(self):
        from rollout_manager.backends.deepseek_v41_lora import lora_target_modules
        text = dict(hidden_size=16, q_lora_rank=8, head_dim=4, num_hidden_layers=2)
        base = {"config_sha256": "a" * 64, "index_sha256": "b" * 64}
        receipts = [{"pid": i + 1, "device_uuid": f"GPU-{i}", "base_identity": base,
                     "lora_config": DistributedLoRAConfig().to_dict()}
                    for i in range(8)]
        calls, registered = [], {}
        class SyncMPClient:
            pass
        class LLM:
            def __init__(self, **values):
                if registered.get("DeepseekV41ForCausalLM") != (
                        "rollout_manager.backends.deepseek_v41_lora_model:DeepseekV41LoRAForCausalLM"):
                    raise AssertionError("the owned class must be registered before LLM configuration")
                calls.append(values)
                self.llm_engine = SimpleNamespace(engine_core=SyncMPClient())
            def collective_rpc(self, name, *, args, timeout):
                self_call = (name, args, timeout)
                self.assertion = self_call
                return receipts
        vllm, envs = ModuleType("vllm"), ModuleType("vllm.envs")
        vllm.__version__, vllm.__file__, vllm.LLM = "0.31.0", __file__, LLM
        vllm.ModelRegistry = SimpleNamespace(register_model=lambda name, target: registered.__setitem__(name, target))
        envs.VLLM_ALLOW_INSECURE_SERIALIZATION = False
        vllm.envs = envs
        core_client = ModuleType("vllm.v1.engine.core_client")
        core_client.SyncMPClient = SyncMPClient
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mounted_config = {"architectures": ["DeepseekV41ForCausalLM"], "text_config": text}
            (root / "config.json").write_text(json.dumps(mounted_config))
            bootstrap = {"model_dir": str(root), "tensor_parallel_size": 8, "seed": 42,
                         "gpu_memory_utilization": .85, "max_model_len": 512,
                         "max_num_seqs": 32, "base_identity": base,
                         "lora_config": DistributedLoRAConfig().to_dict()}
            with patch.dict(sys.modules, {"vllm": vllm, "vllm.envs": envs,
                                         "vllm.v1.engine.core_client": core_client}), \
                    patch("rollout_manager.backends.vllm_lora_triton.configure_lora_triton_source"), \
                    patch("rollout_manager.backends.vllm_batch_admission.qualify_admission_runtime", return_value={"CPU_fixture": True}):
                engine = LoRAActorEngine(bootstrap)
                self.assertEqual(engine.llm.assertion, ("rollout_manager_lora_call", ("stats",), 120))
                self.assertEqual(len(engine.workers), 8)
                self.assertTrue(calls[0]["enable_lora"])
                self.assertFalse(calls[0]["skip_tokenizer_init"])
                self.assertEqual(calls[0]["lora_target_modules"], lora_target_modules(text))
                self.assertEqual(calls[0]["max_lora_rank"], 8)
                self.assertEqual(calls[0]["additional_config"],
                                 {"rollout_manager_lora": bootstrap["lora_config"]})
                self.assertEqual(calls[0]["tensor_parallel_size"], 8)
                self.assertEqual(calls[0]["limit_mm_per_prompt"], {"image": 0})
                self.assertNotIn("quantization", calls[0])
                self.assertNotIn("hf_overrides", calls[0])
                self.assertEqual(json.loads((root / "config.json").read_text()), mounted_config)
                profile = DistributedLoRAConfig(rank=16, alpha=24, layers=(1,))
                bootstrap["lora_config"] = profile.to_dict()
                for worker in receipts:
                    worker["lora_config"] = profile.to_dict()
                configured = LoRAActorEngine(bootstrap)
                self.assertEqual(configured.lora_config, profile)
                self.assertEqual(calls[-1]["max_lora_rank"], 16)
                self.assertEqual(calls[-1]["lora_target_modules"],
                                 ["language_model.model.layers.1.attn.fused_wqa_wkv"])
                self.assertEqual(calls[-1]["additional_config"],
                                 {"rollout_manager_lora": profile.to_dict()})
                receipts.pop()
                with self.assertRaisesRegex(RuntimeError, "every TP worker"):
                    LoRAActorEngine(bootstrap)

    def test_owned_actor_sends_authenticated_json_then_raw_tensor_frame(self):
        from rollout_manager.backends.vllm_actor import receive, send
        from rollout_manager.backends.vllm_lora_transport import receive_update
        manifest, payload, _ = update()
        captured, errors = [], []
        with tempfile.TemporaryDirectory(prefix="lora-rpc-") as temporary:
            path = Path(temporary) / "actor.sock"
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
                listener.bind(str(path))
                listener.listen(1)
                listener.settimeout(2)
                def serve():
                    try:
                        connection, _ = listener.accept()
                        with connection:
                            connection.settimeout(2)
                            captured.append(receive(connection))
                            captured.append(receive_update(connection))
                            send(connection, {"ok": True, "result": {"ack": policy_ack(manifest)}})
                    except BaseException as error:
                        errors.append(error)
                thread = threading.Thread(target=serve, daemon=True)
                thread.start()
                actor = OwnedVLLMActor.__new__(OwnedVLLMActor)
                actor.identity, actor.socket_path, actor.token = {}, path, "unit-private-token"
                with patch("rollout_manager.backends.vllm_controller.check_process"):
                    result = actor.call("load_lora", timeout=2, tensor_update=(manifest, payload))
                thread.join(timeout=3)
                self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(captured[0], {"token": actor.token, "operation": "load_lora"})
        self.assertEqual(captured[1], (manifest, payload))
        self.assertEqual(result, {"ack": policy_ack(manifest)})
        with self.assertRaisesRegex(ValueError, "raw tensor payload"):
            actor.call("load_lora")

    def test_generate_passes_actual_lora_request_and_rejects_stale_binding(self):
        manifest, payload, _ = update()
        engine = LoRAActorEngine.__new__(LoRAActorEngine)
        engine.bootstrap = {"tensor_parallel_size": 2, "base_identity": manifest["base_identity"],
                            "max_model_len": 512, "max_num_seqs": 1, "eos_token_ids": [1]}
        engine.lora_config = DistributedLoRAConfig()
        engine.text_config = dict(hidden_size=4, q_lora_rank=2, head_dim=4, num_hidden_layers=1)
        engine.ack = engine.activation = engine.adapter_request = None
        engine.failed, engine._used_adapter_ids = False, set()
        dispatched = []
        class LLM:
            def collective_rpc(self, name, *, args, timeout):
                return workers(manifest["base_identity"], manifest if args[0] != "observe" else None)
            def reset_prefix_cache(self): return True
            def update_weight_version(self, value): self.version = value
            def get_weight_version(self): return self.version
            def generate(self, prompts, params, *, use_tqdm, lora_request):
                dispatched.append(lora_request)
                choice = SimpleNamespace(token_ids=[2], logprobs=[{2: SimpleNamespace(logprob=-.5)}], finish_reason="length")
                return [SimpleNamespace(request_id="0", finished=True, prompt_token_ids=prompts[0]["prompt_token_ids"], outputs=[choice])]
        engine.llm = LLM()
        vllm, lora, request_module = ModuleType("vllm"), ModuleType("vllm.lora"), ModuleType("vllm.lora.request")
        vllm.SamplingParams = lambda **values: SimpleNamespace(**values)
        request_module.LoRARequest = lambda **values: SimpleNamespace(**values)
        with patch.dict(sys.modules, {"vllm": vllm, "vllm.lora": lora, "vllm.lora.request": request_module}), \
                patch("rollout_manager.backends.vllm_actor.process_identity", return_value={"pid": 123}):
            engine.load_lora(manifest, payload)
            request = {"policy_ack": policy_ack(manifest), "adapter_id": 1, "prompt_ids": [1],
                       "max_tokens": 2, "temperature": .8, "seed": 1}
            engine.generate(request)
            self.assertEqual(dispatched[0].lora_int_id, 1)
            self.assertEqual(dispatched[0].lora_path, "memory://" + manifest["digest"])
            request["adapter_id"] = 2
            with self.assertRaisesRegex(RuntimeError, "immutable"):
                engine.generate(request)



if __name__ == "__main__":
    unittest.main()
