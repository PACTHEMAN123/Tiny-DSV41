import asyncio
import copy
from dataclasses import replace
import json
import http.client
from pathlib import Path
import socket
import sys
import tempfile
import threading
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from tokenizers import Tokenizer, models, pre_tokenizers

from dsv41_train.models.qwen import Qwen3MoeForCausalLM
from dsv41_train.rl.checkpoint import publish_policy_artifact, run_identity
from dsv41_train.rl.config import RLConfig
from dsv41_train.rl.family import build_policy, load_model_config
from rollout_manager.backends.pytorch_controller import SHELL_PARAMETERS
from rollout_manager.backends.vllm_actor import ActorEngine, output_tokens, receive, send
from rollout_manager.backends.vllm_controller import VLLMPolicyController, request_seed
from rollout_manager.backends.vllm_profile import VLLM_POLICY_IDENTITIES, vllm_policy_identities
from rollout_manager.backends.vllm_worker import (CANONICAL_HEAD, checked_artifact, load_head,
    tensor_digest, kernel_receipt, VLLM_KERNEL_CONFIG, TRITON_EXPERT_CLASS,
    guarded_qwen_kernel_warmup, VLLM_WORKER_RPC, captured_graph_receipt)
from rollout_manager.generation_training import TrainingGenerationContract
from rollout_manager.generation_failure import validate_generation_failure, failure_is_terminal
from rollout_manager.model_completion import validate_model_completion_status
from qwen_helpers import tiny_config, write_hf_checkpoint


class FakeWorkerModel:
    def __init__(self, head, *, padded=0):
        self.config = SimpleNamespace(vocab_size=head.shape[0], hidden_size=head.shape[1])
        self.lm_head = torch.nn.Linear(head.shape[1], head.shape[0] + padded, bias=False).to(head.dtype)
        self.lm_head.quant_method = None

    def load_weights(self, weights):
        with torch.no_grad():
            for name, head in weights:
                if name != "lm_head.weight":
                    raise ValueError("wrong translated name")
                self.lm_head.weight[:head.shape[0]].copy_(head)
        return {"lm_head.weight"}


class VLLMArtifactTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.head = torch.arange(48).view(6, 8).to(torch.bfloat16) / 100
        self.identity = {"trainable_scope": "lm_head", "random_init": False, **VLLM_POLICY_IDENTITIES}
        self.model = SimpleNamespace(trainable_state_dict=lambda: {CANONICAL_HEAD: self.head},
                                     frozen_state_dict=lambda **kwargs: {})
        self.artifact = publish_policy_artifact(self.model, 0, self.root, lineage="fixture",
            identity=self.identity, sampling={"temperature": .8, "top_p": 1, "top_k": 0})

    def test_actual_loader_reads_back_hash_excludes_padding_and_keeps_graph_pointer(self):
        worker = FakeWorkerModel(self.head, padded=2)
        pointer = worker.lm_head.weight.data_ptr()
        receipt = load_head(worker, self.artifact)
        self.assertEqual(receipt["loaded_trainable_tensors"], tensor_digest(self.head))
        self.assertEqual(receipt["policy_ack"], {key: self.artifact[key] for key in ("version", "digest", "lineage")})
        self.assertEqual(receipt["padded_vocabulary_rows"], 8)
        self.assertEqual(worker.lm_head.weight.data_ptr(), pointer)
        torch.testing.assert_close(worker.lm_head.weight[:6], self.head, rtol=0, atol=0)

    def test_digest_metadata_scope_and_dtype_fail_before_activation(self):
        for key, value in (("digest", "0" * 64), ("version", 1), ("lineage", "wrong")):
            altered = dict(self.artifact, **{key: value})
            with self.assertRaisesRegex(ValueError, "mismatch"):
                load_head(FakeWorkerModel(self.head), altered)
        worker = FakeWorkerModel(self.head.float())
        with self.assertRaisesRegex(ValueError, "dtype"):
            load_head(worker, self.artifact)
        self.model.frozen_state_dict = lambda **kwargs: {"unexpected": torch.zeros(1)}
        altered = publish_policy_artifact(self.model, 1, self.root, lineage="fixture",
            identity=self.identity, sampling=self.artifact["sampling"])
        with self.assertRaisesRegex(ValueError, "only"):
            checked_artifact(altered)

    def test_replaced_head_pointer_and_wrong_loader_name_cannot_ack(self):
        worker = FakeWorkerModel(self.head)
        def replaced(weights):
            worker.lm_head.weight = torch.nn.Parameter(self.head.clone())
            return {"lm_head.weight"}
        worker.load_weights = replaced
        with self.assertRaisesRegex(RuntimeError, "graph weight allocation"):
            load_head(worker, self.artifact)
        worker = FakeWorkerModel(self.head)
        worker.load_weights = lambda weights: {"other.weight"}
        with self.assertRaisesRegex(RuntimeError, "graph weight allocation"):
            load_head(worker, self.artifact)

    def test_exact_raw_action_ids_probability_alignment_and_eos(self):
        prompt = [1, 4, 9]
        rows = [{7: SimpleNamespace(logprob=-.3)}, {2: SimpleNamespace(logprob=-.2)}]
        choice = SimpleNamespace(token_ids=[7, 2], logprobs=rows, finish_reason="stop")
        output = SimpleNamespace(finished=True, prompt_token_ids=prompt, outputs=[choice])
        result = output_tokens(output, prompt, [2])
        self.assertEqual(result["action_token_ids"], [7, 2])
        self.assertEqual(result["behavior_log_probs"], [-.3, -.2])
        self.assertEqual(result["stop_token_count"], 1)
        self.assertEqual(result["finish_reason"], "eos")
        rows[0] = {4: SimpleNamespace(logprob=-.3)}
        with self.assertRaisesRegex(ValueError, "absent"):
            output_tokens(output, prompt, [2])
        output.prompt_token_ids = [1, 4]
        with self.assertRaisesRegex(ValueError, "prompt differs"):
            output_tokens(output, prompt, [2])

    def test_private_bounded_rpc_framing(self):
        left, right = socket.socketpair()
        self.addCleanup(left.close)
        self.addCleanup(right.close)
        value = {"operation": "stats", "token": "private-test-value"}
        send(left, value)
        self.assertEqual(receive(right), value)
        left.sendall((3 * 1024**2).to_bytes(4, "big"))
        with self.assertRaisesRegex(ValueError, "byte budget"):
            receive(right)

    def test_replay_observer_counts_successful_invocations_without_changing_return(self):
        import rollout_manager.backends.vllm_worker as module
        def original(graph):
            if graph == "fail":
                raise RuntimeError("failed replay fixture")
            return "original-result"
        with patch.object(module, "_GRAPH_OBSERVER_INSTALLED", False), patch.object(module, "_GRAPH_REPLAYS", 0), \
                patch.object(torch.cuda.CUDAGraph, "replay", original):
            module.install_graph_observer()
            self.assertEqual(torch.cuda.CUDAGraph.replay("fixture"), "original-result")
            self.assertEqual(module._GRAPH_REPLAYS, 1)
            with self.assertRaisesRegex(RuntimeError, "fixture"):
                torch.cuda.CUDAGraph.replay("fail")
            self.assertEqual(module._GRAPH_REPLAYS, 1)

    def test_v2_graph_proof_reads_actual_manager_objects_and_rejects_uncaptured_or_metadata(self):
        # CPU fixtures exercise store discovery; they do not qualify CUDA use.
        name = "vllm.v1.worker.gpu.cudagraph_utils"
        source = self.root / "graph-manager-fixture.py"
        source.write_text("fixture source provenance\n")
        graph_type = type("FixtureGraph", (), {})
        manager_type = type("ModelCudaGraphManager", (), {"__module__": name})
        runner_type = type("GPUModelRunner", (), {"__module__": "vllm.v1.worker.gpu.model_runner"})
        manager = manager_type()
        graph = graph_type()
        manager.graphs = {"first": graph, "same_object": graph}
        manager._graphs_captured = True
        runner = runner_type()
        runner.cudagraph_manager = manager
        worker = SimpleNamespace(model_runner=runner, use_v2_model_runner=True)
        module = ModuleType(name)
        module.ModelCudaGraphManager, module.__file__ = manager_type, str(source)
        with patch.dict(sys.modules, {name: module}), patch.object(torch.cuda, "CUDAGraph", graph_type):
            receipt = captured_graph_receipt(worker)
            self.assertEqual(receipt["model_runner_type"], "vllm.v1.worker.gpu.model_runner.GPUModelRunner")
            self.assertIs(receipt["use_v2_model_runner"], True)
            self.assertEqual(receipt["graph"]["captured_graph_count"], 1)
            self.assertEqual(receipt["graph"]["captured_graph_source"], "live_vllm_v2_CudaGraphManager_graphs")
            self.assertEqual(receipt["graph"]["captured_graph_source_module"], name)
            import hashlib
            self.assertEqual(receipt["graph"]["captured_graph_source_sha256"],
                             hashlib.sha256(source.read_bytes()).hexdigest())
            manager._graphs_captured = False
            with self.assertRaisesRegex(RuntimeError, "completed actual graph capture"):
                captured_graph_receipt(worker)
            manager._graphs_captured = True
            manager.graphs["metadata"] = SimpleNamespace(captured=True)
            with self.assertRaisesRegex(RuntimeError, "not an actual CUDA graph"):
                captured_graph_receipt(worker)
            del manager.graphs["metadata"]
            runner.cudagraph_manager = SimpleNamespace(graphs=manager.graphs, _graphs_captured=True)
            with self.assertRaisesRegex(RuntimeError, "completed actual graph capture"):
                captured_graph_receipt(worker)

    def test_v1_graph_proof_scopes_to_current_runner_not_unrelated_global_wrappers(self):
        name = "vllm.compilation.cuda_graph"
        source = self.root / "graph-wrapper-fixture.py"
        source.write_text("fixture source provenance\n")
        graph_type = type("FixtureGraph", (), {})
        wrapper_type = type("CUDAGraphWrapper", (), {"__module__": name})
        runner_type = type("GPUModelRunner", (), {"__module__": "vllm.v1.worker.gpu_model_runner"})
        wrapper = wrapper_type()
        wrapper.concrete_cudagraph_entries = {"captured": SimpleNamespace(cudagraph=graph_type()),
                                             "uncaptured": SimpleNamespace(cudagraph=None)}
        unrelated = wrapper_type()
        unrelated.concrete_cudagraph_entries = {"other": SimpleNamespace(cudagraph=graph_type())}
        wrapper_type._all_instances = [wrapper, unrelated]
        runner = runner_type()
        runner.model = wrapper
        worker = SimpleNamespace(model_runner=runner, use_v2_model_runner=False)
        module = ModuleType(name)
        module.CUDAGraphWrapper, module.__file__ = wrapper_type, str(source)
        with patch.dict(sys.modules, {name: module}), patch.object(torch.cuda, "CUDAGraph", graph_type):
            receipt = captured_graph_receipt(worker)
            self.assertEqual(receipt["graph"]["captured_graph_count"], 1)
            self.assertEqual(receipt["graph"]["captured_graph_source"], "live_vllm_CUDAGraphWrapper_entries")
            runner.model = SimpleNamespace(concrete_cudagraph_entries=wrapper.concrete_cudagraph_entries)
            with self.assertRaisesRegex(RuntimeError, "actual full graph wrapper"):
                captured_graph_receipt(worker)

    def test_worker_kernel_receipt_rejects_configuration_and_actual_expert_mismatch(self):
        expert_type = type("TritonExperts", (), {"__module__": TRITON_EXPERT_CLASS.rsplit(".", 1)[0]})
        method = SimpleNamespace(unquantized_backend=SimpleNamespace(value="TRITON"),
            experts_cls=expert_type, moe_kernel=SimpleNamespace(fused_experts=expert_type()))
        model = SimpleNamespace(config=SimpleNamespace(num_experts=8),
            named_modules=lambda: [("model.layers.0.mlp.experts", SimpleNamespace(quant_method=method))])
        worker = SimpleNamespace(vllm_config=SimpleNamespace(kernel_config=SimpleNamespace(**VLLM_KERNEL_CONFIG)),
            model_runner=SimpleNamespace(model=model))
        receipt = kernel_receipt(worker)
        self.assertEqual(receipt["kernel_config"], VLLM_KERNEL_CONFIG)
        self.assertEqual(receipt["moe_implementations"][0]["experts_class"], TRITON_EXPERT_CLASS)
        worker.vllm_config.kernel_config.enable_jit_warmup = True
        with self.assertRaisesRegex(RuntimeError, "kernel configuration"):
            kernel_receipt(worker)
        worker.vllm_config.kernel_config.enable_jit_warmup = False
        method.moe_kernel.fused_experts = object()
        with self.assertRaisesRegex(RuntimeError, "pinned Triton"):
            kernel_receipt(worker)

    def test_qwen_import_guard_keeps_stock_body_and_restores_modules_even_on_failure(self):
        name = "vllm.model_executor.warmup.minimax_m3_msa_warmup"
        worker = SimpleNamespace(vllm_config=SimpleNamespace(
            kernel_config=SimpleNamespace(**VLLM_KERNEL_CONFIG),
            model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="qwen3_moe"))),
            model_runner=SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(model_type="qwen3_moe"))))
        original_helper = SimpleNamespace(marker="preserved")
        observed = []
        def stock_body(same_worker, *, process_local_only=False):
            observed.append((same_worker, process_local_only, "kv_zeroing_and_other_stock_work"))
            if not process_local_only:
                helper = sys.modules[name].minimax_m3_msa_warmup
                self.assertIsNone(helper(same_worker))
            return "stock-return"
        with patch.dict(sys.modules, {name: original_helper}):
            result = guarded_qwen_kernel_warmup(stock_body, worker, process_local_only=True)
            self.assertEqual(result, "stock-return")
            self.assertEqual(observed, [(worker, True, "kv_zeroing_and_other_stock_work")])
            self.assertIs(sys.modules[name], original_helper)
            self.assertEqual(guarded_qwen_kernel_warmup(stock_body, worker), "stock-return")
            def broken_body(same_worker):
                sys.modules[name].minimax_m3_msa_warmup(same_worker)
                raise RuntimeError("original stock warmup failed")
            with self.assertRaisesRegex(RuntimeError, "stock warmup failed"):
                guarded_qwen_kernel_warmup(broken_body, worker)
            self.assertIs(sys.modules[name], original_helper)
            worker.vllm_config.model_config.hf_config.model_type = "minimax_m3"
            with self.assertRaisesRegex(RuntimeError, "exact Qwen3"):
                guarded_qwen_kernel_warmup(stock_body, worker)
            self.assertIs(sys.modules[name], original_helper)

    def test_actor_receipt_head_activation_and_generation_use_only_named_secure_rpc(self):
        worker = {"pid": 123, "start_ticks": 1, "cmdline_sha256": "a" * 64,
            "device_uuid": "fixture-only", "kernel_config": dict(VLLM_KERNEL_CONFIG),
            "startup_adapter": {"profile": "fixture-only"},
            "worker_rpc_method": VLLM_WORKER_RPC, "insecure_serialization": False,
            "graph": {"captured_graph_count": 1, "replay_calls": 0}}
        calls, sampling_params = [], []
        artifact = {"version": 0, "digest": "b" * 64, "lineage": "fixture"}
        class NamedRPC:
            def collective_rpc(self, method, *, args, timeout):
                self.assert_name(method)
                calls.append((method, args[0]))
                if args[0] == "stats":
                    return [copy.deepcopy(worker)]
                if args[0] == "load_head":
                    self.ack = {key: args[1][key] for key in ("version", "digest", "lineage")}
                    return [{"policy_ack": self.ack, "loaded_trainable_tensors": {"sha256": "c" * 64,
                        "parameter_names": [CANONICAL_HEAD]}, "shape": [32, 8], "dtype": "torch.bfloat16"}]
                raise AssertionError("unexpected RPC operation")
            def assert_name(self, method):
                if type(method) is not str or method != VLLM_WORKER_RPC:
                    raise TypeError("secure serializer rejects callable RPCs")
            def reset_prefix_cache(self):
                return True
            def update_weight_version(self, version):
                self.version = version
            def get_weight_version(self):
                return self.version
            def generate(self, prompt, params, use_tqdm):
                sampling_params.append(params)
                worker["graph"]["replay_calls"] += 1
                return [SimpleNamespace(finished=True, prompt_token_ids=prompt["prompt_token_ids"], metrics=None,
                    outputs=[SimpleNamespace(token_ids=[7, 2], logprobs=[{7: SimpleNamespace(logprob=-.4)},
                        {2: SimpleNamespace(logprob=-.2)}],
                        finish_reason="stop" if 2 in params.stop_token_ids else "length")])]
        engine = ActorEngine.__new__(ActorEngine)
        engine.llm = NamedRPC()
        engine.ack = engine.activation = None
        engine.bootstrap = {"kv_cache_memory_bytes": 32 * 1024**3, "kv_bytes_per_token": 98304,
            "max_model_len": 262144, "eos_token_ids": [2]}
        engine.startup_seconds, engine.source_hashes = .01, {"fixture": "d" * 64}
        fake_vllm = ModuleType("vllm")
        fake_vllm.SamplingParams = lambda **values: SimpleNamespace(**values)
        with patch.dict(sys.modules, {"vllm": fake_vllm}), patch(
                "rollout_manager.backends.vllm_actor.process_identity", return_value={"pid": 122}):
            activation = engine.load(artifact)
            self.assertEqual(activation["ack"], artifact)
            actual = engine.generate({"prompt_ids": [1, 4, 9], "max_tokens": 8,
                "temperature": .8, "seed": 731, "policy_ack": artifact})
            self.assertEqual(actual["action_token_ids"], [7, 2])
            self.assertEqual(actual["generation_metrics"]["graph_replay_calls_delta"], 1)
            self.assertFalse(actual["generation_metrics"]["engine_receipt"]["insecure_serialization"])
            self.assertTrue(sampling_params[-1].ignore_eos)
            self.assertEqual(sampling_params[-1].stop_token_ids, [2])
            engine.bootstrap["eos_token_ids"] = []
            empty_eos = engine.generate({"prompt_ids": [1, 4, 9], "max_tokens": 2,
                "temperature": .8, "seed": 731, "policy_ack": artifact})
            self.assertTrue(sampling_params[-1].ignore_eos)
            self.assertEqual(sampling_params[-1].stop_token_ids, [])
            self.assertEqual(empty_eos["finish_reason"], "length")
            self.assertEqual(empty_eos["stop_token_count"], 0)
        self.assertEqual([operation for _, operation in calls],
                         ["load_head", "stats", "stats", "stats", "stats", "stats"])


class FakeActor:
    instances = []
    def __init__(self, model_dir, config, directory, **kwargs):
        self.config = config
        self.counter = 0
        self.closed = False
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.instances.append(self)

    def call(self, operation, **values):
        if operation == "load":
            artifact = values["artifact"]
            head = checked_artifact(artifact)
            self.ack = {key: artifact[key] for key in ("version", "digest", "lineage")}
            self.activation = {"policy_ack": self.ack, "loaded_trainable_tensors": tensor_digest(head),
                               "shape": list(head.shape), "dtype": str(head.dtype), "storage_pointer_unchanged": True}
            self.engine = {"loaded_policy_ack": self.ack, "workers": [{"pid": 123,
                "start_ticks": 1, "cmdline_sha256": "a" * 64, "device": "cuda:0", "device_uuid": "fixture-only",
                "graph": {"captured_graph_count": 1, "replay_calls": self.counter}}],
                "actor_process": {"pid": 122, "start_ticks": 1, "cmdline_sha256": "b" * 64}}
            return {"ack": self.ack, "activation_receipt": self.activation, "engine": self.engine}
        if operation == "generate":
            self.entered.set()
            self.release.wait(2)
            before = self.counter
            self.counter += 1
            self.engine["workers"][0]["graph"]["replay_calls"] = self.counter
            return {"context_token_ids": values["prompt_ids"], "action_token_ids": [3, 2],
                    "behavior_log_probs": [-.4, -.3], "finish_reason": "eos", "stop_token_count": 1,
                    "logprob_convention": "sampled_distribution", "generation_metrics": {
                        "sampling_seed": values["seed"], "elapsed_seconds": .01,
                        "graph_replay_calls_before": before, "graph_replay_calls_after": self.counter,
                        "graph_replay_calls_delta": 1, "graph_captured_count": 1,
                        "engine_receipt": copy.deepcopy(self.engine)}}
        raise ValueError("unsupported fake operation")

    def close(self):
        self.closed = True


class VLLMControllerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(83)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        write_hf_checkpoint(self.root, Qwen3MoeForCausalLM(tiny_config(max_position_embeddings=128)).to(torch.bfloat16))
        vocabulary = {"[UNK]": 0, "[BOS]": 1, "[EOS]": 2, **{f"token{i}": i for i in range(3, 32)}}
        tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer.add_special_tokens(["[UNK]", "[BOS]", "[EOS]"])
        tokenizer.save(str(self.root / "tokenizer.json"))
        (self.root / "tokenizer_config.json").write_text('{"tokenizer_class":"PreTrainedTokenizerFast","unk_token":"[UNK]","bos_token":"[BOS]","eos_token":"[EOS]"}')
        (self.root / "chat_template.jinja").write_text(
            "{% for m in messages %}{{ m.role }} {{ m.content | default('', true) }} {% endfor %}assistant")
        (self.root / "data.jsonl").write_text('{"prompt":"fixture"}\n')
        self.config = RLConfig(max_new_tokens=8, temperature=.8, trainable_scope="lm_head",
            qwen_last_position_only=True, qwen_kv_cache=True, dtype="bfloat16", rollout_devices=("cuda:1",))
        self.learner = build_policy(self.root, load_model_config(self.root), random_init=False,
            devices=("cpu",), dtype=torch.bfloat16, trainable_scope="lm_head")
        self.identity = run_identity(self.root, self.root / "data.jsonl", False)
        self.identity.update(trainable_scope="lm_head", qwen_logits_path="last_position_head_v1", **vllm_policy_identities(128))
        self.controller = VLLMPolicyController(self.root, self.config, served_model="fixture/qwen")
        evidence = self.root / "evidence"
        evidence.mkdir(mode=0o700)
        if not self._testMethodName.startswith("test_http_"):
            self.controller.configure_diagnostics(evidence)
        self.actor_patch = patch("rollout_manager.backends.vllm_controller.OwnedVLLMActor", FakeActor)
        self.actor_patch.start()
        self.addCleanup(self.actor_patch.stop)
        self.addAsyncCleanup(self.controller.close)
        self.payload = {"model": "fixture/qwen", "messages": [{"role": "user", "content": "token3 token4"}],
            "tools": [{"type": "function", "function": {"name": "shell", "parameters": SHELL_PARAMETERS}}],
            "tool_choice": "auto", "parallel_tool_calls": False, "chat_template_kwargs": {"enable_thinking": False},
            "stream": False, "temperature": .8, "max_tokens": 8}

    def artifact(self, version):
        return publish_policy_artifact(self.learner, version, self.root / "policies", lineage="vllm-fixture",
            identity=self.identity, sampling={"temperature": .8, "top_p": 1, "top_k": 0})

    async def start_http_fixture(self):
        from rollout_manager.backends.native_qwen_loopback import NativeControllerHTTPShim, STATIC_PROOF_FIELDS
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        shim = NativeControllerHTTPShim(self.controller, self.root / "http-fixture", port=port)
        await self.controller.start(self.artifact(0))
        # Portable HTTP-handler fixture, not a GPU/process qualification.
        startup = {name: "HTTP FIXTURE" for name in STATIC_PROOF_FIELDS}
        startup["serving_profile"] = "native-pytorch-v1"
        with patch.object(shim, "profile", return_value=startup):
            shim.start()
        self.addCleanup(shim.close)
        return shim

    def http_generate(self, shim, identity):
        connection = http.client.HTTPConnection("127.0.0.1", shim.port, timeout=5)
        try:
            connection.request("POST", "/v1/chat/completions", json.dumps(self.payload).encode(),
                headers={"Authorization": "Bearer " + shim.token,
                    "X-Rollout-Request-Id": identity, "Content-Type": "application/json"})
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    async def test_actual_evidence_contract_identity_seed_eos_and_diagnostics(self):
        artifact = self.artifact(0)
        await self.controller.start(artifact)
        self.assertEqual(self.controller.capabilities()["serving_profile"], "vllm-training-v1")
        completion = self.controller.generate(self.payload, "a" * 32)
        contract = TrainingGenerationContract(self.controller.policy, self.controller.capabilities())
        trace = contract.validate(self.payload, completion, "a" * 32, 1)
        self.assertEqual(trace.action_token_ids, (3, 2))
        self.assertEqual(trace.stop_token_count, 1)
        self.assertEqual(self.controller.loaded_trainable_tensor_digest(), tensor_digest(self.learner.backend.lm_head.weight))
        diagnostic = json.loads((self.root / "evidence" / ("a" * 32 + ".sampled.json")).read_text())
        self.assertEqual(diagnostic["kind"], "vllm-actual-sampled-action")
        self.assertEqual(diagnostic["vllm_generation"]["graph_replay_calls_delta"], 1)
        self.assertEqual(diagnostic["vllm_generation"]["sampling_seed"], request_seed(self.config.seed, "a" * 32, self.controller.policy))
        with self.assertRaisesRegex(ValueError, "no replay"):
            self.controller.generate(self.payload, "a" * 32)
        actual = self.controller.sample_tokens([1, 4, 9], 8, .8, "b" * 32, ignore_eos=True)
        self.assertEqual(actual["response_ids"], [3, 2])
        self.assertEqual(actual["behavior_log_probs"], [-.4, -.3])
        diagnostic = self.controller.sample_tokens([1, 4, 9], 8, .8, "e" * 32,
            ignore_eos=True, diagnostic_seed=731)
        self.assertEqual(diagnostic["vllm_generation"]["sampling_seed"], 731)
        with self.assertRaisesRegex(ValueError, "no-tool"):
            self.controller.sample_tokens([1, 4, 9], 8, .8, "f" * 32, diagnostic_seed=731)

    async def test_pause_waits_real_generation_and_updates_only_after_drain(self):
        await self.controller.start(self.artifact(0))
        actor = self.controller.model
        actor.release.clear()
        generation = asyncio.create_task(asyncio.to_thread(self.controller.generate, self.payload, "c" * 32))
        await asyncio.to_thread(actor.entered.wait, 1)
        pause = asyncio.create_task(self.controller.pause())
        await asyncio.sleep(.02)
        self.assertFalse(pause.done())
        with self.assertRaisesRegex(RuntimeError, "proven paused"):
            # Same control lock would correctly queue; directly inspect guard
            # via an independent controller rather than deadlocking this pause.
            other = VLLMPolicyController(self.root, self.config, served_model="fixture/qwen")
            await other.activate(self.artifact(1))
        actor.release.set()
        await generation
        receipt = await pause
        self.assertTrue(receipt["drain_complete"])
        with torch.no_grad():
            self.learner.backend.lm_head.weight.add_(.01)
        updated = self.artifact(1)
        ack = await self.controller.activate(updated)
        self.assertEqual(ack["digest"], updated["digest"])
        self.assertEqual(self.controller.loaded_trainable_tensor_digest(), tensor_digest(self.learner.backend.lm_head.weight))
        await self.controller.resume()
        self.assertEqual(self.controller.policy.version, 1)

    async def test_completed_invalid_tools_preserve_actions_and_leave_actor_usable(self):
        await self.controller.start(self.artifact(0))
        head = self.controller.loaded_trainable_tensor_digest()
        texts = [
            '<tool_call>\n{"name":"shell","arguments":{"cmd":"pwd"}}',
            '<tool_call>{"name":"shell","arguments":{"cmd":"pwd"}}</tool_call><tool_call>{',
            '<tool_call>{bad JSON}</tool_call>',
            '<tool_call>{"name":"shell","arguments":{"cmd":"pwd","extra":1}}</tool_call>',
            '<tool_call>{"name":"other","arguments":{"cmd":"pwd"}}</tool_call>',
        ]
        for index, text in enumerate(texts):
            identity = f"{index + 10:032x}"
            with self.subTest(text=text), patch.object(self.controller.tokenizer, "decode", return_value=text):
                completion = self.controller.generate(self.payload, identity)
            self.assertEqual(completion["choices"][0]["message"], {"role": "assistant", "content": text})
            status = validate_model_completion_status(completion, identity)
            self.assertEqual(status["outcome"], "invalid_tool")
            trace = TrainingGenerationContract(self.controller.policy, self.controller.capabilities()).validate(
                self.payload, completion, identity, 1)
            self.assertEqual((trace.action_token_ids, trace.behavior_log_probs, trace.stop_token_count),
                             ((3, 2), (-.4, -.3), 1))
            self.assertEqual(trace.finish_reason, "eos")
            self.assertEqual(self.controller._state, "running")
            self.assertIsNone(self.controller._actor_failure)
            sampled = json.loads((self.root / "evidence" / (identity + ".sampled.json")).read_text())
            self.assertEqual(sampled["decoded_action_text"], text)
            self.assertEqual(sampled["action_token_ids"], [3, 2])
            tampered = copy.deepcopy(completion)
            tampered["model_completion_status"]["error_code"] = "invalid_tool_json"
            tampered["choices"][0]["message"]["content"] += " repaired"
            with self.assertRaisesRegex(ValueError, "raw completion"):
                TrainingGenerationContract(self.controller.policy, self.controller.capabilities()).validate(
                    self.payload, tampered, identity, 1)
        self.assertEqual(self.controller.loaded_trainable_tensor_digest(), head)
        self.assertTrue((await self.controller.pause())["drain_complete"])
        await self.controller.resume()
        self.controller.generate(self.payload, "f" * 32)

    async def test_known_failure_receipt_does_not_convert_infrastructure_to_model_outcome(self):
        await self.controller.start(self.artifact(0))
        identity = "6" * 32
        with patch("rollout_manager.backends.vllm_controller.shell_completion", side_effect=RuntimeError("parser implementation failed")):
            with self.assertRaisesRegex(RuntimeError, "implementation failed"):
                self.controller.generate(self.payload, identity)
        self.assertEqual(self.controller._state, "failed")
        receipt = self.controller.generation_failure_receipt(self.payload, identity, "vllm_shell_completion")
        validated = validate_generation_failure(receipt, self.payload, identity, self.controller.ack)
        self.assertTrue(failure_is_terminal(validated))
        self.assertEqual(receipt["sampler_outcome"], "completed")
        self.assertFalse((self.root / "evidence" / (identity + ".training.json")).exists())
        with self.assertRaises(ValueError):
            validate_generation_failure(receipt, {**self.payload, "max_tokens": 4}, identity, self.controller.ack)
        with self.assertRaises(RuntimeError):
            await self.controller.pause()

    async def test_http_known_failure_has_bound_request_closure_without_success(self):
        shim, identity = await self.start_http_fixture(), "4" * 32
        with patch("rollout_manager.backends.vllm_controller.shell_completion", side_effect=RuntimeError("implementation failure")):
            code, body = await asyncio.to_thread(self.http_generate, shim, identity)
        self.assertEqual(code, 500)
        receipt = validate_generation_failure(body["generation_failure"], self.payload, identity, self.controller.ack)
        self.assertTrue(failure_is_terminal(receipt))
        await asyncio.to_thread(shim.quiesce)
        state = json.loads((shim.output / (identity + ".http-state.json")).read_text())
        self.assertTrue(state["sampler_drained"])
        self.assertFalse(state["http_response_written"])
        self.assertEqual(state["generation_failure"], receipt)
        self.assertFalse((shim.output / (identity + ".training.json")).exists())

    async def test_http_uncertain_actor_rpc_cannot_claim_sampler_drain(self):
        shim, identity = await self.start_http_fixture(), "5" * 32
        with patch.object(self.controller.model, "call", side_effect=TimeoutError("unknown actor outcome")):
            code, body = await asyncio.to_thread(self.http_generate, shim, identity)
        self.assertEqual(code, 500)
        receipt = validate_generation_failure(body["generation_failure"], self.payload, identity, self.controller.ack)
        self.assertFalse(failure_is_terminal(receipt))
        await asyncio.to_thread(shim.quiesce)
        state = json.loads((shim.output / (identity + ".http-state.json")).read_text())
        self.assertFalse(state["sampler_drained"])
        self.assertEqual(state["generation_failure"], receipt)
        self.assertEqual(self.controller._state, "failed")

    async def test_wrong_serving_profile_rejected_and_partial_load_fails_closed(self):
        bad = self.artifact(0)
        bad = copy.deepcopy(bad)
        bad["identity"]["vllm_logprobs_mode"] = "raw_logprobs"
        with self.assertRaisesRegex(ValueError, "profiles"):
            await self.controller.start(bad)
        self.assertEqual(self.controller._state, "failed")
        with self.assertRaisesRegex(RuntimeError, "unqualified"):
            self.controller.generate(self.payload, "d" * 32)

    async def test_uncertain_rpc_outcome_cannot_claim_pause_drain_or_activate(self):
        await self.controller.start(self.artifact(0))
        actor = self.controller.model
        actor.release.clear()
        original = actor.call
        def uncertain(operation, **values):
            if operation == "generate":
                actor.entered.set()
                actor.release.wait(2)
                raise TimeoutError("fixture RPC outcome remains unknown")
            return original(operation, **values)
        actor.call = uncertain
        generation = asyncio.create_task(asyncio.to_thread(self.controller.generate, self.payload, "7" * 32))
        await asyncio.to_thread(actor.entered.wait, 1)
        pause = asyncio.create_task(self.controller.pause())
        await asyncio.sleep(.02)
        self.assertFalse(pause.done())
        actor.release.set()
        with self.assertRaises(TimeoutError):
            await generation
        with self.assertRaisesRegex(RuntimeError, "does not prove"):
            await pause
        self.assertEqual(self.controller._active, 0)  # Only the local call ended.
        self.assertEqual(self.controller._state, "failed")
        self.assertFalse(self.controller._drained)
        with self.assertRaises(RuntimeError):
            self.controller.capabilities()
        with self.assertRaises(RuntimeError):
            await self.controller.activate(self.artifact(1))
        with self.assertRaises(RuntimeError):
            await self.controller.resume()
        failure = json.loads((self.root / "evidence" / ("7" * 32 + ".actor-failure.json")).read_text())
        self.assertTrue(failure["actor_outcome_uncertain"])
        self.assertFalse(failure["actual_actor_drain_proven"])
        self.assertFalse(failure["automatic_replay"])
        receipt = self.controller.generation_failure_receipt(self.payload, "7" * 32, "vllm_sampling")
        self.assertEqual(receipt["sampler_outcome"], "unknown")
        self.assertFalse(failure_is_terminal(validate_generation_failure(receipt, self.payload, "7" * 32, self.controller.ack)))
        with self.assertRaises(ValueError):
            validate_generation_failure({**receipt, "sampler_drained": True}, self.payload, "7" * 32, self.controller.ack)


if __name__ == "__main__":
    unittest.main()
