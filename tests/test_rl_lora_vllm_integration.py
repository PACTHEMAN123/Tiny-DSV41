"""CPU integration of live branch adapters and real authenticated tensor RPC.

Only the GPU actor is substituted. Its eight receipts are simulated CPU
readbacks. The serving fixture folds alpha/rank into BF16 B as vLLM does;
this test does not qualify vLLM kernels, distributed GPUs, or GRPO.
"""
import copy
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from dsv41_train.lora import LoRAConfig, LoRALinear, adapter_parameters
from dsv41_train.rl.lora_config import DistributedLoRAConfig
from dsv41_train.rl.lora_export import export_q_a_update
from rollout_manager.backends.vllm_lora_controller import LoRARolloutConfig, VLLMLoRAController
from rollout_manager.backends.vllm_lora_remote import RemoteLoRAClient, RemoteLoRAServer
from rollout_manager.backends.vllm_lora_transport import policy_ack, unpack_update


def native_model(profile, num_layers):
    model = nn.Module()
    model.config = dict(hidden_size=16, q_lora_rank=8, head_dim=4, num_hidden_layers=num_layers)
    model.distributed_lora_config = profile
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([nn.Module() for _ in range(num_layers)])
    config = LoRAConfig(rank=profile.rank, alpha=profile.alpha, dropout=0,
                        targets=("attention.q_a",))
    for index, layer in enumerate(model.model.layers):
        layer.attention = nn.Module()
        projection = nn.Linear(16, 8, bias=False, dtype=torch.bfloat16)
        projection.requires_grad_(False)
        layer.attention.q_a = (LoRALinear(projection, config)
                               if index in profile.selected_layers(num_layers) else projection)
    return model


class LiveLoRATransportTests(unittest.TestCase):
    def test_optimizer_export_authenticated_activation_and_next_request_version(self):
        self.run_update_transport(DistributedLoRAConfig(), num_layers=1)

    def test_custom_rank_scaling_and_selected_layers_reach_serving_model(self):
        self.run_update_transport(DistributedLoRAConfig(rank=16, alpha=32, layers=(0, 2)),
                                  num_layers=3)

    def run_update_transport(self, profile, *, num_layers):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            learner = native_model(profile, num_layers)
        serving = copy.deepcopy(learner)
        layers = profile.selected_layers(num_layers)
        queries = [learner.model.layers[index].attention.q_a for index in layers]
        base_before = [layer.attention.q_a.weight.detach().clone()
                       for layer in learner.model.layers]
        self.assertTrue(all(isinstance(query, LoRALinear) for query in queries))
        self.assertEqual(sum(parameter.numel() for parameter in adapter_parameters(learner)),
                         profile.parameter_count(16, 8, num_layers))
        self.assertEqual([index for index, layer in enumerate(learner.model.layers)
                          if isinstance(layer.attention.q_a, LoRALinear)], list(layers))
        features = torch.arange(16, dtype=torch.bfloat16).reshape(1, 16) / 16
        optimizer = torch.optim.AdamW(adapter_parameters(learner), lr=.1, weight_decay=0)

        class CPUActor:
            """Actual tensor copies/forward math, with explicitly fake TP receipts."""
            def __init__(self, model_dir, config, directory, *, max_model_len, bootstrap_override):
                self.bootstrap, self.manifest = bootstrap_override, None
                self.received, self.requests, self.projections = [], [], []
                self.receipt = self.stats()

            def stats(self):
                manifest = self.manifest
                ack = policy_ack(manifest) if manifest else None
                workers = [{"device_uuid": f"CPU-fixture-not-GPU-{rank}",
                            "base_identity": self.bootstrap["base_identity"],
                            "lora_config": self.bootstrap["lora_config"],
                            "policy_ack": ack,
                            "adapter_id": manifest["adapter_id"] if manifest else None,
                            "readback_digest": manifest["digest"] if manifest else None}
                           for rank in range(self.bootstrap["tensor_parallel_size"])]
                return {"serving_profile": "deepseek-lora-v1",
                        "base_identity": self.bootstrap["base_identity"],
                        "lora_config": self.bootstrap["lora_config"],
                        "tensor_parallel_size": len(workers), "workers": workers,
                        "loaded_policy_ack": ack,
                        "adapter_id": manifest["adapter_id"] if manifest else None}

            def call(self, operation, *, tensor_update=None, **values):
                if operation == "generate_batch":
                    # This CPU fixture tests tensor transport/math, not batching kernels.
                    results = []
                    for index, spec in enumerate(values["specs"]):
                        result = self.call("generate", **{key: values[key] for key in ("policy_ack", "adapter_id")},
                            prompt_ids=spec["prompt_ids"], max_tokens=spec["max_tokens"],
                            temperature=spec["temperature"], request_id=spec["request_id"], seed=spec["sampling_seed"])
                        result["generation_metrics"].update(request_id=spec["request_id"],
                            sampling_seed=spec["sampling_seed"], batch_request_index=index,
                            engine_request_id=str(index))
                        results.append(result)
                    return {"results": results, "engine_receipt": self.stats(),
                            "batch_id": values["batch_id"]}
                if operation == "load_lora":
                    manifest, payload = tensor_update
                    tensors = unpack_update(manifest, payload)
                    expected = {}
                    for index in layers:
                        query = serving.model.layers[index].attention.q_a
                        prefix = f"language_model.model.layers.{index}.attn.wq_a"
                        expected[prefix + ".lora_A.weight"] = query.lora_a
                        expected[prefix + ".lora_B.weight"] = query.lora_b
                    if tensors.keys() != expected.keys():
                        raise AssertionError("exported names do not match the worker contract")
                    with torch.no_grad():
                        for name, destination in expected.items():
                            destination.copy_(tensors[name])
                            if not torch.equal(destination, tensors[name]):
                                raise AssertionError("CPU tensor readback differs")
                    self.manifest = copy.deepcopy(manifest)
                    self.received.append((copy.deepcopy(manifest), payload))
                    engine = self.stats()
                    return {"ack": policy_ack(manifest), "adapter_id": manifest["adapter_id"],
                            "activation_receipts": engine["workers"], "engine": engine}
                if operation != "generate":
                    raise AssertionError(operation)
                if (values["policy_ack"] != policy_ack(self.manifest)
                        or values["adapter_id"] != self.manifest["adapter_id"]):
                    raise AssertionError("request did not pin the activated adapter")
                self.requests.append((values["request_id"], copy.deepcopy(values["policy_ack"]),
                                      values["adapter_id"]))
                with torch.no_grad():
                    scale = self.manifest["peft"]["lora_alpha"] / self.manifest["peft"]["r"]
                    projections = []
                    for index in layers:
                        query = serving.model.layers[index].attention.q_a
                        scaled_b = query.lora_b * scale
                        projections.append((F.linear(features, query.weight)
                            + F.linear(F.linear(features, query.lora_a), scaled_b)).float())
                    projection = torch.stack(projections).sum(0)
                    log_probs = (projection / values["temperature"]).log_softmax(-1)
                    token = int(log_probs.argmax(-1).item())
                self.projections.append(projection.clone())
                return {"context_token_ids": values["prompt_ids"], "action_token_ids": [token],
                        "behavior_log_probs": [float(log_probs[0, token])],
                        "finish_reason": "length", "stop_token_count": 0,
                        "logprob_convention": "sampled_distribution", "generation_metrics": {
                            "policy_ack": policy_ack(self.manifest),
                            "adapter_request_id": values["adapter_id"]}}

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.json").write_text(json.dumps({"model_type": "deepseek_v41",
                                                         "eos_token_id": 7, **learner.config}))
            (root / "model.safetensors.index.json").write_text("{}")
            config = LoRARolloutConfig(operation_timeout_seconds=5, lora_config=profile)
            controller = VLLMLoRAController(root, config, work_dir=root / "rollout")
            token = "cpu-integration-" + "a" * 32
            server = RemoteLoRAServer(("127.0.0.1", 0), token, controller, io_timeout=5)
            thread = threading.Thread(target=server.serve_forever,
                                      kwargs={"poll_interval": .01}, daemon=True)
            thread.start()
            client = RemoteLoRAClient(*server.server_address, token, timeout=5)
            try:
                with patch("rollout_manager.backends.vllm_lora_controller.OwnedVLLMActor", CPUActor), \
                        patch("torch.save", side_effect=AssertionError("checkpoint publication forbidden")), \
                        patch("torch.load", side_effect=AssertionError("checkpoint loading forbidden")):
                    initialized = (client.initialize() if profile == DistributedLoRAConfig()
                                   else client.initialize(lora_config=profile.to_dict()))
                    self.assertEqual(initialized["base_identity"], controller.base_identity)
                    self.assertEqual(initialized["lora_config"], profile.to_dict())
                    packets = []
                    for version in range(2):
                        if version:
                            self.assertTrue(client.pause()["drain_complete"])
                            optimizer.zero_grad(set_to_none=True)
                            loss = -sum(query(features).float()[0, 0] for query in queries)
                            loss.backward()
                            for query in queries:
                                self.assertTrue(torch.isfinite(query.lora_b.grad).all())
                                self.assertGreater(float(query.lora_b.grad.abs().sum()), 0)
                            optimizer.step()
                            self.assertTrue(all(float(query.lora_b.detach().abs().sum()) > 0
                                                for query in queries))
                        packet = export_q_a_update(learner, optimizer=optimizer,
                            version=version, lineage="cpu-integration", adapter_id=version + 1,
                            base_identity=client.base_identity)
                        packets.append(packet)
                        manifest, payload = packet
                        receipt = client.activate(manifest, payload)
                        self.assertEqual(receipt["ack"], policy_ack(manifest))
                        self.assertEqual(len(receipt["activation_receipts"]), 8)
                        self.assertTrue(all(worker["readback_digest"] == manifest["digest"]
                                            for worker in receipt["activation_receipts"]))
                        self.assertEqual(client.resume(), policy_ack(manifest))
                        result, = client.sample_group([{"prompt_ids": [1, 2], "max_tokens": 1,
                            "temperature": .8, "request_id": f"{version + 1:032x}",
                            "sampling_seed": 42 + version}])
                        self.assertEqual(result["generation_metrics"]["policy_ack"], policy_ack(manifest))
                        self.assertEqual(result["generation_metrics"]["adapter_request_id"], version + 1)
                        native_projection = torch.stack([query(features).float() for query in queries]).sum(0)
                        self.assertTrue(torch.equal(native_projection, controller.actor.projections[-1]))
                        for index, before in enumerate(base_before):
                            query = learner.model.layers[index].attention.q_a
                            self.assertTrue(torch.equal(query.weight, before))
                            self.assertIsNone(query.weight.grad)
                            self.assertFalse(query.weight.requires_grad)
                            self.assertTrue(torch.equal(serving.model.layers[index].attention.q_a.weight, before))
                    self.assertNotEqual(packets[0][0]["digest"], packets[1][0]["digest"])
                    initial = unpack_update(*packets[0])
                    self.assertEqual(len(initial), 2 * len(layers))
                    self.assertTrue(all(float(value.abs().sum()) == 0 for name, value in initial.items()
                                        if name.endswith(".lora_B.weight")))
                    self.assertEqual(controller.actor.received, packets)
                    self.assertEqual([request[1]["version"] for request in controller.actor.requests], [0, 1])
                    self.assertFalse(torch.equal(*controller.actor.projections))
                    self.assertFalse(torch.equal(controller.actor.projections[0].log_softmax(-1),
                                                 controller.actor.projections[1].log_softmax(-1)))
                    self.assertTrue(client.close()["drain_complete"])
            finally:
                server.shutdown()
                thread.join(timeout=5)
                server.server_close()
                self.assertFalse(thread.is_alive())


if __name__ == "__main__":
    unittest.main()
