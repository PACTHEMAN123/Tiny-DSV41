"""CPU integration of live branch adapters and real authenticated tensor RPC.

Only the GPU actor is substituted. Its eight receipts are simulated CPU
readbacks; this test does not qualify vLLM kernels, distributed GPUs, or GRPO.
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

from dsv41_train.lora import LoRAConfig, LoRALinear, adapter_parameters, inject_lora
from dsv41_train.rl.lora_export import export_q_a_update
from rollout_manager.backends.vllm_lora_controller import LoRARolloutConfig, VLLMLoRAController
from rollout_manager.backends.vllm_lora_remote import RemoteLoRAClient, RemoteLoRAServer
from rollout_manager.backends.vllm_lora_transport import policy_ack, unpack_update


def native_model():
    model = nn.Module()
    model.config = dict(hidden_size=16, q_lora_rank=8, head_dim=4, num_hidden_layers=1)
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([nn.Module()])
    layer = model.model.layers[0]
    layer.attention = nn.Module()
    layer.attention.q_a = nn.Linear(16, 8, bias=False, dtype=torch.bfloat16)
    inject_lora(model, LoRAConfig(rank=8, alpha=8, dropout=0, targets=("attention.q_a",)))
    return model


class LiveLoRATransportTests(unittest.TestCase):
    def test_optimizer_export_authenticated_activation_and_next_request_version(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            learner = native_model()
        serving = copy.deepcopy(learner)
        query = learner.model.layers[0].attention.q_a
        serving_query = serving.model.layers[0].attention.q_a
        self.assertIsInstance(query, LoRALinear)
        base_before = query.weight.detach().clone()
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
                            "policy_ack": ack,
                            "adapter_id": manifest["adapter_id"] if manifest else None,
                            "readback_digest": manifest["digest"] if manifest else None}
                           for rank in range(self.bootstrap["tensor_parallel_size"])]
                return {"serving_profile": "deepseek-lora-v1",
                        "base_identity": self.bootstrap["base_identity"],
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
                    prefix = "language_model.model.layers.0.attn.wq_a"
                    expected = {prefix + ".lora_A.weight": serving_query.lora_a,
                                prefix + ".lora_B.weight": serving_query.lora_b}
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
                    projection = serving_query(features).float()
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
            config = LoRARolloutConfig(operation_timeout_seconds=5)
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
                    self.assertEqual(client.initialize()["base_identity"], controller.base_identity)
                    packets = []
                    for version in range(2):
                        if version:
                            self.assertTrue(client.pause()["drain_complete"])
                            optimizer.zero_grad(set_to_none=True)
                            loss = -query(features).float()[0, 0]
                            loss.backward()
                            self.assertTrue(torch.isfinite(query.lora_b.grad).all())
                            self.assertGreater(float(query.lora_b.grad.abs().sum()), 0)
                            optimizer.step()
                            self.assertGreater(float(query.lora_b.detach().abs().sum()), 0)
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
                        self.assertTrue(torch.equal(query.weight, base_before))
                        self.assertIsNone(query.weight.grad)
                        self.assertTrue(torch.equal(serving_query.weight, base_before))
                    self.assertNotEqual(packets[0][0]["digest"], packets[1][0]["digest"])
                    initial = unpack_update(*packets[0])
                    self.assertEqual(float(initial["language_model.model.layers.0.attn.wq_a.lora_B.weight"].abs().sum()), 0)
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
