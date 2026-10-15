"""CPU/browser closure fixtures, without an actor, sandbox or live services."""
import asyncio
from enum import Enum
import json
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from rollout_manager.backends.terminal_bench import TerminalBenchBackend, _identity_key
from rollout_manager.contracts import CancellationToken, request_uncertain
from rollout_manager.generation_training import TrainingGenerationContract, evidence_digest
from rollout_manager.model_completion import invalid_tool_status, ModelCompletionFormatError
from rollout_manager.terminal_bench import controller
from rollout_manager.terminal_bench.browser_relay import relay_server as relay
from rollout_manager.transports import FileRelayTransport
from . import test_harbor_filerelay_agent as fixtures
from .test_classification import trial
from .test_training_profile import plans, profile
from .test_browser_relay_vllm_training import vllm_proof, PROFILE, MODEL

a = fixtures.a


def failure(payload, rid, outcome="completed", ack=None):
    return {"schema_version": 1, "request_id": rid, "request_payload_sha256": evidence_digest(payload),
        "policy_ack": ack or {"version": 2, "digest": "a" * 64, "lineage": "fixture"},
        "stage": "vllm_shell_completion", "sampler_outcome": outcome,
        "sampler_drained": outcome != "unknown", "actual_actor_drain_proven": outcome != "unknown",
        "drain_scope": "request", "proof_kind": "none" if outcome == "unknown" else
        "controller_rejected_before_sampler_dispatch" if outcome == "not_admitted" else
        "owned_vllm_final_request_output", "automatic_replay": False}


def malformed_completion(payload, rid, contract):
    text = '<tool_call>\n{"name":"shell","arguments":{"cmd":"touch /app/DO_NOT_EXECUTE"}}'
    result = {"model": payload["model"], "choices": [{"finish_reason": "stop",
        "message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 3},
        "model_completion_status": invalid_tool_status(text, rid,
            ModelCompletionFormatError("incomplete fixture", "incomplete_tool_envelope"))}
    result["training_evidence"] = {"schema_version": 1, "request_id": rid,
        "request_payload_sha256": evidence_digest(payload), "response_sha256": evidence_digest(result),
        "policy": contract.policy.to_dict(), "logprob_convention": "sampled_distribution",
        "context_token_ids": [10, 11, 12], "action_token_ids": [21, 22, 2],
        "behavior_log_probs": [-.1, -.2, -.3], "finish_reason": "eos", "stop_token_count": 1,
        "context_truncation": "none"}
    return result


class InvalidToolTests(fixtures.RelayFixture):
    make_agent = fixtures.AgentTests.make_agent
    async def test_invalid_tool_preserves_training_tokens_and_never_executes_fragments(self):
        settings = profile(self.root)
        contract = TrainingGenerationContract(settings["training_policy"], settings["training_capabilities"])
        transport = FileRelayTransport(self.spool, training_contract=contract)
        with patch.dict("sys.modules", {"rollout_manager.terminal_bench.harbor_filerelay_agent": a}):
            token = a.relay_factory.set(transport.harbor_relay)
            try:
                agent, environment, context = await self.make_agent(temperature=1)
            finally:
                a.relay_factory.reset(token)
        task = asyncio.create_task(agent.run("public instruction", environment, context))
        published = await self.published()
        request = json.loads(published.read_text())
        response = malformed_completion(request["payload"], request["id"], contract)
        a.write_new(self.spool / "outbox" / published.name,
                    a.json_bytes({"id": request["id"], "ok": True, "response": response}))
        with patch.dict("sys.modules", {"rollout_manager.terminal_bench.harbor_filerelay_agent": a}):
            await task
        environment.exec.assert_not_awaited()
        metadata = context.metadata["file_relay"]
        self.assertEqual((metadata["stop_reason"], metadata["turns"], metadata["error_type"]),
                         ("model_invalid_tool", 1, None))
        self.assertTrue(metadata["verification_owned_by_harbor"])
        self.assertEqual((context.n_input_tokens, context.n_output_tokens), (3, 3))
        trace = transport.training_traces[0]
        self.assertEqual((list(trace.action_token_ids), list(trace.behavior_log_probs),
                          trace.finish_reason, trace.stop_token_count), ([21, 22, 2], [-.1, -.2, -.3], "eos", 1))
        saved = json.loads((agent._private_logs / (request["id"] + ".training.json")).read_text())
        self.assertEqual(saved, trace.to_dict())
        trajectory = json.loads((agent._private_logs / "trajectory.json").read_text())
        self.assertEqual(trajectory[-1], response["choices"][0]["message"])
        self.assertEqual(len(list((self.spool / "inbox").glob("*.json"))), 1)

    async def test_tampered_invalid_tool_marker_is_infrastructure_and_executes_nothing(self):
        for field, value in (("request_id", "e" * 32), ("decoded_action_text_sha256", "f" * 64),
                             ("tools_executable", True)):
            with self.subTest(field=field):
                agent, environment, context = await self.make_agent()
                rid = "d" * 32
                response = {"model": "model008", "choices": [{"finish_reason": "stop", "message": {
                    "role": "assistant", "content": "<tool_call> malformed"}}]}
                response["model_completion_status"] = invalid_tool_status("<tool_call> malformed", rid,
                    ModelCompletionFormatError("fixture", "incomplete_tool_envelope"))
                response["model_completion_status"][field] = value
                agent._relay.generate = AsyncMock(return_value=response)
                agent._relay.request_ids.append(rid)
                with self.assertRaises(ValueError):
                    await agent.run("instruction", environment, context)
                environment.exec.assert_not_awaited()
                self.assertEqual(context.metadata["file_relay"]["stop_reason"], "infrastructure_error")
                # Each native attempt owns a fresh log directory.
                self.output = self.root / ("logs-" + field)
                self.output.mkdir(mode=0o700)


class FailureStateTests(fixtures.RelayFixture):
    async def test_only_exact_request_and_leased_policy_closure_releases_uncertainty(self):
        ack = {"version": 2, "digest": "a" * 64, "lineage": "fixture"}
        payload = {"model": "model008"}
        for index, mutation in enumerate(("completed", "not_admitted", "unknown", "untyped",
                "request", "payload", "policy", "drain")):
            with self.subTest(mutation=mutation):
                rid = f"{index + 1:032x}"
                transport = a.AsyncFileRelay(self.spool, self.output, expected_policy_ack=ack)
                task = asyncio.create_task(transport.generate(payload, 1, request_id=rid))
                async with asyncio.timeout(1):
                    while not (self.spool / "inbox" / (rid + ".json")).exists():
                        await asyncio.sleep(.001)
                receipt = failure(payload, rid, mutation if mutation in {"unknown", "not_admitted"} else "completed")
                if mutation == "request": receipt["request_id"] = "e" * 32
                if mutation == "payload": receipt["request_payload_sha256"] = "f" * 64
                if mutation == "policy": receipt["policy_ack"]["version"] += 1
                if mutation == "drain": receipt["actual_actor_drain_proven"] = False
                negative = {"id": rid, "ok": False, "error": "original actor error"}
                if mutation != "untyped": negative["generation_failure"] = receipt
                a.write_new(self.spool / "outbox" / (rid + ".json"), a.json_bytes(negative))
                with self.assertRaises(a.RelayError): await task
                state = json.loads((self.output / (rid + ".state.json")).read_text())
                known = mutation in {"completed", "not_admitted"}
                self.assertEqual(state["state"], "terminal_failure_received" if known else "published_outcome_unknown")
                self.assertEqual(request_uncertain(state), not known)
                self.assertFalse(state["replayed"])
                self.assertFalse(state["remote_cancellation_proven"])
                self.assertEqual(json.loads((self.output / (rid + ".response.json")).read_text()), negative)
                self.assertFalse((self.output / (rid + ".training.json")).exists())


class FailureRelayTests(unittest.TestCase):
    def test_http_negative_receipt_survives_gpu_poll_and_cpu_commit_without_becoming_a_reward(self):
        for outcome in ("completed", "unknown", "untyped", "cancelled"):
            with self.subTest(outcome=outcome), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp).resolve()
                token = root / "token"
                token.write_text("f" * 64)
                token.chmod(0o600)
                with patch.object(relay, "owned_training_controller", return_value=vllm_proof()):
                    gpu = relay.RelayState(root / "gpu", "gpu", "https://gpu.test/", "https://cpu.test",
                        "http://127.0.0.1:8002", MODEL, serving_profile=PROFILE,
                        training_controller_receipt=root / "receipt.json", training_controller_token_file=token)
                rid = "1" * 32
                payload = {"model": MODEL, "messages": [{"role": "user", "content": "public"}],
                           "temperature": 1, "max_tokens": 32}
                receipt = failure(payload, rid, "unknown" if outcome == "unknown" else "completed")
                body = {"error": "fixture"}
                if outcome != "untyped": body["generation_failure"] = receipt
                connection = Mock()
                response = connection.getresponse.return_value
                response.status, response.length = 503, 0
                response.read.return_value = json.dumps(body).encode()
                if outcome == "cancelled":
                    def read_after_cancel(_limit):
                        gpu.cancel_generation(rid)
                        return json.dumps(body).encode()
                    response.read.side_effect = read_after_cancel
                with patch.object(relay, "open_upstream_connection", return_value=connection):
                    gpu.submit_generation({"id": rid, "payload": payload, "timeout_ms": 10000})
                    entry = gpu.completion_entry(rid)
                    self.assertTrue(entry["event"].wait(2))
                negative = entry["value"]
                self.assertFalse(negative["ok"])
                self.assertNotIn("response", negative)
                self.assertEqual(gpu.gpu_journal[rid]["status"], "failed" if outcome == "completed" else "uncertain")
                cpu = relay.RelayState(root / "cpu", "cpu", "https://cpu.test/", "https://gpu.test",
                    "http://127.0.0.1:8002", MODEL, serving_profile=PROFILE)
                relay.atomic_write(cpu.root / "inbox" / (rid + ".json"), {"id": rid, "operation": "generate",
                    "payload": payload, "deadline_unix": __import__("time").time() + 60})
                cpu.claim(rid)
                commit = {key: value for key, value in negative.items() if key != "pending"}
                cpu.result(commit)
                self.assertEqual(relay.read_json(cpu.root / "outbox" / (rid + ".json")), commit)
                if outcome in {"completed", "unknown"}: self.assertEqual(commit["generation_failure"], receipt)
                else: self.assertNotIn("generation_failure", commit)
                gpu.acknowledge_delivery(rid)
                self.assertEqual(gpu.status()["uncertain"], int(outcome != "completed"))
                if outcome != "completed":
                    with self.assertRaises(relay.CapacityError):
                        gpu.admit_generation("2" * 32, payload, 10000)

    def test_cpu_rejects_receipt_from_another_payload_before_outbox_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            cpu = relay.RelayState(Path(tmp), "cpu", "https://cpu.test/", "https://gpu.test",
                                  "http://127.0.0.1:8002", MODEL, serving_profile=PROFILE)
            rid = "1" * 32
            payload = {"model": MODEL, "messages": [{"role": "user", "content": "public"}], "max_tokens": 32}
            relay.atomic_write(cpu.root / "inbox" / (rid + ".json"), {"id": rid, "operation": "generate",
                "payload": payload, "deadline_unix": __import__("time").time() + 60})
            cpu.claim(rid)
            receipt = failure({**payload, "max_tokens": 64}, rid)
            with self.assertRaises(ValueError):
                cpu.result({"id": rid, "ok": False, "error": "fixture", "generation_failure": receipt})
            self.assertFalse((cpu.root / "outbox" / (rid + ".json")).exists())


class BackendClosureTests(unittest.IsolatedAsyncioTestCase):
    async def test_original_verifier_zero_remains_eligible_after_invalid_tool_and_transport_error_never_does(self):
        for mode in ("invalid_tool", "closed_transport", "tampered_transport"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                settings = profile(root)
                sample = plans(root)[0]
                contract = TrainingGenerationContract(settings["training_policy"], settings["training_capabilities"])
                rid = "d" * 32
                payload = {"model": "model008", "temperature": 1, "max_tokens": 32}
                completed = malformed_completion(payload, rid, contract)
                trace = contract.validate(payload, completed, rid, 1)
                class Transport:
                    training_traces = [trace] if mode == "invalid_tool" else []
                    def with_training(self, _contract): return self
                    def harbor_relay(self, _output): return None
                backend = TerminalBenchBackend(settings, Transport())
                backend._admitted[_identity_key(sample.identity)] = {}
                native = root / ("episode-" + _identity_key(sample.identity)) / "jobs/episode/trial"
                class Job:
                    create = AsyncMock()
                    def add_hook(self, *_): pass
                    async def run(self):
                        native.mkdir(parents=True)
                        trial(native, name=sample.task.task_id, failed=True,
                              exception=None if mode == "invalid_tool" else "RelayError")
                        agent = controller.read_json(native / "agent/file-relay/agent-result.json")
                        agent.update(turns=1, stop_reason="model_invalid_tool" if mode == "invalid_tool" else "error",
                                     error_type=None if mode == "invalid_tool" else "RelayError")
                        (native / "agent/file-relay/agent-result.json").write_text(json.dumps(agent))
                        if mode == "invalid_tool":
                            state = {"id": rid, "state": "response_received"}
                            controller.write_new(native / "agent/file-relay" / (rid + ".training.json"), trace.to_dict())
                        else:
                            receipt = failure(payload, rid)
                            if mode == "tampered_transport": receipt["policy_ack"]["version"] += 1
                            state = {"id": rid, "state": "terminal_failure_received", "generation_failure": "validated",
                                "failure_receipt": receipt, "request_payload_sha256": receipt["request_payload_sha256"]}
                            controller.write_new(native / "agent/file-relay" / (rid + ".request.json"),
                                                 {"id": rid, "operation": "generate", "payload": payload})
                        controller.write_new(native / "agent/file-relay" / (rid + ".state.json"), state)
                Job.create.return_value = Job()
                job_module, hook_module = ModuleType("harbor.job"), ModuleType("harbor.trial.hooks")
                job_module.Job = Job
                hook_module.TrialEvent = Enum("TrialEvent", {"START": "start"})
                modules = {"harbor.job": job_module, "harbor.trial.hooks": hook_module,
                    "rollout_manager.terminal_bench.harbor_filerelay_agent": a}
                config = SimpleNamespace(model_dump=lambda **_: {})
                with patch.dict("sys.modules", modules), patch.object(backend, "_checked_plan", return_value={}), \
                        patch.object(backend, "build_config", return_value=config):
                    result = await backend.execute(sample, CancellationToken())
                self.assertEqual(result.native_outcome["official_reward"], 0)
                self.assertEqual(result.score_eligible, mode == "invalid_tool")
                self.assertEqual(result.cleanup_state, "verified")
                self.assertEqual(request_uncertain(result.request_state), mode == "tampered_transport")
                if mode != "invalid_tool":
                    self.assertEqual(result.execution_state, "infrastructure_error")
                    self.assertIsNone(result.native_outcome["validated_reported_reward"])
                original = controller.read_json(native / "agent/file-relay" / (rid + ".state.json"))
                self.assertNotEqual(original["state"], "terminal_failure_closure_unknown")
