"""Offline serving receipts: no network, model, or benchmark execution."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rollout_manager.contracts import EpisodeIdentity
from rollout_manager.generation_training import TrainingCapabilityError, TrainingGenerationContract, evidence_digest
from rollout_manager.training import PolicyRef
from rollout_manager.transports import FileRelayTransport, RpcTransport


def training_contract():
    policy = PolicyRef(3, "a" * 64, "fixture-lineage", identities={
        "tokenizer_sha256": "b" * 64, "chat_template_sha256": "c" * 64})
    receipt = {"schema_version": 1, "policy": policy.to_dict(), "logprob_convention": "sampled_distribution",
               **{name: True for name in TrainingGenerationContract.REQUIRED}}
    return TrainingGenerationContract(policy, receipt)


def request():
    return {"model": "fixture-model", "messages": [{"role": "user", "content": "public task"}],
            "temperature": 1, "max_tokens": 16}


def completion(payload, request_id, contract, finish="stop"):
    response = {"model": payload["model"], "choices": [{"finish_reason": finish,
        "message": {"role": "assistant", "content": "answer"}}]}
    response["training_evidence"] = {"schema_version": 1, "request_id": request_id,
        "request_payload_sha256": evidence_digest(payload), "response_sha256": evidence_digest(response),
        "policy": contract.policy.to_dict(), "logprob_convention": "sampled_distribution",
        "context_token_ids": [10, 20], "action_token_ids": [30, 40],
        "behavior_log_probs": [-0.4, -0.9], "finish_reason": finish,
        "stop_token_count": 0, "context_truncation": "none"}
    return response


class TrainingEvidenceTests(unittest.TestCase):
    def test_capabilities_and_tokenizer_identity_are_required_before_dispatch(self):
        contract = training_contract()
        for key in TrainingGenerationContract.REQUIRED:
            bad = dict(contract.capabilities, **{key: False})
            with self.assertRaises(TrainingCapabilityError):
                TrainingGenerationContract(contract.policy, bad)
        with self.assertRaises(TrainingCapabilityError):
            TrainingGenerationContract(None, {})
        policy = PolicyRef(3, "a" * 64, "fixture-lineage")
        with self.assertRaisesRegex(TrainingCapabilityError, "tokenizer"):
            TrainingGenerationContract(policy, dict(contract.capabilities, policy=policy.to_dict()))

    def test_raw_tokens_policy_and_binding_are_checked_without_text_retokenization(self):
        contract, payload = training_contract(), request()
        raw = completion(payload, "unique", contract, "tool_calls")
        trace = contract.validate(payload, raw, "unique", 1)
        self.assertEqual(tuple(trace.action_token_ids), (30, 40))
        self.assertEqual(trace.finish_reason, "tool_calls")
        for mutation in ("request", "response", "policy", "logprobs", "tokens"):
            bad = copy.deepcopy(raw)
            if mutation == "request":
                bad["training_evidence"]["request_id"] = "wrong"
            elif mutation == "response":
                bad["choices"][0]["message"]["content"] = "changed"
            elif mutation == "policy":
                bad["training_evidence"]["policy"]["version"] = 4
            elif mutation == "logprobs":
                bad["training_evidence"]["behavior_log_probs"] = [-0.4]
            else:
                bad["training_evidence"]["action_token_ids"] = [True, 40]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                contract.validate(payload, bad, "unique", 1)
        with self.assertRaisesRegex(TrainingCapabilityError, "training_evidence"):
            contract.validate(payload, {"choices": []}, "unique", 1)

    def test_sampling_change_cannot_silently_reuse_behavior_contract(self):
        contract = training_contract()
        for changed in ({"temperature": 0}, {"top_p": .9}, {"top_k": 1}):
            with self.assertRaises(TrainingCapabilityError):
                contract.check_payload(dict(request(), **changed))


class TrainingRpcTests(unittest.IsolatedAsyncioTestCase):
    async def test_bound_native_episode_routes_feedback_turns_and_blocks_late_dispatch(self):
        contract, calls = training_contract(), []
        class Client:
            def __init__(self, *args, **kwargs): pass
            def close(self): pass
            def call(self, path, payload):
                calls.append(path)
                return {"response": completion(payload, path.removeprefix("/generate/"), contract)}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            transport = RpcTransport("https://example.test", root / "creds", root / "ledger", client_type=Client)
            identity = EpisodeIdentity("run", "kernelbench", "task", "group", 0, trial_id="trial")
            transport.bind_episode(identity, contract)
            client = transport.client_factory(transport.url, transport.credentials)
            client.call("/generate/trial-t1", request())
            client.call("/generate/trial-t2", request())
            traces = transport.episode_traces(identity, close=True)
            self.assertEqual([t.turn_id for t in traces], [1, 2])
            self.assertEqual([t.request_id for t in traces], ["trial-t1", "trial-t2"])
            with self.assertRaisesRegex(ValueError, "late dispatch"):
                client.call("/generate/trial-t3", request())
            with self.assertRaisesRegex(ValueError, "binding"):
                client.call("/generate/unadmitted", request())
            with self.assertRaisesRegex(ValueError, "replay"):
                transport.bind_episode(identity, contract)
            self.assertEqual(len(calls), 2)

    async def test_native_factory_captures_before_candidate_parse_and_never_replays(self):
        contract, calls, traces = training_contract(), [], []
        class Client:
            def __init__(self, *args, **kwargs): pass
            def close(self): pass
            def call(self, path, payload):
                calls.append(path)
                return {"response": completion(payload, path.removeprefix("/generate/"), contract)}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = RpcTransport("https://example.test", root / "creds", root / "ledger", client_type=Client)
            transport = base.with_training(contract, traces.append)
            response = await transport.submit({"request_id": "unique", "payload": request()})
            self.assertEqual(response["response"]["choices"][0]["message"]["content"], "answer")
            self.assertEqual([t.request_id for t in traces], ["unique"])
            state = json.loads((root / "ledger/unique.json").read_text())
            self.assertEqual(state["training_evidence"], "validated")
            with self.assertRaises(FileExistsError):
                await transport.submit({"request_id": "unique", "payload": request()})
            self.assertEqual(calls, ["/generate/unique"])

    async def test_evidence_failure_is_known_response_and_not_replayed_or_scored(self):
        calls = []
        class Client:
            def __init__(self, *args, **kwargs): pass
            def close(self): pass
            def call(self, path, payload):
                calls.append(path)
                return {"response": {"model": payload["model"], "choices": []}}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            transport = RpcTransport("https://example.test", root / "creds", root / "ledger",
                client_type=Client).with_training(training_contract())
            with self.assertRaises(TrainingCapabilityError):
                await transport.submit({"request_id": "unique", "payload": request()})
            state = json.loads((root / "ledger/unique.json").read_text())
            self.assertEqual((state["state"], state["uncertain"], state["training_evidence"]),
                             ("response_received", False, "rejected"))
            with self.assertRaises(FileExistsError):
                await transport.submit({"request_id": "unique", "payload": request()})
            self.assertEqual(len(calls), 1)


@unittest.skipUnless(importlib.util.find_spec("harbor"), "optional Harbor contract environment unavailable")
class TrainingFileRelayTests(unittest.IsolatedAsyncioTestCase):
    async def test_browser_hook_captures_actual_tokens_before_harbor_action_parsing(self):
        contract = training_contract()
        identity = "d" * 32
        expected_acks = []
        class Relay:
            request_ids = []
            def __init__(self, *args, expected_policy_ack):
                expected_acks.append(expected_policy_ack)
            async def generate(self, payload, timeout, request_id=None):
                self.request_ids.append(request_id or identity)
                return completion(payload, self.request_ids[-1], contract)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            transport = FileRelayTransport(root).with_training(contract)
            with patch("rollout_manager.terminal_bench.harbor_filerelay_agent.AsyncFileRelay", Relay):
                relay = transport.harbor_relay(root)
                await relay.generate(request(), 360, request_id=identity)
            self.assertEqual(expected_acks, [{key: getattr(contract.policy, key)
                                             for key in ("version", "digest", "lineage")}])
            self.assertEqual(transport.training_traces[0].request_id, identity)
            saved = json.loads((root / (identity + ".training.json")).read_text())
            self.assertEqual(saved["action_token_ids"], [30, 40])
