"""Adapters for the existing TLS and browser spool routes, without replay."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import threading
import time
import uuid
from pathlib import Path


def validate_rpc_origin(url):
    from urllib.parse import urlsplit
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or
            parsed.path not in ("", "/") or parsed.query or parsed.fragment):
        raise ValueError("TLS RPC requires an HTTPS origin without embedded credentials, path, query or fragment")


def _atomic(path, value):
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex)
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, allow_nan=False)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


class RequestLedger:
    """Exclusive request claims survive restarts; terminal or unknown IDs cannot replay."""
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def claim(self, request_id, route, payload, deadline_seconds):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", request_id):
            raise ValueError("invalid durable request ID")
        state = {"request_id": request_id, "route": route, "state": "claimed",
                 "payload_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
                 "deadline_unix": time.time() + deadline_seconds, "automatic_replay": False}
        with self._lock:
            # A separate exclusive claim persists even if process dies before state publication.
            with (self.root / (request_id + ".claim")).open("x") as stream:
                stream.write(request_id); stream.flush(); os.fsync(stream.fileno())
            _atomic(self.root / (request_id + ".json"), state)
        return state

    def update(self, record, **changes):
        record.update(changes)
        with self._lock:
            _atomic(self.root / (record["request_id"] + ".json"), record)


class RpcTransport:
    def __init__(self, url, credentials, ledger_root, *, timeout=900, client_type=None,
                 training_contract=None, trace_sink=None):
        validate_rpc_origin(url)
        self.url, self.credentials = url, Path(credentials)
        self.ledger_root, self.timeout, self.client_type = Path(ledger_root), timeout, client_type
        self._ledger = None
        self._lock = threading.Lock()
        self.training_contract, self.trace_sink = training_contract, trace_sink
        self.training_traces = []
        self._trace_lock = threading.Lock()
        self._episode_bindings = {}

    def with_training(self, contract, trace_sink=None):
        """Bind a policy lease without changing the selected RPC route."""
        return type(self)(self.url, self.credentials, self.ledger_root, timeout=self.timeout,
            client_type=self.client_type, training_contract=contract, trace_sink=trace_sink)

    def _capture(self, trace):
        with self._trace_lock:
            self.training_traces.append(trace)
        if self.trace_sink is not None:
            self.trace_sink(trace)

    def bind_episode(self, identity, contract=None, trace_sink=None):
        """Bind native job and ``job-tN`` request IDs before worker dispatch.

        Explicit bindings cross the native thread boundary without relying on
        ContextVar inheritance. Closing a binding never permits its replay.
        """
        key = identity.trial_id
        if not key or re.fullmatch(r"[A-Za-z0-9_-]{1,100}", key) is None:
            raise ValueError("Training episode requires a durable native trial ID")
        contract = contract or self.training_contract
        if contract is None:
            raise ValueError("Training episode requires a negotiated generation contract")
        with self._trace_lock:
            if key in self._episode_bindings:
                raise ValueError("Native training episode binding cannot replay")
            self._episode_bindings[key] = {"contract": contract, "trace_sink": trace_sink,
                "traces": [], "closed": False, "next_turn": 1, "identity": identity}

    def episode_traces(self, identity, *, close=False):
        with self._trace_lock:
            binding = self._episode_bindings[identity.trial_id]
            if binding["identity"] != identity:
                raise ValueError("Native training episode identity differs")
            if close:
                binding["closed"] = True
            return tuple(binding["traces"])

    def _binding(self, request_id):
        with self._trace_lock:
            binding = self._episode_bindings.get(request_id)
            if binding is None:
                match = re.fullmatch(r"(.+)-t([1-9][0-9]*)", request_id)
                binding = self._episode_bindings.get(match[1]) if match else None
            if binding is not None and binding["closed"]:
                raise ValueError("Native training episode is closed; no late dispatch")
            if self._episode_bindings and binding is None:
                raise ValueError("Generation request lacks an admitted training episode binding")
            return binding

    @property
    def ledger(self):
        with self._lock:
            if self._ledger is None:
                self._ledger = RequestLedger(self.ledger_root)
            return self._ledger

    def client_factory(self, url, credentials, timeout=None):
        if url != self.url or Path(credentials) != self.credentials:
            raise ValueError("native RPC route differs from configured transport")
        from .network import RpcClient
        owner = self
        client = (self.client_type or RpcClient)(url, credentials, timeout=timeout or self.timeout)

        class Client:
            turn_id = 0

            def call(self, path, payload):
                if not path.startswith("/generate/"):
                    return client.call(path, payload)  # Preserve native diagnostic route and measurements.
                request_id = path.removeprefix("/generate/")
                try:
                    binding = owner._binding(request_id)
                    contract = binding["contract"] if binding is not None else owner.training_contract
                    if contract is not None:
                        contract.check_payload(payload)
                except BaseException as error:
                    error.rpc_failure = {"phase": "training_admission", "dispatch_may_have_occurred": False,
                                         "automatic_replay": False}
                    raise
                try:
                    state = owner.ledger.claim(request_id, path, payload, timeout or owner.timeout)
                    owner.ledger.update(state, state="dispatching", uncertain=True)
                except BaseException as error:
                    error.rpc_failure = {"phase": "ledger_claim", "dispatch_may_have_occurred": False,
                                         "automatic_replay": False}
                    raise
                try:
                    result = client.call(path, payload)
                except BaseException as error:
                    failure = getattr(error, "rpc_failure", None)
                    uncertain = True if failure is None else failure["dispatch_may_have_occurred"]
                    owner.ledger.update(state, state="outcome_unknown" if uncertain else "not_dispatched",
                                        uncertain=uncertain, failure=failure)
                    raise
                try:
                    owner.ledger.update(state, state="response_received", uncertain=False,
                                        acknowledged=True)
                except BaseException as error:
                    error.rpc_failure = {"phase": "response_evidence", "dispatch_may_have_occurred": True,
                                         "automatic_replay": False}
                    raise
                if contract is not None:
                    # Receipt validation precedes every native parse/compile.
                    from .network import generation_completion
                    self.turn_id += 1
                    with owner._trace_lock:
                        turn_id = binding["next_turn"] if binding is not None else self.turn_id
                        if binding is not None:
                            binding["next_turn"] += 1
                    try:
                        trace = contract.validate(payload,
                            generation_completion(result), request_id, turn_id)
                        owner.ledger.update(state, training_evidence="validated")
                        if binding is not None:
                            with owner._trace_lock:
                                binding["traces"].append(trace)
                            if binding["trace_sink"] is not None:
                                binding["trace_sink"](trace)
                        owner._capture(trace)
                    except BaseException as error:
                        owner.ledger.update(state, training_evidence="rejected")
                        error.rpc_failure = {"phase": "training_evidence", "dispatch_may_have_occurred": True,
                            "response_received": True, "upstream_completion_known": True, "automatic_replay": False}
                        raise
                return result

            def close(self):
                client.close()
        return Client()

    async def submit(self, request):
        if request.get("route", "tls-rpc") != "tls-rpc":
            raise ValueError("transport route cannot be substituted")
        client = self.client_factory(self.url, self.credentials, request.get("timeout", self.timeout))
        def call():
            try:
                return client.call("/generate/" + request["request_id"], request["payload"])
            finally:
                client.close()
        worker = asyncio.create_task(asyncio.to_thread(call))
        # Cancellation of this waiter does not cancel the dispatching thread or replay it.
        return await asyncio.shield(worker)


class FileRelayTransport:
    def __init__(self, root, *, training_contract=None, trace_sink=None):
        self.root = Path(root)
        self.training_contract, self.trace_sink = training_contract, trace_sink
        self.training_traces = []

    def with_training(self, contract, trace_sink=None):
        """One independent capture scope per admitted episode; route is fixed."""
        return type(self)(self.root, training_contract=contract, trace_sink=trace_sink)

    def harbor_relay(self, output):
        from .terminal_bench.harbor_filerelay_agent import AsyncFileRelay
        policy = self.training_contract.policy if self.training_contract is not None else None
        expected_ack = ({key: getattr(policy, key) for key in ("version", "digest", "lineage")}
                        if policy is not None else None)
        relay = AsyncFileRelay(self.root, Path(output), expected_policy_ack=expected_ack)
        if self.training_contract is None:
            return relay
        owner = self

        class TrainingRelay:
            turn_id = 0

            @property
            def request_ids(self):
                return relay.request_ids

            @property
            def training_traces(self):
                return tuple(owner.training_traces)

            async def generate(self, payload, timeout, request_id=None):
                owner.training_contract.check_payload(payload)
                result = await relay.generate(payload, timeout, request_id=request_id)
                self.turn_id += 1
                identity = relay.request_ids[-1]
                trace = owner.training_contract.validate(payload, result, identity, self.turn_id)
                # Preserve raw evidence in private logs before action parsing.
                from .terminal_bench.harbor_filerelay_agent import write_new, json_bytes
                write_new(Path(output) / (identity + ".training.json"), json_bytes(trace.to_dict()))
                owner.training_traces.append(trace)
                if owner.trace_sink is not None:
                    owner.trace_sink(trace)
                return result

        return TrainingRelay()

    async def submit(self, request):
        if request.get("route", "browser-filerelay") != "browser-filerelay":
            raise ValueError("browser relay cannot be replaced by a direct route")
        relay = self.harbor_relay(Path(request["output_dir"]))
        return await relay.generate(request["payload"], request["timeout"], request_id=request["request_id"])
