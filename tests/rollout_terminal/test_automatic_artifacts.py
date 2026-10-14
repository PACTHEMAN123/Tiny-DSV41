"""Artifact durability and once-only CPU supervision without remote workloads."""
import copy
import hashlib
import json
import os
from pathlib import Path
import socket
import tempfile
import unittest

from rollout_manager.terminal_bench import artifact_transport as artifacts
from rollout_manager.terminal_bench import cpu_supervisor as supervisor
from rollout_manager.terminal_bench.sealed import write_seals
from .test_native_profile import proof


def private(path):
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    return path.resolve()


def deliver(sender, receiver):
    manifest = sender.next()
    progress = receiver.start(manifest)
    for index in range(progress["next_index"], (manifest["bytes"] + artifacts.CHUNK_BYTES - 1) // artifacts.CHUNK_BYTES):
        receiver.receive_chunk(sender.chunk(manifest["id"], index))
    receipt = receiver.commit(manifest["id"])
    sender.acknowledge(receipt)
    return manifest


def lease(version=0, size=8):
    value = proof()
    value["policy"]["version"] = version
    value["policy"]["digest"] = hashlib.sha256(str(version).encode()).hexdigest()
    value["capabilities"]["policy"] = value["policy"]
    return {"schema_version": 1, "kind": "terminalbench-policy-lease", "policy": value["policy"],
            "capabilities": value["capabilities"], "group_size": size,
            "purpose": "policy_evaluation" if size == 1 else "training"}


def closed(output, group, value):
    private(output)
    artifacts.write_new(output / "lease.json", artifacts.encoded(value))
    closure = {"coordinator_closed": True, "cpu_spool_terminal": True,
               "lifecycle_failures": [], "quarantined": {}, "admission_stopped_reason": None,
               "reserved": {"cpu": {"cpu": 0, "memory_bytes": 0, "storage_bytes": 0}}}
    artifacts.write_new(output / "cpu-closure.json", artifacts.encoded(closure))
    return write_seals(output, [{"group_id": group, "registration_file": "lease.json",
        "episode_files": [], "receipt_file": "lease.json", "receipt_kind": "training_group"}], closure)[0]


class ArtifactTests(unittest.TestCase):
    def test_restart_resumes_bytes_and_lost_ack_never_republishes_a_workload(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            source = root / "blob"
            source.write_bytes(os.urandom(artifacts.CHUNK_BYTES * 2 + 17))
            sender = artifacts.ArtifactQueue(private(root / "cpu"), "cpu")
            receiver_root = private(root / "gpu")
            receiver = artifacts.ArtifactQueue(receiver_root, "gpu")
            manifest = sender.publish("sealed_group", "group-0", source)
            receiver.start(manifest)
            first = sender.chunk(manifest["id"], 0)
            receiver.receive_chunk(first)
            receiver.receive_chunk(first)  # a lost chunk ACK repeats identical bytes
            receiver = artifacts.ArtifactQueue(receiver_root, "gpu")
            self.assertEqual(receiver.start(manifest)["next_index"], 1)
            receiver.receive_chunk(sender.chunk(manifest["id"], 1))
            receiver.receive_chunk(sender.chunk(manifest["id"], 2))
            receipt = receiver.commit(manifest["id"])
            self.assertEqual(receiver.commit(manifest["id"]), receipt)
            self.assertIsNotNone(sender.next())
            wrong = {**receipt, "sha256": "f" * 64}
            with self.assertRaisesRegex(ValueError, "content-bound"):
                sender.acknowledge(wrong)
            sender.acknowledge(receipt)
            self.assertIsNone(sender.next())
            self.assertEqual(receiver.received("sealed_group", "group-0")[1].read_bytes(), source.read_bytes())
            self.assertEqual(sender.publish("sealed_group", "group-0", source), manifest)
            source.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "different immutable"):
                sender.publish("sealed_group", "group-0", source)

    def test_role_hash_size_order_and_path_escape_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            sender = artifacts.ArtifactQueue(private(root / "a"))
            receiver = artifacts.ArtifactQueue(private(root / "b"), "cpu")
            source = root / "lease"
            source.write_bytes(b"x" * (artifacts.CHUNK_BYTES + 1))
            manifest = sender.publish("policy_lease", "lease-v1", source)
            receiver.start(manifest)
            with self.assertRaisesRegex(ValueError, "out of order"):
                receiver.receive_chunk(sender.chunk(manifest["id"], 1))
            bad = sender.chunk(manifest["id"], 0)
            bad["sha256"] = "f" * 64
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                receiver.receive_chunk(bad)
            with self.assertRaises((ValueError, FileNotFoundError)):
                receiver.commit(manifest["id"])
            with self.assertRaisesRegex(ValueError, "invalid artifact id"):
                sender.chunk("../escape", 0)
            huge = {**manifest, "bytes": artifacts.MAX_BYTES + 1}
            with self.assertRaises(ValueError):
                receiver.start(huge)
            other = sender.publish("sealed_group", "group-0", source)
            with self.assertRaisesRegex(ValueError, "relay role"):
                receiver.start(other)

    def test_original_seal_roundtrip_is_exact_and_deterministic(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            cpu, gpu = private(root / "cpu"), private(root / "gpu")
            original = root / "original"
            seal = closed(original, 0, lease())
            first = artifacts.publish_sealed(cpu, original, 0, "group-0")
            self.assertEqual(artifacts.publish_sealed(cpu, original, 0, "group-0"), first)
            deliver(artifacts.ArtifactQueue(cpu), artifacts.ArtifactQueue(gpu, "gpu"))
            target = root / "imported"
            imported = artifacts.receive_sealed(gpu, "group-0", target, 1)
            self.assertEqual(imported.read_bytes(), seal.read_bytes())
            self.assertEqual(artifacts.receive_sealed(gpu, "group-0", target, 1), imported)
            (target / "lease.json").write_text("tampered imported member")
            with self.assertRaisesRegex(ValueError, "imported original evidence changed"):
                artifacts.receive_sealed(gpu, "group-0", target, 1)
            (original / "lease.json").write_text("tampered")
            with self.assertRaisesRegex(ValueError, "original member changed"):
                artifacts.publish_sealed(cpu, original, 0, "another-key")


class SupervisorTests(unittest.TestCase):
    def setup_value(self, root, existing=True):
        output, cpu, gpu, source = [private(root / name) for name in ("output", "cpu", "gpu", "source")]
        templates = []
        for group, size in ((0, 8), (1, 1)):
            config = {"source_root": str(source), "expected_hostname": socket.gethostname(),
                "output_dir": str(output / ("training" if group == 0 else "evaluation")),
                "first_group_id": group, "group_size": size, "task_ids": ["sparql-university"],
                "purpose": "training" if size > 1 else "policy_evaluation",
                "terminal_bench": {"relay_root": str(cpu)}}
            phase = {"lease_key": "lease-v%d" % group, "seal_key": "group-%d" % group,
                     "policy_version": group, "config": config}
            if group == 0 and existing:
                phase["existing_output"] = str(root / "original")
                closed(root / "original", 0, lease())
            templates.append(phase)
        value = {"schema_version": 1, "expected_hostname": socket.gethostname(), "expected_uid": os.getuid(),
            "source_root": str(source), "relay_root": str(cpu), "output_root": str(output),
            "wait_seconds": 1, "phases": templates}
        for group, size in ((0, 8), (1, 1)):
            path = root / ("lease-%d.json" % group)
            path.write_bytes(artifacts.encoded(lease(group, size)))
            artifacts.publish_file(gpu, "policy_lease", "lease-v%d" % group, path)
            deliver(artifacts.ArtifactQueue(gpu), artifacts.ArtifactQueue(cpu, "cpu"))
        return value, cpu, gpu

    def test_historical_group_then_fresh_v1_runs_once_and_publishes_both(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            value, cpu, gpu = self.setup_value(root)
            calls = []
            def runner(path, source, control):
                config = artifacts.read_json(path)
                calls.append(config)
                closed(Path(config["output_dir"]), config["first_group_id"], artifacts.read_json(config["lease_capsule"]))
                return 0
            guard_calls = []
            def guard(config, leased, timeout):
                guard_calls.append(leased["policy"]["version"])
                return {"actual_guard_fixture": True}
            self.assertEqual(supervisor.run_supervisor(value, runner=runner, guard_factory=guard), 0)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["group_size"], 1)
            self.assertEqual(guard_calls, [1])
            self.assertEqual(supervisor.run_supervisor(value, runner=runner, guard_factory=guard), 0)
            self.assertEqual(len(calls), 1, "completed immutable phase cannot run again")
            outgoing = artifacts.ArtifactQueue(cpu)
            self.assertEqual({artifacts.read_json(p / "manifest.json")["key"] for p in (outgoing.root / "outgoing").iterdir()}, {"group-0", "group-1"})

    def test_claimed_crash_blocks_relaunch_and_new_lease_version_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            value, cpu, gpu = self.setup_value(root, existing=False)
            calls = []
            def failed(path, source, control):
                calls.append(path)
                raise RuntimeError("owned worker interrupted")
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                supervisor.run_supervisor(value, runner=failed, guard_factory=lambda *a: {})
            with self.assertRaisesRegex(ValueError, "never relaunched"):
                supervisor.run_supervisor(value, runner=failed, guard_factory=lambda *a: {})
            self.assertEqual(len(calls), 1)
        with tempfile.TemporaryDirectory() as temp:
            value, _, _ = self.setup_value(Path(temp).resolve())
            value["phases"][0]["policy_version"] = 2
            with self.assertRaisesRegex(ValueError, "policy version"):
                supervisor.run_supervisor(value, runner=lambda *a: self.fail("no dispatch"))


if __name__ == "__main__":
    unittest.main()
