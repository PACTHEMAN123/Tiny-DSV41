import copy
import socket
import struct
import threading
import unittest
from unittest.mock import patch

import torch

from rollout_manager.backends.vllm_lora_transport import (
    MAX_MANIFEST_BYTES, pack_update, policy_ack, receive_update, send_update,
    unpack_update, validate_update)


def update(version=0, adapter_id=1, base_identity=None):
    tensors = {
        "language_model.model.layers.0.attn.wq_a.lora_A.weight": torch.arange(32, dtype=torch.bfloat16).reshape(8, 4),
        "language_model.model.layers.0.attn.wq_a.lora_B.weight": torch.zeros((2, 8), dtype=torch.bfloat16)}
    manifest, payload = pack_update(tensors, version=version, lineage="unit-fixture", adapter_id=adapter_id,
        peft={"r": 8, "lora_alpha": 8, "target_modules": ["wq_a"]},
        base_identity=base_identity or {"config_sha256": "a" * 64, "index_sha256": "b" * 64})
    return manifest, payload, tensors


class DirectTensorTransportTests(unittest.TestCase):
    def test_snapshot_roundtrip_and_digest_without_checkpoint_calls(self):
        with patch("torch.save", side_effect=AssertionError("checkpoint write")), \
                patch("torch.load", side_effect=AssertionError("checkpoint read")):
            manifest, payload, tensors = update()
            restored = unpack_update(manifest, payload)
            for name, tensor in tensors.items():
                self.assertTrue(torch.equal(restored[name], tensor))
                tensor.fill_(5)
                self.assertFalse(torch.equal(restored[name], tensor))
            self.assertEqual(policy_ack(manifest), {key: manifest[key] for key in ("version", "digest", "lineage")})

    def test_corrupt_bytes_or_manifest_rejected_before_unpack(self):
        manifest, payload, _ = update()
        with self.assertRaisesRegex(ValueError, "bytes differ"):
            unpack_update(manifest, bytes([payload[0] ^ 1]) + payload[1:])
        for mutation in (lambda value: value.update(version=1),
                         lambda value: value.update(checkpoint_path="forbidden"),
                         lambda value: value["tensors"][0].update(offset=1)):
            changed = copy.deepcopy(manifest)
            mutation(changed)
            with self.assertRaises(ValueError):
                validate_update(changed, payload)
        with self.assertRaisesRegex(ValueError, "finite"):
            pack_update({"x.lora_A.weight": torch.full((2, 2), float("nan"))},
                version=0, lineage="fixture", adapter_id=1, peft={}, base_identity={})

    def test_bounded_binary_socket_frame_roundtrip_and_truncation(self):
        manifest, payload, _ = update()
        left, right = socket.socketpair()
        with left, right:
            thread = threading.Thread(target=send_update, args=(left, manifest, payload))
            thread.start()
            self.assertEqual(receive_update(right), (manifest, payload))
            thread.join(timeout=2)
            self.assertFalse(thread.is_alive())
        left, right = socket.socketpair()
        with left, right:
            left.sendall(struct.pack("!IQ", MAX_MANIFEST_BYTES + 1, 1))
            with self.assertRaisesRegex(ValueError, "budget"):
                receive_update(right)
        left, right = socket.socketpair()
        with left, right:
            left.sendall(b"short")
            left.shutdown(socket.SHUT_WR)
            with self.assertRaises(EOFError):
                receive_update(right)


if __name__ == "__main__":
    unittest.main()
