"""Bounded, authenticated-actor tensor transport without checkpoint files."""

from __future__ import annotations
import hashlib
import json
import math
import re
import struct
import sys

MAX_MANIFEST_BYTES = 1024 * 1024
MAX_TENSOR_BYTES = 64 * 1024 * 1024
MAX_TENSORS = 128
_DTYPES = {"float16": 2, "bfloat16": 2, "float32": 4}
_SHA = re.compile("[0-9a-f]{64}\\Z")
_NAME = re.compile("[A-Za-z0-9_.]+\\.lora_[AB]\\.weight\\Z")
_FIELDS = {
    "schema_version",
    "version",
    "lineage",
    "adapter_id",
    "peft",
    "base_identity",
    "tensors",
    "payload_sha256",
    "digest",
}
_TENSOR_FIELDS = {"name", "shape", "dtype", "offset", "nbytes", "sha256"}


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(manifest, payload):
    unsigned = {key: value for key, value in manifest.items() if key != "digest"}
    return hashlib.sha256(_json(unsigned) + b"\x00" + payload).hexdigest()


def policy_ack(manifest):
    return {key: manifest[key] for key in ("version", "digest", "lineage")}


def validate_update(manifest, payload):
    """Validate every byte and descriptor before any worker model mutation."""
    if sys.byteorder != "little":
        raise RuntimeError("direct LoRA tensor transport requires a little-endian host")
    if (
        not isinstance(manifest, dict)
        or set(manifest) != _FIELDS
        or manifest["schema_version"] != 1
        or (type(manifest["version"]) is not int)
        or (manifest["version"] < 0)
        or (type(manifest["adapter_id"]) is not int)
        or (manifest["adapter_id"] < 1)
        or (not isinstance(manifest["lineage"], str))
        or (not 0 < len(manifest["lineage"]) <= 256)
        or (not isinstance(manifest["peft"], dict))
        or (not isinstance(manifest["base_identity"], dict))
        or (not isinstance(payload, bytes))
        or (not 0 < len(payload) <= MAX_TENSOR_BYTES)
    ):
        raise ValueError("invalid direct LoRA update schema or byte budget")
    if len(_json(manifest)) > MAX_MANIFEST_BYTES:
        raise ValueError("direct LoRA manifest exceeds byte budget")
    tensors = manifest["tensors"]
    if not isinstance(tensors, list) or not 0 < len(tensors) <= MAX_TENSORS:
        raise ValueError("invalid direct LoRA tensor count")
    end, names = (0, [])
    for item in tensors:
        if not isinstance(item, dict) or set(item) != _TENSOR_FIELDS:
            raise ValueError("invalid direct LoRA tensor descriptor")
        name, shape, dtype = (item["name"], item["shape"], item["dtype"])
        if (
            not isinstance(name, str)
            or not _NAME.fullmatch(name)
            or (not isinstance(shape, list))
            or (len(shape) != 2)
            or any((type(size) is not int or size < 1 for size in shape))
            or (dtype not in _DTYPES)
            or (type(item["offset"]) is not int)
            or (type(item["nbytes"]) is not int)
            or (item["offset"] != end)
            or (item["nbytes"] != math.prod(shape) * _DTYPES[dtype])
            or (not isinstance(item["sha256"], str))
            or (not _SHA.fullmatch(item["sha256"]))
        ):
            raise ValueError("invalid direct LoRA tensor shape, dtype, offset, or digest")
        end += item["nbytes"]
        if (
            end > len(payload)
            or hashlib.sha256(payload[item["offset"] : end]).hexdigest() != item["sha256"]
        ):
            raise ValueError("direct LoRA tensor bytes differ from manifest")
        names.append(name)
    if names != sorted(set(names)) or end != len(payload):
        raise ValueError("direct LoRA tensors must be unique, sorted, and cover the payload")
    for field in ("payload_sha256", "digest"):
        if not isinstance(manifest[field], str) or not _SHA.fullmatch(manifest[field]):
            raise ValueError("invalid direct LoRA payload digest")
    if (
        hashlib.sha256(payload).hexdigest() != manifest["payload_sha256"]
        or _digest(manifest, payload) != manifest["digest"]
    ):
        raise ValueError("direct LoRA update digest mismatch")
    return manifest


def pack_update(tensors, *, version, lineage, adapter_id, peft, base_identity):
    """Snapshot finite CPU tensor bytes; subsequent trainer mutation is independent."""
    import torch

    if not isinstance(tensors, dict) or not 0 < len(tensors) <= MAX_TENSORS:
        raise ValueError("a bounded nonempty adapter tensor dictionary is required")
    parts, descriptors, offset = ([], [], 0)
    for name, tensor in sorted(tensors.items()):
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("LoRA values must be tensors")
        value = tensor.detach().cpu().contiguous()
        dtype = str(value.dtype).removeprefix("torch.")
        if dtype not in _DTYPES or value.ndim != 2 or (not torch.isfinite(value).all()):
            raise ValueError("LoRA tensors must be finite matrices in a supported dtype")
        raw = value.view(torch.uint8).numpy().tobytes()
        if offset + len(raw) > MAX_TENSOR_BYTES:
            raise ValueError("direct LoRA tensors exceed byte budget")
        descriptors.append(
            {
                "name": name,
                "shape": list(value.shape),
                "dtype": dtype,
                "offset": offset,
                "nbytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
        parts.append(raw)
        offset += len(raw)
    payload = b"".join(parts)
    manifest = {
        "schema_version": 1,
        "version": version,
        "lineage": lineage,
        "adapter_id": adapter_id,
        "peft": peft,
        "base_identity": base_identity,
        "tensors": descriptors,
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
    }
    manifest = json.loads(_json(manifest))
    manifest["digest"] = _digest(manifest, payload)
    validate_update(manifest, payload)
    return (manifest, payload)


def unpack_update(manifest, payload):
    """Return validated finite CPU matrices without deserializing Python objects."""
    import torch

    validate_update(manifest, payload)
    tensors = {}
    for item in manifest["tensors"]:
        raw = bytearray(payload[item["offset"] : item["offset"] + item["nbytes"]])
        value = torch.frombuffer(raw, dtype=getattr(torch, item["dtype"])).reshape(item["shape"])
        if not torch.isfinite(value).all():
            raise ValueError("direct LoRA update contains nonfinite tensor values")
        tensors[item["name"]] = value
    return tensors


def _read_exact(stream, size):
    parts = []
    while size:
        part = stream.recv(size)
        if not part:
            raise EOFError("incomplete direct LoRA tensor frame")
        parts.append(part)
        size -= len(part)
    return b"".join(parts)


def send_update(stream, manifest, payload):
    validate_update(manifest, payload)
    metadata = _json(manifest)
    stream.sendall(struct.pack("!IQ", len(metadata), len(payload)) + metadata + payload)


def receive_update(stream):
    metadata_size, payload_size = struct.unpack("!IQ", _read_exact(stream, 12))
    if not 0 < metadata_size <= MAX_MANIFEST_BYTES or not 0 < payload_size <= MAX_TENSOR_BYTES:
        raise ValueError("direct LoRA frame exceeds byte budget")
    metadata = _read_exact(stream, metadata_size)
    manifest = json.loads(metadata)
    payload = _read_exact(stream, payload_size)
    validate_update(manifest, payload)
    return (manifest, payload)
