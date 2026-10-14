"""Package and import exact original sealed evidence over the artifact protocol."""
import gzip
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile

from .browser_relay.artifacts import (
    CHUNK_BYTES, MAX_BYTES, checked_path, encoded, fsync_directory,
    publish_file, read_json, wait_received, write_new,
)

MAX_FILES = 20000


def publish_sealed(relay_root, output, group_id, key):
    from .sealed import read_json as read_seal, file_sha256
    output = checked_path(output, directory=True)
    if type(group_id) is not int or group_id < 0:
        raise ValueError("nonnegative sealed group id is required")
    seal_path = output / ("group-%d.seal.json" % group_id)
    seal = read_seal(seal_path)
    closure = seal["closure"]
    if (seal.get("kind") != "terminalbench-sealed-group" or seal.get("group_id") != group_id
            or closure.get("coordinator_closed") is not True or closure.get("cpu_spool_terminal") is not True):
        raise ValueError("only original CPU-closed groups may be transported")
    members = {**seal["inventory"], seal_path.name: file_sha256(seal_path)}
    if not 1 <= len(members) <= MAX_FILES or sum(row["bytes"] for row in members.values()) > MAX_BYTES:
        raise ValueError("sealed evidence exceeds automatic artifact bound")
    fd, temporary = tempfile.mkstemp(prefix=".sealed-", dir=checked_path(relay_root, directory=True))
    os.close(fd)
    try:
        with open(temporary, "wb") as raw_archive, gzip.GzipFile(fileobj=raw_archive, mode="wb", filename="", mtime=0) as compressed, tarfile.open(fileobj=compressed, mode="w") as archive:
            for name, expected in sorted(members.items()):
                relative = PurePosixPath(name)
                if relative.is_absolute() or ".." in relative.parts or str(relative) != name:
                    raise ValueError("unsafe sealed member path")
                source = checked_path(output / name)
                if file_sha256(source) != expected:
                    raise ValueError("sealed original member changed")
                info = tarfile.TarInfo(name)
                info.size, info.mode, info.mtime = expected["bytes"], 0o600, 0
                with source.open("rb") as stream:
                    archive.addfile(info, stream)
        return publish_file(relay_root, "sealed_group", key, temporary)
    finally:
        Path(temporary).unlink(missing_ok=True)


def receive_sealed(relay_root, key, output, timeout):
    """Import into exactly the caller-selected new directory; preserve seal bytes."""
    manifest, blob = wait_received(relay_root, "sealed_group", key, timeout)
    output = Path(output).absolute()
    checked_path(output.parent, directory=True)
    receipt_path = output.parent / (output.name + ".artifact-import.json")
    if output.exists():
        receipt = read_json(receipt_path)
        if receipt.get("artifact") != manifest:
            raise ValueError("existing evidence destination is not this immutable artifact")
        from .sealed import read_json as read_seal, file_sha256
        seal_path = checked_path(output / receipt["seal_file"])
        if file_sha256(seal_path)["sha256"] != receipt["seal_sha256"]:
            raise ValueError("imported original seal changed")
        seal = read_seal(seal_path)
        for name, expected in seal["inventory"].items():
            if file_sha256(output / name) != expected:
                raise ValueError("imported original evidence changed")
        return seal_path
    temporary = Path(tempfile.mkdtemp(prefix=".artifact-import-", dir=output.parent))
    try:
        with tarfile.open(blob, "r:gz") as archive:
            members, total, names = [], 0, set()
            for member in archive:
                name = PurePosixPath(member.name)
                if (not member.isfile() or name.is_absolute() or ".." in name.parts or str(name) != member.name
                        or member.name in names or member.size < 0):
                    raise ValueError("unsafe or repeated sealed archive member")
                total += member.size
                names.add(member.name)
                if len(names) > MAX_FILES or total > MAX_BYTES:
                    raise ValueError("sealed archive exceeds extraction bounds")
                members.append(member)
            for member in members:
                destination = temporary / member.name
                destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                stream = archive.extractfile(member)
                with os.fdopen(os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), "wb") as target:
                    shutil.copyfileobj(stream, target, CHUNK_BYTES)
                    target.flush()
                    os.fsync(target.fileno())
        seals = list(temporary.glob("group-*.seal.json"))
        if len(seals) != 1:
            raise ValueError("automatic artifact must contain one original seal")
        from .sealed import read_json as read_seal, file_sha256
        from rollout_manager.generation_training import evidence_digest
        seal = read_seal(seals[0])
        if (seal.get("kind") != "terminalbench-sealed-group" or evidence_digest(seal["inventory"]) != seal["inventory_sha256"]
                or names != set(seal["inventory"]) | {seals[0].name}):
            raise ValueError("transported archive does not exactly cover original sealed inventory")
        for name, expected in seal["inventory"].items():
            if file_sha256(temporary / name) != expected:
                raise ValueError("transported original evidence hash differs")
        receipt = {"schema_version": 1, "artifact": manifest, "seal_file": seals[0].name,
                   "seal_sha256": file_sha256(seals[0])["sha256"]}
        if receipt_path.exists():
            if read_json(receipt_path) != receipt:
                raise ValueError("pending immutable import receipt differs")
        else:
            write_new(receipt_path, encoded(receipt))
        os.rename(temporary, output)
        fsync_directory(output.parent)
        return output / seals[0].name
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
