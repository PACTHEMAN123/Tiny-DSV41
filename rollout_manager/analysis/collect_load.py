"""Export bounded measurement evidence; exclude credentials and build caches."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def collect(root, destination, *, telemetry=True, profiles=False, include=None, exclude_sqlite=False):
    included = []
    excluded = []
    total = 0
    with zipfile.ZipFile(destination, "x", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root)
            if include and relative.parts[0] not in include:
                continue
            if (not path.is_file() or path.is_symlink() or any(part in relative.parts for part in
                    ("credentials", "venv", "build", "inductor-cache", "__pycache__", "tests"))):
                continue
            if not telemetry and path.name.startswith("telemetry-"):
                continue
            if exclude_sqlite and path.suffix == ".sqlite":
                with path.open("rb") as source:
                    digest = hashlib.sha256()
                    for chunk in iter(lambda: source.read(8 * 1024 ** 2), b""):
                        digest.update(chunk)
                excluded.append({"path": str(relative), "bytes": path.stat().st_size,
                    "sha256": digest.hexdigest(), "reason": "explicit --exclude-sqlite; native report retained"})
                continue
            if path.suffix not in ({".json", ".jsonl", ".gz", ".log", ".py", ".cu", ".cpp", ".h", ".md", ".csv", ".txt"}
                                   | ({".nsys-rep", ".sqlite", ".data"} if profiles else set())):
                continue
            if path.stat().st_size > (512 * 1024 ** 2 if path.name.startswith("telemetry-") else 256 * 1024 ** 2):
                raise ValueError("oversize evidence file: " + str(relative))
            content = path.read_bytes()
            total += len(content)
            if total > 1536 * 1024 ** 2:
                raise ValueError("export exceeded 1.5 GiB")
            archive.writestr(str(relative), content)
            included.append({"path": str(relative), "bytes": len(content),
                             "sha256": hashlib.sha256(content).hexdigest()})
        content = json.dumps(excluded, indent=2).encode()
        archive.writestr("EXCLUDED-FILES.json", content)
        included.append({"path": "EXCLUDED-FILES.json", "bytes": len(content),
                         "sha256": hashlib.sha256(content).hexdigest()})
        archive.writestr("SHA256-MANIFEST.json", json.dumps(included, indent=2))
    print(json.dumps({"archive": str(destination), "bytes": destination.stat().st_size,
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(), "files": len(included)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--no-telemetry", action="store_true")
    parser.add_argument("--profiles", action="store_true")
    parser.add_argument("--exclude-sqlite", action="store_true",
                        help="inventory derived SQLite exports without copying them; retain native reports")
    parser.add_argument("--include", nargs="+", help="Exact top-level names to export")
    args = parser.parse_args()
    collect(args.root, args.destination, telemetry=not args.no_telemetry, profiles=args.profiles,
            include=args.include, exclude_sqlite=args.exclude_sqlite)
