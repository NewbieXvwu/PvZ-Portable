"""Lossless git-sized checkpoint parts; failed restores keep their partial file."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

DEFAULT_PART_BYTES = 64 * 1024 * 1024
MAX_PART_BYTES = 95_000_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_file(source: Path, bundle: Path, part_bytes: int = DEFAULT_PART_BYTES) -> dict:
    if type(part_bytes) is not int or not 1 <= part_bytes <= MAX_PART_BYTES:
        raise ValueError("part size must be positive and below the git single-file limit")
    if bundle.exists():
        raise ValueError("bundle exists; keep prior evidence and choose a fresh bundle directory")
    before = source.stat()
    bundle.mkdir(parents=True)
    whole, parts, count = hashlib.sha256(), [], 0
    with source.open("rb") as stream:
        while block := stream.read(part_bytes):
            name = f"part_{len(parts):06d}.bin"
            (bundle / name).write_bytes(block)
            whole.update(block)
            count += len(block)
            parts.append({"path": name, "bytes": len(block), "sha256": hashlib.sha256(block).hexdigest()})
    after = source.stat()
    digest = whole.hexdigest()
    if ((before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)
            or count != before.st_size or digest != sha256(source)):
        raise ValueError("source changed during export; preserve this incomplete bundle")
    manifest = {"schema_version": 1, "source_name": source.name, "source_path": str(source),
                "bytes": count, "sha256": digest, "part_bytes": part_bytes, "parts": parts,
                "semantics": "exact original bytes; no checkpoint fields are removed or reconstructed"}
    temporary = bundle / "manifest.json.partial"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(temporary, bundle / "manifest.json")
    return manifest


def restore_file(bundle: Path, destination: Path) -> dict:
    manifest = json.loads((bundle / "manifest.json").read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported part manifest schema")
    if (type(manifest.get("bytes")) is not int or manifest["bytes"] < 0
            or type(manifest.get("part_bytes")) is not int
            or not 1 <= manifest["part_bytes"] <= MAX_PART_BYTES):
        raise ValueError("invalid declared file or part size")
    if destination.exists():
        if destination.stat().st_size == manifest["bytes"] and sha256(destination) == manifest["sha256"]:
            return {"status": "existing_verified", "destination": str(destination), "sha256": manifest["sha256"]}
        raise ValueError("destination exists with different bytes; refusing to overwrite")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".partial_{time.time_ns()}")
    whole, count = hashlib.sha256(), 0
    with temporary.open("xb") as output:
        for index, part in enumerate(manifest["parts"]):
            expected_name = f"part_{index:06d}.bin"
            if part["path"] != expected_name:
                raise ValueError(f"unexpected/missing/reordered part {index}; partial retained at {temporary}")
            path = bundle / expected_name
            if type(part["bytes"]) is not int or not 1 <= part["bytes"] <= manifest["part_bytes"]:
                raise ValueError(f"invalid part size; partial retained at {temporary}")
            if path.stat().st_size != part["bytes"]:
                raise ValueError(f"part size mismatch: {path}; partial retained at {temporary}")
            block = path.read_bytes()
            if len(block) != part["bytes"] or hashlib.sha256(block).hexdigest() != part["sha256"]:
                raise ValueError(f"part hash/size mismatch: {path}; partial retained at {temporary}")
            output.write(block)
            count += len(block)
            whole.update(block)
        output.flush()
        os.fsync(output.fileno())
    if count != manifest["bytes"] or whole.hexdigest() != manifest["sha256"]:
        raise ValueError(f"reassembled file hash/size mismatch; partial retained at {temporary}")
    # Atomic create without replacing a destination that appeared during restore.
    # Both paths are siblings, so the hard link cannot cross filesystem boundaries.
    os.link(temporary, destination)
    temporary.unlink()
    return {"status": "restored_verified", "destination": str(destination),
            "bytes": count, "sha256": manifest["sha256"]}


def relative_path(root: Path, name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("archive paths must stay relative to their root")
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("archive path escapes its root")
    return path


def restore_archive(archive: Path, destination: Path) -> dict:
    manifest = json.loads((archive / "archive_manifest.json").read_text())
    if manifest.get("schema_version") not in (1, 2) or destination.exists():
        raise ValueError("supported archive and a fresh destination directory are required")
    bundles = manifest.get("large_file_bundles", [])
    prefixes = [entry["bundle_path"].rstrip("/") + "/" for entry in bundles]
    destination.mkdir(parents=True)
    for name, expected in manifest["files"].items():
        source = relative_path(archive, name)
        if source.stat().st_size != expected["bytes"] or sha256(source) != expected["sha256"]:
            raise ValueError(f"archive inventory mismatch: {source}; destination retained")
        if any(name.startswith(prefix) for prefix in prefixes):
            continue
        target = relative_path(destination, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if sha256(target) != expected["sha256"]:
            raise ValueError(f"normal file copy mismatch: {target}; destination retained")
    restored = []
    for entry in bundles:
        result = restore_file(relative_path(archive, entry["bundle_path"]),
                              relative_path(destination, entry["original_path"]))
        if result["sha256"] != entry["sha256"]:
            raise ValueError("bundle digest differs from archive inventory; destination retained")
        restored.append(result)
    pointer = json.loads((destination / "resume.json").read_text())
    checkpoint = relative_path(destination, pointer["checkpoint"])
    if sha256(checkpoint) != pointer["sha256"]:
        raise ValueError("restored resume pointer does not match checkpoint")
    return {"status": "archive_restored_verified", "destination": str(destination),
            "large_files": restored, "checkpoint_sha256": pointer["sha256"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("--source", type=Path, required=True)
    export.add_argument("--bundle-dir", type=Path, required=True)
    export.add_argument("--part-bytes", type=int, default=DEFAULT_PART_BYTES)
    restore = commands.add_parser("restore")
    restore.add_argument("--bundle-dir", type=Path, required=True)
    restore.add_argument("--destination", type=Path, required=True)
    archive = commands.add_parser("restore-archive")
    archive.add_argument("--archive-dir", type=Path, required=True)
    archive.add_argument("--destination-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "export":
        value = export_file(args.source, args.bundle_dir, args.part_bytes)
        print(json.dumps({"status": "exported_verified", "bytes": value["bytes"],
                          "sha256": value["sha256"], "parts": len(value["parts"])}))
    elif args.command == "restore":
        print(json.dumps(restore_file(args.bundle_dir, args.destination)))
    else:
        print(json.dumps(restore_archive(args.archive_dir, args.destination_dir)))


if __name__ == "__main__":
    main()
