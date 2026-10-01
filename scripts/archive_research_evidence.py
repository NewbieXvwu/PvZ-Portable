"""Export an immutable state snapshot in its original relative layout for git delivery.

Training continues in its own directory. No source evidence is moved or removed.
An archive can be resumed as an output directory with the same frozen config and
source/resource fingerprints, because resume.json and checkpoint paths still agree.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
from pvz_seed_jobs import atomic_json
from research_checkpoint_chunks import DEFAULT_PART_BYTES, MAX_PART_BYTES, export_file


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--log", type=Path, action="append", default=[])
    parser.add_argument("--part-bytes", type=int, default=DEFAULT_PART_BYTES)
    parser.add_argument("--single-file-limit-bytes", type=int, default=100_000_000,
                        help="files at or above this threshold are losslessly exported as git-sized parts")
    args = parser.parse_args()
    if (not 1 <= args.single_file_limit_bytes <= 100_000_000
            or not 1 <= args.part_bytes <= MAX_PART_BYTES):
        raise ValueError("invalid export thresholds")
    if args.archive_dir.exists():
        raise ValueError("archive directory already exists; keep old snapshots and choose a new path")
    state_bytes = (args.experiment_dir / "training_state.json").read_bytes()
    state = json.loads(state_bytes)
    args.archive_dir.mkdir(parents=True)
    (args.archive_dir / "training_state.json").write_bytes(state_bytes)
    atomic_json(args.archive_dir / "learning_curve.json", state["learning_curve"])
    files = {"experiment_config.json", "provenance.json", state["checkpoint"]}
    files.update(point["raw_seed_results_path"] for point in state["learning_curve"])
    for point in state["learning_curve"]:
        paths = sorted((args.experiment_dir / "runs/run_1").glob(
            f"update_{point['updates']:06d}_evaluated_*.pt"))
        if not paths:
            raise ValueError("evaluation checkpoint missing")
        files.add(str(paths[0].relative_to(args.experiment_dir)))
    large_files = []
    for relative in sorted(files):
        source, target = args.experiment_dir / relative, args.archive_dir / relative
        if source.stat().st_size >= args.single_file_limit_bytes:
            bundle = export_file(source, Path(str(target) + ".gitparts"), args.part_bytes)
            large_files.append({"original_path": relative, "bundle_path": relative + ".gitparts",
                                "bytes": bundle["bytes"], "sha256": bundle["sha256"]})
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if sha256(source) != sha256(target):
            raise ValueError(f"archive copy hash mismatch: {source}")
    # Build the pointer from the state snapshot, never from a racing live pointer.
    checkpoint_hash = sha256(args.experiment_dir / state["checkpoint"])
    atomic_json(args.archive_dir / "resume.json", {"checkpoint": state["checkpoint"],
                                                   "sha256": checkpoint_hash})
    for log in args.log:
        target = args.archive_dir / "logs" / log.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(log, target)
    inventory = {str(path.relative_to(args.archive_dir)):
                 {"sha256": sha256(path), "bytes": path.stat().st_size}
                 for path in sorted(args.archive_dir.rglob("*")) if path.is_file()}
    atomic_json(args.archive_dir / "archive_manifest.json", {
        "schema_version": 2 if large_files else 1, "source_directory": str(args.experiment_dir),
        "experiment_id": state["experiment_id"], "counters": state["counters"],
        "status": state["status"], "phase": state["phase"], "archived_at_unix_ns": time.time_ns(),
        "state_sha256": hashlib.sha256(state_bytes).hexdigest(), "files": inventory,
        "checkpoint_scope": "all evaluated nodes plus latest complete update; full optimizer/RNG state",
        "rollout_shards": "all original shards retained in source directory; this export does not erase them",
        "log_scope": "log byte snapshots copied during export; a live source log may continue",
        "large_file_bundles": large_files,
        "resume": ("first run research_checkpoint_chunks.py restore-archive into a fresh directory, then resume with original config/fingerprints"
                   if large_files else "use this archive as --output-dir with original --experiment-config and unchanged fingerprints")})
    print("archived", state["experiment_id"], state["counters"], "files", len(inventory), flush=True)


if __name__ == "__main__":
    main()
