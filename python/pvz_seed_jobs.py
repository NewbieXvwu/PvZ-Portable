"""Spawn-based, resumable per-seed collection shared by training and benchmarks."""

from __future__ import annotations

import gzip
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import tempfile
from typing import Any, Callable
import zlib


def atomic_write(path: Path, writer: Callable[[Path], None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        writer(temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_json(path: Path, value: Any, *, compressed: bool = False) -> None:
    def write(temporary_path: Path) -> None:
        if compressed:
            with gzip.open(temporary_path, "wt", encoding="utf-8", compresslevel=6) as stream:
                json.dump(value, stream, separators=(",", ":"), ensure_ascii=False)
        else:
            temporary_path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

    atomic_write(path, write)


def seed_job_directory(output_dir: Path, stage: str, metadata: dict[str, Any]) -> Path:
    identity = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(identity).hexdigest()
    return output_dir / ".seed_jobs" / stage / digest


def _shard_path(directory: Path, seed: int) -> Path:
    return directory / f"seed_{seed}.json.gz"


def _read_shard(path: Path, seed: int, metadata: dict[str, Any]) -> dict[str, Any] | None:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            shard = json.load(stream)
        result = shard["result"]
        if shard.get("metadata") == metadata and isinstance(result, dict) and result.get("seed") == seed:
            return result
    except (OSError, EOFError, ValueError, TypeError, KeyError, zlib.error):
        pass
    path.unlink(missing_ok=True)
    return None


def _collect_and_save(job: tuple[int, Path, dict[str, Any], Callable[[int], dict[str, Any]]]) -> int:
    seed, directory, metadata, worker = job
    result = worker(seed)
    if not isinstance(result, dict) or result.get("seed") != seed:
        raise ValueError(f"seed worker returned a malformed result for seed {seed}")
    atomic_json(_shard_path(directory, seed), {"metadata": metadata, "result": result}, compressed=True)
    return seed


def run_seed_jobs(
    seeds: list[int],
    directory: Path,
    metadata: dict[str, Any],
    worker: Callable[[int], dict[str, Any]],
    *,
    workers: int,
    initializer: Callable[..., None],
    initargs: tuple[Any, ...],
    label: str,
) -> list[dict[str, Any]]:
    metadata = json.loads(json.dumps(metadata, sort_keys=True, separators=(",", ":")))
    if workers < 1:
        raise ValueError("workers must be positive")
    if len(seeds) != len(set(seeds)):
        raise ValueError("seed jobs must be unique")
    directory.mkdir(parents=True, exist_ok=True)
    results: dict[int, dict[str, Any]] = {}
    missing: list[int] = []
    for seed in seeds:
        path = _shard_path(directory, seed)
        result = _read_shard(path, seed, metadata) if path.exists() else None
        if result is None:
            missing.append(seed)
        else:
            results[seed] = result

    if missing:
        context = multiprocessing.get_context("spawn")
        pool = context.Pool(min(workers, len(missing)), initializer=initializer, initargs=initargs)
        try:
            for completed, _ in enumerate(
                pool.imap_unordered(
                    _collect_and_save,
                    ((seed, directory, metadata, worker) for seed in missing),
                    chunksize=1,
                ),
                start=1,
            ):
                if completed % 16 == 0 or completed == len(missing):
                    print(f"{label} {len(seeds) - len(missing) + completed}/{len(seeds)}", flush=True)
            pool.close()
        except BaseException:
            pool.terminate()
            raise
        finally:
            pool.join()

    ordered = []
    for seed in seeds:
        result = results.get(seed) or _read_shard(_shard_path(directory, seed), seed, metadata)
        if result is None:
            raise RuntimeError(f"seed {seed} completed without a readable result shard")
        ordered.append(result)
    return ordered
