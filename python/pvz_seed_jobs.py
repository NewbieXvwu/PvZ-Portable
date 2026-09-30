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
import zipfile

import numpy as np


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


def _archive_encode(value: Any, arrays: dict[str, np.ndarray]) -> Any:
    """Encode nested trajectory data as JSON metadata plus named numeric arrays."""
    if isinstance(value, np.ndarray):
        name = f"array_{len(arrays):04d}"
        arrays[name] = np.ascontiguousarray(value)
        return {"__pvz_array__": name}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _archive_encode(item, arrays) for key, item in value.items()}
    if isinstance(value, tuple):
        return {"__pvz_tuple__": [_archive_encode(item, arrays) for item in value]}
    if isinstance(value, list):
        return [_archive_encode(item, arrays) for item in value]
    return value


def _archive_decode(value: Any, archive: Any) -> Any:
    if isinstance(value, list):
        return [_archive_decode(item, archive) for item in value]
    if isinstance(value, dict):
        if set(value) == {"__pvz_array__"}:
            return archive[value["__pvz_array__"]].copy()
        if set(value) == {"__pvz_tuple__"}:
            return tuple(_archive_decode(item, archive) for item in value["__pvz_tuple__"])
        return {key: _archive_decode(item, archive) for key, item in value.items()}
    return value


def atomic_numpy(path: Path, value: Any, *, compressed: bool = False) -> None:
    """Atomically persist nested Python data while preserving NumPy dtypes.

    Shards use NPZ's numeric arrays and a small JSON manifest.  JSON-only shards
    expanded compact observation arrays into Python numbers and could not preserve
    ``ndarray`` values needed by the PPO forward path.
    """
    def write(temporary_path: Path) -> None:
        arrays: dict[str, np.ndarray] = {}
        encoded = _archive_encode(value, arrays)
        manifest = json.dumps({"schema_version": 1, "value": encoded},
                              separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        arrays["__manifest__"] = np.frombuffer(manifest, dtype=np.uint8)
        with temporary_path.open("wb") as stream:
            saver = np.savez_compressed if compressed else np.savez
            saver(stream, **arrays)

    atomic_write(path, write)


def read_numpy(path: Path) -> Any:
    with np.load(path, allow_pickle=False) as archive:
        manifest = json.loads(archive["__manifest__"].tobytes().decode("utf-8"))
        if manifest.get("schema_version") != 1:
            raise ValueError(f"unsupported NumPy shard schema in {path}")
        return _archive_decode(manifest["value"], archive)


def read_episode(path: Path) -> dict[str, Any]:
    """Read one rollout shard and return the episode inside it.

    ``run_seed_jobs`` writes ``{"metadata": ..., "result": ...}`` so an interrupted
    run can be resumed, but shards written before that wrapper existed hold the
    episode directly.  Both shapes are accepted here rather than at each call site:
    every reader used to open-code the check, and ``attention_benchmark`` had
    silently stopped matching either one.
    """
    stored = read_numpy(path)
    return stored["result"] if "result" in stored else stored


def seed_job_directory(output_dir: Path, stage: str, metadata: dict[str, Any]) -> Path:
    identity = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(identity).hexdigest()
    return output_dir / ".seed_jobs" / stage / digest


def _shard_path(directory: Path, seed: int) -> Path:
    return directory / f"seed_{seed}.npz"


def _read_shard(path: Path, seed: int, metadata: dict[str, Any]) -> dict[str, Any] | None:
    try:
        shard = read_numpy(path)
        result = shard["result"]
        if shard.get("metadata") == metadata and isinstance(result, dict) and result.get("seed") == seed:
            return result
    except (OSError, EOFError, ValueError, TypeError, KeyError, zlib.error, zipfile.BadZipFile):
        pass
    path.unlink(missing_ok=True)
    return None


def _collect_and_save(job: tuple[int, Path, dict[str, Any], Callable[[int], dict[str, Any]]]
                      ) -> tuple[int, dict[str, Any]]:
    """Collect one seed, persist its shard, and hand the result straight back.

    Returning the result matters: the parent used to read every shard back off
    disk, which cost 12.4 s of serial npz decoding per 2000-episode update, while
    pushing the same objects through the pool pipe costs ~1.2 s.  The shard is
    still written, because it is what makes an interrupted run resumable.
    """
    seed, directory, metadata, worker = job
    result = worker(seed)
    if not isinstance(result, dict) or result.get("seed") != seed:
        raise ValueError(f"seed worker returned a malformed result for seed {seed}")
    atomic_numpy(_shard_path(directory, seed), {"metadata": metadata, "result": result}, compressed=True)
    return seed, result


# A worker that dies (a failing initializer, a missing asset, an OOM) does not make
# ``imap_unordered`` raise: the pool respawns it, it dies again, and the parent waits
# forever.  This was observed, not theorised -- a bad initializer signature hung the
# probe for 21 minutes while the workers printed tracebacks.  The watchdog converts
# that silent hang into an error, which matters because the stop-loss budget is spent
# by wall-clock time whether or not anything is happening.
STALL_TIMEOUT_SECONDS = 900.0


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
    stall_timeout: float | None = STALL_TIMEOUT_SECONDS,
) -> list[dict[str, Any]]:
    """Collect every seed, reusing cached shards, and return them in ``seeds`` order.

    ``stall_timeout`` bounds the wait for *one* seed to finish; pass ``None`` to wait
    forever.  The default is generous by construction: the largest ``max_actions`` in
    this repository is 4000 and a PPO rollout decision costs ~2.3-3 ms, so a legitimate
    episode is ~12 s -- 75x under the limit.  Raise it if you drive this with a policy
    whose per-decision cost is orders of magnitude higher.
    """
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
        completed = 0
        try:
            iterator = pool.imap_unordered(
                _collect_and_save,
                ((seed, directory, metadata, worker) for seed in missing),
                chunksize=1,
            )
            while True:
                try:
                    seed, result = iterator.next(timeout=stall_timeout)
                except StopIteration:
                    break
                except multiprocessing.TimeoutError:
                    raise RuntimeError(
                        f"{label} stalled: no seed completed in {stall_timeout:.0f} s "
                        f"after {completed} of {len(missing)}. A worker almost certainly "
                        f"died in its initializer; check the tracebacks above the hang."
                    ) from None
                results[seed] = result
                completed += 1
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
        result = results.get(seed)
        if result is None:
            result = _read_shard(_shard_path(directory, seed), seed, metadata)
        if result is None:
            raise RuntimeError(f"seed {seed} completed without a readable result shard")
        ordered.append(result)
    return ordered
