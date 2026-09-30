"""Decide the worker->parent channel and the pool start method with measurements.

Three questions, each answered by a number rather than an opinion:

1. How much of an update is pool startup?  ``run_seed_jobs`` builds a fresh
   ``multiprocessing`` pool on every update, and every worker imports torch and
   loads the 14 MiB model state before it can act.
2. Is the disk really needed as the worker->parent channel?  Today the worker
   writes an npz shard and the parent reads all 2000 shards back serially.
3. Is ``spawn`` the right start method on POSIX, or would ``fork`` pay less?

Each section is a separate invocation so that a hung pool cannot take the whole
probe down with it::

    for s in startup import fork pipe disk metadata; do
        python scripts/pool_ipc_probe.py --section $s
    done
"""

from __future__ import annotations

import argparse
import faulthandler
import io
import json
import multiprocessing
from pathlib import Path
import pickle
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

import numpy as np  # noqa: E402

PAYLOAD_KIB = 91          # measured compressed shard size for one episode
DEFAULT_RESOURCES = Path("/Users/newbiexvwu/Downloads/Plants_Vs_Zombies_V1.2.0.1073_EN")


def log(message: str) -> None:
    print(message, flush=True)


def _build_payload(seed: int, kib: float = PAYLOAD_KIB) -> bytes:
    """Bytes of the same order and shape as one compressed rollout shard."""
    rng = np.random.default_rng(seed)
    arrays = {
        "tokens": rng.integers(0, 256, size=(int(kib * 1024) - 4096,), dtype=np.uint8),
        "__manifest__": np.frombuffer(
            json.dumps({"schema_version": 1, "value": {"seed": seed}},
                       separators=(",", ":")).encode("utf-8"), dtype=np.uint8),
    }
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


# --------------------------------------------------------------------- workers

_TORCH_VERSION = "not imported"


def _init_torch() -> None:
    """Import torch in the worker: the import itself is the cost being measured."""
    global _TORCH_VERSION
    import torch

    _TORCH_VERSION = torch.__version__


def _init_torch_model_env(resource_dir: str) -> None:
    from pvz_agent_model import GameplayModelV1, configure_torch_threads
    from pvz_env import PvZEnv
    configure_torch_threads(1)
    env = PvZEnv(resource_dir=resource_dir)
    model = GameplayModelV1().eval()
    env.reset(level=1, seed=0)
    del model, env


def _trivial(_: int) -> int:
    return 0


_BLOB = b""


def _init_blob(blob: bytes) -> None:
    global _BLOB
    _BLOB = blob


def _return_blob(_: int) -> bytes:
    """Hand back a fixed blob so only the transfer cost is measured."""
    return _BLOB


def _write_shard(job: tuple[int, str]) -> int:
    """Produce the same bytes as ``_return_bytes`` but put them on disk instead."""
    seed, directory = job
    target = Path(directory) / f"seed_{seed}.npz"
    target.write_bytes(_build_payload(seed))
    return seed


# ------------------------------------------------------------------ experiments

def _run_pool(context_name: str, initializer, initargs: tuple, tasks: list,
              worker, *, workers: int, chunksize: int = 1) -> float:
    context = multiprocessing.get_context(context_name)
    start = time.perf_counter()
    pool = context.Pool(min(workers, len(tasks)), initializer=initializer, initargs=initargs)
    try:
        for _ in pool.imap_unordered(worker, tasks, chunksize=chunksize):
            pass
        pool.close()
    except BaseException:
        pool.terminate()
        raise
    finally:
        pool.join()
    return time.perf_counter() - start


def section_startup(args) -> None:
    available = multiprocessing.get_all_start_methods()
    log(f"available start methods: {available}")
    for name in ("spawn", "fork"):
        if name not in available:
            log(f"  {name:6s} unavailable")
            continue
        empty = _run_pool(name, _init_torch, (), [0], _trivial, workers=1)
        log(f"  {name:6s} import torch + 1 trivial task      {empty:8.3f} s wall")
        full = _run_pool(name, _init_torch_model_env, (str(args.resource_dir),), [0], _trivial,
                         workers=1)
        log(f"  {name:6s} + model + env + first reset        {full:8.3f} s wall")


def section_import(args) -> None:
    for name in ("spawn", "fork"):
        try:
            seconds = _run_pool(name, _init_torch, (), list(range(4)), _trivial, workers=4)
        except Exception as error:            # noqa: BLE001 - report, do not mask
            log(f"  {name:6s} FAILED: {type(error).__name__}: {error}")
            continue
        log(f"  {name:6s} 4 workers, import torch only         {seconds:8.3f} s wall")


def section_fork(args) -> None:
    """Fork after torch is already imported in the parent: the real training case."""
    import torch

    log(f"  parent has torch {torch.__version__} imported: pid {__import__('os').getpid()}")
    seconds = _run_pool("fork", _init_torch, (), list(range(4)), _trivial, workers=4)
    log(f"  fork    4 workers after torch import         {seconds:8.3f} s wall")


def section_pipe(args) -> None:
    for kib in args.payload_kib:
        rng = np.random.default_rng(0)
        blob = rng.integers(0, 256, size=int(kib * 1024), dtype=np.uint8).tobytes()
        seconds = _run_pool("spawn", _init_blob, (blob,), list(range(args.tasks)),
                            _return_blob, workers=args.workers)
        log(f"  pipe: {kib:6.1f} KiB/result, {args.tasks} results   {seconds:8.3f} s wall"
            f"   ({len(blob) * args.tasks / 1024**2:6.1f} MiB, "
            f"{len(blob) * args.tasks / 1024**2 / seconds:5.0f} MiB/s)")


def section_disk(args) -> None:
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        write = _run_pool("spawn", _init_torch, (),
                          [(seed, str(directory)) for seed in range(args.tasks)], _write_shard,
                          workers=args.workers)
        log(f"  disk: worker writes an npz shard             {write:8.3f} s wall")
        start = time.perf_counter()
        total = 0
        for seed in range(args.tasks):
            total += len((directory / f"seed_{seed}.npz").read_bytes())
        read = time.perf_counter() - start
        log(f"  disk: parent re-reads every shard            {read:8.3f} s wall (serial)"
            f"   ({total / 1024**2:6.1f} MiB)")
        log(f"  disk total                                   {write + read:8.3f} s wall")


def section_metadata(args) -> None:
    metadata = {
        "run_number": 1, "update": 1, "model_state_sha256": "0" * 64,
        "assignments": {str(job): {"task_id": f"task-{job % 20:02d}",
                                   "task_seed": job % 64,
                                   "action_seed": 1000 + job}
                        for job in range(args.tasks)},
        "manifest_sha256": "0" * 64, "max_actions": 4000, "potential": "phi-v1",
        "rollout_device": "cpu", "rollout_threads": 1,
    }
    blob = pickle.dumps(metadata, protocol=5)
    log(f"  metadata dict                                {len(blob) / 1024:8.1f} KiB pickled")
    log(f"  re-sent once per task (chunksize=1)          "
        f"{len(blob) * args.tasks / 1024**2:8.1f} MiB serialized per update")
    start = time.perf_counter()
    for _ in range(args.tasks):
        pickle.loads(pickle.dumps(metadata, protocol=5))
    seconds = time.perf_counter() - start
    log(f"  measured cost of that round trip             {seconds:8.3f} s per update")


SECTIONS = {
    "startup": section_startup,
    "import": section_import,
    "fork": section_fork,
    "pipe": section_pipe,
    "disk": section_disk,
    "metadata": section_metadata,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", choices=sorted(SECTIONS), required=True)
    parser.add_argument("--tasks", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=18)
    parser.add_argument("--payload-kib", type=float, nargs="+", default=[87.5],
                        help="result sizes to push through the pool pipe")
    parser.add_argument("--resource-dir", type=Path, default=DEFAULT_RESOURCES)
    parser.add_argument("--guard", type=float, default=120.0,
                        help="seconds before the process dumps its stack and exits")
    args = parser.parse_args()
    args.resource_dir = args.resource_dir.expanduser()
    # A hung pool is a result too: dump where it hung instead of blocking forever.
    faulthandler.dump_traceback_later(args.guard, exit=True)
    log(f"--- {args.section} (tasks={args.tasks} workers={args.workers}) ---")
    SECTIONS[args.section](args)


if __name__ == "__main__":
    main()
