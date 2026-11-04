"""task239: replay archived games on the v6 engine and export value-net features.

The stored compact frames do not carry plant timers, zombie phase/altitude/flying health/mind
control, wave timers, projectiles or grid-item counters.  v6 features are therefore produced by
the C++ engine (`EnvironmentValueNet` encoder 6) while the archived action tape is replayed; this
script only drives that existing replay and writes the engine's own feature trace.

Output is one `.npz` per terrain with:
  board    (N, 139, 60) float16   encoder-6 board channels
  scalars  (N, 125) float32       encoder-6 scalars (no task context)
  context  (N, 35)  float32       task context appended by the engine
  y        (N,)     int8          episode win
  seed, op, level, game

Usage:
  ~/pvz-torch-venv/bin/python scripts/task239_export.py \
      --corpus artifacts/task239/corpus --selection artifacts/task239/pilot_selection.tsv \
      --executable ~/pvz-mac-collect/build-vn6/pvz-portable \
      --value-bin artifacts/task239/tmp/random_v6.bin \
      --out artifacts/task239/features --per-game 64 --workers 4
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
import os
from pathlib import Path
import struct
import sys
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "scripts")]

from human_demo import replay  # noqa: E402
from pvz_episode_archive import read_meta  # noqa: E402

RESOURCE_DIR = Path.home() / ".cache" / "pvz-research-resources"
TERRAINS = ("day", "night", "pool", "fog", "roof")


def net_shape(path: Path) -> tuple[int, int]:
    """(board channels, scalars without context) from a PVZVN001 header."""
    with path.open("rb") as stream:
        magic, _width, version, _context = struct.unpack("<8sIII", stream.read(20))
    if magic != b"PVZVN001":
        raise ValueError(f"not a value network: {path}")
    extra = 47 if version >= 6 else 12 if version == 5 else 8 if version == 4 else 6 if version == 3 else 0
    scalars = 113 + (6 if version >= 2 else 0) + (6 if version >= 6 else 0)
    return 92 + extra, scalars


def sample_operations(count: int, per_game: int) -> list[int]:
    if count <= 0:
        return [0]
    return sorted({int(round(x)) for x in np.linspace(0, count, min(per_game, count + 1))})


def run_one(job: dict) -> dict:
    from pvz_event_env import EventWaitEnv

    archive = Path(job["archive"])
    meta = read_meta(archive)
    # `operations` is not reliable: several imports stored the final tick there.  The frame
    # tape is one frame per decision boundary, so its length is the operation count.
    operations = max(0, int(meta.get("frame_count") or 1) - 1)
    wanted = set(sample_operations(operations, job["per_game"]))
    trace = Path(job["tmp"]) / f"{job['index']}.bin"
    env = EventWaitEnv(str(RESOURCE_DIR), executable=job["executable"], headless=True)
    rows: list[tuple[int, np.ndarray]] = []
    try:
        env.load_value_net(job["value_bin"])

        def observe(operation: int, _obs: dict) -> None:
            if operation not in wanted:
                return
            env.leaf_values(trace_path=trace)
            rows.append((operation, np.fromfile(trace, dtype=np.float32)))

        obs, final_meta, _frames = replay(env, archive, on_observation=observe)
        won = bool(final_meta["outcome"]["won"])
        return dict(ok=True, index=job["index"], terrain=job["terrain"], seed=int(meta["task"]["seed"]),
                    level=int(meta["environment_task"]["level"]), won=won,
                    operations=operations, rows=rows)
    except Exception:
        return dict(ok=False, index=job["index"], archive=str(archive), error=traceback.format_exc()[-800:])
    finally:
        env.close()
        trace.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--value-bin", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-game", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    channels, scalars = net_shape(args.value_bin)
    selection = []
    for line in args.selection.read_text().splitlines():
        if not line.strip():
            continue
        terrain, relative, level, won, frames = line.split("\t")
        selection.append(dict(terrain=terrain, archive=str(args.corpus / relative), level=int(level),
                              recorded_won=bool(int(won)), recorded_frames=int(frames)))
    if args.limit:
        selection = selection[:args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    tmp = args.out / "tmp"
    tmp.mkdir(exist_ok=True)
    jobs = [dict(index=index, terrain=item["terrain"], archive=item["archive"], per_game=args.per_game,
                 executable=str(args.executable), value_bin=str(args.value_bin), tmp=str(tmp))
            for index, item in enumerate(selection)]
    print(f"{len(jobs)} games, board={channels} scalars={scalars} per_game={args.per_game}", flush=True)

    results = []
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        futures = [pool.submit(run_one, job) for job in jobs]
        for done, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            if not result["ok"]:
                print(f"[{done}/{len(jobs)}] FAILED {result['archive']}\n{result['error']}", flush=True)
            elif done % 10 == 0:
                print(f"[{done}/{len(jobs)}] ok", flush=True)

    ok = [result for result in results if result["ok"]]
    failed = [result for result in results if not result["ok"]]
    for terrain in TERRAINS:
        games = [result for result in ok if result["terrain"] == terrain]
        if not games:
            continue
        board = np.asarray([row for game in games for _op, row in game["rows"]], dtype=np.float32)
        expected = channels * 60 + scalars + 35
        if board.shape[1] != expected:
            raise RuntimeError(f"{terrain}: trace has {board.shape[1]} floats, expected {expected}")
        board6 = board[:, :channels * 60].reshape(-1, channels, 60).astype(np.float16)
        scalars6 = board[:, channels * 60:channels * 60 + scalars].astype(np.float32)
        context6 = board[:, channels * 60 + scalars:].astype(np.float32)
        seeds = np.asarray([game["seed"] for game in games for _op, _row in game["rows"]], dtype=np.int64)
        ops = np.asarray([op for game in games for op, _row in game["rows"]], dtype=np.int32)
        levels = np.asarray([game["level"] for game in games for _op, _row in game["rows"]], dtype=np.int16)
        labels = np.asarray([game["won"] for game in games for _op, _row in game["rows"]], dtype=np.int8)
        games_index = np.asarray([game["index"] for game in games for _op, _row in game["rows"]], dtype=np.int32)
        np.savez_compressed(args.out / f"{terrain}.npz", board=board6, scalars=scalars6, context=context6,
                            y=labels, seed=seeds, op=ops, level=levels, game=games_index)
        print(f"{terrain}: {len(games)} games, {len(board6)} rows, wins {int((labels == 1).sum())}", flush=True)

    summary = dict(games=len(ok), failed=len(failed), channels=channels, scalars=scalars,
                   per_game=args.per_game, failed_archives=[item.get("archive") for item in failed],
                   by_terrain={terrain: dict(games=sum(1 for x in ok if x["terrain"] == terrain)) for terrain in TERRAINS})
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n")
    tmp.rmdir()
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
