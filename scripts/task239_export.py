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


def save_features(results, out, channels, scalars, suffix=""):
    for terrain in TERRAINS:
        games = sorted((r for r in results if r["ok"] and r["terrain"] == terrain),
                       key=lambda r: r["index"])
        if not games:
            continue
        features = np.asarray([row for game in games for _op, row in game["rows"]], dtype=np.float32)
        expected = channels * 60 + scalars + 35
        if features.shape[1] != expected:
            raise RuntimeError(f"{terrain}: trace has {features.shape[1]} floats, expected {expected}")
        columns = {name: np.asarray([game[key] for game in games for _ in game["rows"]], dtype=dtype)
                   for name, key, dtype in (("seed", "seed", np.int64), ("level", "level", np.int16),
                                            ("y", "won", np.int8), ("game", "index", np.int32))}
        columns["op"] = np.asarray([op for game in games for op, _ in game["rows"]], dtype=np.int32)
        columns["source"] = np.zeros(len(features), dtype=np.int8)
        columns["candidate"] = np.full(len(features), -1, dtype=np.int16)
        columns["future"] = np.full(len(features), -1, dtype=np.int16)
        columns["truth"] = columns["y"].astype(np.float32)
        columns["live_fraction"] = np.ones(len(features), dtype=np.float32)
        columns["terminal_win_fraction"] = np.zeros(len(features), dtype=np.float32)
        np.savez_compressed(out / f"{terrain}{suffix}.npz",
                            board=features[:, :channels * 60].reshape(-1, channels, 60).astype(np.float16),
                            scalars=features[:, channels * 60:channels * 60 + scalars],
                            context=features[:, channels * 60 + scalars:], **columns)
        print(f"{terrain}{suffix}: {len(games)} games, {len(features)} rows", flush=True)


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
    parser.add_argument("--shard-size", type=int, default=0,
                        help="games per bounded-memory export shard; 0 = one file per terrain")
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
        size = args.shard_size or max(1, len(jobs))
        for start in range(0, len(jobs), size):
            batch = []
            futures = [pool.submit(run_one, job) for job in jobs[start:start + size]]
            for done, future in enumerate(as_completed(futures), start + 1):
                result = future.result()
                batch.append(result)
                if not result["ok"]:
                    print(f"[{done}/{len(jobs)}] FAILED {result['archive']}\n{result['error']}", flush=True)
                elif done % 10 == 0:
                    print(f"[{done}/{len(jobs)}] ok", flush=True)
            save_features(batch, args.out, channels, scalars,
                          f"-{start // size:04d}" if args.shard_size else "")
            for result in batch:
                result.pop("rows", None)
                results.append(result)

    ok = [result for result in results if result["ok"]]
    failed = [result for result in results if not result["ok"]]
    summary = dict(games=len(ok), failed=len(failed), channels=channels, scalars=scalars,
                   per_game=args.per_game, failed_archives=[item.get("archive") for item in failed],
                   failures=failed,
                   by_terrain={terrain: dict(games=sum(1 for x in ok if x["terrain"] == terrain),
                                            wins=sum(x["won"] for x in ok if x["terrain"] == terrain))
                               for terrain in TERRAINS})
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n")
    tmp.rmdir()
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
