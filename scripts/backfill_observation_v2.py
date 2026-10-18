"""精确重放旧档案，只补 v2 原始量。失配整局跳过，源档与标签保持原样。"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from pvz_frame_store import frame_paths, read_frames, write_frames
from relabel_verify import _check_frame
from render_episode import DEFAULT_RESOURCE_DIR, _frame, load_training_task, training_task_kwargs

_ARGS = _ENV = None


def _init(args):
    global _ARGS, _ENV
    from pvz_event_env import EventWaitEnv
    _ARGS = args
    _ENV = EventWaitEnv(args.resource_dir, executable=args.executable, headless=True)


def backfill(path: Path):
    from pvz_env import TaskSpec
    from scripted_baseline import profile_for_deck
    started = time.perf_counter()
    frames = read_frames(path)
    meta = json.loads((path.parent / "meta.json").read_text())
    seed = meta["task"]["seed"]
    kw = training_task_kwargs(load_training_task(_ARGS.manifest, _ARGS.task_id), seed)
    task = TaskSpec(level=kw["level"], seed=seed, playthrough=2,
                    profile=profile_for_deck(tuple(kw["deck"]), kw.get("first_aid_owned", False)),
                    **{key: value for key, value in kw["task_extra"].items() if value is not None})
    obs, _ = _ENV.reset(deck=tuple(kw["deck"]), task=task)
    index = 0
    try:
        for index, record in enumerate(frames):
            if index:
                obs, *_rest, info = _ENV.step(record["action"])
                rejected = not info.get("ok")
                if rejected != record.get("action_rejected", False):
                    raise ValueError(f"{path}: frame {index}, action rejection differs from replay")
                if rejected:
                    obs, *_ = _ENV.step({"type": "wait", "ticks": 60})
            _check_frame(obs, record, path, index)
            record["v2"] = _frame(obs)["v2"]
    except ValueError as error:
        return {"seed": seed, "source": str(path), "mismatch": True, "frame": index,
                "reason": str(error), "frames_checked": index + 1, "frames_written": 0,
                "seconds": time.perf_counter() - started}
    destination = _ARGS.out / "frames" / path.parent.parent.name / path.parent.name
    destination.mkdir(parents=True, exist_ok=False)
    write_frames(destination / "frames.npz", frames)
    meta["observation_backfill"] = {"version": 2, "source": str(path.resolve()),
                                    "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (destination / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return {"seed": seed, "source": str(path), "mismatch": False, "frames_checked": len(frames),
            "frames_written": len(frames), "seconds": time.perf_counter() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--executable", default=str(ROOT / "build/pvz-portable"))
    parser.add_argument("--resource-dir", default=DEFAULT_RESOURCE_DIR)
    parser.add_argument("--seeds", type=int, default=0, help="0 表示全部")
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.out.resolve().is_relative_to(args.data.resolve()):
        parser.error("--out 必须在源档案目录外")
    if args.workers < 1 or args.seeds < 0 or args.seed_offset < 0:
        parser.error("进程数必须为正，局数和起始位置不能为负")
    paths = sorted((path for path in frame_paths(args.data) if path.parent.name == args.label),
                   key=lambda path: int(path.parent.parent.name.removeprefix("seed_")))
    paths = paths[args.seed_offset:args.seed_offset + args.seeds if args.seeds else None]
    if not paths or (args.seeds and len(paths) != args.seeds):
        parser.error("请求的档案范围不存在")
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    started = time.perf_counter()
    with (args.out / "games.jsonl").open("x") as output, mp.get_context("spawn").Pool(
            args.workers, initializer=_init, initargs=(args,)) as pool:
        for result in pool.imap_unordered(backfill, paths):
            results.append(result)
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()
            print(json.dumps(result, ensure_ascii=False), flush=True)
    seconds = time.perf_counter() - started
    summary = {"games": len(results), "mismatch_games": sum(result["mismatch"] for result in results),
               "frames_checked": sum(result["frames_checked"] for result in results),
               "frames_written": sum(result["frames_written"] for result in results), "seconds": seconds}
    summary.update(mismatch_rate=summary["mismatch_games"] / summary["games"],
                   seconds_per_game=seconds / summary["games"], frames_per_second=summary["frames_written"] / seconds)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
