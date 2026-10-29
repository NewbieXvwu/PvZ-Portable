"""按档案实际卡组重放并回填 C++ 富输入，失配整局跳过，保留标签。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "scripts")]

from pvz_env import PlayerProfileContext, TaskSpec
from pvz_event_env import EventWaitEnv
from pvz_frame_store import frame_paths, read_frames, write_frames
from pvz_tasks import profile_for_deck
from render_episode import DEFAULT_RESOURCE_DIR, _frame, load_training_task, training_task_kwargs


def replay_archive(env, path, out, *, manifest=None, task_id=None):
    started = time.perf_counter()
    meta = json.loads((path.parent / "meta.json").read_text())
    frames = read_frames(path)
    index = 0
    try:
        if meta.get("demo_archive") or "_f" in path.parent.parent.name:
            raise ValueError("示范起点/重抽未来需要原示范与 RESEED 时序，本入口不回填")
        recorded = meta["task"]
        if "environment_task" in meta:
            config = dict(meta["environment_task"])
            config["profile"] = PlayerProfileContext(**config["profile"])
            config["preplanted"] = tuple(tuple(p) for p in config.get("preplanted", ()))
            task, deck = TaskSpec(**config), recorded["deck"]
        elif "deck" in recorded and "level" in recorded:
            # Use the actual recorded perturbed deck, never today's deck_for_level rule.
            deck = recorded["deck"]
            task = TaskSpec(level=recorded["level"], seed=recorded["seed"],
                           profile=profile_for_deck(tuple(deck), recorded.get("first_aid_owned", False)),
                           **{key: recorded[key] for key in ("playthrough", "zombie_count_multiplier",
                              "wave_cap", "preplanted", "conveyor_start", "fog_observation") if key in recorded})
        elif manifest and task_id:
            kw = training_task_kwargs(load_training_task(manifest, task_id), recorded["seed"])
            deck = kw["deck"]
            task = TaskSpec(level=kw["level"], seed=kw["seed"], playthrough=2,
                           profile=profile_for_deck(tuple(deck), kw.get("first_aid_owned", False)),
                           **{key: value for key, value in kw["task_extra"].items() if value is not None})
        else:
            raise ValueError("旧档案缺实际任务配置，需 --manifest 与 --task-id")
        if not isinstance(task.preplanted, tuple):
            from dataclasses import replace
            task = replace(task, preplanted=tuple(tuple(p) for p in task.preplanted))
        env.set_token_observation(True)
        obs, _ = env.reset(deck=tuple(deck), task=task)
        for index, record in enumerate(frames):
            if index:
                obs, *_rest, info = env.step(record["action"])
                rejected = not info.get("ok")
                if rejected != record.get("action_rejected", False):
                    raise ValueError(f"frame {index}: action rejection differs")
                if rejected:
                    obs, *_ = env.step({"type": "wait", "ticks": 60})
            actual = json.loads(json.dumps(_frame(obs)))
            for key, value in actual.items():
                # These observation extensions did not exist in early archives.
                if key in ("v2", "v3", "sleep", "boss", "bungee", "fog_amount") and key not in record:
                    continue
                if value != record.get(key):
                    raise ValueError(f"{path}: frame {index}, {key} differs from replay")
            record["v3"] = obs["v3"]
        destination = out / "frames" / path.parent.parent.name / path.parent.name
        destination.mkdir(parents=True, exist_ok=False)
        write_frames(destination / "frames.npz", frames)
        meta["observation_backfill"] = {"version": 3, "source": str(path.resolve())}
        (destination / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
        return dict(source=str(path), mismatch=False, frames_written=len(frames),
                    tokens=sum(len(frame["v3"]["tokens"]) for frame in frames),
                    seconds=time.perf_counter() - started)
    except ValueError as error:
        return dict(source=str(path), mismatch=True, frame=index, reason=str(error), frames_written=0,
                    seconds=time.perf_counter() - started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=3, help="抽样局数，默认只回填三局")
    parser.add_argument("--label")
    parser.add_argument("--manifest")
    parser.add_argument("--task-id")
    parser.add_argument("--executable", default=str(ROOT / "build/pvz-portable"))
    parser.add_argument("--resource-dir", default=DEFAULT_RESOURCE_DIR)
    args = parser.parse_args()
    if args.limit < 1 or args.out.resolve().is_relative_to(args.data.resolve()):
        parser.error("局数须为正，输出须在源目录外")
    paths = sorted(p for p in frame_paths(args.data) if not args.label or p.parent.name == args.label)[:args.limit]
    if not paths:
        parser.error("未找到档案")
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    with EventWaitEnv(args.resource_dir, executable=args.executable, headless=True) as env:
        for path in paths:
            result = replay_archive(env, path, args.out, manifest=args.manifest, task_id=args.task_id)
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    summary = dict(games=len(results), successes=sum(not r["mismatch"] for r in results),
                   seconds=sum(r["seconds"] for r in results), games_detail=results)
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
