"""按档案实际卡组重放并回填 C++ 富输入，失配整局跳过，保留标签。"""
from __future__ import annotations

import argparse
import json
import os
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


def replay_archive(env, path, out, *, manifest=None, task_id=None, inline=False, capture=True):
    started = time.perf_counter()
    meta = json.loads((path.parent / "meta.json").read_text())
    frames = read_frames(path)
    index = 0
    writer = None
    from pvz_research_data import SparseTokenWriter, SIDECAR_NAME
    from pvz_student_tokens import frame_tokens
    try:
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
        env.set_token_observation(capture and not meta.get('demo_archive'))
        if meta.get('demo_archive'):
            from human_demo import replay
            demo = Path(meta['demo_archive'])
            if not demo.is_dir():
                raise ValueError('original demonstration is unavailable: '+str(demo))
            obs, _, _ = replay(env, demo, decision=meta['demo_decision'], fog_observation=task.fog_observation)
            tag = path.parent.parent.name
            if '_f' in tag:
                env.reseed(int(tag.rsplit('_f',1)[1]))
            if capture:
                env.set_token_observation(True)
                obs = env.observe()
        elif '_f' in path.parent.parent.name:
            raise ValueError('RESEED archive is missing its demonstration provenance')
        else:
            obs, _ = env.reset(deck=tuple(deck), task=task)
        if capture and not inline:
            writer = SparseTokenWriter(out)
        replay_started = time.perf_counter()
        for index, record in enumerate(frames):
            if index:
                action = record.get("action") or record.get("input_action")
                obs, *_rest, info = env.step(action)
                rejected = not info.get("ok")
                if rejected != record.get("action_rejected", False):
                    raise ValueError(f"frame {index}: action rejection differs")
                if rejected and meta.get("schema") != "episode_archive_v1":
                    obs, *_ = env.step({"type": "wait", "ticks": 60})
            actual = json.loads(json.dumps(_frame(obs)))
            for key, value in actual.items():
                # These observation extensions did not exist in early archives.
                if key in ("v2", "v3", "sleep", "boss", "bungee", "fog_amount") and key not in record:
                    continue
                # Human capture polls each tick. Grouped waits preserve the board, but move
                # the Python rolling-income event timestamps. v3 does not consume that estimate.
                if key == "sun_income_rate" and meta.get("schema") == "episode_archive_v1":
                    continue
                if value != record.get(key):
                    raise ValueError(f"{path}: frame {index}, {key} differs from replay")
            if inline:
                record['v3'] = obs['v3']
                record.setdefault('v2', obs['v2'])
            elif capture:
                writer.append(frame_tokens(actual, token_input='obs_v3'))
        replay_seconds = time.perf_counter() - replay_started
        if not capture:
            return dict(source=str(path), mismatch=False, frames_written=0, frames_read=len(frames),
                        bytes_written=0,replay_seconds=replay_seconds,seconds=time.perf_counter()-started)
        destination = out / "frames" / path.parent.parent.name / path.parent.name
        destination.mkdir(parents=True, exist_ok=False)
        if inline:
            write_frames(destination / 'frames.npz', frames)
        else:
            # Preserve the compact archive verbatim; never rewrite the production source.
            os.link(path, destination/path.name)
            write_started = time.perf_counter()
            token_meta = writer.finish(destination/SIDECAR_NAME, source_frame=destination/path.name)
            write_seconds = time.perf_counter()-write_started
        meta["observation_backfill"] = {"version": 3, "source": str(path.resolve())}
        (destination / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
        return dict(source=str(path), mismatch=False, frames_written=len(frames),
                    tokens=sum(len(frame['v3']['tokens']) for frame in frames) if inline else token_meta['tokens'],
                    max_tokens=max(len(frame['v3']['tokens']) for frame in frames) if inline else token_meta['max_tokens'],
                    bytes_written=(destination/'frames.npz').stat().st_size if inline else (destination/SIDECAR_NAME).stat().st_size,
                    replay_seconds=replay_seconds, write_seconds=0 if inline else write_seconds,
                    index_kind='inline' if inline else 'frame',
                    seconds=time.perf_counter() - started)
    except (ValueError, RuntimeError, OSError, EOFError, KeyError) as error:
        if writer is not None:
            writer.abort()
        return dict(source=str(path), mismatch=True, frame=index, reason=str(error), frames_written=0,
                    seconds=time.perf_counter() - started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=3, help="抽样局数，默认只回填三局")
    parser.add_argument("--inline", action="store_true", help="legacy inline v3 output; default writes a separate compressed sparse attachment")
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
            if env._process is None or env._process.poll() is not None:
                env.close()
            result = replay_archive(env, path, args.out, manifest=args.manifest, task_id=args.task_id, inline=args.inline)
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    summary = dict(games=len(results), successes=sum(not r["mismatch"] for r in results),
                   seconds=sum(r["seconds"] for r in results), games_detail=results)
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
