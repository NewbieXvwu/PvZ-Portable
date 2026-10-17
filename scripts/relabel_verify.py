"""Replay archived actions exactly, then verify only labels that override the collecting student.

Use --check-only first. New archives live under OUT/frames/seed_N/NEW_LABEL;
train_rollout_policy.py reads them with --data OUT/frames --labels NEW_LABEL.
The collecting student and all original search settings must be supplied unchanged.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

import search_play

_ARGS = None


def _init(args):
    global _ARGS
    _ARGS = args
    search_play._init_worker(args.value, args.rollout_policy, args.rollout_sim_skip, True, args.static_value)
    from pvz_event_env import EventWaitEnv
    env = EventWaitEnv(args.resource_dir, executable=args.executable, headless=True)
    env.load_rollout_net(search_play._W["rollout_file"])
    search_play._W["env"] = env


def _check_frame(obs, expected, path, index):
    from render_episode import _frame
    actual = json.loads(json.dumps(_frame(obs)))
    for key, value in actual.items():
        if value != expected.get(key):
            raise ValueError(f"{path}: frame {index}, {key} differs from replay")


def _relabel(path: Path):
    import torch
    from label_noise import action_key
    from pvz_env import TaskSpec
    from pvz_frame_store import read_frames, write_frames
    from pvz_value_net import observation_context
    from render_episode import load_training_task, training_task_kwargs
    from scripted_baseline import profile_for_deck
    args = _ARGS
    started = time.perf_counter()
    frames = read_frames(path)
    meta = json.loads((path.parent / "meta.json").read_text())
    seed = meta["task"]["seed"]
    kw = training_task_kwargs(load_training_task(args.manifest, args.task_id), seed)
    task = TaskSpec(level=kw["level"], seed=kw["seed"], playthrough=2,
                    profile=profile_for_deck(tuple(kw["deck"])),
                    **{key: value for key, value in kw["task_extra"].items() if value is not None})
    env = search_play._W["env"]
    torch.manual_seed(seed)
    obs, _ = env.reset(deck=tuple(kw["deck"]), task=task)
    search_play._W["context"] = observation_context(obs)
    _check_frame(obs, frames[0], path, 0)
    state = {}
    overrides = cancelled = searches = 0
    search_seconds = verification_seconds = 0.0
    for index in range(1, len(frames)):
        record = frames[index]
        student = search_play._fast_action(obs, state)
        label = record.get("label", record["action"])
        if record.get("targets"):
            searches += 1
            if action_key(label) != action_key(student):
                overrides += 1
                if not args.check_only:
                    stats = {"searched": 0, "overrides": 0, "override_kinds": Counter()}
                    before_search = time.perf_counter()
                    chosen = search_play._search(env, obs, student, args, seed * 10000 + index - 1, stats)
                    search_seconds += time.perf_counter() - before_search
                    verification = stats.get("last_verification")
                    if verification is None or action_key(verification["challenger"]) != action_key(label):
                        raise ValueError(f"{path}: frame {index}, original override did not reproduce")
                    original = dict(zip((action_key(student), action_key(label)), verification["before_values"]))
                    targets = [[action.get("packet", -1), action.get("row", -1), action.get("col", -1),
                                round(original.get(action_key(action), value), 4)]
                               for action, value in stats["last_targets"] if abs(value) != float("inf")]
                    if targets != record["targets"]:
                        raise ValueError(f"{path}: frame {index}, original candidate values did not reproduce")
                    cancelled += action_key(chosen) == action_key(student)
                    verification_seconds += verification["seconds"]
                    record["verification"] = verification
                    record["targets"] = [[action.get("packet", -1), action.get("row", -1), action.get("col", -1),
                                          round(value, 4)] for action, value in stats["last_targets"]
                                         if abs(value) != float("inf")]
                    record["label"] = chosen
        executed = record["action"]
        state["action"] = executed
        obs, _reward, _done, _truncated, info = env.step(executed)
        rejected = not info.get("ok")
        if rejected != record.get("action_rejected", False):
            raise ValueError(f"{path}: frame {index}, action rejection differs from replay")
        if rejected:
            obs, _reward, _done, _truncated, _info = env.step({"type": "wait", "ticks": 60})
        _check_frame(obs, record, path, index)
    if not args.check_only:
        destination = args.out / "frames" / f"seed_{seed}" / args.new_label
        destination.mkdir(parents=True, exist_ok=False)
        write_frames(destination / "frames.npz", frames)
        meta["relabel"] = {"source": str(path.resolve()), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                           "collecting_policy": args.rollout_policy, "verify_samples": args.verify_samples,
                           "searches": searches, "overrides": overrides, "cancelled": cancelled}
        (destination / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return {"seed": seed, "frames_checked": len(frames), "searches": searches, "overrides": overrides,
            "cancelled": cancelled, "seconds": time.perf_counter() - started,
            "search_seconds": search_seconds, "verification_seconds": verification_seconds}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--new-label", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rollout-policy", required=True)
    parser.add_argument("--value", required=True)
    parser.add_argument("--static-value")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--executable", default=str(ROOT / "build/pvz-portable"))
    parser.add_argument("--resource-dir", default=search_play.DEFAULT_RESOURCE_DIR)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--verify-samples", type=int, required=True)
    parser.add_argument("--horizon", type=int, default=2500)
    parser.add_argument("--margin", type=float, default=.02)
    parser.add_argument("--halving", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--per-card", type=int, default=1)
    parser.add_argument("--rollout-sim-skip", type=int, default=61)
    parser.add_argument("--seeds", type=int, default=32)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    args.min_gap = 0
    if args.verify_samples <= 0:
        parser.error("--verify-samples must be positive")
    if Path(args.new_label).name != args.new_label or args.new_label in (".", ".."):
        parser.error("--new-label must be a directory name")
    if args.out.resolve().is_relative_to(args.data.resolve()):
        parser.error("--out must be outside the source archives")
    from pvz_frame_store import frame_paths
    paths = [path for path in frame_paths(args.data) if path.parent.name == args.label]
    paths.sort(key=lambda path: int(path.parent.parent.name.removeprefix("seed_")))
    paths = paths[args.seed_offset:args.seed_offset + args.seeds]
    if len(paths) != args.seeds:
        parser.error("requested archive range is unavailable")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n")
    results = []
    with (args.out / "games.jsonl").open("x") as output, mp.get_context("spawn").Pool(
            args.workers, initializer=_init, initargs=(args,)) as pool:
        for result in pool.imap_unordered(_relabel, paths):
            results.append(result)
            output.write(json.dumps(result) + "\n")
            output.flush()
            print(json.dumps(result), flush=True)
    summary = {"games": len(results), **{key: sum(result[key] for result in results)
               for key in ("frames_checked", "searches", "overrides", "cancelled", "seconds", "search_seconds",
                           "verification_seconds")}}
    summary["seconds_per_game"] = summary["seconds"] / summary["games"]
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
