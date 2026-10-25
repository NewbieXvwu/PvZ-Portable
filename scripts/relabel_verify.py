"""Replay archived actions exactly, then verify only labels that override the collecting student.

Use --check-only to measure candidate reproduction without writing archives.
New archives live under OUT/frames/seed_N/NEW_LABEL;
train_rollout_policy.py reads them with --data OUT/frames --labels NEW_LABEL.
The collecting student and all original search settings must be supplied unchanged.
Original means and snapshots are recovered before any added verification futures.
--on-mismatch fresh discards those means and resamples only the student/original-label pair.
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
    keep, paths = search_play._export_values(args.value, args.static_value)
    search_play._W["value_keep"] = keep
    search_play._init_worker(paths[0], args.rollout_policy, args.rollout_sim_skip, paths[1])
    from pvz_event_env import EventWaitEnv
    env = EventWaitEnv(args.resource_dir, executable=args.executable, headless=True)
    env.load_rollout_net(search_play._W["rollout_file"])
    env.load_value_net(paths[0])
    env.load_value_net(paths[1], static=True)
    search_play._W["env"] = env


def _check_frame(obs, expected, path, index):
    from render_episode import _frame
    actual = json.loads(json.dumps(_frame(obs)))
    for key, value in actual.items():
        if key == "v2" and key not in expected:
            continue
        if value != expected.get(key):
            raise ValueError(f"{path}: frame {index}, {key} differs from replay")


def _relabel(path: Path):
    import torch
    from label_noise import action_key
    from pvz_env import TaskSpec
    from pvz_frame_store import read_frames, write_frames
    from pvz_value_net import observation_context
    from render_episode import load_training_task, training_task_kwargs
    from pvz_tasks import profile_for_deck
    args = _ARGS
    started = time.perf_counter()
    frames = read_frames(path)
    meta = json.loads((path.parent / "meta.json").read_text())
    seed = meta["task"]["seed"]
    kw = training_task_kwargs(load_training_task(args.manifest, args.task_id), seed)
    task = TaskSpec(level=kw["level"], seed=kw["seed"], playthrough=2,
                    profile=profile_for_deck(tuple(kw["deck"]), kw.get("first_aid_owned", False)),
                    **{key: value for key, value in kw["task_extra"].items() if value is not None})
    env = search_play._W["env"]
    originals = {}
    roots = {}
    mismatches = []
    counts = Counter(searches=0, searches_checked=0, overrides=0, reproduced=0, fresh=0, cancelled=0)
    search_args = argparse.Namespace(**{**vars(args), "verify_samples": 0})
    search_seconds = verification_seconds = 0.0
    torch.manual_seed(seed)
    obs, _ = env.reset(deck=tuple(kw["deck"]), task=task)
    search_play._W["context"] = observation_context(obs)
    _check_frame(obs, frames[0], path, 0)
    state = {}
    for index in range(1, len(frames)):
        record = frames[index]
        student = search_play._fast_action(obs, state)
        label = record.get("label", record["action"])
        override = action_key(label) != action_key(student)
        if record.get("targets"):
            counts["searches"] += 1
            counts["overrides"] += override
            if override or args.check_only:
                counts["searches_checked"] += 1
                stats = {"searched": 0, "overrides": 0, "override_kinds": Counter()}
                before_search = time.perf_counter()
                chosen = search_play._search(env, obs, student, search_args, seed * 10000 + index - 1, stats)
                search_seconds += time.perf_counter() - before_search
                targets = [[action.get("packet", -1), action.get("row", -1), action.get("col", -1),
                            round(value, 4)]
                           for action, value in stats["last_targets"] if abs(value) != float("inf")]
                reason = ("override" if action_key(chosen) != action_key(label) else
                          "candidate_values" if targets != record["targets"] else None)
                if reason:
                    mismatches.append({"frame": index, "reason": reason,
                                       "expected": record["targets"], "actual": targets})
                    if args.on_mismatch == "fail" and not args.check_only:
                        raise ValueError(f"{path}: frame {index}, original {reason} did not reproduce")
                else:
                    counts["reproduced"] += 1
                if override:
                    originals[index] = None if reason else stats["last_targets"]
                if not args.check_only:
                    roots[index] = (env.snapshot(), obs, student)
        executed = record["action"]
        state["action"] = executed
        obs, _reward, _done, _truncated, info = env.step(executed)
        rejected = not info.get("ok")
        if rejected != record.get("action_rejected", False):
            raise ValueError(f"{path}: frame {index}, action rejection differs from replay")
        if rejected:
            obs, _reward, _done, _truncated, _info = env.step({"type": "wait", "ticks": 60})
        _check_frame(obs, record, path, index)
    for index, (root, obs, student) in roots.items():
        record = frames[index]
        label = record.get("label", record["action"])
        fresh = originals[index] is None
        if fresh:
            candidates = [student if student["type"] == "plant" else search_play.NOOP, label]
            totals = [0.0, 0.0]
        else:
            candidates, totals = map(list, zip(*originals[index]))
        keys = [action_key(action) for action in candidates]
        stats = {"overrides": 0, "override_kinds": Counter()}
        env.restore(root)
        try:
            chosen = search_play._verified_choice(
                env, obs, root, candidates, totals, [args.samples] * len(candidates),
                keys.index(action_key(label)), keys.index(action_key(student)), student,
                args, seed * 10000 + index - 1, stats, fresh=fresh)
        finally:
            env.release_snapshot(root)
        verification = stats["last_verification"]
        counts["fresh"] += fresh
        counts["cancelled"] += action_key(chosen) == action_key(student)
        verification_seconds += verification["seconds"]
        record["verification"] = verification
        record["targets"] = [[action.get("packet", -1), action.get("row", -1), action.get("col", -1),
                              round(value, 4)] for action, value in stats["last_targets"]
                             if abs(value) != float("inf")]
        record["label"] = chosen
    counts.update(mismatches=len(mismatches), mismatch_games=int(bool(mismatches)),
                  mismatch_values=sum(item["reason"] == "candidate_values" for item in mismatches),
                  mismatch_override=sum(item["reason"] == "override" for item in mismatches),
                  retained=0 if args.check_only else counts["overrides"] - counts["cancelled"])
    if not args.check_only:
        destination = args.out / "frames" / f"seed_{seed}" / args.new_label
        destination.mkdir(parents=True, exist_ok=False)
        write_frames(destination / "frames.npz", frames)
        meta["relabel"] = {"source": str(path.resolve()), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                           "collecting_policy": args.rollout_policy, "verify_samples": args.verify_samples,
                           "on_mismatch": args.on_mismatch, **counts}
        (destination / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return {"seed": seed, "frames_checked": len(frames), **counts, "mismatch_details": mismatches,
            "seconds": time.perf_counter() - started,
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
    parser.add_argument("--on-mismatch", choices=("fail", "fresh"), default="fail")
    args = parser.parse_args()
    args.min_gap = 0
    if args.verify_samples <= 0:
        parser.error("--verify-samples must be positive")
    if Path(args.new_label).name != args.new_label or args.new_label in (".", ".."):
        parser.error("--new-label must be a directory name")
    if args.out.resolve().is_relative_to(args.data.resolve()):
        parser.error("--out must be outside the source archives")
    from pvz_frame_store import archive_seed, frame_paths
    paths = [path for path in frame_paths(args.data) if path.parent.name == args.label]
    paths.sort(key=lambda path: archive_seed(path.parent.parent.name))
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
               for key in ("frames_checked", "searches", "searches_checked", "overrides", "reproduced", "fresh",
                           "cancelled", "retained", "mismatches", "mismatch_games", "mismatch_values",
                           "mismatch_override", "seconds", "search_seconds",
                           "verification_seconds")}}
    summary["seconds_per_game"] = summary["seconds"] / summary["games"]
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
