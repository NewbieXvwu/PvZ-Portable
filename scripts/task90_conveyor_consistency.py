"""Compare Python and native student continuations from three L30 roots."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "scripts")]


def _packets(frame):
    return [list(packet) for packet in frame.get("packets") or []]


def _task(manifest, task_id, seed):
    from pvz_env import TaskSpec
    from pvz_tasks import profile_for_deck
    from render_episode import load_training_task, training_task_kwargs

    definition = load_training_task(manifest, task_id)
    kw = training_task_kwargs(definition, seed)
    return kw, TaskSpec(level=kw["level"], seed=kw["seed"], playthrough=2,
                        profile=profile_for_deck(tuple(kw["deck"]), kw.get("first_aid_owned", False)),
                        **{key: value for key, value in kw["task_extra"].items() if value is not None})


def _replay_to(env, frames, point, kw, task):
    obs, _ = env.reset(deck=tuple(kw["deck"]), task=task)
    for index in range(point):
        action = frames[index + 1].get("action")
        obs, _reward, done, _truncated, info = env.step(action)
        if not info.get("ok"):
            obs, _reward, done, _truncated, info = env.step({"type": "wait", "ticks": 60})
        if done:
            raise RuntimeError(f"archive terminated before decision {point}")
        from render_episode import _frame
        if _packets(_frame(obs)) != _packets(frames[index + 1]):
            raise RuntimeError(f"archive conveyor packets diverged at decision {index + 1}")
    from render_episode import _frame
    if _packets(_frame(obs)) != _packets(frames[point]):
        raise RuntimeError(f"archive conveyor packets diverged at decision {point}")
    return obs


def _continuation(env, root, obs, first, until, future_seed, history, trace_path):
    import search_play
    from copy import deepcopy
    from render_episode import _frame

    native = env.rollout(first, until, snapshot_id=root, seed=future_seed,
                         student_history=history, trace_path=trace_path)
    env.observe()
    native_frames = [_frame(json.loads(line)["observation"])
                     for line in trace_path.read_text().splitlines()]

    env.restore(root)
    env.reseed(future_seed)
    actual_frames = [_frame(env.observe())]
    original_step = env.step

    def record(action):
        result = original_step(action)
        actual_frames.append(_frame(result[0]))
        return result

    env.step = record
    try:
        search_play._W.update(student_history=deepcopy(history), batch_waits=False,
                              lean_rollouts=False, rollout_sim_skip=0)
        search_play._python_rollout_reference(env, obs, first, until, root=None, seed=None)
    finally:
        env.step = original_step

    return actual_frames, native_frames, native


def run(args):
    from pvz_event_env import EventWaitEnv
    from pvz_frame_store import read_frames
    from render_episode import _frame
    from pvz_student_observation import StudentHistory
    import search_play

    search_play._W.clear()
    search_play._load_nets(None, str(args.student))
    net = search_play._W["fast"]
    points = [(91000000, 80), (91000001, 120), (91000002, 160)]
    result = {"horizon": 3000, "points": []}
    env = EventWaitEnv(args.resource_dir, executable=args.executable, headless=True)
    env.load_rollout_net(search_play._W["rollout_file"])
    try:
        for seed, point in points:
            archive = args.frames / f"seed_{seed}" / "task90_fast_search" / "frames.npz"
            frames = read_frames(archive)
            kw, task = _task(args.manifest, "task90_l30_orig_m100", seed)
            obs = _replay_to(env, frames, point, kw, task)
            history = StudentHistory(net.obs) if net.obs != "v1" else None
            if history is not None:
                for frame in frames[:point]:
                    history.extra(frame)
            if env._sun_history_start_tick is None:
                env._sun_history_start_tick = 0
            root = env.snapshot_fast()
            root_tick = int(obs["tick"])
            future_seed = 92000000 + len(result["points"])
            trace = args.out / f"native_{seed}_{point}.jsonl"
            trace.parent.mkdir(parents=True, exist_ok=True)
            actual, native, leaf = _continuation(
                env, root, obs, frames[point + 1]["action"], root_tick + args.horizon,
                future_seed, history, trace)
            env.release_snapshot(root)
            actual_by_tick = {int(frame["tick"]): frame for frame in actual}
            native_by_tick = {int(frame["tick"]): frame for frame in native}
            common = sorted(set(actual_by_tick) & set(native_by_tick))
            mismatches = [tick for tick in common
                          if _packets(actual_by_tick[tick]) != _packets(native_by_tick[tick])]
            result["points"].append({
                "seed": seed, "decision": point, "root_tick": root_tick,
                "future_seed": future_seed, "until_tick": root_tick + args.horizon,
                "actual_observations": len(actual), "native_observations": len(native),
                "common_ticks": len(common), "packet_mismatches": len(mismatches),
                "first_mismatch_tick": mismatches[0] if mismatches else None,
                "native_trace": str(trace), "native_leaf_tick": int(leaf["tick"]),
            })
    finally:
        env.close()
        search_play._W.clear()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--resource-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=3000)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
