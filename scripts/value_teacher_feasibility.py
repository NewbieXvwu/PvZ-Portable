"""Task43 same-root labels, restored and connected to the current native teacher.

The old Python rollout/reference replay is retired. RunRollout returns its actual
student history, and terminal continuations call search_play._play itself.
"""
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np
import search_play as teacher


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, allow_nan=False) + "\n")
    tmp.replace(path)


def continuation_state(obs):
    """Transfer RunRollout history without replaying or recomputing features."""
    from pvz_student_observation import StudentHistory
    raw = obs["rollout_history"]
    history = StudentHistory(teacher._W["fast"].obs)
    history.wave = None if raw["wave"] < 0 else raw["wave"]
    history.wave_tick = raw["wave_tick"]
    history.plants = {tuple(row[:3]): row[3] for row in raw["plants"]}
    return dict(history=history,
                frame={"tick": raw["previous_tick"]} if raw["previous_tick"] >= 0 else None,
                action={"type": "plant" if raw["previous_plant"] else "wait"})


def finish_teacher(env, obs, state, args, seed_base):
    """Continue through the sole production game loop and its native search."""
    teacher._W["env"] = env
    result = teacher._play(dict(seed=0, future_seed=seed_base, arm="fast_search", args=args,
                               continuation=dict(observation=obs, state=state)))
    if not result["terminal"]:
        raise RuntimeError("teacher continuation reached action cap")
    return result


def sample_root(env, obs, state, args, decision_seed, stats, directory, samples=4):
    """Actual SEARCH candidates, common native futures, frozen teacher outcomes.

    Terminal leaves remain in reports and are excluded from value supervision.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    root = env.snapshot_fast()
    baseline = dict(teacher._W)
    trace = directory / "feature.bin"
    rows, features = [], []
    try:
        teacher._search(env, obs, state["action"], args, decision_seed, stats)
        targets = stats.get("last_targets") or []
        write_json(directory / "root.json", dict(decision_seed=decision_seed, tick=obs["tick"],
            wave=obs["wave"], candidates=[a for a, _ in targets], samples=samples))
        for candidate, (action, _) in enumerate(targets):
            for k in range(samples):
                future = (decision_seed * 1000 + k) % 2**32
                teacher._W.clear()
                teacher._W.update(baseline)
                env.restore_fast(root)
                leaf = env.rollout(action, obs["tick"] + args.horizon, snapshot_id=root,
                    seed=future, sim_skip=args.rollout_sim_skip,
                    student_history=deepcopy(state.get("history")))
                terminal = bool(leaf.get("terminal") or leaf.get("rollout_result", -1) >= 0)
                feature = None
                if not terminal:
                    history = leaf["rollout_history"]
                    leaf = env.observe()
                    leaf["rollout_history"] = history
                    env.leaf_values(trace_path=trace, income=leaf.get("sun_income_rate") or 0)
                    feature = np.fromfile(trace, dtype=np.float32)
                start = time.monotonic()
                outcome = (dict(terminal=True, won=leaf.get("rollout_result", leaf.get("result")) == 1)
                           if terminal else finish_teacher(env, leaf, continuation_state(leaf), args, future))
                row = dict(candidate=candidate, action=action, future=k, future_seed=future,
                           leaf_terminal=terminal, leaf_tick=leaf["tick"], won=outcome["won"],
                           terminal=outcome["terminal"], seconds=time.monotonic()-start)
                rows.append(row)
                if feature is not None:
                    features.append((len(rows)-1, feature))
                write_json(directory / "labels.json", rows)
        return rows, features
    finally:
        teacher._W.clear()
        teacher._W.update(baseline)
        env.restore_fast(root)
        env.release_snapshot(root)
        trace.unlink(missing_ok=True)
