"""Compare leaf networks at the first differing decision on a shared teacher prefix.

All actions, feature encoding, continuations and scores use the existing C++ interfaces.
This is an offline diagnostic; it never changes the production teacher.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "python")]
import search_play as play


class Found(Exception):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--candidate", default="v6")
    parser.add_argument("--data", type=Path, default=ROOT / "artifacts/task244")
    args = parser.parse_args()
    base = args.data.resolve()
    (base / "work").mkdir(exist_ok=True)
    models = base / "models"
    os.environ.update(PVZ_NO_POPUP="1", TMPDIR=str(base / "work"),
                      OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    import tempfile
    tempfile.tempdir = str(base / "work")
    from export_value_net import export_value_net
    from pvz_value_net import load_value_net
    paths = {}
    for name in ("production", "static", args.candidate):
        paths[name] = str(base / "work" / f"{args.seed}-{args.candidate}-{name}.bin")
        export_value_net(load_value_net(models / f"{name}.pt"), paths[name])
    task = next(t for t in json.loads((base / "panel-original.json").read_text())["tasks"]
                if t["level"] == args.level)
    game_args = play.parser().parse_args([
        "--rollout-policy", str(models / "student.pt"), "--value", str(models / "production.pt"),
        "--manifest", str(base / "panel-original.json"), "--task-id", task["task_id"],
        "--out", str(base / "work"), "--verify-samples", "9", "--student-candidates", "2",
        "--native-games", "--executable", str(Path.home() / "pvz-mac-collect/build-vn6/pvz-portable")])
    play._load_nets(paths["production"], str(models / "student.pt"), paths["static"])
    play._init_worker(paths["production"], str(models / "student.pt"), 61, paths["static"])
    original = play._search
    prefix = []

    def compare(env, obs, policy, config, decision_seed, stats):
        responses = {}
        # _search restores the common root. Combo/save/shovel are disabled, so there is
        # no pending search state to transfer between these diagnostic calls.
        for name in (args.candidate, "production"):
            env.load_value_net(paths[name])
            local = dict(stats)
            action = original(env, obs, policy, config, decision_seed, local)
            responses[name] = dict(action=action, targets=local.get("last_targets"),
                                   verification=local.get("last_verification"))
            if name == "production":
                stats.update(local)
        prefix.append(dict(decision=decision_seed % 10000, tick=obs["tick"],
                           production=responses["production"]["action"],
                           candidate=responses[args.candidate]["action"]))
        if responses[args.candidate]["action"] == responses["production"]["action"]:
            return responses["production"]["action"]
        root = env.snapshot_fast()
        candidates = [a for a, _ in responses["production"]["targets"]]
        rows = []
        try:
            for action in candidates:
                futures = []
                for k in range(12):
                    future = (decision_seed * 1000 + k) % 2**32
                    env.restore_fast(root)
                    leaf = env.rollout(action, obs["tick"] + config.horizon,
                        snapshot_id=root, seed=future, sim_skip=config.rollout_sim_skip,
                        student_history=play._W.get("student_history"))
                    terminal = leaf.get("terminal") or leaf.get("rollout_result", -1) >= 0
                    values = {}
                    for name in ("production", args.candidate):
                        env.load_value_net(paths[name])
                        values[name] = (float(leaf.get("rollout_result", int(leaf.get("result") == 1)))
                                        if terminal else env.leaf_values(income=leaf.get("sun_income_rate") or 0)[0])
                    futures.append(dict(future=k, tick=leaf["tick"], terminal=terminal, values=values,
                        plants=leaf.get("plants"), sun=leaf.get("sun")))
                rows.append(dict(action=action, futures=futures))
        finally:
            env.restore_fast(root)
            env.release_snapshot(root)
        record = dict(level=args.level, seed=args.seed, candidate=args.candidate,
                      decision=decision_seed % 10000, tick=obs["tick"], policy=policy,
                      root=obs, searches=responses, candidates=rows, prefix=prefix)
        out = base / f"divergence-l{args.level}-{args.seed}-{args.candidate}.json"
        out.write_text(json.dumps(record, indent=1, allow_nan=False) + "\n")
        print(out, flush=True)
        raise Found

    play._search = compare
    try:
        result = play._play(dict(seed=args.seed, future_seed=None, arm="fast_search", args=game_args))
        print("no divergence", result, flush=True)
    except Found:
        pass
    finally:
        play._close_worker_env()


if __name__ == "__main__":
    main()
