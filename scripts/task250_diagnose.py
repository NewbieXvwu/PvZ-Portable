"""task250: first-divergence diagnosis on shared production prefixes.

Replays one panel seed with the production teacher, compares the production and candidate
SEARCH on the same root at every decision, and stops at the first differing action.  At that
root every compared candidate action is rolled to a common leaf, scored by both leaf nets,
and (optionally) continued to the real terminal with the production teacher so the action
can be judged by outcomes instead of by either net.

All actions, features, rollouts, search and continuations use the existing C++ interfaces.
This is offline analysis; it never changes a production entry point.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "python")]
import search_play as play


class Found(Exception):
    pass


def action_key(action):
    if action.get("type") != "plant":
        return ("wait",)
    return ("plant", action.get("packet"), action.get("row"), action.get("col"))


def pick_candidates(responses, candidate, limit):
    """Union of the actions that mattered in either search, best-ranked first."""
    picked = []

    def add(action):
        if action is None:
            return
        key = action_key(action)
        if key not in {action_key(a) for a in picked}:
            picked.append(action)

    # The actual chosen actions are the counterfactual we care about.
    add(responses["production"]["action"])
    add(responses[candidate]["action"])
    # Then the top-ranked leaf of each search, so an override can be compared with its challenger.
    for name in ("production", candidate):
        targets = responses[name].get("targets") or []
        if targets:
            add(targets[0][0])
    # Waiting is the incumbent action the override rule protects; keep it when it was scored.
    for name in ("production", candidate):
        for action, _value in responses[name].get("targets") or []:
            if action.get("type") != "plant":
                add(action)
                break
    return picked[:limit]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--candidate", default="finetune-t3")
    parser.add_argument("--futures", type=int, default=4)
    parser.add_argument("--max-candidates", type=int, default=5)
    parser.add_argument("--no-terminal", action="store_true",
                        help="only score the common leaf; skip the expensive real continuation")
    parser.add_argument("--data", type=Path, default=ROOT / "artifacts/task250")
    parser.add_argument("--executable", type=Path,
                        default=Path.home() / "pvz-mac-collect/build-vnbranch/pvz-portable")
    args = parser.parse_args()
    base = args.data.resolve()
    (base / "work").mkdir(parents=True, exist_ok=True)
    models = base / "models"
    os.environ.update(PVZ_NO_POPUP="1", TMPDIR=str(base / "work"),
                      OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    import tempfile
    tempfile.tempdir = str(base / "work")
    from export_value_net import export_value_net
    from pvz_value_net import load_value_net
    from value_teacher_feasibility import continuation_state, finish_teacher
    paths = {}
    for name in ("production", "static", args.candidate):
        paths[name] = str(base / "work" / f"{args.seed}-{args.candidate}-{name}.bin")
        export_value_net(load_value_net(models / f"{name}.pt"), paths[name])
    panel = json.loads((base / "panel-original.json").read_text())
    task = next(t for t in panel["tasks"] if t["level"] == args.level)
    game_args = play.parser().parse_args([
        "--rollout-policy", str(models / "student.pt"), "--value", str(models / "production.pt"),
        "--manifest", str(base / "panel-original.json"), "--task-id", task["task_id"],
        "--out", str(base / "work"), "--verify-samples", "9", "--student-candidates", "2",
        "--native-games", "--executable", str(args.executable)])
    play._load_nets(paths["production"], str(models / "student.pt"), paths["static"])
    play._init_worker(paths["production"], str(models / "student.pt"), 61, paths["static"])
    original = play._search
    prefix = []

    def compare(env, obs, policy, config, decision_seed, stats):
        responses = {}
        # _search restores the common root.  Combo/save/shovel are disabled, so no pending
        # search state has to be transferred between the two diagnostic calls.
        for name in (args.candidate, "production"):
            env.load_value_net(paths[name])
            local = dict(stats)
            local["override_kinds"] = Counter(stats.get("override_kinds") or {})
            before = {key: local.get(key, 0) for key in ("searched", "overrides")}
            action = original(env, obs, policy, config, decision_seed, local)
            responses[name] = dict(
                action=action, policy=policy, targets=local.get("last_targets"),
                verification=local.get("last_verification"),
                searched=local.get("searched", 0) - before["searched"],
                overrides=local.get("overrides", 0) - before["overrides"],
                override_kinds=dict(local.get("override_kinds") or {}))
            if name == "production":
                stats.update(local)
        prefix.append(dict(decision=decision_seed % 10000, tick=obs["tick"],
                           production=responses["production"]["action"],
                           candidate=responses[args.candidate]["action"]))
        if action_key(responses[args.candidate]["action"]) == action_key(responses["production"]["action"]):
            return responses["production"]["action"]

        root = env.snapshot_fast()
        baseline = dict(play._W)
        candidates = pick_candidates(responses, args.candidate, args.max_candidates)
        rows = []
        try:
            for action in candidates:
                futures = []
                for k in range(args.futures):
                    future = (decision_seed * 1000 + k) % 2**32
                    play._W.clear()
                    play._W.update(baseline)
                    env.restore_fast(root)
                    leaf = env.rollout(action, obs["tick"] + config.horizon,
                        snapshot_id=root, seed=future, sim_skip=config.rollout_sim_skip,
                        student_history=baseline.get("student_history"))
                    terminal = bool(leaf.get("terminal") or leaf.get("rollout_result", -1) >= 0)
                    if not terminal:
                        history = leaf["rollout_history"]
                        leaf = env.observe()
                        leaf["rollout_history"] = history
                    values = {}
                    for name in ("production", args.candidate):
                        env.load_value_net(paths[name])
                        values[name] = (float(leaf.get("rollout_result", int(leaf.get("result") == 1)))
                                        if terminal else float(env.leaf_values(
                                            income=leaf.get("sun_income_rate") or 0)[0]))
                    won = bool(leaf.get("rollout_result", leaf.get("result")) == 1) if terminal else None
                    outcome = None
                    if not terminal and not args.no_terminal:
                        env.load_value_net(paths["production"])
                        start = __import__("time").monotonic()
                        # The continuation must use the real teacher search, not this comparator;
                        # otherwise it would recurse into this diagnostic at its first decision.
                        play._search = original
                        try:
                            outcome = finish_teacher(env, leaf, continuation_state(leaf), config, future)
                        finally:
                            play._search = compare
                        won = outcome["won"]
                        outcome = dict(won=won, terminal=outcome["terminal"],
                                       seconds=__import__("time").monotonic() - start)
                    futures.append(dict(future=k, future_seed=future, tick=leaf["tick"],
                        terminal=terminal, values=values, won=won, continuation=outcome,
                        plants=leaf.get("plants"), sun=leaf.get("sun")))
                rows.append(dict(action=action, futures=futures))
        finally:
            play._W.clear()
            play._W.update(baseline)
            env.restore_fast(root)
            env.release_snapshot(root)
        record = dict(level=args.level, seed=args.seed, candidate=args.candidate,
                      decision=decision_seed % 10000, tick=obs["tick"], policy=policy,
                      root=obs, searches=responses, candidates=rows, prefix=prefix,
                      protocol=dict(futures=args.futures, max_candidates=args.max_candidates,
                          no_terminal=args.no_terminal, horizon=config.horizon,
                          sim_skip=config.rollout_sim_skip,
                          shared_future_formula="(decision_seed*1000+k) mod 2**32",
                          judge="production teacher continuation (finish_teacher)"))
        folder = "diag" if args.no_terminal else "diagT"
        out = base / f"{folder}/l{args.level}-{args.seed}-{args.candidate}.json"
        out.parent.mkdir(exist_ok=True)
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
