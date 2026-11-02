"""Where a native search decision's time goes (task217b part 1).

Plays a game with the student and, at the first ``--decisions`` decisions with a legal plant, runs
one production-budget SEARCH (C++ implementation) and records the engine's own per-section profile:
static scoring, root restores, simulated ticks, student forwards, leaf forwards, verification,
everything else.  Nothing is compared here -- ``csearch_consistency.py`` does that.

Example (Mac):
  PVZ_SEARCH_PROFILE=1 python scripts/csearch_profile.py --level 28 ...
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "python"))

SECTIONS = ("total_seconds", "static_seconds", "restore_seconds", "legal_seconds", "plant_seconds", "leaf_seconds", "terminal_seconds",
            "simulation_seconds", "student_seconds", "other_seconds")
COUNTS = ("ticks_simulated", "student_forwards", "leaf_forwards", "rollouts", "verify_rollouts",
          "static_evaluations", "restores", "legal_calls")


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--level", type=int, required=True)
    p.add_argument("--deck", required=True)
    p.add_argument("--seeds", required=True)
    p.add_argument("--student", type=Path, required=True)
    p.add_argument("--value", type=Path, required=True)
    p.add_argument("--static-value", type=Path, required=True)
    p.add_argument("--executable", type=Path, default=Path.home() / "pvz-csearch/build/pvz-portable")
    p.add_argument("--resource-dir", type=Path, default=Path.home() / ".cache/pvz-research-resources")
    p.add_argument("--decisions", type=int, default=6, help="profiled decisions per seed")
    p.add_argument("--max-steps", type=int, default=4000)
    p.add_argument("--sim-skip", type=int, default=61, help="rollout sim-skip mask (add 256 for sparse tracks)")
    p.add_argument("--out", type=Path)
    return p


def main() -> None:
    args = parser().parse_args()
    import torch
    torch.set_num_threads(1)
    import argparse as ap
    import search_play
    from pvz_event_env import EventWaitEnv
    from pvz_env import TaskSpec
    from pvz_tasks import profile_for_deck
    from render_episode import training_task_kwargs

    deck = [int(part) for part in args.deck.split(",") if part.strip()]
    seeds = [int(part) for part in args.seeds.split(",") if part.strip()]
    value_keep, value_files = search_play._export_values(str(args.value), str(args.static_value))
    search_play._load_nets(value_files[0], str(args.student), value_files[1])
    search_play._W.update(lean_rollouts=True, rollout_sim_skip=args.sim_skip, batch_waits=True)
    teacher = ap.Namespace(
        per_card=1, student_candidates=2, save_candidates=2, save_share=0.5, save_max_wait=1800,
        save_chunk=300, save_interval=600, samples=3, halving=True, halving_first_samples=1,
        combo_candidates=2, combo_max_wait=5000, combo_buffer=25, combo_interval=600, margin_relative=0.0, horizon=2500, verify_samples=9,
        min_gap=300, rollout_sim_skip=args.sim_skip, leaf_probe=None, leaf_probe_limit=0,
        task_id=f"csearch_profile_L{args.level}")
    rows = []
    env = EventWaitEnv(str(args.resource_dir), executable=str(args.executable), headless=True)
    try:
        env.load_rollout_net(search_play._W["rollout_file"])
        env.load_value_net(value_files[0])
        env.load_value_net(value_files[1], static=True)
        for seed in seeds:
            task_definition = {"task_id": f"csearch_profile_L{args.level}", "level": args.level,
                               "deck": deck, "seeds": [seed], "playthrough": 2,
                               "zombie_count_multiplier": 1.0, "wave_cap": None, "preplanted": [],
                               "first_aid_owned": False}
            kw = training_task_kwargs(task_definition, seed)
            torch.manual_seed(seed)
            task = TaskSpec(level=kw["level"], seed=kw["seed"], playthrough=2,
                            profile=profile_for_deck(tuple(kw["deck"]), kw.get("first_aid_owned", False)),
                            **{k: v for k, v in kw["task_extra"].items() if v is not None})
            obs, _ = env.reset(deck=tuple(kw["deck"]), task=task)
            search_play._W["context"] = __import__("pvz_value_net").observation_context(obs)
            search_play._W["demo_actions"] = []
            from collections import Counter
            stats = {"searched": 0, "overrides": 0, "override_kinds": Counter(), "combo_candidate_added": 0,
                     "combo_upgrades": 0, "pending_upgrade": None, "sun_reserve": 0}
            fast_state, steps, profiled = {}, 0, 0
            while not obs["terminal"] and steps < args.max_steps and profiled < args.decisions:
                action = search_play._fast_action(obs, fast_state, env)
                search_play._W["student_history"] = fast_state.get("history")
                legal = (obs.get("legal_actions") or {}).get("plants") or []
                if legal and profiled < args.decisions:
                    stats["last_profile"] = None
                    action = search_play._cpp_search(env, obs, action, teacher, seed * 10000 + steps, stats)
                    profile = stats.get("last_profile")
                    if profile and profile["rollouts"]:
                        rows.append(dict(seed=seed, step=steps, tick=obs["tick"], legal=len(legal), **profile))
                        profiled += 1
                fast_state["action"] = action
                obs, _r, done, _t, info = env.step(action)
                if not info.get("ok"):
                    obs, _r, done, _t, info = env.step({"type": "wait", "ticks": 60})
                steps += 1
    finally:
        env.close()
    total = statistics.median([row["total_seconds"] for row in rows]) if rows else 0.0
    summary = {"share_" + name: (statistics.median([row.get(name, 0) for row in rows]) / total if total else 0.0)
               for name in SECTIONS} if rows else {}
    summary.update({name: statistics.median([row.get(name, 0) for row in rows]) for name in SECTIONS} if rows else {})
    summary.update({name: statistics.median([row.get(name, 0) for row in rows]) for name in COUNTS} if rows else {})
    summary["decisions"] = len(rows)
    summary["level"] = args.level
    print(json.dumps(summary, indent=1))
    if args.out:
        args.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
