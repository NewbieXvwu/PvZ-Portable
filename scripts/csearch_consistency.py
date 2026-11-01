"""Consistency and speed smoke for the native search (task217 phase 1).

Plays a game with the student, and at the first ``--decisions`` decisions that have a legal plant
it runs the *same* decision twice from the *same* root and the same decision seed: once through the
Python teacher (``search_play._search``) and once through the engine's ``SEARCH`` command
(``search_play._cpp_search``).  Nothing is executed between the two, and every rollout reseeds from
the decision seed, so the two must reach the same action; only float noise can differ.

This is the one-off migration comparison of AGENTS.md §6.6, not a resident test.

Example (Mac, one game, 20 decisions):
  python scripts/csearch_consistency.py --level 28 --deck 1,16,35,10,42,39,44,30,21,17 \
      --seeds 3174639343 --student /tmp/task214-models/student/model.pt \
      --value /tmp/task214-models/v5/model.pt --static-value /tmp/task214-models/static/model.pt \
      --executable ~/pvz-csearch/build/pvz-portable --out /tmp/csearch-smoke.json
"""
from __future__ import annotations

import argparse
import json
import copy
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "python"))


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--level", type=int, required=True)
    p.add_argument("--deck", required=True, help="comma-separated seed types")
    p.add_argument("--seeds", required=True, help="comma-separated game seeds")
    p.add_argument("--student", type=Path, required=True)
    p.add_argument("--value", type=Path, required=True)
    p.add_argument("--static-value", type=Path, required=True)
    p.add_argument("--executable", type=Path, default=Path.home() / "pvz-csearch/build/pvz-portable")
    p.add_argument("--resource-dir", type=Path, default=Path.home() / ".cache/pvz-research-resources")
    p.add_argument("--decisions", type=int, default=20, help="compared decisions per game")
    p.add_argument("--max-steps", type=int, default=4000, help="safety cap on student steps per game")
    p.add_argument("--student-candidates", type=int, default=None, help="override the budget's top-k")
    p.add_argument("--budget", choices=("prod", "deep"), default="prod",
                   help="prod = the production teacher budget; deep = the kind of larger budget the "
                        "native search exists to afford (samples 6, verify 15, top-4, horizon 5000)")
    p.add_argument("--combo-candidates", type=int, default=0, help="enable --combo-candidates k")
    p.add_argument("--save-candidates", type=int, default=0, help="enable --save-candidates k")
    p.add_argument("--out", type=Path, help="write the per-decision rows here")
    return p


def main() -> None:
    args = parser().parse_args()
    import torch
    torch.set_num_threads(1)
    import search_play
    from pvz_event_env import EventWaitEnv
    from pvz_env import TaskSpec
    from pvz_tasks import profile_for_deck
    from render_episode import training_task_kwargs

    deck = [int(part) for part in args.deck.split(",") if part.strip()]
    seeds = [int(part) for part in args.seeds.split(",") if part.strip()]
    value_keep, value_files = search_play._export_values(str(args.value), str(args.static_value))
    search_play._load_nets(value_files[0], str(args.student), value_files[1])
    search_play._W.update(lean_rollouts=True, rollout_sim_skip=61, batch_waits=True)
    # The production teacher budget (task214): top-2 student candidates and nine verify futures.
    deep = args.budget == "deep"
    teacher = argparse.Namespace(
        per_card=1, student_candidates=args.student_candidates if args.student_candidates is not None
        else (4 if deep else 2), save_candidates=args.save_candidates, save_share=0.5, save_max_wait=1800,
        save_chunk=300, save_interval=600, samples=6 if deep else 3, halving=True, halving_first_samples=1,
        combo_candidates=args.combo_candidates, combo_max_wait=5000, margin_relative=0.0, horizon=2500,
        verify_samples=9, min_gap=300, rollout_sim_skip=61, leaf_probe=None, leaf_probe_limit=0,
        task_id=f"csearch_L{args.level}")
    rows, compared, agreed = [], 0, 0
    python_seconds, cpp_seconds = [], []
    env = EventWaitEnv(str(args.resource_dir), executable=str(args.executable), headless=True)
    try:
        env.load_rollout_net(search_play._W["rollout_file"])
        env.load_value_net(value_files[0])
        env.load_value_net(value_files[1], static=True)
        if not env._search_version:
            raise SystemExit("the binary has no SEARCH command; rebuild this worktree")
        for seed in seeds:
            task_definition = {"task_id": f"csearch_L{args.level}", "level": args.level, "deck": deck,
                               "seeds": [seed], "playthrough": 2, "zombie_count_multiplier": 1.0,
                               "wave_cap": None, "preplanted": [], "first_aid_owned": False}
            kw = training_task_kwargs(task_definition, seed)
            torch.manual_seed(seed)
            task = TaskSpec(level=kw["level"], seed=kw["seed"], playthrough=2,
                            profile=profile_for_deck(tuple(kw["deck"]), kw.get("first_aid_owned", False)),
                            **{k: v for k, v in kw["task_extra"].items() if v is not None})
            obs, _ = env.reset(deck=tuple(kw["deck"]), task=task)
            search_play._W["context"] = __import__("pvz_value_net").observation_context(obs)
            search_play._W["demo_actions"] = []
            stats = {"searched": 0, "overrides": 0, "shovels": 0, "override_kinds": Counter(),
                     "student_candidate_added": 0, "student_candidate_best": 0,
                     "student_candidate_override": 0, "combo_candidate_added": 0,
                     "combo_upgrades": 0, "combo_held": 0, "save_probes": 0, "save_overrides": 0,
                     "save_wait_ticks": 0, "pending_upgrade": None, "sun_reserve": 0}
            fast_state, steps = {}, 0
            while not obs["terminal"] and steps < args.max_steps and compared < args.decisions * len(seeds):
                action = search_play._fast_action(obs, fast_state, env)
                search_play._W["student_history"] = fast_state.get("history")
                legal = (obs.get("legal_actions") or {}).get("plants") or []
                decision_seed = seed * 10000 + steps
                searched_before = stats.get("searched", 0)
                if legal and compared < args.decisions * len(seeds):
                    # _play clears these before every search; a stale value would look like a diff.
                    stats["last_targets"] = None
                    stats.pop("last_verification", None)
                    # Both sides start from the identical decision state and stats, so the
                    # decision seed, the min-gap clock and the reserve are the same inputs.
                    python_stats, cpp_stats = copy.deepcopy(stats), copy.deepcopy(stats)
                    # Run both sides in the order that alternates per decision: whoever runs second
                    # finds the root restored, the board and the weights warm, so a fixed order would
                    # flatter it (measured: ~5% before alternating).
                    root = env.snapshot_fast()

                    def timed(run):
                        env.restore_fast(root)
                        started = time.perf_counter()
                        result = run()
                        return result, time.perf_counter() - started

                    def run_python():
                        return timed(lambda: search_play._search(
                            env, obs, dict(action), teacher, decision_seed, python_stats))

                    def run_cpp():
                        return timed(lambda: search_play._cpp_search(
                            env, obs, dict(action), teacher, decision_seed, cpp_stats))

                    python_run, cpp_run = ((run_cpp(), run_python()) if compared % 2
                                           else (run_python(), run_cpp()))
                    env.release_snapshot(root)
                    python_action, python_elapsed = python_run
                    cpp_action, cpp_elapsed = cpp_run
                    python_targets = python_stats.get("last_targets")
                    python_verification = python_stats.get("last_verification")
                    cpp_targets = cpp_stats.get("last_targets")
                    cpp_verification = cpp_stats.get("last_verification")
                    stats = cpp_stats
                    same = python_action == cpp_action
                    def keyed(targets):
                        return None if targets is None else [[c, None if abs(v) == float("inf") else v] for c, v in targets]

                    def close(totals_a, totals_b):
                        # Same candidates in the same order, values within float noise (1e-6 relative).
                        if totals_a is None or totals_b is None:
                            return totals_a is None and totals_b is None
                        if len(totals_a) != len(totals_b):
                            return False
                        for (cand_a, value_a), (cand_b, value_b) in zip(totals_a, totals_b):
                            if cand_a != cand_b:
                                return False
                            if abs(value_a) == float("inf") or abs(value_b) == float("inf"):
                                if value_a != value_b:
                                    return False
                            elif abs(value_a - value_b) > 1e-9 + 1e-6 * abs(value_a):
                                return False
                        return True
                    searched = cpp_stats.get("searched", 0) > searched_before
                    # Structural counters must match exactly: they show the halving rounds and the
                    # candidate set ran identically, which the final action alone would hide.
                    counters = ("halving_candidates", "halving_rollouts", "halving_selected_lanes",
                                "overrides", "student_candidate_added", "combo_candidate_added",
                                "combo_upgrades", "combo_held", "save_probes", "save_overrides",
                                "save_wait_ticks")
                    counters_same = all(python_stats.get(name, 0) == cpp_stats.get(name, 0) for name in counters)
                    if not counters_same:
                        print(f"{seed} step {steps} COUNTERS differ " + ", ".join(
                            f"{name} {python_stats.get(name, 0)}/{cpp_stats.get(name, 0)}" for name in counters),
                            flush=True)
                    rows.append(dict(seed=seed, step=steps, tick=obs["tick"], same=same, counters_same=counters_same,
                                     python=python_action, cpp=cpp_action, legal=len(legal),
                                     python_targets=keyed(python_targets), cpp_targets=keyed(cpp_targets),
                                     python_verification=python_verification, cpp_verification=cpp_verification,
                                     searched=searched, targets_same=close(keyed(python_targets), keyed(cpp_targets)),
                                     verification_same=(None if python_verification is None else
                                                        {k: v for k, v in python_verification.items() if k != "seconds"}) ==
                                                       (None if cpp_verification is None else
                                                        {k: v for k, v in cpp_verification.items() if k != "seconds"})))
                    # A min-gap skip runs no rollout; count it only when both sides agree it is one.
                    if python_stats.get("searched", 0) != cpp_stats.get("searched", 0):
                        print(f"{seed} step {steps} SEARCH-FLAG differs", flush=True)
                    if searched:
                        agreed += same and counters_same
                        python_seconds.append(python_elapsed)
                        cpp_seconds.append(cpp_elapsed)
                        compared += 1
                        print(f"{seed} step {steps} tick {obs['tick']}: {'same' if same else 'DIFF'} "
                              f"{python_action} / {cpp_action} "
                              f"({python_elapsed * 1000:.0f} ms python, {cpp_elapsed * 1000:.0f} ms cpp)",
                              flush=True)
                    else:
                        print(f"{seed} step {steps} tick {obs['tick']}: skipped (min gap)", flush=True)
                # _play stores the executed action for the next decision's context; the reference
                # side is the Python search's pick so both sides see the same previous action.
                fast_state["action"] = cpp_action
                obs, _r, done, _t, _info = env.step(action)
                steps += 1
                if done:
                    break
    finally:
        env.close()
        value_keep.cleanup()
    summary = dict(compared=compared, agreed=agreed,
                   targets_agreed=sum(1 for row in rows if row["targets_same"]),
                   verification_compared=sum(1 for row in rows if row["python_verification"] is not None or
                                             row["cpp_verification"] is not None),
                   verification_agreed=sum(1 for row in rows if row["verification_same"] and
                                           (row["python_verification"] is not None or
                                            row["cpp_verification"] is not None)),
                   counters_agreed=sum(1 for row in rows if row["counters_same"] and row["searched"]),
                   searched_decisions=sum(1 for row in rows if row["searched"]),
                   agreement=None if not compared else round(agreed / compared, 4),
                   python_ms_median=round(statistics.median(python_seconds) * 1000, 1) if python_seconds else None,
                   cpp_ms_median=round(statistics.median(cpp_seconds) * 1000, 1) if cpp_seconds else None,
                   speedup=None if not cpp_seconds else round(
                       statistics.median(python_seconds) / statistics.median(cpp_seconds), 2))
    print(json.dumps(summary, indent=1))
    if args.out:
        args.out.write_text(json.dumps(dict(summary=summary, rows=rows), indent=1))


if __name__ == "__main__":
    main()
