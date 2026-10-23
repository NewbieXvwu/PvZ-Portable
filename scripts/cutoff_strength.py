"""task40 strength: does confident cutoff change the teacher's win rate?

beta = 1: the teacher (search) executes its own decisions, the student only supplies the
fast action that gates the search exactly as in production ``search_play`` runs (search
when there are legal plants and the student wants to plant or ``min_gap`` ticks passed).

Each seed is played twice -- once with the production search (``exact``), once with the
same search under the confident-cutoff rollout/scoring swap (``cutoff``, step 3's Python
slow version, theta 0.98, checkpoints 500/1000/1500/2000).  The two games diverge as soon
as their decisions differ, so they cannot share a trajectory; pairing is by game seed.

Gate: the paired win-rate difference (cutoff - exact) over whole seeds, bootstrap 95%
lower bound >= -margin, with every seed of every family played.  Per-family rows are
reported, not gated (same convention as the step 3 quality gate).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

import cutoff_agreement as ca

ARMS = ("exact", "cutoff")


def _play_once(job: dict) -> dict:
    import torch
    import search_play
    family, seed, arm = job["family"], job["seed"], job["arm"]
    args = ca._W["args"]
    torch.manual_seed(seed)
    obs = ca._reset(family, seed)
    env = ca._W["env"]
    fast_state: dict = {}
    search_play._W["student_history"] = None
    config = argparse.Namespace(samples=args.samples, horizon=args.horizon, margin=args.margin,
                                min_gap=args.min_gap, per_card=args.per_card, halving=True,
                                student_candidates=args.student_candidates,
                                verify_samples=args.verify_samples)
    stats = {"searched": 0, "overrides": 0, "override_kinds": Counter()}
    decisions, search_seconds = 0, 0.0
    started = time.perf_counter()
    try:
        while not obs["terminal"] and decisions < args.max_actions:
            action = search_play._fast_action(obs, fast_state)
            search_play._W["student_history"] = fast_state.get("history")
            ca._install(exact=(arm == "exact"))
            before = time.perf_counter()
            try:
                action = search_play._search(env, obs, action, config,
                                             seed * 10000 + decisions, stats)
            finally:
                ca._install(exact=True)
                search_seconds += time.perf_counter() - before
            fast_state["action"] = action
            obs, _r, done, _t, info = env.step(action)
            if not info.get("ok"):
                obs, _r, done, _t, info = env.step({"type": "wait", "ticks": 60})
            decisions += 1
            if done:
                break
    finally:
        ca._install(exact=True)
    return {"family": family, "seed": seed, "arm": arm,
            "won": obs.get("result") == 1, "final_wave": obs["wave"],
            "decisions": decisions, "searches": stats["searched"],
            "overrides": stats["overrides"],
            "seconds_total": round(time.perf_counter() - started, 2),
            "seconds_search": round(search_seconds, 2)}


def _part_path(args, family: str, seed: int, arm: str) -> Path:
    return args.parts / f"{family}_{seed}_{arm}.json"


def _needs_run(args, job: dict) -> bool:
    if args.rerun:
        return True
    path = _part_path(args, job["family"], job["seed"], job["arm"])
    if not path.is_file():
        return True
    try:
        return bool(json.loads(path.read_text()).get("error"))
    except ValueError:
        return True


def _write_part(args, result: dict) -> None:
    path = _part_path(args, result["family"], result["seed"], result["arm"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result) + "\n")
    os.replace(temporary, path)


def _play(job: dict) -> dict:
    from pvz_env import SimulatorExited
    args = ca._W["args"]
    attempts = 1 + max(0, args.retries)
    result = None
    for attempt in range(1, attempts + 1):
        try:
            result = _play_once(job)
            break
        except (SimulatorExited, BrokenPipeError, ConnectionError, EOFError, TimeoutError) as exc:
            print(f"{time.strftime('%H:%M:%S')} engine died {attempt}/{attempts} "
                  f"{job['family']} seed {job['seed']} {job['arm']}: "
                  f"{type(exc).__name__}: {exc}", flush=True)
            if attempt == attempts:
                result = {"family": job["family"], "seed": job["seed"], "arm": job["arm"],
                          "won": None, "final_wave": None, "decisions": 0, "searches": 0,
                          "overrides": 0, "seconds_total": 0.0, "seconds_search": 0.0,
                          "error": f"{type(exc).__name__}: {exc}"}
                break
            ca._recover()
    _write_part(args, result)
    return result


def _bootstrap_pairs(pairs: list[tuple[bool, bool]], *, draws: int, seed: int) -> dict:
    import numpy as np
    exact = sum(1 for exact_won, _ in pairs if exact_won)
    cutoff = sum(1 for _, cutoff_won in pairs if cutoff_won)
    differences = np.array([int(b) - int(a) for a, b in pairs], dtype=np.int8)
    rng = np.random.default_rng(seed)
    picked = rng.integers(len(differences), size=(draws, len(differences)))
    sampled = differences[picked].mean(axis=1)
    interval = np.quantile(sampled, [.025, .975]).tolist()
    return {"games": len(pairs),
            "exact_wins": exact, "cutoff_wins": cutoff,
            "exact_rate": exact / len(pairs), "cutoff_rate": cutoff / len(pairs),
            "difference": (cutoff - exact) / len(pairs),
            "difference_cluster_ci95": interval,
            "bootstrap_draws": draws, "bootstrap_seed": seed,
            "bootstrap_unit": "whole seed, paired win difference"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", default="/home/newbiexvwu/PvZ-search/artifacts/student/dw4i_s1/model.pt")
    p.add_argument("--value", default="/home/newbiexvwu/PvZ-search/artifacts/value_net/v3/model.pt")
    p.add_argument("--static-value", default="/home/newbiexvwu/PvZ-search/artifacts/hard/value_branch1/model.pt")
    p.add_argument("--executable", default=str(ROOT / "artifacts" / "task40" / "engine" / "pvz-portable"))
    p.add_argument("--resource-dir", default="/home/newbiexvwu/.cache/pvz-research-resources")
    p.add_argument("--seeds", type=int, default=64)
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument("--families", default="day_l7,night_l14,pool_24,original_1_10")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--samples", type=int, default=3)
    p.add_argument("--per-card", type=int, default=1)
    p.add_argument("--student-candidates", type=int, default=2)
    p.add_argument("--verify-samples", type=int, default=9)
    p.add_argument("--horizon", type=int, default=2500)
    p.add_argument("--margin", type=float, default=0.02)
    p.add_argument("--min-gap", type=int, default=300)
    p.add_argument("--rollout-sim-skip", type=int, default=61)
    p.add_argument("--max-actions", type=int, default=3000)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--draws", type=int, default=20000)
    p.add_argument("--bootstrap-seed", type=int, default=20261019)
    p.add_argument("--parts", type=Path, default=ROOT / "artifacts" / "task40" / "strength_parts")
    p.add_argument("--rerun", action="store_true")
    p.add_argument("--out", type=Path, default=ROOT / "artifacts" / "task40" / "strength.json")
    args = p.parse_args()

    from render_episode import load_training_task
    jobs = []
    for family in args.families.split(","):
        manifest, task_id = ca.FAMILIES[family]
        seeds = load_training_task(Path(manifest), task_id)["seeds"][args.seed_offset:args.seed_offset + args.seeds]
        if len(seeds) != args.seeds:
            raise SystemExit(f"{family}: task has {len(seeds)} seeds at offset {args.seed_offset}, need {args.seeds}")
        for seed in seeds:
            for arm in ARMS:
                jobs.append({"family": family, "seed": int(seed), "arm": arm})
    started = time.time()
    pending = [job for job in jobs if _needs_run(args, job)]
    if len(pending) < len(jobs):
        print(f"resume: {len(jobs) - len(pending)}/{len(jobs)} games already on disk", flush=True)
    if pending:
        start_method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
        with mp.get_context(start_method).Pool(args.workers, initializer=ca._init, initargs=(args,)) as pool:
            for index, result in enumerate(pool.imap_unordered(_play, pending), 1):
                print(f"{time.strftime('%H:%M:%S')} {index}/{len(pending)} {result['family']} "
                      f"seed {result['seed']} {result['arm']} won={result['won']} "
                      f"searches={result['searches']} seconds={result['seconds_total']}", flush=True)
    results = [json.loads(_part_path(args, job["family"], job["seed"], job["arm"]).read_text())
               for job in jobs]
    errors = [{key: result.get(key) for key in ("family", "seed", "arm", "error")}
              for result in results if result.get("error")]
    by_seed: dict[tuple[str, int], dict[str, dict]] = {}
    for result in results:
        by_seed.setdefault((result["family"], result["seed"]), {})[result["arm"]] = result
    pairs = {key: (value["exact"]["won"], value["cutoff"]["won"])
             for key, value in by_seed.items()
             if all(arm in value and value[arm].get("won") is not None for arm in ARMS)}
    summary = {"config": {key: str(value) for key, value in vars(args).items()},
               "theta": ca.CUTOFF_THETA, "checkpoints": list(ca.CUTOFF_AT),
               "arms": list(ARMS), "seeds": args.seeds,
               "games": len(results), "pairs": len(pairs), "errors": errors,
               "seconds": round(time.time() - started, 1),
               "per_family": {}}
    if pairs:
        summary["summary"] = _bootstrap_pairs(list(pairs.values()),
                                              draws=args.draws, seed=args.bootstrap_seed)
        summary["summary"]["margin"] = 0.03
        summary["summary"]["minimum_pairs"] = len(jobs) // len(ARMS)
        summary["summary"]["gate_passed"] = (len(pairs) == len(jobs) // len(ARMS)
                                             and summary["summary"]["difference_cluster_ci95"][0] >= -0.03)
        for family in sorted({key[0] for key in pairs}):
            family_pairs = [value for key, value in pairs.items() if key[0] == family]
            summary["per_family"][family] = _bootstrap_pairs(family_pairs,
                                                             draws=args.draws, seed=args.bootstrap_seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "games": results}, indent=1) + "\n")
    print(json.dumps({key: summary[key] for key in ("games", "pairs", "errors", "seconds")}, indent=1))
    print(json.dumps(summary.get("summary", {}), indent=1))
    print(json.dumps(summary["per_family"], indent=1))


if __name__ == "__main__":
    main()
