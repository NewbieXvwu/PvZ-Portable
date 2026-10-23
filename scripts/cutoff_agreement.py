"""task40 step 3 quality: does confident cutoff change the teacher's decisions?

The student plays each game (dagger_beta = 0, the task8 gate1 convention).  At every
decision where the teacher searches, the same root and the same resampled futures are
handed to three arms:

``exact``   the production search, horizon 2500
``cutoff``  the same search, but each rollout is cut at the first checkpoint where the
            value net is certain (500/1000/1500/2000 ticks, theta 0.98); a rollout that
            is never certain runs to 2500.  One engine round trip per checkpoint, in
            order, stopping at the first certain state -- the t-tick rollout is a prefix
            of the 2500-tick one (task40 step 1, 32/32), so the states are identical.
``noise``   the production search with a different set of futures (the noise floor)

This is the *Python slow version* from the task book: each checkpoint costs a separate
C++ round trip, so it is slower than a true in-engine check and measures quality only,
not speed.  The engine implementation is a separate step.

Nothing in scripts/search_play.py is modified: the cutoff arm swaps its module-level
``_rollout_policy`` and ``_score_terminal_or_value`` for the duration of that one search.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

CUTOFF_AT = (500, 1000, 1500, 2000)
CUTOFF_THETA = 0.98
NOISE_OFFSET = 7_777_777
DEFAULT_MANIFEST = "/home/newbiexvwu/PvZ-search/artifacts/stage4/day_l7_hard.json"
DEFAULT_TASK = "day_l7_z20_eval"
DEFAULT_CONVEYOR_MANIFEST = "/home/newbiexvwu/PvZ-search/.claude/worktrees/conveyor/artifacts/stage4/day_panel.json"

FAMILIES = {
    "day_l7": ("/home/newbiexvwu/PvZ-search/artifacts/stage4/day_l7_hard.json", "day_l7_z20_eval"),
    "night_l14": ("/home/newbiexvwu/PvZ-search/artifacts/stage4/night_panel_n10.json", "night_l14_eval_n10"),
    "pool_24": ("/home/newbiexvwu/PvZ-search/artifacts/stage4/pool_panel.json", "pool_l24_eval_p9k"),
    "original_1_10": (DEFAULT_CONVEYOR_MANIFEST, "day_l10_orig_eval"),
}

_W: dict = {}


def _open_env(args, binary):
    from pvz_event_env import EventWaitEnv
    env = EventWaitEnv(args.resource_dir, executable=args.executable, headless=True)
    env.load_rollout_net(binary)
    return env


def _init(args) -> None:
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    import torch
    torch.set_num_threads(1)
    from pvz_rollout_policy import load_student, export_student
    from pvz_value_net import load_value_net
    import search_play
    net = load_student(args.policy)
    temporary = tempfile.TemporaryDirectory(prefix="pvz-cutoff-agreement-")
    binary = Path(temporary.name) / "student.bin"
    export_student(net, binary)
    env = _open_env(args, binary)
    search_play._W.update(fast=net, vnet=load_value_net(args.value),
                          static_vnet=load_value_net(args.static_value),
                          lean_rollouts=True, batch_waits=True,
                          rollout_sim_skip=args.rollout_sim_skip)
    _W.update(env=env, net=net, args=args, temporary=temporary, binary=binary,
              search_play=search_play,
              original_rollout=search_play._rollout_policy,
              original_score=search_play._score_terminal_or_value)


def _reset(family: str, seed: int):
    from pvz_env import TaskSpec
    from render_episode import load_training_task, training_task_kwargs
    from pvz_tasks import profile_for_deck
    import search_play
    from pvz_value_net import observation_context
    manifest, task_id = FAMILIES[family]
    kw = training_task_kwargs(load_training_task(Path(manifest), task_id), seed)
    task = TaskSpec(level=kw["level"], seed=kw["seed"], playthrough=2,
                    profile=profile_for_deck(tuple(kw["deck"]), kw.get("first_aid_owned", False)),
                    **{k: v for k, v in kw["task_extra"].items() if v is not None})
    obs, _ = _W["env"].reset(deck=tuple(kw["deck"]), task=task)
    search_play._W["context"] = observation_context(obs)
    return obs


def _cutoff_rollout(env, obs, first, until_tick, root=None, seed=None):
    import search_play
    from render_episode import _frame
    root_tick = int(obs["tick"])
    for checkpoint in CUTOFF_AT:
        if root_tick + checkpoint >= until_tick:
            break
        leaf = _W["original_rollout"](env, obs, first, root_tick + checkpoint, root, seed)
        terminal = _W["original_score"](leaf)
        if terminal is not None:
            leaf["_cutoff_value"] = terminal
            stop = -1
        else:
            value = float(search_play._values([_frame(leaf)])[0])
            if value >= CUTOFF_THETA or value <= 1.0 - CUTOFF_THETA:
                leaf["_cutoff_value"] = value
                stop = checkpoint
        if "_cutoff_value" in leaf:
            stops = _W.setdefault("cutoff_stops", {})
            stops[stop] = stops.get(stop, 0) + 1
            return leaf
    leaf = _W["original_rollout"](env, obs, first, until_tick, root, seed)
    terminal = _W["original_score"](leaf)
    leaf["_cutoff_value"] = (terminal if terminal is not None
                             else float(search_play._values([_frame(leaf)])[0]))
    stops = _W.setdefault("cutoff_stops", {})
    stops[until_tick - root_tick] = stops.get(until_tick - root_tick, 0) + 1
    return leaf


def _cutoff_score(obs):
    if "_cutoff_value" in obs:
        return float(obs["_cutoff_value"])
    return _W["original_score"](obs)


def _install(exact: bool) -> None:
    search_play = _W["search_play"]
    search_play._rollout_policy = _W["original_rollout"] if exact else _cutoff_rollout
    search_play._score_terminal_or_value = _W["original_score"] if exact else _cutoff_score


def _kill_env(env) -> None:
    """Drop a dead engine without env.close(): its wait() has no timeout."""
    process = getattr(env, "_process", None)
    try:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=10)
    except Exception:
        pass
    temporary = getattr(env, "_temporary_save", None)
    if temporary is not None:
        try:
            temporary.cleanup()
        except Exception:
            pass
        env._temporary_save = None
    env._process = None


def _recover() -> None:
    """Replace the worker's engine after SimulatorExited/BrokenPipe and forget per-game state."""
    _install(exact=True)
    _kill_env(_W["env"])
    _W["env"] = _open_env(_W["args"], _W["binary"])


def _part_path(args, family: str, seed: int) -> Path:
    return args.parts / f"{family}_{seed}.json"


def _needs_run(args, job: dict) -> bool:
    if args.rerun:
        return True
    path = _part_path(args, job["family"], job["seed"])
    if not path.is_file():
        return True
    try:
        return bool(json.loads(path.read_text()).get("error"))
    except ValueError:
        return True


def _write_part(args, result: dict) -> None:
    path = _part_path(args, result["family"], result["seed"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result) + "\n")
    os.replace(temporary, path)


def _play(job: dict) -> dict:
    from pvz_env import SimulatorExited
    args = _W["args"]
    attempts = 1 + max(0, args.retries)
    result = None
    for attempt in range(1, attempts + 1):
        try:
            result = _play_once(job)
            break
        except (SimulatorExited, BrokenPipeError, ConnectionError, EOFError, TimeoutError) as exc:
            print(f"{time.strftime('%H:%M:%S')} engine died {attempt}/{attempts} "
                  f"{job['family']} seed {job['seed']}: {type(exc).__name__}: {exc}", flush=True)
            if attempt == attempts:
                result = {"family": job["family"], "seed": job["seed"], "rows": [],
                          "won": None, "final_wave": None, "decisions": 0, "rollouts": 0,
                          "cutoff_stops": {}, "seconds_total": 0.0, "seconds_arms": {},
                          "error": f"{type(exc).__name__}: {exc}"}
                break
            _recover()
    _write_part(args, result)
    return result


def _play_once(job: dict) -> dict:
    import torch
    import search_play
    family, seed = job["family"], job["seed"]
    args = _W["args"]
    torch.manual_seed(seed)
    obs = _reset(family, seed)
    env = _W["env"]
    fast_state: dict = {}
    search_play._W["student_history"] = None
    config = argparse.Namespace(samples=args.samples, horizon=args.horizon, margin=args.margin,
                                min_gap=0, per_card=args.per_card, halving=True,
                                student_candidates=args.student_candidates,
                                verify_samples=args.verify_samples)
    rows, decisions, last_search_tick = [], 0, -10 ** 9
    seconds = {arm: 0.0 for arm in ("exact", "cutoff", "noise")}
    _W["cutoff_stops"] = {}
    started = time.perf_counter()

    def run(decision_seed: int, cutoff: bool) -> tuple[dict, float, dict]:
        _install(exact=not cutoff)
        stats = {"searched": 0, "overrides": 0, "override_kinds": Counter()}
        before = time.perf_counter()
        try:
            chosen = search_play._search(env, obs, action, config, decision_seed, stats)
        finally:
            _install(exact=True)
        return chosen, time.perf_counter() - before, stats

    try:
        while not obs["terminal"] and decisions < args.max_actions:
            action = search_play._fast_action(obs, fast_state)
            search_play._W["student_history"] = fast_state.get("history")
            legal = (obs.get("legal_actions") or {}).get("plants") or []
            searches = bool(legal) and (action["type"] == "plant"
                                        or obs["tick"] - last_search_tick >= args.min_gap)
            if searches:
                last_search_tick = obs["tick"]
                decision_seed = seed * 10000 + decisions
                order = ["exact", "cutoff", "noise"]
                rotation = len(rows) % 3
                order = order[rotation:] + order[:rotation]
                chosen, times = {}, {}
                for arm in order:
                    if arm == "exact":
                        chosen[arm], times[arm], stats = run(decision_seed, False)
                    elif arm == "noise":
                        chosen[arm], times[arm], stats = run(decision_seed + NOISE_OFFSET, False)
                    else:
                        chosen[arm], times[arm], stats = run(decision_seed, True)
                for arm in seconds:
                    seconds[arm] += times[arm]
                rows.append({"family": family, "seed": f"{family}:{seed}", "decision": decisions,
                             "tick": obs["tick"], "exact": chosen["exact"],
                             "approx": chosen["cutoff"], "exact_other_futures": chosen["noise"],
                             "seconds_exact": round(times["exact"], 4),
                             "seconds_cutoff": round(times["cutoff"], 4),
                             "seconds_noise": round(times["noise"], 4)})
            obs, _r, done, _t, info = env.step(action)
            if not info.get("ok"):
                obs, _r, done, _t, info = env.step({"type": "wait", "ticks": 60})
            decisions += 1
            if done:
                break
    finally:
        pass
    return {"family": family, "seed": seed, "rows": rows, "won": obs.get("result") == 1,
            "final_wave": obs["wave"], "decisions": decisions,
            "rollouts": sum(_W["cutoff_stops"].values()),
            "cutoff_stops": dict(sorted(_W["cutoff_stops"].items())),
            "seconds_total": round(time.perf_counter() - started, 2),
            "seconds_arms": {key: round(value, 2) for key, value in seconds.items()}}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", default="/home/newbiexvwu/PvZ-search/artifacts/student/dw4i_s1/model.pt")
    p.add_argument("--value", default="/home/newbiexvwu/PvZ-search/artifacts/value_net/v3/model.pt")
    p.add_argument("--static-value", default="/home/newbiexvwu/PvZ-search/artifacts/hard/value_branch1/model.pt")
    p.add_argument("--executable", default="/home/newbiexvwu/PvZ-search/.claude/worktrees/cq/build/pvz-portable")
    p.add_argument("--resource-dir", default="/home/newbiexvwu/.cache/pvz-research-resources")
    p.add_argument("--seeds", type=int, default=16)
    p.add_argument("--seed-offset", type=int, default=0, help="offset into the task's evaluation seed list")
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
    p.add_argument("--retries", type=int, default=2, help="restart the game after an engine death")
    p.add_argument("--parts", type=Path, default=ROOT / "artifacts" / "task40" / "agreement_parts")
    p.add_argument("--rerun", action="store_true", help="ignore per-game files already on disk")
    p.add_argument("--out", type=Path, default=ROOT / "artifacts" / "task40" / "agreement.json")
    args = p.parse_args()

    from render_episode import load_training_task
    jobs = []
    for family in args.families.split(","):
        manifest, task_id = FAMILIES[family]
        seeds = load_training_task(manifest, task_id)["seeds"][args.seed_offset:args.seed_offset + args.seeds]
        if len(seeds) != args.seeds:
            raise SystemExit(f"{family}: task has {len(seeds)} seeds at offset {args.seed_offset}, need {args.seeds}")
        jobs += [{"family": family, "seed": int(seed)} for seed in seeds]
    started = time.time()
    pending = [job for job in jobs if _needs_run(args, job)]
    if len(pending) < len(jobs):
        print(f"resume: {len(jobs) - len(pending)}/{len(jobs)} games already on disk", flush=True)
    if pending:
        start_method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
        with mp.get_context(start_method).Pool(args.workers, initializer=_init, initargs=(args,)) as pool:
            for index, result in enumerate(pool.imap_unordered(_play, pending), 1):
                print(f"{time.strftime('%H:%M:%S')} {index}/{len(pending)} {result['family']} "
                      f"seed {result['seed']} won={result['won']} searches={len(result['rows'])} "
                      f"stops={result['cutoff_stops']} seconds={result['seconds_total']}", flush=True)
    results = [json.loads(_part_path(args, job["family"], job["seed"]).read_text()) for job in jobs]
    rows = [row for result in results for row in result["rows"]]
    from search_agreement_stats import agreement_intervals
    stops: dict[str, int] = {}
    for result in results:
        for stop, count in result["cutoff_stops"].items():
            stops[str(stop)] = stops.get(str(stop), 0) + count
    errors = [{"family": result["family"], "seed": result["seed"], "error": result["error"]}
              for result in results if result.get("error")]
    summary = {"config": {key: str(value) for key, value in vars(args).items()},
               "checkpoints": list(CUTOFF_AT), "theta": CUTOFF_THETA,
               "games": len(results), "games_completed": sum(1 for result in results if result["rows"]),
               "searches": len(rows), "errors": errors,
               "seeds": [job["seed"] for job in jobs],
               "cutoff_stops": dict(sorted(stops.items(), key=lambda item: int(item[0]))),
               "seconds": round(time.time() - started, 1),
               "per_family": {}, "games_detail": [{key: result[key] for key in
                                                   ("family", "seed", "won", "final_wave", "decisions",
                                                    "rollouts", "cutoff_stops",
                                                    "seconds_total", "seconds_arms")} for result in results]}
    # A game whose engine died after every retry has no rows; keeping minimum_games at the
    # pre-registered count makes the gate fail instead of silently passing on fewer games.
    summary["summary"] = (agreement_intervals(rows, margin=0.03, minimum_games=len(results))
                          if rows else {"error": "no paired search points"})
    for family in sorted({result["family"] for result in results}):
        part = [row for row in rows if row["family"] == family]
        summary["per_family"][family] = (agreement_intervals(part, margin=0.03, minimum_games=args.seeds)
                                         if part else {"error": "no paired search points"})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1) + "\n")
    print(json.dumps({key: summary[key] for key in ("games", "searches", "seconds", "summary")}, indent=1))
    print(json.dumps(summary["per_family"], indent=1))


if __name__ == "__main__":
    main()
