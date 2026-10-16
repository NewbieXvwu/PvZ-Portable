"""How much does an approximate simulator (``SIM_SKIP`` mask) change what matters?

PvZ drives a lot of game logic from animation (plant attack timing, zombie walking and eating,
instant plants), so no ``SIM_SKIP`` bit may be called "cosmetic" from reading code. This script
measures it, two ways:

``games``   the same greedy policy plays the same seeds once on the exact simulator (mask 0) and
            once with ``--mask``. Reports outcome / final-wave agreement, how many games stay
            action-for-action identical, where the first divergence happens, and the speed.

``search``  at every decision where the deep placement search runs (student plays the real game),
            the search is run three times from the same root: exact, approximate (``--mask``
            during the rollouts only, and/or a cheaper ``--approx-top-k/-samples/-horizon``)
            and exact with a different set of resampled futures. The last one is the noise
            floor: if exact-vs-approximate disagree no more often than exact-vs-exact with other
            futures, the approximation costs nothing the search could have noticed.

Usage:
  python3 scripts/sim_skip_agreement.py games --mask 21 --seeds 64 --workers 3 --out /tmp/a.json
  python3 scripts/sim_skip_agreement.py games --mask 21 --manifest artifacts/stage4/night_l12_n1.json \
      --task-id night_l12_n1_eval --policy artifacts/fast_ppo/night_n1_full/model_iter100.pt
  python3 scripts/sim_skip_agreement.py search --mask 21 --seeds 6 --workers 3
  python3 scripts/sim_skip_agreement.py search --mask 0 --approx-samples 2 --seeds 6 --workers 3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = ROOT / "experiments/t7/bc_finetune_level7_v1/progress_evaluation_1024.json"
DEFAULT_TASK = "validation_full_level7_v1_d1"
DEFAULT_POLICY = "/Users/newbiexvwu/PvZAgent/artifacts/student/v3all/model.pt"
DEFAULT_VALUE = "/Users/newbiexvwu/PvZAgent/artifacts/value_net/v2/model.pt"
DEFAULT_RESOURCE_DIR = "/Users/newbiexvwu/Downloads/Plants_Vs_Zombies_V1.2.0.1073_EN"

_W: dict = {}


def _init(args) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    sys.path.insert(0, str(ROOT / "python"))
    sys.path.insert(0, str(ROOT / "scripts"))
    import torch
    torch.set_num_threads(1)
    from pvz_event_env import EventWaitEnv
    from pvz_rollout_policy import load_student, export_student
    net = load_student(args.policy)
    _W.update(net=net, args=args, env=EventWaitEnv(args.resource_dir, executable=args.executable, headless=True))
    if args.cpp_rollouts:
        import tempfile
        _W["temporary"] = tempfile.TemporaryDirectory(prefix="pvz-rollout-agreement-")
        binary = Path(_W["temporary"].name) / "student.bin"
        export_student(net, binary)
        _W["env"].load_rollout_net(binary)
    if args.mode == "search":
        import search_play
        from pvz_value_net import ValueNet
        vnet = ValueNet()
        vnet.load_state_dict(torch.load(args.value, map_location="cpu", weights_only=False)["state_dict"])
        vnet.eval()
        search_play._W.update(fast=net, vnet=vnet, lean_rollouts=True, batch_waits=True)


def _reset(seed: int):
    from pvz_env import TaskSpec
    from render_episode import load_training_task, training_task_kwargs
    from scripted_baseline import profile_for_deck
    args = _W["args"]
    kw = training_task_kwargs(load_training_task(args.manifest, args.task_id), seed)
    task = TaskSpec(level=kw["level"], seed=kw["seed"], playthrough=2, profile=profile_for_deck(tuple(kw["deck"])),
                    **{k: v for k, v in kw["task_extra"].items() if v is not None})
    obs, _ = _W["env"].reset(deck=tuple(kw["deck"]), task=task)
    return obs


def _play_game(job: tuple[int, int, bool]) -> dict:
    """One greedy game with the whole game on ``mask``; the env process is reused across games."""
    import torch
    from pvz_rollout_policy import act
    from render_episode import _frame
    seed, mask, approximate = job
    env = _W["env"]
    torch.manual_seed(seed)
    obs = _reset(seed)
    env.set_sim_skip(mask)
    started = time.perf_counter()
    memo = {"frame": None, "action": None}
    actions = []
    ready_mowers = lambda o: sum(d["state"] == 1 for d in o.get("defenses") or [])
    mowers_start = mowers_left = ready_mowers(obs)
    planted = {}
    if approximate and _W["args"].cpp_rollouts:
        obs = env.rollout(None, 0, sim_skip=mask, max_actions=_W["args"].max_actions)
        details = env._last_response
        env.set_sim_skip(0)
        return {"seed": seed, "mask": mask, "approximate": approximate, "backend": "cpp",
                "won": obs.get("result") == 1, "wave": obs["wave"], "tick": obs["tick"],
                "decisions": details["decisions"], "seconds": time.perf_counter() - started, "actions": None,
                "mowers_used": mowers_start - details["mowers_left"],
                "planted": {index: count for index, count in enumerate(details["planted"]) if count},
                "rng_state_sha256": hashlib.sha256(bytes.fromhex(env.privileged_state()["hidden"]["rand_state_hex"])).hexdigest(),
                "simulation_seconds": details["simulation_seconds"], "network_seconds": details["network_seconds"]}
    while not obs["terminal"] and len(actions) < _W["args"].max_actions:
        frame = _frame(obs)
        frame["action"] = memo["action"]
        action = act(_W["net"], frame, memo["frame"], (obs.get("legal_actions") or {}).get("plants") or [])
        memo["frame"], memo["action"] = frame, action
        obs, _r, done, _t, info = env.step(action)
        if info.get("ok") and action["type"] == "plant":
            planted[action["packet"]] = planted.get(action["packet"], 0) + 1
        if not info.get("ok"):
            obs, _r, done, _t, info = env.step({"type": "wait", "ticks": 60})
        actions.append(f"{obs['tick']}:{action['type'][0]}{action.get('packet', '')}"
                       f"{action.get('row', '')}{action.get('col', '')}")
        if not obs["terminal"]:
            mowers_left = ready_mowers(obs)
        if done:
            break
    seconds = time.perf_counter() - started
    env.set_sim_skip(0)
    return {"seed": seed, "mask": mask, "approximate": approximate, "backend": "python",
            "won": obs.get("result") == 1, "wave": obs["wave"], "tick": obs["tick"],
            "decisions": len(actions), "seconds": round(seconds, 3), "actions": actions,
            "mowers_used": mowers_start - mowers_left, "planted": planted,
            "rng_state_sha256": hashlib.sha256(bytes.fromhex(env.privileged_state()["hidden"]["rand_state_hex"])).hexdigest()}


def _search_decisions(seed: int) -> list[dict]:
    """Student plays the real game exactly; at each search point the search runs three ways."""
    import argparse as ap
    import torch
    import search_play
    from collections import Counter
    args = _W["args"]
    env = _W["env"]
    torch.manual_seed(seed)
    obs = _reset(seed)
    env.set_sim_skip(0)
    # min_gap is applied here, once, so all three arms search at exactly the same decisions.
    sargs = ap.Namespace(scope=args.scope, continuation="policy", top_k=args.top_k, samples=args.samples,
                         horizon=args.horizon, margin=0.02, min_gap=0, per_card=args.per_card, halving=False,
                         halving_gap=0.0)
    # The approximate arm: SIM_SKIP during rollouts and/or a cheaper search configuration.
    approx_sargs = ap.Namespace(**{**vars(sargs), **{k: v for k, v in (
        ("top_k", args.approx_top_k), ("samples", args.approx_samples), ("horizon", args.approx_horizon),
        ("halving", args.approx_halving or None), ("halving_gap", args.approx_halving_gap))
        if v is not None}})
    last_search_tick = -10 ** 9
    fast_state: dict = {}
    rows, decisions = [], 0

    def run(decision_seed: int, mask: int, policy_action: dict, config, cpp=False) -> tuple[dict, list, float]:
        stats = {"searched": 0, "overrides": 0, "override_kinds": Counter()}
        search_play._W["rollout_sim_skip"] = mask
        search_play._W["cpp_rollouts"] = cpp
        started = time.perf_counter()
        try:
            chosen = search_play._search(env, obs, policy_action, config, decision_seed, stats, {})
        finally:
            search_play._W["rollout_sim_skip"] = 0
        return chosen, [(json.dumps(c, sort_keys=True), v) for c, v in (stats.get("last_targets") or [])], \
            time.perf_counter() - started

    while not obs["terminal"] and decisions < args.max_actions:
        action = search_play._fast_action(obs, fast_state)
        legal = (obs.get("legal_actions") or {}).get("plants") or []
        searches = bool(legal) and (action["type"] == "plant" or (
            args.scope != "placement" and obs["tick"] - last_search_tick >= args.min_gap))
        if searches:
            last_search_tick = obs["tick"]
            decision_seed = seed * 10000 + decisions
            exact, exact_totals, t_exact = run(decision_seed, args.base_mask, action, sargs)
            approx, approx_totals, t_approx = run(decision_seed, args.mask, action, approx_sargs, args.cpp_rollouts)
            other, other_totals, _t = run(decision_seed + 7_777_777, args.base_mask, action, sargs)
            if exact_totals:
                rows.append({"seed": seed, "decision": decisions, "tick": obs["tick"],
                             "exact": exact, "approx": approx, "exact_other_futures": other,
                             "exact_totals": exact_totals, "approx_totals": approx_totals,
                             "other_totals": other_totals, "seconds_exact": round(t_exact, 3),
                             "seconds_approx": round(t_approx, 3)})
        obs, _r, done, _t, info = env.step(action)
        if not info.get("ok"):
            obs, _r, done, _t, info = env.step({"type": "wait", "ticks": 60})
        decisions += 1
        if done:
            break
    return rows


def _first_divergence(a: list[str], b: list[str]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def summarize_games(results: list[dict], mask: int) -> dict:
    from math import comb
    by = {(r["seed"], r["approximate"]): r for r in results}
    seeds = sorted({r["seed"] for r in results})
    pairs = [(by[(s, False)], by[(s, True)]) for s in seeds]
    diverged = [_first_divergence(e["actions"], a["actions"]) for e, a in pairs if a["actions"] is not None]
    n = len(pairs)
    exact_only = sum(e["won"] and not a["won"] for e, a in pairs)
    approx_only = sum(a["won"] and not e["won"] for e, a in pairs)
    discordant = exact_only + approx_only
    return {
        "mask": mask, "games": n,
        "wins_exact": sum(e["won"] for e, _a in pairs), "wins_approx": sum(a["won"] for _e, a in pairs),
        "outcome_agreement": round(sum(e["won"] == a["won"] for e, a in pairs) / n, 4),
        "exact_only_wins": sum(e["won"] and not a["won"] for e, a in pairs),
        "approx_only_wins": sum(a["won"] and not e["won"] for e, a in pairs),
        "mcnemar_exact_p": min(1.0, 2 * sum(comb(discordant, index) for index in range(min(exact_only, approx_only) + 1)) / 2 ** discordant),
        "final_wave_agreement": round(sum(e["wave"] == a["wave"] for e, a in pairs) / n, 4),
        "mowers_used_agreement": sum(e["mowers_used"] == a["mowers_used"] for e, a in pairs) / n,
        "planted_agreement": sum(e["planted"] == a["planted"] for e, a in pairs) / n,
        "rng_state_agreement": sum(e["rng_state_sha256"] == a["rng_state_sha256"] for e, a in pairs) / n,
        "mean_wave_exact": round(sum(e["wave"] for e, _a in pairs) / n, 3),
        "mean_wave_approx": round(sum(a["wave"] for _e, a in pairs) / n, 3),
        "identical_trajectories": sum(d is None for d in diverged) if len(diverged) == n else None,
        "first_divergence_decisions": sorted(d for d in diverged if d is not None),
        "seconds_exact": round(sum(e["seconds"] for e, _a in pairs), 2),
        "seconds_approx": round(sum(a["seconds"] for _e, a in pairs), 2),
        "speedup": round(sum(e["seconds"] for e, _a in pairs) / max(1e-9, sum(a["seconds"] for _e, a in pairs)), 3),
        "steps_per_second_exact": sum(e["decisions"] for e, _a in pairs) / sum(e["seconds"] for e, _a in pairs),
        "steps_per_second_approx": sum(a["decisions"] for _e, a in pairs) / sum(a["seconds"] for _e, a in pairs),
    }


def summarize_search(rows: list[dict], mask: int) -> dict:
    def key(a: dict) -> str:
        return json.dumps(a, sort_keys=True)

    def regret(row: dict, mode: str) -> float:
        totals = dict(row["exact_totals"])
        return max(totals.values()) - totals.get(key(row[mode]), max(totals.values()))
    def independent(row: dict, mode: str) -> float:
        totals = dict(row["other_totals"])
        return max(totals.values()) - totals.get(key(row[mode]), max(totals.values()))
    n = len(rows)
    t_exact = sum(r["seconds_exact"] for r in rows)
    t_approx = sum(r["seconds_approx"] for r in rows)
    return {
        "mask": mask, "searches": n,
        "approx_agrees_with_exact": round(sum(key(r["exact"]) == key(r["approx"]) for r in rows) / n, 4),
        "noise_floor_exact_vs_other_futures": round(
            sum(key(r["exact"]) == key(r["exact_other_futures"]) for r in rows) / n, 4),
        "mean_value_regret_approx": round(sum(regret(r, "approx") for r in rows) / n, 5),
        "mean_value_regret_other_futures": round(sum(regret(r, "exact_other_futures") for r in rows) / n, 5),
        # Fair to both arms: judged on the third, independent set of futures.
        "independent_regret_exact": round(sum(independent(r, "exact") for r in rows) / n, 5),
        "independent_regret_approx": round(sum(independent(r, "approx") for r in rows) / n, 5),
        "search_seconds_exact": round(t_exact, 2), "search_seconds_approx": round(t_approx, 2),
        "search_speedup": round(t_exact / max(1e-9, t_approx), 3),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=("games", "search"))
    p.add_argument("--mask", type=int, required=True)
    p.add_argument("--base-mask", type=int, default=0)
    p.add_argument("--cpp-rollouts", action="store_true", help="compare Python against native student rollouts")
    p.add_argument("--seeds", type=int, default=64)
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p.add_argument("--task-id", default=DEFAULT_TASK)
    p.add_argument("--policy", default=DEFAULT_POLICY, help="RolloutNet / fast_ppo checkpoint, played greedily")
    p.add_argument("--value", default=DEFAULT_VALUE)
    p.add_argument("--executable", default=str(ROOT / "build" / "pvz-portable"))
    p.add_argument("--resource-dir", default=DEFAULT_RESOURCE_DIR)
    p.add_argument("--max-actions", type=int, default=3000)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--horizon", type=int, default=3000)
    p.add_argument("--approx-top-k", type=int, help="search mode: cheaper configuration for the approximate arm")
    p.add_argument("--approx-samples", type=int)
    p.add_argument("--approx-horizon", type=int)
    p.add_argument("--approx-halving", action="store_true", help="search mode: successive halving (search_play --halving)")
    p.add_argument("--approx-halving-gap", type=float)
    p.add_argument("--scope", choices=("placement", "all"), default="placement",
                   help="search mode: placement = re-place pi's card; all = wide search incl. wait")
    p.add_argument("--per-card", type=int, default=0, help="search mode: search_play --per-card")
    p.add_argument("--min-gap", type=int, default=0, help="search mode: search_play --min-gap (scope all)")
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    sys.path.insert(0, str(ROOT / "scripts"))
    from render_episode import load_training_task
    seeds = load_training_task(args.manifest, args.task_id)["seeds"][args.seed_offset: args.seed_offset + args.seeds]
    with mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(args,)) as pool:
        if args.mode == "games":
            jobs = [(s, m, approximate) for s in seeds for m, approximate in ((args.base_mask, False), (args.mask, True))]
            results = list(pool.imap_unordered(_play_game, jobs))
            summary = summarize_games(results, args.mask)
        else:
            results = [row for rows in pool.imap_unordered(_search_decisions, seeds) for row in rows]
            summary = summarize_search(results, args.mask)
    summary.update(task_id=args.task_id, policy=str(args.policy), approx_top_k=args.approx_top_k,
                   cpp_rollouts=args.cpp_rollouts, base_mask=args.base_mask,
                   approx_halving=args.approx_halving, approx_halving_gap=args.approx_halving_gap,
                   scope=args.scope, per_card=args.per_card, min_gap=args.min_gap, samples=args.samples,
                   approx_samples=args.approx_samples, approx_horizon=args.approx_horizon,
                   executable_sha256=hashlib.sha256(Path(args.executable).read_bytes()).hexdigest()[:16])
    print(json.dumps(summary, indent=1))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"summary": summary, "results": results}) + "\n")


if __name__ == "__main__":
    main()
