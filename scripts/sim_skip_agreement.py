"""How much does an approximate simulator (``SIM_SKIP`` mask) change what matters?

PvZ drives a lot of game logic from animation (plant attack timing, zombie walking and eating,
instant plants), so no ``SIM_SKIP`` bit may be called "cosmetic" from reading code. This script
measures it, two ways:

``games``   the same greedy policy plays the same seeds once on the exact simulator (mask 0) and
            once with ``--mask``. Reports outcome / final-wave agreement, how many games stay
            action-for-action identical, where the first divergence happens, and the speed.

``search``  at every decision where the teacher's search runs (student plays the real game),
            the search is run three times from the same root: exact, approximate (``--mask``
            during the rollouts only, and/or a cheaper ``--approx-per-card/-samples/-horizon``)
            and exact with a different set of resampled futures. The last one is the noise
            floor: if exact-vs-approximate disagree no more often than exact-vs-exact with other
            futures, the approximation costs nothing the search could have noticed.

Usage:
  python3 scripts/sim_skip_agreement.py games --mask 21 --seeds 64 --workers 3 --out /tmp/a.json
  python3 scripts/sim_skip_agreement.py games --mask 21 --manifest artifacts/stage4/night_l12_n1.json \
      --task-id night_l12_n1_eval --policy artifacts/fast_ppo/night_n1_full/model_iter100.pt
  python3 scripts/sim_skip_agreement.py search --mask 21 --seeds 6 --workers 3
  python3 scripts/sim_skip_agreement.py search --mask 0 --approx-samples 2 --seeds 6 --workers 3
  python3 scripts/sim_skip_agreement.py search --mask 0 --cuda-rollouts --seeds 6 --workers 2
        ^ the GPU arm: same candidates, same futures, same horizon, only the continuation engine differs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import random
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
    if args.cpp_rollouts or args.base_cpp_rollouts:
        import tempfile
        _W["temporary"] = tempfile.TemporaryDirectory(prefix="pvz-rollout-agreement-")
        binary = Path(_W["temporary"].name) / "student.bin"
        export_student(net, binary)
        _W["env"].load_rollout_net(binary)
    if args.mode == "search":
        import search_play
        keep, paths = search_play._export_values(args.value, args.static_value)
        _W['value_keep'] = keep
        search_play._W.update(fast=net, value_files=paths, env=_W['env'], lean_rollouts=True, batch_waits=True)
        _W['env'].load_value_net(paths[0])
        _W['env'].load_value_net(paths[1], static=True)
        if args.cuda_rollouts:
            # The GPU arm needs a place to drop the root it is about to continue (`env.cuda_dump` must run
            # right after `env.restore`, because the income-window history lives in Python).
            import tempfile
            from pvz_cuda_rollout import CudaRolloutClient
            temporary = tempfile.TemporaryDirectory(prefix="pvz-cuda-dump-")
            _W["cuda_dump_tmp"] = temporary
            search_play._W["cuda_dump_tmp"] = temporary
            search_play._W["cuda_dump"] = str(Path(temporary.name) / "root.cudaroot")
            _W["cuda_client"] = CudaRolloutClient(args.cuda_socket)
        if args.approx_static_value:
            approx_keep, approx_paths = search_play._export_values(args.value, args.approx_static_value)
            _W.update(reference_static_file=paths[1], approx_static_file=approx_paths[1], approx_value_keep=approx_keep)


def _reset(seed: int):
    from pvz_env import TaskSpec
    from render_episode import load_training_task, training_task_kwargs
    from pvz_tasks import profile_for_deck
    args = _W["args"]
    kw = training_task_kwargs(load_training_task(args.manifest, args.task_id), seed)
    task = TaskSpec(level=kw["level"], seed=kw["seed"], playthrough=2, profile=profile_for_deck(tuple(kw["deck"]), kw.get("first_aid_owned", False)),
                    **{k: v for k, v in kw["task_extra"].items() if v is not None})
    obs, _ = _W["env"].reset(deck=tuple(kw["deck"]), task=task)
    if args.mode == "search":
        import search_play
        from pvz_value_net import observation_context
        search_play._W["context"] = observation_context(obs)
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
    history = None
    if _W["net"].obs != "v1":
        from pvz_student_observation import StudentHistory
        history = StudentHistory(_W["net"].obs)
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
        action = act(_W["net"], frame, memo["frame"], (obs.get("legal_actions") or {}).get("plants") or [],
                     history=history)
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
    sargs = ap.Namespace(samples=args.samples, horizon=args.horizon, margin=args.margin, min_gap=0, per_card=args.per_card,
                         halving=args.base_halving, student_candidates=args.student_candidates,
                         verify_samples=args.search_verify_samples)
    # The approximate arm: SIM_SKIP during rollouts and/or a cheaper search configuration.
    approx_sargs = ap.Namespace(**{**vars(sargs), **{k: v for k, v in (
        ("per_card", args.approx_per_card), ("samples", args.approx_samples), ("horizon", args.approx_horizon),
        ("halving", args.approx_halving or None))
        if v is not None}})
    if args.label_noise:
        approx_sargs = ap.Namespace(**{**vars(sargs), "samples": args.high_samples, "halving": False})
    last_search_tick = -10 ** 9
    fast_state: dict = {}
    search_play._W["student_history"] = None
    rows, decisions = [], 0
    dagger = random.Random(seed) if args.dagger_beta else None
    stream = (args.jsonl_dir / f"seed_{seed}.jsonl").open("x", buffering=1) if args.jsonl_dir else None

    def run(decision_seed: int, mask: int, policy_action: dict, config, cpp=False,
            cuda=False) -> tuple[dict, list, float, dict | None, dict]:
        stats = {"searched": 0, "overrides": 0, "override_kinds": Counter()}
        search_play._W["rollout_sim_skip"] = mask
        search_play._W["cpp_rollouts"] = cpp
        # The GPU arm swaps only the continuation engine: same candidates, same futures, same horizon.
        search_play._W["cuda"] = _W["cuda_client"] if cuda else None
        if args.approx_static_value:
            env.load_value_net(_W['approx_static_file'] if config is approx_sargs else _W['reference_static_file'], static=True)
        started = time.perf_counter()
        try:
            chosen = search_play._search(env, obs, policy_action, config, decision_seed, stats)
        finally:
            search_play._W["rollout_sim_skip"] = 0
            search_play._W["cuda"] = None
        audit = {"student_candidates": config.student_candidates, "verify_samples": config.verify_samples,
                 "horizon": config.horizon, "rollout_mask": mask,
                 "student_candidates_added": stats.get("student_candidate_added", 0),
                 "verification_samples_used": (stats.get("last_verification") or {}).get("samples", 0),
                 "searched": stats["searched"], "rollout_lanes": stats.get("rollout_lanes", 0)}
        return chosen, [(json.dumps(c, sort_keys=True), v) for c, v in (stats.get("last_targets") or [])], \
            time.perf_counter() - started, stats.get("last_best"), audit

    while not obs["terminal"] and decisions < args.max_actions:
        action = search_play._fast_action(obs, fast_state)
        search_play._W["student_history"] = fast_state.get("history")
        legal = (obs.get("legal_actions") or {}).get("plants") or []
        searches = bool(legal) and (action["type"] == "plant" or obs["tick"] - last_search_tick >= args.min_gap)
        if searches:
            last_search_tick = obs["tick"]
            decision_seed = seed * 10000 + decisions
            lanes_before = _W["cuda_client"].lanes if args.cuda_rollouts else 0
            runs = [(args.base_mask, sargs, args.base_cpp_rollouts, False),
                    (args.mask, approx_sargs, args.cpp_rollouts, args.cuda_rollouts)]
            runs.append((args.base_mask, sargs, args.base_cpp_rollouts, False))
            order = [0, 1, 2] if args.label_noise else ([0, 1, 2] if len(rows) % 2 == 0 else [1, 0, 2])
            if args.label_noise:
                rotation = len(rows) % 3
                order = order[rotation:] + order[:rotation]
                runs[1] = (args.base_mask, approx_sargs, args.base_cpp_rollouts, False)
            offsets = [0, 15_555_557 if args.label_noise else 0, 7_777_777]
            if args.label_noise:
                for samples in args.verify_samples:
                    runs.append((args.base_mask, ap.Namespace(**{**vars(sargs), "verify_samples": samples}),
                                 args.base_cpp_rollouts, False))
                    offsets.append(0)
                order = list(range(len(runs)))
                rotation = len(rows) % len(runs)
                order = order[rotation:] + order[:rotation]
            results = {index: run(decision_seed + offsets[index], runs[index][0], action, runs[index][1], runs[index][2], runs[index][3])
                       for index in order}
            exact, exact_totals, t_exact, _best = results[0][:4]
            approx, approx_totals, t_approx, _best = results[1][:4]
            other, other_totals, _t, _best = results[2][:4]
            if exact_totals:
                rows.append({"seed": seed, "decision": decisions, "tick": obs["tick"],
                             "exact": exact, "approx": approx, "exact_other_futures": other,
                             "exact_totals": exact_totals, "approx_totals": approx_totals,
                             "other_totals": other_totals, "seconds_exact": round(t_exact, 3),
                             "seconds_approx": round(t_approx, 3),
                             "teacher_audit": {"reference": results[0][4], "approximate": results[1][4],
                                               "other_futures": results[2][4]},
                             "cuda_lanes": (_W["cuda_client"].lanes - lanes_before) if args.cuda_rollouts else 0})
                if args.label_noise:
                    rows[-1] = {"seed": seed, "decision": decisions, "tick": obs["tick"], "wave": obs["wave"],
                                "student": action, "margin": args.margin,
                                "arms": {name: {"label": results[index][0], "values": results[index][1],
                                                "seconds": results[index][2], "best": results[index][3],
                                                "decision_seed": decision_seed + offsets[index]}
                                         for name, index in [("A", 0), ("B", 2), ("H", 1)] +
                                         [(f"V{samples}", index + 3) for index, samples in enumerate(args.verify_samples)]}}
                if stream:
                    stream.write(json.dumps(rows[-1], allow_nan=False) + "\n")
        executed = exact if searches and dagger is not None else action
        if dagger is not None and dagger.random() >= args.dagger_beta:
            executed = action
        fast_state["action"] = executed
        obs, _r, done, _t, info = env.step(executed)
        if not info.get("ok"):
            obs, _r, done, _t, info = env.step({"type": "wait", "ticks": 60})
        decisions += 1
        if done:
            break
    if stream:
        stream.close()
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
        # How many of those searches the GPU arm actually ran on the GPU (a root the shared core does not
        # cover falls back to the C++ engine, which would silently raise the agreement).
        "gpu_covered_searches": sum(1 for r in rows if r.get("cuda_lanes")),
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
    p.add_argument("--base-halving", action="store_true", help="search mode: reference teacher uses halving")
    p.add_argument("--base-cpp-rollouts", action="store_true", help="reference teacher also uses native rollouts")
    p.add_argument("--cpp-rollouts", action="store_true", help="approximate arm uses native student rollouts")
    p.add_argument("--cuda-rollouts", action="store_true",
                   help="search mode: approximate arm continues on the GPU (build/rollout_cuda --serve)")
    p.add_argument("--cuda-socket", help="an already-running GPU server; by default start one here")
    p.add_argument("--cuda-binary", default=str(ROOT / "build/rollout_cuda"))
    p.add_argument("--cuda-lanes", type=int, default=2048)
    p.add_argument("--cuda-gather-ms", type=int, default=80)
    p.add_argument("--seeds", type=int, default=64)
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p.add_argument("--task-id", default=DEFAULT_TASK)
    p.add_argument("--policy", default=DEFAULT_POLICY, help="RolloutNet / fast_ppo checkpoint, played greedily")
    p.add_argument("--value", default=DEFAULT_VALUE)
    p.add_argument("--static-value", help="search mode: separate net for choosing cells")
    p.add_argument("--approx-static-value", help="search mode: compare another cell-scoring checkpoint")
    p.add_argument("--executable", default=str(ROOT / "build" / "pvz-portable"))
    p.add_argument("--resource-dir", default=DEFAULT_RESOURCE_DIR)
    p.add_argument("--max-actions", type=int, default=3000)
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--horizon", type=int, default=3000)
    p.add_argument("--approx-per-card", type=int, help="search mode: cheaper configuration for the approximate arm")
    p.add_argument("--approx-samples", type=int)
    p.add_argument("--approx-horizon", type=int)
    p.add_argument("--approx-halving", action="store_true", help="search mode: successive halving (search_play --halving)")
    p.add_argument("--per-card", type=int, default=1, help="search mode: search_play --per-card")
    p.add_argument("--student-candidates", type=int, default=0,
                   help="search mode: same student top-k additions as search_play (default preserves old diagnostics)")
    p.add_argument("--search-verify-samples", type=int, default=0,
                   help="search mode: ordinary verification futures in all three arms; separate from label-noise --verify-samples")
    p.add_argument("--min-gap", type=int, default=300, help="search mode: search_play --min-gap")
    p.add_argument("--margin", type=float, default=0.02)
    p.add_argument("--dagger-beta", type=float, default=0.0, help="search mode: probability of executing reference teacher")
    p.add_argument("--label-noise", action="store_true", help="search mode: independent A/B teachers and full-sample H")
    p.add_argument("--high-samples", type=int, default=12, help="label noise: futures for every H candidate, no halving")
    p.add_argument("--verify-samples", nargs="+", type=int, default=[],
                   help="label noise: compare A with these extra pairwise verification counts")
    p.add_argument("--jsonl-dir", type=Path, help="search mode: stream each search point to a per-seed JSONL")
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    if args.student_candidates < 0 or args.search_verify_samples < 0:
        p.error("--student-candidates and --search-verify-samples must be nonnegative")
    if args.cuda_rollouts:
        if args.mode != "search":
            p.error("--cuda-rollouts is a search-mode arm")
    if not 0 <= args.dagger_beta <= 1:
        p.error("--dagger-beta must be between 0 and 1")
    if args.label_noise and (args.mode != "search" or args.jsonl_dir is None):
        p.error("--label-noise requires search mode and --jsonl-dir")
    if args.verify_samples and (not args.label_noise or min(args.verify_samples) <= 0
                                or len(set(args.verify_samples)) != len(args.verify_samples)):
        p.error("--verify-samples requires --label-noise and distinct positive counts")
    if args.jsonl_dir:
        args.jsonl_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT / "python"))
    from render_episode import load_training_task
    seeds = load_training_task(args.manifest, args.task_id)["seeds"][args.seed_offset: args.seed_offset + args.seeds]
    if len(seeds) != args.seeds:
        p.error("requested seed range exceeds the manifest")
    cuda_keep = cuda_server = cuda_log = None
    cuda_report = ""
    if args.cuda_rollouts and not args.cuda_socket:
        import tempfile
        from pvz_cuda_rollout import serve_policy
        cuda_keep = tempfile.TemporaryDirectory(prefix="pvz-cuda-agreement-")
        cuda_log = Path(cuda_keep.name) / "cuda_server.log"
        with cuda_log.open("w+") as handle:
            cuda_server, args.cuda_socket = serve_policy(args.cuda_binary, args.policy, cuda_keep.name,
                                                         lanes=args.cuda_lanes,
                                                         gather_ms=args.cuda_gather_ms,
                                                         start_timeout_ms=600000, log=handle)
    try:
        with mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(args,)) as pool:
            if args.mode == "games":
                jobs = [(s, m, approximate) for s in seeds for m, approximate in ((args.base_mask, False), (args.mask, True))]
                results = list(pool.imap_unordered(_play_game, jobs))
                summary = summarize_games(results, args.mask)
            else:
                results = []
                completed = 0
                for rows in pool.imap_unordered(_search_decisions, seeds):
                    results.extend(rows)
                    completed += 1
                    partial = summarize_search(results, args.mask) if results else {"searches": 0}
                    print(json.dumps({"completed_seeds": completed, "total_seeds": len(seeds),
                                      "partial": partial}), flush=True)
                    if args.out:
                        args.out.parent.mkdir(parents=True, exist_ok=True)
                        args.out.with_suffix(".partial.json").write_text(json.dumps(
                            {"completed_seeds": completed, "total_seeds": len(seeds),
                             "summary": partial, "results": results}) + "\n")
                if args.label_noise:
                    from label_noise import read_rows, summarize
                    summary = summarize(read_rows(sorted(args.jsonl_dir.glob("seed_*.jsonl"))))
                else:
                    summary = summarize_search(results, args.mask)
    finally:
        if cuda_server is not None:
            import subprocess
            try:
                # Closing the workers' sockets is what shuts the server down; a server that outlives the
                # measurement would hold its VRAM away from the formal experiments.
                cuda_server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                cuda_server.terminate()
            cuda_server.stdout.close()
            cuda_report = cuda_log.read_text(errors="replace").strip()
            cuda_keep.cleanup()
    summary.update(task_id=args.task_id, policy=str(args.policy), approx_per_card=args.approx_per_card,
                   cpp_rollouts=args.cpp_rollouts, base_cpp_rollouts=args.base_cpp_rollouts,
                   cuda_rollouts=args.cuda_rollouts, cuda_server=cuda_report,
                   base_halving=args.base_halving, static_value=args.static_value, base_mask=args.base_mask,
                   approx_static_value=args.approx_static_value,
                   approx_halving=False if args.label_noise else args.approx_halving or args.base_halving,
                   per_card=args.per_card, min_gap=args.min_gap, samples=args.samples,
                   student_candidates=args.student_candidates, search_verify_samples=args.search_verify_samples,
                   approx_samples=args.high_samples if args.label_noise else args.approx_samples,
                   approx_horizon=args.horizon if args.label_noise else args.approx_horizon,
                   margin=args.margin, dagger_beta=args.dagger_beta, label_noise=args.label_noise,
                   high_samples=args.high_samples, horizon=args.horizon, seed_offset=args.seed_offset,
                   workers=args.workers, seeds=seeds, complete=True,
                   cuda_binary_sha256=(hashlib.sha256(Path(args.cuda_binary).read_bytes()).hexdigest()
                                       if args.cuda_rollouts else None),
                   policy_sha256=hashlib.sha256(Path(args.policy).read_bytes()).hexdigest(),
                   value_sha256=hashlib.sha256(Path(args.value).read_bytes()).hexdigest(),
                   executable_sha256=hashlib.sha256(Path(args.executable).read_bytes()).hexdigest()[:16])
    print(json.dumps(summary, indent=1))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"summary": summary, "results": results}) + "\n")


if __name__ == "__main__":
    main()
