"""Play games with the policy pi plus a value-guided search, paired against pi alone (DESIGN.md §13, stage 2).

Search v0 (deliberately cheap; it answers "can the value net correct the policy at all?"):

  At a decision where some plant is legal (packet ready, sun enough, free cell):
  1. Every legal plant is applied from a snapshot (``branch_snapshot``) and the resulting board
     is scored by V. The top ``--top-k`` plants survive.
  2. Each survivor, the "do nothing now" option and pi's own choice are rolled forward
     ``--horizon`` ticks with no further actions, ``--samples`` times, each time after
     ``RESEED`` (so the search never sees the real future). All candidates share the same
     reseed values (common random numbers), and V scores the board at the end.
  3. pi's choice is overridden only when the best candidate beats it by ``--margin``.

pi is played greedily in both arms, so "baseline" and "search" differ only by the search.
Seeds come from the validation task (30000+) -- the value net must be trained on other seeds.

Usage:
  python scripts/search_play.py --policy <pi.pt> --value artifacts/value_net/v2/model.pt \
      --seeds 64 --workers 14 --out artifacts/search/v0_pilot
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "experiments/t7/bc_finetune_level7_v1/progress_evaluation_1024.json"  # first 256 seeds = the 256-seed set
TASK_ID = "validation_full_level7_v1_d1"
DEFAULT_RESOURCE_DIR = "/Users/newbiexvwu/Downloads/Plants_Vs_Zombies_V1.2.0.1073_EN"
NOOP = {"type": "wait", "ticks": 60, "until": "timeout"}

_W: dict = {}


def _init_worker(policy_path: str, value_path: str, rollout_path: str | None = None) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.path.insert(0, str(ROOT / "python"))
    sys.path.insert(0, str(ROOT / "scripts"))
    import torch
    torch.set_num_threads(1)
    from pvz_value_net import ValueNet
    from render_episode import load_checkpoint_policy
    model, _ = load_checkpoint_policy(policy_path)
    vnet = ValueNet()
    vnet.load_state_dict(torch.load(value_path, map_location="cpu", weights_only=False)["state_dict"])
    vnet.eval()
    _W.update(model=model, vnet=vnet)
    if rollout_path:
        from pvz_rollout_policy import RolloutNet
        fast = RolloutNet()
        weights = torch.load(rollout_path, map_location="cpu", weights_only=False)["state_dict"]
        # fast_ppo.py checkpoints are the same net plus a value head; the policy ignores it.
        fast.load_state_dict({k: v for k, v in weights.items() if not k.startswith("value.")})
        fast.eval()
        _W["fast"] = fast


def _fast_action(obs: dict, fast_state: dict) -> dict:
    """The fast rollout policy's action; ``fast_state`` carries the previous frame and action."""
    from pvz_rollout_policy import act
    from render_episode import _frame
    frame = _frame(obs)
    frame["action"] = fast_state.get("action")
    action = act(_W["fast"], frame, fast_state.get("frame"), (obs.get("legal_actions") or {}).get("plants") or [])
    fast_state["frame"] = frame
    fast_state["action"] = action
    return action


def _values(frames: list[dict]) -> list[float]:
    import torch
    from pvz_value_net import featurize
    boards, scalars = zip(*(featurize(f) for f in frames))
    with torch.inference_mode():
        return torch.sigmoid(_W["vnet"](torch.stack(boards), torch.stack(scalars))).tolist()


def _score_terminal_or_value(obs: dict) -> float | None:
    if obs.get("terminal"):
        return 1.0 if obs.get("result") == 1 else 0.0
    return None


def _rollout_policy(env, obs: dict, state: dict, first: dict, until_tick: int) -> dict:
    """Play ``first`` and then let pi -- or the fast rollout policy, if loaded -- continue until
    ``until_tick`` (0 = to the end of the game)."""
    from render_episode import _ppo_action, _frame
    rs = dict(state)
    fast_state = {"frame": _frame(obs), "action": first} if "fast" in _W else None
    until_tick = until_tick if until_tick > obs["tick"] else 10 ** 9
    action = first
    while True:
        obs, _r, done, _t, info = env.step(action)
        if not info.get("ok"):
            action = {"type": "wait", "ticks": 60}
            obs, _r, done, _t, info = env.step(action)
        rs.update(previous_action=action, delta_ticks=info.get("ticks_advanced") or 0,
                  events=info.get("events"), previous_wait_result=info.get("wait_result"))
        if done or obs.get("terminal") or obs["tick"] >= until_tick:
            return obs
        if fast_state is not None:
            action = _fast_action(obs, fast_state)
        else:
            action = _ppo_action(_W["model"], obs, rs, deterministic=True)


def _search(env, obs: dict, policy_action: dict, args, decision_seed: int, stats: dict,
            state: dict | None = None) -> dict:
    from render_episode import _frame
    legal = (obs.get("legal_actions") or {}).get("plants") or []
    if args.scope == "placement":
        # Only *where* to put the card pi chose: same card, same cost, so a short horizon
        # cannot bias the comparison toward "keep the sun" (see learn/02-search.md).
        if policy_action["type"] != "plant":
            return policy_action
        legal = [a for a in legal if a["packet"] == policy_action["packet"]]
    if not legal:
        return policy_action
    if (args.min_gap and policy_action["type"] != "plant"
            and obs["tick"] - stats.get("last_search_tick", -10 ** 9) < args.min_gap):
        return policy_action
    stats["last_search_tick"] = obs["tick"]
    stats["searched"] += 1
    root = env.snapshot()
    income = obs.get("sun_income_rate")
    try:
        plants = [{"type": "plant", "packet": a["packet"], "col": a["col"], "row": a["row"]} for a in legal]
        static = []
        for start in range(0, len(plants), 128):
            chunk = plants[start:start + 128]
            for action, record in zip(chunk, env.branch_snapshot(root, chunk)):
                if record.get("snapshot_id"):
                    env.release_snapshot(record["snapshot_id"])
                if record.get("ok") and record.get("observation"):
                    frame = _frame(record["observation"])
                    frame["sun_income_rate"] = income
                    static.append((action, frame))
        scored = sorted(zip(_values([f for _a, f in static]), [a for a, _f in static]),
                        key=lambda item: -item[0]) if static else []
        candidates = [a for _v, a in scored[: args.top_k]] + ([] if args.scope == "placement" else [NOOP])
        policy_key = json.dumps(policy_action if policy_action["type"] == "plant" else NOOP, sort_keys=True)
        if policy_action["type"] == "plant" and all(json.dumps(c, sort_keys=True) != policy_key for c in candidates):
            candidates.append(policy_action)

        totals = []
        for cand in candidates:
            leaf_frames, fixed = [], []
            for k in range(args.samples):
                env.restore(root)
                env.reseed((decision_seed * 1000 + k) % 2 ** 32)
                if args.continuation == "policy":
                    leaf = _rollout_policy(env, obs, state, cand, obs["tick"] + args.horizon)
                else:
                    if cand["type"] == "plant":
                        env.step(cand)
                    leaf, *_ = env.step({"type": "wait", "ticks": args.horizon})
                terminal = _score_terminal_or_value(leaf)
                if terminal is None:
                    leaf_frames.append(_frame(leaf))
                else:
                    fixed.append(terminal)
            values = fixed + (_values(leaf_frames) if leaf_frames else [])
            totals.append(sum(values) / len(values))
        keys = [json.dumps(c, sort_keys=True) for c in candidates]
        stats["last_targets"] = [(c, v) for c, v in zip(candidates, totals)]
        policy_value = totals[keys.index(policy_key)]
        best = max(range(len(candidates)), key=lambda i: totals[i])
        if totals[best] - policy_value > args.margin and keys[best] != policy_key:
            stats["overrides"] += 1
            stats["override_kinds"][candidates[best]["type"] + "_over_" + policy_action["type"]] += 1
            return candidates[best]
        return policy_action
    finally:
        env.restore(root)
        env.release_snapshot(root)


DELAY = {"type": "wait", "ticks": 300, "until": "timeout"}


def _timing_search(env, obs: dict, policy_action: dict, args, decision_seed: int, stats: dict,
                   state: dict) -> dict:
    """Non-myopic "plant or not now", then (if planting) v0b's placement search.

    Two candidates only, each rolled out ``--timing-horizon`` ticks with pi itself playing on
    (so an investment's payback is inside the horizon), after RESEED, common random numbers:
      * pi wants to plant  -> plant now  vs  wait 300 ticks and let pi decide again;
      * pi wants to wait and has >= ``--timing-sun`` sun -> pi's wait  vs  the best plant by static V.
    Everything after that one decision is pi's own play, so the comparison is fair to both.
    """
    from render_episode import _frame
    legal = (obs.get("legal_actions") or {}).get("plants") or []
    decision = policy_action
    if legal and (policy_action["type"] == "plant" or (
            obs["sun"] >= args.timing_sun
            and obs["tick"] - stats.get("last_timing_tick", -10 ** 9) >= args.timing_gap)):
        stats["last_timing_tick"] = obs["tick"]
        stats["timing_searched"] = stats.get("timing_searched", 0) + 1
        root = env.snapshot()
        try:
            if policy_action["type"] == "plant":
                alternative = DELAY
            else:
                plants = [{"type": "plant", "packet": a["packet"], "col": a["col"], "row": a["row"]} for a in legal]
                frames, acts = [], []
                for start in range(0, len(plants), 128):
                    chunk = plants[start:start + 128]
                    for action, record in zip(chunk, env.branch_snapshot(root, chunk)):
                        if record.get("snapshot_id"):
                            env.release_snapshot(record["snapshot_id"])
                        if record.get("ok") and record.get("observation"):
                            frame = _frame(record["observation"])
                            frame["sun_income_rate"] = obs.get("sun_income_rate")
                            frames.append(frame); acts.append(action)
                alternative = acts[max(range(len(acts)), key=_values(frames).__getitem__)] if acts else None
            if alternative is not None:
                totals = []
                for cand in (policy_action, alternative):
                    leaves, fixed = [], []
                    for k in range(args.samples):
                        env.restore(root)
                        env.reseed((decision_seed * 1000 + 500 + k) % 2 ** 32)
                        leaf = _rollout_policy(env, obs, state, cand, obs["tick"] + args.timing_horizon)
                        terminal = _score_terminal_or_value(leaf)
                        (fixed.append(terminal) if terminal is not None else leaves.append(_frame(leaf)))
                    values = fixed + (_values(leaves) if leaves else [])
                    totals.append(sum(values) / len(values))
                if totals[1] - totals[0] > args.timing_margin:
                    decision = alternative
                    stats["overrides"] += 1
                    stats["override_kinds"][f"timing_{alternative['type']}_over_{policy_action['type']}"] += 1
        finally:
            env.restore(root)
            env.release_snapshot(root)
    if decision["type"] == "plant":
        placement = argparse.Namespace(**{**vars(args), "scope": "placement", "continuation": "passive",
                                          "min_gap": 0})
        decision = _search(env, obs, decision, placement, decision_seed, stats, state)
    return decision


def _transition(obs: dict, action: dict, state: dict, info: dict, config: dict, overridden: bool) -> dict:
    """One behaviour-cloning row in the format of scripts/bc_pipeline (train_bc_gru_sequence.py).

    ``state`` still holds the step *before* this action (previous action, elapsed ticks, events),
    which is exactly what the policy conditioned on when it chose.
    """
    import pvz_agent_model as api
    tensors, metadata = api.observation_tokens(obs, config["input_flags"])
    clean = {k: action[k] for k in ("type", "packet", "row", "col", "ticks", "until") if k in action}
    return {
        "tokens": api.pack_tokens(tensors, metadata),
        "wave": int(obs["wave"]),
        "legal": api.policy_legal_summary(obs, config),
        "previous_action": state["previous_action"],
        "elapsed_since_previous_observation": int(state["delta_ticks"] or 0),
        "events": state["events"] or {},
        "previous_wait_result": state["previous_wait_result"],
        "action": clean,
        "wait_result": info.get("wait_result"),
        "action_duration_ticks": int(info.get("ticks_advanced", 0)),
        "search_override": overridden,
    }


def _play(job: dict) -> dict:
    from collections import Counter
    import torch
    from pvz_event_env import policy_env
    from render_episode import _ppo_action, load_training_task, training_task_kwargs
    from scripted_baseline import profile_for_deck
    from pvz_env import TaskSpec
    args = job["args"]
    manifest = Path(args.manifest) if args.manifest else MANIFEST
    kw = training_task_kwargs(load_training_task(manifest, args.task_id or TASK_ID), job["seed"])
    model = _W["model"]
    env = policy_env(model.config, args.resource_dir, headless=True)
    task = TaskSpec(level=kw["level"], seed=kw["seed"], playthrough=2,
                    profile=profile_for_deck(tuple(kw["deck"])),
                    **{k: v for k, v in kw["task_extra"].items() if v is not None})
    torch.manual_seed(kw["seed"])
    started = time.time()
    stats = {"searched": 0, "overrides": 0, "override_kinds": Counter()}
    try:
        obs, _ = env.reset(deck=tuple(kw["deck"]), task=task)
        state = {"hidden": None, "previous_action": None, "delta_ticks": 0, "events": None,
                 "previous_wait_result": None}
        decisions = 0
        rows = [] if args.record else None
        from render_episode import _frame
        frames = [_frame(obs)] if args.frames else None
        fast_state = {}
        while not obs["terminal"] and decisions < args.max_actions:
            if job["arm"] in ("fast", "fast_search"):
                action = _fast_action(obs, fast_state)
            else:
                action = _ppo_action(model, obs, state, deterministic=args.policy_mode == "greedy")
            policy_choice = action
            if job["arm"] in ("search", "fast_search"):
                stats["last_targets"] = None
                if args.scope == "timing":
                    action = _timing_search(env, obs, action, args, kw["seed"] * 10000 + decisions, stats, state)
                else:
                    action = _search(env, obs, action, args, kw["seed"] * 10000 + decisions, stats, state)
            obs_before = obs
            obs, _r, done, _t, info = env.step(action)
            rejected = not info.get("ok")
            if rejected:
                obs, _r, done, _t, info = env.step({"type": "wait", "ticks": 60})
            elif rows is not None:
                rows.append(_transition(obs_before, action, state, info, model.config,
                                        overridden=action is not policy_choice))
                if stats.get("last_targets"):
                    rows[-1]["search_targets"] = stats["last_targets"]
            if frames is not None:
                frames.append(_frame(obs))
                frames[-1]["action"] = action          # what was executed (after the search), like render_episode
                frames[-1]["action_rejected"] = rejected
            state["previous_action"] = action
            state["delta_ticks"] = info.get("ticks_advanced") or 0
            state["events"] = info.get("events")
            state["previous_wait_result"] = info.get("wait_result")
            decisions += 1
            if done:
                break
    finally:
        env.close()
    if frames is not None:
        # Same layout as episode_query archives, so scripts/train_value_net.py reads it directly.
        archive = Path(args.frames) / f"seed_{job['seed']}" / f"{args.frames_label}_{job['arm']}"
        archive.mkdir(parents=True, exist_ok=True)
        (archive / "frames.jsonl").write_text("".join(json.dumps(f, separators=(",", ":")) + "\n" for f in frames))
        (archive / "meta.json").write_text(json.dumps({
            "task": {"seed": job["seed"], "policy": str(args.policy), "arm": job["arm"]},
            "outcome": {"won": obs.get("result") == 1, "final_wave": obs["wave"]}}))
    if rows is not None:
        import pickle, zlib
        path = Path(args.record) / f"{job['arm']}_{job['seed']}.pkl.z"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(zlib.compress(pickle.dumps({"seed": job["seed"], "won": obs.get("result") == 1,
                                                      "rows": rows}, protocol=5), 3))
    return {"seed": job["seed"], "arm": job["arm"], "won": obs.get("result") == 1,
            "final_wave": obs["wave"], "decisions": decisions, "seconds": round(time.time() - started, 1),
            "searched": stats["searched"], "timing_searched": stats.get("timing_searched", 0), "overrides": stats["overrides"],
            "override_kinds": dict(stats["override_kinds"])}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", required=True)
    p.add_argument("--value", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seeds", type=int, default=64, help="first N validation seeds")
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument("--arms", default="baseline,search")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--horizon", type=int, default=400)
    p.add_argument("--margin", type=float, default=0.02)
    p.add_argument("--continuation", choices=("passive", "policy"), default="passive",
                   help="passive: no actions for --horizon ticks (v0); policy: pi keeps playing (v1)")
    p.add_argument("--scope", choices=("all", "placement", "timing"), default="all",
                   help="all: plant/wait/where (v0); placement: only re-place the card pi chose (v0b)")
    p.add_argument("--rollout-policy", help="fast rollout policy (scripts/train_rollout_policy.py); "
                   "used for look-ahead instead of pi, and by the 'fast' arm")
    p.add_argument("--timing-horizon", type=int, default=1500, help="ticks of look-ahead (0 = to the end of the game)")
    p.add_argument("--timing-sun", type=int, default=300)
    p.add_argument("--timing-gap", type=int, default=1500)
    p.add_argument("--timing-margin", type=float, default=0.02)
    p.add_argument("--min-gap", type=int, default=0,
                   help="ticks between searches unless pi itself wants to plant (0 = search every time)")
    p.add_argument("--max-actions", type=int, default=3000)
    p.add_argument("--record", help="write per-game behaviour-cloning rows here (for distillation)")
    p.add_argument("--frames", help="write per-game frame archives here (for value-network training)")
    p.add_argument("--frames-label", default="play")
    p.add_argument("--policy-mode", choices=("greedy", "sampled"), default="greedy")
    p.add_argument("--manifest", default=None, help="task manifest (default: 1024-seed validation)")
    p.add_argument("--task-id", default=None)
    p.add_argument("--resource-dir", default=DEFAULT_RESOURCE_DIR)
    args = p.parse_args()

    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT / "python"))
    from render_episode import load_training_task
    global MANIFEST, TASK_ID
    MANIFEST = Path(args.manifest) if args.manifest else MANIFEST
    TASK_ID = args.task_id or TASK_ID
    seeds = load_training_task(MANIFEST, TASK_ID)["seeds"][args.seed_offset: args.seed_offset + args.seeds]
    args.out.mkdir(parents=True, exist_ok=True)
    results_path = args.out / "games.jsonl"
    done_keys = set()
    if results_path.exists():
        for line in results_path.read_text().splitlines():
            r = json.loads(line); done_keys.add((r["arm"], r["seed"]))
    jobs = [{"seed": s, "arm": arm, "args": args} for arm in args.arms.split(",") for s in seeds
            if (arm, s) not in done_keys]
    (args.out / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=1))
    print(f"{len(jobs)} games on {args.workers} workers", flush=True)
    started = time.time()
    with results_path.open("a") as out, mp.get_context("spawn").Pool(
            args.workers, initializer=_init_worker, initargs=(args.policy, args.value, args.rollout_policy)) as pool:
        for i, res in enumerate(pool.imap_unordered(_play, jobs), 1):
            out.write(json.dumps(res) + "\n"); out.flush()
            if i % 10 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} ({time.time() - started:.0f}s)", flush=True)
    summarize(results_path)


def summarize(results_path: Path) -> dict:
    from math import comb
    rows = [json.loads(line) for line in results_path.read_text().splitlines()]
    by = {}
    for r in rows:
        by.setdefault(r["arm"], {})[r["seed"]] = r
    summary = {arm: {"games": len(v), "won": sum(r["won"] for r in v.values()),
                     "mean_final_wave": round(sum(r["final_wave"] for r in v.values()) / len(v), 2),
                     "mean_seconds": round(sum(r["seconds"] for r in v.values()) / len(v), 1),
                     "overrides_per_game": round(sum(r["overrides"] for r in v.values()) / len(v), 1),
                     "searched_per_game": round(sum(r["searched"] for r in v.values()) / len(v), 1)}
               for arm, v in by.items()}
    if "baseline" in by and "search" in by:
        common = sorted(set(by["baseline"]) & set(by["search"]))
        b = sum(1 for s in common if by["search"][s]["won"] and not by["baseline"][s]["won"])
        c = sum(1 for s in common if by["baseline"][s]["won"] and not by["search"][s]["won"])
        n = b + c
        p_value = min(1.0, 2 * sum(comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n) if n else 1.0
        summary["paired"] = {"seeds": len(common), "search_only_wins": b, "baseline_only_wins": c,
                             "mcnemar_exact_p": p_value}
    (results_path.parent / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    main()
