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


def _init_worker(policy_path: str, value_path: str, rollout_path: str | None = None,
                 lean_rollouts: bool = False, rollout_sim_skip: int = 0, batch_waits: bool = False) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _W.update(lean_rollouts=lean_rollouts, rollout_sim_skip=rollout_sim_skip, batch_waits=batch_waits)
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
    # Rollouts by the fast policy only read what render_episode._frame keeps, so they can ask for
    # the compact observation (identical frames, ~1/4 of the bytes). pi reads the full one.
    lean = fast_state is not None and _W.get("lean_rollouts", False)
    batch_waits = fast_state is not None and _W.get("batch_waits", False) and hasattr(env, "wait_until_legal")
    sim_skip = _W.get("rollout_sim_skip", 0)
    if lean:
        env.set_lean_observation(True)
    if sim_skip:
        env.set_sim_skip(sim_skip)
    try:
        while True:
            obs, _r, done, _t, info = env.step(action)
            if not info.get("ok"):
                action = {"type": "wait", "ticks": 60}
                obs, _r, done, _t, info = env.step(action)
            rs.update(previous_action=action, delta_ticks=info.get("ticks_advanced") or 0,
                      events=info.get("events"), previous_wait_result=info.get("wait_result"))
            if done or obs.get("terminal") or obs["tick"] >= until_tick:
                return obs
            if batch_waits and not (obs.get("legal_actions") or {}).get("plants"):
                # With no legal plant the fast policy can only wait 60 ticks, again and again: let
                # the simulator repeat that itself. Its features only read the previous frame's
                # tick and whether the previous action was a plant, so this state is exact.
                obs, _r, done, _t, info = env.wait_until_legal(until_tick)
                fast_state["frame"], fast_state["action"] = {"tick": info["previous_tick"]}, dict(NOOP)
                if done or obs.get("terminal") or obs["tick"] >= until_tick:
                    return obs
            if fast_state is not None:
                action = _fast_action(obs, fast_state)
            else:
                action = _ppo_action(_W["model"], obs, rs, deterministic=True)
    finally:
        if sim_skip:
            env.set_sim_skip(0)
        if lean:
            env.set_lean_observation(False)


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
        if args.per_card:
            # The best cells of *every* card, not the overall top-k: the static value net was trained on
            # games where some cards (cherry bomb, wall-nut) were never played, so it cannot rank them
            # against shooters right after planting -- only the rollout can show what they are worth.
            by_card: dict[int, list] = {}
            for v, a in scored:
                by_card.setdefault(a["packet"], []).append(a)
            candidates = [a for cells in by_card.values() for a in cells[: args.per_card]]
        else:
            candidates = [a for _v, a in scored[: args.top_k]]
        candidates += [] if args.scope == "placement" else [NOOP]
        policy_key = json.dumps(policy_action if policy_action["type"] == "plant" else NOOP, sort_keys=True)
        if policy_action["type"] == "plant" and all(json.dumps(c, sort_keys=True) != policy_key for c in candidates):
            candidates.append(policy_action)

        if getattr(args, "halving", False) and args.continuation == "policy":
            return _halving_choice(env, obs, root, candidates, policy_action, policy_key, args, decision_seed,
                                   stats, state)
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


def _halving_choice(env, obs: dict, root: int, candidates: list[dict], policy_action: dict, policy_key: str,
                    args, decision_seed: int, stats: dict, state: dict | None) -> dict:
    """``--halving``: successive halving over the same resampled futures as the full search.

    Round 1 rolls every candidate out once (future k = 0). Each later round keeps the better half
    (never fewer than two, and always pi's own choice, because the override rule compares against
    it) and doubles their futures (k = 1, then k = 2..3, ... up to ``--samples``). Survivors have
    always been scored on exactly the same futures, so comparing them stays a common-random-numbers
    comparison. With ``--halving-gap G`` the search also stops as soon as the leader is more than G
    ahead of every other survivor. The decision rule (override pi only by more than ``--margin``)
    is unchanged. With 8 candidates and 4 futures this is 8 + 4 + 2x2 = 16 rollouts instead of 32.
    """
    from render_episode import _frame
    keys = [json.dumps(c, sort_keys=True) for c in candidates]
    policy_index = keys.index(policy_key)
    values: list[list[float]] = [[] for _ in candidates]
    alive = list(range(len(candidates)))
    done = 0                                 # futures every survivor has been scored on
    target = 1
    gap = getattr(args, "halving_gap", 0.0) or 0.0
    while True:
        target = min(target, args.samples)
        leaves: list[tuple[int, dict]] = []
        for i in alive:
            for k in range(done, target):
                env.restore(root)
                env.reseed((decision_seed * 1000 + k) % 2 ** 32)
                leaf = _rollout_policy(env, obs, state, candidates[i], obs["tick"] + args.horizon)
                terminal = _score_terminal_or_value(leaf)
                if terminal is None:
                    leaves.append((i, _frame(leaf)))
                else:
                    values[i].append(terminal)
        if leaves:
            for (i, _f), v in zip(leaves, _values([f for _i, f in leaves])):
                values[i].append(v)
        done = target
        mean = {i: sum(values[i]) / len(values[i]) for i in alive}
        ranked = sorted(alive, key=lambda i: -mean[i])
        if done >= args.samples:
            break
        if gap and len(ranked) > 1 and mean[ranked[0]] - mean[ranked[1]] > gap:
            stats["halving_early_stops"] = stats.get("halving_early_stops", 0) + 1
            break
        keep = ranked[: max(2, (len(ranked) + 1) // 2)]
        if policy_index not in keep:
            keep.append(policy_index)
        alive = sorted(keep)
        target = done * 2
    stats["halving_rollouts"] = stats.get("halving_rollouts", 0) + sum(len(v) for v in values)
    totals = [sum(v) / len(v) if v else float("-inf") for v in values]
    stats["last_targets"] = [(c, v) for c, v in zip(candidates, totals)]
    best = max(alive, key=lambda i: totals[i])
    if totals[best] - totals[policy_index] > args.margin and best != policy_index:
        stats["overrides"] += 1
        stats["override_kinds"][candidates[best]["type"] + "_over_" + policy_action["type"]] += 1
        return candidates[best]
    return policy_action


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
    # One simulator process per worker, reset for every game: starting the process and loading
    # resources costs ~0.7 s, as much as a whole student-only game. RESET_V2 rebuilds the board
    # from the seed, so a reused process plays the same game (scripts/perf_rollout_check.py).
    env = _W.get("env")
    if env is None:
        env = _W["env"] = policy_env(model.config, args.resource_dir, headless=True)
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
        ready_mowers = lambda o: sum(1 for d in o.get("defenses") or [] if d["state"] == 1)
        mowers_start = mowers_left = ready_mowers(obs)
        planted: dict[int, int] = {}
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
            if not rejected and action["type"] == "plant":
                planted[action["packet"]] = planted.get(action["packet"], 0) + 1
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
            if not obs["terminal"]:      # a won level fires the remaining mowers; count before that
                mowers_left = ready_mowers(obs)
            if done:
                break
    except BaseException:
        _W.pop("env", None)
        env.close()
        raise
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
            "mowers_used": mowers_start - mowers_left, "planted": planted,
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
    p.add_argument("--per-card", type=int, default=0,
                   help="instead of the overall top-k, keep the best N cells of each card (scope all)")
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--halving", action="store_true",
                   help="successive halving over the candidates (1, 2, 4.. futures; needs --continuation policy)")
    p.add_argument("--halving-gap", type=float, default=0.0,
                   help="with --halving, stop once the leader is this far ahead of every other survivor")
    p.add_argument("--horizon", type=int, default=400)
    p.add_argument("--margin", type=float, default=0.02)
    p.add_argument("--continuation", choices=("passive", "policy"), default="passive",
                   help="passive: no actions for --horizon ticks (v0); policy: pi keeps playing (v1)")
    p.add_argument("--scope", choices=("all", "placement", "timing"), default="all",
                   help="all: plant/wait/where (v0); placement: only re-place the card pi chose (v0b)")
    p.add_argument("--rollout-policy", help="fast rollout policy (scripts/train_rollout_policy.py); "
                   "used for look-ahead instead of pi, and by the 'fast' arm")
    p.add_argument("--no-lean-rollouts", dest="lean_rollouts", action="store_false",
                   help="fast-policy rollouts read full observations instead of OBS_LEAN (same frames, slower)")
    p.add_argument("--no-batch-waits", dest="batch_waits", action="store_false",
                   help="fast-policy rollouts send every no-legal-plant wait separately (same result, slower)")
    p.add_argument("--rollout-sim-skip", type=int, default=0,
                   help="SIM_SKIP mask during rollouts only (0 = exact; see scripts/sim_skip_agreement.py)")
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
            args.workers, initializer=_init_worker, initargs=(args.policy, args.value, args.rollout_policy,
                                                     args.lean_rollouts, args.rollout_sim_skip,
                                                     args.batch_waits)) as pool:
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
                     "searched_per_game": round(sum(r["searched"] for r in v.values()) / len(v), 1),
                     "mowers_used_per_game": round(sum(r.get("mowers_used", 0) for r in v.values()) / len(v), 2)}
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
