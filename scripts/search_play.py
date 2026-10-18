"""The teacher of expert iteration: the student plus a value-guided search (DESIGN.md §13–§14).

At a decision where some plant is legal (packet ready, sun enough, free cell) or, with shovels on, some
cell can be dug up:
  1. Every legal plant is applied from a snapshot (``branch_snapshot``) and the board right after planting is
     scored by the static value net (``--static-value``, default ``--value``). The best ``--per-card`` cells of
     every card survive, plus "do nothing now" and the student's own choice. With ``--shovel-candidates k``
     the legal shovel actions are scored the same way and the best ``k`` join the candidates; the continuation
     then decides what the freed cell is worth (the student may replant it). With ``--student-candidates k``
     the student's own top-k plants (by its policy probabilities, ``pvz_rollout_policy.ranked_plants``) join
     the candidates too -- the static net only ever offers one cell per card, so cells the student itself
     would pick were previously reachable only by shovelling first. 0 = off for both, and off is the old
     candidate set exactly.
  2. Each candidate is played and then continued by the student for ``--horizon`` ticks in C++, ``--samples``
     times with different random futures (the same futures for every candidate). The leaf is scored by
     ``--value`` (a game that ended inside the horizon scores its result). ``--halving`` (default) rolls out
     the weaker half fewer times.
  3. The student's choice is overridden only when the best candidate beats it by ``--margin``.
     ``--verify-samples`` adds independent futures for just that pair before accepting an override.
Between searches at least ``--min-gap`` ticks pass unless the student itself wants to plant.

Arms: ``fast`` = the student alone, ``fast_search`` = the teacher. ``--frames`` writes per-decision archives
for training (with ``--dagger-beta`` < 1 the student executes part of the moves, the search still labels all).

Removed 2026-10-16 (results in DESIGN.md / learn/, code in git history): pi (the PPO policy) as the player and
its arms ``baseline`` / ``search``; passive continuation (no actions during the look-ahead); ``--scope
placement`` and ``--scope timing`` (timing search: no gain, learn/05); ``--top-k`` (replaced by ``--per-card``);
``--halving-gap`` (no extra gain); ``--record`` (distillation into pi, replaced by the student, learn/06).

Usage:
  python scripts/search_play.py --rollout-policy artifacts/student/v8/model.pt --value artifacts/value_net/v2/model.pt \
      --manifest artifacts/stage4/day_l7_hard.json --task-id day_l7_z13_eval --seeds 128 --out OUT
"""
from __future__ import annotations

import argparse
import random
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESOURCE_DIR = "/Users/newbiexvwu/Downloads/Plants_Vs_Zombies_V1.2.0.1073_EN"
NOOP = {"type": "wait", "ticks": 60, "until": "timeout"}

_W: dict = {}


def _init_worker(value_path: str, rollout_path: str, rollout_sim_skip: int = 0, cpp_rollouts: bool = True,
                 static_value_path: str | None = None) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    _W.update(lean_rollouts=True, rollout_sim_skip=rollout_sim_skip, batch_waits=True, cpp_rollouts=cpp_rollouts)
    sys.path.insert(0, str(ROOT / "python"))
    sys.path.insert(0, str(ROOT / "scripts"))
    import torch
    torch.set_num_threads(1)
    from pvz_rollout_policy import load_student
    from pvz_value_net import load_value_net
    _W["vnet"] = load_value_net(value_path)
    if static_value_path:
        _W["static_vnet"] = load_value_net(static_value_path)
    _W["fast"] = load_student(rollout_path)
    if cpp_rollouts:
        import tempfile
        from pvz_rollout_policy import export_student
        _W["rollout_tmp"] = tempfile.TemporaryDirectory(prefix="pvz-rollout-")
        _W["rollout_file"] = str(Path(_W["rollout_tmp"].name) / "student.bin")
        export_student(_W["fast"], _W["rollout_file"])


def _fast_action(obs: dict, fast_state: dict) -> dict:
    """The student's action; ``fast_state`` carries the previous frame and action."""
    from pvz_rollout_policy import act
    from render_episode import _frame
    frame = _frame(obs)
    frame["action"] = fast_state.get("action")
    previous = fast_state.get("frame")
    if _W["fast"].obs != "v1" and "history" not in fast_state:
        from pvz_student_observation import StudentHistory
        fast_state["history"] = StudentHistory(_W["fast"].obs)
    action = act(_W["fast"], frame, fast_state.get("frame"), (obs.get("legal_actions") or {}).get("plants") or [],
                 context=_W.get("context"), history=fast_state.get("history"))
    # --student-candidates needs exactly the frame / history the student just read, so keep them.
    _W["student_frame"], _W["student_previous_frame"] = frame, previous
    fast_state["frame"] = frame
    fast_state["action"] = action
    return action


def _values(frames: list[dict], static: bool = False) -> list[float]:
    import torch
    from pvz_value_net import featurize
    net = _W.get("static_vnet", _W["vnet"]) if static else _W["vnet"]
    boards, scalars = zip(*(featurize(f) for f in frames))
    scalars = torch.stack(scalars)
    if net.context:          # the game's multiplier and zombie roster, set by _play at reset
        scalars = torch.cat([scalars, torch.from_numpy(_W["context"]).expand(len(frames), -1)], dim=1)
    with torch.inference_mode():
        return torch.sigmoid(net(torch.stack(boards), scalars)).tolist()


def _score_terminal_or_value(obs: dict) -> float | None:
    if "rollout_result" in obs:
        return float(obs["rollout_result"])
    if obs.get("terminal"):
        return 1.0 if obs.get("result") == 1 else 0.0
    return None


def _rollout_policy(env, obs: dict, first: dict, until_tick: int,
                    root: int | None = None, seed: int | None = None) -> dict:
    """Play ``first`` and then let the student continue until ``until_tick`` (0 = to the end of the game).

    In C++ by default; the Python loop below is the reference the C++ path is checked against
    (scripts/perf_rollout_check.py)."""
    student_history = _W.get("student_history") if _W["fast"].obs != "v1" else None
    if _W.get("cpp_rollouts"):
        leaf = env.rollout(first, until_tick, sim_skip=_W.get("rollout_sim_skip", 0), snapshot_id=root, seed=seed,
                           student_history=student_history)
        metrics = _W.setdefault("rollout_metrics", {})
        for key in ("decisions", "simulation_seconds", "network_seconds"):
            metrics[key] = metrics.get(key, 0) + env._last_response[key]
        return leaf
    if root is not None:
        env.restore(root)
    if seed is not None:
        env.reseed(seed)
    from render_episode import _frame
    fast_state = {"frame": _frame(obs), "action": first}
    if _W["fast"].obs != "v1":
        from copy import deepcopy
        from pvz_student_observation import StudentHistory
        fast_state["history"] = deepcopy(student_history) if student_history is not None else StudentHistory(_W["fast"].obs)
        fast_state["history"].extra(fast_state["frame"])
    until_tick = until_tick if until_tick > obs["tick"] else 10 ** 9
    action = first
    # The student only reads what render_episode._frame keeps, so rollouts can ask for the compact
    # observation (identical frames, ~1/4 of the bytes).
    lean = _W.get("lean_rollouts", False)
    batch_waits = _W.get("batch_waits", False) and hasattr(env, "wait_until_legal") and _W["fast"].obs == "v1"
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
            if done or obs.get("terminal") or obs["tick"] >= until_tick:
                return obs
            if batch_waits and not (obs.get("legal_actions") or {}).get("plants"):
                # With no legal plant the student can only wait 60 ticks, again and again: let
                # the simulator repeat that itself. Its features only read the previous frame's
                # tick and whether the previous action was a plant, so this state is exact.
                obs, _r, done, _t, info = env.wait_until_legal(until_tick)
                fast_state["frame"], fast_state["action"] = {"tick": info["previous_tick"]}, dict(NOOP)
                if done or obs.get("terminal") or obs["tick"] >= until_tick:
                    return obs
            action = _fast_action(obs, fast_state)
    finally:
        if sim_skip:
            env.set_sim_skip(0)
        if lean:
            env.set_lean_observation(False)


def _static_frames(env, root: int, actions: list[dict], income) -> list[tuple[dict, dict]]:
    """Apply every action from ``root`` and return the frames the static value net scores."""
    from render_episode import _frame
    out = []
    for start in range(0, len(actions), 128):
        chunk = actions[start:start + 128]
        for action, record in zip(chunk, env.branch_snapshot(root, chunk, retain_snapshots=False)):
            if record.get("ok") and record.get("observation"):
                frame = _frame(record["observation"])
                frame["sun_income_rate"] = income
                out.append((action, frame))
    return out


def _search(env, obs: dict, policy_action: dict, args, decision_seed: int, stats: dict) -> dict:
    legal = (obs.get("legal_actions") or {}).get("plants") or []
    shovels = ([{"type": "shovel", "col": int(col), "row": int(row)}
                for col, row in (obs.get("legal_actions") or {}).get("shovels") or []]
               if getattr(args, "shovel_candidates", 0) > 0 else [])
    # A full board has no legal plant but does have legal shovels: that is exactly the position where
    # "shovel the front row and replant it behind" has to be considered, so shovels may trigger a search.
    if not legal and not shovels:
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
        static = _static_frames(env, root, plants, income)
        static_shovels = _static_frames(env, root, shovels, income)
        scored = sorted(zip(_values([f for _a, f in static], static=True), [a for a, _f in static]),
                        key=lambda item: -item[0]) if static else []
        scored_shovels = sorted(zip(_values([f for _a, f in static_shovels], static=True),
                                    [a for a, _f in static_shovels]), key=lambda item: -item[0]) \
            if static_shovels else []
        # The best cells of *every* card, not the overall top-k: the static value net was trained on
        # games where some cards (cherry bomb, wall-nut) were never played, so it cannot rank them
        # against shooters right after planting -- only the rollout can show what they are worth.
        by_card: dict[int, list] = {}
        for v, a in scored:
            by_card.setdefault(a["packet"], []).append(a)
        candidates = ([a for cells in by_card.values() for a in cells[: args.per_card]]
                      + [a for _v, a in scored_shovels[: args.shovel_candidates]] + [NOOP])
        policy_key = json.dumps(policy_action if policy_action["type"] == "plant" else NOOP, sort_keys=True)
        if policy_action["type"] == "plant" and all(json.dumps(c, sort_keys=True) != policy_key for c in candidates):
            candidates.append(policy_action)

        # --student-candidates k: the student's own top-k plants join the candidates
        # (pvz_rollout_policy.ranked_plants). The static value net picks one cell per card, so a cell
        # the student would actually pick -- or its second choice -- never gets compared; the search
        # could only reach it indirectly by shovelling and letting the student react (learn/06).
        # Deduplicated against everything already there. k = 0 (default) leaves the set untouched.
        stats["last_student_keys"] = set()
        if getattr(args, "student_candidates", 0) and _W.get("student_frame") is not None:
            from pvz_rollout_policy import ranked_plants
            present = {json.dumps(c, sort_keys=True) for c in candidates}
            added = 0
            for _score, entry in ranked_plants(_W["fast"], _W["student_frame"], _W.get("student_previous_frame"),
                                               legal, context=_W.get("context"),
                                               history=_W.get("student_history")):
                action = {"type": "plant", "packet": int(entry["packet"]),
                          "col": int(entry["col"]), "row": int(entry["row"])}
                key = json.dumps(action, sort_keys=True)
                if key in present:
                    continue
                candidates.append(action)
                present.add(key)
                stats["last_student_keys"].add(key)
                added += 1
                if added >= args.student_candidates:
                    break
            stats["student_candidate_added"] = stats.get("student_candidate_added", 0) + added

        if args.halving:
            return _halving_choice(env, obs, root, candidates, policy_action, policy_key, args, decision_seed, stats)
        totals = []
        for cand in candidates:
            leaf_frames, fixed = [], []
            for k in range(args.samples):
                leaf = _rollout_policy(env, obs, cand, obs["tick"] + args.horizon, root,
                                       (decision_seed * 1000 + k) % 2 ** 32)
                terminal = _score_terminal_or_value(leaf)
                if terminal is None:
                    leaf_frames.append(_frame(leaf))
                else:
                    fixed.append(terminal)
            values = fixed + (_values(leaf_frames) if leaf_frames else [])
            totals.append(sum(values) / len(values))
        keys = [json.dumps(c, sort_keys=True) for c in candidates]
        best = max(range(len(candidates)), key=lambda i: totals[i])
        return _verified_choice(env, obs, root, candidates, totals, [args.samples] * len(candidates),
                                best, keys.index(policy_key), policy_action, args, decision_seed, stats)
    finally:
        env.restore(root)
        env.release_snapshot(root)


def _halving_choice(env, obs: dict, root: int, candidates: list[dict], policy_action: dict, policy_key: str,
                    args, decision_seed: int, stats: dict) -> dict:
    """``--halving``: successive halving over the same resampled futures as the full search.

    Round 1 rolls every candidate out once (future k = 0). Each later round keeps the better half
    (never fewer than two, and always the student's own choice, because the override rule compares
    against it) and doubles their futures (k = 1, then k = 2..3, ... up to ``--samples``). Survivors
    have always been scored on exactly the same futures, so comparing them stays a common-random-numbers
    comparison. The decision rule (override only by more than ``--margin``) is unchanged. With 8
    candidates and 4 futures this is 8 + 4 + 2x2 = 16 rollouts instead of 32.
    """
    from render_episode import _frame
    keys = [json.dumps(c, sort_keys=True) for c in candidates]
    policy_index = keys.index(policy_key)
    values: list[list[float]] = [[] for _ in candidates]
    alive = list(range(len(candidates)))
    done = 0                                 # futures every survivor has been scored on
    target = 1
    while True:
        target = min(target, args.samples)
        leaves: list[tuple[int, dict]] = []
        for i in alive:
            for k in range(done, target):
                leaf = _rollout_policy(env, obs, candidates[i], obs["tick"] + args.horizon,
                                       root, (decision_seed * 1000 + k) % 2 ** 32)
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
        keep = ranked[: max(2, (len(ranked) + 1) // 2)]
        if policy_index not in keep:
            keep.append(policy_index)
        alive = sorted(keep)
        target = done * 2
    stats["halving_rollouts"] = stats.get("halving_rollouts", 0) + sum(len(v) for v in values)
    totals = [sum(v) / len(v) if v else float("-inf") for v in values]
    best = max(alive, key=lambda i: totals[i])
    return _verified_choice(env, obs, root, candidates, totals, [len(v) for v in values], best,
                            policy_index, policy_action, args, decision_seed, stats)


def _verified_choice(env, obs: dict, root: int, candidates: list[dict], totals: list[float],
                     counts: list[int], best: int, policy_index: int, policy_action: dict,
                     args, decision_seed: int, stats: dict, *, fresh: bool = False) -> dict:
    from render_episode import _frame
    before = fresh or (best != policy_index and totals[best] - totals[policy_index] > args.margin)
    samples = getattr(args, "verify_samples", 0) + (args.samples if fresh else 0)
    if fresh:
        counts = [0] * len(candidates)
    if samples and before:
        prior = None if fresh else [totals[policy_index], totals[best]]
        started = time.perf_counter()
        for index in (policy_index, best):
            fixed, leaves = [], []
            for future in range(args.samples, args.samples + samples):
                leaf = _rollout_policy(env, obs, candidates[index], obs["tick"] + args.horizon,
                                       root, (decision_seed * 1000 + future) % 2 ** 32)
                terminal = _score_terminal_or_value(leaf)
                if terminal is None:
                    leaves.append(_frame(leaf))
                else:
                    fixed.append(terminal)
            added = fixed + (_values(leaves) if leaves else [])
            totals[index] = (totals[index] * counts[index] + sum(added)) / (counts[index] + samples)
        stats["last_verification"] = {
            "samples": samples, "student": policy_action, "challenger": candidates[best],
            "before_values": prior, "after_values": [totals[policy_index], totals[best]],
            "before_override": True, "after_override": totals[best] - totals[policy_index] > args.margin,
            "original_samples": [counts[policy_index], counts[best]], "decision_seed": decision_seed,
            "mode": "fresh" if fresh else "reproduced",
            "seconds": time.perf_counter() - started}
    # Did the winner come from --student-candidates? Counted here so both the halving and the
    # full search report it; the override counter below is the subset that changed the action.
    if json.dumps(candidates[best], sort_keys=True) in (stats.get("last_student_keys") or ()):
        stats["student_candidate_best"] = stats.get("student_candidate_best", 0) + 1
        if best != policy_index and totals[best] - totals[policy_index] > args.margin:
            stats["student_candidate_override"] = stats.get("student_candidate_override", 0) + 1
    stats["last_targets"] = list(zip(candidates, totals))
    stats["last_best"] = candidates[best]
    if totals[best] - totals[policy_index] > args.margin and best != policy_index:
        stats["overrides"] += 1
        stats["override_kinds"][candidates[best]["type"] + "_over_" + policy_action["type"]] += 1
        return candidates[best]
    return policy_action


def _play(job: dict) -> dict:
    from collections import Counter
    import torch
    from pvz_event_env import EventWaitEnv
    from render_episode import _frame, load_training_task, training_task_kwargs
    from scripted_baseline import profile_for_deck
    from pvz_env import TaskSpec
    args = job["args"]
    kw = training_task_kwargs(load_training_task(Path(args.manifest), args.task_id), job["seed"])
    # One simulator process per worker, reset for every game: starting the process and loading
    # resources costs ~0.7 s, as much as a whole student-only game. RESET_V2 rebuilds the board
    # from the seed, so a reused process plays the same game (scripts/perf_rollout_check.py).
    env = _W.get("env")
    if env is None:
        env = _W["env"] = EventWaitEnv(args.resource_dir, headless=True)
        if _W.get("cpp_rollouts"):
            env.load_rollout_net(_W["rollout_file"])
    task = TaskSpec(level=kw["level"], seed=kw["seed"], playthrough=2,
                    profile=profile_for_deck(tuple(kw["deck"]), kw.get("first_aid_owned", False)),
                    **{k: v for k, v in kw["task_extra"].items() if v is not None})
    torch.manual_seed(kw["seed"])
    started = time.time()
    stats = {"searched": 0, "overrides": 0, "shovels": 0, "override_kinds": Counter(),
             "student_candidate_added": 0, "student_candidate_best": 0, "student_candidate_override": 0}
    try:
        obs, _ = env.reset(deck=tuple(kw["deck"]), task=task)
        from pvz_value_net import observation_context
        _W["context"] = observation_context(obs)
        task_context = {"level": kw["level"], "multiplier": obs.get("zombie_count_multiplier", 1.0),
                        "roster": (obs.get("loadout_context") or {}).get("zombie_roster") or []}
        decisions = 0
        frames = [_frame(obs)] if args.frames else None
        fast_state = {}
        _W["student_history"] = None
        # DAgger (DESIGN.md §14.6): the search still labels every decision, but with probability
        # 1 - dagger_beta the student's own action is executed, so the archive covers the positions the
        # student itself gets into. The label is the anchored search decision (overrides only past --margin).
        dagger = random.Random(kw["seed"]) if args.dagger_beta < 1.0 else None
        ready_mowers = lambda o: sum(1 for d in o.get("defenses") or [] if d["state"] == 1)
        mowers_start = mowers_left = ready_mowers(obs)
        planted: dict[int, int] = {}
        # 整局 C++ 接口只返回末状态，逐决策档案需要走 Python 对局循环。
        if job["arm"] == "fast" and _W.get("cpp_rollouts") and not args.frames:
            obs = env.rollout(None, 0, max_actions=args.max_actions)
            details = env._last_response
            decisions, mowers_left = details["decisions"], details["mowers_left"]
            planted = {index: count for index, count in enumerate(details["planted"]) if count}
        while not obs["terminal"] and decisions < args.max_actions:
            action = policy_choice = _fast_action(obs, fast_state)
            _W["student_history"] = fast_state.get("history")
            if job["arm"] == "fast_search":
                stats["last_targets"] = None
                stats.pop("last_verification", None)
                action = _search(env, obs, action, args, kw["seed"] * 10000 + decisions, stats)
                if action["type"] == "shovel":
                    stats["shovels"] += 1
                    cell = next((p for p in obs.get("plants") or []
                                 if p["row"] == action["row"] and p["col"] == action["col"]), None)
                    stats.setdefault("shovel_events", []).append({
                        "tick": obs["tick"], "row": action["row"], "col": action["col"],
                        "plant": cell.get("type") if cell else None,
                        "plant_health": cell.get("health") if cell else None,
                        "student": policy_choice["type"], "decision": decisions})
            label = action
            if dagger is not None and dagger.random() >= args.dagger_beta:
                action = policy_choice
            fast_state["action"] = action    # the student's next features read the executed action
            obs, _r, done, _t, info = env.step(action)
            rejected = not info.get("ok")
            if not rejected and action["type"] == "plant":
                planted[action["packet"]] = planted.get(action["packet"], 0) + 1
            if rejected:
                obs, _r, done, _t, info = env.step({"type": "wait", "ticks": 60})
            if frames is not None:
                frames.append(_frame(obs))
                if stats.get("last_targets"):
                    # the search's estimate for every candidate it compared: [packet (-1 = wait), row, col, value];
                    # a soft target for the student (train_rollout_policy --soft-weight). Before "action" and
                    # "label", because the trainer finds the target by scanning for the last of those.
                    # Shovel candidates are left out: the student has no shovel output yet, so a shovel
                    # row would be read as a wait target by the trainer (phase 2 adds the head).
                    frames[-1]["targets"] = [[c.get("packet", -1), c.get("row", -1), c.get("col", -1), round(v, 4)]
                                             for c, v in stats["last_targets"]
                                             if abs(v) != float("inf") and c["type"] != "shovel"]
                frames[-1]["action"] = action          # what was executed (after the search), like render_episode
                frames[-1]["action_rejected"] = rejected
                if stats.get("last_verification"):
                    frames[-1]["verification"] = stats["last_verification"]
                if dagger is not None:
                    frames[-1]["label"] = label        # last field: train_rollout_policy scans for it
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
        from pvz_frame_store import write_frames
        write_frames(archive / f"frames.{getattr(args, 'frames_format', 'npz')}", frames)
        (archive / "meta.json").write_text(json.dumps({
            "task": {"seed": job["seed"], "policy": str(args.rollout_policy), "arm": job["arm"]},
            "context": task_context,
            "outcome": {"won": obs.get("result") == 1, "final_wave": obs["wave"]}}))
    return {"seed": job["seed"], "arm": job["arm"], "won": obs.get("result") == 1,
            "final_wave": obs["wave"], "decisions": decisions, "seconds": round(time.time() - started, 1),
            "mowers_used": mowers_start - mowers_left, "planted": planted,
            "searched": stats["searched"], "overrides": stats["overrides"], "shovels": stats["shovels"],
            "student_candidate_added": stats["student_candidate_added"],
            "student_candidate_best": stats["student_candidate_best"],
            "student_candidate_override": stats["student_candidate_override"],
            "shovel_events": stats.get("shovel_events", []),
            "override_kinds": dict(stats["override_kinds"])}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rollout-policy", required=True, help="the student (scripts/train_rollout_policy.py)")
    p.add_argument("--value", required=True, help="value net that scores the leaves")
    p.add_argument("--static-value", help="value net that picks the candidate cells right after planting "
                   "(default --value)")
    p.add_argument("--manifest", required=True)
    p.add_argument("--task-id", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seeds", type=int, default=64)
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument("--arms", default="fast_search", help="fast (the student alone), fast_search (the teacher)")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--per-card", type=int, default=1, help="best cells of each card kept as candidates")
    p.add_argument("--student-candidates", type=int, default=0,
                   help="the student's own top-k plant actions (its policy probabilities) added to the "
                        "candidates; 0 = off, which is the old candidate set exactly")
    p.add_argument("--shovel-candidates", type=int, default=0,
                   help="best legal shovel cells (scored by --static-value right after shovelling) added to "
                        "the candidates; 0 = off, which is the old candidate set exactly")
    p.add_argument("--samples", type=int, default=3)
    p.add_argument("--halving", action=argparse.BooleanOptionalAction, default=True,
                   help="successive halving over the candidates (1, 2, 4.. futures); --no-halving = every "
                        "candidate gets all --samples")
    p.add_argument("--horizon", type=int, default=2500)
    p.add_argument("--margin", type=float, default=0.02)
    p.add_argument("--verify-samples", type=int, default=0,
                   help="extra independent futures for student and challenger before an override (0 = disabled)")
    p.add_argument("--min-gap", type=int, default=300,
                   help="ticks between searches unless the student itself wants to plant (0 = search every time)")
    p.add_argument("--rollout-sim-skip", type=int, default=61,
                   help="SIM_SKIP mask during rollouts only (0 = exact; see scripts/sim_skip_agreement.py)")
    p.add_argument("--cpp-rollouts", action=argparse.BooleanOptionalAction, default=True,
                   help="run student continuations in C++ (--no-cpp-rollouts: the Python reference loop)")
    p.add_argument("--max-actions", type=int, default=3000)
    p.add_argument("--frames", help="write per-game frame archives here (for training)")
    p.add_argument("--frames-label", default="play")
    p.add_argument("--frames-format", choices=["npz", "jsonl"], default="npz",
                   help="lossless archive with cached training features (jsonl: legacy text)")
    p.add_argument("--dagger-beta", type=float, default=1.0,
                   help="probability of executing the search's decision (1 = the teacher plays; < 1 = DAgger: "
                        "the student plays the rest, every frame still records the search's label)")
    p.add_argument("--resource-dir", default=DEFAULT_RESOURCE_DIR)
    args = p.parse_args()
    if args.verify_samples < 0:
        p.error("--verify-samples must be nonnegative")
    if args.shovel_candidates < 0:
        p.error("--shovel-candidates must be nonnegative")
    if args.student_candidates < 0:
        p.error("--student-candidates must be nonnegative")

    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT / "python"))
    from render_episode import load_training_task
    seeds = load_training_task(Path(args.manifest), args.task_id)["seeds"][args.seed_offset: args.seed_offset + args.seeds]
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
            args.workers, initializer=_init_worker,
            initargs=(args.value, args.rollout_policy, args.rollout_sim_skip, args.cpp_rollouts,
                      args.static_value)) as pool:
        for i, res in enumerate(pool.imap_unordered(_play, jobs), 1):
            out.write(json.dumps(res) + "\n"); out.flush()
            if i % 10 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} ({time.time() - started:.0f}s)", flush=True)
    summarize(results_path)


def summarize(results_path: Path) -> dict:
    rows = [json.loads(line) for line in results_path.read_text().splitlines()]
    by = {}
    for r in rows:
        by.setdefault(r["arm"], {})[r["seed"]] = r
    summary = {arm: {"games": len(v), "won": sum(r["won"] for r in v.values()),
                     "mean_final_wave": round(sum(r["final_wave"] for r in v.values()) / len(v), 2),
                     "mean_seconds": round(sum(r["seconds"] for r in v.values()) / len(v), 1),
                     "overrides_per_game": round(sum(r["overrides"] for r in v.values()) / len(v), 1),
                     "searched_per_game": round(sum(r["searched"] for r in v.values()) / len(v), 1),
                     "shovels_per_game": round(sum(r.get("shovels", 0) for r in v.values()) / len(v), 2),
                     "games_with_shovel": sum(1 for r in v.values() if r.get("shovels")),
                     "mowers_used_per_game": round(sum(r.get("mowers_used", 0) for r in v.values()) / len(v), 2)}
               for arm, v in by.items()}
    (results_path.parent / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    main()
