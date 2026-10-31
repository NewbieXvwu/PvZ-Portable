"""The teacher of expert iteration: the student plus a value-guided search (DESIGN.md §13–§14).

At a decision where some plant is legal (packet ready, sun enough, free cell):
  1. Every legal plant is applied from a snapshot (``branch_snapshot``) and the board right after planting is
     scored by the static value net (``--static-value``, default ``--value``). The best ``--per-card`` cells of
     every card survive, plus "do nothing now" and the student's own choice. With ``--student-candidates k``
     the student's own top-k plants (by its policy probabilities, ``pvz_rollout_policy.ranked_plants``) join
     the candidates too -- the static net only ever offers one cell per card, so cells the student itself
     would pick were previously reachable only by shovelling first. 0 = off, and off is the old candidate
     set exactly.
  2. Each candidate is played and then continued by the student for ``--horizon`` ticks in C++, ``--samples``
     times with different random futures (the same futures for every candidate). The leaf is scored by
     ``--value`` (a game that ended inside the horizon scores its result). ``--halving`` (default) rolls out
     the weaker half fewer times.
  3. The student's choice is overridden only when the best candidate beats it by ``MARGIN``.
     ``--verify-samples`` adds independent futures for just that pair before accepting an override.
Between searches at least ``--min-gap`` ticks pass unless the student itself wants to plant.

Arms: ``fast`` = the student alone, ``fast_search`` = the teacher. ``--frames`` writes per-decision archives
for training (with ``--dagger-beta`` < 1 the student executes part of the moves, the search still labels all).

The comparison harness ``scripts/b3_noninferiority.py`` can set a private ``_W["candidate_mode"] =
"student-topk"`` (task42 B3 arm: the student's global top-k plants, wait, and its own action). There is no
CLI flag, and the formal collection path never sets it.

Continuations run in the original C++ simulator. ``--rollout-sim-skip`` is a CPU presentation-work
bitmask: mask 29 skips colour, height, attachment matrices and messages, while preserving game ticks
and animation lifecycle updates.

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
import math
import random
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from growable_pool import GIB, GrowablePool  # noqa: E402  (task138: growable worker pool)
DEFAULT_RESOURCE_DIR = os.environ.get("PVZ_RESOURCE_DIR",
                                      str(Path.home() / ".cache" / "pvz-research-resources"))
NOOP = {"type": "wait", "ticks": 60, "until": "timeout"}
# The search overrides the student only when the best candidate beats it by more than this. It was
# --margin, set to 0.02 in all 201 run configurations (task33 audit); a constant now, not a flag.
MARGIN = 0.02
# Action budget of one game: --max-actions was 3000 in every run configuration that recorded one.
MAX_ACTIONS = 3000

_W: dict = {}

ROOF_BACKGROUNDS = frozenset({"BACKGROUND_5_ROOF", "BACKGROUND_6_BOSS"})
PRIOR_BACKGROUNDS = frozenset({"BACKGROUND_4_FOG", "BACKGROUND_5_ROOF", "BACKGROUND_6_BOSS"})


def _leaf_value(value_path: str, roof_value_path: str | None, task: dict) -> str:
    """Roof levels (and the boss) score leaves with ``--value-roof``; every other terrain keeps ``--value``."""
    if not roof_value_path:
        return value_path
    from pvz_constants import background_for_level
    roof = background_for_level(int(task["level"])) in ROOF_BACKGROUNDS
    return roof_value_path if roof else value_path


def _terrain_policy(student_path: str, gpt_path: str | None, task: dict) -> str:
    """Fog, roof and boss levels act and continue with the GPT prior; other terrain keeps the student."""
    if not gpt_path:
        return student_path
    from pvz_constants import background_for_level
    return gpt_path if background_for_level(int(task["level"])) in PRIOR_BACKGROUNDS else student_path


def _export_values(value_path: str, static_value_path: str | None):
    """Export once in the driver; workers keep paths instead of torch value networks."""
    import tempfile
    from export_value_net import export_value_net
    from pvz_value_net import load_value_net
    keep = tempfile.TemporaryDirectory(prefix="pvz-values-")
    paths = []
    for index, path in enumerate((value_path, static_value_path or value_path)):
        output = str(Path(keep.name) / f"value{index}.bin")
        export_value_net(load_value_net(path), output)
        paths.append(output)
    return keep, paths

def _close_worker_env():
    env = _W.pop("env", None)
    if env is not None:
        env.close()


def _load_nets(value_path: str, rollout_path: str, static_value_path: str | None = None) -> None:
    """Load the nets and the student's C++ export into the worker state dict.

    Either every worker calls this, or -- where the platform can fork -- the parent calls it
    once before the pool starts and the children inherit the pages copy-on-write: a run of
    sixteen workers then keeps one copy of torch and of the weights instead of sixteen (task18).
    Nothing ever writes to a weight, so the shared pages stay shared.
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    import torch
    torch.set_num_threads(1)
    from pvz_rollout_policy import load_student
    _W["value_files"] = (value_path, static_value_path or value_path)
    _W["fast"] = load_student(rollout_path)
    import tempfile
    from pvz_rollout_policy import export_student
    _W["rollout_tmp"] = tempfile.TemporaryDirectory(prefix="pvz-rollout-")
    _W["rollout_file"] = str(Path(_W["rollout_tmp"].name) / "student.bin")
    export_student(_W["fast"], _W["rollout_file"])


def _release_freed_memory() -> None:
    """Hand the C allocator's free pages back to the OS after a game.

    glibc keeps freed blocks on its arenas, so a worker's RSS stays at the high-water mark of
    the biggest game it ever played even though almost none of it is live (task18).  Linux only:
    the other platforms either have no such call or return nothing for a Python heap.
    """
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def _freeze_startup() -> None:
    """Take the imported modules and the weights out of the garbage collector's generations.

    Nothing allocated at startup is cyclic garbage, and a forked worker that leaves them
    tracked pays twice: every collection walks the whole inherited heap, and each header it
    touches is a copy-on-write page it now owns privately.  Freezing them keeps the shared
    pages shared and makes the collections that do run cheaper (task18).
    """
    import gc
    gc.collect()
    gc.freeze()


def _init_worker(value_path: str, rollout_path: str, rollout_sim_skip: int = 0,
                 static_value_path: str | None = None, metrics_counters=None) -> None:
    _W["metrics_counters"] = metrics_counters
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    _W.update(lean_rollouts=True, rollout_sim_skip=rollout_sim_skip, batch_waits=True)
    sys.path.insert(0, str(ROOT / "python"))
    sys.path.insert(0, str(ROOT / "scripts"))
    from pvz_common import OOM_SCORE_WORKER, set_oom_score_adj
    # Leave the WSL entry point its memory: under pressure the kernel should pick this game worker first.
    set_oom_score_adj(OOM_SCORE_WORKER)
    from multiprocessing.util import Finalize
    Finalize(None, _close_worker_env, exitpriority=10)
    if "fast" not in _W:        # a preloading parent already loaded them, and we inherited the pages
        _load_nets(value_path, rollout_path, static_value_path)
    import torch
    # Forking does not carry the parent's intra-op thread count into the child; a worker is
    # single threaded either way.
    torch.set_num_threads(1)
    _freeze_startup()


def _fast_action(obs: dict, fast_state: dict, native_env=None) -> dict:
    """The student's action; ``fast_state`` carries the previous frame and action."""
    from pvz_rollout_policy import act
    from render_episode import _frame
    frame = _frame(obs)
    frame["action"] = fast_state.get("action")
    previous = fast_state.get("frame")
    if _W["fast"].obs != "v1" and "history" not in fast_state:
        from pvz_student_observation import StudentHistory
        fast_state["history"] = StudentHistory(_W["fast"].obs)
    if native_env is not None:
        if fast_state.get('history') is not None:
            fast_state['history'].extra(frame)
        if (obs.get('legal_actions') or {}).get('plants'):
            action = native_env.rollout_predict(previous.get('tick') if previous else None,
                (fast_state.get('action') or {}).get('type') == 'plant',
                student_history=fast_state.get('history'))['action']
        else:
            action = dict(NOOP)
    else:
        action = act(_W["fast"], frame, previous, (obs.get("legal_actions") or {}).get("plants") or [],
                     context=_W.get("context"), history=fast_state.get("history"))
    # --student-candidates needs exactly the frame / history the student just read, so keep them.
    _W["student_frame"], _W["student_previous_frame"] = frame, previous
    fast_state["frame"] = frame
    fast_state["action"] = action
    return action


def _score_terminal_or_value(obs: dict) -> float | None:
    if "rollout_result" in obs:
        return float(obs["rollout_result"])
    if obs.get("terminal"):
        return 1.0 if obs.get("result") == 1 else 0.0
    return None


def _rollout_policy(env, obs: dict, first: dict, until_tick: int,
                    root: int | None = None, seed: int | None = None) -> dict:
    """Play ``first`` and then let the student continue until ``until_tick`` (0 = to the end of the game) in C++."""
    student_history = _W.get("student_history") if _W["fast"].obs != "v1" else None
    leaf = env.rollout(first, until_tick, sim_skip=_W.get("rollout_sim_skip", 0), snapshot_id=root, seed=seed,
                       student_history=student_history, score_value=True)
    metrics = _W.setdefault("rollout_metrics", {})
    for key in ("decisions", "simulation_seconds", "network_seconds"):
        metrics[key] = metrics.get(key, 0) + env._last_response[key]
    return leaf


# LEGACY(python-rollout-reference): the Python continuation loop, kept as the reference the C++ path is
# checked against. Test and diagnostic only -- python/test_cpp_rollout.py and scripts/perf_rollout_check.py
# call it; nothing on the collection path does.


def _continuations(env, obs: dict, root: int, jobs: list[tuple[dict, int]], args,
                   stats: dict) -> list:
    """Play each candidate and continue with the student for ``--horizon`` ticks; one handle per job."""
    stats["rollout_batches"] = stats.get("rollout_batches", 0) + 1
    stats["rollout_lanes"] = stats.get("rollout_lanes", 0) + len(jobs)
    return [_rollout_policy(env, obs, candidate, obs["tick"] + args.horizon, root, seed % 2 ** 32)
            for candidate, seed in jobs]


def _leaf_values(handles: list) -> list[float]:
    """Terminal results or engine value scores, in continuation order."""
    return [handle["value"] if (terminal := _score_terminal_or_value(handle)) is None else terminal
            for handle in handles]


def _static_scores(env, root: int, actions: list[dict], income) -> list[tuple[float, dict]]:
    """Score candidates in engine order; rejected branches do not enter the candidate set."""
    return [(score, action) for score, action in zip(env.static_scores(root, actions, income or 0.0), actions)
            if score is not None]

def _student_topk_candidates(legal: list[dict], k: int) -> tuple[list[dict], set[str]]:
    """The student's global top-k legal plants, best first (``ranked_plants``).

    Comparison arm only (scripts/b3_noninferiority.py, task42). The formal path never sets
    ``_W["candidate_mode"]``, so this function is not reached by collection runs.
    """
    from pvz_rollout_policy import ranked_plants
    if _W.get("student_frame") is None:
        raise ValueError("student-topk candidates require the current student frame")
    ranked = ranked_plants(_W["fast"], _W["student_frame"], _W.get("student_previous_frame"),
                           legal, context=_W.get("context"), history=_W.get("student_history"))
    candidates, keys = [], set()
    for _score, entry in ranked[:k]:
        action = {"type": "plant", "packet": int(entry["packet"]),
                  "col": int(entry["col"]), "row": int(entry["row"])}
        key = json.dumps(action, sort_keys=True)
        if key not in keys:
            candidates.append(action)
            keys.add(key)
    return candidates, keys


_LEAF_PROBE_COUNT = 0


def _probe_decision(args, obs: dict, candidates: list[dict], totals: list[float], chosen: dict,
                    decision_seed: int, policy_action: dict, best: int, policy_index: int) -> None:
    """Opt-in diagnostic: one JSON line per search decision with candidate-leaf dispersion.

    Off unless --leaf-probe is given, so default behaviour and every recorded result stay
    unchanged. Writes per worker process (<path>.<pid>.jsonl) to avoid cross-process locking.
    Pruned halving candidates carry -inf and are excluded; candidates vs scored keeps that
    visible instead of silently shrinking the set.
    """
    global _LEAF_PROBE_COUNT
    path = getattr(args, "leaf_probe", None)
    if not path:
        return
    limit = getattr(args, "leaf_probe_limit", 0)
    if limit and _LEAF_PROBE_COUNT >= limit:
        return
    import math
    finite = sorted((float(v) for v in totals if math.isfinite(v)), reverse=True)
    if len(finite) < 2:
        return
    mean = sum(finite) / len(finite)
    std = (sum((v - mean) ** 2 for v in finite) / len(finite)) ** 0.5
    row = dict(task=args.task_id, seed=decision_seed // 10000, decision=decision_seed % 10000,
               tick=int(obs.get("tick", -1)), candidates=len(candidates), scored=len(finite),
               mean=round(mean, 6), std=round(std, 6), top1_top2=round(finite[0] - finite[1], 6),
               best=best, policy_index=policy_index, overridden=chosen is not policy_action,
               chosen=json.dumps(chosen, sort_keys=True), policy=json.dumps(policy_action, sort_keys=True))
    with open(f"{path}.{os.getpid()}.jsonl", "a") as stream:
        stream.write(json.dumps(row) + "\n")
    _LEAF_PROBE_COUNT += 1


def _search(env, obs: dict, policy_action: dict, args, decision_seed: int, stats: dict) -> dict:
    legal = (obs.get("legal_actions") or {}).get("plants") or []
    if not legal:
        return policy_action
    if (args.min_gap and policy_action["type"] != "plant"
            and obs["tick"] - stats.get("last_search_tick", -10 ** 9) < args.min_gap):
        return policy_action
    stats["last_search_tick"] = obs["tick"]
    stats["searched"] += 1
    root = env.snapshot_fast()
    income = obs.get("sun_income_rate")
    try:
        plants = [{"type": "plant", "packet": a["packet"], "col": a["col"], "row": a["row"]} for a in legal]
        policy_key = json.dumps(policy_action if policy_action["type"] == "plant" else NOOP, sort_keys=True)
        mode = _W.get("candidate_mode", "static")
        if mode == "student-topk":
            # task42 B3 arm: global top-k plants by the student's own probabilities, plus wait and
            # the student's own action. No static net, no per-card floor.
            candidates, student_keys = _student_topk_candidates(legal, int(_W.get("student_topk", 3)))
            candidates.append(NOOP)
            if (policy_action["type"] == "plant"
                    and all(json.dumps(c, sort_keys=True) != policy_key for c in candidates)):
                candidates.append(policy_action)
            stats["last_student_keys"] = student_keys
        elif mode == "static":
            scored = sorted(_static_scores(env, root, plants, income), key=lambda item: -item[0])
            # The best cells of *every* card, not the overall top-k: the static value net was trained on
            # games where some cards (cherry bomb, wall-nut) were never played, so it cannot rank them
            # against shooters right after planting -- only the rollout can show what they are worth.
            by_card: dict[int, list] = {}
            for v, a in scored:
                by_card.setdefault(a["packet"], []).append(a)
            candidates = ([a for cells in by_card.values() for a in cells[: args.per_card]] + [NOOP])
            if (policy_action["type"] == "plant"
                    and all(json.dumps(c, sort_keys=True) != policy_key for c in candidates)):
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
                student_ranked = ranked_plants(_W["fast"], _W["student_frame"], _W.get("student_previous_frame"),
                                              legal, context=_W.get("context"), history=_W.get("student_history"))
                for _score, entry in student_ranked:
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
        else:
            raise ValueError(f"unknown candidate mode {mode!r}")

        present = {json.dumps(candidate, sort_keys=True) for candidate in candidates}
        legal_keys = {json.dumps(plant, sort_keys=True) for plant in plants}
        for tick, action in _W.get('demo_actions', ()):
            key = json.dumps(action, sort_keys=True)
            if tick >= obs['tick'] and key in legal_keys:
                if key not in present:
                    candidates.append(action)
                    stats['demo_candidate_added'] = stats.get('demo_candidate_added', 0) + 1
                break

        if args.halving:
            return _halving_choice(env, obs, root, candidates, policy_action, policy_key, args, decision_seed, stats)
        # One candidate at a time, terminal futures first: the order the labels were produced in.
        totals = []
        for cand in candidates:
            fixed, frames = [], []
            for k in range(args.samples):
                leaf = _rollout_policy(env, obs, cand, obs["tick"] + args.horizon,
                                       root, (decision_seed * 1000 + k) % 2 ** 32)
                terminal = _score_terminal_or_value(leaf)
                if terminal is None:
                    frames.append(leaf)
                else:
                    fixed.append(terminal)
            scored_values = fixed + _leaf_values(frames)
            totals.append(sum(scored_values) / len(scored_values))
        best = max(range(len(candidates)), key=lambda i: totals[i])
        keys = [json.dumps(c, sort_keys=True) for c in candidates]
        return _verified_choice(env, obs, root, candidates, totals, [args.samples] * len(candidates),
                                best, keys.index(policy_key), policy_action, args, decision_seed, stats)
    finally:
        env.restore_fast(root)
        env.release_snapshot(root)


def _halving_choice(env, obs: dict, root: int, candidates: list[dict], policy_action: dict, policy_key: str,
                    args, decision_seed: int, stats: dict) -> dict:
    """``--halving``: successive halving over the same resampled futures as the full search.

    Round 1 rolls every candidate out once (future k = 0). Each later round keeps the better half
    (never fewer than two, and always the student's own choice, because the override rule compares
    against it) and doubles their futures (k = 1, then k = 2..3, ... up to ``--samples``). Survivors
    have always been scored on exactly the same futures, so comparing them stays a common-random-numbers
    comparison. The decision rule (override only by more than ``MARGIN``) is unchanged. With 8
    candidates and 4 futures this is 8 + 4 + 2x2 = 16 rollouts instead of 32.
    """
    keys = [json.dumps(c, sort_keys=True) for c in candidates]
    policy_index = keys.index(policy_key)
    values: list[list[float]] = [[] for _ in candidates]
    alive = list(range(len(candidates)))
    stats["halving_candidates"] = stats.get("halving_candidates", 0) + len(candidates)
    selected_lanes = 0
    done = 0                                 # futures every survivor has been scored on
    target = 1
    while True:
        target = min(target, args.samples)
        order = [(i, k) for i in alive for k in range(done, target)]
        selected_lanes += len(order)
        jobs = [(candidates[i], decision_seed * 1000 + k) for i, k in order]
        handles = _continuations(env, obs, root, jobs, args, stats)
        scored = list(zip(order, handles, _leaf_values(handles)))
        # Terminal futures first, as the original CPU accumulation order had them.
        scored.sort(key=lambda row: _score_terminal_or_value(row[1]) is None)
        for (i, _k), _handle, value in scored:
            if value is not None:
                values[i].append(value)
        done = target
        mean = {i: sum(values[i]) / max(1, len(values[i])) for i in alive}
        ranked = sorted(alive, key=lambda i: -mean[i])
        if done >= args.samples:
            break
        keep = ranked[: max(2, (len(ranked) + 1) // 2)]
        if policy_index not in keep:
            keep.append(policy_index)
        alive = sorted(keep)
        target = done * 2
    stats["halving_rollouts"] = stats.get("halving_rollouts", 0) + sum(len(v) for v in values)
    stats["halving_selected_lanes"] = stats.get("halving_selected_lanes", 0) + selected_lanes
    totals = [sum(v) / len(v) if v else float("-inf") for v in values]
    best = max(alive, key=lambda i: totals[i])
    return _verified_choice(env, obs, root, candidates, totals, [len(v) for v in values], best,
                            policy_index, policy_action, args, decision_seed, stats)


def _save_value(env, obs: dict, action: dict, snapshot: int, seed: int, until_tick: int) -> float:
    leaf = _rollout_policy(env, obs, action, until_tick, snapshot, seed)
    terminal = _score_terminal_or_value(leaf)
    if terminal is not None:
        return terminal
    return _leaf_values([leaf])[0]


def _save_probe(env, obs: dict, root: int, packet: int, cost: int, args, decision_seed: int,
                until_tick: int) -> dict | None:
    """Wait in the engine until `packet` is affordable, then score planting it there.

    The engine only lists plant actions the player can pay for, so a card above the sun the
    policy ever holds is invisible to the candidate set (task120: a 125 card had a legal cell in
    0.5% of decisions, a 300 card in none). This probe answers "what would it be worth if we
    saved for it" without touching the candidate set: restore the decision root, wait, and roll
    the plant out from the future state. The caller restores the root again afterwards.
    """
    waited = 0
    current = obs
    env.restore_fast(root)
    while current['sun'] < cost and waited < args.save_max_wait and not current['terminal']:
        step = min(args.save_chunk, args.save_max_wait - waited)
        current, _reward, _done, _truncated, info = env.step({'type': 'wait', 'ticks': step})
        advanced = int(info.get('ticks_advanced') or 0)
        if advanced <= 0:
            break
        waited += advanced
    if current['terminal'] or current['sun'] < cost:
        return None
    actions = [{'type': 'plant', 'packet': entry['packet'], 'col': entry['col'], 'row': entry['row']}
               for entry in (current.get('legal_actions') or {}).get('plants') or []
               if entry['packet'] == packet]
    if not actions:
        return None
    future = env.snapshot_fast()
    try:
        scores = env.static_scores(future, actions, current.get('sun_income_rate') or 0.0)
        pairs = [(score, action) for score, action in zip(scores, actions) if score is not None]
        if not pairs:
            return None
        action = max(pairs, key=lambda item: item[0])[1]
        # The normal candidates are decided after --samples + --verify-samples futures; a plan with
        # fewer would win against them on noise alone.
        samples = max(1, args.samples + max(0, args.verify_samples))
        values = [_save_value(env, current, action, future,
                              (decision_seed * 1000 + k) % 2 ** 32, until_tick) for k in range(samples)]
    finally:
        env.release_snapshot(future)
    return dict(packet=packet, cost=cost, ticks=waited, cell=[action['col'], action['row']],
                value=sum(values) / len(values), sun=int(current['sun']), samples=samples)


def _save_plan(env, obs: dict, root: int, candidates: list[dict], totals: list[float], args,
               decision_seed: int, stats: dict) -> dict | None:
    """The best "save up and plant it" plan, when it beats every current candidate by MARGIN."""
    if not getattr(args, 'save_candidates', 0):
        return None
    present = {candidate.get('packet') for candidate in candidates if candidate['type'] == 'plant'}
    # obs['packets'] is in deck order: the slot index is the action's packet field, while
    # entry['type'] is the SeedType (a different numbering).
    costs = {slot: entry.get('cost') or 0 for slot, entry in enumerate(obs.get('packets') or [])
             if entry.get('type', -1) >= 0}
    last = stats.setdefault('save_last_probe', {})
    reachable = [(packet, cost) for packet, cost in costs.items()
                 if packet not in present and cost > obs['sun']
                 and obs['sun'] >= getattr(args, 'save_share', 0.5) * cost
                 and obs['tick'] - last.get(packet, -10 ** 9) >= getattr(args, 'save_interval', 600)]
    if not reachable:
        return None
    reachable.sort(key=lambda item: item[1])
    # The probe advances the engine and restores it; keep it out of the export operation stream so
    # a recording run cannot replay it as if the teacher had waited.
    episode = env.episode
    env.episode = None
    plans = []
    try:
        for packet, cost in reachable[: args.save_candidates]:
            last[packet] = obs['tick']
            plan = _save_probe(env, obs, root, packet, cost, args, decision_seed,
                               obs['tick'] + args.horizon)
            env.restore_fast(root)
            if plan is not None:
                plans.append(plan)
    finally:
        env.episode = episode
    stats['save_probes'] = stats.get('save_probes', 0) + len(plans)
    finite = [float(value) for value in totals if math.isfinite(float(value))]
    if not plans or not finite:
        return None
    plan = max(plans, key=lambda item: item['value'])
    if plan['value'] > max(finite) + MARGIN:
        stats['save_overrides'] = stats.get('save_overrides', 0) + 1
        stats['save_wait_ticks'] = stats.get('save_wait_ticks', 0) + plan['ticks']
        return plan
    return None


def _verified_choice(env, obs: dict, root: int, candidates: list[dict], totals: list[float],
                     counts: list[int], best: int, policy_index: int, policy_action: dict,
                     args, decision_seed: int, stats: dict, *, fresh: bool = False) -> dict:
    if not fresh:
        plan = _save_plan(env, obs, root, candidates, totals, args, decision_seed, stats)
        if plan is not None:
            stats['last_targets'] = None
            wait = {'type': 'wait', 'ticks': plan['ticks'], 'until': 'timeout'}
            _probe_decision(args, obs, candidates, totals, wait, decision_seed, policy_action,
                            best, policy_index)
            return wait
    before = fresh or (best != policy_index and totals[best] - totals[policy_index] > MARGIN)
    samples = getattr(args, "verify_samples", 0) + (args.samples if fresh else 0)
    if fresh:
        counts = [0] * len(candidates)
    if samples and before:
        prior = None if fresh else [totals[policy_index], totals[best]]
        started = time.perf_counter()
        # The original value batches and terminal-first sum order: a different order can change the
        # last bits even though the continuation jobs are identical.
        stats["rollout_batches"] = stats.get("rollout_batches", 0) + 1
        stats["rollout_lanes"] = stats.get("rollout_lanes", 0) + 2 * samples
        for index in (policy_index, best):
            fixed, leaves = [], []
            for future in range(args.samples, args.samples + samples):
                leaf = _rollout_policy(env, obs, candidates[index], obs["tick"] + args.horizon,
                                       root, (decision_seed * 1000 + future) % 2 ** 32)
                terminal = _score_terminal_or_value(leaf)
                if terminal is None:
                    leaves.append(leaf)
                else:
                    fixed.append(terminal)
            added = fixed + _leaf_values(leaves)
            totals[index] = (totals[index] * counts[index] + sum(added)) / (counts[index] + samples)
        stats["last_verification"] = {
            "samples": samples, "student": policy_action, "challenger": candidates[best],
            "before_values": prior, "after_values": [totals[policy_index], totals[best]],
            "before_override": True, "after_override": totals[best] - totals[policy_index] > MARGIN,
            "original_samples": [counts[policy_index], counts[best]], "decision_seed": decision_seed,
            "mode": "fresh" if fresh else "reproduced",
            "seconds": time.perf_counter() - started}
    # Did the winner come from --student-candidates? Counted here so both the halving and the
    # full search report it; the override counter below is the subset that changed the action.
    if json.dumps(candidates[best], sort_keys=True) in (stats.get("last_student_keys") or ()):
        stats["student_candidate_best"] = stats.get("student_candidate_best", 0) + 1
        if best != policy_index and totals[best] - totals[policy_index] > MARGIN:
            stats["student_candidate_override"] = stats.get("student_candidate_override", 0) + 1
    stats["last_targets"] = list(zip(candidates, totals))
    stats["last_best"] = candidates[best]
    if totals[best] - totals[policy_index] > MARGIN and best != policy_index:
        stats["overrides"] += 1
        stats["override_kinds"][candidates[best]["type"] + "_over_" + policy_action["type"]] += 1
        chosen = candidates[best]
    else:
        chosen = policy_action
    _probe_decision(args, obs, candidates, totals, chosen, decision_seed, policy_action, best, policy_index)
    return chosen


def _frame_json(frame: dict) -> str:
    """One archive line: byte for byte what pvz_frame_store.write_frames serializes."""
    return json.dumps(frame, separators=(",", ":")) + "\n"


def _final_zombies(obs: dict) -> list[dict]:
    # ROLLOUT returns a lean board even when ordinary root observations are full.
    from render_episode import _frame
    return [dict(row=z[0], type=z[1], x=z[2], health=z[4])
            for z in _frame(obs)['zombies'] if z[5]]


def _play(job: dict, observer=None) -> dict:
    args = job["args"]
    from collections import Counter
    import torch
    from pvz_event_env import EventWaitEnv
    from render_episode import _frame, load_training_task, training_task_kwargs
    from pvz_tasks import profile_for_deck
    from pvz_env import TaskSpec
    task_definition = (json.loads(args.task_json) if getattr(args, 'task_json', None) else
                       load_training_task(Path(args.manifest), args.task_id))
    kw = training_task_kwargs(task_definition, job["seed"])
    # One simulator process per worker, reset for every game: starting the process and loading
    # resources costs ~0.7 s, as much as a whole student-only game. RESET_V2 rebuilds the board
    # from the seed, so a reused process plays the same game (scripts/perf_rollout_check.py).
    env = _W.get("env")
    if env is None:
        env = _W["env"] = EventWaitEnv(args.resource_dir, executable=getattr(args, "executable", None), headless=True)
        env.load_rollout_net(_W["rollout_file"])
    if job["arm"] == "fast_search" and not getattr(env, "_teacher_value_loaded", False):
        env.load_value_net(_W["value_files"][0])
        env.load_value_net(_W["value_files"][1], static=True)
        env._teacher_value_loaded = True
    token_observation = bool(getattr(args, "obs_v3", False) or getattr(_W["fast"], "token_input", None) == "obs_v3")
    initial_tokens = token_observation and not getattr(args, 'demo_archive', None)
    if initial_tokens or getattr(env, '_token_observation', False):
        env.set_token_observation(initial_tokens)
    env._token_observation = initial_tokens
    from pvz_research_data import SparseTokenWriter, capture_frame, SIDECAR_NAME
    token_writer = None
    if getattr(args, 'obs_v3', False) and args.frames:
        tag = job['seed'] if job.get('future_seed') is None else f"{job['seed']}_f{job['future_seed']}"
        token_writer = SparseTokenWriter(Path(args.frames)/f'seed_{tag}'/f'{args.frames_label}_{job["arm"]}')
    task = TaskSpec(level=kw["level"], seed=kw["seed"], playthrough=2,
                    profile=profile_for_deck(tuple(kw["deck"]), kw.get("first_aid_owned", False)),
                    **{k: v for k, v in kw["task_extra"].items() if v is not None})
    # A demonstration start draws its future from RESEED instead of the task seed, so the teacher's
    # own sampling (torch draws and the search's decision seed) follows the same number.
    future_seed = job.get("future_seed")
    torch.manual_seed(kw["seed"] if future_seed is None else future_seed)
    started = time.time()
    stats = {"searched": 0, "overrides": 0, "shovels": 0, "override_kinds": Counter(),
             "student_candidate_added": 0, "student_candidate_best": 0, "student_candidate_override": 0}
    try:
        demo_archive = getattr(args, 'demo_archive', None)
        prefix = []
        _W['demo_actions'] = []
        if demo_archive:
            from dataclasses import asdict
            from human_demo import replay
            demo_meta = json.loads((Path(demo_archive) / 'meta.json').read_text())
            # Compare every field the archive recorded. A field added to TaskSpec after the
            # demonstration was recorded (fog_observation, task53) did not exist then and must not
            # block the start. fog_observation is exempt outright (task70 B6): it is presentation
            # only, so a demo start cell may replay a null-mode recording under human_v2, and the
            # _hv2 start cells deliberately carry task ids that extend the recorded one.
            recorded = demo_meta['environment_task']
            current = json.loads(json.dumps(asdict(task)))
            if (demo_meta['task']['id'] != args.task_id and
                    not (isinstance(args.task_id, str) and args.task_id.startswith(demo_meta['task']['id'] + '_')) or
                    demo_meta['task']['deck'] != list(kw['deck']) or
                    any(current.get(key) != value for key, value in recorded.items()
                        if key != 'fog_observation')):
                raise ValueError('demonstration task, deck, seed or environment differs from the manifest')
            obs, _, prefix = replay(env, Path(demo_archive), decision=args.demo_decision,
                                    fog_observation=task.fog_observation)
            if future_seed is not None:
                # The restored snapshot carries the demonstration's RNG, so replaying it would hand the
                # teacher the true future (the clairvoyance problem, scripts/verify_reseed.py). RESEED
                # re-picks every wave that has not started yet and leaves the board untouched.
                env.reseed(future_seed)
            if token_observation:
                # Prefix ticks are restored, not collected or passed to the student.
                env.set_token_observation(True)
                env._token_observation = True
                obs = env.observe()
            from pvz_frame_store import frame_path, read_frames
            demo_frames = read_frames(frame_path(Path(demo_archive)))
            _W['demo_actions'] = [(before['tick'], after['action'])
                                  for before, after in zip(demo_frames, demo_frames[1:])
                                  if (after.get('action') or {}).get('type') == 'plant']
        else:
            obs, _ = env.reset(deck=tuple(kw["deck"]), task=task)
        fog_diagnostics = None
        native_game = (getattr(args, 'native_games', False) and job['arm'] == 'fast'
                       and not args.frames and observer is None)
        if (task_definition.get('fog_observation') and obs.get('fog')
                and (not native_game or getattr(args, 'fog_diagnostics', False))):
            from pvz_fog import FogEpisodeDiagnostics
            fog_diagnostics = FogEpisodeDiagnostics()
            fog_diagnostics.sample(env, obs)
        from pvz_value_net import observation_context
        _W["context"] = observation_context(obs)
        task_context = {"level": kw["level"], "multiplier": obs.get("zombie_count_multiplier", 1.0),
                        "roster": (obs.get("loadout_context") or {}).get("zombie_roster") or []}
        decisions = 0
        # One JSON line per decision, exactly what write_frames would serialize -- kept as bytes
        # rather than as parsed dicts, so the archive costs 2.8 MB instead of ~18 MB of Python
        # objects for the whole game (task18).
        frames = [_frame_json(capture_frame(obs, token_writer))] if args.frames else None
        fast_state = {}
        if prefix:
            fast_state['frame'] = prefix[-2] if len(prefix) > 1 else None
            fast_state['action'] = prefix[-1].get('action')
            if _W['fast'].obs != 'v1':
                from pvz_student_observation import StudentHistory
                fast_state['history'] = StudentHistory(_W['fast'].obs)
                for frame in prefix[:-1]:
                    fast_state['history'].extra(frame)
        _W["student_history"] = None
        # DAgger (DESIGN.md §14.6): the search still labels every decision, but with probability
        # 1 - dagger_beta the student's own action is executed, so the archive covers the positions the
        # student itself gets into. The label is the anchored search decision (overrides only past MARGIN).
        dagger = random.Random(kw["seed"]) if args.dagger_beta < 1.0 else None
        ready_mowers = lambda o: sum(1 for d in o.get("defenses") or [] if d["state"] == 1)
        mowers_start = mowers_left = ready_mowers(obs)
        planted: dict[int, int] = {}
        planted_by_type: dict[int, int] = {}
        from pvz_episode_stats import planted_by_type as read_types, planted_slots, plant_type_before_action
        # 整局 C++ 接口只返回末状态，逐决策档案需要走 Python 对局循环。
        if job["arm"] == "fast" and not args.frames and fog_diagnostics is None and observer is None:
            if fast_state.get("history") is not None:
                _W["student_history"] = fast_state["history"]   # a demonstration start has history
            # ENV_SKIP_HEIGHT (bit 4) drops PlantDrawHeightOffset, which only moves a plant's
            # reanim on screen; the student never reads it, and the final board state is
            # unchanged (MEMORY/docs/task166c-fog-profile.md).
            obs = env.rollout(None, 0, max_actions=MAX_ACTIONS, sim_skip=4)
            details = env._last_response
            decisions, mowers_left = details["decisions"], details["mowers_left"]
            planted = planted_slots(details)
            planted_by_type = read_types(details,task_definition)
            if planted_by_type is None or planted is None:
                raise RuntimeError('native executable lacks recoverable type counts; rebuild this worktree')
        native_root = (getattr(args, 'native_games', False) and observer is None
                       and getattr(_W['fast'], 'token_input', None) != 'obs_v3')
        while not obs["terminal"] and decisions < MAX_ACTIONS:
            if observer is not None:
                observer.before(env, obs, fast_state, decisions)
            action = policy_choice = _fast_action(obs, fast_state, env if native_root else None)
            _W["student_history"] = fast_state.get("history")
            if job["arm"] == "fast_search":
                searched_before = stats['searched']
                stats["last_targets"] = None
                stats.pop("last_verification", None)
                # Capture root inputs only. CNN teacher rollouts need no rich JSON observations.
                pause_tokens = token_writer is not None and getattr(_W['fast'], 'token_input', None) != 'obs_v3'
                if pause_tokens:
                    env.set_token_observation(False)
                try:
                    action = _search(env, obs, action, args,
                                     (kw["seed"] if future_seed is None else future_seed) * 10000 + decisions, stats)
                finally:
                    if pause_tokens:
                        env.set_token_observation(True)
                if fog_diagnostics is not None:
                    fog_diagnostics.compare(obs, policy_choice, action, stats['searched'] > searched_before)
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
            # Planting on an occupied cell only became legal with wall-nut first aid (a damaged
            # wall-nut takes another nut of the same type, Board::CanPlantAt) -- record what stood
            # there, so the report can say how often the teacher actually used it. Harmless for
            # ordinary plants: the legal list never offers an occupied cell without first aid.
            planted_type = plant_type_before_action(obs,action['packet']) if action['type']=='plant' else None
            if action["type"] == "plant":
                occupant = next((p for p in obs.get("plants") or []
                                 if p["row"] == action["row"] and p["col"] == action["col"]), None)
                if occupant is not None:
                    stats.setdefault("replant_events", []).append({
                        "tick": obs["tick"], "row": action["row"], "col": action["col"],
                        "packet": action["packet"], "occupant": occupant.get("type"),
                        "occupant_health": occupant.get("health"),
                        "plant_type":planted_type,"same_type": occupant.get("type") == planted_type})
            obs, _r, done, _t, info = env.step(action)
            rejected = not info.get("ok")
            if not rejected and action["type"] == "plant":
                planted[action["packet"]] = planted.get(action["packet"], 0) + 1
                assert planted_type is not None and planted_type >= 0
                planted_by_type[planted_type] = planted_by_type.get(planted_type, 0) + 1
                if fog_diagnostics is not None:
                    fog_diagnostics.planted(planted_type, action)
            if rejected:
                obs, _r, done, _t, info = env.step({"type": "wait", "ticks": 60})
            if observer is not None:
                observer.after(obs, action, policy_choice, rejected, info)
            if fog_diagnostics is not None:
                fog_diagnostics.sample(env, obs)
            if frames is not None:
                frame = capture_frame(obs, token_writer)
                if stats.get("last_targets"):
                    # the search's estimate for every candidate it compared: [packet (-1 = wait), row, col, value];
                    # a soft target for the student (train_rollout_policy --soft-weight). Before "action" and
                    # "label", because the trainer finds the target by scanning for the last of those.
                    # Shovel candidates are left out: the student has no shovel output yet, so a shovel
                    # row would be read as a wait target by the trainer (phase 2 adds the head).
                    frame["targets"] = [[c.get("packet", -1), c.get("row", -1), c.get("col", -1), round(v, 4)]
                                        for c, v in stats["last_targets"]
                                        if abs(v) != float("inf") and c["type"] != "shovel"]
                frame["action"] = action          # what was executed (after the search), like render_episode
                frame["action_rejected"] = rejected
                if observer is not None:
                    frame["eaten"] = int((info.get("events") or {}).get("plants_eaten", 0))
                    frame["eaten_free"] = int((info.get("events") or {}).get("plants_eaten_free", 0))
                if stats.get("last_verification"):
                    # Wall time is not part of the label: keeping it in the archive makes two runs of
                    # the same arm differ byte for byte (and the GPU schedule comparison impossible).
                    verification = dict(stats["last_verification"])
                    frame["verification"] = verification
                if dagger is not None:
                    frame["label"] = label        # last field: train_rollout_policy scans for it
                frames.append(_frame_json(frame))
            decisions += 1
            counters = _W.get("metrics_counters")
            if counters is not None and decisions % 32 == 0:
                counters[job['metrics_index'] * 2 + 1] = decisions
            if not obs["terminal"]:      # a won level fires the remaining mowers; count before that
                mowers_left = ready_mowers(obs)
            if done:
                break
    except BaseException:
        if token_writer is not None:
            token_writer.abort()
        _W.pop("env", None)
        env.close()
        raise
    if frames is not None:
        # Same layout as episode_query archives, so scripts/train_value_net.py reads it directly.
        seed_tag = job["seed"] if future_seed is None else f"{job['seed']}_f{future_seed}"
        archive = Path(args.frames) / f"seed_{seed_tag}" / f"{args.frames_label}_{job['arm']}"
        archive.mkdir(parents=True, exist_ok=True)
        from pvz_frame_store import write_frames_from_source
        write_frames_from_source(archive / "frames.npz",
                                 "".join(frames).encode())
        attachment_info = None
        if token_writer is not None:
            token_writer.finish(archive / SIDECAR_NAME, source_frame=archive / 'frames.npz')
            attachment_info = dict(file=SIDECAR_NAME, version=3,
                                   bytes=(archive/SIDECAR_NAME).stat().st_size,
                                   frames=token_writer.rows)
        from dataclasses import asdict
        from pvz_episode_archive import write_meta
        archive_task = dict(id=args.task_id, seed=job['seed'], level=kw['level'], deck=list(kw['deck']),
                            policy='student', checkpoint=str(Path(args.rollout_policy).resolve()),
                            arm=job['arm'], playthrough=2,
                            **{k: v for k, v in kw['task_extra'].items() if v is not None})
        archive_outcome = dict(result=obs.get('result') or 0, won=obs.get('result') == 1,
                               final_wave=obs['wave'], final_tick=obs['tick'],
                               wave_count=obs.get('wave_count'),
                               reason='win' if obs.get('result') == 1 else 'not_won')
        if (demo_archive or 'deck_perturb' in task_definition or 'deck_random' in task_definition
                or task_definition.get('live193')
                or task_definition.get('conveyor_start') == 'original'
                or task_definition.get('fog_observation')):
            archive_task.update(level=kw['level'], deck=list(kw['deck']), playthrough=2,
                                policy='student', checkpoint=str(Path(args.rollout_policy).resolve()),
                                task_id=args.task_id, first_aid_owned=kw.get('first_aid_owned', False),
                                **{k: v for k, v in kw['task_extra'].items() if v is not None})
            if 'deck_perturb' in task_definition:
                archive_task['deck_perturb'] = task_definition['deck_perturb']
                archive_task['base_deck'] = task_definition['deck']
            archive_outcome.update(final_tick=obs['tick'], wave_count=_frame(obs)['scene'].get('wave_count', 0),
                                   reason='win' if obs.get('result') == 1 else 'not_won')
        write_meta(archive, {
            "label_source": "teacher",
            **({'observation_attachment': attachment_info} if attachment_info else {}),
            **({"demo_archive": str(Path(demo_archive).resolve()), "demo_decision": args.demo_decision,
                "demo_note": demo_meta.get('note', '')} if demo_archive else {}),
            "task": archive_task,
            "environment_task": asdict(task),
            "frame_count": len(frames), "operations": decisions,
            "context": task_context,
            "outcome": archive_outcome})
    result = {"seed": job["seed"], "future_seed": future_seed, "arm": job["arm"], "won": obs.get("result") == 1,
            **({'deck': list(kw['deck'])} if 'deck_random' in task_definition else {}),
            **({'fog_diagnostics': fog_diagnostics.result(), 'deck': list(kw['deck']),
                'final_tick': obs['tick']} if fog_diagnostics is not None else {}),
            "final_wave": obs["wave"], "decisions": decisions, "seconds": round(time.time() - started, 1),
            "final_tick": obs["tick"], "terminal": bool(obs.get("terminal")),
            **({'obs_v3_bytes': attachment_info['bytes']} if args.frames and attachment_info else {}),
            "wave_count": obs.get("wave_count"),
            # Public final-board evidence, including the row/type of a house breach.
            # Keep all on-board zombies: nearest-x alone is ambiguous for bungees/vehicles.
            "final_zombies": _final_zombies(env.privileged_state()
                if task_definition.get('fog_observation') == 'storm' else obs),
            "mowers_used": mowers_start - mowers_left, "planted_slots": planted,
            "planted_types": planted_by_type,
            "searched": stats["searched"], "overrides": stats["overrides"],
            **{name: stats.get(name, 0) for name in ("rollout_batches", "rollout_lanes",
                "halving_candidates", "halving_rollouts", "halving_selected_lanes")},
            "student_candidate_added": stats["student_candidate_added"],
            "student_candidate_best": stats["student_candidate_best"],
            "student_candidate_override": stats["student_candidate_override"],
            "demo_candidate_added": stats.get('demo_candidate_added', 0),
            "shovel_events": stats.get("shovel_events", []),
            "replant_events": stats.get("replant_events", []),
            "override_kinds": dict(stats["override_kinds"])}
    _release_freed_memory()      # this game's archives are written; its buffers are garbage now
    counters = _W.get("metrics_counters")
    if counters is not None:
        # Each job owns its slots. A killed worker cannot strand a shared lock.
        counters[job['metrics_index'] * 2] = 1
        counters[job['metrics_index'] * 2 + 1] = decisions
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--native-games', action=argparse.BooleanOptionalAction, default=True,
                   help='use C++ fog student games and CNN collection root forwards (deploy19)')
    p.add_argument('--fog-diagnostics', action='store_true',
                   help='collect per-decision fog diagnostics, using the Python game loop')
    p.add_argument("--obs-v3", action="store_true", help="write separate sparse float32 obs-v3 attachment beside compact frames")
    p.add_argument("--rollout-policy", required=True, help="the student (scripts/train_rollout_policy.py)")
    p.add_argument("--rollout-policy-gpt",
                   help="GPT behavior-clone prior for fog/roof/boss levels "
                        "(BACKGROUND_4_FOG/5_ROOF/6_BOSS); default --rollout-policy")
    p.add_argument("--value", required=True, help="value net that scores the leaves")
    p.add_argument("--value-roof", help="value net that scores the leaves on roof levels 41-50 "
                   "(including the boss); default --value")
    p.add_argument("--static-value", help="value net that picks the candidate cells right after planting "
                   "(default --value)")
    p.add_argument("--task-json", help="inline curriculum task definition (task193; no derived manifest)")
    p.add_argument("--manifest", required=True)
    p.add_argument("--task-id", required=True)
    p.add_argument("--demo-archive", type=Path, help="verified human demonstration used as the starting state")
    p.add_argument("--demo-decision", type=int, default=1, help="start immediately before this demo decision")
    p.add_argument("--future-seeds",
                   help="comma-separated RESEED values drawn right after the demonstration is restored; one "
                        "game per value, and the teacher's own sampling follows the same number. Without it "
                        "the restored RNG replays the demonstration's true future (clairvoyant, "
                        "scripts/verify_reseed.py)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seeds", type=int, default=64)
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument("--seed-list", help="explicit comma-separated seeds (overrides --seed-offset/--seeds); "
                   "task31 uses it to sample the seeds whose perturbed deck drops one specific base card")
    p.add_argument("--arms", default="fast_search", help="fast (the student alone), fast_search (the teacher)")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                   help="hard cap on game workers; the pool may grow up to this many "
                        "while the group runs (task138)")
    p.add_argument("--initial-workers", type=int,
                   help="workers to start with (default --workers). The controller passes the "
                        "width it admitted at group start and lets the pool grow from there")
    p.add_argument("--worker-memory-bytes", type=int, default=0,
                   help="measured RSS of one worker; 0 = unmeasured, which disables growth")
    p.add_argument("--memory-headroom-bytes", type=int, default=0,
                   help="memory reserved for a running background job; subtracted from "
                        "MemAvailable before admitting another worker")
    p.add_argument("--grow-interval", type=float, default=30.0,
                   help="seconds between two memory readings while the group runs")
    p.add_argument("--per-card", type=int, default=1, help="best cells of each card kept as candidates")
    p.add_argument("--student-candidates", type=int, default=0,
                   help="the student's own top-k plant actions (its policy probabilities) added to the "
                        "candidates; 0 = off, which is the old candidate set exactly")
    p.add_argument("--save-candidates", type=int, default=0,
                   help="cards the player cannot afford yet but has at least --save-share of the cost for: "
                        "wait in the engine until they are affordable and score the plant from that future "
                        "state; the decision becomes the wait when it beats the best candidate. "
                        "0 = off (task120)")
    p.add_argument("--save-share", type=float, default=0.5,
                   help="only probe a card once the current sun reaches this share of its cost")
    p.add_argument("--save-max-wait", type=int, default=1800, help="longest wait a save probe may simulate")
    p.add_argument("--save-chunk", type=int, default=300, help="wait step of a save probe")
    p.add_argument("--save-interval", type=int, default=600,
                   help="minimum ticks between two probes of the same card")
    p.add_argument("--samples", type=int, default=3)
    p.add_argument("--halving", action=argparse.BooleanOptionalAction, default=True,
                   help="successive halving over the candidates (1, 2, 4.. futures); --no-halving = every "
                        "candidate gets all --samples")
    p.add_argument("--horizon", type=int, default=2500)
    p.add_argument("--leaf-probe", help="diagnostic: write one JSON line per search decision with "
                   "candidate-leaf mean/std/top1-top2 to <path>.<pid>.jsonl; off by default")
    p.add_argument("--leaf-probe-limit", type=int, default=0,
                   help="per worker decision cap for --leaf-probe; 0 = unlimited")
    p.add_argument("--verify-samples", type=int, default=0,
                   help="extra independent futures for student and challenger before an override (0 = disabled)")
    p.add_argument("--min-gap", type=int, default=300,
                   help="ticks between searches unless the student itself wants to plant (0 = search every time)")
    p.add_argument("--rollout-sim-skip", type=int, default=61,
                   help="SIM_SKIP mask during rollouts only (0 = exact; see scripts/sim_skip_agreement.py)")
    p.add_argument("--executable", default=str(ROOT / "build/pvz-portable"),
                   help="fixed original-simulator binary used for roots and real game decisions")
    p.add_argument("--frames", help="write per-game frame archives here (for training)")
    p.add_argument("--frames-label", default="play")
    p.add_argument("--dagger-beta", type=float, default=1.0,
                   help="probability of executing the search's decision (1 = the teacher plays; < 1 = DAgger: "
                        "the student plays the rest, every frame still records the search's label)")
    p.add_argument("--resource-dir", default=DEFAULT_RESOURCE_DIR)
    return p


def main() -> None:
    p = parser()
    args = p.parse_args()
    if args.verify_samples < 0:
        p.error("--verify-samples must be nonnegative")
    if args.student_candidates < 0:
        p.error("--student-candidates must be nonnegative")
    if args.save_candidates < 0:
        p.error("--save-candidates must be nonnegative")
    if not 0 < args.save_share <= 1:
        p.error("--save-share must be in (0, 1]")
    if args.save_max_wait <= 0 or args.save_chunk <= 0 or args.save_interval < 0:
        p.error("--save-max-wait and --save-chunk must be positive, --save-interval nonnegative")
    if args.workers < 1:
        p.error("--workers must be positive")
    initial_workers = args.workers if args.initial_workers is None else args.initial_workers
    if not 1 <= initial_workers <= args.workers:
        p.error("--initial-workers must be within 1..--workers")
    if args.worker_memory_bytes < 0 or args.memory_headroom_bytes < 0:
        p.error("--worker-memory-bytes and --memory-headroom-bytes must be nonnegative")
    if args.grow_interval <= 0:
        p.error("--grow-interval must be positive")

    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT / "python"))
    from render_episode import load_training_task
    task_definition = (json.loads(args.task_json) if args.task_json else
                       load_training_task(Path(args.manifest), args.task_id))
    task_seeds = task_definition["seeds"]
    args.leaf_value = _leaf_value(args.value, args.value_roof, task_definition)
    args.rollout_policy = _terrain_policy(args.rollout_policy, args.rollout_policy_gpt, task_definition)
    if args.seed_list:
        seeds = [int(s) for s in args.seed_list.split(',') if s.strip()]
        unknown = [s for s in seeds if s not in set(task_seeds)]
        if unknown:
            p.error(f"--seed-list seeds are not in the task manifest: {unknown[:5]}")
    else:
        seeds = task_seeds[args.seed_offset: args.seed_offset + args.seeds]
    if args.demo_archive:
        demo = json.loads((args.demo_archive / 'meta.json').read_text())
        if not demo.get('verified'):
            p.error('verify the human demonstration before starting a teacher game')
        seed = demo['task']['seed']
        if seed not in seeds:
            p.error('demonstration seed is outside the selected manifest seeds')
        seeds = [seed]
    futures = [int(v) for v in args.future_seeds.split(",") if v.strip()] if args.future_seeds else [None]
    if args.future_seeds and not args.demo_archive:
        p.error("--future-seeds only applies to a demonstration start (--demo-archive)")
    if len(set(futures)) != len(futures):
        p.error("--future-seeds contains duplicates")
    args.out.mkdir(parents=True, exist_ok=True)
    results_path = args.out / "games.jsonl"
    done_keys = set()
    if results_path.exists():
        for line in results_path.read_text().splitlines():
            r = json.loads(line)
            done_keys.add((r["arm"], r["seed"] if r.get("future_seed") is None else r["future_seed"]))
    jobs = [{"seed": s, "future_seed": f, "arm": arm, "args": args}
            for arm in args.arms.split(",") for s in seeds for f in futures
            if (arm, s if f is None else f) not in done_keys]
    (args.out / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=1))
    # Keep the exact "N games on W workers" shape: scripts/qc_log_throughput.py parses it.
    print(f"{len(jobs)} games on {initial_workers} workers", flush=True)
    if initial_workers != args.workers:
        print(f"worker pool may grow to {args.workers} (per-worker "
              f"{args.worker_memory_bytes / GIB:.2f} GiB, headroom "
              f"{args.memory_headroom_bytes / GIB:.2f} GiB, every {args.grow_interval:g}s)",
              flush=True)
    value_keep, value_files = _export_values(args.leaf_value, args.static_value)
    _release_freed_memory()
    # Student weights remain shared on fork; value weights live only in engine processes.
    start_method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
    if start_method == "fork":
        _load_nets(value_files[0], args.rollout_policy, value_files[1])
        _freeze_startup()
    started = time.time()
    counters = mp.get_context(start_method).RawArray("q", len(jobs) * 2)
    for index, job in enumerate(jobs):
        job['metrics_index'] = index
    metrics = None
    # task138: the width is no longer frozen at group start. The pool starts at the
    # admitted width and adds workers when MemAvailable allows, up to --workers.
    pool = GrowablePool(mp.get_context(start_method), _play, init=_init_worker,
                        initargs=(value_files[0], args.rollout_policy, args.rollout_sim_skip,
                                  value_files[1], counters),
                        initial=initial_workers, upper=args.workers,
                        interval=args.grow_interval,
                        per_worker=args.worker_memory_bytes,
                        background=args.memory_headroom_bytes)
    try:
        with results_path.open("a") as out:
            from pvz_metrics import Metrics
            metrics = Metrics("search_play", progress_source=lambda: dict(
                games=sum(counters[::2]), decisions=sum(counters[1::2]))).start()
            width = initial_workers
            for i, res in enumerate(pool.run(jobs), 1):
                out.write(json.dumps(res) + "\n"); out.flush()
                if len(pool.processes) != width:
                    print(f"workers {width}->{len(pool.processes)} "
                          f"({time.time() - started:.0f}s)", flush=True)
                    width = len(pool.processes)
                if i % 10 == 0 or i == len(jobs):
                    print(f"{i}/{len(jobs)} ({time.time() - started:.0f}s)", flush=True)
            # A graceful close runs worker finalizers and reaps their simulator children. The
            # parent's RUSAGE_CHILDREN now includes all worker startup, games and teardown, once.
            pool.close()
    finally:
        pool.close()
        if metrics is not None:
            metrics.close()
        value_keep.cleanup()
    summarize(results_path)


def summarize(results_path: Path) -> dict:
    rows = [json.loads(line) for line in results_path.read_text().splitlines()]
    by = {}
    for r in rows:
        # A cold-start game has future_seed = null; the key must fall back to the game seed.
        by.setdefault(r["arm"], {})[r["seed"] if r.get("future_seed") is None else r["future_seed"]] = r
    summary = {arm: {"games": len(v), "won": sum(r["won"] for r in v.values()),
                     "mean_final_wave": round(sum(r["final_wave"] for r in v.values()) / len(v), 2),
                     "mean_seconds": round(sum(r["seconds"] for r in v.values()) / len(v), 1),
                     "overrides_per_game": round(sum(r["overrides"] for r in v.values()) / len(v), 1),
                     "searched_per_game": round(sum(r["searched"] for r in v.values()) / len(v), 1),
                     "shovels_per_game": round(sum(r.get("shovels", 0) for r in v.values()) / len(v), 2),
                     "games_with_shovel": sum(1 for r in v.values() if r.get("shovels")),
                     "replants_per_game": round(sum(len(r.get("replant_events") or []) for r in v.values()) / len(v), 2),
                     "first_aid_replants_per_game": round(sum(
                         sum(1 for e in (r.get("replant_events") or []) if e.get("same_type"))
                         for r in v.values()) / len(v), 2),
                     "mowers_used_per_game": round(sum(r.get("mowers_used", 0) for r in v.values()) / len(v), 2)}
               for arm, v in by.items()}
    (results_path.parent / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    main()
