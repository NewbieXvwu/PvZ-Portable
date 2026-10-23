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
     ``--candidate-coverage-model`` optionally adds legal cells from a learned, set-conditioned
     marginal-coverage predictor, bounded by ``--candidate-coverage-budget`` (default 2). Without
     a checkpoint the feature is off, with no extra model inference or candidate changes.
  2. Each candidate is played and then continued by the student for ``--horizon`` ticks in C++, ``--samples``
     times with different random futures (the same futures for every candidate). The leaf is scored by
     ``--value`` (a game that ended inside the horizon scores its result). ``--halving`` (default) rolls out
     the weaker half fewer times.
  3. The student's choice is overridden only when the best candidate beats it by ``--margin``.
     ``--verify-samples`` adds independent futures for just that pair before accepting an override.
Between searches at least ``--min-gap`` ticks pass unless the student itself wants to plant.

Arms: ``fast`` = the student alone, ``fast_search`` = the teacher. ``--frames`` writes per-decision archives
for training (with ``--dagger-beta`` < 1 the student executes part of the moves, the search still labels all).

Continuations run in the original C++ simulator by default. ``--rollout-backend cuda`` sends them to
``build/rollout_cuda --serve`` (python/pvz_cuda_rollout.py) instead, which gathers the decisions of several
workers into one GPU wave and returns the leaves as the value net's own features -- and simulates *every*
tick. ``--rollout-sim-skip`` is a CPU presentation-work bitmask: mask 29 skips colour, height,
attachment matrices and messages, while preserving game ticks and animation lifecycle updates.

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
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESOURCE_DIR = "/Users/newbiexvwu/Downloads/Plants_Vs_Zombies_V1.2.0.1073_EN"
NOOP = {"type": "wait", "ticks": 60, "until": "timeout"}

_W: dict = {}


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


def _wait_for_gpu_pause(args, seed: int) -> None:
    """Yield the GPU between games while training owns the pause file."""
    if getattr(args, "rollout_backend", "cpp") != "cuda":
        return
    filename = getattr(args, "gpu_pause_file", None) or os.environ.get("PVZ_GPU_PAUSE_FILE")
    if not filename:
        return
    path = Path(filename).expanduser()
    announced = False
    while path.exists():
        if not announced:
            # A queue paused before its very first game must greet the server too:
            # otherwise its startup timeout expires while every worker is waiting.
            client = _W.get("cuda")
            if client is not None:
                client.connect()
            print(f"GPU collection paused before seed {seed}: {path}", flush=True)
            _gpu_pause_event(args, "paused", seed)
            announced = True
        time.sleep(10)
    if announced:
        print(f"GPU collection resumed before seed {seed}", flush=True)
        _gpu_pause_event(args, "resumed", seed)


def _gpu_pause_event(args, event: str, seed: int) -> None:
    """The same news as the printed line, for programs: one JSON line per event in <out>/gpu_pause.jsonl."""
    with (Path(args.out) / "gpu_pause.jsonl").open("a") as f:
        f.write(json.dumps({"event": event, "seed": seed, "time": time.time()}) + "\n")


def _close_worker_env():
    env = _W.pop("env", None)
    if env is not None:
        env.close()
    client = _W.pop("cuda", None)
    if client is not None:
        client.close()


def _load_nets(value_path: str, rollout_path: str, static_value_path: str | None = None,
               cpp_rollouts: bool = True, coverage_path: str | None = None,
               coverage_budget: int = 2) -> None:
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
    _W.pop("coverage", None)
    if coverage_path:
        from pvz_candidate_coverage import load
        _W["coverage"] = load(coverage_path)
        _W["coverage_budget"] = coverage_budget
    from pvz_rollout_policy import load_student
    _W["value_files"] = (value_path, static_value_path or value_path)
    _W["fast"] = load_student(rollout_path)
    if cpp_rollouts:
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


def _init_worker(value_path: str, rollout_path: str, rollout_sim_skip: int = 0, cpp_rollouts: bool = True,
                 static_value_path: str | None = None, cuda_socket: str | None = None,
                 coverage_path: str | None = None, coverage_budget: int = 2) -> None:
    if rollout_sim_skip & 256 and cuda_socket is not None:
        raise ValueError("sparse tracks are a CPU-only rollout feature")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    _W.update(lean_rollouts=True, rollout_sim_skip=rollout_sim_skip, batch_waits=True, cpp_rollouts=cpp_rollouts,
              cuda_socket=cuda_socket)
    sys.path.insert(0, str(ROOT / "python"))
    sys.path.insert(0, str(ROOT / "scripts"))
    from pvz_common import OOM_SCORE_WORKER, set_oom_score_adj
    # Leave the WSL entry point its memory: under pressure the kernel should pick this game worker first.
    set_oom_score_adj(OOM_SCORE_WORKER)
    from multiprocessing.util import Finalize
    Finalize(None, _close_worker_env, exitpriority=10)
    if "fast" not in _W:        # a preloading parent already loaded them, and we inherited the pages
        _load_nets(value_path, rollout_path, static_value_path, cpp_rollouts, coverage_path, coverage_budget)
    import torch
    # Forking does not carry the parent's intra-op thread count into the child; a worker is
    # single threaded either way.
    torch.set_num_threads(1)
    _freeze_startup()
    if cuda_socket:
        # One connection per worker; the *server* is what turns the workers' separate decisions into one
        # GPU wave.  The dump file is this worker's only channel of positions to it, and a worker is single
        # threaded, so rewriting it between requests cannot race with a wave still reading it -- the server
        # parses (and copies) a request before it answers it, and the client does not touch the file again
        # until the answer arrives.
        import tempfile
        from pvz_cuda_rollout import CudaRolloutClient
        _W["cuda_dump_tmp"] = tempfile.TemporaryDirectory(prefix="pvz-cuda-dump-")
        _W["cuda_dump"] = str(Path(_W["cuda_dump_tmp"].name) / "root.cudaroot")
        _W["cuda"] = CudaRolloutClient(cuda_socket)


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
                           student_history=student_history, score_value=True)
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
                obs["value"] = _score_terminal_or_value(obs)
                if obs["value"] is None:
                    obs["value"] = env.leaf_values()[0]
                return obs
            if batch_waits and not (obs.get("legal_actions") or {}).get("plants"):
                # With no legal plant the student can only wait 60 ticks, again and again: let
                # the simulator repeat that itself. Its features only read the previous frame's
                # tick and whether the previous action was a plant, so this state is exact.
                obs, _r, done, _t, info = env.wait_until_legal(until_tick)
                fast_state["frame"], fast_state["action"] = {"tick": info["previous_tick"]}, dict(NOOP)
                if done or obs.get("terminal") or obs["tick"] >= until_tick:
                    obs["value"] = _score_terminal_or_value(obs)
                    if obs["value"] is None:
                        obs["value"] = env.leaf_values()[0]
                    return obs
            action = _fast_action(obs, fast_state)
    finally:
        if sim_skip:
            env.set_sim_skip(0)
        if lean:
            env.set_lean_observation(False)


def _cuda_continuations(env, root: int, jobs: list[tuple[dict, int]], client, args) -> list | None:
    """Every continuation of one decision as a single request to the GPU server.

    Returns one handle per job, or ``None`` when this position cannot go to the GPU at all -- a root the
    shared core does not cover, or a server that went away -- and the caller runs the C++ rollout instead.
    A decision is 10-40 lanes and a GPU round costs the same at 10 lanes as at 1000, so sending the whole
    batch in one request (and letting the server merge the other workers' batches into the same wave) is
    the only way this path can ever be cheaper than the CPU it replaces.

    ``max_actions`` is the original's own default, which the C++ path gets by not passing one: the horizon
    is what stops a search continuation, not the action budget.
    """
    from pvz_cuda_rollout import CudaServerError, leaf_tensors
    import numpy as np

    dump_path = _W["cuda_dump"]
    env.restore(root)
    try:
        response = env.cuda_dump(dump_path, write_level=True, student_history=_W.get("student_history"))
    except (RuntimeError, OSError):
        return None
    if not response.get("supported"):
        _W["cuda_unsupported"] = _W.get("cuda_unsupported", 0) + 1
        return None
    pairs = [(0, seed % 2 ** 32, candidate) for candidate, seed in jobs]
    try:
        records, leaves = client.run(dump_path, pairs, horizon=args.horizon, max_actions=10 ** 6)
    except CudaServerError as error:
        _W["cuda"] = None
        _W["cuda_failed"] = str(error)
        print(f"cuda rollout unavailable, falling back to C++: {error}", flush=True)
        return None
    boards, scalars, context = leaf_tensors(leaves)
    return [(record, (boards[index], scalars[index], context[index]))
            for index, record in enumerate(records)]


def _continuations(env, obs: dict, root: int, jobs: list[tuple[dict, int]], args,
                   stats: dict) -> list:
    """Play each candidate and continue with the student for ``--horizon`` ticks; one handle per job.

    A handle is either the leaf observation (the C++/Python rollout) or a ``(record, features)`` pair from
    the GPU.  ``_leaf_values`` scores both the same way, so the search logic does not care which backend ran.
    """
    stats["rollout_batches"] = stats.get("rollout_batches", 0) + 1
    stats["rollout_lanes"] = stats.get("rollout_lanes", 0) + len(jobs)
    client = _W.get("cuda")
    if client is not None:
        handles = _cuda_continuations(env, root, jobs, client, args)
        if handles is not None:
            stats["cuda_batches"] = stats.get("cuda_batches", 0) + 1
            stats["cuda_lanes"] = stats.get("cuda_lanes", 0) + len(jobs)
            stats["cuda_error_lanes"] = stats.get("cuda_error_lanes", 0) + sum(
                bool(record["errorFlags"]) for record, _features in handles)
            return handles
    if getattr(args, "rollout_backend", "cpp") == "cuda" or _W.get("cuda_socket"):
        stats["cuda_fallback_batches"] = stats.get("cuda_fallback_batches", 0) + 1
        stats["cuda_fallback_lanes"] = stats.get("cuda_fallback_lanes", 0) + len(jobs)
    return [_rollout_policy(env, obs, candidate, obs["tick"] + args.horizon, root, seed % 2 ** 32)
            for candidate, seed in jobs]


def _leaf_values(handles: list) -> list[float | None]:
    """Terminal results and value-net scores, in job order; ``None`` for a lane that never ran.

    Dropping those matters: a false 0.0 for a lane whose root would not import would be read by the search
    as "this candidate loses", which is the opposite of what an unusable position means.
    """
    values: list[float | None] = [None] * len(handles)
    rows, row_slots = [], []
    for slot, handle in enumerate(handles):
        if isinstance(handle, tuple):
            record, features = handle
            if record["errorFlags"]:
                continue
            if record["result"]:
                values[slot] = 1.0 if record["result"] > 0 else 0.0
            else:
                rows.append(features)
                row_slots.append(slot)
        else:
            terminal = _score_terminal_or_value(handle)
            values[slot] = handle["value"] if terminal is None else terminal
    if rows:
        # External simulator features also use the engine forward, never a Python value net.
        import numpy as np
        import struct
        import tempfile
        header = Path(_W["value_files"][0]).read_bytes()[:20]
        _magic, _width, encoder, context_size = struct.unpack("<8sIII", header)
        if encoder != 1:
            raise ValueError("CUDA leaves do not include value encoder 2 scene/grave features")
        with tempfile.NamedTemporaryFile(suffix=".features") as features:
            for board, scalars, context in rows:
                features.write(np.asarray(board, dtype="<f4").tobytes())
                features.write(np.asarray(scalars, dtype="<f4").tobytes())
                if context_size:
                    features.write(np.asarray(context, dtype="<f4").tobytes())
            features.flush()
            scored = _W["env"].leaf_feature_values(features.name, len(rows))
        for slot, value in zip(row_slots, scored):
            values[slot] = value
    return values


def _static_scores(env, root: int, actions: list[dict], income) -> list[tuple[float, dict]]:
    """Score candidates in engine order; rejected branches do not enter the candidate set."""
    return [(score, action) for score, action in zip(env.static_scores(root, actions, income or 0.0), actions)
            if score is not None]


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
        generator = getattr(args, "candidate_generator", "static")
        if generator not in ("static", "student-topk", "student-per-card"):
            raise ValueError("unknown candidate generator")
        student_ranked = None
        if generator != "static":
            if shovels or _W.get("coverage") is not None:
                raise ValueError("student candidate generators require shovels and coverage off")
            if _W.get("student_frame") is None:
                raise ValueError("student candidate generators require the current student frame")
            from pvz_rollout_policy import ranked_plants
            student_ranked = ranked_plants(_W["fast"], _W["student_frame"], _W.get("student_previous_frame"),
                                          legal, context=_W.get("context"), history=_W.get("student_history"))
        scored = sorted(_static_scores(env, root, plants, income), key=lambda item: -item[0]) if generator == "static" else []
        scored_shovels = sorted(_static_scores(env, root, shovels, income), key=lambda item: -item[0]) if generator == "static" else []
        # The best cells of *every* card, not the overall top-k: the static value net was trained on
        # games where some cards (cherry bomb, wall-nut) were never played, so it cannot rank them
        # against shooters right after planting -- only the rollout can show what they are worth.
        by_card: dict[int, list] = {}
        ranking = scored if generator == "static" else [
            (v, {"type": "plant", **{k: int(a[k]) for k in ("packet", "col", "row")}})
            for v, a in student_ranked]
        for v, a in ranking:
            by_card.setdefault(a["packet"], []).append(a)
        base_plants = ([a for _v, a in ranking[:args.student_candidates]] if generator == "student-topk"
                       else [a for cells in by_card.values() for a in cells[: args.per_card]])
        candidates = (base_plants
                      + [a for _v, a in scored_shovels[: getattr(args, "shovel_candidates", 0)]] + [NOOP])
        policy_key = json.dumps(policy_action if policy_action["type"] == "plant" else NOOP, sort_keys=True)
        if policy_action["type"] == "plant" and all(json.dumps(c, sort_keys=True) != policy_key for c in candidates):
            candidates.append(policy_action)

        # --student-candidates k: the student's own top-k plants join the candidates
        # (pvz_rollout_policy.ranked_plants). The static value net picks one cell per card, so a cell
        # the student would actually pick -- or its second choice -- never gets compared; the search
        # could only reach it indirectly by shovelling and letting the student react (learn/06).
        # Deduplicated against everything already there. k = 0 (default) leaves the set untouched.
        stats["last_student_keys"] = set()
        if (generator != "student-topk" and getattr(args, "student_candidates", 0)
                and _W.get("student_frame") is not None):
            from pvz_rollout_policy import ranked_plants
            present = {json.dumps(c, sort_keys=True) for c in candidates}
            added = 0
            if student_ranked is None:
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

        if _W.get("coverage") is not None and _W.get("coverage_budget", 0):
            from pvz_candidate_coverage import features, propose
            from pvz_rollout_policy import ranked_plants
            from render_episode import _frame
            if student_ranked is None:
                student_ranked = ranked_plants(_W["fast"], _W["student_frame"], _W.get("student_previous_frame"),
                                              legal, context=_W.get("context"), history=_W.get("student_history"))
            frame = _frame(obs)
            state, action_features = features(frame, _W["context"], plants + [NOOP],
                                              scored + [(_static_scores(env, root, [dict(type="wait", ticks=0)], income)[0][0], NOOP)], student_ranked)
            added = propose(_W["coverage"], state, action_features, plants + [NOOP], candidates,
                            _W["coverage_budget"])
            candidates.extend(added)
            stats["coverage_candidate_added"] = stats.get("coverage_candidate_added", 0) + len(added)
            stats["last_coverage_actions"] = added

        if args.halving:
            return _halving_choice(env, obs, root, candidates, policy_action, policy_key, args, decision_seed, stats)
        jobs = [(cand, decision_seed * 1000 + k) for cand in candidates for k in range(args.samples)]
        if _W.get("cuda") is None:
            # The CPU reference scores one candidate at a time, terminal futures first.
            totals = []
            for cand in candidates:
                fixed, leaves = [], []
                for k in range(args.samples):
                    leaf = _rollout_policy(env, obs, cand, obs["tick"] + args.horizon,
                                           root, (decision_seed * 1000 + k) % 2 ** 32)
                    terminal = _score_terminal_or_value(leaf)
                    if terminal is None:
                        leaves.append(leaf)
                    else:
                        fixed.append(terminal)
                scored = fixed + _leaf_values(leaves)
                totals.append(sum(scored) / len(scored))
            best = max(range(len(candidates)), key=lambda i: totals[i])
            keys = [json.dumps(c, sort_keys=True) for c in candidates]
            return _verified_choice(env, obs, root, candidates, totals, [args.samples] * len(candidates),
                                    best, keys.index(policy_key), policy_action, args, decision_seed, stats)
        scores = _leaf_values(_continuations(env, obs, root, jobs, args, stats))
        totals = []
        for index in range(len(candidates)):
            scored = [v for v in scores[index * args.samples:(index + 1) * args.samples] if v is not None]
            totals.append(sum(scored) / len(scored) if scored else float("-inf"))
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
    keys = [json.dumps(c, sort_keys=True) for c in candidates]
    policy_index = keys.index(policy_key)
    values: list[list[float]] = [[] for _ in candidates]
    alive = list(range(len(candidates)))
    stats["halving_candidates"] = stats.get("halving_candidates", 0) + len(candidates)
    # Speculate all independent (candidate, future) continuations in one real batch. Later stages
    # still score only their original survivors, in the original order and batch sizes: unused
    # futures cannot affect pruning, labels or the anchored override rule. This spends extra GPU
    # work to remove device-wave dependencies, rather than merely combining protocol messages.
    speculative = None
    verification_cache = None
    if getattr(args, "cuda_speculative_halving", False) and _W.get("cuda") is not None:
        order = [(i, k) for i in alive for k in range(args.samples)]
        verification_order = ([(i, k) for i in alive
                               for k in range(args.samples, args.samples + args.verify_samples)]
                              if getattr(args, "cuda_prefetch_verify", False) else [])
        jobs = [(candidates[i], decision_seed * 1000 + k) for i, k in order + verification_order]
        handles = _continuations(env, obs, root, jobs, args, stats)
        speculative = dict(zip(order, handles[:len(order)]))
        if verification_order:
            # Prefetch futures, never scores: pruning and verification still evaluate
            # their original subsets, in their original order and value-net batches.
            verification_cache = dict(zip(verification_order, handles[len(order):]))
            stats["cuda_prefetch_verify_lanes"] = stats.get("cuda_prefetch_verify_lanes", 0) + len(verification_order)
        stats["halving_speculative_lanes"] = stats.get("halving_speculative_lanes", 0) + len(order)
    selected_lanes = 0
    done = 0                                 # futures every survivor has been scored on
    target = 1
    while True:
        target = min(target, args.samples)
        order = [(i, k) for i in alive for k in range(done, target)]
        selected_lanes += len(order)
        if speculative is None:
            jobs = [(candidates[i], decision_seed * 1000 + k) for i, k in order]
            handles = _continuations(env, obs, root, jobs, args, stats)
        else:
            handles = [speculative[key] for key in order]
        scored = list(zip(order, handles, _leaf_values(handles)))
        if _W.get("cuda") is None:
            # Match the original CPU accumulation order within each stage.
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
    if speculative is not None:
        stats["halving_discarded_lanes"] = stats.get("halving_discarded_lanes", 0) + len(speculative) - selected_lanes
    totals = [sum(v) / len(v) if v else float("-inf") for v in values]
    best = max(alive, key=lambda i: totals[i])
    kwargs = {"prefetched": verification_cache} if verification_cache is not None else {}
    used_before = stats.get("cuda_prefetch_verify_used_lanes", 0)
    chosen = _verified_choice(env, obs, root, candidates, totals, [len(v) for v in values], best,
                              policy_index, policy_action, args, decision_seed, stats, **kwargs)
    if verification_cache is not None:
        used = stats.get("cuda_prefetch_verify_used_lanes", 0) - used_before
        stats["cuda_prefetch_verify_discarded_lanes"] = stats.get("cuda_prefetch_verify_discarded_lanes", 0) + len(verification_cache) - used
    return chosen


def _verified_choice(env, obs: dict, root: int, candidates: list[dict], totals: list[float],
                     counts: list[int], best: int, policy_index: int, policy_action: dict,
                     args, decision_seed: int, stats: dict, *, fresh: bool = False, prefetched=None) -> dict:
    before = fresh or (best != policy_index and totals[best] - totals[policy_index] > args.margin)
    samples = getattr(args, "verify_samples", 0) + (args.samples if fresh else 0)
    if fresh:
        counts = [0] * len(candidates)
    if samples and before:
        prior = None if fresh else [totals[policy_index], totals[best]]
        started = time.perf_counter()
        if _W.get("cuda") is None:
            # Retain the original CPU value batches and terminal-first sum order. A different
            # order can change the last bits even though the continuation jobs are identical.
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
        else:
            # The extra futures are ordinary continuations: route them through _continuations so the GPU
            # arm runs them on the GPU too.  Running them on the C++ simulator here would make every
            # override pay 2 x --verify-samples CPU rollouts, which is exactly the cost the GPU arm
            # exists to remove (task4).  Job order is unchanged, so the C++ arm stays bit-identical.
            indexes = (policy_index, best)
            jobs = [(candidates[index], (decision_seed * 1000 + future) % 2 ** 32)
                    for index in indexes for future in range(args.samples, args.samples + samples)]
            if prefetched is None:
                handles = _continuations(env, obs, root, jobs, args, stats)
            else:
                handles = [prefetched[(index, future)] for index in indexes
                           for future in range(args.samples, args.samples + samples)]
                stats["cuda_prefetch_verify_used_lanes"] = stats.get("cuda_prefetch_verify_used_lanes", 0) + len(handles)
            for slot, index in enumerate(indexes):
                added = [value for value in _leaf_values(handles[slot * samples:(slot + 1) * samples])
                         if value is not None]
                if added:
                    totals[index] = (totals[index] * counts[index] + sum(added)) / (counts[index] + len(added))
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


def _frame_json(frame: dict) -> str:
    """One archive line: byte for byte what pvz_frame_store.write_frames serializes."""
    return json.dumps(frame, separators=(",", ":")) + "\n"


def _play(job: dict) -> dict:
    args = job["args"]
    _wait_for_gpu_pause(args, job["seed"])
    from collections import Counter
    import torch
    from pvz_event_env import EventWaitEnv
    from render_episode import _frame, load_training_task, training_task_kwargs
    from pvz_tasks import profile_for_deck
    from pvz_env import TaskSpec
    task_definition = load_training_task(Path(args.manifest), args.task_id)
    kw = training_task_kwargs(task_definition, job["seed"])
    # One simulator process per worker, reset for every game: starting the process and loading
    # resources costs ~0.7 s, as much as a whole student-only game. RESET_V2 rebuilds the board
    # from the seed, so a reused process plays the same game (scripts/perf_rollout_check.py).
    env = _W.get("env")
    if env is None:
        env = _W["env"] = EventWaitEnv(args.resource_dir, executable=getattr(args, "executable", None), headless=True)
        env.load_value_net(_W["value_files"][0])
        env.load_value_net(_W["value_files"][1], static=True)
        if _W.get("cpp_rollouts"):
            env.load_rollout_net(_W["rollout_file"])
        if _W.get("rollout_sim_skip", 0) & 256:
            env.require_sparse_tracks()
    task = TaskSpec(level=kw["level"], seed=kw["seed"], playthrough=2,
                    profile=profile_for_deck(tuple(kw["deck"]), kw.get("first_aid_owned", False)),
                    **{k: v for k, v in kw["task_extra"].items() if v is not None})
    torch.manual_seed(kw["seed"])
    started = time.time()
    stats = {"searched": 0, "overrides": 0, "shovels": 0, "override_kinds": Counter(),
             "student_candidate_added": 0, "student_candidate_best": 0, "student_candidate_override": 0}
    # The client counts per worker, not per game, so a game reports the delta it caused.
    cuda_client = _W.get("cuda")
    cuda_started_seconds = cuda_client.seconds if cuda_client is not None else 0.0
    try:
        obs, _ = env.reset(deck=tuple(kw["deck"]), task=task)
        from pvz_value_net import observation_context
        _W["context"] = observation_context(obs)
        task_context = {"level": kw["level"], "multiplier": obs.get("zombie_count_multiplier", 1.0),
                        "roster": (obs.get("loadout_context") or {}).get("zombie_roster") or []}
        decisions = 0
        # One JSON line per decision, exactly what write_frames would serialize -- kept as bytes
        # rather than as parsed dicts, so the archive costs 2.8 MB instead of ~18 MB of Python
        # objects for the whole game (task18).
        frames = [_frame_json(_frame(obs))] if args.frames else None
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
            # Planting on an occupied cell only became legal with wall-nut first aid (a damaged
            # wall-nut takes another nut of the same type, Board::CanPlantAt) -- record what stood
            # there, so the report can say how often the teacher actually used it. Harmless for
            # ordinary plants: the legal list never offers an occupied cell without first aid.
            if action["type"] == "plant":
                occupant = next((p for p in obs.get("plants") or []
                                 if p["row"] == action["row"] and p["col"] == action["col"]), None)
                if occupant is not None:
                    stats.setdefault("replant_events", []).append({
                        "tick": obs["tick"], "row": action["row"], "col": action["col"],
                        "packet": action["packet"], "occupant": occupant.get("type"),
                        "occupant_health": occupant.get("health"),
                        "same_type": occupant.get("type") == action["packet"]})
            obs, _r, done, _t, info = env.step(action)
            rejected = not info.get("ok")
            if not rejected and action["type"] == "plant":
                planted[action["packet"]] = planted.get(action["packet"], 0) + 1
            if rejected:
                obs, _r, done, _t, info = env.step({"type": "wait", "ticks": 60})
            if frames is not None:
                frame = _frame(obs)
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
                if stats.get("last_verification"):
                    # Wall time is not part of the label: keeping it in the archive makes two runs of
                    # the same arm differ byte for byte (and the GPU schedule comparison impossible).
                    verification = dict(stats["last_verification"])
                    if getattr(args, "rollout_backend", "cpp") == "cuda":
                        verification.pop("seconds", None)
                    frame["verification"] = verification
                if dagger is not None:
                    frame["label"] = label        # last field: train_rollout_policy scans for it
                frames.append(_frame_json(frame))
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
        from pvz_frame_store import write_frames_from_source
        write_frames_from_source(archive / f"frames.{getattr(args, 'frames_format', 'npz')}",
                                 "".join(frames).encode())
        archive_task = {"seed": job["seed"], "policy": str(args.rollout_policy), "arm": job["arm"]}
        archive_outcome = {"won": obs.get("result") == 1, "final_wave": obs["wave"]}
        if 'deck_perturb' in task_definition or task_definition.get('conveyor_start') == 'original':
            import hashlib
            archive_task.update(level=kw['level'], deck=list(kw['deck']), playthrough=2,
                                policy='student', checkpoint=str(Path(args.rollout_policy).resolve()),
                                checkpoint_sha256=hashlib.sha256(Path(args.rollout_policy).read_bytes()).hexdigest(),
                                task_id=args.task_id, first_aid_owned=kw.get('first_aid_owned', False),
                                **{k: v for k, v in kw['task_extra'].items() if v is not None})
            if 'deck_perturb' in task_definition:
                archive_task['deck_perturb'] = task_definition['deck_perturb']
                archive_task['base_deck'] = task_definition['deck']
            archive_outcome.update(final_tick=obs['tick'], wave_count=_frame(obs)['scene'].get('wave_count', 0),
                                   reason='win' if obs.get('result') == 1 else 'not_won')
        (archive / "meta.json").write_text(json.dumps({
            "task": archive_task,
            "context": task_context,
            "outcome": archive_outcome}))
    result = {"seed": job["seed"], "arm": job["arm"], "won": obs.get("result") == 1,
            "final_wave": obs["wave"], "decisions": decisions, "seconds": round(time.time() - started, 1),
            "mowers_used": mowers_start - mowers_left, "planted": planted,
            "searched": stats["searched"], "overrides": stats["overrides"],
            "cuda_coverage_version": 1,
            **{name: stats.get(name, 0) for name in ("rollout_batches", "rollout_lanes", "cuda_batches",
                "cuda_lanes", "cuda_error_lanes", "cuda_fallback_batches", "cuda_fallback_lanes",
                "halving_candidates", "halving_rollouts", "halving_selected_lanes",
                "halving_speculative_lanes", "halving_discarded_lanes")},
            "cuda_seconds": round(cuda_client.seconds - cuda_started_seconds, 2) if cuda_client is not None else 0.0,
            **({name: stats.get(name, 0) for name in ("cuda_prefetch_verify_lanes",
                "cuda_prefetch_verify_used_lanes", "cuda_prefetch_verify_discarded_lanes")}
               if getattr(args, "cuda_prefetch_verify", False) else {}),
            "student_candidate_added": stats["student_candidate_added"],
            "student_candidate_best": stats["student_candidate_best"],
            "student_candidate_override": stats["student_candidate_override"],
            "shovel_events": stats.get("shovel_events", []),
            "replant_events": stats.get("replant_events", []),
            "override_kinds": dict(stats["override_kinds"])}
    if getattr(args, "candidate_coverage_model", None):
        result["coverage_candidate_added"] = stats.get("coverage_candidate_added", 0)
    if getattr(args, "measure_cpu", False):
        from pvz_cpu_usage import usage, process_usage, difference
        worker = usage()
        simulator = process_usage(env._process.pid)
        zero = {"user_s": 0.0, "system_s": 0.0}
        result["host_cpu"] = {
            "worker": difference(worker, _W.get("cpu_accounted", zero)),
            "simulator": difference(simulator, _W.get("sim_cpu_accounted", zero))}
        _W["cpu_accounted"], _W["sim_cpu_accounted"] = worker, simulator
    _release_freed_memory()      # this game's archives are written; its buffers are garbage now
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rollout-policy", required=True, help="the student (scripts/train_rollout_policy.py)")
    p.add_argument("--value", required=True, help="value net that scores the leaves")
    p.add_argument("--static-value", help="value net that picks the candidate cells right after planting "
                   "(default --value)")
    p.add_argument("--manifest", required=True)
    p.add_argument("--task-id", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--measure-cpu", action="store_true",
                   help="record host user/system CPU seconds for workers, simulators and owned CUDA service")
    p.add_argument("--seeds", type=int, default=64)
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument("--arms", default="fast_search", help="fast (the student alone), fast_search (the teacher)")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--per-card", type=int, default=1, help="best cells of each card kept as candidates")
    p.add_argument("--candidate-generator", choices=["static", "student-topk", "student-per-card"],
                   default="static", help="candidate ablation, default static (unchanged); student-topk uses "
                   "--student-candidates as a global prefix, without a per-card floor; student-per-card "
                   "uses student probabilities for per-card cells plus the usual additional student candidates. "
                   "Both student modes skip the static net and require shovel/coverage candidates off")
    p.add_argument("--student-candidates", type=int, default=0,
                   help="the student's own top-k plant actions (its policy probabilities) added to the "
                        "candidates; 0 = off, which is the old candidate set exactly")
    p.add_argument("--candidate-coverage-model", help="optional learned candidate proposer checkpoint; default off")
    p.add_argument("--candidate-coverage-budget", type=int, default=2,
                   help="maximum additional legal plants per search when the coverage model is enabled")
    p.add_argument("--shovel-candidates", type=int, default=0,
                   help="best legal shovel cells (scored by --static-value right after shovelling) added to "
                        "the candidates; 0 = off, which is the old candidate set exactly")
    p.add_argument("--samples", type=int, default=3)
    p.add_argument("--halving", action=argparse.BooleanOptionalAction, default=True,
                   help="successive halving over the candidates (1, 2, 4.. futures); --no-halving = every "
                        "candidate gets all --samples")
    p.add_argument("--cuda-prefetch-verify", action="store_true",
                   help="prefetch verification futures in the speculative GPU wave; spends extra lanes "
                        "to avoid a serial verification request, preserving scoring and selection")
    p.add_argument("--cuda-speculative-halving", action="store_true",
                   help="batch all candidate/future continuations once, then use only the original "
                        "halving survivors for scores and labels; spends extra CUDA work")
    p.add_argument("--horizon", type=int, default=2500)
    p.add_argument("--margin", type=float, default=0.02)
    p.add_argument("--verify-samples", type=int, default=0,
                   help="extra independent futures for student and challenger before an override (0 = disabled)")
    p.add_argument("--min-gap", type=int, default=300,
                   help="ticks between searches unless the student itself wants to plant (0 = search every time)")
    p.add_argument("--rollout-sim-skip", type=int, default=61,
                   help="SIM_SKIP mask during rollouts only (0 = exact; see scripts/sim_skip_agreement.py)")
    p.add_argument("--cpu-sparse-tracks", action="store_true",
                   help="CPU-only: skip inactive animation tracks, preserving mask61 results; requires a rebuilt simulator")
    p.add_argument("--cpp-rollouts", action=argparse.BooleanOptionalAction, default=True,
                   help="run student continuations in C++ (--no-cpp-rollouts: the Python reference loop)")
    p.add_argument("--rollout-backend", choices=["cpp", "cuda"], default="cpp",
                   help="where the search's continuations run: cpp = the original simulator (default, and the "
                        "reference the GPU is checked against), cuda = build/rollout_cuda --serve")
    p.add_argument("--executable", default=str(ROOT / "build/pvz-portable"),
                   help="fixed original-simulator binary used for roots and real game decisions")
    p.add_argument("--cuda-binary", default=str(ROOT / "build/rollout_cuda"))
    p.add_argument("--cuda-socket", default=None, help="an already-running server; by default start one here")
    p.add_argument("--cuda-lanes", type=int, default=2048,
                   help="the server's lane capacity (about 113 MB of VRAM per 1024 lanes)")
    p.add_argument("--cuda-gather-ms", type=int, default=80,
                   help="how long the server holds a wave open for other workers before running it; a wave "
                        "costs its rounds, not its lanes, so merging is what pays (0 = no waiting)")
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
    if args.cpu_sparse_tracks:
        args.rollout_sim_skip |= 256
    if args.rollout_sim_skip & 256 and (
            args.rollout_backend != "cpp" or args.cuda_socket is not None or not args.cpp_rollouts
            or not args.rollout_sim_skip & 32):
        p.error("CPU sparse tracks require --cpp-rollouts, backend cpp, no CUDA socket and LIGHT_ANIMATION bit32")
    if args.rollout_backend == "cuda" and os.environ.get("PVZ_GPU_PAUSE_FILE"):
        args.gpu_pause_file = str(Path(os.environ["PVZ_GPU_PAUSE_FILE"]).expanduser().resolve())
    if args.cuda_speculative_halving and (not args.halving or args.rollout_backend != "cuda"):
        p.error("--cuda-speculative-halving requires --halving and --rollout-backend cuda")
    if args.cuda_prefetch_verify and (not args.cuda_speculative_halving or args.verify_samples <= 0):
        p.error("--cuda-prefetch-verify requires --cuda-speculative-halving and positive --verify-samples")
    if args.verify_samples < 0:
        p.error("--verify-samples must be nonnegative")
    if args.shovel_candidates < 0:
        p.error("--shovel-candidates must be nonnegative")
    if args.student_candidates < 0:
        p.error("--student-candidates must be nonnegative")
    if args.candidate_coverage_budget < 0:
        p.error("--candidate-coverage-budget must be nonnegative")
    if args.candidate_generator != "static":
        if args.shovel_candidates or args.candidate_coverage_model:
            p.error("student candidate generators require shovel/coverage candidates off")
        if args.candidate_generator == "student-topk" and args.student_candidates <= 0:
            p.error("student-topk requires positive --student-candidates")
        if args.candidate_generator == "student-per-card" and args.per_card <= 0:
            p.error("student-per-card requires positive --per-card")

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
    cuda_socket, cuda_server, cuda_keep = args.cuda_socket, None, None
    if args.rollout_backend == "cuda" and cuda_socket is None:
        # The server is the parent's child so it outlives every worker and dies with the run; the workers
        # hold nothing but a socket to it.  Its own stderr goes to a file because the lanes-per-wave it
        # achieved is the number this seam exists for.
        import tempfile
        from pvz_cuda_rollout import serve_policy
        cuda_keep = tempfile.TemporaryDirectory(prefix="pvz-cuda-")
        with (args.out / "cuda_server.log").open("wb") as log:
            cuda_server, cuda_socket = serve_policy(args.cuda_binary, args.rollout_policy, cuda_keep.name,
                                                    lanes=args.cuda_lanes, gather_ms=args.cuda_gather_ms,
                                                    start_timeout_ms=600000, log=log)
        print(f"cuda rollout server: {cuda_socket}, {args.cuda_lanes} lanes", flush=True)
        (args.out / "cuda_server.json").write_text(json.dumps({"socket": cuda_socket, "lanes": args.cuda_lanes}) + "\n")
    value_keep, value_files = _export_values(args.value, args.static_value)
    _release_freed_memory()
    # Student weights remain shared on fork; value weights live only in engine processes.
    start_method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
    if start_method == "fork":
        _load_nets(value_files[0], args.rollout_policy, value_files[1], args.cpp_rollouts,
                   args.candidate_coverage_model, args.candidate_coverage_budget)
        _freeze_startup()
    if args.measure_cpu:
        import resource
        from pvz_cpu_usage import usage, difference, wait_with_cpu_usage
        children_before = usage(resource.RUSAGE_CHILDREN)
    server_cpu = None
    started = time.time()
    # A worker's resident size only ever grows -- macOS keeps freed pages in the process (task18).
    # A long-lived worker therefore pays for the high-water of every game it played, so where the
    # platform can fork, recycle the process after each game: the next one starts from the parent's
    # copy-on-write baseline instead of on top of the last game's peak. Same games, same bytes.
    recycle = 1 if start_method == "fork" else None
    try:
        with results_path.open("a") as out, mp.get_context(start_method).Pool(
                args.workers, initializer=_init_worker, maxtasksperchild=recycle,
                initargs=(value_files[0], args.rollout_policy, args.rollout_sim_skip, args.cpp_rollouts,
                          value_files[1], cuda_socket) +
                         ((args.candidate_coverage_model, args.candidate_coverage_budget)
                          if args.candidate_coverage_model else ())) as pool:
            for i, res in enumerate(pool.imap_unordered(_play, jobs), 1):
                out.write(json.dumps(res) + "\n"); out.flush()
                if i % 10 == 0 or i == len(jobs):
                    print(f"{i}/{len(jobs)} ({time.time() - started:.0f}s)", flush=True)
            # A graceful join runs worker finalizers and reaps their simulator children. The parent's
            # RUSAGE_CHILDREN now includes all worker startup, games and teardown, exactly once.
            pool.close()
            pool.join()
    finally:
        value_keep.cleanup()
        if args.measure_cpu:
            workers_cpu = difference(usage(resource.RUSAGE_CHILDREN), children_before)
        if cuda_server is not None:
            try:
                # The workers closing their sockets is what shuts it down; only wait for that before
                # stopping it ourselves, because a server that outlives the run holds its VRAM.
                if args.measure_cpu:
                    server_cpu = wait_with_cpu_usage(cuda_server)
                else:
                    cuda_server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                cuda_server.terminate()
                if args.measure_cpu:
                    server_cpu = wait_with_cpu_usage(cuda_server)
                else:
                    cuda_server.wait(timeout=30)
            print("cuda server: " + (args.out / "cuda_server.log").read_text(errors="replace").strip(),
                  flush=True)
        if cuda_keep is not None:
            cuda_keep.cleanup()
    if args.measure_cpu:
        driver_cpu = difference(usage(), {"user_s": 0.0, "system_s": 0.0})
        total = workers_cpu["total_s"] + driver_cpu["total_s"] + (server_cpu["total_s"] if server_cpu else 0)
        (args.out / "cpu_usage.json").write_text(json.dumps({
            "method": "getrusage(RUSAGE_CHILDREN) after graceful worker/simulator joins; wait4 owned service",
            "games": len(jobs), "workers_and_simulators": workers_cpu, "owned_service": server_cpu,
            "driver": driver_cpu, "total_host_cpu_s": total,
            "host_cpu_s_per_game": total / len(jobs) if jobs else None,
            "external_service_excluded": bool(args.cuda_socket)}, indent=1) + "\n")
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
                     # With --rollout-backend cuda these say what the GPU actually got to chew on: lanes per
                     # request is the batching, and cuda_seconds/seconds is how much of the game was waiting
                     # for it.
                     "cuda_lanes_per_game": round(sum(r.get("cuda_lanes", 0) for r in v.values()) / len(v), 1),
                     "cuda_batches_per_game": round(sum(r.get("cuda_batches", 0) for r in v.values()) / len(v), 1),
                     "cuda_seconds_per_game": round(sum(r.get("cuda_seconds", 0.0) for r in v.values()) / len(v), 1),
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
