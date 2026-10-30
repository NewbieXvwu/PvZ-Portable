#!/usr/bin/env python3
"""Pure-student win-rate benchmark with two execution paths.

The promotion gate answers "is the candidate non-inferior to the incumbent"; this tool
answers "how often does this student actually win, where, and how does it compare with
another student". No search, no value net: every decision is the student's own greedy
action (``--arms fast`` semantics in scripts/search_play.py).

Paths
-----
cpp      export the checkpoint with pvz_rollout_policy.export_student and play the whole
         game inside the C++ simulator (``PvZEnv.rollout(None, 0)``). This is the fast,
         production-shaped path and only accepts RolloutNet / PVZRN006 checkpoints.
python   play in the Python decision loop (``pvz_rollout_policy.act`` + ``env.step``).
         Slower, but it is the seam for structures that cannot be exported yet and the
         reference the native path is checked against.
auto     cpp when the checkpoint exports, python otherwise.

Panel and seeds
---------------
The default panel is configs/gate_v5_panel.json (the same adventure panel the gate runs).
A subset is allowed but must keep all five terrains (day, night, pool, fog, roof) and the
deck-heldout cells, because a subset that drops a terrain cannot say anything about it.
Seeds are per task and identical for every model, so two runs of the same panel are paired.
Use --seeds-from <decision.json|manifest.json> to replay the exact gate seeds of a round.

Statistics
----------
Per task, per terrain and overall: wins, games, win rate and a 95% Wilson interval.
``--compare A B`` pairs two run directories by (task_id, seed) and reports the paired
contingency table, the Newcombe interval of the win-rate difference (the gate's interval)
and an exact two-sided McNemar p-value, per terrain as well as overall.

Resume: games.jsonl is appended as games finish; a rerun with the same --out and the same
model/backend/panel skips completed (task, seed) pairs, never repairs corrupt rows.

Usage:
  python scripts/bench_play.py --model artifacts/student/model.pt --out artifacts/bench/s1
  python scripts/bench_play.py --model m.pt --tasks task101_l7_day_d7,task101_l19_night_n10 \
      --per-task 4 --workers 4 --out artifacts/bench/small
  python scripts/bench_play.py --model m.pt --seeds-from /path/to/decision.json \
      --tasks task101_l7_day_d7 --out artifacts/bench/r36_l7
  python scripts/bench_play.py --compare artifacts/bench/a artifacts/bench/b
  python scripts/bench_play.py --model m.pt --verify-parity --tasks task101_l7_day_d7 \
      --per-task 1 --out artifacts/bench/parity
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import sys
import time
from math import comb
from pathlib import Path
from statistics import NormalDist

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

DEFAULT_PANEL = ROOT / "configs/gate_v5_panel.json"
DEFAULT_EXECUTABLE = ROOT / "build/pvz-portable"
DEFAULT_RESOURCE_DIR = Path(os.environ.get(
    "PVZ_RESOURCE_DIR", "/Users/newbiexvwu/Downloads/Plants_Vs_Zombies_V1.2.0.1073_EN"))
REQUIRED_ZONES = ("day", "night", "pool", "fog", "roof")
HELDOUT_ZONE = "heldout"
MAX_ACTIONS = 3000
MAX_WORKERS = 8
# Bench-only scratch seed segment: above every registered pool (gate 100M, start 200M),
# so a benchmark can never consume or collide with a gate ledger block.
DEFAULT_SEED_START = 900_000_000
WAIT = {"type": "wait", "ticks": 60}

_W: dict = {}


# --------------------------------------------------------------------------- stats

def wilson_interval(wins: int, total: int, alpha: float = 0.05) -> list[float]:
    """95% Wilson score interval for a single win rate (same formula as pvz_promotion)."""
    z = NormalDist().inv_cdf(1 - alpha / 2)
    p, z2 = wins / total, z * z
    center = (p + z2 / (2 * total)) / (1 + z2 / total)
    radius = z * math.sqrt(p * (1 - p) / total + z2 / (4 * total * total)) / (1 + z2 / total)
    return [round(max(0.0, center - radius), 6), round(min(1.0, center + radius), 6)]


def win_summary(rows: list[dict]) -> dict:
    games = len(rows)
    if not games:
        return {"games": 0, "wins": 0, "win_rate": None, "wilson95": None}
    wins = sum(bool(row["won"]) for row in rows)
    return {"games": games, "wins": wins, "win_rate": round(wins / games, 6),
            "wilson95": wilson_interval(wins, games)}


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Exact two-sided McNemar p-value over the discordant pairs."""
    discordant = only_a + only_b
    if not discordant:
        return 1.0
    tail = sum(comb(discordant, i) for i in range(min(only_a, only_b) + 1))
    return min(1.0, 2 * tail / 2 ** discordant)


def paired_summary(rows: list[tuple[bool, bool]]) -> dict:
    """Paired (a, b) outcomes -> counts, delta, Newcombe interval and McNemar p."""
    from pvz_promotion import paired_interval
    both_win = sum(1 for a, b in rows if a and b)
    both_lose = sum(1 for a, b in rows if not a and not b)
    only_a = sum(1 for a, b in rows if a and not b)
    only_b = sum(1 for a, b in rows if b and not a)
    pairs = len(rows)
    low, high = paired_interval(both_win, only_a, only_b, both_lose)
    return {"pairs": pairs, "a_wins": both_win + only_a, "b_wins": both_win + only_b,
            "both_win": both_win, "a_only": only_a, "b_only": only_b, "both_lose": both_lose,
            "delta": round((only_a - only_b) / pairs, 6),
            "delta_ci95": [round(low, 6), round(high, 6)],
            "mcnemar_p": round(mcnemar_exact(only_a, only_b), 6)}


def summarize_run(rows: list[dict]) -> dict:
    zones = sorted({row["zone"] for row in rows})
    tasks = sorted({row["task_id"] for row in rows})
    return {"overall": win_summary(rows),
            "zones": {zone: win_summary([r for r in rows if r["zone"] == zone]) for zone in zones},
            "tasks": {task: win_summary([r for r in rows if r["task_id"] == task]) for task in tasks}}


# --------------------------------------------------------------------------- panel/seeds

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_panel(path: Path) -> tuple[dict, list[dict]]:
    document = json.loads(Path(path).read_text())
    if document.get("panel_version") != "v5":
        raise ValueError(f"{path} is not the gate v5 panel (panel_version="
                         f"{document.get('panel_version')!r})")
    table = {task["task_id"]: task for task in document["tasks"]}
    tasks = []
    for task_id, zone in document["groups"].items():
        if task_id not in table:
            raise ValueError(f"panel group {task_id!r} has no task definition")
        tasks.append({**table[task_id], "zone": zone})
    return document, tasks


def select_tasks(panel_tasks: list[dict], ids: str | None, zones: str | None,
                 allow_partial: bool) -> list[dict]:
    by_id = {task["task_id"]: task for task in panel_tasks}
    selected = panel_tasks
    if ids:
        wanted = [name.strip() for name in ids.split(",") if name.strip()]
        unknown = [name for name in wanted if name not in by_id]
        if unknown:
            raise ValueError(f"tasks not in the panel: {unknown[:5]}")
        selected = [by_id[name] for name in wanted]
    elif zones:
        wanted = {name.strip() for name in zones.split(",") if name.strip()}
        unknown = wanted - {task["zone"] for task in panel_tasks}
        if unknown:
            raise ValueError(f"zones not in the panel: {sorted(unknown)}")
        selected = [task for task in panel_tasks if task["zone"] in wanted]
    found = {task["zone"] for task in selected}
    missing = [zone for zone in REQUIRED_ZONES if zone not in found]
    if HELDOUT_ZONE not in found:
        missing.append(HELDOUT_ZONE)
    if missing and not allow_partial:
        raise ValueError(f"the selected panel has no cells for {missing}; the benchmark needs all "
                         f"five terrains and the deck heldout (pass --allow-partial to override)")
    if len({task["task_id"] for task in selected}) != len(selected):
        raise ValueError("duplicate task ids in the selected panel")
    return sorted(selected, key=lambda task: task["task_id"])


def plan_seeds(tasks: list[dict], per_task: int, seed_start: int) -> dict[str, list[int]]:
    if per_task < 1:
        raise ValueError("--per-task must be positive")
    plan = {}
    for index, task in enumerate(tasks):
        first = seed_start + index * per_task
        plan[task["task_id"]] = list(range(first, first + per_task))
    flat = [seed for seeds in plan.values() for seed in seeds]
    if any(not 0 <= seed < 2 ** 32 for seed in flat):
        raise ValueError("seed plan falls outside the unsigned 32-bit seed range")
    return plan


def seeds_from(path: Path, tasks: list[dict]) -> dict[str, list[int]]:
    """Read the exact per-task seeds of a gate decision or frozen manifest."""
    document = json.loads(Path(path).read_text())
    selected = {task["task_id"] for task in tasks}
    plan: dict[str, set[int]] = {}
    if isinstance(document.get("paired"), list):
        for row in document["paired"]:
            if row.get("task_id") in selected:
                plan.setdefault(row["task_id"], set()).add(int(row["seed"]))
    elif isinstance(document.get("tasks"), list):
        for task in document["tasks"]:
            if task.get("task_id") in selected:
                plan.setdefault(task["task_id"], set()).update(int(seed) for seed in task.get("seeds") or ())
    else:
        raise ValueError(f"{path} is neither a gate decision (paired rows) nor a frozen manifest")
    missing = sorted(selected - set(plan))
    if missing:
        raise ValueError(f"{path} has no seeds for {missing}")
    return {task_id: sorted(seeds) for task_id, seeds in plan.items()}


def build_jobs(tasks: list[dict], plan: dict[str, list[int]]) -> list[dict]:
    return [{"task_id": task["task_id"], "seed": seed}
            for task in tasks for seed in plan[task["task_id"]]]


# --------------------------------------------------------------------------- policy seam

def load_policy_module(module_path: str | None):
    if not module_path:
        return None
    import importlib.util
    spec = importlib.util.spec_from_file_location("bench_policy", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_policy(path: str, module=None):
    if module is not None:
        return module.load_policy(path)
    from pvz_rollout_policy import load_student
    return load_student(path)


def policy_act(policy, frame, previous, legal, *, context=None, history=None, module=None):
    if module is not None:
        return module.act(policy, frame, previous, legal, context=context, history=history)
    from pvz_rollout_policy import act
    return act(policy, frame, previous, legal, context=context, history=history)


def policy_obs(policy) -> str:
    return getattr(policy, "obs", "v1")


def policy_context_enabled(policy) -> bool:
    return bool(getattr(policy, "context", False))


# --------------------------------------------------------------------------- workers

def _init_worker(cfg: dict) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    from pvz_event_env import EventWaitEnv
    _W["cfg"] = cfg
    _W["env"] = EventWaitEnv(cfg["resource_dir"], executable=Path(cfg["executable"]), headless=True)
    if cfg["backend"] in ("cpp", "parity"):
        _W["env"].load_rollout_net(cfg["native"])
    if cfg["backend"] in ("python", "parity"):
        import torch
        torch.set_num_threads(1)
        _W["module"] = load_policy_module(cfg.get("policy_module"))
        _W["policy"] = load_policy(cfg["model"], _W["module"])


def _task_spec(task_definition: dict, seed: int):
    from pvz_env import TaskSpec
    from pvz_tasks import profile_for_deck
    from render_episode import training_task_kwargs
    kw = training_task_kwargs(task_definition, seed)
    return TaskSpec(level=kw["level"], seed=kw["seed"],
                    playthrough=int(task_definition.get("playthrough", 2)),
                    profile=profile_for_deck(tuple(kw["deck"]), kw.get("first_aid_owned", False)),
                    **{k: v for k, v in kw["task_extra"].items() if v is not None}), kw["deck"]


def _ready_mowers(obs: dict) -> int:
    return sum(1 for defense in obs.get("defenses") or [] if defense["state"] == 1)


def _row(task: dict, seed: int, obs: dict, started: float, decisions: int,
         mowers_start: int, mowers_left: int, backend: str) -> dict:
    return {"task_id": task["task_id"], "zone": task["zone"], "level": task["level"],
            "seed": seed, "won": obs.get("result") == 1, "terminal": bool(obs.get("terminal")),
            "backend": backend, "decisions": decisions,
            "seconds": round(time.time() - started, 3), "final_wave": obs.get("wave"),
            "final_tick": obs.get("tick"), "mowers_used": mowers_start - mowers_left}


def _play_cpp(task: dict, seed: int, env) -> dict:
    started = time.time()
    spec, deck = _task_spec(task, seed)
    obs, _ = env.reset(deck=tuple(deck), task=spec)
    mowers_start = _ready_mowers(obs)
    obs = env.rollout(None, 0, max_actions=MAX_ACTIONS)
    details = getattr(env, "_last_response", {}) or {}
    return _row(task, seed, obs, started, int(details.get("decisions") or 0),
                mowers_start, int(details.get("mowers_left", mowers_start)), "cpp")


def _play_python(task: dict, seed: int, env, policy) -> dict:
    from pvz_value_net import observation_context
    from pvz_student_observation import StudentHistory
    from render_episode import _frame
    started = time.time()
    spec, deck = _task_spec(task, seed)
    obs, _ = env.reset(deck=tuple(deck), task=spec)
    mowers_start = mowers_left = _ready_mowers(obs)
    context = observation_context(obs) if policy_context_enabled(policy) else None
    history = StudentHistory(policy_obs(policy)) if policy_obs(policy) != "v1" else None
    state = {}
    decisions = 0
    while not obs.get("terminal") and decisions < MAX_ACTIONS:
        frame = _frame(obs)
        frame["action"] = state.get("action")
        legal = (obs.get("legal_actions") or {}).get("plants") or []
        action = policy_act(policy, frame, state.get("frame"), legal,
                            context=context, history=history, module=_W.get("module"))
        state["frame"], state["action"] = frame, action
        obs, _reward, done, _truncated, info = env.step(action)
        if not info.get("ok"):
            obs, _reward, done, _truncated, info = env.step(WAIT)
        decisions += 1
        if not obs.get("terminal"):
            mowers_left = _ready_mowers(obs)
        if done:
            break
    return _row(task, seed, obs, started, decisions, mowers_start, mowers_left, "python")


def _play_one(job: dict) -> dict:
    cfg = _W["cfg"]
    task = cfg["tasks"][job["task_id"]]
    env = _W["env"]
    if cfg["backend"] == "cpp":
        return _play_cpp(task, job["seed"], env)
    return _play_python(task, job["seed"], env, _W["policy"])


def _parity_one(job: dict) -> dict:
    """Same task/seed through both paths; compare every Python-visited decision with native."""
    from pvz_value_net import observation_context
    from pvz_student_observation import StudentHistory
    from render_episode import _frame
    cfg = _W["cfg"]
    task, seed, env, policy = cfg["tasks"][job["task_id"]], job["seed"], _W["env"], _W["policy"]
    spec, deck = _task_spec(task, seed)

    started = time.time()
    obs, _ = env.reset(deck=tuple(deck), task=spec)
    cpp_obs = env.rollout(None, 0, max_actions=MAX_ACTIONS)
    details = getattr(env, "_last_response", {}) or {}
    cpp = {"won": cpp_obs.get("result") == 1, "decisions": int(details.get("decisions") or 0),
           "tick": cpp_obs.get("tick"), "seconds": round(time.time() - started, 3)}

    started = time.time()
    obs, _ = env.reset(deck=tuple(deck), task=spec)
    context = observation_context(obs) if policy_context_enabled(policy) else None
    history = StudentHistory(policy_obs(policy)) if policy_obs(policy) != "v1" else None
    state = {}
    decisions = compared = 0
    first_mismatch = None
    while not obs.get("terminal") and decisions < MAX_ACTIONS:
        frame = _frame(obs)
        frame["action"] = state.get("action")
        legal = (obs.get("legal_actions") or {}).get("plants") or []
        action = policy_act(policy, frame, state.get("frame"), legal, context=context,
                            history=history, module=_W.get("module"))
        previous = state.get("frame")
        previous_action = state.get("action")
        native = env.rollout_predict(previous["tick"] if previous else None,
                                     bool(previous_action and previous_action.get("type") == "plant"),
                                     student_history=history)
        compared += 1
        if first_mismatch is None and native.get("action") != action:
            first_mismatch = {"decision": decisions, "tick": obs.get("tick"),
                              "python": action, "native": native.get("action")}
        state["frame"], state["action"] = frame, action
        obs, _reward, done, _truncated, info = env.step(action)
        if not info.get("ok"):
            obs, _reward, done, _truncated, info = env.step(WAIT)
        decisions += 1
        if done:
            break
    python = {"won": obs.get("result") == 1, "decisions": decisions,
              "tick": obs.get("tick"), "seconds": round(time.time() - started, 3)}
    return {"task_id": task["task_id"], "zone": task["zone"], "seed": seed,
            "cpp": cpp, "python": python, "actions_compared": compared,
            "first_mismatch": first_mismatch,
            "outcome_match": (cpp["won"] == python["won"] and cpp["decisions"] == python["decisions"]
                              and cpp["tick"] == python["tick"])}


# --------------------------------------------------------------------------- run/compare

def _write(path: Path, document) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".building")
    temporary.write_text(json.dumps(document, indent=1) + "\n")
    temporary.replace(path)


def _read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _resolve_backend(model: Path, choice: str, parity: bool, module: str | None):
    """Return (backend, native_export_path_or_None); auto prefers the C++ path when exportable."""
    if parity:
        return "parity", _export(model, module)
    if module:
        if choice == "cpp":
            raise ValueError("--backend cpp requires an exportable RolloutNet checkpoint, not --policy-module")
        return "python", None
    if choice in ("auto", "cpp"):
        try:
            return "cpp", _export(model, None)
        except Exception as exc:
            if choice == "cpp":
                raise
            print(f"C++ export unavailable ({exc}); using the Python path", flush=True)
    return "python", None


def _export(model: Path, module: str | None) -> Path:
    if module:
        raise ValueError("a custom policy module cannot be exported to C++ yet")
    import tempfile
    from pvz_rollout_policy import export_student
    policy = load_policy(str(model))
    directory = Path(tempfile.mkdtemp(prefix="bench-play-"))
    target = directory / "student.bin"
    export_student(policy, target)
    return target


def _plan(args, tasks: list[dict]) -> dict[str, list[int]]:
    if args.seeds_from:
        return seeds_from(args.seeds_from, tasks)
    return plan_seeds(tasks, args.per_task, args.seed_start)


def _run(args) -> int:
    model = Path(args.model).expanduser().resolve()
    panel = Path(args.panel).expanduser().resolve()
    executable = Path(args.executable).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    if not model.is_file():
        raise SystemExit(f"model not found: {model}")
    if not executable.is_file():
        raise SystemExit(f"simulator not found: {executable}")
    if not (Path(args.resource_dir) / "main.pak").is_file():
        raise SystemExit(f"main.pak not found in {args.resource_dir}")
    _, panel_tasks = load_panel(panel)
    tasks = select_tasks(panel_tasks, args.tasks, args.zones, args.allow_partial)
    plan = _plan(args, tasks)
    jobs = build_jobs(tasks, plan)
    backend, native = _resolve_backend(model, args.backend, args.verify_parity, args.policy_module)
    model_sha = sha256_file(model)
    config = {"model": str(model), "model_sha256": model_sha, "backend": backend,
              "policy_module": args.policy_module, "panel": str(panel),
              "panel_sha256": sha256_file(panel), "executable": str(executable),
              "executable_sha256": sha256_file(executable),
              "tasks": {task["task_id"]: {"zone": task["zone"], "seeds": plan[task["task_id"]]}
                        for task in tasks},
              "workers": args.workers, "verify_parity": bool(args.verify_parity)}
    out.mkdir(parents=True, exist_ok=True)
    config_path = out / "config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        key = ("model_sha256", "backend", "panel_sha256", "tasks", "policy_module")
        if any(previous.get(name) != config.get(name) for name in key):
            raise SystemExit(f"{out} holds a different run (see {config_path}); use a new --out")
    else:
        _write(config_path, config)
    results_path = out / ("parity.jsonl" if args.verify_parity else "games.jsonl")
    done = {(row["task_id"], row["seed"]) for row in _read_rows(results_path)}
    if done:
        planned = {(job["task_id"], job["seed"]) for job in jobs}
        stale = done - planned
        if stale:
            raise SystemExit(f"{results_path} has rows outside the planned jobs, e.g. {sorted(stale)[:3]}")
        for row in _read_rows(results_path):
            if row.get("backend") not in (backend, "cpp", "python"):
                raise SystemExit(f"row with unknown backend in {results_path}")
    pending = [job for job in jobs if (job["task_id"], job["seed"]) not in done]
    print(f"{len(jobs)} games, {len(pending)} pending, backend={backend}, workers={args.workers}",
          flush=True)

    worker_cfg = {"backend": backend, "model": str(model), "native": str(native) if native else None,
                  "policy_module": args.policy_module, "resource_dir": str(args.resource_dir),
                  "executable": str(executable), "tasks": {task["task_id"]: task for task in tasks}}
    start_method = "spawn" if sys.platform == "darwin" else (
        "fork" if "fork" in mp.get_all_start_methods() else "spawn")
    started = time.time()
    function = _parity_one if args.verify_parity else _play_one
    if pending:
        context = mp.get_context(start_method)
        pool = context.Pool(args.workers, initializer=_init_worker, initargs=(worker_cfg,))
        try:
            with results_path.open("a") as stream:
                for index, row in enumerate(pool.imap_unordered(function, pending, chunksize=1), 1):
                    stream.write(json.dumps(row) + "\n")
                    stream.flush()
                    if args.verify_parity:
                        print(f"{index}/{len(pending)} {row['task_id']} seed {row['seed']}: "
                              f"actions {row['actions_compared']}, outcome_match={row['outcome_match']}, "
                              f"first_mismatch={row['first_mismatch']}", flush=True)
                    else:
                        print(f"{index}/{len(pending)} {row['task_id']} seed {row['seed']}: "
                              f"won={row['won']} {row['seconds']}s", flush=True)
        finally:
            pool.close()
            pool.join()
    wall = time.time() - started
    rows = _read_rows(results_path)
    if args.verify_parity:
        mismatches = [row for row in rows if not row["outcome_match"]]
        action_mismatches = [row for row in rows if row["first_mismatch"]]
        summary = {"games": len(rows), "outcome_mismatches": len(mismatches),
                   "action_mismatches": len(action_mismatches),
                   "actions_compared": sum(row["actions_compared"] for row in rows),
                   "mismatches": [{"task_id": row["task_id"], "seed": row["seed"],
                                   "first_mismatch": row["first_mismatch"]}
                                  for row in action_mismatches + mismatches]}
        _write(out / "parity_summary.json", summary)
        print(json.dumps(summary, indent=1), flush=True)
        return 0 if not summary["mismatches"] else 1

    summary = summarize_run(rows)
    summary.update(model=str(model), model_sha256=model_sha, backend=backend,
                   panel=str(panel), panel_sha256=config["panel_sha256"],
                   wall_seconds=round(wall, 1), workers=args.workers,
                   games_per_hour=(round(len(rows) * 3600 / wall, 1) if wall and pending else None),
                   games_per_hour_per_process=(round(len(rows) * 3600 / wall / args.workers, 1)
                                               if wall and pending else None),
                   mean_seconds=round(sum(row["seconds"] for row in rows) / len(rows), 3)
                   if rows else None)
    previous_summary = out / "summary.json"
    if not pending and previous_summary.exists():
        # A fully resumed run spends no time playing; keep the measured speed of the run
        # that actually played the games instead of reporting wall_seconds ~= 0.
        previous = json.loads(previous_summary.read_text())
        for name in ("wall_seconds", "games_per_hour", "games_per_hour_per_process"):
            summary[name] = previous.get(name, summary[name])
        summary["resumed"] = True
    _write(out / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in
                      ("games_per_hour", "games_per_hour_per_process", "mean_seconds")}, indent=1),
          flush=True)
    print(json.dumps(summary["overall"], indent=1), flush=True)
    print(f"{out / 'summary.json'}", flush=True)
    return 0


def _load_run(path: Path) -> tuple[dict, dict[tuple[str, int], dict]]:
    config_path, games_path = path / "config.json", path / "games.jsonl"
    if not config_path.is_file() or not games_path.is_file():
        raise SystemExit(f"{path} is not a bench run (needs config.json and games.jsonl)")
    rows = {(row["task_id"], row["seed"]): row for row in _read_rows(games_path)}
    return json.loads(config_path.read_text()), rows


def compare(path_a: Path, path_b: Path, label_a: str, label_b: str) -> dict:
    _, rows_a = _load_run(path_a)
    _, rows_b = _load_run(path_b)
    keys = sorted(set(rows_a) & set(rows_b))
    if not keys:
        raise SystemExit("the two runs have no common (task, seed) pairs")
    only_a, only_b = set(rows_a) - set(rows_b), set(rows_b) - set(rows_a)
    if only_a or only_b:
        raise SystemExit(f"runs differ in games (only {label_a}: {len(only_a)}, "
                         f"only {label_b}: {len(only_b)}); use the same panel and seeds")
    zones = sorted({rows_a[key]["zone"] for key in keys})
    paired = {key: (bool(rows_a[key]["won"]), bool(rows_b[key]["won"])) for key in keys}

    def block(selected_keys):
        return {**paired_summary([paired[key] for key in selected_keys]),
                label_a: win_summary([rows_a[key] for key in selected_keys]),
                label_b: win_summary([rows_b[key] for key in selected_keys])}

    return {"label_a": label_a, "label_b": label_b, "pairs": len(keys),
            "overall": block(keys),
            "zones": {zone: block([key for key in keys if rows_a[key]["zone"] == zone])
                      for zone in zones},
            "tasks": {task: block([key for key in keys if key[0] == task])
                      for task in sorted({key[0] for key in keys})}}


def _compare(args) -> int:
    path_a, path_b = (Path(path).expanduser().resolve() for path in args.compare)
    result = compare(path_a, path_b, args.label_a or path_a.name, args.label_b or path_b.name)
    if args.out:
        _write(Path(args.out), result)
    print(json.dumps(result, indent=1))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"),
                        help="compare two finished run directories by paired (task, seed)")
    parser.add_argument("--label-a")
    parser.add_argument("--label-b")
    parser.add_argument("--model", help="student checkpoint to benchmark")
    parser.add_argument("--policy-module",
                        help="Python file exposing load_policy(path) and act(policy, frame, previous, "
                             "legal, context=..., history=...) for structures without a C++ export")
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--tasks", help="comma-separated task ids (a subset of the panel)")
    parser.add_argument("--zones", help="comma-separated zones: day,night,pool,fog,roof,heldout")
    parser.add_argument("--allow-partial", action="store_true",
                        help="allow a panel subset without all five terrains + heldout")
    parser.add_argument("--per-task", type=int, default=8, help="seeds per task (default 8)")
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START,
                        help=f"first seed of the per-task block (default {DEFAULT_SEED_START}, a "
                             f"bench-only segment above every registered pool)")
    parser.add_argument("--seeds-from",
                        help="gate decision.json or frozen manifest.json: reuse its exact per-task seeds")
    parser.add_argument("--backend", choices=("auto", "cpp", "python"), default="auto")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--executable", default=str(DEFAULT_EXECUTABLE))
    parser.add_argument("--resource-dir", type=Path, default=DEFAULT_RESOURCE_DIR)
    parser.add_argument("--out", help="run directory (default artifacts/bench_play/<model>-<backend>)")
    parser.add_argument("--verify-parity", action="store_true",
                        help="run both paths on the selected games and compare every decision")
    args = parser.parse_args()

    if args.compare:
        if args.model or args.verify_parity:
            parser.error("--compare does not take --model/--verify-parity")
        try:
            return _compare(args)
        except ValueError as exc:
            parser.error(str(exc))
    if not args.model:
        parser.error("--model is required unless --compare is used")
    if not 1 <= args.workers <= MAX_WORKERS:
        parser.error(f"--workers must be from 1 to {MAX_WORKERS}")
    if args.out is None:
        backend = "parity" if args.verify_parity else args.backend
        args.out = ROOT / "artifacts/bench_play" / f"{Path(args.model).stem}-{backend}"
    try:
        return _run(args)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    sys.exit(main())
