"""Rollout throughput benchmark: the workload the teacher's search actually pays for.

One rollout = restore a root position, play the candidate action, let the student continue for
``--horizon`` ticks in the simulator, stop at the leaf. This script measures that on the original
C++ simulator (``--backend cpp``) and reports per-rollout simulation / student-network seconds as
the C++ side times them, plus the wall time of the whole call (which additionally covers the root
restore, the observation round trip, and Python parsing).

``roots`` are taken from real games: the student plays the task's seed and we snapshot at fixed
tick intervals, exactly where the teacher would search. Candidates are the same set the teacher
compares (best cell per card by the static value net, plus "do nothing now").

Usage:
  python3 scripts/rollout_bench.py --backend cpp --manifest ~/PvZ-search/artifacts/stage4/day_l7_hard.json \
      --task-id day_l7_z15_train --seeds 4 --rollout-policy .../student/v14/model.pt \
      --static-value .../hard/value_branch1/model.pt --value .../value_net/v2/model.pt \
      --resource-dir ~/.cache/pvz-research-resources --workers 1 --out artifacts/cuda/bench_cpp.json
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
NOOP = {"type": "wait", "ticks": 60, "until": "timeout"}
_W: dict = {}


def _init(job_args: dict) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    sys.path.insert(0, str(ROOT / "python"))
    sys.path.insert(0, str(ROOT / "scripts"))
    import torch
    torch.set_num_threads(1)
    from pvz_rollout_policy import load_student, export_student
    from pvz_value_net import load_value_net
    _W["vnet"] = load_value_net(job_args["value"])
    _W["static_vnet"] = load_value_net(job_args["static_value"] or job_args["value"])
    _W["fast"] = load_student(job_args["rollout_policy"])
    import tempfile
    _W["rollout_tmp"] = tempfile.TemporaryDirectory(prefix="pvz-bench-")
    _W["rollout_file"] = str(Path(_W["rollout_tmp"].name) / "student.bin")
    export_student(_W["fast"], _W["rollout_file"])
    _W["args"] = job_args
    if job_args["backend"] == "cpp":
        from pvz_env import PvZEnv
        _W["env"] = PvZEnv(job_args["resource_dir"], executable=job_args["executable"], headless=True)
        _W["env"].load_rollout_net(_W["rollout_file"])


def _candidates(env, obs: dict, income) -> list[dict]:
    """The teacher's candidate set: best ``--per-card`` cells of every legal card, plus NOOP."""
    import torch
    from pvz_value_net import featurize
    from render_episode import _frame
    args = _W["args"]
    legal = (obs.get("legal_actions") or {}).get("plants") or []
    plants = [{"type": "plant", "packet": a["packet"], "col": a["col"], "row": a["row"]} for a in legal]
    if not plants:
        return []
    per_card = args["per_card"]
    scored = []
    for start in range(0, len(plants), 128):
        chunk = plants[start:start + 128]
        records = env.branch_snapshot(_W["root"], chunk, retain_snapshots=False)
        frames = []
        for action, record in zip(chunk, records):
            if not record.get("ok") or not record.get("observation"):
                continue
            frame = _frame(record["observation"])
            frame["sun_income_rate"] = income
            frames.append((action, frame))
        if not frames:
            continue
        boards, scalars = zip(*(featurize(f) for _a, f in frames))
        scalars = torch.stack(scalars)
        net = _W["static_vnet"]
        if net.context:
            from pvz_value_net import observation_context
            scalars = torch.cat([scalars, torch.from_numpy(observation_context(obs))[None].expand(len(frames), -1)], dim=1)
        with torch.inference_mode():
            values = torch.sigmoid(net(torch.stack(boards), scalars)).tolist()
        scored.extend(zip(values, [a for a, _f in frames]))
    scored.sort(key=lambda item: -item[0])
    by_card: dict[int, list] = {}
    for value, action in scored:
        by_card.setdefault(action["packet"], []).append(action)
    return [a for cells in by_card.values() for a in cells[:per_card]]


def _worker(seed: int) -> dict:
    args = _W["args"]
    env = _W["env"]
    from render_episode import _frame
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT / "python"))
    from pvz_env import TaskSpec
    from render_episode import load_training_task, training_task_kwargs
    from pvz_tasks import profile_for_deck
    kw = load_training_task(Path(args["manifest"]), args["task_id"])
    task = TaskSpec(level=kw["level"], seed=seed, playthrough=2,
                    profile=profile_for_deck(tuple(kw["deck"])),
                    **{k: v for k, v in training_task_kwargs(kw, seed)["task_extra"].items() if v is not None})
    obs, _ = env.reset(deck=tuple(kw["deck"]), task=task)
    env.set_lean_observation(True)
    env.set_sim_skip(args["sim_skip"])
    rows, root_tick, roots, skipped = [], args["root_start"], 0, 0
    while not obs["terminal"] and roots < args["roots_per_game"] and obs["tick"] < args["max_root_tick"]:
        obs = env.rollout(None, root_tick, max_actions=args["max_actions"])
        if obs["terminal"]:
            break
        root_tick = obs["tick"] + args["root_stride"]
        if not (obs.get("legal_actions") or {}).get("plants"):
            # the teacher only searches where something is plantable (search_play._search returns
            # the policy's own action otherwise), so a root with no candidate is not representative
            skipped += 1
            continue
        roots += 1
        _W["root"] = root = env.snapshot()
        income = obs.get("sun_income_rate")
        candidates = _candidates(env, obs, income)
        for index, action in enumerate(candidates + [NOOP]):
            for k in range(args["samples"]):
                future = (seed * 1000 + k) % 2 ** 32
                started = time.perf_counter()
                leaf = env.rollout(action, obs["tick"] + args["horizon"], sim_skip=args["sim_skip"],
                                   snapshot_id=root, seed=future)
                wall = time.perf_counter() - started
                details = env._last_response
                rows.append({"wall": wall, "sim": details["simulation_seconds"],
                             "net": details["network_seconds"], "decisions": details["decisions"],
                             "net_calls": details.get("net_calls", 0), "ticks": details.get("ticks", 0),
                             "tick": obs["tick"], "root_seed": seed, "candidate": index})
        env.restore(root)
        env.release_snapshot(root)
    env.set_sim_skip(0)
    env.set_lean_observation(False)
    return {"seed": seed, "rows": [{k: v for k, v in r.items()} for r in rows]}


def summarize(batches: list[dict], workers: int) -> dict:
    rows = [r for batch in batches for r in batch["rows"]]
    if not rows:
        raise SystemExit("no rollouts measured")
    wall = sum(r["wall"] for r in rows)
    sim = sum(r["sim"] for r in rows)
    net = sum(r["net"] for r in rows)
    count = len(rows)
    ticks = sum(r.get("ticks", 0) for r in rows)
    calls = sum(r.get("net_calls", 0) for r in rows)
    return {
        "rollouts": count,
        "workers": workers,
        "mean_decisions": round(sum(r["decisions"] for r in rows) / count, 1),
        "mean_net_calls": round(calls / count, 1),
        "mean_ticks": round(ticks / count, 1),
        "sim_microseconds_per_tick": round(sim / ticks * 1e6, 2) if ticks else None,
        "net_milliseconds_per_call": round(net / calls * 1e3, 3) if calls else None,
        "seconds_per_rollout_cpp": round((sim + net) / count, 4),
        "seconds_per_rollout_wall": round(wall / count, 4),
        "cpp_sim_share": round(sim / (sim + net), 3),
        "rollouts_per_second_single_thread": round(count / (sim + net), 2),
        "rollouts_per_second_wall": round(count / wall, 2),
        "note": "single_thread = one C++ simulator thread; wall includes root restore, "
                "observation JSON round trip and Python parsing",
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", default="cpp")
    p.add_argument("--manifest", required=True)
    p.add_argument("--task-id", required=True)
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument("--roots-per-game", type=int, default=8)
    p.add_argument("--root-start", type=int, default=600)
    p.add_argument("--root-stride", type=int, default=600)
    p.add_argument("--max-root-tick", type=int, default=24000)
    p.add_argument("--horizon", type=int, default=2500)
    p.add_argument("--sim-skip", type=int, default=29)
    p.add_argument("--samples", type=int, default=3)
    p.add_argument("--per-card", type=int, default=1)
    p.add_argument("--max-actions", type=int, default=3000)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--rollout-policy", required=True)
    p.add_argument("--value", required=True)
    p.add_argument("--static-value")
    p.add_argument("--executable", default=str(ROOT / "build" / "pvz-portable"))
    p.add_argument("--resource-dir", default=os.path.expanduser("~/.cache/pvz-research-resources"))
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    job = vars(args)
    job["resource_dir"] = os.path.expanduser(args.resource_dir)
    manifest = os.path.expanduser(args.manifest)
    job["manifest"] = manifest
    seeds = load_seeds(manifest, args.task_id)[args.seed_offset: args.seed_offset + args.seeds]
    started = time.time()
    if args.workers == 1:
        _init(job)
        batches = [_worker(seed) for seed in seeds]
    else:
        with mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(job,)) as pool:
            batches = pool.map(_worker, seeds)
    summary = summarize(batches, args.workers)
    summary["elapsed"] = round(time.time() - started, 1)
    summary["load_avg"] = os.getloadavg()
    summary["cpu_count"] = os.cpu_count()
    summary["task"] = {"manifest": manifest, "task_id": args.task_id, "seeds": seeds,
                       "horizon": args.horizon, "sim_skip": args.sim_skip, "samples": args.samples,
                       "roots_per_game": args.roots_per_game, "backend": args.backend}
    print(json.dumps(summary, indent=1))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"summary": summary, "rows": [r for b in batches for r in b["rows"]]}, indent=1))


def load_seeds(manifest: str, task_id: str) -> list[int]:
    sys.path.insert(0, str(ROOT / "scripts"))
    from render_episode import load_training_task
    return load_training_task(Path(manifest), task_id)["seeds"]


if __name__ == "__main__":
    main()
