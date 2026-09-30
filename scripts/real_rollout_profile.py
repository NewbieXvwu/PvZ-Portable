#!/usr/bin/env python3
"""Re-measure the per-episode rollout cost split against the real game.

``PPO_UPDATE_ANATOMY.md`` §5 quotes the four ``mean_profile_seconds`` stages from
``artifacts/t5/perf/worker_batch_sweep_2000.json``, which was recorded on 09-29 --
*before* relation-bias fusion became the default.  Those numbers are stale: they
imply 6.8 ms/decision on the ``model`` stage while the same code measures about
1.8 ms/decision once fusion is on.  This script produces the replacement numbers
from the real ``build/pvz-portable`` binary and the real game resources, so the
document does not have to cite a measurement of code that no longer exists.

It drives the *production* ``collect_task_episode`` -- the exact function the T5
rollout workers call -- so the four stages reported here are the same four the
trainer writes into ``mean_episode_profile_seconds``.  Nothing is re-implemented.

The first episode pays resource loading and ``torch.compile`` warmup, so the
steady-state figures exclude it and it is reported on its own.

**Two configurations matter, and the recorded artifacts mix them.**  Every
``throughput.json`` configuration uses one thread per worker, but the serial
(``workers=1``) run measures the model stage at 6.84 ms/decision while the selected
18-worker run measures 13.89 ms/decision -- the workers contend for cores.  Comparing
today's code against only one of those numbers would either hide the code improvement
or overstate it, so both paths are available here:

* ``--workers 1`` (default) is the serial, uncontended figure, comparable to
  ``throughput.json``'s ``workers=1`` row and to ``single_core_episodes_per_hour``.
* ``--workers 18`` goes through the production ``run_seed_jobs`` pool, comparable to
  the selected configuration and to ``worker_batch_sweep_2000.json``.

Usage::

    PY=/Users/newbiexvwu/.local/share/mise/installs/python/3.14/bin/python3
    RES=~/Downloads/Plants_Vs_Zombies_V1.2.0.1073_EN

    # uncontended, one thread, one process (the 6.84 ms/decision baseline)
    $PY scripts/real_rollout_profile.py --resource-dir $RES --episodes 20

    # the production pool (the 13.89 ms/decision baseline)
    $PY scripts/real_rollout_profile.py --resource-dir $RES --episodes 360 --workers 18
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import statistics
import sys
import tempfile
import time
from typing import Any

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from pvz_agent_model import (  # noqa: E402
    GameplayModelV1,
    configure_torch_threads,
    resolve_device,
)
from pvz_env import PvZEnv  # noqa: E402
from pvz_seed_jobs import atomic_json, run_seed_jobs  # noqa: E402
from train_pvz_ppo import collect_task_episode  # noqa: E402
import train_pvz_ppo_task_family as family  # noqa: E402

# The four stages ``collect_task_episode`` accumulates.  ``environment`` includes the
# reset, which is why it is the only stage that also carries process start-up cost.
STAGES = ("model", "environment", "tokenization", "critic_inputs")


def build_plan(episodes: int, task_limit: int | None,
               curriculum: str = "all") -> list[dict[str, Any]]:
    """Cycle the frozen training tasks, giving each episode a distinct task seed.

    ``curriculum`` selects the task subset the same way the trainer does, because the
    two subsets have different token counts per decision and therefore different
    per-decision costs: ``cap1`` is what stage 0 trains on, ``all`` is what the later
    stages and the T5 throughput measurement use.
    """
    train, _ = family._task_family()
    tasks = family._curriculum_tasks(train["tasks"], curriculum)
    if task_limit is not None:
        tasks = tasks[:task_limit]
    plan = []
    for index in range(episodes):
        task = tasks[index % len(tasks)]
        seed = task["seeds"][(index // len(tasks)) % len(task["seeds"])]
        plan.append({"task": task, "task_seed": seed, "action_seed": 1_000 + index})
    return plan


def _record(index: int, row: dict[str, Any], episode: dict[str, Any]) -> dict[str, Any]:
    return {
        "index": index,
        "task_id": row["task"]["task_id"],
        "task_seed": row["task_seed"],
        "won": episode["won"],
        "decisions": len(episode["transitions"]),
        "seconds": episode["seconds"],
        "profile_seconds": episode["profile_seconds"],
    }


def run_serial(model: GameplayModelV1, env: PvZEnv, plan: list[dict[str, Any]],
               max_actions: int, on_done: Any) -> list[dict[str, Any]]:
    rows = []
    for index, row in enumerate(plan):
        torch.manual_seed(row["action_seed"])
        episode = collect_task_episode(
            model, env, row["task"], row["task_seed"], index, max_actions)
        record = _record(index, row, episode)
        rows.append(record)
        on_done(record)
    return rows


def run_parallel(resource_dir: Path, plan: list[dict[str, Any]], max_actions: int,
                 workers: int, threads: int, device: str,
                 on_done: Any) -> list[dict[str, Any]]:
    """Drive the production ``run_seed_jobs`` pool so CPU contention is reproduced.

    The recorded ``worker_*`` artifacts were measured with 18 single-threaded workers
    on one machine, where the workers contend for cores and roughly double the
    per-episode model time.  Measuring only the serial path would understate that, so
    this path goes through the same pool, initializer and worker function the trainer
    uses.  The shard directory is a fresh temporary one, so no cached episode can be
    reused instead of measured.
    """
    torch.manual_seed(20260930)
    state = GameplayModelV1().state_dict()
    assignments = {index: dict(row) for index, row in enumerate(plan)}
    directory = Path(tempfile.mkdtemp(prefix="real-rollout-profile-"))
    try:
        episodes = run_seed_jobs(
            list(assignments),
            directory,
            {"probe": "real_rollout_profile", "curriculum": "all"},
            family._rollout_worker,
            workers=workers,
            initializer=family._init_worker,
            initargs=(str(resource_dir), state, assignments, max_actions, threads, device),
            label="real rollout profile",
        )
    finally:
        shutil.rmtree(directory, ignore_errors=True)
    rows = []
    for index, episode in enumerate(episodes):
        record = _record(index, plan[index], episode)
        rows.append(record)
        on_done(record)
    return rows


def summarize(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    decisions = [episode["decisions"] for episode in episodes]
    total_decisions = sum(decisions)
    stages = {
        stage: sum(episode["profile_seconds"][stage] for episode in episodes)
        for stage in STAGES
    }
    wall = sum(episode["seconds"] for episode in episodes)
    return {
        "episodes": len(episodes),
        "decisions": total_decisions,
        "decisions_per_episode": total_decisions / len(episodes),
        "wall_seconds": wall,
        "seconds_per_episode": wall / len(episodes),
        "stage_seconds": stages,
        "stage_ms_per_decision": {
            stage: 1000.0 * value / total_decisions for stage, value in stages.items()
        },
        "stage_share": {
            stage: value / sum(stages.values()) for stage, value in stages.items()
        },
        "profile_total_seconds": sum(stages.values()),
        "unaccounted_seconds": wall - sum(stages.values()),
    }


def print_report(title: str, summary: dict[str, Any]) -> None:
    print(f"\n=== {title} ===")
    print(f"episodes {summary['episodes']}  decisions {summary['decisions']}  "
          f"mean {summary['decisions_per_episode']:.1f} decisions/episode")
    print(f"wall {summary['wall_seconds']:.2f} s  "
          f"({summary['seconds_per_episode']:.3f} s/episode)")
    print(f"{'stage':<16}{'s total':>10}{'ms/decision':>14}{'share':>9}")
    for stage in STAGES:
        print(f"{stage:<16}{summary['stage_seconds'][stage]:>10.3f}"
              f"{summary['stage_ms_per_decision'][stage]:>14.3f}"
              f"{summary['stage_share'][stage] * 100:>8.1f}%")
    print(f"{'profiled total':<16}{summary['profile_total_seconds']:>10.3f}")
    print(f"{'unaccounted':<16}{summary['unaccounted_seconds']:>10.3f} "
          f"(host-side bookkeeping outside the four timers)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resource-dir", type=Path, default=family.DEFAULT_RESOURCE_DIR)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--curriculum", choices=("all", "cap1"), default="all",
                        help="task subset: 'all' (20 tasks, default) or 'cap1' (stage 0)")
    parser.add_argument("--task-limit", type=int, default=None,
                        help="use only the first N selected tasks (default: all)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1,
                        help="torch threads per rollout worker (default 1, matching "
                             "throughput.json's selected_torch_threads_per_worker; "
                             "batch-1 forward passes are slower with 4 threads)")
    parser.add_argument("--workers", type=int, default=1,
                        help="1 = serial in-process; >1 = the production run_seed_jobs "
                             "pool, which reproduces the CPU contention the recorded "
                             "18-worker artifacts were measured under")
    parser.add_argument("--max-actions", type=int, default=family.WORKER_MAX_ACTIONS)
    parser.add_argument("--output", type=Path, default=None,
                        help="write the raw per-episode rows here as JSON")
    args = parser.parse_args()

    resource_dir = args.resource_dir.expanduser().resolve()
    if not (resource_dir / "main.pak").is_file():
        print(f"main.pak not found under {resource_dir}", file=sys.stderr)
        return 2

    configure_torch_threads(args.threads)
    device = resolve_device(args.device)
    plan = build_plan(args.episodes, args.task_limit, args.curriculum)

    def on_done(record: dict[str, Any]) -> None:
        print(f"  [{record['index'] + 1:>3}/{len(plan)}] {record['task_id']:<16} "
              f"{record['decisions']:>4} dec  {record['seconds']:>7.3f} s  "
              f"model {record['profile_seconds']['model']:>7.3f} s  "
              f"env {record['profile_seconds']['environment']:>6.3f} s", flush=True)

    print(f"resource dir  {resource_dir}")
    print(f"device        {device}  threads/worker {args.threads}  workers {args.workers}")
    print(f"episodes      {args.episodes}  tasks {len({row['task']['task_id'] for row in plan})}"
          f"  curriculum {args.curriculum}")
    print(f"max_actions   {args.max_actions}")

    started = time.perf_counter()
    if args.workers > 1:
        rows = run_parallel(resource_dir, plan, args.max_actions, args.workers,
                            args.threads, args.device, on_done)
        # Every worker pays its own resource load and warmup on its first episode, and
        # the pool finishes episodes out of order, so there is no single "cold" episode
        # to exclude.  The cost is spread over ``workers`` of the ``len(rows)`` rows.
        warm, cold = rows, []
        print(f"\n(note: {args.workers} workers each pay one cold episode; that is "
              f"{args.workers}/{len(rows)} of the rows below)")
    else:
        model = GameplayModelV1().eval().to(device)
        env = PvZEnv(resource_dir=resource_dir)
        rows = run_serial(model, env, plan, args.max_actions, on_done)
        warm, cold = rows[1:], rows[:1]
    wall = time.perf_counter() - started

    warm = rows[1:]
    cold = rows[:1]
    print_report("all episodes", summarize(rows))
    if cold:
        print_report("first episode (cold: resource load + torch.compile warmup)",
                     summarize(cold))
    if warm:
        print_report("steady state (episodes 2..N)", summarize(warm))
    print(f"\ntotal wall {wall:.2f} s")
    print(f"model ms/decision spread (steady state): "
          f"{min(1000.0 * r['profile_seconds']['model'] / r['decisions'] for r in warm):.3f}"
          f" .. {max(1000.0 * r['profile_seconds']['model'] / r['decisions'] for r in warm):.3f}"
          if warm else "")
    if warm:
        medians = {stage: statistics.median(
            1000.0 * r["profile_seconds"][stage] / r["decisions"] for r in warm)
            for stage in STAGES}
        print("median ms/decision (steady state): "
              + "  ".join(f"{stage} {value:.3f}" for stage, value in medians.items()))

    if args.output is not None:
        atomic_json(args.output, {
            "resource_dir": str(resource_dir),
            "device": str(device),
            "threads": args.threads,
            "workers": args.workers,
            "curriculum": args.curriculum,
            "episodes": rows,
            "all": summarize(rows),
            "steady_state": summarize(warm) if warm else None,
            "first_episode": summarize(cold) if cold else None,
            "total_wall_seconds": wall,
        })
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
