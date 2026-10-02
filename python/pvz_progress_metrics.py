"""Separate task difficulty and pair policy results with a frozen idle control.

Short-course wins can be produced by lawnmowers without any policy action.
Controls keep the exact task, seed and native environment; they never supply
training trajectories or alter acceptance thresholds.
"""
from __future__ import annotations

from collections import defaultdict
import gzip
import json
from pathlib import Path

from pvz_common import canonical_digest
from pvz_seed_jobs import atomic_json


def task_group(task: dict) -> str:
    cap = task["wave_cap"]
    aided = bool(task["preplanted"]) or task["sun_start"] > 50 or task["zombie_count_multiplier"] < 1
    if cap is not None and cap < 10:
        if task.get("terrain") == "roof":
            return ("aided_" if aided else "") + ("roof_short" if cap < 3 else "roof_multiwave")
        return "short_regression"
    if aided:
        return "aided_full" if cap is None else "aided_long"
    if task["sun_start"] != 50 or task["zombie_count_multiplier"] != 1:
        return "variant_full" if cap is None else "variant_long"
    return "ordinary_full" if cap is None else "ordinary_long"


def _seed_map(records: list[dict]) -> dict:
    mapped = {row["seed"]: row for row in records}
    if len(mapped) != len(records):
        raise ValueError("duplicate evaluation seeds")
    return mapped


def _check_rows(tasks: list[dict], grouped: dict) -> None:
    if set(grouped) != {task["task_id"] for task in tasks}:
        raise ValueError("evaluation task set differs from frozen manifest")
    for task in tasks:
        if set(_seed_map(grouped[task["task_id"]])) != set(task["seeds"]):
            raise ValueError("evaluation seed set differs from frozen manifest")


def idle_baseline(tasks: list[dict], resource_dir: Path, output: Path,
                  experiment_identity: str, max_actions: int) -> dict:
    import t4_capability_profile as profile
    from pvz_env import PvZEnv
    identity = canonical_digest({"schema_version": 1, "experiment_identity": experiment_identity,
                                 "tasks": tasks, "max_actions": max_actions, "fixed_wait_ticks": 300})
    path = output / "evaluations/idle_control.json.gz"
    if path.exists():
        with gzip.open(path, "rt") as stream:
            saved = json.load(stream)
        if saved["identity"] != identity:
            raise ValueError("idle control cache identity differs; preserve the existing evidence")
        grouped = saved["seed_results"]
    else:
        with PvZEnv(resource_dir=resource_dir) as env:
            grouped = {task["task_id"]: [profile.run_episode(
                env, task, seed, "idle", max_actions=max_actions, allow_truncation=True)
                for seed in task["seeds"]] for task in tasks}
        atomic_json(path, {"schema_version": 1, "identity": identity,
                          "experiment_identity": experiment_identity, "fixed_wait_ticks": 300,
                          "tasks": tasks, "seed_results": grouped,
                          "progress_groups": progress_summary(tasks, grouped)}, compressed=True)
    _check_rows(tasks, grouped)
    return grouped


def progress_summary(tasks: list[dict], grouped: dict) -> dict:
    import t4_capability_profile as profile
    _check_rows(tasks, grouped)
    groups = defaultdict(list)
    for task in tasks:
        groups[task_group(task)].extend(grouped[task["task_id"]])
    return {name: {**profile.summarize_episodes(rows),
                   "truncated": sum(row["truncated"] for row in rows)}
            for name, rows in groups.items()}


def paired_idle_summary(tasks: list[dict], policy: dict, idle: dict) -> dict:
    import t4_capability_profile as profile
    _check_rows(tasks, policy)
    _check_rows(tasks, idle)
    per_task, groups = {}, defaultdict(lambda: ([], []))

    def compare(policy_rows: list[dict], idle_rows: list[dict]) -> dict:
        p, b = _seed_map(policy_rows), _seed_map(idle_rows)
        # Group summaries may contain the same seed in different tasks. Pairing
        # is calculated per task below; do not pool by seed across tasks.
        improved = sum(p[s]["won"] and not b[s]["won"] for s in p)
        degraded = sum(b[s]["won"] and not p[s]["won"] for s in p)
        return {"cases": len(p), "improved_pairs": improved, "degraded_pairs": degraded,
                "net_win_gain": (improved - degraded) / len(p),
                "policy": profile.summarize_episodes(policy_rows),
                "idle": profile.summarize_episodes(idle_rows),
                "policy_truncated": sum(row["truncated"] for row in policy_rows),
                "idle_truncated": sum(row["truncated"] for row in idle_rows)}

    for task in tasks:
        key, group = task["task_id"], task_group(task)
        per_task[key] = compare(policy[key], idle[key])
        groups[group][0].extend(policy[key])
        groups[group][1].extend(idle[key])
    by_group = {}
    for name, (p_rows, b_rows) in groups.items():
        components = [per_task[t["task_id"]] for t in tasks if task_group(t) == name]
        improved = sum(row["improved_pairs"] for row in components)
        degraded = sum(row["degraded_pairs"] for row in components)
        by_group[name] = {"cases": len(p_rows), "improved_pairs": improved,
            "degraded_pairs": degraded, "net_win_gain": (improved - degraded) / len(p_rows),
            "policy": profile.summarize_episodes(p_rows), "idle": profile.summarize_episodes(b_rows),
            "policy_truncated": sum(row["truncated"] for row in p_rows),
            "idle_truncated": sum(row["truncated"] for row in b_rows)}
    return {"per_task": per_task, "by_group": by_group,
            "scope": "Exact task/seed pairs, descriptive win differences; aided scores and short regression scores are not full-course acceptance."}
