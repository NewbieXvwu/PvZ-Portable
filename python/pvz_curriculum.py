"""Explicit terrain coverage and positive learning-progress sampling.

This sampler uses only completed training outcomes. Validation never supplies
its history. Window lengths and coverage floor are experimental choices, not
claims about an optimal curriculum. It cannot manufacture successes from a
pool where every task remains unsolved.
"""
from __future__ import annotations

from collections import defaultdict
import math
from typing import Any


COURSE_METHODS = frozenset({"terrain_balanced", "learning_progress"})
COURSE_SCHEMA = 1


def validate_settings(settings: dict[str, Any]) -> None:
    if set(settings) != {"window_episodes", "minimum_window_episodes", "uniform_fraction"}:
        raise ValueError("curriculum window sizes and uniform_fraction must be explicit")
    window, minimum = settings["window_episodes"], settings["minimum_window_episodes"]
    if (type(window) is not int or type(minimum) is not int or not 1 <= minimum <= window
            or type(settings["uniform_fraction"]) not in (int, float)
            or not math.isfinite(settings["uniform_fraction"])
            or not 0 < settings["uniform_fraction"] <= 1):
        raise ValueError("invalid curriculum windows or coverage floor")


def initial_state(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    ids = [task["task_id"] for task in tasks]
    if not ids or len(ids) != len(set(ids)) or any(not task.get("terrain") for task in tasks):
        raise ValueError("curriculum needs unique task IDs and explicit terrains")
    return {"schema_version": COURSE_SCHEMA, "history": {key: [] for key in ids},
            "completed": {key: 0 for key in ids}, "ignored_truncations": {key: 0 for key in ids}}


def validate_state(state: dict[str, Any], tasks: list[dict[str, Any]], settings: dict[str, Any]) -> None:
    validate_settings(settings)
    expected = {task["task_id"] for task in tasks}
    if state.get("schema_version") != COURSE_SCHEMA:
        raise ValueError("curriculum state schema changed")
    for field in ("history", "completed", "ignored_truncations"):
        if set(state[field]) != expected:
            raise ValueError("curriculum state task pool differs from frozen task pool")
    for key in expected:
        history = state["history"][key]
        if len(history) > 2 * settings["window_episodes"] or any(type(x) is not bool for x in history):
            raise ValueError("invalid curriculum history")
        if (type(state["completed"][key]) is not int or state["completed"][key] < len(history)
                or type(state["ignored_truncations"][key]) is not int
                or state["ignored_truncations"][key] < 0):
            raise ValueError("invalid curriculum observation counters")


def observe(state: dict[str, Any], episodes: list[dict[str, Any]], settings: dict[str, Any]) -> None:
    for episode in episodes:
        key = episode["task_id"]
        if key not in state["history"]:
            raise ValueError("curriculum cannot ingest outcomes outside its training pool")
        if episode["truncated"]:
            if episode["terminated"] or episode["won"]:
                raise ValueError("truncated episode cannot be a normal terminal or a win")
            state["ignored_truncations"][key] += 1
            continue
        if not episode["terminated"] or type(episode["won"]) is not bool:
            raise ValueError("curriculum requires explicit normal-terminal labels")
        state["history"][key].append(episode["won"])
        del state["history"][key][:-2 * settings["window_episodes"]]
        state["completed"][key] += 1


def probabilities(tasks: list[dict[str, Any]], state: dict[str, Any],
                  settings: dict[str, Any], method: str) -> tuple[list[float], dict[str, Any]]:
    if method not in COURSE_METHODS:
        raise ValueError("unsupported curriculum method")
    validate_state(state, tasks, settings)
    groups = defaultdict(list)
    task_details = {}
    window, minimum = settings["window_episodes"], settings["minimum_window_episodes"]
    for task in tasks:
        key = task["task_id"]
        groups[task["terrain"]].append(key)
        history = state["history"][key]
        # Equal adjacent windows; at warm-up use every available pair. There is
        # no comparison against a differently sized or differently sampled window.
        size = min(window, len(history) // 2)
        previous, recent, gain = None, None, 0.0
        if size >= minimum:
            previous = sum(history[-2 * size:-size]) / size
            recent = sum(history[-size:]) / size
            gain = max(0.0, recent - previous)
        task_details[key] = {"terrain": task["terrain"], "window_size": size,
                             "previous_pass_rate": previous, "recent_pass_rate": recent,
                             "positive_progress": gain, "completed": state["completed"][key],
                             "ignored_truncations": state["ignored_truncations"][key]}
    terrain_mass = 1 / len(groups)
    uniform = settings["uniform_fraction"]
    for keys in groups.values():
        total_gain = sum(task_details[key]["positive_progress"] for key in keys)
        for key in keys:
            within = 1 / len(keys)
            if method == "learning_progress" and total_gain > 0:
                within = uniform / len(keys) + (1 - uniform) * task_details[key]["positive_progress"] / total_gain
            task_details[key]["probability"] = terrain_mass * within
    weights = [task_details[task["task_id"]]["probability"] for task in tasks]
    if not math.isclose(sum(weights), 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError("curriculum probabilities do not sum to one")
    return weights, {"method": method, "settings": dict(settings),
                     "terrain_probabilities": {terrain: terrain_mass for terrain in groups},
                     "tasks": task_details,
                     "interpretation": "positive empirical progress heuristic; no optimality or confidence claim"}
