"""Explicit terrain/task-length coverage and positive learning-progress sampling.

This sampler uses only completed training outcomes. Validation never supplies
its history. Window lengths and coverage floor are experimental choices, not
claims about an optimal curriculum. It cannot manufacture successes from a
pool where every task remains unsolved.
"""
from __future__ import annotations

from collections import defaultdict
import math
from typing import Any


COURSE_METHODS = frozenset({"terrain_balanced", "learning_progress", "frontier_v1"})
COURSE_SCHEMA = 2


def validate_settings(settings: dict[str, Any], method: str | None = None) -> None:
    fields = {"window_episodes", "minimum_window_episodes", "uniform_fraction", "coverage"}
    frontier_fields = fields | {"frontier_pass_range"}
    if (set(settings) not in (fields, frontier_fields)
            or (method == "frontier_v1" and set(settings) != frontier_fields)
            or (method is not None and method != "frontier_v1" and set(settings) != fields)):
        raise ValueError("curriculum window sizes, uniform_fraction and coverage must be explicit")
    if "frontier_pass_range" in settings:
        band = settings["frontier_pass_range"]
        if (not isinstance(band, list) or len(band) != 2
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in band)
                or not 0 < band[0] < band[1] < 1):
            raise ValueError("frontier_pass_range must explicitly define two interior pass rates")
    if settings["coverage"] not in ("terrain", "terrain_wave_cap"):
        raise ValueError("curriculum coverage must be terrain or terrain_wave_cap")
    window, minimum = settings["window_episodes"], settings["minimum_window_episodes"]
    if (type(window) is not int or type(minimum) is not int or not 1 <= minimum <= window
            or type(settings["uniform_fraction"]) not in (int, float)
            or not math.isfinite(settings["uniform_fraction"])
            or not 0 < settings["uniform_fraction"] <= 1):
        raise ValueError("invalid curriculum windows or coverage floor")


def task_metadata(tasks: list[dict[str, Any]]) -> dict[str, dict]:
    return {task["task_id"]: {"terrain": task["terrain"], "has_wave_cap": "wave_cap" in task,
                             "wave_cap": task.get("wave_cap")} for task in tasks}


def initial_state(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    ids = [task["task_id"] for task in tasks]
    if (not ids or len(ids) != len(set(ids)) or any(type(key) is not str or not key for key in ids)
            or any(type(task.get("terrain")) is not str or not task["terrain"] for task in tasks)):
        raise ValueError("curriculum needs unique task IDs and explicit terrains")
    return {"schema_version": COURSE_SCHEMA, "task_metadata": task_metadata(tasks),
            "history": {key: [] for key in ids},
            "completed": {key: 0 for key in ids}, "ignored_truncations": {key: 0 for key in ids}}


def validate_state(state: dict[str, Any], tasks: list[dict[str, Any]], settings: dict[str, Any]) -> None:
    validate_settings(settings)
    expected_metadata = initial_state(tasks)["task_metadata"]
    expected = {task["task_id"] for task in tasks}
    if state.get("schema_version") != COURSE_SCHEMA:
        raise ValueError("curriculum state schema changed")
    for field in ("history", "completed", "ignored_truncations"):
        if set(state[field]) != expected:
            raise ValueError("curriculum state task pool differs from frozen task pool")
    if state.get("task_metadata") != expected_metadata:
        raise ValueError("curriculum frozen task terrain/wave-cap metadata changed")
    if settings["coverage"] == "terrain_wave_cap":
        for task in tasks:
            cap = task.get("wave_cap")
            if "wave_cap" not in task or (cap is not None and (type(cap) is not int or not 1 <= cap <= 50)):
                raise ValueError("task-length coverage requires explicit wave_cap (1..50 or None for full)")
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
    validate_settings(settings, method)
    validate_state(state, tasks, settings)
    groups = defaultdict(lambda: defaultdict(list))
    task_details = {}
    window, minimum = settings["window_episodes"], settings["minimum_window_episodes"]
    for task in tasks:
        key = task["task_id"]
        cap = task.get("wave_cap")
        cap_label = "all" if settings["coverage"] == "terrain" else ("full" if cap is None else f"cap{cap}")
        groups[task["terrain"]][cap_label].append(key)
        history = state["history"][key]
        # Equal adjacent windows; at warm-up use every available pair. There is
        # no comparison against a differently sized or differently sampled window.
        size = min(window, len(history) // 2)
        previous, recent, gain = None, None, 0.0
        if size >= minimum:
            previous = sum(history[-2 * size:-size]) / size
            recent = sum(history[-size:]) / size
            gain = max(0.0, recent - previous)
        task_details[key] = {"terrain": task["terrain"], "coverage_group": cap_label, "window_size": size,
                             "previous_pass_rate": previous, "recent_pass_rate": recent,
                             "positive_progress": gain, "completed": state["completed"][key],
                             "ignored_truncations": state["ignored_truncations"][key]}
        if method == "frontier_v1":
            # A stable intermediate win rate is a frontier even when adjacent
            # windows show no positive gain. Only normal training terminals
            # supply this estimate; cold tasks still receive the coverage floor.
            frontier_size = min(window, len(history))
            frontier_rate = (sum(history[-frontier_size:]) / frontier_size
                             if frontier_size >= minimum else None)
            low, high = settings["frontier_pass_range"]
            task_details[key].update(frontier_window_size=frontier_size,
                frontier_pass_rate=frontier_rate,
                frontier_score=float(frontier_rate is not None and low <= frontier_rate <= high))
    terrain_mass = 1 / len(groups)
    uniform = settings["uniform_fraction"]
    coverage_probabilities = {}
    for terrain, caps in groups.items():
        group_mass = terrain_mass / len(caps)
        coverage_probabilities[terrain] = {cap: group_mass for cap in caps}
        for keys in caps.values():
            total_gain = sum(task_details[key]["positive_progress"] for key in keys)
            for key in keys:
                within = 1 / len(keys)
                if method == "learning_progress" and total_gain > 0:
                    within = uniform / len(keys) + (1 - uniform) * task_details[key]["positive_progress"] / total_gain
                task_details[key]["probability"] = group_mass * within
    if method == "frontier_v1":
        total_score = sum(row["frontier_score"] for row in task_details.values())
        for row in task_details.values():
            # Mix a terrain/cap-balanced floor with a global frontier allocation.
            # Priority can cross singleton coverage groups, unlike the old LP
            # sampler; no task or failed seed is removed from the frozen pool.
            floor = row["probability"]
            row["coverage_base_probability"] = floor
            row["probability"] = (uniform * floor + (1 - uniform) * row["frontier_score"] / total_score
                                  if total_score > 0 else floor)
        coverage_probabilities = {
            terrain: {cap: sum(task_details[key]["probability"] for key in keys)
                      for cap, keys in caps.items()} for terrain, caps in groups.items()}
    weights = [task_details[task["task_id"]]["probability"] for task in tasks]
    if not math.isclose(sum(weights), 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError("curriculum probabilities do not sum to one")
    return weights, {"method": method, "settings": dict(settings),
                     "terrain_probabilities": ({terrain: sum(caps.values())
                                                for terrain, caps in coverage_probabilities.items()}
                                               if method == "frontier_v1" else
                                               {terrain: terrain_mass for terrain in groups}),
                     "coverage_probabilities": coverage_probabilities,
                     "tasks": task_details,
                     "interpretation": "episode-assignment probabilities, not equal decision/tick/time budgets; explicit curriculum heuristic, no optimality claim"}
