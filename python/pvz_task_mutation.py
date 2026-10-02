"""Append-only, explicitly bounded training-task mutation.

Each round has its own deterministic generator seed. Sampling RNG and validation
outcomes never enter this generator. The pool and round events belong to the
complete training checkpoint, so replaying an interrupted collection recreates
the same task definitions, environment seeds and assignments.
"""
from __future__ import annotations

import copy
import math
import random
from typing import Any

TERRAINS = ("day", "night", "pool", "fog", "roof")


def signature(task: dict[str, Any]) -> tuple:
    return (task["level"], tuple(task["deck"]), task["zombie_count_multiplier"],
            task["wave_cap"], task["sun_start"], tuple(map(tuple, task["preplanted"])))


def validate_settings(settings: dict, base: list[dict], evaluation: list[dict]) -> None:
    fields = {"method", "generator_seed", "interval_decisions", "max_rounds",
              "seed_start", "seeds_per_task", "levels", "decks", "multipliers"}
    if set(settings) != fields or settings["method"] != "append_neighbors_v1":
        raise ValueError("task mutation requires explicit append_neighbors_v1 settings")
    for key in ("generator_seed", "interval_decisions", "max_rounds", "seed_start", "seeds_per_task"):
        if type(settings[key]) is not int or settings[key] < (0 if key == "generator_seed" else 1):
            raise ValueError(f"invalid task mutation {key}")
    if settings["seeds_per_task"] < 64 or settings["seed_start"] + settings["max_rounds"] * 5 * settings["seeds_per_task"] > 2**32:
        raise ValueError("mutation needs at least64 seeds/task within uint32")
    multipliers = settings["multipliers"]
    if (not multipliers or len(multipliers) != len(set(multipliers))
            or any(type(x) not in (float, int) or not math.isfinite(x) or not 1 <= x <= 10 for x in multipliers)):
        raise ValueError("mutation multipliers must stay in the original legal range1..10")
    if set(settings["levels"]) != set(TERRAINS) or set(settings["decks"]) != set(TERRAINS):
        raise ValueError("mutation requires explicit level/deck bounds for all five terrains")
    for i, terrain in enumerate(TERRAINS):
        levels, decks = settings["levels"][terrain], settings["decks"][terrain]
        if (not levels or len(levels) != len(set(levels))
                or any(type(x) is not int or not 1+10*i <= x <= min(10+10*i, 49) for x in levels)):
            raise ValueError("mutation levels must retain their declared terrain")
        if (not decks or len({tuple(deck) for deck in decks}) != len(decks)
                or any(not deck or len(deck) > 6 or len(set(deck)) != len(deck)
                       or any(type(x) is not int or not 0 <= x <= 48 for x in deck) for deck in decks)):
            raise ValueError("mutation deck bounds must contain unique legal six-slot decks")
    if ({task["terrain"] for task in base} != set(TERRAINS)
            or any(task["sun_start"] != 50 or task["preplanted"] or task["playthrough"] != 2
                   or task["wave_cap"] not in (1, 3, 5) for task in base)):
        raise ValueError("append_neighbors_v1 needs ordinary Adventure-II1/3/5-wave parents")
    first = settings["seed_start"]
    last = first + settings["max_rounds"] * 5 * settings["seeds_per_task"]
    if any(first <= seed < last for task in base + evaluation for seed in task["seeds"]):
        raise ValueError("mutation seed block overlaps frozen training/evaluation seeds")


def initial_state(base: list[dict]) -> dict:
    return {"schema_version": 1, "rounds": 0, "pool": copy.deepcopy(base), "events": []}


def validate_state(state: dict, settings: dict, base: list[dict], evaluation: list[dict]) -> None:
    validate_settings(settings, base, evaluation)
    if (state.get("schema_version") != 1 or type(state.get("rounds")) is not int
            or not 0 <= state["rounds"] <= settings["max_rounds"]
            or len(state.get("events", [])) != state["rounds"]
            or state.get("pool", [])[:len(base)] != base):
        raise ValueError("mutation state changed original tasks or round history")
    pool = state["pool"]
    ids = [task["task_id"] for task in pool]
    if len(ids) != len(set(ids)) or len({signature(task) for task in pool}) != len(pool):
        raise ValueError("mutation pool contains duplicate task IDs/definitions")
    parents = {task["task_id"]: task for task in base}
    children = {task["task_id"]: task for task in pool[len(base):]}
    seen = set()
    blocked = {signature(task) for task in evaluation}
    for round_index, event in enumerate(state["events"], 1):
        if (event["round"] != round_index
                or event["trigger_decisions"] != round_index * settings["interval_decisions"]
                or event["at_decisions"] < event["trigger_decisions"]):
            raise ValueError("mutation round schedule changed")
        event_terrains = set()
        for row in event["additions"]:
            child, parent = children[row["task_id"]], parents[row["parent_task_id"]]
            terrain = child["terrain"]
            if terrain in event_terrains or terrain != parent["terrain"]:
                raise ValueError("mutation round repeats or changes parent terrain")
            event_terrains.add(terrain)
            expected_id = f"mutation_v1_r{round_index:03d}_{terrain}"
            offset = ((round_index - 1) * 5 + TERRAINS.index(terrain)) * settings["seeds_per_task"]
            first = settings["seed_start"] + offset
            expected = copy.deepcopy(parent)
            axis = row["axis"]
            allowed = {"level": settings["levels"][terrain], "deck": settings["decks"][terrain],
                       "zombie_count_multiplier": settings["multipliers"]}
            if axis not in allowed or child[axis] == parent[axis] or child[axis] not in allowed[axis]:
                raise ValueError("mutation changed more than one bounded task axis")
            expected[axis] = copy.deepcopy(child[axis])
            expected.update(task_id=expected_id, seeds=list(range(first, first + settings["seeds_per_task"])))
            if child != expected or signature(child) in blocked or expected_id in seen:
                raise ValueError("mutation child differs from frozen bounds or overlaps evaluation")
            seen.add(expected_id)
    if seen != set(children):
        raise ValueError("mutation pool and generation history differ")


def advance(state: dict, settings: dict, base: list[dict], evaluation: list[dict],
            curriculum: dict, course_settings: dict, decisions: int) -> list[dict]:
    """Add at most one new neighbor per terrain per due round; never remove tasks."""
    added = []
    blocked = {signature(task) for task in state["pool"] + evaluation}
    due_rounds = min(decisions // settings["interval_decisions"], settings["max_rounds"])
    while state["rounds"] < due_rounds:
        round_index = state["rounds"] + 1
        rng = random.Random(f"{settings['generator_seed']}:{round_index}")
        event = {"round": round_index, "trigger_decisions": round_index * settings["interval_decisions"],
                 "at_decisions": decisions, "additions": [], "exhausted_terrains": []}
        for terrain_index, terrain in enumerate(TERRAINS):
            parents = [task for task in base if task["terrain"] == terrain]
            edge = []
            low, high = course_settings["frontier_pass_range"]
            for parent in parents:
                history = curriculum["history"][parent["task_id"]][-course_settings["window_episodes"]:]
                if len(history) >= course_settings["minimum_window_episodes"] and low <= sum(history) / len(history) <= high:
                    edge.append(parent)

            def neighbors(sources):
                candidates = []
                unique = set()
                for parent in sources:
                    for axis, values in (("deck", settings["decks"][terrain]),
                                         ("zombie_count_multiplier", settings["multipliers"]),
                                         ("level", settings["levels"][terrain])):
                        for value in values:
                            child = copy.deepcopy(parent)
                            child[axis] = copy.deepcopy(value)
                            key = signature(child)
                            if value != parent[axis] and key not in blocked and key not in unique:
                                candidates.append((parent, axis, child))
                                unique.add(key)
                return candidates

            candidates = neighbors(edge) if edge else []
            if not candidates:
                candidates = neighbors(parents)
            if not candidates:
                event["exhausted_terrains"].append(terrain)
                continue
            parent, axis, child = rng.choice(candidates)
            child["task_id"] = f"mutation_v1_r{round_index:03d}_{terrain}"
            offset = ((round_index - 1) * 5 + terrain_index) * settings["seeds_per_task"]
            first = settings["seed_start"] + offset
            child["seeds"] = list(range(first, first + settings["seeds_per_task"]))
            state["pool"].append(child)
            blocked.add(signature(child))
            added.append(child)
            event["additions"].append({"task_id": child["task_id"], "parent_task_id": parent["task_id"],
                                      "axis": axis, "parent_source": "frontier" if parent in edge else "all_original"})
        state["events"].append(event)
        state["rounds"] = round_index
    return added
