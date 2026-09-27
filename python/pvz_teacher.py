"""Rule-based candidate generator used as a prior by the search teacher."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from pvz_agent_model import WAIT_DECISION_TICKS


@dataclass(frozen=True)
class TeacherAdvice:
    action: dict[str, Any]
    candidates: list[tuple[dict[str, Any], float]]
    search_policy: list[float] | None = None
    best_second_margin: float | None = None
    search_depth: int = 0
    simulation_count: int = 0
    terminal_outcome: int | None = None
    search_elapsed_ticks: int = 0


def teacher_advice(observation: dict[str, Any], sunflower_placements: int = 0) -> TeacherAdvice:
    rows = sorted({cell["row"] for cell in observation["cells"] if cell["row_type"] > 0})
    zombies = observation["zombies"]
    plants = observation["plants"]
    zombie_multiplier = max(1.0, float(observation.get("zombie_count_multiplier", 1.0)))
    safe_shot_gap = 240.0 + 30.0 * (zombie_multiplier - 1.0)
    emergency_front = max(180.0, 450.0 - 50.0 * (zombie_multiplier - 1.0))
    packets = {packet["index"]: packet for packet in observation["packets"]}
    legal = observation["legal_actions"]
    plants_by_row = {row: [] for row in rows}
    zombies_by_row = {row: [zombie for zombie in zombies if zombie["row"] == row] for row in rows}

    def plant_type(plant: dict[str, Any]) -> int:
        return plant["imitater_type"] if plant["type"] == 48 and plant["imitater_type"] >= 0 else plant["type"]

    for plant in plants:
        if plant["row"] in plants_by_row:
            plants_by_row[plant["row"]].append(plant)

    def zombie_urgency(zombie: dict[str, Any]) -> float:
        return max(0.0, min(1.0, (650.0 - zombie["x"]) / 420.0))

    def zombie_hp(zombie: dict[str, Any]) -> float:
        return sum(max(0.0, zombie.get(key, 0.0)) for key in ("body_health", "helm_health", "shield_health"))

    def expected_shots(zombie: dict[str, Any], plant_x: float) -> float:
        speed = abs(zombie.get("velocity_x", 0.0))
        if speed <= 0.01:
            return 0.0
        firing_window = max(0.0, (zombie["x"] - plant_x - 80.0) / speed)
        projectile_travel = max(0.0, zombie["x"] - plant_x) / 5.0
        return max(0.0, firing_window - projectile_travel) / 150.0

    attackers = {
        row: [p for p in plants_by_row[row]
              if plant_type(p) in (0, 5) and not p["squished"]
              and p["health"] > 0.1 * max(1, p["max_health"])
              and min((z["x"] for z in zombies_by_row[row]), default=9999.0) > 160 + 80 * p["col"]]
        for row in rows
    }
    wallnuts = {row: [p for p in plants_by_row[row] if plant_type(p) == 3 and not p["squished"]]
                for row in rows}
    mine_rows = {row for row in rows if any(plant_type(p) == 4 for p in plants_by_row[row])}
    uncovered_rows = {row for row in rows if zombies_by_row[row] and not attackers[row]}
    ready_mowers = {defense["row"] for defense in observation["defenses"] if defense["state"] == 1}
    sunflowers = [p for p in plants if plant_type(p) == 1]
    active_shooter_rows = sum(bool(attackers[row]) for row in rows)
    shooter_rows_built = sum(any(plant_type(p) in (0, 5) and not p["squished"] and
                                 p["health"] > 0.1 * max(1, p["max_health"])
                                 for p in plants_by_row[row]) for row in rows)
    shooter_rows_before_economy = min(len(rows), max(2, math.ceil(zombie_multiplier)))
    sunflower_goal = (min(8, 2 + math.ceil(zombie_multiplier - 1.0))
                      if observation["wave"] < 3 else min(8, 3 + round(zombie_multiplier - 1.0)))
    if observation["wave"] >= 10:
        sunflower_goal = 8
    elif active_shooter_rows == len(rows):
        sunflower_goal = max(sunflower_goal, 5)
    mower_defense_front = 500.0 + 40.0 * (zombie_multiplier - 1.0)
    sunflower_safety_front = mower_defense_front
    rake_rows = {item["row"] for item in observation["grid_items"] if item["type"] == 11 and item["state"] == 26}
    reserve_target = round(100.0 + 25.0 * (zombie_multiplier - 1.0))
    coverage_bonus = 24.0 + 12.0 * (zombie_multiplier - 1.0)
    sun = observation["sun"]
    scores: list[tuple[dict[str, Any], float]] = []

    def add(action: dict[str, Any], score: float) -> None:
        scores.append((action, score))

    for placement in legal["plants"]:
        packet = packets[placement["packet"]]
        seed_type = packet["imitater_type"] if packet["type"] == 48 else packet["type"]
        row, col = placement["row"], placement["col"]
        lane_zombies = zombies_by_row[row]
        front = min((z["x"] for z in lane_zombies), default=9999.0)
        urgency = max((zombie_urgency(z) for z in lane_zombies), default=0.0)
        pressure = sum((1.0 + zombie_urgency(z)) * min(3.0, max(0.5, zombie_hp(z) / 200.0)) for z in lane_zombies)
        count_attackers = len(attackers[row])
        mower_risk = max(0.0, min(1.0, (450.0 - front) / 300.0)) if row in ready_mowers else 0.0
        plant_x = 80 + 80 * col
        action = {"type": "plant", **placement}
        cost = packet["cost"]
        if sun < cost:
            continue
        reserve_penalty = 100.0 if sun - cost < reserve_target else 0.0

        if seed_type in (0, 5):
            if front <= emergency_front or not attackers[row]:
                reserve_penalty = 0.0
            if seed_type == 0 and not attackers[row] and front > mower_defense_front and sun < 150:
                reserve_penalty = 200.0
            targets = [z for z in lane_zombies if z["x"] > plant_x + 80]
            if targets:
                useful_count = sum(1 for z in targets if z["x"] > plant_x + 180)
                value = (25.0 if seed_type == 5 else 22.0) + 10.0 * urgency
                value += min(30.0, pressure * 4.0) / (1 + count_attackers * 0.5)
                value += min(4.0, useful_count * 0.7) - count_attackers * 4.0
                gap = min(z["x"] - plant_x for z in targets)
                value += max(-5.0, 7.0 - abs(gap - safe_shot_gap) / 35.0) + 12.0 * mower_risk
                value += coverage_bonus if row in uncovered_rows else -coverage_bonus * len(uncovered_rows)
                needed_attackers = min(3, max(1, (len(targets) + 1) // 2))
                value += 20.0 * max(0, needed_attackers - count_attackers)
                extra_damage = 0.0
                for zombie in targets:
                    shots = expected_shots(zombie, plant_x) * (1.75 if seed_type == 5 else 1.0)
                    current = min(zombie_hp(zombie), count_attackers * shots * 20.0)
                    added = min(zombie_hp(zombie), (count_attackers + 1) * shots * 20.0)
                    extra_damage += added - current
                value += min(200.0, extra_damage * 0.5)
                if seed_type == 5:
                    value += 40.0 + min(8.0, pressure) + 8.0 * (zombie_multiplier - 1.0)
                    reserve_penalty = 0.0
                if row in ready_mowers and front < mower_defense_front:
                    value += 600.0
                    reserve_penalty = 0.0
                add(action, value - cost * 0.01 - reserve_penalty)
            elif not lane_zombies and count_attackers == 0:
                value = (8.0 if seed_type == 0 else 5.0) + (2.0 if col == 2 else 0.0)
                if seed_type == 0:
                    value += 12.0 * (zombie_multiplier - 1.0)
                add(action, value - cost * 0.01 - reserve_penalty)
            continue

        if seed_type == 3 and lane_zombies and not wallnuts[row] and plant_x < front - 35:
            value = 6.0 + 11.0 * urgency + min(4.0, pressure) + 10.0 * mower_risk
            value += max(-5.0, 7.0 - abs((front - plant_x) - safe_shot_gap) / 35.0)
            value -= cost * 0.01
            if row in ready_mowers and front < emergency_front + 70 and (attackers[row] or front < 200):
                reserve_penalty = 0.0
            add(action, (value if attackers[row] else value - 5.0) - reserve_penalty)
        elif seed_type == 4 and lane_zombies and row not in mine_rows:
            viable = []
            for zombie in lane_zombies:
                speed = abs(zombie.get("velocity_x", 0.0))
                distance = zombie["x"] - plant_x - 25
                if speed > 0.01 and distance > 0:
                    arrival = distance / speed
                    if arrival >= 3200:
                        viable.append(arrival)
            if viable:
                arrival = min(viable)
                value = 3.0 + min(3.0, len(lane_zombies)) * 1.2 + 2.0 * urgency
                value -= abs(arrival - 2600.0) / 900.0 + min(8.0, count_attackers * 5.0)
                if not attackers[row]:
                    value += 16.0 + max(0.0, 8.0 - abs(arrival - 2600.0) / 350.0)
                    if observation["wave"] <= 1:
                        reserve_penalty = 0.0
                add(action, value - cost * 0.01 - reserve_penalty)
        elif seed_type == 2:
            targets = [z for z in zombies if abs(z["row"] - row) <= 1 and abs(z["x"] - plant_x) <= 115]
            if targets:
                value = 8.0 + sum(
                    15.0 + 20.0 * zombie_urgency(z) + min(8.0, zombie_hp(z) / 180.0)
                    + 10.0 * (zombie_multiplier - 1.0)
                    for z in targets
                )
                if len(targets) >= 2 and max(zombie_urgency(z) for z in targets) > 0.35:
                    reserve_penalty = 0.0
                if any(z["row"] in ready_mowers and z["x"] < emergency_front - 70 for z in targets):
                    value += 14.0
                    reserve_penalty = 0.0
                if any(z["row"] in ready_mowers and z["x"] < mower_defense_front for z in targets):
                    value += 50.0
                    reserve_penalty = 0.0
                uncovered_targets = sum(z["row"] in uncovered_rows for z in targets)
                value += coverage_bonus * uncovered_targets
                if not uncovered_targets and uncovered_rows:
                    value -= coverage_bonus * len(uncovered_rows)
                add(action, value - cost * 0.015 - reserve_penalty)
        elif seed_type == 1:
            opening_sunflowers = max(1, round(2.0 / zombie_multiplier))
            if sunflower_placements >= opening_sunflowers and shooter_rows_built < shooter_rows_before_economy:
                continue
            if len(sunflowers) >= sunflower_goal:
                continue
            row_front = min((z["x"] for z in lane_zombies), default=9999.0)
            if lane_zombies and row_front < sunflower_safety_front and not any(p["col"] > col for p in wallnuts[row]):
                continue
            reserve_penalty = 0.0
            safety = max(0.0, min(1.0, (row_front - 420.0) / 300.0))
            economy = max(-3.0, 16.0 - len(sunflowers) * 3.5)
            economy += max(0, sunflower_goal - max(len(sunflowers), sunflower_placements)) * 8.0
            add(action, economy + safety * 4.0 + (1.0 if col == 1 else 0.0) - cost * 0.01)

    for col, row in legal["shovels"]:
        plant = next((p for p in plants_by_row[row] if p["col"] == col), None)
        if plant is None:
            continue
        durability = plant["health"] / max(1, plant["max_health"])
        nearest = min((z["x"] for z in zombies_by_row[row]), default=9999.0)
        if plant.get("squished") or (durability < 0.12 and nearest > 620):
            add({"type": "shovel", "col": col, "row": row}, 4.0 - durability)

    if not zombies:
        wait_action, wait_score = {"type": "wait_decision", "max_ticks": WAIT_DECISION_TICKS}, 3.0
    else:
        nearest = min(zombie["x"] for zombie in zombies)
        mower_front = min((zombie["x"] for zombie in zombies if zombie["row"] in ready_mowers), default=9999.0)
        if mower_front < emergency_front or nearest < 360:
            wait_action = {"type": "wait", "ticks": 60}
        elif mower_front < mower_defense_front or nearest < 540:
            wait_action = {"type": "wait", "ticks": 150}
        else:
            wait_action = {"type": "wait_decision", "max_ticks": WAIT_DECISION_TICKS}
        wait_score = 0.0 if nearest < 540 else 4.0
        if any(zombie["row"] in ready_mowers and zombie["x"] < emergency_front for zombie in zombies):
            wait_score -= 6.0
        if any(zombie["row"] in rake_rows and zombie["x"] > 600 for zombie in zombies):
            wait_score += 3.0
    add(wait_action, wait_score)

    scores.sort(key=lambda item: item[1], reverse=True)
    candidates = scores[:8]
    return TeacherAdvice(action=candidates[0][0], candidates=candidates)


class TeacherPolicy:
    def __init__(self) -> None:
        self.sunflower_placements = 0

    def advice(self, observation: dict[str, Any]) -> TeacherAdvice:
        return teacher_advice(observation, self.sunflower_placements)

    def record_action(self, observation: dict[str, Any], action: dict[str, Any]) -> None:
        if action.get("type") != "plant":
            return
        packet = next((item for item in observation["packets"] if item["index"] == action["packet"]), None)
        if packet is None:
            return
        seed_type = packet["imitater_type"] if packet["type"] == 48 else packet["type"]
        if seed_type == 1:
            self.sunflower_placements += 1
