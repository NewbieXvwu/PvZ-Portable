"""Public-only row context; it describes choices and never selects an action."""
from __future__ import annotations

from collections import defaultdict


INPUT_ON_BOARD = 1
INPUT_ROW_CONTEXT = 2
INPUT_TARGET_RELATIONS = 4


def require_public_fields(observation: dict, flags: int) -> None:
    if type(flags) is not int or not 0 <= flags <= 7:
        raise ValueError("input_flags must be an explicit integer bitmask from 0 to 7")
    if flags & (INPUT_ON_BOARD | INPUT_ROW_CONTEXT):
        if any(type(z.get("on_board")) is not bool for z in observation["zombies"]):
            raise ValueError("new input requires explicit public zombie.on_board")
    if flags & INPUT_TARGET_RELATIONS:
        ids = [z.get("id") for z in observation["zombies"]]
        if (any(type(key) is not int or key in (0, -1) for key in ids)
                or len(ids) != len(set(ids))):
            raise ValueError("target relations require unique non-null public zombie IDs")


def effective_plant_type(entity: dict) -> int:
    kind = int(entity["type"])
    return int(entity["imitater_type"]) if kind == 48 and int(entity.get("imitater_type", -1)) >= 0 else kind


def row_context(observation: dict, economic: set | frozenset, shooters: set | frozenset) -> tuple[list[tuple], list[dict]]:
    enemies = [z for z in observation["zombies"] if z["on_board"] and 0 <= z["row"] < 6]
    legal_by_row = [set() for _ in range(6)]
    cells_by_row = [set() for _ in range(6)]
    for action in observation["legal_actions"].get("plants", ()):
        row = action["row"]
        if 0 <= row < 6:
            legal_by_row[row].add(action["packet"])
            cells_by_row[row].add(action["col"])
    roles = (economic, shooters, frozenset({16, 33}))
    additions = []
    for row in range(6):
        zombies = [z for z in enemies if z["row"] == row]
        plants = [p for p in observation["plants"] if p["row"] == row]
        values = [len(zombies) / 10, sum(z["helm_health"] > 0 for z in zombies) / 10,
                  sum(z["shield_health"] > 0 for z in zombies) / 10,
                  sum(z["is_eating"] for z in zombies) / 10,
                  sum(z["chilled"] > 0 or z["ice_trap"] > 0 for z in zombies) / 10,
                  sum(effective_plant_type(p) in economic for p in plants) / 10,
                  len(cells_by_row[row]) / 9]
        for role in roles:
            packets = [p for p in observation["packets"] if effective_plant_type(p) in role]
            cooldowns = [max(0, p["cooldown"]) for p in packets]
            low, high = (min(cooldowns), max(cooldowns)) if cooldowns else (0, 0)
            values.extend((len(packets) / 10, low / (low + 3000), high / (high + 3000),
                           sum(p["active"] and p["cooldown"] == 0 and p["cost"] <= observation["sun"]
                               for p in packets) / 10,
                           sum(p["index"] in legal_by_row[row] for p in packets) / 10))
        additions.append(tuple(values))
    grouped = defaultdict(list)
    for zombie in enemies:
        grouped[zombie["row"], zombie["type"]].append(zombie)
    composition = []
    for (row, kind), zombies in sorted(grouped.items()):
        composition.append({"row": row, "type": kind,
                            "values": (len(zombies) / 10,
                                       sum(z["body_health"] for z in zombies) / 6000,
                                       sum(z["helm_health"] for z in zombies) / 6000,
                                       sum(z["shield_health"] for z in zombies) / 6000,
                                       min(z["x"] for z in zombies) / 900,
                                       sum(z["velocity_x"] for z in zombies) / len(zombies) / 5)})
    return additions, composition
