"""A deliberately simple hand-written policy, used as a difficulty probe.

The point is not to write a good bot; it is to answer one question the win rate
cannot: is Adventure-II level 7 (30 waves, 6 base cards) winnable at all, or is
the 0/1024 result explained by the task?  If a 40-line rule bot gets much further
than the search teacher, the teacher is the problem.

    python scripts/scripted_baseline.py --resource-dir <PvZ 1.2.0.1073 dir> --seeds 30000 20000
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from pvz_env import PvZEnv, TaskSpec, training_task  # noqa: E402

DECK = (0, 1, 2, 3, 4, 5)
LEVEL = 7
PEASHOOTER, SUNFLOWER, CHERRY, WALLNUT, POTATO, SNOWPEA = range(6)


def rows_with(plants: list[dict], kind: int) -> Counter:
    return Counter(plant["row"] for plant in plants if plant["type"] == kind)


def choose(observation: dict) -> dict:
    legal = observation["legal_actions"]["plants"]
    if not legal:
        return {"type": "wait", "ticks": 60}
    packets = observation["packets"]
    by_type: dict[int, list[dict]] = {}
    for placement in legal:
        kind = packets[placement["packet"]]["type"]
        by_type.setdefault(kind, []).append(placement)
    plants = observation["plants"]
    zombies = observation["zombies"]
    count = Counter(plant["type"] for plant in plants)
    occupied = {(plant["col"], plant["row"]) for plant in plants}

    def pick(kind, filter_fn, key_fn):
        options = [item for item in by_type.get(kind, []) if filter_fn(item)]
        if not options:
            return None
        return {"type": "plant", **max(options, key=key_fn)}

    def row_threat(row: int) -> float:
        xs = [z["x"] for z in zombies if z["row"] == row]
        return min(xs) if xs else 9999.0

    # 1. emergency wall-nut in front of anything close to the house
    for row in range(6):
        if row_threat(row) < 260 and not any(
                p["type"] == WALLNUT and p["row"] == row and p["col"] >= 5 for p in plants):
            action = pick(WALLNUT, lambda item: item["row"] == row and item["col"] >= 5,
                          lambda item: item["col"])
            if action:
                return action

    # 2. cherry bomb on a clustered breach
    cluster = [z for z in zombies if z["x"] < 320]
    if len(cluster) >= 4:
        row = Counter(z["row"] for z in cluster).most_common(1)[0][0]
        action = pick(CHERRY, lambda item: item["row"] == row, lambda item: -item["col"])
        if action:
            return action

    # 3. economy first: two columns of sunflowers at the back
    if count[SUNFLOWER] < 10:
        action = pick(SUNFLOWER, lambda item: item["col"] <= 2,
                      lambda item: (-item["col"], rows_with(plants, SUNFLOWER)[item["row"]]))
        if action:
            return action

    # 4. offence: peashooters (with a few snow peas) in the middle columns
    shooters = count[PEASHOOTER] + count[SNOWPEA]
    if shooters < 14:
        if count[SNOWPEA] < 4 and observation["sun"] >= 275:
            action = pick(SNOWPEA, lambda item: 2 <= item["col"] <= 5,
                          lambda item: -row_threat(item["row"]))
            if action:
                return action
        action = pick(PEASHOOTER, lambda item: 2 <= item["col"] <= 5,
                      lambda item: (-row_threat(item["row"]), -item["col"]))
        if action:
            return action

    # 5. cheap early mines on the lane the first zombies walk down
    if count[POTATO] < 4 and observation["wave"] <= 4:
        action = pick(POTATO, lambda item: item["col"] >= 6, lambda item: -row_threat(item["row"]))
        if action:
            return action

    return {"type": "wait", "ticks": 60}


def run(env: PvZEnv, seed: int, level: int = LEVEL, max_actions: int = 4000,
        task: TaskSpec | None = None) -> dict:
    observation, _ = env.reset(deck=DECK, task=task or training_task(seed, level))
    initial_plants = observation["plants"]
    initial_sun = observation["sun"]
    initial_off_board_zombies = sum(not zombie["on_board"] for zombie in observation["zombies"])
    if initial_off_board_zombies == 0 or observation["enemy_zombies_on_screen"]:
        raise AssertionError("preview zombies must be observed but excluded from enemy presence")
    actions = 0
    mower_triggered = 0
    while not observation["terminal"] and actions < max_actions:
        action = choose(observation)
        observation, _, done, _, info = env.step(action)
        if not info.get("ok"):
            # wait whenever the chosen placement was rejected
            observation, _, done, _, info = env.step({"type": "wait", "ticks": 60})
        mower_triggered += info["events"].get("mower_triggered", 0)
        actions += 1
        if done:
            break
    result = int(observation["result"])
    level_lost = bool(info.get("events", {}).get("level_lost"))
    if result == 1:
        terminal_reason = "won"
    elif result == 2 and level_lost:
        terminal_reason = "zombie_breach"
    elif result == 2:
        terminal_reason = "lost_other"
    elif observation["terminal"]:
        terminal_reason = "other_terminal"
    else:
        terminal_reason = "action_limit"
    if result == 1 and observation["enemy_zombies_on_screen"]:
        raise AssertionError("a winning terminal state must have no enemy zombies on screen")
    return {
        "seed": seed,
        "level": level,
        "won": result == 1,
        "result": result,
        "terminal_reason": terminal_reason,
        "terminal": bool(observation["terminal"]),
        "wave": observation["wave"],
        "wave_count": observation["wave_count"],
        "tick": observation["tick"],
        "actions": actions,
        "initial_plants": initial_plants,
        "initial_sun": initial_sun,
        "initial_off_board_zombies": initial_off_board_zombies,
        "initial_enemy_zombies_on_screen": False,
        "final_enemy_zombies_on_screen": bool(observation["enemy_zombies_on_screen"]),
        "level_lost_event": level_lost,
        "mower_triggered": mower_triggered,
        "plants": Counter(plant["type"] for plant in observation["plants"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resource-dir", required=True)
    parser.add_argument("--level", type=int, default=LEVEL)
    parser.add_argument("--seeds", type=int, nargs="+", default=[30000])
    args = parser.parse_args()
    env = PvZEnv(args.resource_dir)
    for seed in args.seeds:
        result = run(env, seed, args.level)
        print(f"level {args.level} seed {seed}: won={result['won']} result={result['result']} "
              f"reason={result['terminal_reason']} level_lost={result['level_lost_event']} "
              f"mowers={result['mower_triggered']} terminal={result['terminal']} "
              f"wave {result['wave']}/{result['wave_count']} "
              f"tick={result['tick']} actions={result['actions']} "
              f"previews={result['initial_off_board_zombies']} "
              f"enemy_query={result['initial_enemy_zombies_on_screen']}"
              f"->{result['final_enemy_zombies_on_screen']} board={dict(result['plants'])}")
    env.close()


if __name__ == "__main__":
    main()
