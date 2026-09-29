"""Trace what the SearchTeacher actually does on a single seed.

The teacher scores 0/256 on development; the point of this script is to make the
failure visible as a decision log instead of a win rate: what it plants, when it
plants it, what the search thought each candidate was worth, and which wave kills
it.

    python scripts/diagnose_teacher.py --resource-dir <PvZ 1.2.0.1073 dir> --seed 30000
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from pvz_env import PvZEnv, training_task  # noqa: E402
from pvz_search import SearchTeacher  # noqa: E402

SEED_NAMES = {
    0: "Peashooter", 1: "Sunflower", 2: "CherryBomb", 3: "Wallnut",
    4: "PotatoMine", 5: "SnowPea", 6: "Chomper", 7: "Repeater",
}
DECK = (0, 1, 2, 3, 4, 5)


def describe(action: dict[str, Any]) -> str:
    kind = action.get("type")
    if kind == "plant":
        return f"plant {SEED_NAMES.get(action['packet'], action['packet'])} @ r{action['row']}c{action['col']}"
    if kind == "shovel":
        return f"shovel @ r{action['row']}c{action['col']}"
    return f"wait {action['ticks']}"


def plants_summary(observation: dict[str, Any]) -> str:
    counts = Counter(SEED_NAMES.get(p.get("type"), str(p.get("type"))) for p in observation["plants"])
    return ",".join(f"{name}x{n}" for name, n in sorted(counts.items())) or "-"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resource-dir", required=True)
    parser.add_argument("--seed", type=int, default=30000)
    parser.add_argument("--level", type=int, default=7)
    parser.add_argument("--max-actions", type=int, default=400)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    env = PvZEnv(args.resource_dir)
    teacher = SearchTeacher(env)  # no value model: cold-start evaluator
    task = training_task(args.seed, args.level)
    observation, _ = env.reset(deck=DECK, task=task)
    print(f"wave_count={observation.get('wave_count')} tick={observation['tick']} "
          f"sun={observation['sun']} scene={observation.get('loadout_context', {}).get('scene')}")
    print(f"roster={observation.get('loadout_context', {}).get('zombie_roster')}")

    started = time.perf_counter()
    actions = 0
    while not observation["terminal"] and actions < args.max_actions:
        advice = teacher.advice(observation)
        zombies = observation["zombies"]
        nearest = min((z["x"] for z in zombies), default=9999.0)
        line = (
            f"#{actions:3d} t={observation['tick']:6d} w={observation['wave']:2d} "
            f"sun={observation['sun']:4d} z={len(zombies):2d} near={nearest:6.0f} "
            f"| {plants_summary(observation):40s} | -> {describe(advice.action)}"
        )
        top = ", ".join(
            f"{describe(a)}={s:+.3f}" for a, s in advice.candidates[:3]
        )
        print(line)
        if args.verbose:
            print(f"      sims={advice.simulation_count} margin={advice.best_second_margin} "
                  f"top: {top}")
        observation, _, done, _, info = env.step(advice.action)
        if not info.get("ok"):
            print(f"ILLEGAL ACTION: {advice.action}")
            break
        actions += 1
        if done:
            break
    elapsed = time.perf_counter() - started
    print(f"\nterminal={observation['terminal']} result={observation.get('result')} "
          f"wave={observation['wave']}/{observation.get('wave_count')} "
          f"tick={observation['tick']} actions={actions} wall={elapsed:.1f}s")
    print(f"final board: {plants_summary(observation)}")
    env.close()


if __name__ == "__main__":
    main()
