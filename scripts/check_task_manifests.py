"""Generate the enumerable training task pool and check its frozen held-out split."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = ROOT / "artifacts/task_family/train.json"
HELDOUT_PATH = ROOT / "artifacts/task_family/heldout.json"
SEEDS_PER_TASK = 64
TRAIN_SEED_START = 60000
HELDOUT_SEED_START = 50000
TERRAIN_ORDER = ("day", "night", "pool", "fog", "roof")

FROZEN_SEED_FILES = (
    "artifacts/adventure2_level7/seeds/dev.json",
    "artifacts/adventure2_level7/seeds/test.json",
    "artifacts/multiterrain/seeds/day_level8.json",
    "artifacts/multiterrain/seeds/night_level12.json",
    "artifacts/multiterrain/seeds/pool_level26.json",
    "artifacts/multiterrain/seeds/fog_level31.json",
    "artifacts/multiterrain/seeds/roof_level41.json",
)
RESERVED_SEED_RANGES = (
    {"name": "train", "first_seed": 0, "count": 64},
    {"name": "dagger", "first_seed": 10000, "count": 64},
    {"name": "value_bootstrap", "first_seed": 20000, "count": 32},
    {"name": "value_refinement", "first_seed": 21000, "count": 32},
    {"name": "development", "first_seed": 30000, "count": 256},
    {"name": "final_test", "first_seed": 40000, "count": 1024},
)

TRAIN_BASES = {
    "day": {"levels": [1, 2, 3, 4], "decks": [[0, 1, 2, 3, 4, 5], [0, 1, 3, 4, 7, 9],
                                              [0, 1, 2, 3, 4, 5, 7]]},
    "night": {"levels": [11, 13, 14, 16], "decks": [[8, 9, 10, 12, 14, 15], [8, 9, 10, 12, 13, 14]]},
    "pool": {"levels": [21, 22, 23, 24], "decks": [[0, 1, 2, 3, 4, 16], [0, 1, 3, 4, 16, 19]]},
    "fog": {"levels": [32, 33, 34, 36], "decks": [[8, 9, 10, 14, 15, 16], [8, 9, 10, 13, 14, 16]]},
    "roof": {"levels": [42, 43, 44, 49], "decks": [[0, 1, 2, 3, 4, 33], [0, 1, 3, 4, 7, 33]]},
}
TRAIN_MUTATIONS = (
    {"deck_index": 0, "zombie_count_multiplier": 1.0, "wave_cap": 1},
    {"deck_index": 1, "zombie_count_multiplier": 1.5, "wave_cap": 3},
    {"deck_index": 0, "zombie_count_multiplier": 2.0, "wave_cap": 3},
    {"deck_index": 1, "zombie_count_multiplier": 1.0, "wave_cap": 5},
)

HELDOUT_BASES = {
    "day": {"levels": [6, 7, 9, 10], "decks": TRAIN_BASES["day"]["decks"]},
    "night": {"levels": [17, 18, 19, 20], "decks": TRAIN_BASES["night"]["decks"]},
    "pool": {"levels": [27, 28, 29, 30], "decks": TRAIN_BASES["pool"]["decks"]},
    "fog": {"levels": [37, 38, 39, 40], "decks": TRAIN_BASES["fog"]["decks"]},
    "roof": {"levels": [45, 46, 47, 48], "decks": TRAIN_BASES["roof"]["decks"]},
}
HELDOUT_VARIANTS = (
    {"deck_index": 1, "zombie_count_multiplier": 1.0, "wave_cap": 3},
    {"deck_index": 0, "zombie_count_multiplier": 1.5, "wave_cap": 3},
    {"deck_index": 1, "zombie_count_multiplier": 2.0, "wave_cap": 5},
    {"deck_index": 0, "zombie_count_multiplier": 1.0, "wave_cap": 3},
)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _make_tasks(bases: dict[str, Any], variants: tuple[dict[str, Any], ...], seed_start: int,
                split: str) -> list[dict[str, Any]]:
    tasks = []
    for terrain in TERRAIN_ORDER:
        base = bases[terrain]
        if len(base["levels"]) != len(variants):
            raise ValueError(f"{terrain}: each mutation needs one level")
        for index, (level, variant) in enumerate(zip(base["levels"], variants, strict=True)):
            task_number = len(tasks)
            tasks.append({
                "task_id": f"{split}_{terrain}_{index + 1}",
                "terrain": terrain,
                "level": level,
                "deck": list(base["decks"][variant["deck_index"]]),
                "zombie_count_multiplier": variant["zombie_count_multiplier"],
                "wave_cap": variant["wave_cap"],
                "sun_start": 50,
                "preplanted": [],
                "playthrough": 2,
                "seeds": list(range(seed_start + task_number * SEEDS_PER_TASK,
                                     seed_start + (task_number + 1) * SEEDS_PER_TASK)),
            })
    return tasks


def build_train_manifest() -> dict[str, Any]:
    generation = {
        "seed_start": TRAIN_SEED_START,
        "seeds_per_task": SEEDS_PER_TASK,
        "bases": TRAIN_BASES,
        "mutations": list(TRAIN_MUTATIONS),
    }
    return {
        "schema_version": 1,
        "split": "train",
        "task_definition": ["level", "deck", "zombie_count_multiplier", "wave_cap", "sun_start", "preplanted"],
        "generation": generation,
        "tasks": _make_tasks(TRAIN_BASES, TRAIN_MUTATIONS, TRAIN_SEED_START, "train"),
    }


def _frozen_seed_sets() -> list[dict[str, Any]]:
    result = []
    for relative_path in FROZEN_SEED_FILES:
        path = ROOT / relative_path
        data = json.loads(path.read_text(encoding="utf-8"))
        result.append({
            "path": relative_path,
            "role": data["role"],
            "level": data["level"],
            "playthrough": data["playthrough"],
            "first_seed": data["first_seed"],
            "count": data["count"],
            "included_in_heldout": False,
        })
    return result


def build_heldout_manifest() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "split": "heldout",
        "frozen": True,
        "task_definition": ["level", "deck", "zombie_count_multiplier", "wave_cap", "sun_start", "preplanted"],
        "sun_start_semantics": "All tasks record the current Adventure-II reset value of 50; RESET_V2 does not override it.",
        "seed_start": HELDOUT_SEED_START,
        "seeds_per_task": SEEDS_PER_TASK,
        "reserved_seed_ranges": list(RESERVED_SEED_RANGES),
        "frozen_seed_sets": _frozen_seed_sets(),
        "tasks": _make_tasks(HELDOUT_BASES, HELDOUT_VARIANTS, HELDOUT_SEED_START, "heldout"),
    }


def terrain_for_level(level: int) -> str:
    if 1 <= level <= 10:
        return "day"
    if 11 <= level <= 20:
        return "night"
    if 21 <= level <= 30:
        return "pool"
    if 31 <= level <= 40:
        return "fog"
    if 41 <= level <= 49:
        return "roof"
    raise AssertionError(f"level {level} does not belong to the five task terrains")


def _task_signature(task: dict[str, Any]) -> tuple[Any, ...]:
    return (
        task["level"], tuple(task["deck"]), task["zombie_count_multiplier"], task["wave_cap"],
        task["sun_start"], tuple(tuple(item) for item in task["preplanted"]),
    )


def _validate_task(task: dict[str, Any]) -> None:
    required = {"task_id", "terrain", "level", "deck", "zombie_count_multiplier", "wave_cap",
                "sun_start", "preplanted", "playthrough", "seeds"}
    if set(task) != required:
        raise AssertionError(f"{task.get('task_id')}: incomplete task fields")
    if terrain_for_level(task["level"]) != task["terrain"]:
        raise AssertionError(f"{task['task_id']}: terrain does not match level")
    if task["playthrough"] != 2 or not 1 <= task["sun_start"] <= 10000:
        raise AssertionError(f"{task['task_id']}: unsupported playthrough or sun_start")
    # 上限 12 = 模型的 packet 头容量（见 pvz_agent_model 的 previous_packet_embedding(12, 64)）。
    # 早先写死 6，挡住了"加一张牌"这类实验（实测加双发射手 +5 个百分点）。
    if not task["deck"] or len(task["deck"]) > 12 or len(set(task["deck"])) != len(task["deck"]):
        raise AssertionError(f"{task['task_id']}: invalid deck")
    if any(type(seed) is not int or not 0 <= seed <= 48 for seed in task["deck"]):
        raise AssertionError(f"{task['task_id']}: invalid deck seed type")
    if not 1.0 <= task["zombie_count_multiplier"] <= 10.0:
        raise AssertionError(f"{task['task_id']}: invalid zombie_count_multiplier")
    if type(task["wave_cap"]) is not int or not 1 <= task["wave_cap"] <= 50:
        raise AssertionError(f"{task['task_id']}: invalid wave_cap")
    if any(len(plant) != 3 or not 0 <= plant[0] <= 48 or not 0 <= plant[1] <= 5 or not 0 <= plant[2] <= 8
           for plant in task["preplanted"]):
        raise AssertionError(f"{task['task_id']}: invalid preplanted plants")
    seeds = task["seeds"]
    if len(seeds) < 64 or len(seeds) != len(set(seeds)) or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise AssertionError(f"{task['task_id']}: each task needs at least 64 unique seeds")


def _seed_set(data: dict[str, Any]) -> set[int]:
    return set(range(data["first_seed"], data["first_seed"] + data["count"]))


def check_manifests(train: dict[str, Any], heldout: dict[str, Any]) -> dict[str, Any]:
    expected_train = build_train_manifest()
    if train != expected_train:
        raise AssertionError("training manifest differs from its enumerable mutation generator")
    if heldout.get("schema_version") != 1 or heldout.get("split") != "heldout" or heldout.get("frozen") is not True:
        raise AssertionError("held-out manifest header is invalid")
    if heldout.get("task_definition") != expected_train["task_definition"]:
        raise AssertionError("held-out task definition is incomplete")
    if heldout.get("reserved_seed_ranges") != list(RESERVED_SEED_RANGES):
        raise AssertionError("reserved historical seed ranges are missing or changed")
    if heldout.get("frozen_seed_sets") != _frozen_seed_sets():
        raise AssertionError("frozen development/final-test seed relationships are missing or changed")

    train_tasks, heldout_tasks = train["tasks"], heldout.get("tasks", [])
    for task in train_tasks + heldout_tasks:
        _validate_task(task)
    if len({task["task_id"] for task in train_tasks}) != len(train_tasks):
        raise AssertionError("training task IDs are not unique")
    if len({task["task_id"] for task in heldout_tasks}) != len(heldout_tasks):
        raise AssertionError("held-out task IDs are not unique")

    train_signatures = {_task_signature(task) for task in train_tasks}
    overlap = train_signatures & {_task_signature(task) for task in heldout_tasks}
    if overlap:
        raise AssertionError(f"training and held-out task parameters overlap: {next(iter(overlap))}")
    train_seeds = {seed for task in train_tasks for seed in task["seeds"]}
    heldout_seeds = {seed for task in heldout_tasks for seed in task["seeds"]}
    if train_seeds & heldout_seeds:
        raise AssertionError("training and held-out seed IDs overlap")
    if len(train_seeds) != sum(len(task["seeds"]) for task in train_tasks):
        raise AssertionError("training seed IDs are reused between tasks")
    if len(heldout_seeds) != sum(len(task["seeds"]) for task in heldout_tasks):
        raise AssertionError("held-out seed IDs are reused between tasks")

    reserved = set()
    for item in RESERVED_SEED_RANGES:
        reserved.update(range(item["first_seed"], item["first_seed"] + item["count"]))
    frozen = set()
    for item in heldout["frozen_seed_sets"]:
        frozen.update(_seed_set(item))
    if heldout_seeds & (reserved | frozen):
        raise AssertionError("held-out seeds overlap an existing frozen seed set")
    if train_seeds & reserved:
        raise AssertionError("training seeds overlap a reserved historical seed range")

    terrain_counts = Counter(task["terrain"] for task in heldout_tasks)
    if any(terrain_counts[terrain] < 4 for terrain in TERRAIN_ORDER):
        raise AssertionError(f"held-out task count per terrain is insufficient: {dict(terrain_counts)}")
    return {
        "train_tasks": len(train_tasks),
        "heldout_tasks": len(heldout_tasks),
        "heldout_tasks_by_terrain": dict(sorted(terrain_counts.items())),
        "seeds_per_task": min(len(task["seeds"]) for task in heldout_tasks),
        "train_seed_count": len(train_seeds),
        "heldout_seed_count": len(heldout_seeds),
        "theta_overlap_count": 0,
        "seed_overlap_count": 0,
        "reserved_seed_overlap_count": 0,
    }


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_environment(train: dict[str, Any], heldout: dict[str, Any], resource_dir: str) -> None:
    sys.path.insert(0, str(ROOT / "python"))
    from pvz_env import PvZEnv, TaskSpec

    failures = []
    with PvZEnv(resource_dir) as env:
        for task in train["tasks"] + heldout["tasks"]:
            spec = TaskSpec(
                level=task["level"], seed=task["seeds"][0], playthrough=2,
                zombie_count_multiplier=task["zombie_count_multiplier"],
                wave_cap=task["wave_cap"], preplanted=tuple(tuple(item) for item in task["preplanted"]),
            )
            observation, _ = env.reset(deck=task["deck"], task=spec)
            if observation["sun"] != task["sun_start"] or observation["wave_count"] != task["wave_cap"]:
                failures.append(
                    f"{task['task_id']}: reset sun={observation['sun']} wave_count={observation['wave_count']} "
                    f"expected sun={task['sun_start']} wave_count={task['wave_cap']}"
                )
    if failures:
        raise AssertionError("; ".join(failures))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=TRAIN_PATH)
    parser.add_argument("--heldout", type=Path, default=HELDOUT_PATH)
    parser.add_argument("--write-train", type=Path)
    parser.add_argument("--write-heldout", type=Path)
    parser.add_argument("--resource-dir")
    args = parser.parse_args()
    if args.write_train:
        _write_json(args.write_train, build_train_manifest())
    if args.write_heldout:
        if args.write_heldout.resolve() == HELDOUT_PATH.resolve():
            parser.error("write a temporary candidate first; the frozen held-out manifest must not be regenerated in place")
        _write_json(args.write_heldout, build_heldout_manifest())
    if args.write_train or args.write_heldout:
        return
    try:
        train, heldout = _load(args.train), _load(args.heldout)
        metrics = check_manifests(train, heldout)
        if args.resource_dir:
            _validate_environment(train, heldout, args.resource_dir)
            metrics["all_task_resets_match_manifest"] = True
        print(json.dumps({"gate_result": "pass", "metrics": metrics}, sort_keys=True))
    except (AssertionError, OSError, ValueError, KeyError) as error:
        print(json.dumps({"gate_result": "fail", "error": str(error)}), file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
