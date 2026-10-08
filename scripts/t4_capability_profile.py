"""Profile a gameplay checkpoint on the frozen task-family manifests."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import random
from pathlib import Path
import statistics
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from pvz_agent_model import (  # noqa: E402
    GameplayModelV1,
    model_architecture_version,
    configure_torch_threads,
    predict_action,
    select_action,
)
from pvz_common import ENV_PROTOCOL_VERSION, OBSERVATION_VERSION, TASK_VERSION, sha256_file  # noqa: E402
from pvz_env import PvZEnv, TaskSpec, profile_for_deck  # noqa: E402
from pvz_event_env import policy_env, require_policy_env  # noqa: E402
from pvz_wait_events import summarize_wait_records, validate_wait_result  # noqa: E402
from pvz_seed_jobs import atomic_json  # noqa: E402
from pvz_value import SEARCH_LABEL_VERSION, VALUE_SEMANTICS  # noqa: E402
import scripted_baseline  # noqa: E402
from check_task_manifests import check_manifests  # noqa: E402

TRAIN_PATH = ROOT / "artifacts/task_family/train.json"
HELDOUT_PATH = ROOT / "artifacts/task_family/heldout.json"
OFFENSE_TYPES = {0, 5, 7, 8, 10, 13, 18, 24, 26, 28, 29, 32, 34, 39, 40, 42, 43, 44, 47, 48}
CURVE_INTERVAL = 3000
WILSON_Z = 1.959963984540054
MAX_ACTIONS = 4000


def _state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_checkpoint(path: Path) -> tuple[GameplayModelV1, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    provenance = checkpoint["provenance"]
    research = checkpoint.get("research_version") == 1
    if (checkpoint["model_architecture_version"] != model_architecture_version(checkpoint["config"])
            or (not research and checkpoint.get("value_semantics") != VALUE_SEMANTICS)
            or (research and checkpoint.get("value_semantics") not in {
                "research_explicit_return_v1", VALUE_SEMANTICS})
            or (not research and provenance.get("search_label_version") != SEARCH_LABEL_VERSION)
            or provenance.get("protocol_version") != ENV_PROTOCOL_VERSION
            or provenance["observation_version"] != OBSERVATION_VERSION
            or provenance["task_version"] != TASK_VERSION):
        raise ValueError("checkpoint does not match the current model/environment semantics")
    model = GameplayModelV1(checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, {"kind": "checkpoint_file", "path": str(path), "sha256": sha256_file(path),
                   "state_sha256": _state_sha256(model.state_dict())}


def random_action(observation: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    legal = observation["legal_actions"]
    actions = [{"type": "plant", **item} for item in legal["plants"]]
    actions.extend({"type": "shovel", "col": col, "row": row} for col, row in legal["shovels"])
    if legal.get("wait", True):
        actions.extend({"type": "wait", "ticks": ticks} for ticks in (60, 150, 300))
    if not actions:
        raise RuntimeError("observation has no legal action")
    return rng.choice(actions)


def _task_spec(task: dict[str, Any], seed: int) -> TaskSpec:
    return TaskSpec(
        level=task["level"], seed=seed, playthrough=task["playthrough"],
        profile=profile_for_deck(task["deck"]),
        zombie_count_multiplier=task["zombie_count_multiplier"], wave_cap=task["wave_cap"],
        preplanted=tuple(tuple(item) for item in task["preplanted"]),
    )


def run_episode(env: PvZEnv, task: dict[str, Any], seed: int, strategy: str,
                model: GameplayModelV1 | None = None, *, deterministic: bool = True,
                max_actions: int = MAX_ACTIONS,
                allow_truncation: bool = False) -> dict[str, Any]:
    if model is not None:
        require_policy_env(model.config, env)
    observation, _ = env.reset(deck=task["deck"], task=_task_spec(task, seed))
    if observation["sun"] != task["sun_start"]:
        raise RuntimeError(f"{task['task_id']} seed {seed}: observed sun does not match sun_start")
    rng = random.Random(seed)
    hidden = None
    previous_action = None
    previous_wait_result = None
    delta_ticks = 0
    events: dict[str, Any] = {}
    totals: Counter[str] = Counter()
    actions = 0
    peak_offense = sum(plant["type"] in OFFENSE_TYPES for plant in observation["plants"])
    curve = [{"tick": 0, "sun": observation["sun"], "sun_produced": 0, "sun_spent": 0}]
    next_curve_tick = CURVE_INTERVAL

    previous_planted_cell = None
    immediate_shovels = 0
    action_counts: Counter[str] = Counter()
    zero_tick_actions = 0
    wait_records = []
    while not observation["terminal"] and actions < max_actions:
        if strategy == "random":
            action = random_action(observation, rng)
        elif strategy == "idle":
            action = {"type": "wait", "ticks": 300}
        elif strategy == "scripted":
            action = scripted_baseline.choose(observation)
        elif strategy == "checkpoint":
            if model is None:
                raise RuntimeError("checkpoint strategy has no model")
            with torch.inference_mode():
                output = model.step(observation, hidden, previous_action, delta_ticks, events,
                                    previous_wait_result)
                action, _, _ = select_action(model, output, observation, deterministic=deterministic)
                hidden = output["hidden"]
        else:
            raise ValueError(f"unsupported strategy: {strategy}")

        observation, _, done, _, info = env.step(action)
        if not info.get("ok"):
            raise RuntimeError(f"{task['task_id']} seed {seed}: illegal action {action}")
        events = info["events"]
        totals.update(events)
        previous_action = action
        previous_wait_result = info.get("wait_result")
        delta_ticks = info["ticks_advanced"]
        if action.get("until") is not None:
            validate_wait_result(action, previous_wait_result, delta_ticks)
        elif previous_wait_result is not None:
            raise ValueError("fixed actions cannot carry event wait metadata")
        if model is not None and "wait_mode" in model.config and action["type"] == "wait":
            wait_records.append({"decision_index": actions, "action": dict(action),
                                 "actual_ticks": delta_ticks, "terminal_tick": observation["tick"],
                                 "wait_result": previous_wait_result})
        action_counts[action["type"]] += 1
        zero_tick_actions += delta_ticks == 0
        cell = (action.get("row"), action.get("col"))
        immediate_shovels += action["type"] == "shovel" and cell == previous_planted_cell
        previous_planted_cell = cell if action["type"] == "plant" else None
        actions += 1
        peak_offense = max(peak_offense, sum(plant["type"] in OFFENSE_TYPES
                                             for plant in observation["plants"]))
        while observation["tick"] >= next_curve_tick:
            curve.append({"tick": next_curve_tick, "sun": observation["sun"],
                          "sun_produced": totals["sun_produced"], "sun_spent": totals["sun_spent"]})
            next_curve_tick += CURVE_INTERVAL
        if done:
            break

    truncated = not observation["terminal"]
    if truncated and not allow_truncation:
        raise RuntimeError(f"{task['task_id']} seed {seed}: exceeded {max_actions} actions without terminal result")
    record = {
        "seed": seed,
        "won": observation["result"] == 1,
        "result": observation["result"],
        "terminated": not truncated,
        "truncated": truncated,
        "action_counts": dict(action_counts),
        "events": dict(totals),
        "zero_tick_actions": zero_tick_actions,
        "immediate_plant_shovels": immediate_shovels,
        "terminal_wave": observation["wave"],
        "wave_count": observation["wave_count"],
        "terminal_tick": observation["tick"],
        "peak_offense": peak_offense,
        "actions": actions,
        "economy_curve": curve,
    }
    if model is not None and "wait_mode" in model.config:
        record["wait_mode"] = model.config["wait_mode"]
        record["wait_records"] = wait_records
    return record


def wilson_95(passes: int, count: int) -> list[float]:
    rate = passes / count
    z2 = WILSON_Z * WILSON_Z
    denominator = 1 + z2 / count
    center = (rate + z2 / (2 * count)) / denominator
    margin = WILSON_Z * ((rate * (1 - rate) / count + z2 / (4 * count * count)) ** 0.5) / denominator
    return [round(center - margin, 6), round(center + margin, 6)]


def summarize_episodes(records: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(records)
    passes = sum(record["won"] for record in records)
    wave_hist = Counter(record["terminal_wave"] for record in records)
    tick_hist = Counter(f"{record['terminal_tick'] // 5000 * 5000}-{record['terminal_tick'] // 5000 * 5000 + 4999}"
                        for record in records)
    curve_values: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        for point in record["economy_curve"]:
            if point["tick"] % CURVE_INTERVAL == 0:
                curve_values[point["tick"]].append(point)
    economy_curve = [
        {"tick": tick, "sample_count": len(points),
         "mean_sun": round(statistics.mean(point["sun"] for point in points), 3),
         "mean_sun_produced": round(statistics.mean(point["sun_produced"] for point in points), 3),
         "mean_sun_spent": round(statistics.mean(point["sun_spent"] for point in points), 3)}
        for tick, points in sorted(curve_values.items())
    ]
    summary = {
        "sample_count": count,
        "pass_rate": round(passes / count, 6),
        "wilson_95": wilson_95(passes, count),
        "mean_terminal_wave": round(statistics.mean(record["terminal_wave"] for record in records), 3),
        "mean_survival_ticks": round(statistics.mean(record["terminal_tick"] for record in records), 3),
        "peak_offense": round(statistics.mean(record["peak_offense"] for record in records), 3),
        "terminal_wave_histogram": dict(sorted(wave_hist.items())),
        "terminal_tick_histogram_5000_tick_bins": dict(sorted(tick_hist.items())),
        "economy_curve": economy_curve,
    }
    if any("wait_records" in record for record in records):
        if any("wait_records" not in record for record in records):
            raise ValueError("mixed legacy/explicit-wait evaluation rows")
        summary["wait_summary"] = summarize_wait_records([row for record in records for row in record["wait_records"]])
    return summary


def _evaluate_split(env: PvZEnv, tasks: list[dict[str, Any]], strategy: str,
                    model: GameplayModelV1 | None) -> dict[str, Any]:
    return {
        task["task_id"]: [run_episode(env, task, seed, strategy, model) for seed in task["seeds"]]
        for task in tasks
    }


def _profile(split: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return {task_id: summarize_episodes(records) for task_id, records in split.items()}


def _fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _control_task(seed_start: int, count: int) -> dict[str, Any]:
    return {
        "task_id": "t0_level7_wave_cap_3_control", "terrain": "day", "level": 7,
        "deck": [0, 1, 2, 3, 4, 5], "zombie_count_multiplier": 1.0, "wave_cap": 3,
        "sun_start": 50, "preplanted": [], "playthrough": 2,
        "seeds": list(range(seed_start, seed_start + count)),
    }


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resource-dir", required=True)
    parser.add_argument("--train", type=Path, default=TRAIN_PATH)
    parser.add_argument("--heldout", type=Path, default=HELDOUT_PATH)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--checkpoint", type=Path)
    choice.add_argument("--checkpoint-seed", type=int,
                        help="profile a deterministic, randomly initialized checkpoint (no training)")
    parser.add_argument("--control-seed-start", type=int, default=90000)
    parser.add_argument("--control-seed-count", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    train = json.loads(args.train.read_text(encoding="utf-8"))
    heldout = json.loads(args.heldout.read_text(encoding="utf-8"))
    check_manifests(train, heldout)
    configure_torch_threads(1)
    if args.checkpoint:
        model, checkpoint_info = load_checkpoint(args.checkpoint)
    else:
        torch.manual_seed(args.checkpoint_seed)
        model = GameplayModelV1().eval()
        checkpoint_info = {
            "kind": "random_initialized_checkpoint",
            "seed": args.checkpoint_seed,
            "trained": False,
            "state_sha256": _state_sha256(model.state_dict()),
        }

    with policy_env(model.config, args.resource_dir) as env:
        first_run = {
            "train": _evaluate_split(env, train["tasks"], "checkpoint", model),
            "heldout": _evaluate_split(env, heldout["tasks"], "checkpoint", model),
        }
        first_fingerprint = _fingerprint(first_run)
        second_run = {
            "train": _evaluate_split(env, train["tasks"], "checkpoint", model),
            "heldout": _evaluate_split(env, heldout["tasks"], "checkpoint", model),
        }
        second_fingerprint = _fingerprint(second_run)
        deterministic = first_run == second_run

        control = _control_task(args.control_seed_start, args.control_seed_count)
        random_control = [run_episode(env, control, seed, "random") for seed in control["seeds"]]
        scripted_control = [run_episode(env, control, seed, "scripted") for seed in control["seeds"]]

    profiles = {split: _profile(records) for split, records in first_run.items()}
    train_mean_pass = statistics.mean(item["pass_rate"] for item in profiles["train"].values())
    heldout_mean_pass = statistics.mean(item["pass_rate"] for item in profiles["heldout"].values())
    random_summary = summarize_episodes(random_control)
    scripted_summary = summarize_episodes(scripted_control)
    gate_result = (
        random_summary["pass_rate"] <= 0.10
        and scripted_summary["pass_rate"] >= 0.80
        and deterministic
        and len(random_control) >= 64
        and len(scripted_control) >= 64
    )
    result = {
        "schema_version": 1,
        "task_id": "T4",
        "checkpoint": checkpoint_info,
        "manifests": {
            "train_sha256": sha256_file(args.train),
            "heldout_sha256": sha256_file(args.heldout),
        },
        "metrics": {
            "per_task": profiles,
            "aggregate": {
                "train_mean_pass": round(train_mean_pass, 6),
                "heldout_mean_pass": round(heldout_mean_pass, 6),
                "generalization_gap": round(train_mean_pass - heldout_mean_pass, 6),
            },
            "controls": {
                "task": {key: control[key] for key in (
                    "task_id", "level", "deck", "zombie_count_multiplier", "wave_cap", "sun_start", "preplanted",
                )},
                "random": random_summary,
                "t0_rule_script": scripted_summary,
            },
            "determinism": {
                "identical_results": deterministic,
                "first_run_sha256": first_fingerprint,
                "second_run_sha256": second_fingerprint,
                "episodes_per_run": sum(len(records) for split in first_run.values() for records in split.values()),
            },
        },
        "thresholds": {
            "random_wave_cap_3_pass_rate_max": 0.10,
            "t0_script_wave_cap_3_pass_rate_min": 0.80,
            "checkpoint_repeat_results_identical": True,
            "minimum_samples_for_wilson_conclusion": 64,
            "single_seed_failure": "propagate as a nonzero evaluator exit; never skip a seed",
        },
        "control_seed_results": {"random": random_control, "t0_rule_script": scripted_control},
        "seed_results": first_run,
        "gate_result": "pass" if gate_result else "fail",
        "offense_seed_types": sorted(OFFENSE_TYPES),
        "economy_curve_interval_ticks": CURVE_INTERVAL,
    }
    atomic_json(args.output, result)
    print(json.dumps({
        "gate_result": result["gate_result"],
        "checkpoint": checkpoint_info,
        "random_cap3_pass_rate": random_summary["pass_rate"],
        "t0_script_cap3_pass_rate": scripted_summary["pass_rate"],
        "deterministic": deterministic,
        "episodes_per_run": result["metrics"]["determinism"]["episodes_per_run"],
        "output": str(args.output),
    }, sort_keys=True))
    if not gate_result:
        raise SystemExit(1)


if __name__ == "__main__":
    _main()
