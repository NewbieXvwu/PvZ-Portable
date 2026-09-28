"""Compare search-teacher labels across horizons on similar visible states."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from itertools import combinations
import gzip
import json
from pathlib import Path
from typing import Any


def visible_state_key(observation: dict[str, Any]) -> str:
    plants = sorted(
        (int(item["type"]), int(item["row"]), int(item["col"]) // 3,
         min(4, int(4 * item.get("health", 0) / max(1, item.get("max_health", 1)))))
        for item in observation["plants"] if not item.get("squished") and item.get("health", 0) > 0
    )
    zombies = sorted(
        (int(item["type"]), int(item["row"]), int(float(item["x"]) // 60),
         min(4, int((item.get("body_health", 0) + item.get("helm_health", 0)
                     + item.get("shield_health", 0)) // 400)))
        for item in observation["zombies"]
    )
    packets = sorted(
        (int(item["type"]), int(item.get("imitater_type", -1)), bool(item.get("active")),
         min(4, int(4 * item.get("cooldown", 0) / max(1, item.get("refresh_time", 1)))))
        for item in observation.get("packets", [])
    )
    return json.dumps({
        "level": observation["level"],
        "wave": observation["wave"],
        "tick": int(observation["tick"]) // 300,
        "sun": int(observation["sun"]) // 100,
        "zombie_count_multiplier": observation.get("zombie_count_multiplier", 1.0),
        "scene": [bool(observation.get(name)) for name in ("night", "pool", "fog", "roof")],
        "plants": plants,
        "zombies": zombies,
        "packets": packets,
        "defenses": sorted((int(item["row"]), int(item["state"])) for item in observation.get("defenses", [])),
    }, sort_keys=True, separators=(",", ":"))


def _distribution(samples: list[dict[str, Any]], policy: bool) -> dict[str, float]:
    counts: Counter[str] = Counter()
    total = 0.0
    for sample in samples:
        if policy:
            labels = sample.get("policy", []) or [{"action": sample["action"], "probability": 1.0}]
            mass = sum(max(0.0, float(item["probability"])) for item in labels)
            if mass <= 0.0:
                continue
            for item in labels:
                action = json.dumps(item["action"], sort_keys=True, separators=(",", ":"))
                counts[action] += max(0.0, float(item["probability"])) / mass
        else:
            action = json.dumps(sample["action"], sort_keys=True, separators=(",", ":"))
            counts[action] += 1.0
        total += 1.0
    return {action: count / total for action, count in counts.items()} if total else {}


def compare_label_samples(samples_by_horizon: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
    clusters: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for horizon, samples in samples_by_horizon.items():
        for sample in samples:
            clusters[sample["visible_state_key"]][horizon].append(sample)

    action_rates: list[float] = []
    policy_distances: list[float] = []
    matched_samples = 0
    matched_clusters = 0
    for by_horizon in clusters.values():
        horizons = sorted(by_horizon)
        if len(horizons) < 2:
            continue
        matched_clusters += 1
        matched_samples += sum(len(by_horizon[horizon]) for horizon in horizons)
        actions = {horizon: _distribution(by_horizon[horizon], False) for horizon in horizons}
        policies = {horizon: _distribution(by_horizon[horizon], True) for horizon in horizons}
        for left, right in combinations(horizons, 2):
            overlap = sum(probability * actions[right].get(action, 0.0)
                          for action, probability in actions[left].items())
            action_rates.append(1.0 - overlap)
            keys = policies[left].keys() | policies[right].keys()
            policy_distances.append(0.5 * sum(abs(policies[left].get(action, 0.0)
                                                 - policies[right].get(action, 0.0)) for action in keys))
    return {
        "matched_clusters": matched_clusters,
        "matched_samples": matched_samples,
        "horizon_pairs": len(action_rates),
        "action_disagreement_rate": sum(action_rates) / len(action_rates) if action_rates else None,
        "mean_search_policy_total_variation": sum(policy_distances) / len(policy_distances) if policy_distances else None,
    }


def _parse_sample_arg(value: str) -> tuple[int, Path]:
    horizon_text, separator, path_text = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("samples must use HORIZON_TICKS=PATH")
    try:
        horizon = int(horizon_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("horizon must be an integer") from error
    if horizon < 1 or not path_text:
        raise argparse.ArgumentTypeError("horizon and sample path must be valid")
    return horizon, Path(path_text).expanduser()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", action="append", type=_parse_sample_arg, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len({horizon for horizon, _ in args.samples}) < 2:
        parser.error("provide sample files for at least two different horizons")
    samples_by_horizon: dict[int, list[dict[str, Any]]] = {}
    run_signatures: list[dict[str, Any]] = []
    for horizon, path in args.samples:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            data = json.load(stream)
        if data.get("evaluation_role") != "development":
            parser.error(f"{path} is not a development-set diagnostic")
        if data.get("horizon_ticks") != horizon:
            parser.error(f"{path} does not contain horizon {horizon}")
        run_signatures.append({
            key: data.get(key) for key in (
                "evaluation_role", "seed_file_sha256", "task_signature", "simulation_budget",
                "beam_width", "candidate_limit", "max_decisions", "search_value_sha256",
            )
        })
        samples_by_horizon[horizon] = data["samples"]
    if any(signature != run_signatures[0] for signature in run_signatures[1:]):
        parser.error("diagnostic runs must share the same development seeds, task, value checkpoint, and search budgets")
    result = {
        "evaluation_role": "development",
        "horizons_ticks": sorted(samples_by_horizon),
        **run_signatures[0],
        **compare_label_samples(samples_by_horizon),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
