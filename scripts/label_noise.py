"""Summarize anchored A/B/H labels from JSONL, without running the simulator.

Collect with sim_skip_agreement.py search --label-noise --base-halving
--base-cpp-rollouts --cpp-rollouts --base-mask 61 --mask 61 --samples 3
--horizon 2500 --dagger-beta 0.5 --jsonl-dir OUT (plus model/task paths).
Then: python3 scripts/label_noise.py OUT/seed_*.jsonl --out OUT/summary.json
"""
from __future__ import annotations

import argparse
import json
from math import sqrt
from pathlib import Path

import numpy as np


def action_key(action: dict) -> str:
    if action["type"] == "wait":
        return "wait"
    return json.dumps(action, sort_keys=True)


def read_rows(paths: list[Path]) -> list[dict]:
    rows = [json.loads(line) for path in paths for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("no search points")
    if len({(row["seed"], row["decision"]) for row in rows}) != len(rows):
        raise ValueError("duplicate search points")
    return rows


def fraction(numerator: int, denominator: int) -> dict:
    if not denominator:
        return {"count": numerator, "n": 0, "rate": None, "ci95": None}
    rate = numerator / denominator
    zscore = 1.959963984540054
    scale = 1 + zscore ** 2 / denominator
    center = (rate + zscore ** 2 / (2 * denominator)) / scale
    radius = zscore * sqrt(rate * (1 - rate) / denominator + zscore ** 2 / (4 * denominator ** 2)) / scale
    return {"count": numerator, "n": denominator, "rate": rate,
            "ci95": [max(0, center - radius), min(1, center + radius)]}


def distribution(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": None, "p50": None, "p90": None, "mean_ci95": None}
    array = np.asarray(values)
    draws = np.random.default_rng(0).integers(len(array), size=(2000, len(array)))
    interval = np.quantile(array[draws].mean(axis=1), [0.025, 0.975]).tolist()
    return {"n": len(array), "mean": float(array.mean()), "p50": float(np.quantile(array, 0.5)),
            "p90": float(np.quantile(array, 0.9)), "mean_ci95": interval}


def relabel(row: dict, arm: str, margin: float) -> str:
    record = row["arms"][arm]
    values = {action_key(json.loads(action)): value for action, value in record["values"]}
    student = action_key(row["student"])
    best = action_key(record["best"])
    return best if values[best] - values[student] > margin else student


def summarize_group(rows: list[dict]) -> dict:
    counts = dict.fromkeys(("a", "both", "same_both", "b_not_a", "h_same", "h_diff", "h_no", "h_not_a"), 0)
    regrets, student_regrets, agreed, disagreed = [], [], [], []
    margins = {str(margin): [0, 0] for margin in (0.02, 0.05, 0.1)}
    for row in rows:
        student = action_key(row["student"])
        labels = {name: action_key(arm["label"]) for name, arm in row["arms"].items()}
        a_override = labels["A"] != student
        b_override = labels["B"] != student
        h_override = labels["H"] != student
        counts["a"] += a_override
        counts["both"] += a_override and b_override
        counts["same_both"] += a_override and b_override and labels["A"] == labels["B"]
        counts["b_not_a"] += not a_override and b_override
        counts["h_same"] += a_override and h_override and labels["A"] == labels["H"]
        counts["h_diff"] += a_override and h_override and labels["A"] != labels["H"]
        counts["h_no"] += a_override and not h_override
        counts["h_not_a"] += not a_override and h_override
        values = {action_key(json.loads(action)): value for action, value in row["arms"]["H"]["values"]}
        best_value = max(values.values())
        regrets.append(best_value - values[labels["A"]])
        student_regrets.append(best_value - values[student])
        if a_override:
            target = agreed if h_override and labels["A"] == labels["H"] else disagreed
            target.append(values[labels["A"]] - values[student])
        for margin, tally in margins.items():
            label = relabel(row, "A", float(margin))
            tally[0] += label != student
            tally[1] += label != student and h_override and label == labels["H"]
    total = len(rows)
    overrides = counts["a"]
    return {
        "searches": total,
        "a_override": fraction(overrides, total),
        "b_override_given_a": fraction(counts["both"], overrides),
        "same_action_given_both": fraction(counts["same_both"], counts["both"]),
        "b_override_given_not_a": fraction(counts["b_not_a"], total - overrides),
        "h_same_given_a": fraction(counts["h_same"], overrides),
        "h_different_given_a": fraction(counts["h_diff"], overrides),
        "h_no_override_given_a": fraction(counts["h_no"], overrides),
        "h_override_given_not_a": fraction(counts["h_not_a"], total - overrides),
        "a_value_loss": distribution(regrets),
        "student_value_loss": distribution(student_regrets),
        "gain_h_agrees": distribution(agreed),
        "gain_h_disagrees": distribution(disagreed),
        "margin_sweep": {margin: {"override": fraction(tally[0], total),
                                   "h_agrees": fraction(tally[1], tally[0])} for margin, tally in margins.items()},
    }


def summarize(rows: list[dict]) -> dict:
    def wave_group(row):
        return "1-9" if row["wave"] < 10 else "10-19" if row["wave"] < 20 else "20-30"

    groups = {"all": rows}
    for kind in ("wait", "plant"):
        groups[kind] = [row for row in rows if row["student"]["type"] == kind]
    for wave in ("1-9", "10-19", "20-30"):
        groups[wave] = [row for row in rows if wave_group(row) == wave]
        for kind in ("wait", "plant"):
            groups[f"{kind}/{wave}"] = [row for row in groups[wave] if row["student"]["type"] == kind]
    return {"games": len({row["seed"] for row in rows}),
            "seeds": sorted({row["seed"] for row in rows}),
            "intervals": "pointwise Wilson 95%; means: point bootstrap 2000, seed 0; within-game dependence not adjusted",
            "split": "student action; wave 0 (opening) included in 1-9",
            "h_reference": "independent finite-horizon value estimates, not terminal ground truth; H margin fixed at recorded value",
            "groups": {name: summarize_group(group) for name, group in groups.items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", nargs="+", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = summarize(read_rows(args.jsonl))
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.out:
        args.out.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
