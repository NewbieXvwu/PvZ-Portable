#!/usr/bin/env python3
"""task158: aggregate the terrain-arm runs into the comparison the task asks for.

Reads ``<runs>/<arm>_s<seed>.json`` (and ``experts_s<seed>.json``) and reports, per terrain:

  seen    rows whose decks trained; reported on the seed-validation split
  unseen  whole decks withheld from training (task153's unseen-deck split)

For every arm and metric: mean and range over seeds, then the three task questions
(single nets beat joint64 somewhere -> interference; joint128 catches up -> capacity;
experts >= joint128 at joint64 inference cost -> recommend experts).

Parameters and inference cost are per-row network evaluations: joint64=1x, joint128~2.6x,
experts=1x (one of five nets per row, but five checkpoints), hard_moe~1x, soft_moe~2.4x,
film~1x.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

TERRAINS = ("day", "night", "pool", "fog", "roof")
ARMS = ("joint64", "joint128", "experts", "hard_moe", "soft_moe", "film")
METRICS = ("action_acc", "auc", "false_plant", "missed_plant", "packet_acc", "cell_acc")
COST = {"joint64": "1x", "joint128": "3.14x", "experts": "1x (one of five files)",
        "hard_moe": "1x (2.35x weights)", "soft_moe": "1.52x", "film": "1x"}


def load_runs(runs: Path, arms, seeds):
    reports = {}
    for arm in arms:
        per_seed = []
        for seed in seeds:
            path = runs / f"{arm}_s{seed}.json"
            if path.is_file():
                per_seed.append(json.loads(path.read_text()))
        if per_seed:
            reports[arm] = per_seed
    return reports


def all_block(report, split):
    """``seen_all``/``unseen_all`` when present; the experts arm reports per terrain only, so pool
    the rates that pool exactly (by rows / wait rows / plant rows). AUC is not poolable."""
    block = report.get(f"{split}_all")
    if block:
        return block
    parts = [value for value in report.get(split, {}).values() if value]
    rows = sum(part["rows"] for part in parts)
    waits = sum(part["rows"] * (1 - part["plant_rate"]) for part in parts)
    plants = sum(part["rows"] * part["plant_rate"] for part in parts)
    if not rows:
        return None
    return {"rows": rows,
            "plant_rate": plants / rows,
            "action_acc": sum(part["action_acc"] * part["rows"] for part in parts) / rows,
            "auc": None,
            "false_plant": sum(part["false_plant"] * part["rows"] * (1 - part["plant_rate"])
                               for part in parts) / waits if waits else None,
            "missed_plant": sum(part["missed_plant"] * part["rows"] * part["plant_rate"]
                                for part in parts) / plants if plants else None,
            "packet_acc": sum(part["packet_acc"] * part["rows"] * part["plant_rate"]
                              for part in parts) / plants if plants else None,
            "cell_acc": sum(part["cell_acc"] * part["rows"] * part["plant_rate"]
                            for part in parts) / plants if plants else None}


def values(reports, arm, split, terrain, metric):
    out = []
    for report in reports.get(arm, []):
        block = all_block(report, split) if terrain == "all" else report.get(split, {}).get(terrain)
        if block and block.get(metric) is not None and metric in block:
            out.append(float(block[metric]))
    return out


def summary(values_list):
    if not values_list:
        return None
    return dict(mean=sum(values_list) / len(values_list),
                low=min(values_list), high=max(values_list), seeds=len(values_list))


def parameters(reports, arm):
    values_list = [report.get("parameters") for report in reports.get(arm, [])]
    values_list = [value for value in values_list if value]
    return values_list[0] if values_list else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--seeds", default="1,2")
    args = parser.parse_args()
    arms = [arm for arm in args.arms.split(",") if arm]
    seeds = [int(seed) for seed in args.seeds.split(",")]
    reports = load_runs(args.runs, arms, seeds)
    if not reports:
        raise SystemExit(f"no reports under {args.runs}")

    table = {"arms": {}, "per_terrain": {}}
    for arm in reports:
        table["arms"][arm] = dict(parameters=parameters(reports, arm), cost=COST.get(arm),
                                  seeds=[report.get("seed") for report in reports[arm]])
    for split in ("seen", "unseen"):
        table["per_terrain"][split] = {}
        for terrain in TERRAINS + ("all",):
            block = {}
            for arm in reports:
                block[arm] = {metric: summary(values(reports, arm, split, terrain, metric))
                              for metric in METRICS}
            table["per_terrain"][split][terrain] = block

    def render(lines, split, metric, title):
        lines += ["", f"## {split} decks: {title} (mean [range] over seeds)", "",
                  "| terrain | " + " | ".join(reports) + " |", "|---" * (len(reports) + 1) + "|"]
        for terrain in TERRAINS + ("all",):
            cells = []
            for arm in reports:
                value = table["per_terrain"][split][terrain][arm][metric]
                cells.append("-" if value is None else
                             f"{value['mean']:.4f} [{value['low']:.4f},{value['high']:.4f}]")
            lines.append(f"| {terrain} | " + " | ".join(cells) + " |")

    lines = ["# task158 terrain arms", "",
             "| arm | parameters | inference cost | seeds |", "|---|---|---|---|"]
    for arm, info in table["arms"].items():
        lines.append(f"| {arm} | {info['parameters']} | {info['cost']} | {info['seeds']} |")
    for split in ("seen", "unseen"):
        for metric, title in (("action_acc", "action accuracy"), ("auc", "plant/wait AUC"),
                              ("false_plant", "wait rows called plant"), ("missed_plant", "plant rows called wait"),
                              ("packet_acc", "packet accuracy"), ("cell_acc", "cell accuracy")):
            render(lines, split, metric, title)
    lines += ["", "## routing (soft_moe)", ""]
    for report in reports.get("soft_moe", []):
        lines.append(json.dumps(dict(seed=report.get("seed"), seen=report.get("routing"),
                                     unseen=report.get("routing_unseen"))))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    (args.out.with_suffix(".json")).write_text(json.dumps(table, indent=1) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
