#!/usr/bin/env python3
"""task84: episode-level forensics for the heldout-zone regression (r13/r14 vs r11).

The gate v2 evaluations re-evaluated the incumbent (r11) on the same fresh seeds as
each candidate, so candidate/incumbent games.jsonl form true paired outcomes. This
script breaks down day_l7_eval_d7p_holdout1 (the regressing task) and its controls:
which seeds flipped, how far losing candidates got (final wave), how many mowers
they burned, and how the planted-plant mix differs from the incumbent.
Statistics only - no gameplay rules.
Run: python3 scripts/task84_heldout_forensics.py
"""
from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path

PIPELINE = Path("/home/newbiexvwu/pvz-logs/curriculum_state.pipeline")
EVALUATIONS = {
    13: PIPELINE / "round_000013/evaluations/a41fdfa96075ae608a40",
    14: PIPELINE / "round_000014/evaluations/e3c036387cbf7d74606d",
}
TASKS = ("day_l7_eval_d7p_holdout1", "day_l7_eval_d7", "day_l9_eval_d7p_holdout1")


def rows(evaluation, side, task):
    path = evaluation / side / task / "games.jsonl"
    return {row["seed"]: row for row in map(json.loads, path.read_text().splitlines())}


def plant_mix(games):
    mix = Counter()
    for row in games:
        for plant, count in (row.get("planted_types") or {}).items():
            mix[int(plant)] += count
    return mix


def quartiles(values):
    if len(values) < 4:
        return ""
    q = statistics.quantiles(values, n=4)
    return " p25 %.0f p75 %.0f" % (q[0], q[2])


def describe(tag, games):
    wins = [row for row in games if row["won"]]
    losses = [row for row in games if not row["won"]]
    line = "  %s: %d/%d wins" % (tag, len(wins), len(games))
    if losses:
        waves = [row["final_wave"] for row in losses]
        mowers = [row["mowers_used"] for row in losses]
        line += (" | losses: wave median %.0f (min %d%s)"
                 % (statistics.median(waves), min(waves), quartiles(waves)))
        line += ", mowers median %.0f" % statistics.median(mowers)
    print(line)


def mix_delta(tag, candidate_games, incumbent_games, seeds_n):
    mix_c, mix_i = plant_mix(candidate_games), plant_mix(incumbent_games)
    delta = {p: (mix_c[p] - mix_i[p]) / seeds_n for p in sorted(set(mix_c) | set(mix_i))}
    text = " ".join("%d:%+.2f" % (p, d) for p, d in delta.items() if abs(d) >= 0.02)
    if text:
        print("    %s: %s" % (tag, text))


def main():
    for round_index, evaluation in sorted(EVALUATIONS.items()):
        manifest = json.loads((evaluation / "manifest.json").read_text())
        print("=== round %d vs r11 ===" % round_index)
        for task in TASKS:
            if task not in manifest["groups"]:
                continue
            incumbent = rows(evaluation, "incumbent", task)
            candidate = rows(evaluation, "candidate", task)
            seeds = sorted(incumbent)
            flips = [s for s in seeds if incumbent[s]["won"] and not candidate[s]["won"]]
            gains = [s for s in seeds if not incumbent[s]["won"] and candidate[s]["won"]]
            print("%s: r11 %d/%d, r%d %d/%d, flips(r11 win->cand loss) %d, gains %d"
                  % (task, sum(r["won"] for r in incumbent.values()), len(seeds),
                     round_index, sum(r["won"] for r in candidate.values()), len(seeds),
                     len(flips), len(gains)))
            describe("r11 all", [incumbent[s] for s in seeds])
            describe("r11 on flips", [incumbent[s] for s in flips])
            describe("cand on flips", [candidate[s] for s in flips])
            describe("cand all", [candidate[s] for s in seeds])
            if flips:
                mix_delta("plant mix per flip game (cand-inc)",
                          [candidate[s] for s in flips], [incumbent[s] for s in flips], len(flips))
            mix_delta("plant mix per game all seeds (cand-inc)",
                      [candidate[s] for s in seeds], [incumbent[s] for s in seeds], len(seeds))
            print()


if __name__ == "__main__":
    main()
