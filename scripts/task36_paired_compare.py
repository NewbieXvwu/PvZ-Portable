"""task36 step 1: same-seed paired comparison of two student panels.

Reads the per-game games.jsonl of two unified-panel output directories, pairs games by
evaluation seed (both panels use the same frozen manifest, so the seed lists are equal),
and reports discordant pairs with an exact McNemar (binomial) p-value.

  python3 scripts/task36_paired_compare.py \
      --a artifacts/task27/baseline_v2_dw4i_s1 --b artifacts/task27/panel_dw5i_v2 \
      --out artifacts/task36/paired_dw4i_vs_dw5i.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

EARLY = {"day_l1_eval_d7", "day_l3_eval_d7", "day_l4_eval_d7", "day_l6_eval_d7"}


def read_games(directory: Path, task: str) -> dict[int, dict]:
    path = directory / task / "games.jsonl"
    if not path.exists():
        return {}
    games = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        games[int(record["seed"])] = record
    return games


def exact_mcnemar(b: int, c: int) -> float:
    """Two-sided exact binomial test of b vs c discordant pairs under p=1/2."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def compare(a_dir: Path, b_dir: Path, tasks: list[str]) -> list[dict]:
    rows = []
    for task in tasks:
        a_games, b_games = read_games(a_dir, task), read_games(b_dir, task)
        if not a_games or not b_games:
            continue
        seeds = sorted(set(a_games) & set(b_games))
        both = sum(1 for s in seeds if a_games[s]["won"] and b_games[s]["won"])
        only_a = sum(1 for s in seeds if a_games[s]["won"] and not b_games[s]["won"])
        only_b = sum(1 for s in seeds if b_games[s]["won"] and not a_games[s]["won"])
        neither = len(seeds) - both - only_a - only_b
        rows.append(dict(
            task_id=task, paired=len(seeds), a_won=both + only_a, b_won=both + only_b,
            both=both, only_a=only_a, only_b=only_b, neither=neither,
            delta_pp=100.0 * (only_b - only_a) / len(seeds) if seeds else 0.0,
            mcnemar_p=exact_mcnemar(only_a, only_b),
            unpaired_a=len(a_games), unpaired_b=len(b_games),
            same_seed_set=set(a_games) == set(b_games),
        ))
    return rows


def summarise(rows: list[dict], name: str) -> dict:
    paired = sum(r["paired"] for r in rows)
    only_a = sum(r["only_a"] for r in rows)
    only_b = sum(r["only_b"] for r in rows)
    return dict(
        name=name, tasks=len(rows), paired_games=paired,
        a_won=sum(r["a_won"] for r in rows), b_won=sum(r["b_won"] for r in rows),
        only_a=only_a, only_b=only_b, delta_pp=100.0 * (only_b - only_a) / paired if paired else 0.0,
        mcnemar_p=exact_mcnemar(only_a, only_b),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", type=Path, required=True, help="reference panel directory")
    parser.add_argument("--b", type=Path, required=True, help="candidate panel directory")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tasks", help="optional comma-separated task ids (default: every shared task)")
    args = parser.parse_args()
    if args.tasks:
        tasks = args.tasks.split(",")
    else:
        tasks = sorted(p.name for p in args.a.iterdir()
                       if p.is_dir() and (p / "games.jsonl").exists()
                       and (args.b / p.name / "games.jsonl").exists())
    rows = compare(args.a, args.b, tasks)
    night = [r for r in rows if r["task_id"].startswith("night_")]
    early = [r for r in rows if r["task_id"] in EARLY]
    report = dict(
        schema_version=1,
        a=str(args.a), b=str(args.b),
        per_task=rows,
        groups=[summarise(night, "night")] if night else [],
        early_guard=summarise(early, "early_L1_L3_L4_L6") if early else None,
        all_shared=summarise(rows, "all_shared_tasks"),
        note=("Same evaluation seeds in both panels (frozen manifest); McNemar exact two-sided "
              "binomial over discordant pairs. Descriptive for one training seed."),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    for group in report["groups"] + ([report["early_guard"]] if report["early_guard"] else []):
        print(f"{group['name']:>20}: A {group['a_won']}/{group['paired_games']} "
              f"B {group['b_won']}/{group['paired_games']} delta {group['delta_pp']:+.2f}pp "
              f"(onlyA {group['only_a']}, onlyB {group['only_b']}, p={group['mcnemar_p']:.4g})")
    for row in rows:
        print(f"  {row['task_id']:<32} {row['a_won']:>3}/{row['paired']:<3} -> {row['b_won']:>3} "
              f"{row['delta_pp']:+7.2f}pp p={row['mcnemar_p']:.4g}")


if __name__ == "__main__":
    main()
