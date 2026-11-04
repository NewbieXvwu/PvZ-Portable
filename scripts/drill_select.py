"""Filter a mined drill bank by the production teacher's short-horizon success (task235).

All-zero items teach nothing (no successful label exists) and all-one items add little
(the student already handles them); keep the middle band.  This script only reads the
``drill_run.py`` summary, so calibration stays a replay measurement, not a new rule.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--results", type=Path,
                        help="drill_run results.jsonl; overrides --summary and aggregates extra seeds")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--low", type=float, default=0.25)
    parser.add_argument("--high", type=float, default=0.90)
    parser.add_argument("--min-games", type=int, default=3)
    parser.add_argument("--max-per-level-type", type=int, default=0,
                        help="optional balance cap; 0 keeps every passing drill")
    parser.add_argument("--require-levels", default="",
                        help="comma/range list; a required level with no in-band drill gets all-1 coverage items")
    parser.add_argument("--coverage-per-level", type=int, default=0,
                        help="how many all-1 items to add for a required level with no in-band drill")
    args = parser.parse_args()
    bank = json.loads(args.bank.read_text())
    if args.results:
        grouped = {}
        for line in args.results.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") != "ok":
                continue
            entry = grouped.setdefault(row["drill_id"], {"games": 0, "wins": 0})
            entry["games"] += 1
            entry["wins"] += int(bool(row.get("success")))
        rates = {key: {"teacher": value} for key, value in grouped.items()}
    elif args.summary:
        rates = json.loads(args.summary.read_text()).get("drills") or {}
    else:
        parser.error("one of --summary or --results is required")
    kept, stats = [], Counter()
    per_level_type = Counter()
    seen = set()
    for drill in bank.get("drills") or []:
        if drill["id"] in seen:
            stats["duplicate_id"] += 1
            continue
        seen.add(drill["id"])
        item = (rates.get(drill["id"]) or {}).get("teacher") or {}
        games, wins = int(item.get("games") or 0), int(item.get("wins") or 0)
        if games < args.min_games:
            stats["too_few_games"] += 1
            continue
        rate = wins / games
        key = (drill["level"], drill.get("trigger", {}).get("name"))
        if not args.low <= rate <= args.high:
            stats["all_or_nothing" if rate in (0.0, 1.0) else "outside_band"] += 1
            continue
        if args.max_per_level_type and per_level_type[key] >= args.max_per_level_type:
            stats["balanced_out"] += 1
            continue
        per_level_type[key] += 1
        drill = dict(drill, teacher=dict(games=games, wins=wins, rate=round(rate, 4)))
        kept.append(drill)
    # A few levels (L36 digger is the known case) can have teacher 0/3 or 3/3 only.
    # For those, keep a small number of all-1 items as coverage anchors: they still
    # carry expert labels at a rescue window, but cannot discriminate difficulty, so
    # they are marked and counted separately. All-0 items are never added.
    required = set()
    for part in (piece.strip() for piece in args.require_levels.split(",")):
        if not part:
            continue
        low, _, high = part.partition("-")
        required.update(range(int(low), int(high or low) + 1))
    coverage = []
    if args.coverage_per_level:
        present = Counter(drill["level"] for drill in kept)
        kept_ids = {drill["id"] for drill in kept}
        for level in sorted(required):
            need = args.coverage_per_level - present[level]
            if need <= 0:
                continue
            candidates = []
            for drill in bank.get("drills") or []:
                if drill["level"] != level or drill["id"] in kept_ids:
                    continue
                item = (rates.get(drill["id"]) or {}).get("teacher") or {}
                games, wins = int(item.get("games") or 0), int(item.get("wins") or 0)
                if games >= args.min_games and wins == games:
                    candidates.append(dict(drill, teacher=dict(games=games, wins=wins, rate=1.0),
                                           coverage_only=True))
            for drill in candidates[:need]:
                coverage.append(drill)
                kept.append(drill)
                kept_ids.add(drill["id"])
                stats[f"coverage_level_{level}"] += 1
    for drill in kept:
        stats[f"level_{drill['level']}"] += 1
    by_level = Counter(drill["level"] for drill in kept)
    by_type = Counter(drill.get("trigger", {}).get("name") for drill in kept)
    by_level_type = Counter(f"{drill['level']}|{drill.get('trigger', {}).get('name')}"
                            for drill in kept)
    report = dict(schema="pvz-drill-calibration-v1", task="task235",
                  band=[args.low, args.high], min_games=args.min_games,
                  mined=len(bank.get("drills") or []), kept=len(kept),
                  dropped=dict(stats), by_level=dict(sorted(by_level.items())),
                  by_type=dict(sorted(by_type.items())),
                  by_level_type=dict(sorted(by_level_type.items())),
                  coverage_only=[{"id": drill["id"], "level": drill["level"],
                                  "trigger": drill.get("trigger", {}).get("name")}
                                 for drill in coverage],
                  teacher_rate=dict(
                      games=sum(d["teacher"]["games"] for d in kept),
                      wins=sum(d["teacher"]["wins"] for d in kept),
                      mean=round(sum(d["teacher"]["rate"] for d in kept) / len(kept), 4)
                      if kept else None))
    document = dict(bank, drills=kept, calibration=report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, ensure_ascii=False, indent=1) + "\n")
    if args.report:
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({key: report[key] for key in
                      ("mined", "kept", "dropped", "by_level", "by_type", "teacher_rate")},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
