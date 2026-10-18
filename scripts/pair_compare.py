"""Paired comparison of two search_play runs on the same seeds.

Both inputs are the output directory of scripts/search_play.py (games.jsonl). The pairing is by
seed, so the comparison is a paired one: same task, same seeds, one flag different.

    python scripts/pair_compare.py artifacts/teacher/A artifacts/teacher/B --label-a first_aid_off --label-b first_aid_on

Reports wins, solo wins (games only one side won), the exact McNemar p-value over the discordant
pairs, mowers lost, per-game searches / overrides / seconds, and the first-aid replant counts when
the runs recorded them (search_play "replant_events", same_type = nut on a damaged nut of the same
type). It never writes anything: the numbers go into the report.
"""
from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path


def load(path: Path) -> dict[int, dict]:
    rows = [json.loads(line) for line in (path / "games.jsonl").read_text().splitlines() if line.strip()]
    by_seed: dict[int, dict] = {}
    for row in rows:
        by_seed.setdefault(row["seed"], row)      # first arm wins; every run here has one arm
    return by_seed


def mcnemar_exact(only_a: int, only_b: int) -> float:
    discordant = only_a + only_b
    if not discordant:
        return 1.0
    tail = sum(comb(discordant, i) for i in range(min(only_a, only_b) + 1))
    return min(1.0, 2 * tail / 2 ** discordant)


def mean(rows: list[dict], key: str) -> float:
    values = [r.get(key) for r in rows if isinstance(r.get(key), (int, float))]
    return round(sum(values) / len(values), 2) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("a", type=Path)
    parser.add_argument("b", type=Path)
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    args = parser.parse_args()

    a, b = load(args.a), load(args.b)
    seeds = sorted(set(a) & set(b))
    if not seeds:
        raise SystemExit("两个目录没有共同种子")
    rows_a = [a[s] for s in seeds]
    rows_b = [b[s] for s in seeds]
    wins_a = sum(r["won"] for r in rows_a)
    wins_b = sum(r["won"] for r in rows_b)
    only_a = sum(r["won"] and not b[s]["won"] for s, r in zip(seeds, rows_a))
    only_b = sum(r["won"] and not a[s]["won"] for s, r in zip(seeds, rows_b))

    def replants(rows: list[dict]) -> tuple[float, float]:
        all_events = [e for r in rows for e in (r.get("replant_events") or [])]
        same = [e for e in all_events if e.get("same_type")]
        return round(len(all_events) / len(rows), 2), round(len(same) / len(rows), 2)

    ra, sa = replants(rows_a)
    rb, sb = replants(rows_b)
    out = {
        "seeds": len(seeds),
        args.label_a: {"won": wins_a, "solo_wins": only_a,
                       "mowers_per_game": mean(rows_a, "mowers_used"),
                       "searches_per_game": mean(rows_a, "searched"),
                       "overrides_per_game": mean(rows_a, "overrides"),
                       "seconds_per_game": mean(rows_a, "seconds"),
                       "replants_per_game": ra, "first_aid_replants_per_game": sa,
                       "student_candidate_added": mean(rows_a, "student_candidate_added"),
                       "student_candidate_best": mean(rows_a, "student_candidate_best"),
                       "student_candidate_override": mean(rows_a, "student_candidate_override")},
        args.label_b: {"won": wins_b, "solo_wins": only_b,
                       "mowers_per_game": mean(rows_b, "mowers_used"),
                       "searches_per_game": mean(rows_b, "searched"),
                       "overrides_per_game": mean(rows_b, "overrides"),
                       "seconds_per_game": mean(rows_b, "seconds"),
                       "replants_per_game": rb, "first_aid_replants_per_game": sb,
                       "student_candidate_added": mean(rows_b, "student_candidate_added"),
                       "student_candidate_best": mean(rows_b, "student_candidate_best"),
                       "student_candidate_override": mean(rows_b, "student_candidate_override")},
        "mcnemar_exact_p": round(mcnemar_exact(only_a, only_b), 4),
        "same_outcome": sum(a[s]["won"] == b[s]["won"] for s in seeds),
    }
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
