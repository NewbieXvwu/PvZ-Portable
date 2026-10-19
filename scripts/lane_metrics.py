"""The three lane-mismatch metrics (MEMORY/notes/lane-mismatch.md) over a set of recorded games.

    python scripts/lane_metrics.py --data artifacts/student/frames --label d27S_z17_fast

Per game, then averaged over games (decisions inside one game are not independent):
  A  share of shooter plantings placed behind the front zombie of their row (plant column >= that zombie's column;
     only rows that have a zombie on the board count);
  B  the same share, but only for plantings made while some *other* row was under pressure with no shooter able to
     hit its front zombie ("bare lane");
  C  bare lanes per frame, waves 5-15 (the wave window keeps early deaths from looking good).
Uses the repository's own reachability rule, `render_episode._reachable_dps`. Lower is better for all three.
The state an action was taken in is the frame before it (the same pairing as the student's training rows).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1] / "python")]
from pvz_frame_store import frame_paths, read_frames  # noqa: E402
from render_episode import SHOOTER_TYPES, _col_of, _reachable_dps  # noqa: E402


def front_x(frame: dict, row: int) -> float | None:
    xs = [z[2] for z in frame["zombies"] if z[0] == row and z[5]]
    return min(xs) if xs else None


def bare_rows(frame: dict) -> set[int]:
    out = set()
    for row in frame.get("lanes") or range(5):
        x = front_x(frame, row)
        if x is not None and _reachable_dps(frame, row, x)[1] == 0:
            out.add(row)
    return out


def game_metrics(frames: list[dict]) -> dict:
    behind = placed = behind_b = placed_b = 0
    bare_counts = []
    for i in range(1, len(frames)):
        state = frames[i - 1]
        if 5 <= state["wave"] <= 15:
            bare_counts.append(len(bare_rows(state)))
        action = frames[i].get("action")
        if not action or action.get("type") != "plant" or frames[i].get("action_rejected"):
            continue
        kind = state["packets"][action["packet"]][1]
        if kind not in SHOOTER_TYPES:
            continue
        x = front_x(state, action["row"])
        if x is None:
            continue
        is_behind = action["col"] >= _col_of(x)
        placed += 1
        behind += is_behind
        if bare_rows(state) - {action["row"]}:
            placed_b += 1
            behind_b += is_behind
    return {"A": behind / placed if placed else None, "B": behind_b / placed_b if placed_b else None,
            "C": sum(bare_counts) / len(bare_counts) if bare_counts else None,
            "won": frames[-1].get("result") == 1}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--label", required=True, help="archive directory name, e.g. d27S_z17_fast")
    args = parser.parse_args()
    games = [game_metrics(read_frames(p)) for p in frame_paths(args.data) if p.parent.name == args.label]

    def mean(rows, key):
        values = [r[key] for r in rows if r[key] is not None]
        return round(sum(values) / len(values), 4) if values else None

    out = {"games": len(games), "wins": sum(g["won"] for g in games)}
    for name, rows in (("all", games), ("won", [g for g in games if g["won"]]), ("lost", [g for g in games if not g["won"]])):
        out[name] = {key: mean(rows, key) for key in ("A", "B", "C")}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
