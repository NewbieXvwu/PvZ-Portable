"""Does moving the student's early front sunflower / close potato mine actually save mowers?  Many seeds, paired.

    python3 scripts/early_fix_check.py --student artifacts/student/v30/model.pt --seeds 30000:30256 --workers 2 \
        --out artifacts/teacher_diag/early_fix_v30_z20.jsonl

One `whatif` on one game is a single sample of a chaotic system: at decision 1 of v30 z20 seed 30005, 8 of the
29 legal first moves lose and the mower count of the winning ones spans 0-5, so "this change gave 0 mowers" says
nothing by itself.  This script asks the same question over many games instead.  Per seed it captures the
student's game (episode_query capture), finds
  sun   the first sunflower planted in waves 0-3 at column >= 4, and
  mine  the first potato mine planted in waves 0-10 with the front zombie of its row at most one column ahead,
and replays each from that decision (the `whatif` replay, the student plays on) with three choices:
  orig     the student's own action (reproduces the game),
  fix      the same plant in the same row moved back (sun: leftmost empty column; mine: the empty column nearest
           to three columns in front of the zombie, at least two),
  placebo  a perturbation of the same size that does not test the idea (sun: another empty column >= 4 of the
           same row; mine: wait 60 ticks and let the student decide again).
fix - placebo is the effect of the idea; placebo - orig is the noise any change brings.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from episode_query import DEFAULT_RESOURCE_DIR, _run_variant  # noqa: E402
from render_episode import _col_of  # noqa: E402


def frames_of(archive: Path) -> list[dict]:
    from pvz_frame_store import frame_path, read_frames
    return read_frames(frame_path(archive))


def empty(frame: dict, row: int, col: int) -> bool:
    return not any(p[0] == row and p[1] == col for p in frame["plants"]) and \
        not any(g[0] == row and g[1] == col for g in frame.get("graves") or [])


def front_col(frame: dict, row: int) -> int | None:
    cols = [_col_of(z[2]) for z in frame["zombies"] if z[0] == row and z[5]]
    return min(cols) if cols else None


def find_cases(frames: list[dict]) -> dict:
    out = {}
    for d in range(1, len(frames)):
        a, state = frames[d].get("action"), frames[d - 1]
        if not a or a.get("type") != "plant" or frames[d].get("action_rejected"):
            continue
        kind, row, col = state["packets"][a["packet"]][1], a["row"], a["col"]
        plant = lambda c: {"type": "plant", "packet": a["packet"], "row": row, "col": c}
        if kind == 1 and "sun" not in out and state["wave"] <= 3 and col >= 4:
            back = [c for c in range(col) if empty(state, row, c)]
            other = [c for c in range(4, 9) if c != col and empty(state, row, c)]
            if back and other:
                out["sun"] = {"decision": d, "wave": state["wave"], "orig": plant(col), "fix": plant(back[0]),
                              "placebo": plant(other[len(other) // 2])}
        if kind == 4 and "mine" not in out and state["wave"] <= 10:
            z = front_col(state, row)
            if z is not None and z - col <= 1:
                far = sorted((c for c in range(min(z - 1, 9)) if empty(state, row, c)), key=lambda c: abs(z - 3 - c))
                if far:
                    out["mine"] = {"decision": d, "wave": state["wave"], "zombie_col": z, "orig": plant(col),
                                   "fix": plant(far[0]), "placebo": {"type": "wait", "ticks": 60}}
        if len(out) == 2:
            break
    return out


def run_seed(args: tuple) -> dict:
    seed, student, manifest, task_id = args
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "ep"
        subprocess.run([sys.executable, str(ROOT / "scripts/episode_query.py"), "capture", "--policy", "student",
                        "--checkpoint", student, "--task-manifest", manifest, "--task-id", task_id,
                        "--seed", str(seed), "--out", str(archive)], check=True, capture_output=True, cwd=ROOT)
        frames = frames_of(archive)
        meta = json.loads((archive / "meta.json").read_text())
        record = {"seed": seed, "won": meta["outcome"]["won"]}
        for name, case in find_cases(frames).items():
            # the same replay `episode_query.py whatif` runs: recorded actions up to the decision, the override, then the student
            prefix = [f.get("action") for f in frames[1:case["decision"]]]
            case["result"] = {}
            for key in ("orig", "fix", "placebo"):
                o = _run_variant(meta, {case["decision"]: case[key]}, DEFAULT_RESOURCE_DIR, prefix=prefix)
                case["result"][key] = {k: o[k] for k in ("won", "final_wave", "mowers_used", "plants_eaten", "override_rejected")}
            record[name] = case
        return record


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--student", required=True)
    p.add_argument("--seeds", default="30000:30256")
    p.add_argument("--manifest", default="artifacts/stage4/day_l7_hard.json")
    p.add_argument("--task-id", default="day_l7_z20_eval")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    lo, hi = map(int, args.seeds.split(":"))
    done = {json.loads(line)["seed"] for line in args.out.open()} if args.out.exists() else set()
    todo = [(s, args.student, args.manifest, args.task_id) for s in range(lo, hi) if s not in done]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(args.workers) as pool, args.out.open("a") as out:
        for record in pool.map(run_seed, todo):
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()


if __name__ == "__main__":
    main()
