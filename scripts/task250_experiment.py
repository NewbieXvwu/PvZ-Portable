"""task250 matched games: quick 16-pair validation and the full new-seed panel.

Runs the existing search_play teacher and task243 paired statistics.  The quick phase reuses
the task246 production outcomes for the same seeds; the full panel runs both arms on a fresh
seed segment so the routing estimate and any fix are measured out of sample.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "python")]
from task243_teacher import paired


def outcome_rows(target):
    games = [json.loads(line) for line in (target / "games.jsonl").read_text().splitlines()]
    return games


def run_arm(py, base, panel, label, model, args, levels, seeds, margins, out):
    env = dict(os.environ, PVZ_NO_POPUP="1", OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1",
               TMPDIR=str(base / "work"))
    rows = []
    for task in panel["tasks"]:
        if task["level"] not in levels:
            continue
        task = dict(task)
        task["seeds"] = seeds[task["level"]]
        task_panel = {"tasks": [task]}
        manifest = out / f"manifest-l{task['level']}.json"
        manifest.write_text(json.dumps(task_panel, indent=1))
        target = out / f"l{task['level']}"
        target.mkdir(parents=True, exist_ok=True)
        command = [py, str(ROOT / "scripts/search_play.py"),
            "--rollout-policy", str(base / "models/student.pt"),
            "--rollout-policy-gpt", str(base / "models/gptbc.pt"),
            "--value", str(model["value"]),
            "--value-roof", str(model["roof"]),
            "--static-value", str(base / "models/static.pt"),
            "--manifest", str(manifest), "--task-id", task["task_id"],
            "--seed-list", ",".join(str(s) for s in seeds[task["level"]]),
            "--arms", "fast_search", "--workers", str(args.workers),
            "--verify-samples", "9", "--student-candidates", "2", "--native-games",
            "--margin-relative", str(margins.get(task["level"], args.margin_relative)),
            "--executable", str(Path.home() / "pvz-mac-collect/build-vnbranch/pvz-portable"),
            "--out", str(target)]
        with (out / f"l{task['level']}.log").open("a") as log:
            subprocess.run(command, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=log, check=True)
        games = outcome_rows(target)
        if len(games) != len(seeds[task["level"]]) or not all(g["terminal"] for g in games):
            raise RuntimeError(f"L{task['level']}: incomplete terminal pairs")
        rows.extend(dict(level=task["level"], seed=g["seed"], won=g["won"], terminal=g["terminal"],
                         overrides=g["overrides"], searched=g["searched"], planted=g.get("planted_types"),
                         seconds=g["seconds"]) for g in games)
        print(label, task["level"], sum(g["won"] for g in games), "/", len(games), flush=True)
    (out / "outcomes.json").write_text(json.dumps(rows, indent=1) + "\n")
    return rows


def report(rows, reference):
    ref = {(r["level"], r["seed"]): r["won"] for r in reference}
    pairs = [dict(level=r["level"], seed=r["seed"], candidate=r["won"],
                  production=ref[(r["level"], r["seed"])])
             for r in rows if (r["level"], r["seed"]) in ref]
    per_level = {str(level): paired([p for p in pairs if p["level"] == level],
                                    "candidate", "production")
                 for level in sorted({p["level"] for p in pairs})}
    pooled = paired(pairs, "candidate", "production")
    return dict(per_level=per_level, pooled=pooled, pairs=pairs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "artifacts/task250")
    parser.add_argument("--phase", choices=("quick", "panel"), required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--model", default="finetune-t3")
    parser.add_argument("--margin-relative", type=float, default=0.02)
    parser.add_argument("--margins", default="", help="per-level overrides, e.g. 21:0.015,22:0.05")
    parser.add_argument("--levels", default="")
    parser.add_argument("--pairs", type=int, default=0)
    parser.add_argument("--seed-start", type=int, default=250100000)
    parser.add_argument("--seed-stride", type=int, default=1000)
    parser.add_argument("--stride", type=int, default=1,
                        help="quick phase: take every Nth seed of the task246 panel")
    parser.add_argument("--offset", type=int, default=0, help="quick phase: first panel seed index")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    base = args.data.resolve()
    (base / "work").mkdir(exist_ok=True)
    panel = json.loads((base / "panel-original.json").read_text())
    margins = {int(item.split(":")[0]): float(item.split(":")[1])
               for item in args.margins.split(",") if item}
    levels = [int(x) for x in args.levels.split(",")] if args.levels else (
        [21, 22, 24] if args.phase == "quick" else [t["level"] for t in panel["tasks"]])
    model = {"value": base / "models" / f"{args.model}.pt"}
    model["roof"] = base / "models" / ("roof.pt" if args.model == "production" else f"{args.model}.pt")
    if args.phase == "quick":
        # The task246 panel overwrote each level's seed block, so the authoritative seed list
        # is the production outcomes, not the pre-panel manifest.
        reference = json.loads((base / "production-outcomes.json").read_text())
        seeds = {}
        for level in levels:
            measured = sorted({r["seed"] for r in reference if r["level"] == level})
            seeds[level] = measured[args.offset:][::args.stride][:args.pairs or 16]
        out = base / f"quick-{args.label}"
    else:
        seeds = {t["level"]: [args.seed_start + i * args.seed_stride + k
                              for k in range(args.pairs or 32)]
                 for i, t in enumerate(panel["tasks"]) if t["level"] in levels}
        out = base / f"panel-{args.label}"
        reference = None
    out.mkdir(exist_ok=True)
    rows = run_arm(sys.executable, base, panel, args.label, model, args, levels, seeds, margins, out)
    if args.phase == "quick":
        result = report(rows, reference)
        (out / "report.json").write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps(result["per_level"]), flush=True)
        print(json.dumps(result["pooled"]), flush=True)


if __name__ == "__main__":
    main()
