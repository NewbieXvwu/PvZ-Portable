#!/usr/bin/env python3
"""task104 面板评估：把一份面板清单跑成"每个模型 × 每个任务"的胜局表。

面板文件由 task104 冻结（`MEMORY/docs/task104-regression-panel.json` 与屋顶面板），
每个任务自带显式种子列表。脚本把它们写成 search_play 认的任务清单，然后对每个模型
每关调一次 `scripts/search_play.py --arms fast --seed-list ...`（纯学生，不搜索、不代打）。

  python3 scripts/task104_panel_eval.py --panel MEMORY/docs/task104-regression-panel.json \
      --model a_s1=artifacts/task104/train/a_s1/student/model.pt \
      --model b_s1=artifacts/task104/train/b_s1/student/model.pt \
      --value artifacts/hard_mac/value_v3/model.pt --out artifacts/task104/eval/regression

  # 只跑一个模型的一关，用于冒烟
  python3 scripts/task104_panel_eval.py ... --levels 42 --models a_s1

每个模型的 games.jsonl 留在 `<out>/<model>/<task_id>/`，search_play 自己跳过已完成的种子，
所以中断后重跑不会重复。汇总写 `<out>/summary.json`。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from eval_panel import read_games, summarize  # noqa: E402

TASK_KEYS = ("task_id", "level", "deck", "seeds", "wave_cap", "zombie_count_multiplier",
             "preplanted", "playthrough", "conveyor_start", "fog_observation",
             "first_aid_owned", "deck_random", "splits", "sun_start")


def manifest_for(panel: dict, path: Path, levels: list[int] | None) -> dict:
    tasks = []
    for task in panel["tasks"]:
        if levels and int(task["level"]) not in levels:
            continue
        entry = {key: task[key] for key in TASK_KEYS if key in task}
        if not entry.get("seeds"):
            raise ValueError(f"{task['task_id']}: panel task has no seeds")
        tasks.append(entry)
    if not tasks:
        raise ValueError("panel selection is empty")
    document = {"tasks": tasks}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=1) + "\n")
    return document


def run_task(args, model_path: Path, task: dict, out: Path) -> list[dict]:
    seeds = ",".join(str(seed) for seed in task["seeds"])
    command = [sys.executable, str(ROOT / "scripts/search_play.py"),
               "--rollout-policy", str(model_path),
               "--value", str(args.value),
               "--manifest", str(args.manifest),
               "--task-id", task["task_id"],
               "--seed-list", seeds,
               "--arms", "fast",
               "--workers", str(args.workers),
               "--out", str(out)]
    if args.static_value:
        command += ["--static-value", str(args.static_value)]
    if args.resource_dir:
        command += ["--resource-dir", str(args.resource_dir)]
    if args.executable:
        command += ["--executable", str(args.executable)]
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)
    return read_games(out / "games.jsonl")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--panel", type=Path, required=True)
    p.add_argument("--model", action="append", default=[], metavar="LABEL=PATH",
                   help="student checkpoint, repeatable")
    p.add_argument("--models", default="", help="comma separated labels to run (default: all)")
    p.add_argument("--value", type=Path, required=True)
    p.add_argument("--static-value", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=2, help="game processes; task104 caps this at 2")
    p.add_argument("--levels", default="", help="comma separated levels (default: all)")
    p.add_argument("--resource-dir", type=Path)
    p.add_argument("--executable", type=Path)
    args = p.parse_args()

    models = {}
    for item in args.model:
        label, _, path = item.partition("=")
        if not label or not path:
            p.error(f"--model needs LABEL=PATH, got {item!r}")
        models[label] = Path(path)
    if args.models:
        wanted = [name.strip() for name in args.models.split(",") if name.strip()]
        models = {name: models[name] for name in wanted}
    if not models:
        p.error("no models selected")

    panel = json.loads(args.panel.read_text())
    levels = [int(level) for level in args.levels.split(",")] if args.levels else None
    args.out.mkdir(parents=True, exist_ok=True)
    args.manifest = args.out / "manifest.json"
    document = manifest_for(panel, args.manifest, levels)

    summary = {"panel": str(args.panel), "manifest": str(args.manifest), "arms": "fast",
               "workers": args.workers, "models": {}}
    for label, model_path in models.items():
        per_task = []
        for task in document["tasks"]:
            games = run_task(args, model_path, task, args.out / label / task["task_id"])
            row = {"task_id": task["task_id"], "level": task["level"],
                   "terrain": next((t.get("terrain") for t in panel["tasks"]
                                    if t["task_id"] == task["task_id"]), None),
                   "seeds": task["seeds"], **summarize(games)}
            per_task.append(row)
            print(f"{label} {task['task_id']}: {row.get('won', 0)}/{row.get('games', 0)}", flush=True)
        won = sum(row.get("won", 0) for row in per_task)
        played = sum(row.get("games", 0) for row in per_task)
        terrains = {}
        for row in per_task:
            bucket = terrains.setdefault(row["terrain"] or "unknown", {"won": 0, "games": 0})
            bucket["won"] += row.get("won", 0)
            bucket["games"] += row.get("games", 0)
        summary["models"][label] = {"checkpoint": str(model_path), "won": won, "games": played,
                                    "win_rate": round(won / played, 4) if played else None,
                                    "terrains": terrains, "tasks": per_task}
        (args.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
        print(f"{label}: {won}/{played}", flush=True)

    print(json.dumps({name: {"won": row["won"], "games": row["games"],
                             "win_rate": row["win_rate"]} for name, row in summary["models"].items()},
                     indent=1))
    print(f"wrote {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
