#!/usr/bin/env python3
"""白天面板评估：一个学生模型（或老师）在 `day_panel.json` 的每关各跑 N 局。

面板任务由 `scripts/make_day_panel.py` 生成（原版难度：倍率 1.0、不限波、
卡组 = `pvz_tasks.deck_for_level`、种子段 30000..31023）。本脚本只是一个薄包装：
每关调用一次 `scripts/search_play.py`，把 `games.jsonl` 汇总成每关的
胜局数、平均丢车数（`mowers_used`）和平均到达波数（`final_wave`）。

  * 学生：`--arms fast`（纯学生，不搜索）。
  * 老师：`--arms fast_search` 加上 `artifacts/hard_mac/round15_flags`
    里的参数（`--verify-samples 9 --value ... --static-value ...`）。

每个关卡的 `games.jsonl` 会保留，重跑时 search_play 自己跳过已完成的种子，
所以中途停掉再跑不会重复。

用法：
  python3 scripts/eval_panel.py --rollout-policy artifacts/student/v32/model.pt \
      --value artifacts/hard_mac/value_v3/model.pt \
      --out artifacts/day_panel/v32 --seeds 256

  python3 scripts/eval_panel.py --rollout-policy artifacts/student/v32/model.pt \
      --value artifacts/hard_mac/value_v3/model.pt \
      --static-value artifacts/hard_mac/value_branch1_desk/model.pt --verify-samples 9 \
      --arms fast_search --out artifacts/day_panel/teacher --seeds 32
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run_level(args, task_id: str, out: Path) -> dict:
    command = [sys.executable, str(ROOT / "scripts/search_play.py"),
               "--rollout-policy", str(args.rollout_policy),
               "--value", str(args.value),
               "--manifest", str(args.manifest),
               "--task-id", task_id,
               "--seeds", str(args.seeds),
               "--arms", args.arms,
               "--workers", str(args.workers),
               "--out", str(out)]
    if args.static_value:
        command += ["--static-value", str(args.static_value)]
    if args.verify_samples:
        command += ["--verify-samples", str(args.verify_samples)]
    if args.student_candidates:
        command += ["--student-candidates", str(args.student_candidates)]
    if args.resource_dir:
        command += ["--resource-dir", str(args.resource_dir)]
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)
    return read_games(out / "games.jsonl")


def read_games(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize(games: list[dict]) -> dict:
    games = [game for game in games if game.get("arm") == "fast" or game.get("arm") == "fast_search"]
    if not games:
        return {"games": 0}
    count = len(games)
    return {
        "games": count,
        "won": sum(bool(game["won"]) for game in games),
        "win_rate": round(sum(bool(game["won"]) for game in games) / count, 4),
        "mean_mowers_used": round(sum(game.get("mowers_used", 0) for game in games) / count, 2),
        "mean_final_wave": round(sum(game.get("final_wave", 0) for game in games) / count, 2),
        "mean_seconds": round(sum(game.get("seconds", 0.0) for game in games) / count, 2),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rollout-policy", type=Path, required=True)
    p.add_argument("--value", type=Path, required=True)
    p.add_argument("--static-value", type=Path)
    p.add_argument("--verify-samples", type=int, default=0)
    p.add_argument("--student-candidates", type=int, default=0)
    p.add_argument("--manifest", type=Path, default=ROOT / "artifacts/stage4/day_panel.json")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seeds", type=int, default=256)
    p.add_argument("--arms", default="fast", choices=("fast", "fast_search"))
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--levels", default="", help="逗号分隔的关卡子集，默认面板全部")
    p.add_argument("--resource-dir", type=Path,
                   help="游戏资源目录（不传则用 search_play.py 的默认值）")
    args = p.parse_args()

    document = json.loads(args.manifest.read_text())
    tasks = {task["level"]: task for task in document["tasks"]}
    levels = [int(level) for level in args.levels.split(",")] if args.levels else sorted(tasks)
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for level in levels:
        task = tasks[level]
        out = args.out / f"l{level}"
        games = run_level(args, task["task_id"], out)
        row = {"level": level, "deck": task["deck"], **summarize(games)}
        rows.append(row)
        print(f"L{level}: {row['won']}/{row['games']} won, "
              f"{row['mean_mowers_used']} mowers, wave {row['mean_final_wave']}", flush=True)

    panel = {
        "manifest": str(args.manifest),
        "rollout_policy": str(args.rollout_policy),
        "value": str(args.value),
        "static_value": str(args.static_value) if args.static_value else None,
        "verify_samples": args.verify_samples,
        "student_candidates": args.student_candidates,
        "arms": args.arms,
        "seeds_requested": args.seeds,
        "levels": rows,
    }
    (args.out / "panel.json").write_text(json.dumps(panel, indent=1) + "\n")

    print()
    print("| 关卡 | 卡组 | 胜局 | 胜率 | 平均丢车 | 平均到达波数 |")
    print("|---|---|---|---|---|---|")
    for row in rows:
        print(f"| 1-{row['level']} | {row['deck']} | {row['won']}/{row['games']} | "
              f"{row['win_rate']:.3f} | {row['mean_mowers_used']} | {row['mean_final_wave']} |")
    print(f"\nwrote {args.out / 'panel.json'}")


if __name__ == "__main__":
    main()
