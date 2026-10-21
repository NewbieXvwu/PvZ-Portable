#!/usr/bin/env python3
"""把若干 `eval_panel.py` 的结果合成任务书要的那张表，并回答“哪几关学生明显不如老师”。

输入可以是一个 `panel.json`，也可以是一个目录（读其中 `lN/games.jsonl`，
这样评估中途被停掉的部分结果也能用）。列 = 每个学生标签的“胜局/局数”，
再加老师一列、平均丢车、平均到达波数；缺的格子写 `-`。

用法（6 卡基线 + 台式机 7 卡结果回来后）：

  python3 scripts/day_panel_report.py \
      --student v32=artifacts/day_panel/v32 \
      --student d35rows2=artifacts/day_panel/d35rows2_d7 \
      --teacher 老师=artifacts/day_panel/teacher_d7 \
      --deck 0,1,2,3,4,5,7

输出 markdown 到 stdout（`--out` 可写文件）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_games(path: Path) -> dict[int, dict]:
    """level -> 汇总。path 是 panel.json 或含 lN/games.jsonl 的目录。"""
    if path.is_file():
        document = json.loads(path.read_text())
        return {row["level"]: row for row in document["levels"]}
    rows = {}
    for level_dir in sorted(path.glob("l*")):
        try:
            level = int(level_dir.name[1:])
        except ValueError:
            continue
        games = []
        for line in (level_dir / "games.jsonl").read_text().splitlines() if (level_dir / "games.jsonl").exists() else []:
            if line.strip():
                games.append(json.loads(line))
        if not games:
            continue
        count = len(games)
        rows[level] = {
            "level": level,
            "deck": None,
            "games": count,
            "won": sum(bool(game["won"]) for game in games),
            "win_rate": round(sum(bool(game["won"]) for game in games) / count, 4),
            "mean_mowers_used": round(sum(game.get("mowers_used", 0) for game in games) / count, 2),
            "mean_final_wave": round(sum(game.get("final_wave", 0) for game in games) / count, 2),
        }
    return rows


def cell(row: dict | None) -> str:
    if not row:
        return "-"
    return f"{row['won']}/{row['games']}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--student", action="append", default=[], metavar="LABEL=PATH",
                   help="学生结果，可重复")
    p.add_argument("--teacher", default="", metavar="LABEL=PATH")
    p.add_argument("--deck", default="", help="表头里显示的卡组，默认从结果里取")
    p.add_argument("--gap", type=float, default=0.10, help="“明显不如老师”的胜率差阈值")
    p.add_argument("--area", type=int, default=1, help="冒险模式区号，只影响关卡标签（1 -> 1-7，2 -> 2-7）")
    p.add_argument("--out", type=Path)
    args = p.parse_args()

    def label(level: int) -> str:
        return f"{args.area}-{level - (args.area - 1) * 10}"

    students = []
    for spec in args.student:
        label, _, path = spec.partition("=")
        students.append((label, load_games(Path(path))))
    teacher = None
    if args.teacher:
        label, _, path = args.teacher.partition("=")
        teacher = (label, load_games(Path(path)))

    levels = sorted({level for _, rows in students for level in rows} |
                    ({level for level in teacher[1]} if teacher else set()))
    deck = args.deck
    if not deck:
        for _, rows in students:
            for row in rows.values():
                if row.get("deck"):
                    deck = ",".join(str(seed) for seed in row["deck"])
                    break
            if deck:
                break

    lines = [f"| 关卡 | 卡组 | " + " | ".join(f"{label} 胜局" for label, _ in students) +
             (f" | {teacher[0]} 胜局" if teacher else "") + " | 平均丢车 | 平均到达波数 |",
             "|---" * (3 + len(students) + (1 if teacher else 0)) + "|"]
    gaps = []
    for level in levels:
        first = next((rows[level] for _, rows in students if level in rows), None)
        if not first:
            continue
        mowers = first["mean_mowers_used"]
        waves = first["mean_final_wave"]
        cells = [cell(rows.get(level)) for _, rows in students]
        row = f"| {label(level)} | [{deck}] | " + " | ".join(cells)
        if teacher:
            teacher_row = teacher[1].get(level)
            row += f" | {cell(teacher_row)}"
            if teacher_row:
                gaps.append((teacher_row["win_rate"] - first["win_rate"], level,
                             first["win_rate"], teacher_row["win_rate"]))
        row += f" | {mowers} | {waves} |"
        lines.append(row)

    lines.append("")
    if teacher and gaps:
        compared = sorted(level for _, level, _, _ in gaps)
        lines.append(f"学生明显不如老师（胜率差 > {args.gap:.0%}，按差值排序；"
                     f"只比较有老师数据的关卡 {', '.join(label(level) for level in compared)}）：")
        worse = [item for item in sorted(gaps, reverse=True) if item[0] > args.gap]
        if worse:
            for gap, level, student_rate, teacher_rate in worse:
                lines.append(f"- {label(level)}：学生 {student_rate:.1%} vs 老师 {teacher_rate:.1%}"
                             f"（差 {gap:+.1%}）")
        else:
            lines.append("- 无（这些关卡的学生胜率都在阈值内）")
        missing = [level for level in levels if level not in compared]
        if missing:
            lines.append(f"- 未比较（没有老师数据）：{', '.join(label(level) for level in missing)}")
    elif teacher:
        lines.append("老师结果还没有，无法比较。")
    else:
        lines.append("没有给老师结果（--teacher），无法比较。")

    text = "\n".join(lines) + "\n"
    if args.out:
        args.out.write_text(text)
        print(f"wrote {args.out}")
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
