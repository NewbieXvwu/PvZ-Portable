#!/usr/bin/env python3
"""task85：L50 僵王博士原生传送带族。"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from make_night_panel import original_task_document  # noqa: E402


def build_tasks():
    tasks = []
    for split, base, count in (("eval", 85_000_000, 32), ("train", 85_100_000, 8192)):
        task = original_task_document(50, f"boss_l50_orig_{split}", list(range(base, base + count)))
        task["terrain"] = "boss"
        tasks.append(task)
    return tasks


if __name__ == "__main__":
    path = ROOT / "artifacts/task85/boss_panel.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema_version": 1, "tasks": build_tasks()}, indent=1) + "\n")
    print(path)
