#!/usr/bin/env python3
"""task75：L25 / L30 / L40 原生传送带清单，不覆盖既有面板。"""
import json
from pathlib import Path

from make_night_panel import original_task_document

ROOT = Path(__file__).resolve().parents[1]
LEVELS = (25, 30, 40)
TRAIN_BASE = 60_000_000
EVAL_BASE = 61_000_000


def build_tasks():
    tasks = []
    for index, level in enumerate(LEVELS):
        prefix = 'storm' if level == 40 else 'pool'
        for split, base, count in (('eval', EVAL_BASE, 32), ('train', TRAIN_BASE, 8192)):
            start = base + index * 100_000
            task = original_task_document(level, f'{prefix}_l{level}_orig_{split}',
                                          list(range(start, start + count)))
            task['terrain'] = 'fog' if level == 40 else 'pool'
            if level == 40:
                task['fog_observation'] = 'storm'
            tasks.append(task)
    return tasks


def main():
    path = ROOT / 'artifacts/task75/native_conveyor_panel.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(schema_version=1, tasks=build_tasks()), indent=1) + '\n')
    print(path)


if __name__ == '__main__':
    main()
