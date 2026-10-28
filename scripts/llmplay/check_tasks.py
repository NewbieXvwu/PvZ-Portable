#!/usr/bin/env python3
"""task81c：无头 reset 核对——llmplay 任务清单里每个族都要能起局。

对 `configs/llmplay_seeds.json` 里的每个任务：用包里那套推导（Game → TaskSpec）
reset 一次、推进 120 tick，打印关卡 / 倍率 / 波次 / 是否终止。
任何一个起不来就非零退出。

    python3 scripts/llmplay/check_tasks.py [--executable PATH] [--resource-dir DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))

from llmplay.game import Game  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--executable')
    parser.add_argument('--resource-dir')
    args = parser.parse_args()

    # 核对脚本用一次性会话文件，不污染宿主真实任务的搜索/重写计数。
    os.environ.setdefault('PVZ_LLMPLAY_SESSION', str(Path(tempfile.mkdtemp(prefix='llmplay_check_')) / 'budget.json'))

    plan = json.loads((ROOT / 'configs' / 'llmplay_seeds.json').read_text(encoding='utf-8'))
    tasks = {task['task_id']: task for task in
             json.loads((ROOT / 'configs' / 'human_demo_tasks.json').read_text(encoding='utf-8'))['tasks']}
    failures = []
    for task_id in plan['order']:
        task = tasks[task_id]
        layer = task.get('layer')
        seed = plan['tasks'][task_id][0]
        try:
            # 逐关起局只验证任务本身；层序/卡组规则由 workflow 的单测覆盖。
            deck = task.get('deck') or (task.get('reference_deck') or [])
            if task.get('deck_selection') == 'player' and len(deck) < 6:
                deck = [1, 33, 39, 44, 15, 3] if task['level'] >= 41 else [0, 1, 2, 3, 4, 5]
            game = Game(task_id, seed=seed, deck=deck, reason='check_tasks 起局核对',
                        strategy='核对', executable=args.executable,
                        resource_dir=args.resource_dir, _gate=False)
            game.wait(120)
            spec = game.spec
            print('%-4s %-32s L%-2d m=%-4s tick=%-5d wave=%-2d terminal=%s'
                  % (layer, task_id, spec.level, spec.zombie_count_multiplier,
                     game.tick, int(game.obs.get('wave') or 0), game.terminal))
            game.close()
        except Exception as error:      # noqa: BLE001 - 报错就是结论
            failures.append((task_id, '%s: %s' % (type(error).__name__, error)))
            print('FAIL %-4s %-32s %s' % (layer, task_id, failures[-1][1][:160]))
    if failures:
        print('CHECK FAILED: %d/%d' % (len(failures), len(plan['order'])))
        return 1
    print('CHECK OK: %d tasks' % len(plan['order']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
