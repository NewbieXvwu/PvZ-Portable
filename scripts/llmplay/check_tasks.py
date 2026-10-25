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
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))

from llmplay.game import Game  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--executable')
    parser.add_argument('--resource-dir')
    args = parser.parse_args()

    plan = json.loads((ROOT / 'configs' / 'llmplay_seeds.json').read_text(encoding='utf-8'))
    failures = []
    for layer in ('A', 'B', 'C', 'D', 'E'):
        for task_id in plan['layers'][layer]:
            seed = plan['tasks'][task_id][0]
            try:
                game = Game(task_id, seed=seed, executable=args.executable,
                            resource_dir=args.resource_dir)
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
        print('CHECK FAILED: %d/%d' % (len(failures), len(plan['tasks'])))
        return 1
    print('CHECK OK: %d tasks' % len(plan['tasks']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
