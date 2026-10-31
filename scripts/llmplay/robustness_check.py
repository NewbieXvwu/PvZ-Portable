#!/usr/bin/env python3
"""task81d：稳健性接口的三条实测（要引擎，单进程，最多 2 个游戏进程）。

1. **可复现**：同一打法、同一检查点种子，两次 `robustness()` 的胜局数逐字相同。
2. **RESEED 只改未来**：换种子前后当前状态（tick / 阳光 / 植物数）不变，
   但之后出的僵尸不同。
3. **预算计数**：两次检查点 0 试打，精确记 2 × k × max_ticks 个搜索 tick。

    python3 scripts/llmplay/robustness_check.py [--executable PATH] [--resource-dir DIR]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))

from llmplay.game import Game  # noqa: E402

TASK = 'demo_roof_41_doom_m200'


def policy(game):
    game.wait(30)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--executable')
    parser.add_argument('--resource-dir')
    args = parser.parse_args()
    failures = []

    game = Game(TASK, executable=args.executable, resource_dir=args.resource_dir)
    game.wait(600)

    before = dict(game.session.level(game.budget_key))
    first = game.robustness(policy, k=2, checkpoints=[0], max_ticks=300)
    second = game.robustness(policy, k=2, checkpoints=[0], max_ticks=300)
    after = dict(game.session.level(game.budget_key))
    print('可复现：%s' % ('OK' if first == second else 'FAILED'))
    if first != second:
        failures.append('同一打法两次结果不同')

    spent = after['search_ticks'] - before['search_ticks']
    want = 2 * 2 * 300
    print('预算：搜索 tick +%d（应为 %d）%s' % (spent, want, 'OK' if spent == want else 'FAILED'))
    if spent != want:
        failures.append('搜索预算不符')

    state = []
    futures = []
    for seed in (83_000_000, 83_000_001):
        trial = Game(TASK, executable=args.executable, resource_dir=args.resource_dir)
        trial.wait(600)
        before_reseed = trial.env.observe()
        trial.env.reseed(seed)
        trial.obs = trial.env.observe()
        visible = ('tick', 'sun', 'plants', 'zombies', 'defenses', 'packets')
        state.append(all(before_reseed.get(key) == trial.obs.get(key) for key in visible))
        trial.wait(2400)
        futures.append(trial.obs.get('zombies') or [])
        trial.close()
    same_state = all(state)
    different_future = futures[0] != futures[1]
    print('RESEED 只改未来：状态不变 %s / 未来不同 %s'
          % ('OK' if same_state else 'FAILED', 'OK' if different_future else 'FAILED'))
    if not same_state:
        failures.append('RESEED 改动了当前状态')
    if not different_future:
        failures.append('两个种子的未来一样（RESEED 没生效？）')

    game.close()
    if failures:
        print('ROBUSTNESS CHECK FAILED: ' + '; '.join(failures))
        return 1
    print('ROBUSTNESS CHECK OK')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
