#!/usr/bin/env python3
"""task137 冒烟：v5 随机卡组格能被引擎接受，且 0.5 / 0.75 档按 lround 生效。

对同一个格子、同一个种子，只改 multiplier 跑前几波（不放植物，按 5 tick 步进），
用 1.0 档的每波数量当原版基数，检查低档是否等于 lround(base × mult)。

用法（Mac 上串行跑，最多 4 个进程）：
  python3 scripts/smoke_random_deck_cells.py --task-id rd137_l7_train_d01 --seed 210000000
  python3 scripts/smoke_random_deck_cells.py --task-id rd137_l24_train_d01 --seed 213000000 --waves 3
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]

from pvz_env import PvZEnv  # noqa: E402
from smoke_zombie_multiplier import count_waves  # noqa: E402

DEFAULT_PANEL = ROOT / 'artifacts/task137/training_panel.json'
LOW = (0.5, 0.75, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--panel', type=Path, default=DEFAULT_PANEL)
    parser.add_argument('--task-id', default='rd137_l7_train_d01')
    parser.add_argument('--seed', type=int, default=213_000_000)
    parser.add_argument('--waves', type=int, default=3)
    parser.add_argument('--executable', type=Path, default=ROOT / 'build/pvz-portable')
    parser.add_argument('--resource-dir', type=Path, default=Path.home() / '.cache/pvz-research-resources')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()

    panel = json.loads(args.panel.read_text())
    table = {task['task_id']: task for task in panel['tasks']}
    base = table[args.task_id]
    deck = tuple(base['deck'])
    level = base['level']
    counts, dirty = {}, {}
    with PvZEnv(args.resource_dir, executable=args.executable) as env:
        for multiplier in LOW:
            counts[multiplier], dirty[multiplier] = count_waves(
                env, level, args.seed, deck, multiplier, args.waves)
            print(f'{args.task_id} m{round(multiplier * 100):03d} deck={list(deck)} '
                  f'waves={counts[multiplier]} dirty={sorted(dirty[multiplier])}', flush=True)

    # 1.0 档就是原版基数；低档必须是 lround(base × mult)，且只有 ≥2 只的波会变。
    base_waves = counts[1.0]
    checked = 0
    for wave, original in sorted(base_waves.items()):
        if wave in dirty[1.0]:
            continue
        for multiplier in (0.5, 0.75):
            actual = counts[multiplier].get(wave, 0)
            expected = int(math.floor(original * multiplier + 0.5))
            assert actual == expected, (wave, multiplier, original, actual, expected)
            checked += 1
    result = dict(task_id=args.task_id, level=level, seed=args.seed, deck=list(deck),
                  waves={str(m): counts[m] for m in LOW},
                  dirty={str(m): sorted(dirty[m]) for m in LOW},
                  checked_pairs=checked,
                  note='aSpawnCount = lround(base * multiplier); graveyard/pool/sky extras are not scaled')
    print(json.dumps(result, ensure_ascii=False))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1) + '\n')


if __name__ == '__main__':
    main()
