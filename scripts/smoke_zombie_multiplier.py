#!/usr/bin/env python3
"""task29 第 1 步冒烟：确认 `zombie_count_multiplier` 真的按 `lround(base * mult)` 放大常规波次。

做法：同一关、同一种子、同一卡组，只改 multiplier；不放任何植物，按 5 tick 步进，
统计**每一波新增的僵尸数**（相邻观测的僵尸数增量，跨波时归到新波）。前几波僵尸还没走到
房子，所以增量就是这一波实际刷出的数量，可以直接和 `std::lround(base * mult)` 对。
某波里如果同时有刷怪和小推车碾死（僵尸数下降），这一波的增量会被低估，冒烟把它标成
dirty、不参与严格断言（本脚本只证明倍率生效，不统计死亡）。

坟头 / 泳池 / 屋顶在最后一波之后的额外刷怪不读 multiplier（`Board::SpawnZombiesFromGraves`
Board.cpp:4850、`SpawnZombiesFromPool` Board.cpp:4728、`SpawnZombiesFromSky`），本冒烟只覆盖常规波次。

用法：
  python3 scripts/smoke_zombie_multiplier.py --level 7 --seed 1800000 --waves 4
  python3 scripts/smoke_zombie_multiplier.py --level 10 --seed 1880000 --waves 3 --conveyor-original
  python3 scripts/smoke_zombie_multiplier.py --level 12 --seed 2210000 --waves 4 --out artifacts/task29/smoke_l12.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]

from pvz_env import PvZEnv, TaskSpec  # noqa: E402
from pvz_tasks import deck_for_level, profile_for_deck  # noqa: E402

MULTIPLIERS = (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0)


def count_waves(env, level: int, seed: int, deck, multiplier: float, waves: int,
                conveyor_original: bool = False, step_ticks: int = 5,
                max_ticks: int = 200_000):
    """Return (per-wave spawned counts, waves where a zombie count dropped)."""
    task = TaskSpec(level=level, seed=seed, profile=profile_for_deck(deck),
                    zombie_count_multiplier=multiplier,
                    conveyor_start='original' if conveyor_original else None)
    obs, _ = env.reset(deck=deck, task=task)
    spawned: dict[int, int] = {}
    dirty: set[int] = set()
    previous = len(obs['zombies'])
    ticks = 0
    while ticks < max_ticks:
        obs, *_ = env.step({'type': 'wait', 'ticks': step_ticks})
        ticks += step_ticks
        current = len(obs['zombies'])
        wave = int(obs['wave'])
        if current > previous:
            spawned[wave] = spawned.get(wave, 0) + (current - previous)
        elif current < previous:
            dirty.add(wave)
        previous = current
        if obs['terminal'] or wave >= waves:
            break
    return spawned, dirty


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--level', type=int, default=7)
    parser.add_argument('--seed', type=int, default=1800000)
    parser.add_argument('--waves', type=int, default=4)
    parser.add_argument('--conveyor-original', action='store_true',
                        help='原版传送带开局（deck 为空，卡由引擎发）')
    parser.add_argument('--executable', type=Path, default=ROOT / 'build/pvz-portable')
    parser.add_argument('--resource-dir', type=Path, default=Path.home() / '.cache/pvz-research-resources')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    deck = () if args.conveyor_original else deck_for_level(args.level)
    results = {}
    dirty = {}
    with PvZEnv(args.resource_dir, executable=args.executable) as env:
        for multiplier in MULTIPLIERS:
            results[multiplier], dirty[multiplier] = count_waves(
                env, args.level, args.seed, deck, multiplier, args.waves,
                conveyor_original=args.conveyor_original)
            print(f'mult {multiplier}: {results[multiplier]}', flush=True)
    base = results[1.0]
    print(f"{'wave':>4} {'base':>5} " + ' '.join(f'{m:>7}' for m in MULTIPLIERS))
    for wave in range(args.waves + 1):
        row = [results[m].get(wave, 0) for m in MULTIPLIERS]
        print(f'{wave:>4} {base.get(wave, 0):>5} ' + ' '.join(f'{v:>7}' for v in row))
    # 断言：干净波次都按 C++ std::lround(base * mult) 放大（.5 远离零，不是 Python 的银行家舍入）。
    checked = 0
    skipped = 0
    for multiplier in MULTIPLIERS:
        for wave, count in results[multiplier].items():
            if wave in dirty[multiplier]:
                skipped += 1
                continue
            expected = math.floor(base.get(wave, 0) * multiplier + 0.5)
            assert count == expected, (multiplier, wave, count, expected)
            checked += 1
    print(f'OK: {checked} clean wave observations equal lround(base * multiplier)'
          + (f'; skipped {skipped} waves with a mower kill' if skipped else ''))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(dict(level=args.level, seed=args.seed, deck=list(deck),
                                            conveyor_original=args.conveyor_original,
                                            waves=args.waves,
                                            counts={str(k): v for k, v in results.items()},
                                            dirty={str(k): sorted(v) for k, v in dirty.items()}),
                                       indent=1) + '\n')


if __name__ == '__main__':
    main()
