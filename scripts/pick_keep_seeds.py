#!/usr/bin/env python3
"""task31：在某个扰动族任务里挑出“某张底卡会被换掉”的种子（分层补样本用）。

判据是**条件概率** P(老师赢 | 这张底卡被换掉)。条件采样（只在被换掉的种子里抽）
对这个条件概率是无偏的，而且比随机抽种子省很多局：随机抽时一张卡被换掉的概率
只有 8%–20%，条件采样是 100%。

用法：
  python3 scripts/pick_keep_seeds.py \
      --manifest /home/newbiexvwu/PvZ-search/artifacts/stage4/night_panel_n10p.json \
      --task-id night_l12_train_n10p --card 8 --count 8 \
      --played /home/newbiexvwu/PvZ-search/.claude/worktrees/n10p/artifacts/task28/teacher_l12_16 \
      --out artifacts/task31/seeds_l12_c8.json

挑法：符合条件的种子按数值排序后**等距**取 count 个（确定性，不用随机数），
跳过 --played 与 --exclude 里已经跑过的种子。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]

from pvz_tasks import deck_for_seed  # noqa: E402
from render_episode import load_training_task  # noqa: E402


def played_seeds(folders) -> set[int]:
    seeds = set()
    for folder in folders:
        path = Path(folder) / 'games.jsonl'
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if line.strip():
                seeds.add(int(json.loads(line)['seed']))
    return seeds


def evenly_spaced(items: list[int], count: int) -> list[int]:
    if count >= len(items):
        return list(items)
    if count == 1:
        return [items[len(items) // 2]]
    step = (len(items) - 1) / (count - 1)
    return sorted({items[round(i * step)] for i in range(count)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--task-id', required=True)
    parser.add_argument('--card', type=int, required=True, help='底卡 id（植物 id）')
    parser.add_argument('--count', type=int, default=8)
    parser.add_argument('--played', action='append', default=[], metavar='DIR',
                        help='已经跑过的 run 目录（games.jsonl），其中的种子会被跳过')
    parser.add_argument('--exclude', action='append', default=[], metavar='DIR', help='同 --played')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()

    task = load_training_task(args.manifest, args.task_id)
    base = tuple(task['deck'])
    if args.card not in base:
        raise SystemExit(f'card {args.card} is not in the base deck {base}')
    if 'deck_perturb' not in task:
        raise SystemExit('task has no deck_perturb: every seed uses the same deck')
    if args.card in set(task['deck_perturb'].get('keep_plants', [])):
        raise SystemExit(f'card {args.card} is already a keep card; it is never removed')

    taken = played_seeds(args.played + args.exclude)
    eligible, available = [], []
    for seed in task['seeds']:
        if args.card not in deck_for_seed(task, int(seed)):
            eligible.append(int(seed))
            if int(seed) not in taken:
                available.append(int(seed))
    chosen = evenly_spaced(available, args.count)
    print(f'{args.task_id} L{task["level"]} card {args.card}: '
          f'{len(eligible)}/{len(task["seeds"])} seeds drop it, {len(available)} unused, picked {len(chosen)}')
    print(','.join(str(seed) for seed in chosen))
    if len(chosen) < args.count:
        print(f'WARNING: only {len(chosen)} of {args.count} requested seeds available', file=sys.stderr)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(dict(task_id=args.task_id, level=int(task['level']), card=args.card,
                                            base=list(base), seeds=chosen, eligible=len(eligible),
                                            available=len(available), total=len(task['seeds'])), indent=1) + '\n')


if __name__ == '__main__':
    main()
