"""task28 第 3 步：把 n10p 的输赢与“换了哪两张卡”对上，找出结构性打不过的替换。"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]
from pvz_constants import PLANT_NAME
from pvz_tasks import deck_for_seed
from render_episode import load_training_task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--run', nargs='+', required=True, help='<task_id>=<dir>')
    parser.add_argument('--markdown', action='store_true', help='按老师逐局表输出 markdown（含按类型种植数）')
    args = parser.parse_args()
    for item in args.run:
        task_id, _, path = item.partition('=')
        task = load_training_task(args.manifest, task_id)
        base = set(task['deck'])
        rows = []
        for line in (Path(path) / 'games.jsonl').read_text().splitlines():
            if not line.strip():
                continue
            game = json.loads(line)
            deck = set(deck_for_seed(task, game['seed']))
            by_type = Counter()
            for slot, count in (game.get('planted') or {}).items():
                by_type[deck_for_seed(task, game['seed'])[int(slot)]] += count
            rows.append(dict(seed=game['seed'], won=game['won'], wave=game['final_wave'],
                             added=sorted(deck - base), removed=sorted(base - deck),
                             mowers=game.get('mowers_used'),
                             by_type=by_type,
                             plants=sum(by_type.values())))
        if args.markdown:
            print(f'#### {task_id}')
            print()
            print('| 种子 | 结果 | 波 | 丢车 | 种植 | 换入/换出 | 按类型 |')
            print('|---|---|---:|---:|---:|---|---|')
            for row in sorted(rows, key=lambda r: r['seed']):
                types = ' '.join(f'{PLANT_NAME.get(t, t)}{n}' for t, n in row['by_type'].most_common())
                print(f"| {row['seed']} | {'胜' if row['won'] else '负'} | {row['wave']} | {row['mowers']} | "
                      f"{row['plants']} | +{'/'.join(map(str, row['added']))} "
                      f"-{'/'.join(map(str, row['removed']))} | {types} |")
            print()
            continue
        losses = [r for r in rows if not r['won']]
        wins = [r for r in rows if r['won']]
        print(f'== {task_id}: {len(wins)}/{len(rows)} 胜')
        for row in sorted(rows, key=lambda r: (r['won'], r['wave'])):
            print(f"  {'W' if row['won'] else 'L'} seed={row['seed']} wave={row['wave']:>2} "
                  f"plants={row['plants']:>3} +{row['added']} -{row['removed']}")
        added_loss = Counter(p for r in losses for p in r['added'])
        added_win = Counter(p for r in wins for p in r['added'])
        print('  added in losses:', dict(sorted(added_loss.items())))
        print('  added in wins:  ', dict(sorted(added_win.items())))
        early = [r for r in losses if (r['wave'] or 0) < 20]
        print(f'  losses before wave 20: {len(early)}', [(r['seed'], r['wave'], r['added'], r['plants']) for r in early])


if __name__ == '__main__':
    main()
