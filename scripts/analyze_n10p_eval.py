"""task28 第 3 步：把 n10p 的每局结果摊开（卡组、胜负、按类型种植数）。

`search_play` 的 `planted` 键是**卡槽号**；n10p 是固定卡组（只有 `deck_perturb` 决定初始牌），
所以 `slot -> deck[slot]` 就是植物类型。
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]
from pvz_tasks import deck_for_seed
from render_episode import load_training_task


def summarize(run_dir: Path, manifest: Path, task_id: str):
    task = load_training_task(manifest, task_id)
    rows = []
    for line in (run_dir / 'games.jsonl').read_text().splitlines():
        if not line.strip():
            continue
        game = json.loads(line)
        deck = list(deck_for_seed(task, game['seed']))
        by_type = {}
        for slot, count in (game.get('planted') or {}).items():
            slot = int(slot)
            kind = deck[slot] if 0 <= slot < len(deck) else -1
            by_type[kind] = by_type.get(kind, 0) + count
        rows.append(dict(seed=game['seed'], won=game['won'], final_wave=game['final_wave'],
                         mowers_used=game['mowers_used'], planted=sum(by_type.values()),
                         planted_by_type=dict(sorted(by_type.items())), deck=deck,
                         searched=game.get('searched'), overrides=game.get('overrides')))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--run', nargs='+', required=True, help='<task_id>=<dir> pairs')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    result = {}
    for item in args.run:
        task_id, _, path = item.partition('=')
        rows = summarize(Path(path), args.manifest, task_id)
        wins = sum(r['won'] for r in rows)
        print(f'== {task_id}  {path}  {wins}/{len(rows)} 胜')
        for row in rows:
            kinds = ' '.join(f'{k}:{v}' for k, v in row['planted_by_type'].items())
            print(f"  seed={row['seed']} won={int(row['won'])} wave={row['final_wave']:>2} "
                  f"mowers={row['mowers_used']} plants={row['planted']:>2} [{kinds}] deck={row['deck']}")
        result[task_id] = dict(games=len(rows), wins=wins,
                               mean_final_wave=round(sum(r['final_wave'] or 0 for r in rows) / max(1, len(rows)), 2),
                               mean_plants=round(sum(r['planted'] for r in rows) / max(1, len(rows)), 1),
                               rows=rows)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1) + '\n')


if __name__ == '__main__':
    main()
