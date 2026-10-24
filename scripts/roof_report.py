"""Summarize task46 runs with per-loss final-board evidence; never infer missing games."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from pvz_constants import ZOMBIE_NAME


def summarize(manifest, runs, arms, count):
    document = json.loads(manifest.read_text())
    cells = []
    for task in document['tasks']:
        for arm in arms:
            path = runs / f"{task['task_id']}_{arm}_{count}" / 'games.jsonl'
            rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
            expected = set(task['seeds'][:count])
            expected_arm = 'fast_search' if arm == 'teacher' else 'fast'
            if any(row['seed'] not in expected or row['arm'] != expected_arm for row in rows):
                raise ValueError(f'{path}: unexpected seed/arm')
            if len({row['seed'] for row in rows}) != len(rows):
                raise ValueError(f'{path}: duplicate game')
            if any(row.get('terminal') is not True for row in rows):
                raise ValueError(f'{path}: nonterminal game cannot be counted as completed evaluation')
            losses = []
            for row in rows:
                if row['won']:
                    continue
                zombies = sorted(row['final_zombies'], key=lambda z: z['x'])
                # Record the actual public board, not a guessed policy mistake. The leftmost
                # living zombie is evidence of the breach; all others remain in games.jsonl.
                left = zombies[0] if zombies else None
                losses.append(dict(seed=row['seed'], wave=row['final_wave'], tick=row['final_tick'],
                    terminal=row['terminal'], leftmost_zombie=left,
                    zombie_name=ZOMBIE_NAME.get(left['type']) if left else None,
                    interpretation='leftmost living final-board zombie; inspect x for house breach'
                                   if left else 'no living zombie recorded; cause unproven'))
            cells.append(dict(task_id=task['task_id'], level=task['level'], deck=task['deck'], arm=arm,
                multiplier=task['zombie_count_multiplier'], games=len(rows), expected=len(expected),
                complete={row['seed'] for row in rows} == expected, wins=sum(row['won'] for row in rows),
                win_rate=sum(row['won'] for row in rows) / len(rows) if rows else None,
                mean_wave_fraction=sum(row['final_wave'] / row['wave_count'] for row in rows) / len(rows)
                                   if rows else None,
                losses=losses, results_path=str(path)))
    return dict(cells=cells, complete=all(cell['complete'] for cell in cells))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--runs', type=Path, default=ROOT / 'artifacts/task46/runs')
    parser.add_argument('--arms', default='teacher,student')
    parser.add_argument('--count', type=int, default=16)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    summary = summarize(args.manifest, args.runs, args.arms.split(','), args.count)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=1) + '\n')
    for cell in summary['cells']:
        print(f"{cell['task_id']} {cell['arm']}: {cell['wins']}/{cell['games']} "
              f"complete={cell['complete']}")
    print(f"all complete: {summary['complete']}")


if __name__ == '__main__':
    main()
