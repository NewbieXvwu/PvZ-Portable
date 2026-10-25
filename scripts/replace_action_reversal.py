"""Descriptive reversal and seed-cost breakdown of the task69 replacement events.

Reads only the frozen manifest, the raw after-arm games.jsonl batches and the
engine's own seed table; never launches games.  This is a post-hoc explanation
of why the preregistered A/B failed, not a new test: it does not touch the
preregistered win-rate comparison, its intervals or the Holm correction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_seed_tables(root):
    source = root / 'src/Lawn/Plant.cpp'
    text = source.read_text(encoding='utf-8', errors='replace')
    costs = {}
    pattern = (r'\.mSeedType = SeedType::SEED_(\w+).*?\.mSeedCost = (\d+)'
               r'.*?\.mPlantName = "(\w+)"')
    for match in re.finditer(pattern, text):
        costs[match.group(1)] = dict(cost=int(match.group(2)), name=match.group(3))
    enums = (root / 'src/ConstEnums.h').read_text(encoding='utf-8', errors='replace')
    start = enums.index('enum SeedType')
    ids = {}
    for match in re.finditer(r'\bSEED_(\w+) = (\d+)', enums[start:start + 4000]):
        ids[int(match.group(2))] = match.group(1)
    return costs, ids, source


def summarize(rows, costs, ids):
    events = reversals = free = 0
    upgrade = equal = downgrade = unknown = 0
    cells = 0
    repeated_cells = 0
    games_with_reversal = 0
    for row in rows:
        game_events = sorted(row.get('replace_events') or [], key=lambda e: e['tick'])
        events += len(game_events)
        by_cell = defaultdict(list)
        for event in game_events:
            by_cell[(event['row'], event['col'])].append(event)
            if event.get('sun_spent') == 0:
                free += 1
            added = costs.get(ids.get(event['added'][0], ''), {}).get('cost')
            removed = costs.get(ids.get(event['removed'][0], ''), {}).get('cost')
            if added is None or removed is None:
                unknown += 1
            elif added > removed:
                upgrade += 1
            elif added == removed:
                equal += 1
            else:
                downgrade += 1
        game_reversals = 0
        for cell_events in by_cell.values():
            cells += 1
            if len(cell_events) >= 2:
                repeated_cells += 1
            for previous, current in zip(cell_events, cell_events[1:]):
                if set(current['added']) == set(previous['removed']):
                    game_reversals += 1
        reversals += game_reversals
        games_with_reversal += int(game_reversals > 0)
    return dict(games=len(rows), events=events, free_events=free,
                free_share=free / events if events else None,
                same_cell_reversals=reversals,
                same_cell_reversal_share=reversals / events if events else None,
                games_with_any_reversal=games_with_reversal,
                cells_replaced=cells, cells_replaced_twice_or_more=repeated_cells,
                seed_cost_delta=dict(upgrade=upgrade, equal=equal,
                                     downgrade=downgrade, unknown=unknown))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('raw', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--root', type=Path, default=Path('.'))
    args = parser.parse_args()
    costs, ids, source = load_seed_tables(args.root)
    manifest = json.loads(args.manifest.read_text())
    tasks = {}
    pooled_rows = []
    for task in manifest['tasks']:
        path = args.raw / task['task_id'] / 'after' / 'games.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        tasks[task['task_id']] = summarize(rows, costs, ids)
        pooled_rows.extend(rows)
    result = dict(
        manifest_sha256=sha(args.manifest),
        scope='after arm only; post-hoc description of the observed behaviour, '
              'does not alter the preregistered win-rate test or its intervals',
        reversal_rule='consecutive replacements on one cell where the later plant '
                      'is exactly the type the earlier replacement removed',
        cost_source=dict(path=str(source), sha256=sha(source)),
        cost_proxy_note='engine seed cost is used only as a descriptive proxy, not as '
                        'a strength or strategy rule; zero-cost plants make a free '
                        'replacement legal',
        tasks=tasks,
        pooled=summarize(pooled_rows, costs, ids),
    )
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result['tasks'], ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
