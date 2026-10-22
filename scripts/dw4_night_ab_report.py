"""task23 step 6.1 report: n9 vs n10 per-plant-type comparison from a panel run.

games.jsonl 'planted' is keyed by SEED-BANK SLOT, not by plant type (src/main.cpp indexes
'std::array<int,10> planted' with action.packet, and search_play does the same for the Python arm).
Comparing two decks by those keys therefore compares different cards under the same number. This script
translates every slot through the task deck before aggregating, and prints the paired table plus the
within-n10 win rate by grave-buster count.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def bucket(count):
    if count == 0:
        return '0'
    if count <= 4:
        return '1-4'
    if count <= 9:
        return '5-9'
    return '10+'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--panel', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    manifest = json.loads(args.manifest.read_text())
    tasks = {task['task_id']: task for task in manifest['tasks']}
    result = {}
    for name, task in tasks.items():
        deck = task['deck']
        rows = [json.loads(line) for line in (args.panel/name/'games.jsonl').read_text().splitlines()]
        totals = Counter(); per_game = {kind: [] for kind in set(deck)}
        for row in rows:
            assert all(int(slot) < len(deck) for slot in row['planted']), (name, row['planted'], deck)
            counts = Counter()
            for slot, count in row['planted'].items():
                counts[deck[int(slot)]] += count
            for kind in set(deck):
                per_game[kind].append(counts[kind])
        result[name] = dict(
            deck=deck, games=len(rows), won=sum(row['won'] for row in rows),
            win_rate=sum(row['won'] for row in rows)/len(rows),
            mean_final_wave=sum(row['final_wave'] for row in rows)/len(rows),
            mean_mowers=sum(row['mowers_used'] for row in rows)/len(rows),
            planted_per_game={str(kind): sum(values)/len(values) for kind, values in sorted(per_game.items())},
            planted_total={str(kind): sum(values) for kind, values in sorted(per_game.items())},
        )
    pairs = {}
    for name in tasks:
        if not name.endswith('_n10'):
            continue
        level = name[:-len('_n10')]
        left, right = result[level + '_n9'], result[name]
        pairs[level] = dict(
            n9_won=left['won'], n10_won=right['won'], games=left['games'],
            delta_pp=100*(right['win_rate']-left['win_rate']),
            sunshroom_n9=left['planted_per_game'].get('9', 0.0),
            sunshroom_n10=right['planted_per_game'].get('9', 0.0),
            gravebuster_n9=left['planted_per_game'].get('11', 0.0),
            gravebuster_n10=right['planted_per_game'].get('11', 0.0),
        )
        buckets = Counter(); wins = Counter()
        for row in (args.panel/name/'games.jsonl').read_text().splitlines():
            row = json.loads(row)
            count = sum(count for slot, count in row['planted'].items() if tasks[name]['deck'][int(slot)] == 11)
            key = bucket(count); buckets[key] += 1; wins[key] += row['won']
        pairs[level]['n10_win_rate_by_gravebuster'] = {
            key: dict(games=buckets[key], won=wins[key], win_rate=wins[key]/buckets[key]) for key in sorted(buckets)}
    report = dict(manifest=str(args.manifest), panel=str(args.panel), tasks=result, pairs=pairs,
                  note='per-game counts are per plant TYPE after mapping the deck; games.jsonl keys are slots')
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'pairs': pairs, 'tasks': {k: {kk: vv for kk, vv in v.items() if kk != 'deck'} for k, v in result.items()}}, indent=2))


if __name__ == '__main__':
    main()
