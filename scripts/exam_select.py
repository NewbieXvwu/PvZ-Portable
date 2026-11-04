#!/usr/bin/env python3
"""Select a stratified exam from paired teacher/incumbent calibration results."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path

from tactical_exam import check_split, training_split


def select(bank, rows, training, *, limit=100, seed_start=242200000):
    outcomes = defaultdict(dict)
    first_seconds = {}
    for row in rows:
        if row['status'] != 'ok':
            continue
        key = (row['item'], row['model'])
        if row['seed'] in outcomes[key]:
            raise ValueError(f'duplicate calibration result: {key} {row["seed"]}')
        outcomes[key][row['seed']] = row['won']
        first_seconds.setdefault(row['item'], row.get('seconds', 0))
    groups, excluded = defaultdict(list), Counter()
    split = training_split(training)
    for item in bank['items']:
        try:
            check_split(dict(items=[item]), training, split=split)
        except ValueError:
            excluded['training_overlap'] += 1
            continue
        teacher = outcomes[(item['id'], 'teacher')]
        student = outcomes[(item['id'], 'incumbent')]
        seeds = set(item['seeds'])
        if len(seeds) < 3 or set(teacher) != seeds or set(student) != seeds:
            excluded['incomplete_calibration'] += 1
            continue
        rates = {name: sum(values.values()) / len(seeds)
                 for name, values in [('teacher', teacher), ('incumbent', student)]}
        if not (0.2 <= rates['teacher'] <= 0.8 or rates['teacher'] != rates['incumbent']):
            excluded['no_observed_discrimination'] += 1
            continue
        entry = copy.deepcopy(item)
        entry['calibration'] = {name: dict(games=len(seeds), wins=sum(values.values()),
                                          rate=rates[name])
                                for name, values in [('teacher', teacher), ('incumbent', student)]}
        entry['calibration']['seeds'] = sorted(seeds)
        entry['calibration']['first_game_seconds'] = first_seconds[item['id']]
        groups[item['skill']].append(entry)
    # Round robin over terrain x trigger, deterministic within each stratum.
    for group in groups.values():
        group.sort(key=lambda item: (item['calibration']['first_game_seconds'], item['id']))
    selected = []
    while groups and len(selected) < limit:
        for skill in sorted(list(groups)):
            if len(selected) == limit:
                break
            entry = groups[skill].pop(0)
            seed = seed_start + len(selected)
            if seed in entry['calibration']['seeds']:
                raise ValueError('exam seed overlaps calibration')
            entry['seeds'] = [seed]
            entry['task']['drill_start'].update(future_seed_start=seed, future_seed_count=1)
            selected.append(entry)
            if not groups[skill]:
                del groups[skill]
    result = dict(schema='pvz-tactical-exam-v1', task='task242',
                  status='ready' if 100 <= len(selected) <= 200 else 'insufficient_calibration',
                  veto=False, items=selected, excluded=dict(excluded),
                  counts=dict(items=len(selected), skills=dict(Counter(i['skill'] for i in selected))),
                  seed_block=[seed_start, seed_start + len(selected)])
    check_split(result, training)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--training-manifest', type=Path, default=Path('configs/task235_drills.json'))
    parser.add_argument('--limit', type=int, default=100)
    parser.add_argument('--seed-start', type=int, default=242200000)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = select(json.loads(args.bank.read_text()),
                    [json.loads(line) for line in args.results.read_text().splitlines()],
                    json.loads(args.training_manifest.read_text()),
                    limit=args.limit, seed_start=args.seed_start)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'items'}))


if __name__ == '__main__':
    main()
