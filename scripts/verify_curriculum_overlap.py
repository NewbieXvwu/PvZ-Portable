#!/usr/bin/env python3
"""Audit task49's isolated 2 rounds x 2 groups x 8 full games from saved runtime evidence."""
import argparse
import hashlib
import json
from pathlib import Path


def audit(directory: Path):
    state_path = directory / 'state.json'
    state = json.loads(state_path.read_text())
    assert state['round'] == 2 and len(state['log']) == 2
    groups = []
    seeds = set()
    for entry in state['log']:
        assert len(entry['groups']) == 2
        for group in entry['groups']:
            assert group['games'] == 8 and 1 <= group['workers'] <= 2
            run = directory / 'runs' / f'{group["label"]}_8'
            config = json.loads((run / 'config.json').read_text())
            rows = [json.loads(line) for line in (run / 'games.jsonl').read_text().splitlines()]
            assert len(rows) == 8 and config['rollout_policy'] == group['student']
            assert float(config['dagger_beta']) == 0 and int(config['workers']) <= 2
            for row in rows:
                assert row['seed'] not in seeds, 'duplicate collection seed'
                seeds.add(row['seed'])
                archive = directory / 'frames' / f'seed_{row["seed"]}' / (group['label'] + '_fast_search')
                assert (archive / 'frames.npz').is_file()
            groups.append(dict(round=entry['round'], label=group['label'], games=len(rows),
                               start=group['collection_started'], end=group['collection_finished'],
                               student_round=group['student_round'], student=group['student'],
                               workers=group['workers']))
    jobs = state['training_jobs']
    assert len(jobs) == 2 and all(job['status'] == 'done' for job in jobs)
    phases = []
    for job in jobs:
        folder = Path(job['recipe']).parent
        complete = json.loads((folder / 'complete.json').read_text())
        assert complete == job['timing'] and Path(job['model']).is_file()
        pack_lines = [json.loads(line) for line in (folder / 'pack.log').read_text().splitlines()
                      if line.startswith('{')]
        built = next(row for row in pack_lines if 'built' in row)
        assert built['built'] == 2
        assert built['reused'] == (0 if job['round'] == 1 else 2)
        for phase in ('pack', 'train'):
            start, end = complete[phase + '_started'], complete[phase + '_finished']
            overlaps = [dict(label=group['label'], seconds=max(0, min(end, group['end']) - max(start, group['start'])))
                        for group in groups if group['round'] > job['round']]
            overlaps = [row for row in overlaps if row['seconds'] > 0]
            if job['round'] == 1:
                assert overlaps, f'{phase} did not overlap collection'
            phases.append(dict(round=job['round'], phase=phase, seconds=end - start, overlaps=overlaps,
                               start=start, end=end))
    assert groups[2]['student_round'] == 0  # next round starts immediately with the old student
    assert groups[3]['student_round'] == 1  # publish only at next group boundary
    idle = groups[2]['start'] - groups[1]['end']
    assert 0 <= idle < 5, f'unexpected round gap: {idle}s'
    return dict(passed=True, games=len(seeds), rounds=2, groups=groups, phases=phases,
                round_gap_seconds=idle, incremental_round_2=dict(built=2, reused=2),
                state_sha256=hashlib.sha256(state_path.read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not args.run.is_absolute() or not args.out.is_absolute():
        parser.error('use absolute paths')
    evidence = audit(args.run)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps(evidence), flush=True)


if __name__ == '__main__':
    main()
