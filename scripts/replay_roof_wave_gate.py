#!/usr/bin/env python3
"""task168: offline v6 decisions from copied decision.json and roof games.jsonl."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from pvz_promotion import decide_scored, policy


def replay(root):
    rounds, duplicates, seen = [], [], {}
    monitor = {task['task_id'] for task in policy(5)['tasks'] if task['monitor']}
    for path in sorted(root.glob('round_*/evaluations/*/decision.json')):
        original = json.loads(path.read_text())
        identity = tuple(original['contract'][side + '_sha256'] for side in ('candidate', 'incumbent'))
        if identity in seen and original['paired'] == seen[identity]:
            duplicates.append(str(path.relative_to(root)))
            continue
        seen[identity] = original['paired']
        pairs = [dict(row) for row in original['paired'] if row['task_id'] not in monitor]
        files = {}
        for name in sorted({row['task_id'] for row in pairs if row['zone'] == 'roof'}):
            rows = [row for row in pairs if row['task_id'] == name]
            for side in ('candidate', 'incumbent'):
                if all(side + '_wave' in row for row in rows):
                    continue
                source = path.parent / side / name / 'games.jsonl'
                payload = source.read_bytes()
                files[str(source.relative_to(path.parent))] = hashlib.sha256(payload).hexdigest()
                games = {game['seed']: game for game in map(json.loads, payload.splitlines())}
                for row in rows:
                    game = games[row['seed']]
                    if game['won'] is not row[side]:
                        raise ValueError(f'episode outcome differs from decision: {source}, seed {row["seed"]}')
                    row[side + '_wave'] = game['final_wave']
        result = decide_scored(pairs, version=6)
        rounds.append(dict(round=int(path.parents[2].name.split('_')[1]),
                           source=str(path.relative_to(root)),
                           decision_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                           candidate_sha256=identity[0], incumbent_sha256=identity[1],
                           roof_episode_sha256=files, old_approved=original['approved'],
                           old_reasons=original['reasons'], new_approved=result['approved'],
                           new_reasons=result['reasons'], overall=result['overall'], zones=result['zones'],
                           roof_pairs={name: [[row['seed'], row['candidate_wave'], row['incumbent_wave'],
                                              row['candidate'], row['incumbent']]
                                             for row in pairs if row['task_id'] == name]
                                       for name in result['zones']['roof']['wave_tasks']}))
    if not rounds:
        raise ValueError(f'no round_*/evaluations/*/decision.json in {root}')
    return dict(schema='task168_roof_wave_replay_v1', policy=policy(6), rounds=rounds,
                duplicates=duplicates)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='local copy of the pipeline round directories')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = replay(args.root)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1) + '\n')
    for row in result['rounds']:
        roof = row['zones']['roof']['wave_tasks']
        waves = row['zones']['roof']['waves']
        print(f'r{row["round"]}: {row["old_approved"]} -> {row["new_approved"]}; ' +
              ', '.join(f'{name}: delta={value["mean_delta_wave"]:+.4f}, '
                        f'p={value["wilcoxon_worse_p"]:.6g}' for name, value in roof.items()) +
              f'; roof p={waves["wilcoxon_worse_p"]:.6g}, veto={waves["veto"]}')


if __name__ == '__main__':
    main()
