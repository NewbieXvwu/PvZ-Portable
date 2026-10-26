#!/usr/bin/env python3
"""从 qc 完成行与采集目录回填轮日志，默认只打印计划。

主控确认计划、将 qc 停在组边界后，才可加 --write。已结束轮补进 log，
当前轮补进 completed_groups，交给控制器轮末归档。不更新窗口或种子使用记录。
应在采集机器的本地时区运行。
"""
import argparse
import json
import re
from datetime import datetime
from pathlib import Path


DONE = re.compile(
    r'^(?P<stamp>\S+) round (?P<round>\d+) (?P<label>\S+) (?P<task_id>\S+) '
    r'mult=(?P<multiplier>\S+)(?: start=(?P<start>\d+))? '
    r'offset=(?P<offset>\d+) games=(?P<games>\d+) wins=\d+ '
    r'rate=\S+ beta=(?P<beta>\S+) rc=(?P<rc>-?\d+) (?P<seconds>\d+)s$')
START = re.compile(r'^(?P<stamp>\S+) round (?P<round>\d+) (?P<label>\S+): workers=')


def reconstruct(done, started, folder, student_rounds):
    config = json.loads((folder / 'config.json').read_text())
    games = int(done['games'])
    with (folder / 'games.jsonl').open() as handle:
        count = sum(1 for line in handle if line.strip())
    if games <= 0 or count != games or int(config['seeds']) != games:
        raise ValueError(f'games.jsonl/config/completion counts differ: {count}/{config["seeds"]}/{games}')
    if (config['task_id'] != done['task_id'] or config['frames_label'] != done['label']
            or float(config['dagger_beta']) != float(done['beta'])):
        raise ValueError('config does not match completion task/label/beta')
    demo = done['start'] is not None
    if int(config['seed_offset']) != (0 if demo else int(done['offset'])):
        raise ValueError('config does not match completion offset')
    multiplier = float(done['multiplier'])
    task_id = done['task_id']
    suffix = re.search(r'_m(\d+)$', task_id)
    base = task_id[:suffix.start()] if not demo and suffix and int(suffix[1]) / 100 == multiplier else task_id
    student = config['rollout_policy']
    model_round = re.search(r'/round_(\d+)/student/model\.pt$', student)
    student_round = int(model_round[1]) if model_round else student_rounds.get(student, 0)
    finished = datetime.fromisoformat(done['stamp']).timestamp()
    group = dict(base=base, multiplier=multiplier, task_id=task_id, offset=int(done['offset']),
                 games=games, manifest=config['manifest'], label=done['label'],
                 collection_started=(datetime.fromisoformat(started).timestamp() if started
                                     else finished - int(done['seconds'])),
                 collection_finished=finished, student=student, student_round=student_round,
                 workers=int(config['workers']), beta=float(done['beta']))
    if demo:
        if config.get('demo_archive') in (None, 'None') or int(config['demo_decision']) != int(done['start']):
            raise ValueError('config does not match demonstration start')
        seeds = [int(seed) for seed in config['future_seeds'].split(',')]
        if len(seeds) != games:
            raise ValueError('future seed count does not match games')
        group.update(demo_archive=config['demo_archive'], demo_decision=int(done['start']),
                     future_seeds=seeds)
    return group


def backfill(state, qc_out, runs):
    rounds = {entry['round']: entry for entry in state['log']}
    known = {group['label'] for entry in state['log'] for group in entry['groups']}
    known.update(group['label'] for group in state.get('completed_groups', []))
    student_rounds = {group['student']: group.get('student_round', 0)
                      for entry in state['log'] for group in entry['groups'] if 'student' in group}
    if state.get('student'):
        student_rounds[state['student']] = state.get('student_round', 0)
    starts, completions = {}, {}
    with Path(qc_out).open() as handle:
        for line in handle:
            start = START.match(line)
            if start:
                starts[(int(start['round']), start['label'])] = start['stamp']
            done = DONE.match(line.strip())
            if done:
                key = (int(done['round']), done['label'])
                completions[key] = (done.groupdict(), starts.get(key))
    added, skipped = [], []
    for (round_index, label), (done, started) in completions.items():
        if label in known or int(done['rc']) != 0:
            continue
        if round_index not in rounds and round_index != state['round'] + 1:
            skipped.append(dict(round=round_index, label=label, reason='round is not in saved log'))
            continue
        folder = Path(runs) / f'{label}_{done["games"]}'
        try:
            group = reconstruct(done, started, folder, student_rounds)
        except (OSError, ValueError, KeyError) as error:
            skipped.append(dict(round=round_index, label=label, reason=str(error)))
            continue
        target = (rounds[round_index]['groups'] if round_index in rounds
                  else state.setdefault('completed_groups', []))
        target.append(group)
        known.add(label)
        added.append(dict(round=round_index, **group))
    for round_index in {group['round'] for group in added}:
        target = (rounds[round_index]['groups'] if round_index in rounds
                  else state['completed_groups'])
        target.sort(key=lambda group: group['collection_finished'])
    return added, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--qc-out', type=Path, required=True)
    parser.add_argument('--runs', type=Path, required=True, help='artifacts/hard 目录')
    parser.add_argument('--write', action='store_true', help='主控确认且 qc 已停在组边界后写入')
    args = parser.parse_args()
    original = args.state.read_bytes()
    state = json.loads(original)
    added, skipped = backfill(state, args.qc_out, args.runs)
    labels = {group.get('frame_label', group['label'] + '_fast_search')
              for entry in state['log'] for group in entry['groups']}
    print(json.dumps(dict(dry_run=not args.write, added=added, skipped=skipped,
                          added_groups=len(added), added_games=sum(group['games'] for group in added),
                          added_log_groups=sum(group['round'] <= state['round'] for group in added),
                          added_pending_groups=sum(group['round'] > state['round'] for group in added),
                          log_groups=sum(len(entry['groups']) for entry in state['log']),
                          recipe_log_labels=len(labels)), indent=2, ensure_ascii=False))
    if args.write and added:
        if args.state.read_bytes() != original:
            raise SystemExit('state changed during backfill; stop qc at a group boundary and retry')
        temporary = args.state.with_suffix(args.state.suffix + '.building')
        temporary.write_text(json.dumps(state, indent=1) + '\n')
        temporary.replace(args.state)


if __name__ == '__main__':
    main()
