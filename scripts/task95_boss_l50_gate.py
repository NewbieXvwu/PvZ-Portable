#!/usr/bin/env python3
"""Compare four boss_l50_orig_eval games with the unchanged A2 student policy.

Only the candidate must expose Boss fields. Both engines use the same production-shaped
command and manifest; actions and logical terminal outcomes must match exactly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
MAIN = Path('/home/newbiexvwu/PvZ-search')
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]

from curriculum_queue import command_for
from pvz_frame_store import read_frames
from task70_redeploy_gate import COMPARE_FIELDS, action_stream, frames_for

MANIFEST = ROOT / 'artifacts/task85/boss_panel.json'
TASK_ID = 'boss_l50_orig_eval'
BOSS_KEYS = {'active', 'stage', 'head', 'action', 'target_row', 'target_col', 'projectile'}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_game(old: dict, new: dict, old_frames: list[dict], new_frames: list[dict]) -> dict:
    if not old_frames or not new_frames:
        raise ValueError('requires complete nonempty action archives')
    differences = {field: {'old': old.get(field), 'new': new.get(field)}
                   for field in COMPARE_FIELDS if old.get(field) != new.get(field)}
    old_actions, new_actions = action_stream(old_frames), action_stream(new_frames)
    blocks = [frame['boss'] for frame in new_frames if isinstance(frame.get('boss'), dict)]
    reset = new_frames[0].get('boss') or {}
    return {
        'seed': old['seed'],
        'old': {field: old.get(field) for field in COMPARE_FIELDS},
        'new': {field: new.get(field) for field in COMPARE_FIELDS},
        'logical_differences': differences,
        'action_stream_match': old_actions == new_actions,
        'action_count': {'old': len(old_actions), 'new': len(new_actions)},
        'old_boss_block_present': any('boss' in frame for frame in old_frames),
        'boss_reset': reset,
        'new_boss_fields_present': bool(blocks) and bool(reset.get('active'))
            and all(BOSS_KEYS <= set(block) for block in blocks),
        'boss_frames': len(blocks),
        'boss_stages': sorted({block['stage'] for block in blocks if 'stage' in block}),
        'boss_actions': sorted({block['action'] for block in blocks if 'action' in block}),
        'all_terminal': bool(old.get('terminal')) and bool(new.get('terminal')),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old', type=Path, required=True)
    parser.add_argument('--new', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=ROOT / 'artifacts/task95/boss_l50.json')
    args = parser.parse_args()
    task = next(t for t in json.loads(MANIFEST.read_text())['tasks'] if t['task_id'] == TASK_ID)
    seeds = task['seeds'][:4]
    assert len(seeds) == 4 and task['level'] == 50 and task['conveyor_start'] == 'original'
    args.out.parent.mkdir(parents=True, exist_ok=True)
    models = {
        'rollout_policy': MAIN / 'artifacts/student/dw4i_s1/model.pt',
        'value': MAIN / 'artifacts/value_net/v4/model.pt',
        'static_value': MAIN / 'artifacts/hard/value_branch1/model.pt',
    }
    for path in models.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    model_hashes = {name: sha(path) for name, path in models.items()}
    common = SimpleNamespace(python=Path(sys.executable), workers=2, **models,
                             resource_dir=Path('/home/newbiexvwu/.cache/pvz-research-resources'))
    records, archives, commands = {}, {}, {}
    for tag, executable in (('old', args.old), ('new', args.new)):
        common.executable = executable.resolve(strict=True)
        label = 'boss_l50_' + tag
        frame_dir = args.out.parent / ('boss_l50_frames_' + tag)
        command = command_for(common, MANIFEST, TASK_ID, 0, 4, label, frames=frame_dir,
                              out=args.out.parent / ('boss_l50_runs_' + tag), beta=0.0)
        metadata = dict(command=command, executable_sha256=sha(common.executable),
                        models=model_hashes, manifest_sha256=sha(MANIFEST))
        metadata_path = args.out.parent / ('boss_l50_command_' + tag + '.json')
        if metadata_path.exists() and json.loads(metadata_path.read_text()) != metadata:
            raise RuntimeError('existing Boss run belongs to different inputs')
        metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
        result_path = Path(command[command.index('--out') + 1]) / 'games.jsonl'
        have = [json.loads(line) for line in result_path.read_text().splitlines()] if result_path.exists() else []
        have_seeds = [row['seed'] for row in have]
        if len(have_seeds) != len(set(have_seeds)) or set(have_seeds) - set(seeds):
            raise RuntimeError('existing Boss run has duplicate or unexpected seeds')
        if set(have_seeds) != set(seeds):
            env = {**os.environ, 'PVZ_NO_POPUP': '1', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
            with (args.out.parent / ('boss_l50_' + tag + '.txt')).open('a') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, check=True)
        rows = [json.loads(line) for line in result_path.read_text().splitlines()]
        if len(rows) != 4 or sorted(row['seed'] for row in rows) != sorted(seeds):
            raise RuntimeError('incomplete Boss evaluation')
        records[tag] = {row['seed']: row for row in rows}
        archives[tag] = {seed: frames_for(frame_dir, label, seed) for seed in seeds}
        commands[tag] = metadata
    games = [compare_game(records['old'][seed], records['new'][seed],
                          archives['old'][seed], archives['new'][seed]) for seed in seeds]
    checks = {
        'new_boss_fields_present': all(game['new_boss_fields_present'] for game in games),
        'logical_fields_match': all(not game['logical_differences'] for game in games),
        'action_streams_match': all(game['action_stream_match'] for game in games),
        'all_terminal': all(game['all_terminal'] for game in games),
    }
    evidence = dict(task_id=TASK_ID, seeds=seeds, policy='unchanged A2 dw4i_s1, beta=0',
                    commands=commands, games=games, checks=checks, **{'pass': all(checks.values())})
    args.out.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + '\n')
    print('BOSS L50', 'PASS' if evidence['pass'] else 'FAIL', checks, flush=True)
    return 0 if evidence['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
