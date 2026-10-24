#!/usr/bin/env python3
"""Frozen unified-v2 paired student promotion evaluation (task49A).

This always evaluates the preregistered 64 pairs (128 episodes). Statistical
rejection is a successful evaluation with approved=false; incomplete/corrupt data
is an error and cannot produce a promotion certificate.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]
from pvz_promotion import GATE_VERSION, decide, freeze_panel, policy


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.building')
    temporary.write_text(json.dumps(document, indent=2) + '\n')
    temporary.replace(path)


def contract(candidate, incumbent, source_panel, executable, value, resource_dir, workers=1):
    return dict(version=GATE_VERSION, policy=policy(), candidate=str(candidate), incumbent=str(incumbent),
                candidate_sha256=sha(candidate), incumbent_sha256=sha(incumbent),
                source_panel=str(source_panel), source_panel_sha256=sha(source_panel),
                executable=str(executable), executable_sha256=sha(executable),
                value=str(value), value_sha256=sha(value), resource_dir=str(resource_dir), workers=workers,
                runner_sha256=sha(ROOT / 'scripts/search_play.py'),
                policy_module_sha256=sha(ROOT / 'python/pvz_promotion.py'), evaluator_sha256=sha(__file__))


def contract_key(document):
    return hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest()[:20]


def valid_certificate(result, frozen, directory=None):
    """Recompute the decision from complete paired outcomes, not only an approved flag."""
    try:
        if (result['contract'] != frozen or result['contract_key'] != contract_key(frozen)
                or type(result['approved']) is not bool or directory is None):
            return False
        current = contract(Path(frozen['candidate']), Path(frozen['incumbent']), Path(frozen['source_panel']),
                           Path(frozen['executable']), Path(frozen['value']), Path(frozen['resource_dir']), frozen['workers'])
        if current != frozen:
            return False
        directory = Path(directory)
        if not directory.is_absolute():
            return False
        manifest = freeze_panel(read(frozen['source_panel']))
        fresh = decide(result['paired'], manifest)
        if not all(result.get(key) == value for key, value in fresh.items()):
            return False
        actual_pairs, files = [], {}
        args = SimpleNamespace(executable=Path(frozen['executable']), workers=frozen['workers'])
        for task in manifest['tasks']:
            outcomes = {}
            for side in ('candidate', 'incumbent'):
                folder = directory / side / task['task_id']
                outcomes[side] = _task_rows(folder, task, Path(frozen[side]), args)
                files[str((folder / 'games.jsonl').relative_to(directory))] = sha(folder / 'games.jsonl')
            actual_pairs.extend(dict(task_id=task['task_id'], seed=seed, zone=manifest['groups'][task['task_id']],
                                     candidate=outcomes['candidate'][seed]['won'], incumbent=outcomes['incumbent'][seed]['won'])
                                for seed in task['seeds'])
        return actual_pairs == result['paired'] and files == result['episode_files_sha256']
    except (KeyError, ValueError, OSError, TypeError):
        return False


def _task_rows(folder, task, model, args):
    config = read(folder / 'config.json')
    if (config['rollout_policy'] != str(model) or config['task_id'] != task['task_id']
            or config['arms'] != 'fast' or float(config['dagger_beta']) != 0
            or config['executable'] != str(args.executable) or int(config['workers']) != args.workers):
        raise ValueError(f'evaluation config mismatch: {folder}')
    rows = [json.loads(line) for line in (folder / 'games.jsonl').read_text().splitlines()]
    if (len(rows) != len(task['seeds']) or {row['seed'] for row in rows} != set(task['seeds'])
            or any(row['arm'] != 'fast' or row['searched'] != 0 or type(row['won']) is not bool for row in rows)):
        raise ValueError(f'incomplete/non-student/duplicate evaluation games: {folder}')
    return {row['seed']: row for row in rows}


def evaluate(args):
    args.out.mkdir(parents=True, exist_ok=True)
    # A resumed run cannot race the still-live evaluator or its game children.
    with (args.out / 'evaluation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        frozen = contract(args.candidate, args.incumbent, args.source_panel, args.executable,
                          args.value, args.resource_dir, args.workers)
        manifest = freeze_panel(read(args.source_panel))
        manifest_path = args.out / 'manifest.json'
        registration = args.out / 'registration.json'
        if registration.exists():
            if read(registration) != frozen or read(manifest_path) != manifest:
                raise ValueError('cannot resume a different model/build/panel/policy')
        else:
            if list(args.out.glob('*/*/games.jsonl')):
                raise ValueError('unregistered existing evaluation games')
            write(manifest_path, manifest)
            write(registration, frozen)  # before the first episode
        started = time.time()
        paired = []
        files = {}
        for index, task in enumerate(manifest['tasks']):
            outcomes = {}
            # Alternate model order by task; seed pairing never depends on execution order.
            sides = ('incumbent', 'candidate') if index % 2 == 0 else ('candidate', 'incumbent')
            for side in sides:
                model = getattr(args, side)
                folder = args.out / side / task['task_id']
                command = [sys.executable, str(ROOT / 'scripts/search_play.py'),
                           '--rollout-policy', str(model), '--value', str(args.value),
                           '--manifest', str(manifest_path), '--task-id', task['task_id'],
                           '--executable', str(args.executable), '--resource-dir', str(args.resource_dir),
                           '--seeds', str(len(task['seeds'])), '--workers', str(args.workers),
                           '--arms', 'fast', '--dagger-beta', '0', '--out', str(folder)]
                folder.mkdir(parents=True, exist_ok=True)
                command_path = folder / 'command.json'
                if command_path.exists():
                    if read(command_path) != command:
                        raise ValueError(f'command changed: {folder}')
                else:
                    write(command_path, command)
                complete = False
                if (folder / 'games.jsonl').exists():
                    # search_play resumes missing (arm,seed) keys. Corruption is never repaired by
                    # silently dropping rows; the complete-data validator below will reject it.
                    complete = len((folder / 'games.jsonl').read_text().splitlines()) == len(task['seeds'])
                if not complete:
                    with (folder / 'run.log').open('a') as log:
                        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                                pass_fds=(lock.fileno(),),
                                                env={**os.environ, 'CUDA_VISIBLE_DEVICES': '',
                                                     'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})
                    if result.returncode:
                        raise RuntimeError(f'{side} {task["task_id"]}: exit {result.returncode}; see {folder / "run.log"}')
                outcomes[side] = _task_rows(folder, task, model, args)
                if contract(args.candidate, args.incumbent, args.source_panel, args.executable,
                            args.value, args.resource_dir, args.workers) != frozen:
                    raise ValueError('model/build/panel changed during paired evaluation')
                files[str((folder / 'games.jsonl').relative_to(args.out))] = sha(folder / 'games.jsonl')
                print(f'{side} {task["task_id"]}: {sum(r["won"] for r in outcomes[side].values())}/8', flush=True)
            paired.extend(dict(task_id=task['task_id'], seed=seed, zone=manifest['groups'][task['task_id']],
                               candidate=outcomes['candidate'][seed]['won'], incumbent=outcomes['incumbent'][seed]['won'])
                          for seed in task['seeds'])
        result = decide(paired, manifest)
        result.update(contract=frozen, contract_key=contract_key(frozen), paired=paired,
                      episode_files_sha256=files, started=started, finished=time.time())
        if not valid_certificate(result, frozen, args.out):
            raise ValueError('paired certificate failed independent episode validation')
        write(args.out / 'decision.json', result)
        print(json.dumps({key: result[key] for key in ('approved', 'reasons', 'overall', 'zones')}), flush=True)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('candidate', 'incumbent', 'source-panel', 'executable', 'value', 'resource-dir', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    if args.workers not in (1, 2):
        parser.error('--workers must be 1 or 2')
    if any(not getattr(args, name).is_absolute() for name in
           ('candidate', 'incumbent', 'source_panel', 'executable', 'value', 'resource_dir', 'out')):
        parser.error('all paths must be absolute')
    evaluate(args)


if __name__ == '__main__':
    main()
