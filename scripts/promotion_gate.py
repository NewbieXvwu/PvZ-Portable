#!/usr/bin/env python3
"""Paired student promotion evaluation.

Gate version 1 (task49A): frozen unified-v2 paired evaluation on the registered
64 pairs (128 episodes). Statistical rejection is a successful evaluation with
approved=false; incomplete/corrupt data is an error and cannot produce a
promotion certificate.

Gate version 2 (task73b): fresh seeds every round, allocated in order from the
gate-dedicated pool [31024, 65536) via an out-of-git ledger
(~/pvz-logs/gate_seed_ledger.json); 1536 pairs; the incumbent is re-evaluated on
the same fresh seeds (cached incumbent outcomes are never used). Pool exhaustion
aborts the gate instead of wrapping around to reuse seeds.
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
import pvz_promotion
from pvz_promotion import GATE_VERSION, SEED_LEDGER_DEFAULT, decide, freeze_panel, policy


class SeedPoolExhausted(RuntimeError):
    """The gate seed pool has no contiguous block left; never wrap around and reuse."""


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.building')
    temporary.write_text(json.dumps(document, indent=2) + '\n')
    temporary.replace(path)


def contract(candidate, incumbent, source_panel, executable, value, resource_dir, workers=1,
             version=GATE_VERSION, seed_block=None):
    document = dict(version=version, policy=policy(version), candidate=str(candidate), incumbent=str(incumbent),
                    candidate_sha256=sha(candidate), incumbent_sha256=sha(incumbent),
                    source_panel=str(source_panel), source_panel_sha256=sha(source_panel),
                    executable=str(executable), executable_sha256=sha(executable),
                    value=str(value), value_sha256=sha(value), resource_dir=str(resource_dir), workers=workers,
                    runner_sha256=sha(ROOT / 'scripts/search_play.py'),
                    policy_module_sha256=sha(ROOT / 'python/pvz_promotion.py'), evaluator_sha256=sha(__file__))
    if version >= 2:
        document['seed_block'] = [int(seed_block[0]), int(seed_block[1])] if seed_block else None
    return document


def allocate_seed_block(ledger_path, total, pool=None):
    """Reserve the next unused contiguous block from the gate pool (task73b).

    The ledger lives outside git and is the sole authority for gate seed
    consumption: blocks are handed out in pool order, never reused, never
    wrapped. Exhaustion raises SeedPoolExhausted.
    """
    pool = pvz_promotion.SEED_POOL_V2 if pool is None else pool
    ledger_path, lock_path = Path(ledger_path), Path(str(ledger_path) + '.lock')
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if ledger_path.exists():
            document = read(ledger_path)
            if tuple(document.get('pool', ())) != tuple(pool):
                raise ValueError(f'seed ledger {ledger_path} was created for pool {document.get("pool")}')
            blocks = document.get('allocated')
            if (type(blocks) is not list
                    or any(type(b) is not list or len(b) != 2 or any(type(s) is not int for s in b)
                           or b[0] < pool[0] or b[1] > pool[1] or b[0] >= b[1] for b in blocks)):
                raise ValueError(f'corrupt seed ledger: {ledger_path}')
            ordered = sorted(tuple(b) for b in blocks)
            if any(a[1] > b[0] for a, b in zip(ordered, ordered[1:])):
                raise ValueError(f'overlapping blocks in seed ledger: {ledger_path}')
        else:
            blocks, ordered = [], []
        cursor = pool[0]
        for low, high in ordered:
            if cursor + total <= low:
                break
            cursor = max(cursor, high)
        if cursor + total > pool[1]:
            raise SeedPoolExhausted(
                f'gate seed pool {list(pool)} exhausted: need {total} more seeds after '
                f'{len(blocks)} allocated blocks; refusing to reuse seeds')
        blocks.append([cursor, cursor + total])
        write(ledger_path, dict(pool=list(pool), allocated=blocks))
        return cursor, cursor + total


def require_allocated_block(ledger_path, seed_block):
    """The ledger is the sole seed authority: a caller-supplied block must be recorded."""
    document = read(Path(ledger_path))
    if [int(seed_block[0]), int(seed_block[1])] not in document.get('allocated', []):
        raise ValueError(f'seed block {[int(seed_block[0]), int(seed_block[1])]} is not allocated '
                         f'in the ledger {ledger_path}; allocate it via allocate_seed_block first')


def contract_key(document):
    return hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest()[:20]


def valid_certificate(result, frozen, directory=None):
    """Recompute the decision from complete paired outcomes, not only an approved flag."""
    try:
        if (result['contract'] != frozen or result['contract_key'] != contract_key(frozen)
                or type(result['approved']) is not bool or directory is None):
            return False
        # Version-aware rebuild: a certificate stays verifiable under the gate
        # version that issued it, even after GATE_VERSION moves on.
        current = contract(Path(frozen['candidate']), Path(frozen['incumbent']), Path(frozen['source_panel']),
                           Path(frozen['executable']), Path(frozen['value']), Path(frozen['resource_dir']),
                           frozen['workers'], version=frozen.get('version', GATE_VERSION),
                           seed_block=frozen.get('seed_block'))
        if current != frozen:
            return False
        directory = Path(directory)
        if not directory.is_absolute():
            return False
        manifest = freeze_panel(read(frozen['source_panel']), version=frozen.get('version', GATE_VERSION),
                                seed_block=frozen.get('seed_block'))
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
        version = getattr(args, 'gate_version', GATE_VERSION)
        registration = args.out / 'registration.json'
        # A resumed v2 run must reuse the block recorded at first registration,
        # otherwise resuming would silently consume another ledger block.
        existing = read(registration) if registration.exists() else None
        if version >= 2:
            if existing is not None:
                seed_block = tuple(existing.get('seed_block') or ())
                if len(seed_block) != 2:
                    raise ValueError('registered v2 evaluation has no usable seed block')
            elif getattr(args, 'seed_block', None):
                seed_block = tuple(int(part) for part in str(args.seed_block).split(','))
                require_allocated_block(args.seed_ledger, seed_block)
            else:
                seed_block = allocate_seed_block(args.seed_ledger, policy(version)['pair_count'])
                print(f'seed ledger: allocated [{seed_block[0]}, {seed_block[1]}) from {args.seed_ledger}', flush=True)
        else:
            seed_block = None
        frozen = contract(args.candidate, args.incumbent, args.source_panel, args.executable,
                          args.value, args.resource_dir, args.workers, version=version,
                          seed_block=seed_block)
        manifest = freeze_panel(read(args.source_panel), version=version, seed_block=seed_block)
        manifest_path = args.out / 'manifest.json'
        if existing is not None:
            if existing != frozen or read(manifest_path) != manifest:
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
                            args.value, args.resource_dir, args.workers, version=version,
                            seed_block=seed_block) != frozen:
                    raise ValueError('model/build/panel changed during paired evaluation')
                files[str((folder / 'games.jsonl').relative_to(args.out))] = sha(folder / 'games.jsonl')
                print(f'{side} {task["task_id"]}: {sum(r["won"] for r in outcomes[side].values())}/{len(task["seeds"])}', flush=True)
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
    parser.add_argument('--gate-version', type=int, choices=(1, 2), default=GATE_VERSION,
                        help='1 = task49A fixed 64-pair panel; 2 = task73b fresh-seed 1536-pair panel')
    parser.add_argument('--seed-ledger', type=Path, default=Path(SEED_LEDGER_DEFAULT),
                        help='out-of-git ledger of allocated gate seed blocks (v2 only)')
    parser.add_argument('--seed-block', default=None,
                        help='START,END already allocated in the ledger (v2 pipeline flow); '
                             'omit to allocate the next free block')
    args = parser.parse_args()
    if args.workers not in (1, 2):
        parser.error('--workers must be 1 or 2')
    absolute = ('candidate', 'incumbent', 'source_panel', 'executable', 'value', 'resource_dir')
    if args.gate_version >= 2:
        absolute += ('seed_ledger',)
    if any(not getattr(args, name).is_absolute() for name in absolute):
        parser.error('all paths must be absolute')
    evaluate(args)


if __name__ == '__main__':
    main()
