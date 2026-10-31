#!/usr/bin/env python3
"""Paired student promotion evaluation (production v7).

The panel is the registered adventure set; the two heldout cells run for
monitoring only. Every round takes a fresh ledger-allocated seed block, and the
incumbent is re-evaluated on the same seeds (cached incumbent outcomes are never
used). Statistical rejection is a successful evaluation with approved=false;
incomplete data is an error and cannot produce a promotion certificate.

Parallelism (task73c): --eval-units N runs up to N search_play children
concurrently (each with the contract's own --workers). Game outcomes are a
function of (model, task, seed) only, so unit order and concurrency never
change the paired data; unit folders stay resumable per (side, task). The unit
count is an execution detail, deliberately absent from the contract.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]
import pvz_promotion
from pvz_promotion import GATE_VERSION, SEED_LEDGER_DEFAULT, decide, freeze_panel, policy


class SeedPoolExhausted(RuntimeError):
    """The gate seed pool has no contiguous block left; never wrap around and reuse."""


def read(path):
    return json.loads(Path(path).read_text())


def write(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.building')
    temporary.write_text(json.dumps(document, indent=2) + '\n')
    temporary.replace(path)


def contract(candidate, incumbent, source_panel, executable, value, resource_dir, workers=1,
             seed_block=None, candidates=None, calibration=None):
    result = dict(version=GATE_VERSION, policy=policy(), candidate=str(candidate),
                incumbent=str(incumbent), source_panel=str(source_panel), executable=str(executable),
                value=str(value), resource_dir=str(resource_dir), workers=workers,
                seed_block=[int(seed_block[0]), int(seed_block[1])] if seed_block else None)
    result.update(candidates=candidates, calibration=calibration)
    return result


def allocate_seed_block(ledger_path, total, pool=None, pools=None):
    """Reserve the next unused contiguous block from the gate seed pools.

    The ledger lives outside git and is the sole authority for gate seed
    consumption: blocks are handed out in pool order, never reused, never
    wrapped. Pools are walked in the order registered in SEED_POOLS_V2, so the
    second window only starts handing out blocks once the first one is spent;
    exhaustion of every pool raises SeedPoolExhausted.
    """
    if pool is not None:
        pools = (pool,)
    pools = tuple(tuple(int(bound) for bound in entry) for entry in (pools or pvz_promotion.SEED_POOLS_V2))
    ledger_path, lock_path = Path(ledger_path), Path(str(ledger_path) + '.lock')
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if ledger_path.exists():
            document = read(ledger_path)
            if tuple(tuple(p) for p in document.get('pools', ())) != pools:
                raise ValueError(f'seed ledger {ledger_path} was created for pools '
                                 f'{document.get("pools")}')
            blocks = document.get('allocated')
            if (type(blocks) is not list
                    or any(type(b) is not list or len(b) != 2 or any(type(s) is not int for s in b)
                           or b[0] >= b[1]
                           or not any(low <= b[0] and b[1] <= high for low, high in pools)
                           for b in blocks)):
                raise ValueError(f'corrupt seed ledger: {ledger_path}')
            ordered = sorted(tuple(b) for b in blocks)
            if any(a[1] > b[0] for a, b in zip(ordered, ordered[1:])):
                raise ValueError(f'overlapping blocks in seed ledger: {ledger_path}')
        else:
            blocks, ordered = [], []
        block = None
        for pool_low, pool_high in pools:
            cursor = pool_low
            for low, high in ordered:
                if high <= pool_low or low >= pool_high:
                    continue
                if cursor + total <= low:
                    break
                cursor = max(cursor, high)
            if cursor + total <= pool_high:
                block = (cursor, cursor + total)
                break
        if block is None:
            raise SeedPoolExhausted(
                f'gate seed pools {[list(entry) for entry in pools]} exhausted: need {total} '
                f'more seeds after {len(blocks)} allocated blocks; refusing to reuse seeds')
        blocks.append([block[0], block[1]])
        write(ledger_path, dict(pools=[list(entry) for entry in pools], allocated=blocks))
        return block


def require_allocated_block(ledger_path, seed_block):
    """The ledger is the sole seed authority: a caller-supplied block must be recorded."""
    document = read(Path(ledger_path))
    if [int(seed_block[0]), int(seed_block[1])] not in document.get('allocated', []):
        raise ValueError(f'seed block {[int(seed_block[0]), int(seed_block[1])]} is not allocated '
                         f'in the ledger {ledger_path}; allocate it via allocate_seed_block first')


def contract_key(document):
    return hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest()[:20]


def valid_certificate(result, frozen):
    """The certificate answers this contract with complete panel coverage."""
    if type(result) is not dict or result.get('contract') != frozen:
        return False
    if type(result.get('approved')) is not bool:
        return False
    reasons = result.get('reasons')
    if type(reasons) is not list or any(not isinstance(reason, str) for reason in reasons):
        return False
    if result['approved'] == bool(reasons):
        return False
    if any(reason.startswith('evaluation_error:') for reason in reasons):
        return True
    registered = (frozen.get('policy') or {}).get('tasks') or []
    adventure = sum(task['seeds'] for task in registered if not task.get('monitor'))
    overall, zones = result.get('overall'), result.get('zones')
    if result.get('sequential', {}).get('stopped'):
        fraction = frozen['policy']['sequential']['looks'][result['sequential']['look'] - 1]
        adventure = sum(math.ceil(task['seeds'] * fraction) for task in registered if not task.get('monitor'))
    return (result.get('version') == frozen.get('version') and result.get('policy') == frozen.get('policy')
            and type(overall) is dict and overall.get('pairs') == adventure
            and type(zones) is dict and bool(registered)
            and set(zones) == {task['zone'] for task in registered if not task.get('monitor')})




def _task_rows(folder, task):
    rows = [json.loads(line) for line in (folder / 'games.jsonl').read_text().splitlines()]
    if task.get('gate_key'):
        rows = [dict(row, seed=row.get('future_seed') if row.get('future_seed') is not None else row['seed'])
                for row in rows]
        rows = [row for row in rows if row['seed'] in set(task['seeds'])]
    if (len(rows) != len(task['seeds']) or {row['seed'] for row in rows} != set(task['seeds'])
            or any(row['arm'] != 'fast' or row['searched'] != 0 or type(row['won']) is not bool
                   for row in rows)):
        raise ValueError(f'incomplete/non-student/duplicate evaluation games: {folder}')
    return {row['seed']: row for row in rows}


def _unit_command(args, manifest_path, task, side, model):
    command = [sys.executable, str(ROOT / 'scripts/search_play.py'),
            '--rollout-policy', str(model), '--value', str(args.value),
            '--manifest', str(manifest_path), '--task-id', task['task_id'],
            '--executable', str(args.executable), '--resource-dir', str(args.resource_dir),
            '--seeds', str(len(task['seeds'])), '--workers', str(args.workers),
            '--arms', 'fast', '--dagger-beta', '0',
            '--native-games' if getattr(args, 'native_games', True) else '--no-native-games',
            '--out', str(args.out / side / task['task_id'])]
    if task.get('demo_start'):
        demo = task['demo_start']
        command += ['--demo-archive', str(ROOT / demo['archive']),
                    '--demo-decision', str(demo['decisions'][demo['index']]),
                    '--seed-list', str(task['demo_seed']),
                    '--future-seeds', ','.join(map(str, task['seeds']))]
    return command


def _run_unit(args, lock, manifest_path, task, side, model):
    """One (task, side) evaluation unit: resumable search_play over its own seeds."""
    folder = args.out / side / task['task_id']
    command = _unit_command(args, manifest_path, task, side, model)
    folder.mkdir(parents=True, exist_ok=True)
    complete = False
    if (folder / 'games.jsonl').exists():
        # search_play resumes missing (arm,seed) keys. Corruption is never repaired by
        # silently dropping rows; the complete-data validator below will reject it.
        count = len((folder / 'games.jsonl').read_text().splitlines())
        complete = count >= len(task['seeds']) if task.get('gate_key') else count == len(task['seeds'])
    if not complete:
        with (folder / 'run.log').open('a') as log:
            offset = log.tell()
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                    pass_fds=(lock.fileno(),),
                                    env={**os.environ, 'CUDA_VISIBLE_DEVICES': '',
                                         'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})
        if result.returncode:
            with (folder / 'run.log').open('rb') as log:
                log.seek(max(offset, (folder / 'run.log').stat().st_size - 8192))
                diagnostic = log.read().decode('utf-8', errors='replace')
            raise RuntimeError(f'{side} {task["task_id"]}: exit {result.returncode}; '
                               f'see {folder / "run.log"}\n{diagnostic}')
    rows = _task_rows(folder, task)
    print(f'{side} {task["task_id"]}: {sum(r["won"] for r in rows.values())}/{len(task["seeds"])}', flush=True)
    return side, task['task_id'], rows




def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('candidate', 'incumbent', 'source-panel', 'executable', 'value', 'resource-dir', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--candidates', required=True, help='JSON object mapping A/B family to model paths')
    parser.add_argument('--calibration', required=True, help='JSON object frozen from curriculum_state.gate_panel')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--native-games', action=argparse.BooleanOptionalAction, default=True,
                        help='enable whole-game C++ fog student games (deploy19)')
    parser.add_argument('--eval-units', type=int, default=1,
                        help='concurrent search_play children (task73c); execution detail, '
                             'never part of the contract, outcomes are identical for any value')
    parser.add_argument('--seed-ledger', type=Path, default=Path(SEED_LEDGER_DEFAULT),
                        help='out-of-git ledger of allocated gate seed blocks')
    parser.add_argument('--seed-block', default=None,
                        help='START,END already allocated in the ledger (pipeline flow); '
                             'omit to allocate the next free block')
    args = parser.parse_args()
    if args.workers not in (1, 2):
        parser.error('--workers must be 1 or 2')
    if args.eval_units < 1:
        parser.error('--eval-units must be positive')
    absolute = ('candidate', 'incumbent', 'source_panel', 'executable', 'value', 'resource_dir', 'seed_ledger')
    if any(not getattr(args, name).is_absolute() for name in absolute):
        parser.error('all paths must be absolute')
    evaluate(args)



def evaluate(args):
    """One shared incumbent run per seed, both candidates at every balanced look.

    Existing registration/manifest/decision files are reused. decision.json is
    replaced after each look, so restart resumes search_play's cumulative keys.
    """
    import pvz_promotion as sequential
    candidates = json.loads(args.candidates) if isinstance(args.candidates, str) else args.candidates
    calibration = json.loads(args.calibration) if isinstance(args.calibration, str) else args.calibration
    models = dict(candidates, incumbent=str(args.incumbent))
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / 'evaluation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        registration = args.out / 'registration.json'
        existing = read(registration) if registration.exists() else None
        if existing:
            block = existing['seed_block']
        elif args.seed_block:
            block = [int(x) for x in args.seed_block.split(',')]
            require_allocated_block(args.seed_ledger, block)
        else:
            block = allocate_seed_block(args.seed_ledger, policy()['pair_count'])
        frozen = contract(args.candidate, args.incumbent, args.source_panel, args.executable,
                          args.value, args.resource_dir, args.workers, seed_block=block,
                          candidates=candidates, calibration=calibration)
        manifest = freeze_panel(read(args.source_panel), seed_block=block,
                                calibration=calibration, demos=read(ROOT / 'configs/human_demo_tasks.json'))
        manifest_path = args.out / 'manifest.json'
        if existing is not None:
            if existing != frozen:
                raise ValueError('cannot resume a different model/build/panel/policy')
        else:
            write(manifest_path, manifest)
            write(registration, frozen)
        started = time.time()
        outcomes = {side: {} for side in models}
        for look, fraction in enumerate(sequential.LOOKS, 1):
            tasks = []
            for task in manifest['tasks']:
                seeds = task.get('gate_seeds', task['seeds'])
                count = len(seeds) if task['task_id'] in manifest['monitor'] else max(1, math.ceil(len(seeds) * fraction))
                tasks.append(dict(task, seeds=seeds[:count],
                                  **(dict(demo_seed=task['seeds'][0]) if task.get('demo_start') else {})))
            with ThreadPoolExecutor(max_workers=args.eval_units, thread_name_prefix='gate-eval') as executor:
                futures = [executor.submit(_run_unit, args, lock, manifest_path, task, side, model)
                           for task in tasks for side, model in models.items()]
                for future in futures:
                    side, name, rows = future.result()
                    outcomes[side][name] = rows
            def pairs(left, right):
                rows = []
                for task in tasks:
                    name = task['task_id']
                    for seed in task['seeds']:
                        a, b = outcomes[left][name][seed], outcomes[right][name][seed]
                        rows.append(dict(task_id=name, seed=seed, zone=manifest['groups'][name],
                                         candidate=a['won'], incumbent=b['won'],
                                         candidate_wave=a['final_wave'], incumbent_wave=b['final_wave'],
                                         endpoint=task['endpoint'],
                                         wave_count=a.get('wave_count') or b.get('wave_count')))
                return rows
            results = {side: sequential.decide(pairs(side, 'incumbent'), manifest, look) for side in candidates}
            stopped = all(r['status'] != 'continue' for r in results.values())
            eligible = [side for side in candidates if results[side]['approved']] if stopped else []
            winner = max(eligible, key=lambda side: results[side]['overall']['delta_rate'], default=None)
            representative = winner or max(results, key=lambda side: results[side]['overall']['delta_rate'])
            result = dict(results[representative])
            result.update(approved=bool(winner), winner=candidates.get(winner), winner_family=winner,
                          candidates=results, contract=frozen,
                          sequential=dict(look=look, fraction=fraction, stopped=stopped,
                                          adventure_pairs=result['overall']['pairs']),
                          started=started, finished=time.time())
            if not stopped:
                result['reasons'] = ['sequential_continue']
            elif not winner:
                result['reasons'] = ['no_candidate_passed']
            if stopped:
                result['next_panel'] = sequential.next_panel(calibration, manifest, outcomes)
                if 'A' in candidates and 'B' in candidates:
                    result['ab'] = sequential.ab_summary(pairs('A', 'B'))
            write(args.out / 'decision.json', result)
            print(f'sequential look {look}: ' + json.dumps({s: dict(status=r['status'],
                  delta=r['overall']['delta_rate'], reasons=r['reasons']) for s, r in results.items()}), flush=True)
            if stopped:
                return result

if __name__ == '__main__':
    main()
