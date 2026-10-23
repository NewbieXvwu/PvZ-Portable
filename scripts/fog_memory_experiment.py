"""Task45 B: same-data current-frame / K=4 public-zombie-stack comparison.

The memory model is evaluated through Python, without a new native rollout format.
Each stage is restartable; collection and evaluation use at most four game processes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
MAIN = Path('/home/newbiexvwu/PvZ-search')
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]
from curriculum_queue import command_for
from fog_cold_start import completed, sha
from pvz_recipe import load_recipe, training_row_options, write_recipe
from pvz_training_pack import pack_flags, pack_options

MEMORY = ROOT / 'artifacts/task45/memory'
MANIFEST = ROOT / 'artifacts/task45/fog_panel.json'
NETS = {}
ENV = None


def run_command(command, logfile, timeout=None):
    print('RUN ' + ' '.join(command), flush=True)
    with logfile.open('a') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=timeout,
                       env={**os.environ, 'PVZ_NO_POPUP': '1'})


def own_engines():
    from pvz_fog import task45_engine_pids
    return task45_engine_pids()


def collect():
    if own_engines():
        raise RuntimeError('another task45 stage has live game processes; wait for it to finish')
    tasks = {t['task_id']: t for t in json.loads(MANIFEST.read_text())['tasks']}
    args = SimpleNamespace(python=Path(sys.executable),
        rollout_policy=MAIN / 'artifacts/student/dw4i_s1/model.pt',
        static_value=MAIN / 'artifacts/hard/value_branch1/model.pt',
        value=MAIN / 'artifacts/value_net/v4/model.pt', executable=ROOT / 'build/pvz-portable',
        resource_dir=Path('/home/newbiexvwu/.cache/pvz-research-resources'), workers=4,
        frames=MEMORY / 'frames', out=MEMORY / 'collection')
    for level in range(31, 35):
        task_id = f'fog_l{level}_train_f8p'
        command = command_for(args, MANIFEST, task_id, 0, 8, f'fogmem{level}')
        command[command.index('--dagger-beta') + 1] = '1'
        cell = dict(out=command[command.index('--out') + 1], arm='fast_search',
                    seeds=tasks[task_id]['seeds'][:8])
        if completed(cell):
            continue
        folder = Path(cell['out'])
        folder.mkdir(parents=True, exist_ok=True)
        run_command(command, folder / 'run.log')
        assert completed(cell)
    files = sorted((MEMORY / 'frames').glob('seed_*/*/frames.npz'))
    assert len(files) == 32
    fingerprint = {str(p.relative_to(MEMORY)): sha(p) for p in files}
    (MEMORY / 'corpus.json').write_text(json.dumps(fingerprint, indent=2) + '\n')


def train():
    import numpy as np
    from pvz_frame_store import read_frames
    frozen = json.loads((MEMORY / 'corpus.json').read_text())
    for name, digest in frozen.items():
        assert sha(MEMORY / name) == digest
    rows = sum(len(read_frames(MEMORY / name)) - 1 for name in frozen)
    # Conservative dense upper bound plus 3 GiB for Python, encoder buffers, optimizer and CUDA.
    estimate = rows * 204 * 6 * 10 * 4 * 2 + 3 * 1024 ** 3
    gate = dict(archive_rows=rows, estimated_peak_bytes=estimate, memory_limit_bytes=10 * 1024 ** 3,
                gpu_time_limit_seconds=7200, epochs=12, same_data_sha256=sha(MEMORY / 'corpus.json'))
    (MEMORY / 'resource_gate.json').write_text(json.dumps(gate, indent=2) + '\n')
    if rows > 100000 or estimate >= 10 * 1024 ** 3:
        raise RuntimeError('memory experiment exceeds preregistered resource gate; ask controller to schedule')
    base = load_recipe(ROOT / 'recipes/dw4i.json', 'training')
    for name, obs in [('current', 'v2fog'), ('stack4', 'v2fogstack4')]:
        recipe = {**base, 'data': str(MEMORY / 'frames'), 'out': str(MEMORY / name),
                  'labels': [f'fogmem{level}_fast_search' for level in range(31, 35)],
                  'label_weight': {}, 'obs': obs, 'seed': 1, 'epochs': 12,
                  'init': str(MAIN / 'artifacts/student/dw4i_s1/model.pt'),
                  'cache_dir': str(MEMORY / 'feature_cache'), 'read_cache_dir': None,
                  'precompute_workers': 1, 'batch_size': 256, 'validation_batch_size': 256,
                  'provenance': {'notes': 'task45 same 32 human-fog teacher episodes; final epoch only'}}
        recipe_path = MEMORY / f'{name}.json'
        write_recipe(recipe_path, recipe)
        options = pack_options(**training_row_options(recipe))
        command = [sys.executable, str(ROOT / 'scripts/pack_training_rows.py'), '--data', recipe['data'],
                   '--labels', ','.join(recipe['labels']), *pack_flags(options),
                   '--cache-dir', str(MEMORY / 'feature_cache'), '--precompute-workers', '1']
        run_command(command, MEMORY / f'{name}-pack.log')
        if not (MEMORY / name / 'model.pt').exists():
            # /usr/bin/time records actual peak RSS and elapsed time; subprocess timeout caps GPU occupancy.
            run_command(['/usr/bin/time', '-v', sys.executable,
                        str(ROOT / 'scripts/train_rollout_policy.py'), '--recipe', str(recipe_path)],
                        MEMORY / f'{name}-train.log', timeout=3600)
        with np.load(__import__('pvz_training_pack').pack_path(
                MEMORY / 'frames', {**options, 'sources': __import__('pvz_training_pack').archive_fingerprint(
                    MEMORY / 'frames', set(recipe['labels']))}), allow_pickle=False) as pack:
            assert int(pack['rows']) <= rows


def initialize_eval():
    import torch
    from pvz_rollout_policy import load_student
    torch.set_num_threads(1)
    for name in ('current', 'stack4'):
        NETS[name] = load_student(MEMORY / name / 'model.pt')


def evaluate_game(job):
    global ENV
    from pvz_event_env import EventWaitEnv
    from pvz_env import TaskSpec
    from pvz_tasks import profile_for_deck
    from pvz_rollout_policy import act
    from pvz_student_observation import StudentHistory
    from render_episode import _frame, training_task_kwargs
    if ENV is None:
        ENV = EventWaitEnv('/home/newbiexvwu/.cache/pvz-research-resources',
                           executable=ROOT / 'build/pvz-portable')
        __import__('atexit').register(ENV.close)
        __import__('multiprocessing.util').util.Finalize(ENV, ENV.close, exitpriority=10)
    task, seed, arm = job
    kw = training_task_kwargs(task, seed)
    net = NETS[arm]
    obs, _ = ENV.reset(deck=kw['deck'], task=TaskSpec(level=kw['level'], seed=seed,
                  profile=profile_for_deck(tuple(kw['deck'])),
                  **{k: v for k, v in kw['task_extra'].items() if v is not None}))
    history, previous, action = StudentHistory(net.obs), None, None
    decisions = 0
    blover = False
    clear = False
    returns = []
    recently_visible_rows = set()
    planting_after_return = Counter()
    while not obs['terminal'] and decisions < 3000:
        frame = _frame(obs)
        frame['action'] = action
        action = act(net, frame, previous, obs['legal_actions']['plants'], history=history)
        previous = frame
        kind = obs['packets'][action['packet']]['type'] if action['type'] == 'plant' else None
        before = obs
        obs, _, _, _, info = ENV.step(action)
        if info['ok'] and kind == 27:
            blover = True
        opacity = sum(map(sum, obs['fog_amount']))
        if blover and opacity == 0:
            clear = True
            recently_visible_rows.update(z['row'] for z in obs['zombies'] if z['on_board'])
        elif blover and clear and opacity > 0:
            returns.append(dict(tick=obs['tick'], seen_rows=sorted(recently_visible_rows)))
            clear, blover = False, False
        if (info['ok'] and kind is not None and returns and
                0 <= before['tick'] - returns[-1]['tick'] <= 600):
            planting_after_return[action['row'], int(action['row'] in returns[-1]['seen_rows'])] += 1
        decisions += 1
    return dict(task_id=task['task_id'], seed=seed, arm=arm, won=obs['result'] == 1,
                terminal=obs['terminal'], final_wave=obs['wave'], final_tick=obs['tick'],
                returns=returns, planting_after_return=[dict(row=r, seen_before=bool(s), count=n)
                       for (r, s), n in sorted(planting_after_return.items())])


def evaluate():
    if own_engines():
        raise RuntimeError('another task45 stage has live games')
    tasks = [t for t in json.loads(MANIFEST.read_text())['tasks'] if
             t['level'] in range(31, 35) and t['task_id'].endswith(('_eval_f8', '_eval_f8p'))]
    jobs = [(t, seed, arm) for t in tasks for seed in t['seeds'] for arm in ('current', 'stack4')]
    path = MEMORY / 'evaluation.jsonl'
    prior = [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    done = {(r['task_id'], r['seed'], r['arm']) for r in prior}
    assert len(done) == len(prior)
    jobs = [j for j in jobs if (j[0]['task_id'], j[1], j[2]) not in done]
    with path.open('a') as stream, mp.get_context('fork').Pool(4, initializer=initialize_eval) as pool:
        for i, result in enumerate(pool.imap_unordered(evaluate_game, jobs), 1):
            stream.write(json.dumps(result) + '\n')
            stream.flush()
            if i % 16 == 0:
                print(f'evaluated {i}/{len(jobs)}', flush=True)
        pool.close()
        pool.join()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['collect', 'train', 'evaluate'])
    args = p.parse_args()
    MEMORY.mkdir(parents=True, exist_ok=True)
    dict(collect=collect, train=train, evaluate=evaluate)[args.stage]()


if __name__ == '__main__':
    main()
