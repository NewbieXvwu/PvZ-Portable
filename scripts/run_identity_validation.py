"""Isolated task19 dw3 training / fixed-deck evaluation; never control external queues."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'python')]
import dagger_experiment_runner as guard
from pvz_recipe import load_recipe, materialize_recipe

OUT = ROOT / 'artifacts/task19'
SOURCE = Path.home() / 'PvZ-search'
guard.OUT = OUT
guard.ENV = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MALLOC_ARENA_MAX': '2',
             'PYTHONPATH': str(ROOT / 'python') + ':' + str(ROOT / 'scripts'),
             'PVZ_EXECUTABLE': str(ROOT / 'build/pvz-portable'),
             'PVZ_RESOURCE_DIR': str(Path.home() / '.cache/pvz-research-resources')}


def descendants(pid):
    """Pool handler threads may fork: inspect every task, not only the main thread."""
    found = set()
    todo = [pid]
    while todo:
        current = todo.pop()
        if current in found:
            continue
        found.add(current)
        for entry in Path(f'/proc/{current}/task').glob('*/children'):
            try:
                todo.extend(map(int, entry.read_text().split()))
            except (FileNotFoundError, ProcessLookupError):
                pass
    return list(found)


guard.descendants = descendants


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def trainers():
    result = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            args = (entry / 'cmdline').read_bytes().split(b'\0')[1:]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if any(Path(v.decode(errors='replace')).name.startswith('train_') and v.endswith(b'.py') for v in args):
            result.append(int(entry.name))
    return result


def prepare():
    target = OUT / 'validation_plan.json'
    if target.exists():
        return json.loads(target.read_text())
    labels = load_recipe(ROOT/'recipes/identity_dw3_s1.json')['labels']
    manifest = OUT / 'day_panel_frozen.json'
    shutil.copyfile(SOURCE / 'artifacts/stage4/day_panel.json', manifest)
    counts = {v: len(list((SOURCE / 'artifacts/student/frames_v2').glob('*/' + v + '/frames.npz'))) for v in labels}
    assert all(counts.values()), counts
    source_names = ['scripts/train_rollout_policy.py', 'python/pvz_rollout_policy.py',
                    'python/pvz_value_net.py', 'python/pvz_student_observation.py', 'python/pvz_frame_store.py',
                    'python/pvz_recipe.py', 'recipes/identity_dw3_s1.json']
    recipe = materialize_recipe(ROOT/'recipes/identity_dw3_s1.json', OUT/'training.recipe.json',
                                data=SOURCE/'artifacts/student/frames_v2', out=OUT/'identity_dw3_s1',
                                cache_dir=OUT/'cache')
    command = [guard.PYTHON, str(ROOT/'scripts/train_rollout_policy.py'), '--recipe', str(recipe)]
    plan = dict(registered_at=datetime.now(timezone.utc).isoformat(), training_seed=1, labels=labels,
                archive_counts=counts, command=command, training_sources={v: sha(ROOT / v) for v in source_names},
                recipe_sha256=sha(recipe),
                manifest=str(manifest), manifest_sha256=sha(manifest),
                task_ids=[f'day_l{level}_eval{suffix}' for suffix in ('_d7', '') for level in (7, 8, 9, 10)],
                evaluation_seeds=list(range(30000, 30256)), workers=4,
                comparison='report all 8x256 games, no best seed/epoch, no production change; L10 is fixed deck',
                reference_dw3s1_d7_wins=[248, 251, 229, 237])
    target.write_text(json.dumps(plan, indent=2) + '\n')
    return plan


def verify(plan):
    if 'recipe_sha256' in plan:
        assert sha(OUT/'training.recipe.json') == plan['recipe_sha256']
    for name, digest in plan['training_sources'].items():
        assert sha(ROOT / name) == digest, name
    assert sha(plan['manifest']) == plan['manifest_sha256']


def train(plan):
    marker = OUT / 'identity_training_complete.json'
    if marker.exists():
        verify(plan)
        record = json.loads(marker.read_text())
        assert record['plan_sha256'] == sha(OUT / 'validation_plan.json')
        assert sha(OUT / 'identity_dw3_s1/model.pt') == record['model_sha256']
        return
    stable_since = None
    while True:
        busy = trainers()
        if not busy and guard.available() >= 14 * 1024 * 1024:
            if stable_since is None:
                stable_since = time.monotonic()
            if time.monotonic() - stable_since >= 30:
                break
        else:
            stable_since = None
        print(json.dumps({'training_admission_wait': busy, 'available_kib': guard.available()}), flush=True)
        time.sleep(10)
    verify(plan)
    print(json.dumps({'training_admitted': True, 'foreign_trainers': trainers(),
                      'available_kib': guard.available()}), flush=True)
    result = guard.run(plan['command'], 'identity_training', extra={'CUDA_VISIBLE_DEVICES': '0'})
    assert result['returncode'] == 0
    verify(plan)
    import torch
    model = torch.load(OUT / 'identity_dw3_s1/model.pt', map_location='cpu', weights_only=True)
    assert model['packet_head'] == 'identity' and model['row_pool'] and len(model['history']) == 12
    assert set(model['labels'].split(',')) == set(plan['labels'])
    assert all(torch.isfinite(v).all() for v in model['state_dict'].values())
    marker.write_text(json.dumps(dict(model_sha256=sha(OUT / 'identity_dw3_s1/model.pt'),
                                     plan_sha256=sha(OUT / 'validation_plan.json'), result=result), indent=2) + '\n')


def evaluate(plan):
    verify(plan)
    complete = json.loads((OUT / 'identity_training_complete.json').read_text())
    assert complete['plan_sha256'] == sha(OUT / 'validation_plan.json')
    model = OUT / 'identity_dw3_s1/model.pt'
    assert sha(model) == complete['model_sha256']
    freeze_path = OUT / 'evaluation_freeze.json'
    source_names = [*plan['training_sources'], 'scripts/run_identity_validation.py',
                    'scripts/dagger_experiment_runner.py', 'scripts/search_play.py', 'scripts/render_episode.py',
                    'scripts/pvz_tasks.py', 'python/pvz_env.py', 'src/EnvironmentRollout.cpp',
                    'src/EnvironmentRollout.h', 'src/LawnApp.cpp', 'src/main.cpp']
    if not freeze_path.exists():
        freeze_path.write_text(json.dumps(dict(model_sha256=sha(model), manifest_sha256=sha(plan['manifest']),
            executable_sha256=sha(ROOT / 'build/pvz-portable'), sources={n: sha(ROOT / n) for n in source_names}), indent=2) + '\n')
    frozen = json.loads(freeze_path.read_text())

    def verify_evaluation():
        assert sha(model) == frozen['model_sha256']
        assert sha(plan['manifest']) == frozen['manifest_sha256']
        assert sha(ROOT / 'build/pvz-portable') == frozen['executable_sha256']
        for name, digest in frozen['sources'].items():
            assert sha(ROOT / name) == digest, name

    verify_evaluation()
    evidence = dict(model_sha256=sha(model), executable_sha256=sha(ROOT / 'build/pvz-portable'),
                    search_source_sha256=sha(ROOT / 'scripts/search_play.py'),
                    env_source_sha256=sha(ROOT / 'python/pvz_env.py'), tasks={})
    for task in plan['task_ids']:
        verify_evaluation()
        dest = OUT / 'eval' / task
        summary = dest / 'summary.json'
        if not summary.exists():
            command = [guard.PYTHON, 'scripts/search_play.py', '--value', str(SOURCE / 'artifacts/value_net/v2/model.pt'),
                       '--rollout-policy', str(model), '--manifest', plan['manifest'], '--task-id', task,
                       '--seeds', '256', '--workers', '4', '--resource-dir', guard.ENV['PVZ_RESOURCE_DIR'],
                       '--executable', str(ROOT / 'build/pvz-portable'), '--arms', 'fast', '--out', str(dest)]
            result = guard.run(command, 'eval_' + task, extra={'CUDA_VISIBLE_DEVICES': ''})
            assert result['returncode'] == 0
        rows = [json.loads(v) for v in (dest / 'games.jsonl').read_text().splitlines()]
        config = json.loads((dest / 'config.json').read_text())
        assert config['rollout_policy'] == str(model) and config['task_id'] == task
        assert config['executable'] == str(ROOT / 'build/pvz-portable') and config['workers'] == '4'
        assert len(rows) == 256 and {r['seed'] for r in rows} == set(plan['evaluation_seeds'])
        assert all(r['arm'] == 'fast' and r['searched'] == 0 for r in rows)
        wins = sum(r['won'] for r in rows)
        assert json.loads(summary.read_text())['fast']['won'] == wins
        evidence['tasks'][task] = dict(won=wins, games=256, games_sha256=sha(dest / 'games.jsonl'))
        print(json.dumps({'completed_task': task, 'wins': wins}), flush=True)
    evidence['complete'] = True
    verify_evaluation()
    evidence['evaluation_freeze_sha256'] = sha(freeze_path)
    (OUT / 'identity_evaluation_complete.json').write_text(json.dumps(evidence, indent=2) + '\n')


def finish(plan):
    """Reuse the live trainer, then run exactly the previously registered eight evaluation groups."""
    watcher = json.loads((OUT / 'train_watcher.process.json').read_text())['pid']
    while not (OUT / 'identity_training_complete.json').exists():
        try:
            args = Path(f'/proc/{watcher}/cmdline').read_bytes().split(b'\0')
        except FileNotFoundError:
            args = []
        if not guard.live(watcher) or b'scripts/run_identity_validation.py' not in args:
            raise RuntimeError('registered training watcher stopped without a completed model; inspect its existing logs')
        time.sleep(10)
    assert json.loads((ROOT / 'MEMORY/docs/task19-perturbation-smoke.json').read_text())['complete']
    evaluate(plan)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['prepare', 'train', 'evaluate', 'finish'], required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    plan = prepare()
    if args.phase == 'train':
        train(plan)
    elif args.phase == 'evaluate':
        evaluate(plan)
    elif args.phase == 'finish':
        finish(plan)
