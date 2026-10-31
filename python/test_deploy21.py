"""task183 targeted lifecycle cases; no production controller or large corpus."""
import copy
import json
import sys
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'python')]
import curriculum_queue as queue
import promotion_gate as gate
import pvz_promotion as v7
from pvz_recipe import load_recipe, write_recipe


def test_deploy21_three_way_early_stop_and_higher_passing_arm(tmp_path, monkeypatch):
    source = ROOT / 'configs/gate_v5_panel.json'
    settings = v7.initial_panel(gate.read(source))
    args = SimpleNamespace(candidate=tmp_path / 'published.pt', incumbent=tmp_path / 'inc.pt',
        candidates={'A': str(tmp_path / 'A.pt'), 'B': str(tmp_path / 'B.pt')}, calibration=settings,
        source_panel=source, executable=Path('/engine'), value=Path('/value'), resource_dir=Path('/resources'),
        workers=1, eval_units=2, out=tmp_path / 'eval', seed_ledger=tmp_path / 'ledger.json', seed_block=None)
    calls = []
    def run(args, lock, manifest_path, task, side, model):
        calls.append((side, task['task_id'], tuple(task['seeds'])))
        wins = {'incumbent': 5, 'A': 7, 'B': 9}[side]
        return side, task['task_id'], {seed: dict(won=seed % 10 < wins,
            final_wave=10 if seed % 10 < wins else 3, wave_count=10) for seed in task['seeds']}
    monkeypatch.setattr(gate, '_run_unit', run)
    result = gate.evaluate(args)
    assert result['winner_family'] == 'B' and result['approved']
    assert all(r['approved'] for r in result['candidates'].values())
    assert result['sequential']['look'] == 1
    assert result['overall']['pairs'] < 1728
    assert result['ab']['overall']['delta_rate'] < 0
    for name in {c[1] for c in calls}:
        arms = [c for c in calls if c[1] == name]
        assert {c[0] for c in arms} == {'A', 'B', 'incumbent'}
        assert len({c[2] for c in arms}) == 1
    assert gate.valid_certificate(result, result['contract'])


def test_deploy21_drift_guard_wave_endpoint_and_clear_rejection():
    manifest = dict(monitor=[], groups={})
    pairs = [dict(task_id='day', zone='day', seed=i, candidate=True, incumbent=True,
                  candidate_wave=20, incumbent_wave=20, wave_count=20, endpoint='win') for i in range(500)]
    pairs[0].update(candidate=False)
    result = v7.decide(pairs, manifest, 4)
    assert result['overall']['score_ci'][0] > -.03
    assert 'overall_point_estimate_negative' in result['reasons']
    assert result['status'] == 'reject' and not result['approved']
    for row in pairs:
        row.update(candidate=False, incumbent=False, endpoint='waves', candidate_wave=5, incumbent_wave=10)
    result = v7.decide(pairs, manifest, 1)
    assert result['overall']['delta_rate'] == -.25
    assert result['zones']['day']['holm_adjusted_p'] < .001
    assert result['status'] == 'reject'


def test_deploy21_staircase_keeps_decks_and_uses_persisted_endpoints():
    source = gate.read(ROOT / 'configs/gate_v5_panel.json')
    settings = v7.initial_panel(source)
    demos = gate.read(ROOT / 'configs/human_demo_tasks.json')
    panel = v7.freeze_panel(source, (100_000_000, 100_001_920), settings, demos)
    table = {task['gate_key']: task for task in panel['tasks']}
    for old in source['tasks']:
        assert table[old['task_id']]['deck'] == old['deck']
    assert table['task101_l34_fog_selected']['zombie_count_multiplier'] == .75
    assert table['task101_l34_fog_selected']['endpoint'] == 'waves'
    assert table['task101_l46_roof_demo']['demo_start']['index'] == 3
    outcomes = {side: {task['task_id']: {seed: {'won': task['level'] < 30}
                    for seed in task.get('gate_seeds', task['seeds'])} for task in panel['tasks']}
                for side in ('A', 'B', 'incumbent')}
    updated = v7.next_panel(settings, panel, outcomes)
    assert updated['task101_l13_night_n10']['multiplier'] == 1.75
    assert updated['task101_l34_fog_selected']['multiplier'] == .5
    assert updated['task101_l46_roof_demo']['demo_index'] == 2
    assert settings['task101_l34_fog_selected']['multiplier'] == .75


def test_clean_b_lineage_plan_waits_for_previous_b_and_recovers_after_restart(tmp_path, monkeypatch):
    from test_curriculum_pipeline import setup
    args, state = setup(tmp_path)
    args.recipe = ROOT / 'recipes/dw4i2_v2timer_rebal_seed2cold.json'
    args.alternate_recipe = ROOT / 'recipes/dw4i2_v2timer_clean_b.json'
    args.b_init = tmp_path / 'gptbc.pt'
    args.b_init.write_bytes(b'gptbc')
    pipeline = queue.TrainingPipeline(args, state)
    monkeypatch.setattr(pipeline.executor, 'submit', lambda *a: Future())
    try:
        for round_index in (1, 2):
            state['round'] = round_index
            pipeline.enqueue()
        first, second = state['training_jobs']
        a = load_recipe(Path(first['recipe']), 'training')
        b_path = Path(first['recipe']).with_name('recipe_b.json')
        b = load_recipe(b_path, 'training')
        assert a['init'] is None and a['epochs'] == 12
        assert a['source_weight']['llm_demo'] == 0
        assert b['init'] == str(args.b_init) and b['epochs'] == 8
        assert state['b_student'] == str(args.b_init)
        assert b['source_weight'] == dict(default=0., llm_demo=1., teacher_r52=1., teacher_gptbc=1.)
        assert first['training_seeds'] == [1, 2]
        parent = Path(first['recipe']).parent / 'seeds/seed_000002/student/model.pt'
        parent.parent.mkdir(parents=True)
        parent.write_bytes(b'first B')
        marker = parent.parent.parent / 'trained.json'
        marker.write_text(json.dumps(dict(status='trained', model=str(parent))))
        next_b = pipeline._b_recipe(second)
        assert next_b['init'] == str(parent) and next_b['epochs'] == 4
        for job in state['training_jobs']:
            job['status'] = 'done'
        restored = queue.TrainingPipeline(args, state)
        restored.executor.shutdown(wait=True)
        assert state['b_student'] == str(parent) and state['b_round'] == 1
        assert gate.read(args.state)['b_student'] == str(parent)
        print('recipe_b.json:', json.dumps({k:b[k] for k in ('init', 'epochs', 'source_weight')}))
        print('next recipe_b.json:', json.dumps({k:next_b[k] for k in ('init', 'epochs', 'source_weight')}))
    finally:
        pipeline.executor.shutdown(wait=True)


def test_deploy21_new_rows_weighted_without_changing_validation(tmp_path):
    import torch
    from pvz_training_rows import ReplayAdjustment
    adjust = ReplayAdjustment(tmp_path, ['seed_1/old'], [], 5)
    adjust.observe(tmp_path / 'seed_1/old/frames.npz', [0, 1])
    adjust.observe(tmp_path / 'seed_2/new/frames.npz', [0, 1])
    weights = torch.ones(4)
    adjust.apply(weights, torch.tensor([False, True, False, True]))
    assert weights.tolist() == [1, 1, 5, 1]


def test_deploy21_activation_is_saved_before_logging_and_boundary(tmp_path, monkeypatch):
    from test_curriculum_pipeline import setup, certificate
    import qc_boundary as boundary
    args, state = setup(tmp_path)
    pipeline = queue.TrainingPipeline(args, state)
    try:
        job = dict(round=3, model=str(tmp_path / 'new.pt'), recipe=str(args.recipe), status='queued')
        pipeline._seeded(job)
        state['training_jobs'].append(job)
        result = dict(model=job['model'], promotion=certificate(pipeline, job))
        result['promotion']['next_panel'] = {'sentinel': {'multiplier': 1.25}}
        future = Future(); future.set_result(result)
        original = queue.log_line
        def logged(path, message):
            if 'student activated' in message:
                assert gate.read(args.state)['student_round'] == 3
                assert gate.read(args.state)['gate_panel'] == {'sentinel': {'multiplier': 1.25}}
                assert gate.read(args.state)['training_jobs'][-1]['promotion_approved']
            original(path, message)
        monkeypatch.setattr(queue, 'log_line', logged)
        pipeline._poll_job(0, job, future, False)
        state['completed_groups'] = [dict(label='g')]
        queue.save_state(args.state, state)
        with args.log.open('a') as log:
            log.write('round 3 g rc=0\nbackground round 4 evaluate done: approved=True\n')
        assert not boundary.committed_boundary(args.log, args.state)
        state['student_round'] = 4
        queue.save_state(args.state, state)
        assert boundary.committed_boundary(args.log, args.state)
    finally:
        pipeline.executor.shutdown(wait=True)


def test_deploy21_e2e_real_engine_once(monkeypatch):
    """Opt-in tiny real collection/pack/two-trainer/gate/activation, CPU only.

    Shrink the existing panel in memory and cap real games to one decision for
    the gate. This is a wiring rehearsal, never a model-quality measurement.
    All outputs and ledgers live in one TemporaryDirectory and are deleted.
    """
    import os
    import subprocess
    import tempfile
    if os.environ.get('PVZ_DEPLOY21_E2E') != '1':
        pytest.skip('explicit one-time rehearsal only')
    production = Path.home() / 'PvZ-search'
    incumbent = Path(gate.read(Path.home() / 'pvz-logs/curriculum_state.json')['student'])
    with tempfile.TemporaryDirectory(prefix='pvz-deploy21-') as temporary:
        root = Path(temporary)
        recipe = load_recipe(ROOT / 'recipes/dw4i2_v2timer.json', 'training')
        recipe.update(data=str(root / 'frames'), labels=['rehearsal_fast_search'], label_weight={},
                      source_weight={}, out=str(root / 'unused'), device='cpu', max_steps=1,
                      batch_size=16, validation_batch_size=16, precompute_workers=1,
                      source_row_report=False, cache_dir=str(root / 'cache'),
                      validation=dict(seed=0, fraction=.5))
        write_recipe(root / 'recipe.json', recipe)
        warm = load_recipe(ROOT / 'recipes/dw4i2_v2timer_warm.json', 'training')
        warm.update({key: recipe[key] for key in ('data', 'labels', 'out', 'device', 'max_steps',
                    'batch_size', 'validation_batch_size', 'precompute_workers', 'source_row_report',
                    'cache_dir', 'validation')})
        write_recipe(root / 'warm.json', warm)
        args = SimpleNamespace(state=root / 'state.json', log=root / 'curriculum.log',
            frames=root / 'frames', recipe=root / 'recipe.json', alternate_recipe=root / 'warm.json',
            python=Path(sys.executable), rollout_policy=incumbent, workers=2, min_eval_units=1,
            training_parallelism=1, gate_version=7, native_games=False, stop_flag=root / 'stop',
            out=root / 'runs', executable=production / 'artifacts/production/deploy20/pvz-portable',
            value=production / 'artifacts/value_net/v5/model.pt',
            resource_dir=Path.home() / '.cache/pvz-research-resources',
            promotion_panel=ROOT / 'configs/gate_v5_panel.json', seed_ledger=root / 'ledger.json')
        def limited(command, actions):
            code = (f'import sys; sys.path.insert(0, {str(ROOT / "scripts")!r}); '
                    f'import search_play as s; s.MAX_ACTIONS={actions}; s.main()')
            return [command[0], '-c', code, *command[2:]]
        collect = [sys.executable, str(ROOT / 'scripts/search_play.py'),
            '--rollout-policy', str(incumbent), '--value', str(args.value),
            '--manifest', str(ROOT / 'configs/student_selection_v2.json'), '--task-id', 'day_l1_train_d7_m250',
            '--seeds', '2', '--workers', '1', '--arms', 'fast_search', '--dagger-beta', '0',
            '--no-native-games', '--samples', '1', '--horizon', '1', '--no-halving',
            '--executable', str(args.executable), '--resource-dir', str(args.resource_dir),
            '--frames', str(args.frames), '--frames-label', 'rehearsal', '--obs-v3',
            '--out', str(root / 'collect')]
        with (root / 'collect.log').open('w') as output:
            completed = subprocess.run(limited(collect, 20), stdout=output, stderr=subprocess.STDOUT, cwd=ROOT)
        assert completed.returncode == 0, (root / 'collect.log').read_text()[-5000:]
        print('E2E collection: 2 real games, 20 decisions each', flush=True)
        state = dict(round=1, student=str(incumbent), student_round=0, log=[], cells={}, offsets={},
                     student_corpus_archives=[])
        pipeline = queue.TrainingPipeline(args, state)
        original_freeze, original_command = gate.freeze_panel, gate._unit_command
        original_policy = v7.policy
        def tiny_policy():
            result = original_policy()
            tasks, seen = [], set()
            for task in result['tasks']:
                if task['monitor']:
                    tasks.append(dict(task, seeds=2))
                elif task['zone'] not in seen:
                    seen.add(task['zone'])
                    tasks.append(dict(task, seeds=32))
            result['tasks'] = tasks
            return result
        monkeypatch.setattr(v7, 'policy', tiny_policy)
        def tiny_panel(*a, **kw):
            panel = original_freeze(*a, **kw)
            tasks, seen = [], set()
            for task in panel['tasks']:
                zone = panel['groups'][task['task_id']]
                if task['task_id'] in panel['monitor']:
                    tasks.append(dict(task, seeds=task['seeds'][:2]))
                elif zone not in seen:
                    seen.add(zone)
                    tasks.append(dict(task, seeds=task.get('gate_seeds', task['seeds'])[:32]))
            panel['tasks'] = tasks
            return panel
        monkeypatch.setattr(gate, 'freeze_panel', tiny_panel)
        monkeypatch.setattr(gate, '_unit_command', lambda *a, **kw: limited(original_command(*a, **kw), 1))
        def command(command, log):
            if command[1].endswith('/promotion_gate.py'):
                def arg(name): return command[command.index('--' + name) + 1]
                eval_args = SimpleNamespace(candidate=Path(arg('candidate')), incumbent=Path(arg('incumbent')),
                    candidates=arg('candidates'), calibration=arg('calibration'), source_panel=Path(arg('source-panel')),
                    executable=args.executable, value=args.value, resource_dir=args.resource_dir,
                    workers=1, eval_units=1, native_games=False, out=Path(arg('out')),
                    seed_ledger=args.seed_ledger, seed_block=arg('seed-block'))
                gate.evaluate(eval_args)
            else:
                with log.open('a') as output:
                    completed = subprocess.run(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
                assert completed.returncode == 0, log.read_text()[-5000:]
                print('E2E phase: ' + Path(command[1]).stem, flush=True)
            return 0
        monkeypatch.setattr(pipeline, '_command', command)
        try:
            pipeline.enqueue()
            for _, future in pipeline.futures:
                future.result()
            pipeline.poll(retry=False)
            saved = gate.read(args.state)
            job = saved['training_jobs'][-1]
            report = job['timing']
            assert saved['student_round'] == 1 and job['promotion_approved']
            assert len(report['candidates']) == 2
            assert report['promotion']['sequential']['look'] == 3
            assert (Path(job['recipe']).parent / 'ab_result.json').exists()
            assert not (Path(job['recipe']).parent / 'selection_manifest.json').exists()
            print('E2E promotion saved: A/B 1 optimizer step each; look 3/4, 120 adventure pairs; temp removed on exit', flush=True)
        finally:
            pipeline.executor.shutdown(wait=True)
