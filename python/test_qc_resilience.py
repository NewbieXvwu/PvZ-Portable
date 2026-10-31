"""task98: injected failures must not stop unrelated cells or reuse reservations."""
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'python')]
import curriculum_queue as q
from test_curriculum_pipeline import prepared_selection_without_real_games


def cell():
    return dict(level=1, mult_index=0, last_round=0, windows={})


def state():
    return dict(round=0, cells={'bad': cell(), 'good': cell()}, offsets={}, log=[])


def args(root):
    return SimpleNamespace(state=root / 'state.json', log=root / 'curriculum.log',
                           out=root / 'runs', stop_flag=root / 'stop', workers=2,
                           rollout_policy=root / 'student.pt', beta_threshold=.1,
                           starved_beta=.5, probe_interval=3, rounds=1,
                           round_games=4, games_per_group=2,
                           # task140: argparse always sets these; a fixture that forgot them
                           # made collect_rounds raise AttributeError and spin forever.
                           save_candidates_levels=frozenset(),
                           random_deck_share=q.RANDOM_DECK_SHARE)


def plan():
    return [dict(base=n, task_id=n, multiplier=1., manifest='/manifest',
                 label=f'c1{n}', games=2, offset=0) for n in ('bad', 'good')]


@pytest.mark.parametrize('damage', [dict(mult_index=9), dict(windows=[]),
                                    dict(windows={'1.0': {'recent': ['bad']}}),
                                    dict(ladder=['bad']), dict(level='bad')])
def test_bad_cell_is_retired_before_shared_sorting(damage):
    s = state()
    s['cells']['bad'].update(damage)
    tasks = {n: dict(seeds=list(range(10))) for n in ('bad', 'good')}
    jobs, changes = q.plan_round(s, tasks, {}, [1.], {n: Path('/m') for n in tasks}, 4, 2)
    assert [j['base'] for j in jobs] == ['good']
    assert 'bad' in s['retired_cells'] and 'bad' in s['cell_errors']
    assert any('cell retired: bad' in line for line in changes)


def test_missing_task_only_retires_its_cell():
    s = state()
    jobs, _ = q.plan_round(s, {'good': {'seeds': [1, 2]}}, {}, [1.],
                            {'good': Path('/m')}, 4, 2)
    assert [j['base'] for j in jobs] == ['good']
    assert 'bad' in s['retired_cells']


def test_plain_native_multiplier_ladder_is_not_a_demo_start():
    base = 'pool_l25_orig_train'
    c = dict(cell(), level=25, ladder=[.5, 1., 1.25])
    s = dict(round=0, offsets={}, cells={base: c}, retired_cells={}, log=[])
    q.reconcile_start_cells(s, {})
    assert base in s['cells']
    tasks = {base + '_m050': dict(level=25, seeds=[1, 2])}
    jobs, _ = q.plan_round(s, tasks, {'m050': .5}, [1., 1.25],
                            {base + '_m050': Path('/m')}, 2, 2)
    assert len(jobs) == 1 and 'demo_archive' not in jobs[0]
    assert jobs[0]['multiplier'] == .5


def test_exhaustion_is_not_resurrected_on_reconcile():
    s = state()
    spec = dict(cell(), ladder=[100, 50], seeds=list(range(256)), task_id='bad',
                family='demo_start', archive='/numeric/demo')
    s['cells']['bad'] = copy.deepcopy(spec)
    s['offsets']['bad'] = 256
    jobs, _ = q.plan_round(s, {'good': {'seeds': [1, 2]}}, {}, [1.],
                            {n: Path('/m') for n in ('bad', 'good')}, 4, 2)
    assert [j['base'] for j in jobs] == ['good']
    q.reconcile_start_cells(s, {'bad': spec})
    assert 'bad' not in s['cells']
    # task99 preserves the retired table until an explicit restore; extension
    # then resumes at the old absolute offset without wrapping.
    spec['seeds'] += [256, 257]
    q.reconcile_start_cells(s, {'bad': spec}, restore=('bad',))
    assert s['offsets']['bad'] == 256 and 'bad' in s['cells']


def test_bad_history_and_non_numeric_seed_directory_do_not_stop_import(tmp_path):
    for name, mult in [('seed_f44300000', 1), ('broken', 'not-a-number')]:
        p = tmp_path / name
        p.mkdir()
        (p / 'config.json').write_text(json.dumps(dict(task_id='good',
            zombie_count_multiplier=mult, seed_offset=4, seeds=2)))
        (p / 'games.jsonl').write_text('{"won":true}\n')
    with patch.object(q, 'training_base_ids', return_value={'good'}):
        windows, offsets = q.init_from_runs([str(tmp_path / '*' / 'config.json')], {})
    assert offsets == {'good': 6}
    assert windows['good']['1.0']['games'] == 1


def test_optional_teacher_entries_are_independent(tmp_path):
    p = tmp_path / 'rates.json'
    p.write_text(json.dumps(dict(bases={'good': .75, 'bad': {'wins': 1, 'games': 'x'},
                                     'nan': float('nan')}, levels=[])))
    assert q.load_teacher_rates(p) == {'good': .75}
    p.write_text('[]')
    assert q.load_teacher_rates(p) == {}


@pytest.mark.parametrize('failure', ['exit', 'system_exit', 'bad_json', 'missing_results'])
@pytest.mark.parametrize('restart', [False, True])
def test_bad_group_is_saved_and_next_group_runs(tmp_path, failure, restart):
    a, s, jobs = args(tmp_path), state(), plan()
    seen = []
    def collect(command, **kwargs):
        task = command[0]
        seen.append(task)
        if restart and task == 'bad':
            a.stop_flag.touch()
        assert json.loads(a.state.read_text())['offsets'][task] == 2
        if task == 'bad' and failure == 'system_exit':
            raise SystemExit('bad group')
        if task == 'bad' and failure == 'missing_results':
            return SimpleNamespace(returncode=0)
        folder = a.out / f'c1{task}_2'
        folder.mkdir(parents=True)
        (folder / 'games.jsonl').write_text('broken' if task == 'bad' and failure == 'bad_json'
                                           else '{"won":true,"decisions":2}\n' * 2)
        return SimpleNamespace(returncode=1 if task == 'bad' and failure == 'exit' else 0)
    with patch.object(q, 'command_for', side_effect=lambda *p, **kw: [p[2]]), \
         patch.object(q.subprocess, 'run', side_effect=collect), \
         patch.object(q, 'mem_available_bytes', return_value=None):
        if restart:
            assert not q.run_round(a, s, jobs, [])
            s = json.loads(a.state.read_text())
            assert s['completed_groups'][0]['status'] == 'failed'
            a.stop_flag.unlink()
            assert q.run_round(a, s, jobs[1:], [])
        else:
            assert q.run_round(a, s, jobs, [])
    assert seen == ['bad', 'good']
    saved = json.loads(a.state.read_text())
    assert saved['round'] == 1 and saved['offsets'] == {'bad': 2, 'good': 2}
    assert [j['status'] for j in saved['log'][0]['groups']] == ['failed', 'done']
    assert q.pack_labels(saved) == ['c1good_fast_search']
    assert saved['cells']['bad']['windows'] == {}


def test_zero_valid_round_does_not_enqueue_or_mark_cells_completed(tmp_path):
    arguments, current = args(tmp_path), state()
    with patch.object(q, 'command_for', return_value=['collect']), \
         patch.object(q.subprocess, 'run', return_value=SimpleNamespace(returncode=1)), \
         patch.object(q, 'mem_available_bytes', return_value=None), \
         patch.object(q, 'pipeline_action') as background:
        assert q.run_round(arguments, current, plan(), [])
    assert all(entry['last_round'] == 0 for entry in current['cells'].values())
    assert all(group['valid_games'] == 0 for group in current['log'][0]['groups'])
    assert 'round 1 done: 0 groups, 0 games' in arguments.log.read_text()
    assert all(call.args[1] != 'enqueue' for call in background.call_args_list)


def test_systemic_zero_groups_pause_and_resume_same_round_with_fresh_seeds(tmp_path):
    arguments, current = args(tmp_path), state()
    arguments.round_games = 8
    current['cells'] = {name: cell() for name in ('a', 'b', 'c', 'd')}
    tasks = {name: dict(seeds=list(range(20))) for name in current['cells']}
    manifests = {name: Path('/manifest') for name in tasks}
    health = q.GameHealth(arguments)
    seen, probes = [], []

    def collect(command, **kwargs):
        task, offset, games, label = command
        seen.append((task, offset))
        if len(seen) > 3:
            folder = arguments.out / f'{label}_{games}'
            folder.mkdir(parents=True)
            (folder / 'games.jsonl').write_text('{"won":false}\n' * games)
        return SimpleNamespace(returncode=1 if len(seen) <= 3 else 0)

    def probe():
        assert len(seen) == 3
        assert current['round'] == 0
        assert all(entry['last_round'] == 0 for entry in current['cells'].values())
        probes.append(1)
        return len(probes) == 2

    with patch.object(q, 'command_for', side_effect=lambda *pos, **kw: list(pos[2:6])), \
         patch.object(q.subprocess, 'run', side_effect=collect), \
         patch.object(q, 'mem_available_bytes', return_value=None), \
         patch.object(health, 'probe', side_effect=probe), \
         patch.object(health.closed, 'wait', return_value=False) as delay, \
         patch.object(q, 'pipeline_action') as background:
        q.collect_rounds(arguments, current, tasks, {}, [1.], manifests, {})
    assert probes == [1, 1]
    assert [call.args[0] for call in delay.call_args_list] == [180, 180]
    assert len(seen) == 7 and len(set(seen)) == 7
    assert current['round'] == 1
    assert all(entry['last_round'] == 1 for entry in current['cells'].values())
    assert sum(call.args[1] == 'enqueue' for call in background.call_args_list) == 1
    assert 'round 1 done: 4 groups, 8 games' in arguments.log.read_text()


def test_systemic_pause_survives_restart_and_honors_stop(tmp_path):
    arguments, current = args(tmp_path), state()
    current['zero_game_groups'] = 3
    health = q.GameHealth(arguments)

    def stop(seconds):
        arguments.stop_flag.touch()
        return False

    with patch.object(health.closed, 'wait', side_effect=stop), \
         patch.object(health, 'probe') as probe, \
         patch.object(q, '_run_group') as collect:
        assert q.run_round(arguments, current, plan(), []) is False
    probe.assert_not_called()
    collect.assert_not_called()
    assert current['round'] == 0


@pytest.mark.parametrize('result', ['valid', 'nonzero', 'empty', 'invalid'])
def test_recovery_probe_uses_one_isolated_game(tmp_path, result):
    arguments = args(tmp_path)
    arguments.promotion_panel = tmp_path / 'panel.json'
    arguments.promotion_panel.write_text('{"tasks":[{"task_id":"probe"}]}')
    health = q.GameHealth(arguments)

    def command(*pos, **kwargs):
        assert pos[2:6] == ('probe', 0, 1, 'recovery')
        assert kwargs['workers'] == 1
        assert kwargs['out'] != arguments.out
        folder = kwargs['out'] / 'recovery_1'
        folder.mkdir()
        (folder / 'games.jsonl').write_text(
            '' if result == 'empty' else '{"won":null}\n' if result == 'invalid' else '{"won":false}\n')
        return ['probe']

    with patch.object(q, 'command_for', side_effect=command), \
         patch.object(q.subprocess, 'run', return_value=SimpleNamespace(
             returncode=1 if result == 'nonzero' else 0)) as process:
        assert health.probe() is (result == 'valid')
    assert process.call_args.kwargs['timeout'] == q.PROBE_TIMEOUT_SECONDS


def test_recovery_probe_supplies_seed_for_production_panel(tmp_path):
    from test_curriculum_pipeline import setup
    arguments, _ = setup(tmp_path)
    arguments.static_value = tmp_path / 'static.pt'
    source = arguments.promotion_panel.read_bytes()
    original = json.loads(source)['tasks'][0]
    assert 'seeds' not in original

    def collect(command, **kwargs):
        manifest = Path(command[command.index('--manifest') + 1])
        assert manifest != arguments.promotion_panel
        task = json.loads(manifest.read_text())['tasks'][0]
        assert task == dict(original, seeds=[0])
        assert command[command.index('--seeds') + 1] == '1'
        assert command[command.index('--workers') + 1] == '1'
        folder = Path(command[command.index('--out') + 1])
        folder.mkdir()
        (folder / 'games.jsonl').write_text('{"won":false}\n')
        return SimpleNamespace(returncode=0)

    with patch.object(q.subprocess, 'run', side_effect=collect):
        assert q.GameHealth(arguments).probe()
    assert arguments.promotion_panel.read_bytes() == source


@pytest.mark.parametrize('failure', ['preparation', 'pack', 'train', 'evaluation'])
def test_systemic_background_failures_wait_without_spending_attempts(tmp_path, failure):
    from test_curriculum_pipeline import setup, write_evaluation
    arguments, current = setup(tmp_path)
    current.update(round=1, log=[dict(groups=[dict(label='c1a')])])
    pipeline = q.TrainingPipeline(arguments, current)
    failures = []
    original_prepare = pipeline._prepare_corpus

    def fail(phase):
        if phase == failure and len(failures) < 3:
            failures.append(phase)
            raise RuntimeError('PvZ-Portable exited unexpectedly with code 127')

    def prepare(job):
        fail('preparation')
        return original_prepare(job)

    def command(command, logfile):
        fail(logfile.stem)
        if logfile.name == 'train.log':
            model = logfile.parent / 'student/model.pt'
            model.parent.mkdir(parents=True)
            model.write_bytes(b'trained')
        elif logfile.name == 'evaluation.log':
            write_evaluation(pipeline, command, approved=False)
        return 0

    try:
        with patch.object(pipeline, '_prepare_corpus', side_effect=prepare), \
             patch.object(pipeline, '_command', side_effect=command), \
             patch.object(pipeline.game_health, 'probe', side_effect=[False, True] * 3) as probe, \
             patch.object(pipeline.game_health.closed, 'wait', return_value=False):
            pipeline.enqueue()
            pipeline.futures[0][1].result(timeout=5)
            pipeline.poll()
        job = current['training_jobs'][0]
        assert failures == [failure] * 3 and probe.call_count == 6
        assert job.get('attempts', 0) == 0 and job['status'] == 'done'
    finally:
        pipeline.close()


def test_systemic_background_stop_is_resumable_without_spending_attempts(tmp_path):
    from test_curriculum_pipeline import setup
    arguments, current = setup(tmp_path)
    pipeline = q.TrainingPipeline(arguments, current)
    job = dict(round=1, status='queued', attempts=1,
               recipe=str(arguments.recipe), model=str(arguments.rollout_policy), seed_block=[1, 3])
    current['training_jobs'].append(job)
    with patch.object(pipeline, '_run_snapshot', side_effect=RuntimeError(
            'PvZ-Portable exited unexpectedly with code 127')):
        pipeline.game_health.closed.set()
        future = pipeline.executor.submit(pipeline._run, dict(job))
        pipeline.futures.append((job, future))
        with pytest.raises(q.GameRecoveryStopped):
            future.result(timeout=5)
        pipeline.close()
    assert job['status'] == 'paused' and job['attempts'] == 1
    restored = json.loads(arguments.state.read_text())
    del arguments.game_health
    with patch.object(q.TrainingPipeline, '_run_snapshot', return_value=dict(superseded_by=2)), \
         patch.object(q.GameHealth, 'probe', return_value=True) as probe, \
         patch.object(q, 'GAME_RECOVERY_SECONDS', 0):
        restarted = q.TrainingPipeline(arguments, restored)
        restarted.futures[0][1].result(timeout=5)
        restarted.close()
    probe.assert_called_once()
    assert restored['training_jobs'][0]['status'] == 'superseded'
    assert restored['training_jobs'][0]['attempts'] == 1


@pytest.mark.parametrize('stale', [False, True])
def test_background_failure_diagnostics_only_use_current_attempt(tmp_path, stale):
    from test_curriculum_pipeline import setup
    arguments, current = setup(tmp_path)
    pipeline = q.TrainingPipeline(arguments, current)
    logfile = tmp_path / 'pack.log'
    marker = 'Inconsistency detected by ld.so: Assertion failed'
    if stale:
        logfile.write_text(marker + '\n')

    def process(command, **kwargs):
        kwargs['stdout'].write('bad archive\n' if stale else marker + '\n')
        kwargs['stdout'].flush()
        return SimpleNamespace(pid=12345, poll=lambda: 1)

    try:
        with patch.object(q.subprocess, 'Popen', side_effect=process):
            with pytest.raises(RuntimeError) as failure:
                pipeline._command(['fail'], logfile)
        assert q.systemic_game_failure(failure.value) is (not stale)
    finally:
        pipeline.close()


@pytest.mark.parametrize('legacy_label', [False, True])
def test_restart_after_partial_group_isolates_new_seed_results(tmp_path, legacy_label):
    arguments, current = args(tmp_path), state()
    current['cells'].pop('bad')
    tasks = {'good': dict(seeds=list(range(8)))}

    def replan():
        return q.plan_round(current, tasks, {}, [1.], {'good': Path('/manifest')}, 2, 2)[0]

    first = replan()
    if legacy_label:
        first[0]['label'] = first[0]['label'].removesuffix('_o0')
    old_results = arguments.out / f'{first[0]["label"]}_2' / 'games.jsonl'

    def collect(command, **kwargs):
        results = arguments.out / f'{command[0]}_2' / 'games.jsonl'
        results.parent.mkdir(parents=True, exist_ok=True)
        with results.open('a') as output:
            output.write('{"won":true}\n' * (1 if results == old_results else 2))
        if results == old_results:
            raise KeyboardInterrupt
        return SimpleNamespace(returncode=0)

    with patch.object(q, 'command_for', side_effect=lambda *pos, **kw: [pos[5]]), \
         patch.object(q.subprocess, 'run', side_effect=collect), \
         patch.object(q, 'mem_available_bytes', return_value=None):
        with pytest.raises(KeyboardInterrupt):
            q.run_round(arguments, current, first, [])
        current = json.loads(arguments.state.read_text())
        assert current['offsets']['good'] == 2
        restarted = replan()
        assert q.run_round(arguments, current, restarted, [])

    completed = current['log'][0]['groups'][0]
    new_results = arguments.out / f'{completed["label"]}_2' / 'games.jsonl'
    assert completed['status'] == 'done'
    assert completed['offset'] == 2 and current['offsets']['good'] == 4
    assert new_results != old_results
    assert len(old_results.read_text().splitlines()) == 1
    assert len(new_results.read_text().splitlines()) == 2
    assert q.pack_labels(current) == [completed['label'] + '_fast_search']


def test_state_io_retries_once_without_starting_a_game(tmp_path):
    original = Path.replace
    calls = []
    def replace(p, target):
        calls.append(p)
        if len(calls) == 1:
            raise OSError('temporary IO')
        return original(p, target)
    with patch.object(Path, 'replace', replace):
        q.save_state(tmp_path / 'state.json', state())
    assert len(calls) == 2
    assert json.loads((tmp_path / 'state.json').read_text())['offsets'] == {}


def test_persistent_state_error_blocks_collection_until_saved(tmp_path):
    a, s = args(tmp_path), state()
    events = []
    def save(*unused):
        events.append('save')
        if len(events) == 1:
            raise q.StateWriteError('disk unavailable')
    def run(*unused):
        events.append('collect')
        return False
    with patch.object(q, 'save_state', side_effect=save), \
         patch.object(q, 'plan_round', return_value=([], [])), \
         patch.object(q, 'startup_worker_estimate', return_value=1), \
         patch.object(q, 'run_round', side_effect=run), \
         patch.object(q.time, 'sleep') as sleep:
        q.collect_rounds(a, s, {}, {}, [1.], {}, {})
    assert events == ['save', 'save', 'collect']
    sleep.assert_called_once_with(120)


def test_round_boundary_retries_are_bounded_then_give_up(tmp_path):
    a = args(tmp_path)
    with patch.object(q, 'save_state'), \
         patch.object(q, 'plan_round', side_effect=OSError('manifest unreadable')), \
         patch.object(q.time, 'sleep') as sleep:
        with pytest.raises(OSError, match='manifest unreadable'):
            q.collect_rounds(a, state(), {}, {}, [1.], {}, {})
    assert [call.args[0] for call in sleep.call_args_list] == [120, 240, 480, 960, 1800]
    # one retry line per attempt plus the fatal line, never an unbounded log
    assert len(a.log.read_text().splitlines()) == q.ROUND_RETRY_LIMIT + 1


def test_programming_error_at_round_boundary_is_not_retried(tmp_path):
    a = args(tmp_path)
    with patch.object(q, 'save_state'), \
         patch.object(q, 'plan_round', side_effect=AttributeError('save_candidates_levels')), \
         patch.object(q.time, 'sleep') as sleep:
        with pytest.raises(AttributeError):
            q.collect_rounds(a, state(), {}, {}, [1.], {}, {})
    sleep.assert_not_called()
    assert len(a.log.read_text().splitlines()) == 1


def test_keyboard_interrupt_is_not_swallowed(tmp_path):
    with patch.object(q, 'save_state', side_effect=KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt):
            q.collect_rounds(args(tmp_path), state(), {}, {}, [1.], {}, {})


def test_configuration_errors_still_fail_at_startup():
    with patch.object(sys, 'argv', ['curriculum_queue.py', '--workers', '0']), \
         patch.object(q, 'set_oom_score_adj'):
        with pytest.raises(SystemExit) as error:
            q.main()
    assert error.value.code == 2


def test_bad_retired_cell_does_not_break_reconcile():
    s = state()
    s.update(cells={}, retired_cells={'bad': 'broken'})
    with patch.object(q, 'training_base_ids', return_value={'bad'}):
        q.reconcile_cells(s, {'bad': {'level': 1}}, {}, {})
    assert not s['cells'] and 'bad' in s['cell_errors']


def test_bad_saved_jobs_are_isolated_without_training(tmp_path):
    from test_curriculum_pipeline import setup
    a, s = setup(tmp_path)
    s['training_jobs'] = [42, {'status': 'running'}]
    p = q.TrainingPipeline(a, s)
    p.close()
    assert all(j['isolated'] and j['status'] == 'failed' for j in s['training_jobs'])


def test_gate_ledger_io_retries_once(tmp_path):
    p = object.__new__(q.TrainingPipeline)
    p.args = SimpleNamespace(seed_ledger=tmp_path / 'ledger.json')
    job = {}
    with patch.object(q, 'allocate_seed_block', side_effect=[OSError('temporary'), (1, 3)]) as allocate:
        p._seeded(job)
    assert allocate.call_count == 2 and job['seed_block'] == [1, 3]


def test_orphaned_snapshot_recovery_failure_isolated(tmp_path):
    from test_curriculum_pipeline import setup
    a, s = setup(tmp_path)
    p = q.TrainingPipeline(a, s)
    s['training_jobs'].append(dict(round=1, status='queued', recipe='/r', model='/m'))
    with patch.object(p, '_seeded', side_effect=ValueError('pool exhausted')):
        p.poll()
    p.close()
    assert s['training_jobs'][0]['isolated']


def test_twice_failed_background_snapshot_does_not_poison_next_round(tmp_path):
    from test_curriculum_pipeline import setup, write_evaluation
    a, s = setup(tmp_path)
    s.update(round=1, log=[dict(groups=[dict(label='c1a')])])
    def command(self, cmd, log):
        if log.parent.name == 'round_000001':
            raise RuntimeError('bad archive directory')
        if log.name == 'train.log':
            model = log.parent / 'student/model.pt'
            model.parent.mkdir(parents=True)
            model.write_bytes(b'trained')
        if log.name == 'evaluation.log':
            write_evaluation(self, cmd, approved=False)
    with patch.object(q.TrainingPipeline, '_command', command), \
         patch.object(q, 'mem_available_bytes', return_value=64 * q.GIB):
        p = q.TrainingPipeline(a, s)
        p.enqueue()
        for _ in range(2):
            with pytest.raises(RuntimeError):
                p.futures[0][1].result(timeout=5)
            p.poll()
        assert s['training_jobs'][0]['isolated']
        s['round'] = 2
        s['log'].append(dict(groups=[dict(label='c2a')]))
        p.enqueue()
        p.close()
    assert s['training_jobs'][1]['status'] == 'done'
    assert a.rollout_policy == tmp_path / 'initial.pt'


@pytest.mark.parametrize('problem', ['missing', 'wrong_seed', 'unverified'])
def test_bad_demo_configuration_retires_only_its_cell_before_dispatch(tmp_path, problem):
    archive = tmp_path / 'demo'
    if problem != 'missing':
        archive.mkdir()
        (archive / 'meta.json').write_text(json.dumps(
            dict(verified=problem != 'unverified', task=dict(seed=99))))
    manifest = tmp_path / 'tasks.json'
    manifest.write_text(json.dumps(dict(tasks=[dict(
        task_id='bad', level=1, seeds=[42], demo_start=dict(
            archive=str(archive), decisions=[100, 50], future_seeds=[900, 901]))])))
    specs, manifests = q.load_demo_start_cells(manifest)
    s = state()
    s['cells']['bad'] = specs['bad']
    tasks = dict(bad=dict(seeds=[42]), good=dict(seeds=[1, 2]))
    manifests['good'] = manifest
    jobs, changes = q.plan_round(s, tasks, {}, [1.], manifests, 4, 2)
    assert [job['base'] for job in jobs] == ['good']
    assert 'bad' in s['retired_cells'] and 'bad' not in s['cells']
    assert 'configuration invalid' in s['cell_errors']['bad']
    assert any('cell retired: bad:' in line for line in changes)
    # A restart must leave the configuration error retired.
    q.reconcile_start_cells(s, specs)
    assert 'bad' not in s['cells']

