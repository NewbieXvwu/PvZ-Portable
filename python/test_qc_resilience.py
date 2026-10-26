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
                           round_games=4, games_per_group=2)


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
    # An explicitly extended block can be resumed without wrapping the old one.
    spec['seeds'] += [256, 257]
    q.reconcile_start_cells(s, {'bad': spec})
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
    from test_curriculum_pipeline import setup, write_evaluation, write_selection
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
        if log.name == 'selection.log':
            write_selection(self, cmd)
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
