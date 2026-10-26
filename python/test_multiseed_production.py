"""task92c: parallel trainers, fallback, durable panels, fresh locked seed blocks."""
import copy
import json
from pathlib import Path
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]
import curriculum_queue as queue
from pvz_selection_refresh import make_plan, SELECTION_POOLS, SELECTION_BLOCK_SIZE
from pvz_student_selection import assert_disjoint, seed_spans
from promotion_gate import allocate_seed_block
from test_curriculum_pipeline import setup, write_evaluation, write_selection


@pytest.fixture(autouse=True)
def no_game_engine(monkeypatch):
    monkeypatch.setattr(queue.TrainingPipeline, '_validate_selection_panel', lambda self, panel: panel)
    monkeypatch.setattr(queue, 'gpu_free_bytes', lambda: 16 * queue.GIB)
    monkeypatch.setattr(queue, 'mem_available_bytes', lambda: 64 * queue.GIB)
    monkeypatch.setattr(queue, 'mem_total_bytes', lambda: 64 * queue.GIB)


def run_round(root, monkeypatch, trainer):
    args, state = setup(root)
    state.update(round=1, log=[dict(round=1, groups=[dict(label='c1a')])])
    calls = []
    def command(self, command, logfile):
        calls.append(logfile.name)
        if logfile.name == 'train.log':
            trainer(command, logfile)
        elif logfile.name == 'selection.log':
            write_selection(self, command)
        elif logfile.name == 'evaluation.log':
            write_evaluation(self, command)
        return queue.GIB
    monkeypatch.setattr(queue.TrainingPipeline, '_command', command)
    pipeline = queue.TrainingPipeline(args, state)
    pipeline.enqueue()
    pipeline.futures[0][1].result(timeout=10)
    pipeline.poll()
    pipeline.close()
    return args, state['training_jobs'][0], calls


def save_model(command, logfile):
    model = logfile.parent / 'student/model.pt'
    model.parent.mkdir(parents=True, exist_ok=True)
    model.write_bytes(logfile.parent.name.encode())


def test_two_trainers_overlap_failure_is_local_and_selection_precedes_gate(tmp_path, monkeypatch):
    barrier = threading.Barrier(2)
    threads = set()
    def train(command, logfile):
        threads.add(threading.get_ident())
        barrier.wait(timeout=5)  # a serial implementation deadlocks and fails this test
        if logfile.parent.name == 'seed_000002':
            raise RuntimeError('one seed fails')
        save_model(command, logfile)
    args, job, calls = run_round(tmp_path, monkeypatch, train)
    assert len(threads) == 2
    assert job['timing']['training_mode']['mode'] == 'parallel'
    assert [c['status'] for c in job['timing']['candidates']] == ['trained', 'failed']
    assert calls.index('selection.log') < calls.index('evaluation.log')
    assert job['promotion_approved']
    folder = Path(job['recipe']).parent
    assert (folder / 'training_mode.json').is_file()
    assert (folder / 'candidates.json').is_file()
    assert (folder / 'seeds/seed_000002/failed.json').is_file()


@pytest.mark.parametrize('resource', ['host_memory', 'gpu'])
def test_insufficient_resources_fall_back_to_serial(tmp_path, monkeypatch, resource):
    threads = set()
    if resource == 'host_memory':
        monkeypatch.setattr(queue, 'mem_available_bytes', lambda: 9 * queue.GIB)
    else:
        monkeypatch.setattr(queue, 'gpu_free_bytes', lambda: 3 * queue.GIB)  # one fits, two do not
    def train(command, logfile):
        threads.add(threading.get_ident())
        save_model(command, logfile)
    args, job, _ = run_round(tmp_path, monkeypatch, train)
    mode = job['timing']['training_mode']
    assert mode['mode'] == 'serial' and mode['workers'] == 1
    assert mode['fallback_reasons']
    assert len(threads) == 1
    assert len(job['timing']['selection']['rejected']) == 1
    assert (Path(job['recipe']).parent / 'student/selection.json').is_file()


def test_prepared_panel_rejects_an_altered_round_seed_allocation(tmp_path, monkeypatch):
    args, job, _ = run_round(tmp_path, monkeypatch, save_model)
    changed = copy.deepcopy(job)
    changed['selection_refresh']['seed_block'] = [value + 384 for value in changed['selection_refresh']['seed_block']]
    pipeline = object.__new__(queue.TrainingPipeline)
    pipeline.args = args
    with pytest.raises(ValueError, match='preparation changed'):
        pipeline._prepare_selection(changed)


def test_each_round_allocates_once_and_panels_never_overlap(tmp_path, monkeypatch):
    args, state = setup(tmp_path)
    state.update(round=1, log=[dict(round=1, groups=[dict(label='c1a')])])
    def command(self, command, logfile):
        if logfile.name == 'train.log':
            save_model(command, logfile)
        elif logfile.name == 'selection.log':
            write_selection(self, command)
        elif logfile.name == 'evaluation.log':
            write_evaluation(self, command)
        return queue.GIB
    monkeypatch.setattr(queue.TrainingPipeline, '_command', command)
    pipeline = queue.TrainingPipeline(args, state)
    try:
        pipeline.enqueue()
        pipeline.enqueue()  # same round, no second allocation
        pipeline.futures[0][1].result(timeout=10)
        pipeline.poll()
        state.update(round=2)
        state['log'].append(dict(round=2, groups=[dict(label='c2a')]))
        pipeline.enqueue()
        pipeline.futures[1][1].result(timeout=10)
        pipeline.poll()
    finally:
        pipeline.close()
    jobs = state['training_jobs']
    first, second = [json.loads(Path(job['selection_manifest']).read_text()) for job in jobs]
    assert_disjoint(first, list(seed_spans(second)))
    ledger = json.loads((tmp_path / 'selection_seed_ledger.json').read_text())
    assert len(ledger['allocated']) == 2
    assert all(high - low == SELECTION_BLOCK_SIZE for low, high in ledger['allocated'])
    assert all((Path(job['recipe']).parent / 'selection_state.json').is_file() for job in jobs)
    assert all((Path(job['recipe']).parent / 'selection_plan.json').is_file() for job in jobs)


def test_current_incumbent_windows_refresh_eligible_cells(tmp_path):
    model = tmp_path / 'new_incumbent.pt'
    model.write_bytes(b'new policy')
    source = '/original_panel.json'
    task = dict(task_id='t', level=7, deck=[0, 1, 2, 3, 4, 5], seeds=[30000])
    templates = json.loads((ROOT / 'configs/student_selection_v2.json').read_text())
    def state_with(label, wins):
        folder = tmp_path / f'{label}_2'
        folder.mkdir()
        (folder / 'config.json').write_text(json.dumps(dict(rollout_policy=str(model), dagger_beta=0, task_id='t')))
        (folder / 'games.jsonl').write_text('\n'.join(json.dumps(dict(won=bool(v), terminal=True)) for v in wins))
        group = dict(base='t', multiplier=1, beta=0, student=str(model), label=label, games=2,
                     task_id='t', manifest=source)
        return dict(cells=dict(t=dict(windows={'1.0': dict(recent=wins)})), log=[dict(groups=[group])])
    ledger = tmp_path / 'ledger.json'
    first_block = allocate_seed_block(ledger, 384, pools=SELECTION_POOLS)
    second_block = allocate_seed_block(ledger, 384, pools=SELECTION_POOLS)
    first = make_plan(state_with('one', [1, 0]), model, tmp_path, templates, {source: dict(tasks=[task])}, first_block)
    second = make_plan(state_with('two', [1, 1]), model, tmp_path, templates, {source: dict(tasks=[task])}, second_block)
    assert len(first['cells']) == 1 and len(second['cells']) == 0
    assert len(first['tasks']) == 4 and len(second['tasks']) == 3
    assert_disjoint(first, list(seed_spans(second)))


def test_locked_selection_ledger_handles_simultaneous_allocations(tmp_path):
    ledger = tmp_path / 'ledger.json'
    with ThreadPoolExecutor(max_workers=2) as executor:
        blocks = list(executor.map(lambda _: allocate_seed_block(ledger, 384, pools=SELECTION_POOLS), range(2)))
    assert blocks[0] != blocks[1]
    assert sorted(blocks)[0][1] <= sorted(blocks)[1][0]


def test_selection_retry_reuses_models_panel_and_preserves_actual_mode(tmp_path, monkeypatch):
    args, state = setup(tmp_path)
    state.update(round=1, log=[dict(round=1, groups=[dict(label='c1a')])])
    counts = dict(train=0, selection=0)
    def command(self, command, logfile):
        if logfile.name == 'train.log':
            counts['train'] += 1
            save_model(command, logfile)
        elif logfile.name == 'selection.log':
            counts['selection'] += 1
            if counts['selection'] == 1:
                raise RuntimeError('selection interrupted')
            write_selection(self, command)
        elif logfile.name == 'evaluation.log':
            write_evaluation(self, command)
        return queue.GIB
    monkeypatch.setattr(queue.TrainingPipeline, '_command', command)
    pipeline = queue.TrainingPipeline(args, state)
    try:
        pipeline.enqueue()
        with pytest.raises(RuntimeError, match='selection interrupted'):
            pipeline.futures[0][1].result(timeout=10)
        pipeline.poll()
        pipeline.futures[0][1].result(timeout=10)
        pipeline.poll()
    finally:
        pipeline.close()
    job = state['training_jobs'][0]
    assert job['status'] == 'done' and counts == dict(train=2, selection=2)
    assert [mode['mode'] for mode in job['timing']['training_mode']['attempt_history']] == ['parallel', 'reused']
    assert len(json.loads((tmp_path / 'selection_seed_ledger.json').read_text())['allocated']) == 1


def test_reservation_counts_both_live_training_trees(tmp_path, monkeypatch):
    from types import SimpleNamespace
    args, state = setup(tmp_path)
    pipeline = queue.TrainingPipeline(args, state)
    pipeline.futures.append(({}, SimpleNamespace(done=lambda: False)))
    pipeline.last_train_peak, pipeline.train_parallelism = 3 * queue.GIB, 2
    pipeline.processes = {1: SimpleNamespace(pid=1), 3: SimpleNamespace(pid=3)}
    monkeypatch.setattr(queue, '_proc_table', lambda: {1: (0, queue.GIB), 2: (1, queue.GIB),
                                                     3: (0, queue.GIB), 4: (3, queue.GIB)})
    try:
        assert pipeline.estimate_peak() == 6 * queue.GIB
        assert pipeline.reservation() == 2 * queue.GIB
    finally:
        pipeline.futures.clear()
        pipeline.close()


def test_resources_changing_during_admission_still_fall_back(tmp_path, monkeypatch):
    args, state = setup(tmp_path)
    pipeline = queue.TrainingPipeline(args, state)
    values = iter([64 * queue.GIB, 9 * queue.GIB])
    monkeypatch.setattr(queue, 'mem_available_bytes', lambda: next(values))
    pipeline.train_parallelism, pipeline.training_admission = 2, True
    try:
        pipeline._admit()
        assert pipeline.train_parallelism == 1
        assert 'resources_changed_before_launch' in pipeline.train_admission_record['fallback_reasons']
    finally:
        pipeline.close()


def test_controller_dry_run_never_allocates_or_starts_a_process(tmp_path, monkeypatch, capsys):
    args, state = setup(tmp_path)
    args.state.write_text(json.dumps(state))
    before = args.state.read_bytes()
    selection_ledger, gate_ledger = tmp_path / 'selection.json', tmp_path / 'gate.json'
    monkeypatch.setattr(queue, 'set_oom_score_adj', lambda _: None)
    monkeypatch.setattr(queue, 'load_panel', lambda _: ({}, [1.], {}, {}))
    monkeypatch.setattr(queue, 'reconcile_cells', lambda *a: [])
    monkeypatch.setattr(queue, 'load_teacher_rates', lambda _: {'a': .5})
    monkeypatch.setattr(queue, 'plan_round', lambda *a: ([], []))
    def forbidden(*args, **kwargs):
        raise AssertionError('dry-run must not start a process')
    monkeypatch.setattr(queue.subprocess, 'Popen', forbidden)
    monkeypatch.setattr(sys, 'argv', ['curriculum_queue.py', '--dry-run',
        '--manifest', str(args.promotion_panel), '--promotion-panel', str(args.promotion_panel),
        '--state', str(args.state), '--log', str(args.log), '--recipe', str(args.recipe),
        '--frames', str(args.frames), '--out', str(args.out), '--rollout-policy', str(args.rollout_policy),
        '--executable', str(args.executable), '--resource-dir', str(args.resource_dir),
        '--demo-start-manifest', str(tmp_path / 'absent.json'), '--selection-seed-ledger', str(selection_ledger),
        '--seed-ledger', str(gate_ledger)])
    queue.main()
    assert args.state.read_bytes() == before
    assert not selection_ledger.exists() and not gate_ledger.exists()
    assert 'selection preview (no allocation, no engine, no state write)' in capsys.readouterr().out


def test_window_that_disagrees_with_incumbent_records_sits_out(tmp_path):
    # 10-26 production dry-run: six windows held games the log could not attribute; one bad
    # window must drop that cell from the panel, not abort the whole selection.
    from pvz_selection_refresh import current_cells
    model = tmp_path / 'r11.pt'
    model.write_bytes(b'r11')
    folder = tmp_path / 'g_2'
    folder.mkdir()
    (folder / 'config.json').write_text(json.dumps(dict(rollout_policy=str(model), dagger_beta='0', task_id='t')))
    (folder / 'games.jsonl').write_text('\n'.join(json.dumps(dict(won=bool(w), terminal=True)) for w in [1, 0]))
    group = dict(base='t', multiplier=1., beta=0, student=str(model.resolve()), label='g', games=2,
                 task_id='t', manifest='/m.json')
    failed = dict(group, label='failed', status='failed')
    broken = tmp_path / 'failed_2'
    broken.mkdir()
    (broken / 'games.jsonl').write_text('broken')
    state = dict(log=[dict(groups=[group, failed, dict(failed, label='missing')])],
                 cells=dict(t=dict(windows={'1.0': dict(recent=[1, 1, 1, 0])})))
    cells, _, excluded = current_cells(state, model, tmp_path)
    assert [c['rate'] for c in cells] == [.5]
    state['cells']['t']['windows']['1.0']['recent'][-1] = 1
    cells, _, excluded = current_cells(state, model, tmp_path)
    assert cells == []
    assert excluded[-1]['reason'] == 'current-policy records do not match window suffix'
