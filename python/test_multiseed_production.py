"""task92c: parallel trainers, fallback, durable panels, fresh locked seed blocks."""
import json
from pathlib import Path
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]
import curriculum_queue as queue
from pvz_selection_refresh import make_plan, selection_templates, SELECTION_POOLS, SELECTION_BLOCK_SIZE
from pvz_student_selection import assert_disjoint, seed_spans
from promotion_gate import allocate_seed_block
from test_curriculum_pipeline import setup, write_evaluation, write_selection


@pytest.fixture(autouse=True)
def no_game_engine(monkeypatch):
    monkeypatch.setattr(queue, 'gpu_free_bytes', lambda: 16 * queue.GIB)
    monkeypatch.setattr(queue, 'mem_available_bytes', lambda: 64 * queue.GIB)
    monkeypatch.setattr(queue, 'mem_total_bytes', lambda: 64 * queue.GIB)


def run_round(root, monkeypatch, trainer, alternate=None):
    args, state = setup(root)
    args.alternate_recipe = alternate
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


def test_two_recipes_share_one_pack_and_record_each_selection_score(tmp_path, monkeypatch):
    from pvz_recipe import load_recipe, write_recipe
    alternate = tmp_path / 'balanced.json'
    write_recipe(alternate, dict(schema_version=1, kind='training', data=str(tmp_path / 'frames'),
                                labels=['base'], out=str(tmp_path / 'unused'), epochs=2,
                                precompute_workers=2, source_weight=dict(level_balance=.25)))
    seen = []
    def train(command, logfile):
        seen.append(load_recipe(Path(command[command.index('--recipe') + 1])))
        save_model(command, logfile)
    _, job, calls = run_round(tmp_path, monkeypatch, train, alternate)
    assert calls.count('pack.log') == 1
    assert sorted(recipe['source_weight'].get('level_balance', 0) for recipe in seen) == [0, .25]
    assert len({recipe['data'] for recipe in seen}) == 1
    candidates = job['timing']['candidates']
    assert [candidate['selection_score']['total'] for candidate in candidates] == [10, 9]
    assert candidates[1]['source_recipe'] == str(alternate)
    assert job['promotion_approved']


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


@pytest.mark.parametrize('resource', ['host_memory', 'gpu'])
def test_insufficient_resources_fall_back_to_serial(tmp_path, monkeypatch, resource):
    threads = set()
    if resource == 'host_memory':
        monkeypatch.setattr(queue, 'mem_available_bytes', lambda: 6 * queue.GIB)
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
    assert all((Path(job['recipe']).parent / 'selection_plan.json').is_file() for job in jobs)


def test_consecutive_rounds_score_the_adventure_body_on_disjoint_seeds(tmp_path):
    # task128: the panel no longer follows the incumbent's windows; every round
    # re-plans the same adventure body on a fresh block from the selection ledger.
    model = tmp_path / 'new_incumbent.pt'
    model.write_bytes(b'new policy')
    templates = selection_templates(ROOT / 'configs/student_selection_holdout.json',
                                    ROOT / 'configs/gate_v5_panel.json')
    ledger = tmp_path / 'ledger.json'
    first = make_plan(templates, allocate_seed_block(ledger, SELECTION_BLOCK_SIZE, pools=SELECTION_POOLS), model)
    second = make_plan(templates, allocate_seed_block(ledger, SELECTION_BLOCK_SIZE, pools=SELECTION_POOLS), model)
    assert len(first['adventure_tasks']) == 38
    assert len(first['tasks']) == 48                      # 38 adventure + 8 random + 2 unseen
    assert {len(task['seeds']) for task in first['tasks']} == {8, 16}
    assert sum(len(task['seeds']) for task in first['tasks']) == 456
    assert_disjoint(first, list(seed_spans(second)))
    assert first['score_only'] == first['selection_holdout_tasks']


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
    monkeypatch.setattr(queue, '_anon_bytes', lambda pid: queue.GIB)
    try:
        assert pipeline.estimate_peak() == 6 * queue.GIB
        assert pipeline.reservation() == 2 * queue.GIB
    finally:
        pipeline.futures.clear()
        pipeline.close()


def test_training_admission_uses_anonymous_peak(tmp_path, monkeypatch):
    args, state = setup(tmp_path)
    pipeline = queue.TrainingPipeline(args, state)
    pipeline.last_train_peak = int(3.5 * queue.GIB)
    pipeline.train_parallelism, pipeline.training_admission = 2, True
    monkeypatch.setattr(queue, 'mem_available_bytes', lambda: 24 * queue.GIB)
    monkeypatch.setattr(queue, 'mem_total_bytes', lambda: 25 * queue.GIB)
    try:
        assert pipeline.estimate_peak() == 7 * queue.GIB
        pipeline._admit()
        assert pipeline.train_parallelism == 2
        assert pipeline.train_admission_record['fallback_reasons'] == []
        assert pipeline.train_admission_record['required_bytes'] == 10 * queue.GIB
    finally:
        pipeline.close()


def test_restart_reserves_anonymous_peak_without_mapped_data(tmp_path):
    args, state = setup(tmp_path)
    model = tmp_path / 'seed_000001/student/model.pt'
    model.parent.mkdir(parents=True)
    (model.parent / 'metrics.json').write_text(json.dumps(dict(data_memory=dict(
        rows=19649496, board_bytes=6 * queue.GIB, scalar_bytes=2 * queue.GIB,
        packet_bytes=queue.GIB, tensor_bytes=queue.GIB))))
    state['training_jobs'] = [dict(round=1, status='done', recipe=str(args.recipe),
        model=str(tmp_path / 'selected/model.pt'), timing=dict(
            train_peak_anon_bytes=3 * queue.GIB, train_data_bytes=10 * queue.GIB,
            selection=dict(best=str(model))))]
    pipeline = queue.TrainingPipeline(args, state)
    try:
        assert pipeline.estimate_peak() == 3 * queue.GIB
    finally:
        pipeline.close()


def test_pack_size_does_not_add_to_measured_anonymous_peak(tmp_path):
    args, state = setup(tmp_path)
    pipeline = queue.TrainingPipeline(args, state)
    segments = pipeline.directory / 'frames/packs/segments'
    segments.mkdir(parents=True)
    (segments / 'one.npz').write_bytes(b'pack data')
    pipeline.last_train_peak = 3 * queue.GIB
    try:
        assert pipeline.estimate_peak() == 3 * queue.GIB
    finally:
        pipeline.close()


def test_resources_changing_during_admission_still_fall_back(tmp_path, monkeypatch):
    args, state = setup(tmp_path)
    pipeline = queue.TrainingPipeline(args, state)
    values = iter([64 * queue.GIB, 6 * queue.GIB])
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
    start_ledger = tmp_path / 'start.json'
    start_ledger.write_text(json.dumps({'pool': list(queue.START_SEED_POOL),
                                      'allocated': [[200000000, 200000256]]}))
    start_before = start_ledger.read_bytes()
    monkeypatch.setattr(queue, 'set_oom_score_adj', lambda _: None)
    monkeypatch.setattr(queue, 'load_panel', lambda *_: ({}, [1.], {}, {}))
    monkeypatch.setattr(queue, 'reconcile_cells', lambda *a: [])
    monkeypatch.setattr(queue, 'load_teacher_rates', lambda _: {'a': .5})
    def preview(*args, start_seed_ledger, **kwargs):
        assert start_seed_ledger != start_ledger
        assert start_seed_ledger.read_bytes() == start_before
        # The planner may reserve blocks, but only on the disposable ledger copy.
        queue.allocate_seed_block(start_seed_ledger, 256, pool=queue.START_SEED_POOL)
        return [], []
    monkeypatch.setattr(queue, 'plan_round', preview)
    def forbidden(*args, **kwargs):
        raise AssertionError('dry-run must not start a process')
    monkeypatch.setattr(queue.subprocess, 'Popen', forbidden)
    monkeypatch.setattr(sys, 'argv', ['curriculum_queue.py', '--dry-run',
        '--manifest', str(args.promotion_panel), '--promotion-panel', str(args.promotion_panel),
        '--state', str(args.state), '--log', str(args.log), '--recipe', str(args.recipe),
        '--frames', str(args.frames), '--out', str(args.out), '--rollout-policy', str(args.rollout_policy),
        '--executable', str(args.executable), '--resource-dir', str(args.resource_dir),
        '--demo-start-manifest', str(tmp_path / 'absent.json'), '--selection-seed-ledger', str(selection_ledger),
        '--seed-ledger', str(gate_ledger), '--start-seed-ledger', str(start_ledger)])
    queue.main()
    assert start_ledger.read_bytes() == start_before
    assert args.state.read_bytes() == before
    assert not selection_ledger.exists() and not gate_ledger.exists()
    assert 'selection preview (no allocation, no engine, no state write)' in capsys.readouterr().out
