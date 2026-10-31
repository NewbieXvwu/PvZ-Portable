"""task49: overlap, group-boundary publication, memory headroom and durable failures."""
import json
import shutil
import sys
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'python')]
import curriculum_queue as queue
from pvz_recipe import write_recipe
from pvz_promotion import GATE_VERSION, decide, freeze_panel
from promotion_gate import contract_key


@pytest.fixture(autouse=True)
def serial_mock_trainers(monkeypatch):
    monkeypatch.setattr(queue, 'gpu_free_bytes', lambda: None)


def setup(root):
    recipe = root / 'recipe.json'
    write_recipe(recipe, dict(schema_version=1, kind='training', data=str(root / 'frames'),
                             labels=['base'], out=str(root / 'unused'), epochs=2, precompute_workers=2))
    args = SimpleNamespace(state=root / 'state.json', log=root / 'curriculum.log', frames=root / 'frames',
                           recipe=recipe, python=Path(sys.executable), rollout_policy=root / 'initial.pt',
                           workers=2, stop_flag=root / 'stop', out=root / 'runs',
                           executable=root / 'engine', value=root / 'value.pt', resource_dir=root / 'resources',
                           promotion_panel=root / 'panel.json', beta_threshold=0.1, starved_beta=0.5,
                           probe_interval=3, seed_ledger=root / 'gate_ledger.json')
    args.rollout_policy.write_bytes(b'initial')
    args.alternate_recipe = recipe
    args.executable.write_bytes(b'engine')
    args.value.write_bytes(b'value')
    args.resource_dir.mkdir()
    # The pipeline runs the current gate version, so its panel is the committed v5
    # adventure panel; older-version tests write their own v2/v3 panel.
    args.promotion_panel.write_text((ROOT / 'configs/gate_v5_panel.json').read_text())
    for label in ('base', 'c1a_fast_search', 'c1b_fast_search', 'c2a_fast_search', 'c2b_fast_search'):
        archive = args.frames / 'seed_1' / label
        archive.mkdir(parents=True)
        (archive / 'frames.jsonl').write_text('{}\n')
    cells = {name: dict(windows={}) for name in ('a', 'b')}
    state = dict(round=0, log=[], cells=cells, offsets={}, student_corpus_archives=[])
    return args, state


def certificate(pipeline, job, approved=True):
    job.setdefault('recipe', str(pipeline.args.recipe))
    trained = Path(job['recipe']).parent / 'trained.json'
    if not trained.exists():
        trained.write_text(json.dumps(dict(candidates=[dict(recipe_family='A', model=job['model'], status='trained')])))
    frozen = pipeline.evaluation_contract(job)
    manifest = freeze_panel(json.loads(pipeline.args.promotion_panel.read_text()),
        seed_block=frozen['seed_block'], calibration=frozen['calibration'],
        demos=json.loads((ROOT / 'configs/human_demo_tasks.json').read_text()))
    pairs = [dict(task_id=t['task_id'], zone=manifest['groups'][t['task_id']], seed=seed,
                  candidate=approved, incumbent=not approved, endpoint=t['endpoint'],
                  candidate_wave=20 if approved else 0, incumbent_wave=0 if approved else 20, wave_count=20)
             for t in manifest['tasks'] for seed in t.get('gate_seeds', t['seeds'])]
    result = decide(pairs, manifest, 4)
    winner = next(iter(frozen['candidates'])) if approved else None
    return dict(result, contract=frozen, contract_key=contract_key(frozen), paired=pairs,
        winner=frozen['candidates'].get(winner), winner_family=winner,
        sequential=dict(look=4, stopped=True))


def write_evaluation(pipeline, command, approved=True):
    out = Path(command[command.index('--out') + 1])
    out.mkdir(parents=True, exist_ok=True)
    job = dict(model=command[command.index('--candidate') + 1], recipe=str(out.parent.parent / 'recipe.json'))
    job['seed_block'] = [int(x) for x in command[command.index('--seed-block') + 1].split(',')]
    (out / 'decision.json').write_text(json.dumps(certificate(pipeline, job, approved)))


def retried_success(self, command, logfile):
    """_command stand-in: the retry trains and gates without an engine."""
    if logfile.name == 'train.log':
        model = logfile.parent / 'student/model.pt'
        model.parent.mkdir(parents=True)
        model.write_bytes(b'retried')
    elif logfile.name == 'evaluation.log':
        write_evaluation(self, command)


def plan(round_index):
    return [dict(base=name, multiplier=1., task_id=name, offset=(round_index - 1) * 8,
                 games=8, manifest='/manifest', label=f'c{round_index}{name}') for name in ('a', 'b')]


def test_collection_overlaps_both_phases_and_switches_only_between_groups():
    with tempfile.TemporaryDirectory() as folder:
        args, state = setup(Path(folder))
        pack_started, release_pack, train_started, release_train, eval_started, release_eval = [threading.Event() for _ in range(6)]
        seen = []
        def background(self, command, logfile):
            if 'round_000001' in str(logfile):
                if logfile.name == 'pack.log':
                    pack_started.set()
                    assert release_pack.wait(5)
                elif logfile.name == 'train.log':
                    train_started.set()
                    assert release_train.wait(5)
                elif logfile.name == 'evaluation.log':
                    eval_started.set()
                    assert release_eval.wait(5)
            if logfile.name == 'train.log':
                model = logfile.parent / 'student/model.pt'
                model.parent.mkdir(parents=True)
                model.write_bytes(b'trained')
            if logfile.name == 'evaluation.log':
                write_evaluation(self, command)
        def command(_args, manifest, task, offset, games, label, **kwargs):
            return [label, str(_args.rollout_policy)]
        def collect(command, **kwargs):
            label, student = command
            seen.append((label, student))
            if label == 'c2a':
                assert pack_started.wait(5)
                release_pack.set()
                assert train_started.wait(5)
                release_train.set()
                assert eval_started.wait(5)
                assert args.rollout_policy == Path(folder) / 'initial.pt'
                release_eval.set()
                args.pipeline.futures[0][1].result(timeout=5)
            out = args.out / f'{label}_8'
            out.mkdir(parents=True)
            (out / 'games.jsonl').write_text('\n'.join(json.dumps({'won': True}) for _ in range(8)))
            return SimpleNamespace(returncode=0)
        with patch.object(queue.TrainingPipeline, '_command', background), \
             patch.object(queue, 'mem_available_bytes', return_value=64 * queue.GIB), \
             patch.object(queue, 'command_for', command), patch.object(queue.subprocess, 'run', collect):
            args.pipeline = queue.TrainingPipeline(args, state)
            try:
                assert queue.run_round(args, state, plan(1), [], queue.GIB)
                assert queue.run_round(args, state, plan(2), [], queue.GIB)
            finally:
                args.pipeline.close()
        assert seen[2][1] == str(Path(folder) / 'initial.pt')
        assert seen[3][1].endswith('round_000001/student/model.pt')
        assert [j['status'] for j in state['training_jobs']] == ['done', 'done']
        assert state['student_round'] == 2
        assert state['log'][1]['groups'][0]['student_round'] == 0
        assert state['log'][1]['groups'][1]['student_round'] == 1
        # A restart trusts the completion marker and does not train the successful snapshot again.
        with patch.object(queue.TrainingPipeline, '_command', side_effect=AssertionError('duplicate training')):
            pipeline = queue.TrainingPipeline(args, state)
            pipeline.enqueue()
            pipeline.close()


def test_background_failure_is_saved_and_can_be_retried():
    with tempfile.TemporaryDirectory() as folder:
        args, state = setup(Path(folder))
        state.update(round=1, log=[dict(groups=[dict(label='c1a')])])
        with patch.object(queue.TrainingPipeline, '_command', side_effect=RuntimeError('pack failed')), \
             patch.object(queue, 'mem_available_bytes', return_value=64 * queue.GIB):
            pipeline = queue.TrainingPipeline(args, state)
            pipeline.enqueue()
            pipeline.close()   # task70 B7: one failure during the final drain is logged, not raised
        saved = json.loads(args.state.read_text())
        assert saved['round'] == 1 and saved['training_jobs'][0]['status'] == 'failed'
        assert 'pack failed' in saved['training_jobs'][0]['error']
        with patch.object(queue.TrainingPipeline, '_command', retried_success), \
             patch.object(queue, 'mem_available_bytes', return_value=64 * queue.GIB):
            pipeline = queue.TrainingPipeline(args, saved)
            pipeline.close()
        assert saved['training_jobs'][0]['status'] == 'done'


def test_memory_forecast_counts_only_background_headroom():
    with tempfile.TemporaryDirectory() as folder:
        args, state = setup(Path(folder))
        pipeline = queue.TrainingPipeline(args, state)
        try:
            assert queue.worker_count(16 * queue.GIB, queue.GIB, 16, background=4 * queue.GIB) == 9
            pending = SimpleNamespace(done=lambda: False)
            pipeline.futures.append(({}, pending))
            pipeline.process = SimpleNamespace(pid=123)
            with patch.object(queue, '_proc_table', return_value={123: (1, 6 * queue.GIB), 124: (123, 6 * queue.GIB)}), \
                    patch.object(queue, '_anon_bytes', return_value=queue.GIB // 2):
                assert pipeline.reservation() == queue.GIB
            pipeline.futures.clear()
        finally:
            pipeline.close()


def test_candidate_transition_keeps_collection_workers():
    with tempfile.TemporaryDirectory() as folder:
        args, state = setup(Path(folder))
        peak, resident = 2652 * 1024 ** 2, 1096 * 1024 ** 2
        state['training_jobs'] = [dict(round=1, status='done', recipe=str(args.recipe),
            model=str(args.rollout_policy), timing=dict(train_peak_anon_bytes=peak,
                                                       train_data_bytes=12 * queue.GIB))]
        pipeline = queue.TrainingPipeline(args, state)
        try:
            pipeline.futures.append(({}, SimpleNamespace(done=lambda: False)))
            pipeline.process = SimpleNamespace(pid=123)
            with patch.object(queue, '_proc_table', return_value={123: (1, 14 * queue.GIB)}), \
                    patch.object(queue, '_anon_bytes', return_value=resident):
                assert pipeline.reservation() == peak - resident
                pipeline.process = None
                assert pipeline.reservation() == peak
                assert queue.worker_count(22 * queue.GIB, queue.GIB, 16,
                                          pipeline.reservation()) == 16
                pipeline.process = SimpleNamespace(pid=123)
                assert pipeline.reservation() == peak - resident
            pipeline.futures.clear()
        finally:
            pipeline.close()


@pytest.mark.parametrize('requested,workers,expected', [(4, 16, 4), (6, 16, 6), (1, 16, 1),
                                                       (4, 2, 1), (4, 1, 1)])
def test_min_eval_units_reserves_slots_and_memory(tmp_path, requested, workers, expected):
    args, state = setup(tmp_path)
    args.workers, args.min_eval_units = workers, requested
    pipeline = queue.TrainingPipeline(args, state)
    try:
        assert pipeline.foreground_limit() == workers  # idle queue reserves nothing
        pipeline.futures.append(({}, SimpleNamespace(done=lambda: False)))
        pipeline.per_worker_bytes = queue.GIB
        assert pipeline.foreground_limit() == max(1, workers - expected)
        with patch.object(pipeline, 'estimate_peak', return_value=queue.GIB // 2):
            assert pipeline.reservation() == expected * queue.GIB
        pipeline.foreground_workers = max(1, workers - expected)
        with patch.object(queue, 'mem_available_bytes', return_value=20 * queue.GIB):
            assert pipeline._evaluation_units() == expected
    finally:
        pipeline.futures.clear()
        pipeline.close()


def test_min_eval_units_waits_for_old_group_and_memory(tmp_path):
    args, state = setup(tmp_path)
    args.workers = 16  # default min_eval_units=4, including historical jobs
    pipeline = queue.TrainingPipeline(args, state)
    pipeline.per_worker_bytes = queue.GIB
    pipeline.foreground_workers = 16
    available = [20 * queue.GIB]
    waits = []

    def release(_):
        waits.append(1)
        assert len(waits) <= 2
        pipeline.foreground_workers = 12
        available[0] = (queue.MEMORY_RESERVE_BYTES + (3 if len(waits) == 1 else 4) * queue.GIB)

    try:
        with patch.object(queue, 'mem_available_bytes', side_effect=lambda: available[0]), \
             patch.object(queue.time, 'sleep', side_effect=release):
            assert pipeline._evaluation_units() == 4
        assert len(waits) == 2
        assert 'min_units=4 free_slots=1' in args.log.read_text()
        with patch.object(queue, 'mem_total_bytes', return_value=6 * queue.GIB):
            with pytest.raises(RuntimeError, match='evaluation cannot fit 4 units'):
                pipeline._evaluation_units()
    finally:
        pipeline.close()


def test_min_eval_units_first_group_preserves_memory_and_growth_slots(tmp_path):
    args, state = setup(tmp_path)
    args.workers = 16
    args.pipeline = pipeline = queue.TrainingPipeline(args, state)
    pipeline.futures.append(({}, SimpleNamespace(done=lambda: False)))
    item = plan(1)[0]

    def command(*a, **kw):
        # 16 available - 3 reserve - 4 evaluation = 9 at launch; later growth stops at12.
        assert kw['workers'] == 9
        assert kw['upper_workers'] == pipeline.foreground_workers == 12
        assert kw['headroom_bytes'] == 4 * queue.GIB
        return ['synthetic']

    def collect(*a, **kw):
        out = args.out / f'{item["label"]}_8'
        out.mkdir(parents=True)
        (out / 'games.jsonl').write_text('{"won":true}\n' * 8)
        return SimpleNamespace(returncode=0)

    try:
        with patch.object(queue, 'mem_available_bytes', return_value=16 * queue.GIB), \
             patch.object(pipeline, 'estimate_peak', return_value=queue.GIB), \
             patch.object(queue, 'command_for', side_effect=command), \
             patch.object(queue.subprocess, 'run', side_effect=collect):
            queue._run_group(args, state, item, 1, queue.GIB)
        assert pipeline.foreground_workers == 0
    finally:
        pipeline.futures.clear()
        pipeline.close()


def test_recipe_and_collection_sources_merge_without_copying_or_dropping_labels():
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        args, state = setup(root)
        base = root / 'recipe_frames/seed_2/base'
        base.mkdir(parents=True)
        (base / 'frames.jsonl').write_text('{"from_recipe":true}\n')
        recipe = json.loads(args.recipe.read_text())
        recipe['data'] = str(root / 'recipe_frames')
        write_recipe(args.recipe, recipe)
        pipeline = queue.TrainingPipeline(args, state)
        try:
            pipeline._prepare_corpus(['base', 'c1a_fast_search'])
            view = pipeline.directory / 'frames'
            assert (view / 'seed_2/base').is_symlink()
            assert (view / 'seed_2/base').resolve() == base.resolve()
            assert (view / 'seed_1/base').resolve() == (args.frames / 'seed_1/base').resolve()
            assert (view / 'seed_1/c1a_fast_search/frames.jsonl').read_bytes() == b'{}\n'
            try:
                pipeline._prepare_corpus(['missing'])
                assert False, 'missing labels must fail'
            except RuntimeError as error:
                assert 'missing' in str(error)
        finally:
            pipeline.close()


def test_rejection_keeps_student_and_next_round_still_accumulates_data():
    with tempfile.TemporaryDirectory() as folder:
        args, state = setup(Path(folder))
        state.update(round=1, log=[dict(round=1, groups=[dict(label='c1a')])])
        original = args.rollout_policy
        def background(self, command, logfile):
            if logfile.name == 'train.log':
                model = logfile.parent / 'student/model.pt'
                model.parent.mkdir(parents=True)
                model.write_bytes(b'weak')
            if logfile.name == 'evaluation.log':
                write_evaluation(self, command, approved=False)
        def collect(command, **kwargs):
            label = command[0]
            out = args.out / f'{label}_8'
            out.mkdir(parents=True)
            (out / 'games.jsonl').write_text('\n'.join(json.dumps({'won': False}) for _ in range(8)))
            return SimpleNamespace(returncode=0)
        with patch.object(queue.TrainingPipeline, '_command', background), \
             patch.object(queue, 'mem_available_bytes', return_value=64 * queue.GIB), \
             patch.object(queue, 'command_for', lambda a, m, t, o, g, label, **kw: [label]), \
             patch.object(queue.subprocess, 'run', collect):
            args.pipeline = queue.TrainingPipeline(args, state)
            args.pipeline.enqueue()
            args.pipeline.futures[0][1].result(timeout=5)
            args.pipeline.poll()
            assert args.rollout_policy == original and state.get('student_round', 0) == 0
            assert state['training_jobs'][0]['promotion_approved'] is False
            assert state['training_jobs'][0]['promotion_reasons']
            assert queue.run_round(args, state, plan(2), [], queue.GIB)
            args.pipeline.close()
        assert state['round'] == 2 and len(state['log']) == 2
        assert args.rollout_policy == original
        assert all(not job['promotion_approved'] for job in state['training_jobs'])


def test_stale_comparator_is_reevaluated_without_retraining_on_close():
    with tempfile.TemporaryDirectory() as folder:
        args, state = setup(Path(folder))
        state.update(round=1, log=[dict(round=1, groups=[dict(label='c1a')])])
        calls = []
        def background(self, command, logfile):
            # Seed trainers log inside seeds/seed_NNNNNN/; attribute them to the round folder.
            calls.append((logfile.parents[2].name if logfile.parent.name.startswith('seed_')
                          else logfile.parent.name, logfile.name))
            if logfile.name == 'train.log':
                model = logfile.parent / 'student/model.pt'
                model.parent.mkdir(parents=True)
                model.write_bytes(logfile.parent.name.encode())
            if logfile.name == 'evaluation.log':
                write_evaluation(self, command)
        with patch.object(queue.TrainingPipeline, '_command', background), \
             patch.object(queue, 'mem_available_bytes', return_value=64 * queue.GIB):
            pipeline = queue.TrainingPipeline(args, state)
            pipeline.enqueue()
            pipeline.futures[0][1].result(timeout=5)
            # Background can finish two snapshots before a long foreground group ends.
            state.update(round=2)
            state['log'].append(dict(round=2, groups=[dict(label='c2a')]))
            pipeline.enqueue()
            pipeline.futures[1][1].result(timeout=5)
            assert state.get('student_round', 0) == 0
            pipeline.close()
        assert state['student_round'] == 2
        assert sum(1 for c in calls if c == ('round_000001', 'train.log')) == 2
        assert sum(1 for c in calls if c == ('round_000002', 'train.log')) == 2
        assert calls.count(('round_000002', 'evaluation.log')) == 2
        assert state['training_jobs'][1]['timing']['promotion']['contract']['incumbent'].endswith('round_000001/student/model.pt')


def test_evaluation_failure_is_closed_gate_but_does_not_poison_training_queue():
    with tempfile.TemporaryDirectory() as folder:
        args, state = setup(Path(folder))
        state.update(round=1, log=[dict(round=1, groups=[dict(label='c1a')])])
        def background(self, command, logfile):
            if logfile.name == 'train.log':
                model = logfile.parent / 'student/model.pt'
                model.parent.mkdir(parents=True)
                model.write_bytes(b'trained')
            if logfile.name == 'evaluation.log':
                raise RuntimeError('evaluation engine failure')
        with patch.object(queue.TrainingPipeline, '_command', background), \
             patch.object(queue, 'mem_available_bytes', return_value=64 * queue.GIB):
            pipeline = queue.TrainingPipeline(args, state)
            pipeline.enqueue()
            pipeline.futures[0][1].result(timeout=5)
            pipeline.poll()
            assert not pipeline.failed
            state['round'] = 2
            state['log'].append(dict(round=2, groups=[dict(label='c2a')]))
            pipeline.enqueue()
            pipeline.close()
        assert all(job['status'] == 'done' and not job['promotion_approved'] for job in state['training_jobs'])
        assert all('evaluation_error:' in job['promotion_reasons'][0] for job in state['training_jobs'])
        assert args.rollout_policy == Path(folder) / 'initial.pt'


def test_all_seeds_failed_still_reaches_the_B7_retriable_failure():
    # task92: an entirely failed round keeps the pre-existing B7 retry/stop semantics.
    with tempfile.TemporaryDirectory() as folder:
        args, state = setup(Path(folder))
        state.update(round=1, log=[dict(round=1, groups=[dict(label='c1a')])])
        def background(self, command, logfile):
            if logfile.name == 'train.log':
                raise RuntimeError('every seed failed')
        with patch.object(queue.TrainingPipeline, '_command', background), \
             patch.object(queue, 'mem_available_bytes', return_value=64 * queue.GIB):
            args.pipeline = queue.TrainingPipeline(args, state)
            args.pipeline.enqueue()
            args.pipeline.close()   # B7: logged, not raised
        job = state['training_jobs'][0]
        assert job['status'] == 'failed'
        assert 'all training seeds failed' in job['error']
        def success(self, command, logfile):
            if logfile.name == 'train.log':
                model = logfile.parent / 'student/model.pt'
                model.parent.mkdir(parents=True)
                model.write_bytes(b'retried')
            elif logfile.name == 'evaluation.log':
                write_evaluation(self, command)
        with patch.object(queue.TrainingPipeline, '_command', retried_success), \
             patch.object(queue, 'mem_available_bytes', return_value=64 * queue.GIB):
            args.pipeline = queue.TrainingPipeline(args, state)
            args.pipeline.close()
        assert state['training_jobs'][0]['status'] == 'done'
