"""task49: overlap, group-boundary publication, memory headroom and durable failures."""
import json
import sys
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'python')]
import curriculum_queue as queue
from pvz_recipe import write_recipe
from pvz_promotion import PANEL_TASKS, decide, freeze_panel
from promotion_gate import contract_key, sha


def setup(root):
    recipe = root / 'recipe.json'
    write_recipe(recipe, dict(schema_version=1, kind='training', data=str(root / 'frames'),
                             labels=['base'], out=str(root / 'unused'), epochs=2, precompute_workers=2))
    args = SimpleNamespace(state=root / 'state.json', log=root / 'curriculum.log', frames=root / 'frames',
                           recipe=recipe, python=Path(sys.executable), rollout_policy=root / 'initial.pt',
                           workers=2, stop_flag=root / 'stop', out=root / 'runs',
                           executable=root / 'engine', value=root / 'value.pt', resource_dir=root / 'resources',
                           promotion_panel=root / 'panel.json', beta_threshold=0.1, starved_beta=0.5,
                           probe_interval=3)
    args.rollout_policy.write_bytes(b'initial')
    args.executable.write_bytes(b'engine')
    args.value.write_bytes(b'value')
    args.resource_dir.mkdir()
    args.promotion_panel.write_text(json.dumps(dict(panel_version='v2',
        groups={name: zone for zone, name in PANEL_TASKS},
        tasks=[dict(task_id=name, seeds=list(range(30000, 30128)), wave_cap=None,
                    zombie_count_multiplier=1, level=1, deck=[0, 1, 2, 3, 4, 5]) for _, name in PANEL_TASKS])))
    for label in ('base', 'c1a_fast_search', 'c1b_fast_search', 'c2a_fast_search', 'c2b_fast_search'):
        archive = args.frames / 'seed_1' / label
        archive.mkdir(parents=True)
        (archive / 'frames.jsonl').write_text('{}\n')
    cells = {name: dict(windows={}) for name in ('a', 'b')}
    state = dict(round=0, log=[], cells=cells, offsets={})
    return args, state


def certificate(pipeline, job, approved=True):
    frozen = pipeline.evaluation_contract(job)
    manifest = freeze_panel(json.loads(pipeline.args.promotion_panel.read_text()))
    pairs = [dict(task_id=t['task_id'], zone=manifest['groups'][t['task_id']], seed=seed,
                  candidate=approved, incumbent=not approved) for t in manifest['tasks'] for seed in t['seeds']]
    return {**decide(pairs, manifest), 'contract': frozen, 'contract_key': contract_key(frozen), 'paired': pairs}


def write_evaluation(pipeline, command, approved=True):
    out = Path(command[command.index('--out') + 1])
    out.mkdir(parents=True, exist_ok=True)
    job = dict(model=command[command.index('--candidate') + 1])
    report = certificate(pipeline, job, approved)
    manifest = freeze_panel(json.loads(pipeline.args.promotion_panel.read_text()))
    files = {}
    for task in manifest['tasks']:
        for side, won in [('candidate', approved), ('incumbent', not approved)]:
            folder = out / side / task['task_id']
            folder.mkdir(parents=True, exist_ok=True)
            (folder / 'config.json').write_text(json.dumps(dict(rollout_policy=report['contract'][side],
                task_id=task['task_id'], arms='fast', dagger_beta='0',
                executable=report['contract']['executable'], workers='1')))
            (folder / 'games.jsonl').write_text('\n'.join(json.dumps(dict(seed=seed, arm='fast', searched=0, won=won))
                                                           for seed in task['seeds']))
            files[str((folder / 'games.jsonl').relative_to(out))] = sha(folder / 'games.jsonl')
    report['episode_files_sha256'] = files
    (out / 'decision.json').write_text(json.dumps(report))


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
            try:
                pipeline.close()
                assert False, 'failure must reach the controller'
            except RuntimeError as error:
                assert 'pack failed' in str(error)
        saved = json.loads(args.state.read_text())
        assert saved['round'] == 1 and saved['training_jobs'][0]['status'] == 'failed'
        def success(self, command, logfile):
            if logfile.name == 'train.log':
                model = logfile.parent / 'student/model.pt'
                model.parent.mkdir(parents=True)
                model.write_bytes(b'retried')
            if logfile.name == 'evaluation.log':
                write_evaluation(self, command)
        with patch.object(queue.TrainingPipeline, '_command', success), \
             patch.object(queue, 'mem_available_bytes', return_value=64 * queue.GIB):
            pipeline = queue.TrainingPipeline(args, saved)
            pipeline.close()
        assert saved['training_jobs'][0]['status'] == 'done'


def test_memory_forecast_counts_only_background_headroom():
    with tempfile.TemporaryDirectory() as folder:
        args, state = setup(Path(folder))
        pipeline = queue.TrainingPipeline(args, state)
        try:
            assert queue.worker_count(16 * queue.GIB, queue.GIB, 16, background=4 * queue.GIB) == 6
            pending = SimpleNamespace(done=lambda: False)
            pipeline.futures.append(({}, pending))
            pipeline.process = SimpleNamespace(pid=123)
            with patch.object(queue, '_proc_table', return_value={123: (1, queue.GIB // 2), 124: (123, queue.GIB // 2)}):
                assert pipeline.reservation() == queue.GIB
            pipeline.futures.clear()
        finally:
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
            calls.append((logfile.parent.name, logfile.name))
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
        assert calls.count(('round_000001', 'train.log')) == 1
        assert calls.count(('round_000002', 'train.log')) == 1
        assert calls.count(('round_000002', 'evaluation.log')) == 2
        assert state['training_jobs'][1]['timing']['promotion']['contract']['incumbent'].endswith('round_000001/student/model.pt')


def test_pre_a_completion_gets_evaluated_and_can_be_rejected_without_retraining():
    with tempfile.TemporaryDirectory() as folder:
        args, state = setup(Path(folder))
        snapshot = Path(folder) / 'snapshot'
        snapshot.mkdir()
        recipe = snapshot / 'recipe.json'
        recipe.write_bytes(args.recipe.read_bytes())
        model = snapshot / 'student/model.pt'
        model.parent.mkdir()
        model.write_bytes(b'previously trained, never evaluated')
        job = dict(round=1, recipe=str(recipe), model=str(model), status='queued')
        (snapshot / 'complete.json').write_text(json.dumps(dict(round=1, model=str(model), train_finished=1)))
        state.update(round=1, training_jobs=[job])
        def evaluate_only(self, command, logfile):
            assert logfile.name == 'evaluation.log', 'pre-A trained models must not be retrained'
            write_evaluation(self, command, approved=False)
        with patch.object(queue.TrainingPipeline, '_command', evaluate_only), \
             patch.object(queue, 'mem_available_bytes', return_value=64 * queue.GIB):
            pipeline = queue.TrainingPipeline(args, state)
            pipeline.close()
        assert job['status'] == 'done' and not job['promotion_approved']
        assert args.rollout_policy == Path(folder) / 'initial.pt'
        assert (snapshot / 'trained.json').is_file()


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


def test_corruption_after_evaluation_before_group_boundary_cannot_promote():
    with tempfile.TemporaryDirectory() as folder:
        args, state = setup(Path(folder))
        state.update(round=1, log=[dict(round=1, groups=[dict(label='c1a')])])
        def background(self, command, logfile):
            if logfile.name == 'train.log':
                model = logfile.parent / 'student/model.pt'
                model.parent.mkdir(parents=True)
                model.write_bytes(b'trained')
            if logfile.name == 'evaluation.log':
                write_evaluation(self, command)
        with patch.object(queue.TrainingPipeline, '_command', background), \
             patch.object(queue, 'mem_available_bytes', return_value=64 * queue.GIB):
            pipeline = queue.TrainingPipeline(args, state)
            pipeline.enqueue()
            report = pipeline.futures[0][1].result(timeout=5)
            assert report['promotion']['approved']
            path = next(Path(report['evaluation_directory']).glob('candidate/*/games.jsonl'))
            path.write_text('corrupted')
            pipeline.poll()
            assert not pipeline.failed
            pipeline.close()
        assert args.rollout_policy == Path(folder) / 'initial.pt'
        assert state['training_jobs'][0]['status'] == 'done'
        assert not state['training_jobs'][0]['promotion_approved']
        assert 'invalid paired evaluation artifacts' in state['training_jobs'][0]['promotion_reasons'][0]
