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


def setup(root):
    recipe = root / 'recipe.json'
    write_recipe(recipe, dict(schema_version=1, kind='training', data=str(root / 'frames'),
                             labels=['base'], out=str(root / 'unused'), epochs=2, precompute_workers=2))
    args = SimpleNamespace(state=root / 'state.json', log=root / 'curriculum.log', frames=root / 'frames',
                           recipe=recipe, python=Path(sys.executable), rollout_policy=root / 'initial.pt',
                           workers=2, stop_flag=root / 'stop', out=root / 'runs')
    args.rollout_policy.write_bytes(b'initial')
    cells = {name: dict(windows={}) for name in ('a', 'b')}
    state = dict(round=0, log=[], cells=cells, offsets={})
    return args, state


def plan(round_index):
    return [dict(base=name, multiplier=1., task_id=name, offset=(round_index - 1) * 8,
                 games=8, manifest='/manifest', label=f'c{round_index}{name}') for name in ('a', 'b')]


def test_collection_overlaps_both_phases_and_switches_only_between_groups():
    with tempfile.TemporaryDirectory() as folder:
        args, state = setup(Path(folder))
        pack_started, release_pack, train_started, release_train = [threading.Event() for _ in range(4)]
        seen = []
        def background(self, command, logfile):
            if 'round_000001' in str(logfile):
                if logfile.name == 'pack.log':
                    pack_started.set()
                    assert release_pack.wait(5)
                else:
                    train_started.set()
                    assert release_train.wait(5)
            if logfile.name == 'train.log':
                model = logfile.parent / 'student/model.pt'
                model.parent.mkdir(parents=True)
                model.write_bytes(b'trained')
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
            with patch.object(queue, '_proc_table', return_value={123: (1, queue.GIB), 124: (123, queue.GIB)}):
                assert pipeline.reservation() == 2 * queue.GIB
            pipeline.futures.clear()
        finally:
            pipeline.close()
