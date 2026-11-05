"""task233: B 从现任学生蒸馏（Learning without Forgetting）的定向测试。"""
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'python')]

import train_rollout_policy as trainer
from pack_training_rows import pack
from pvz_recipe import load_recipe, resolve_paths, training_row_options, validate
from test_incremental_pack import fixture, options


def _deterministic():
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)


def _recipe(data, out, **updates):
    document = dict(schema_version=1, kind='training', data=str(data), labels=['a', 'b'],
                    out=str(out), device='cpu', threads=1, width=4, epochs=1, seed=7,
                    batch_size=32, validation_batch_size=32, cell_soft_weight=3.0,
                    validation=dict(seed=0, fraction=.34))
    document.update(updates)
    return validate(document, 'training')


def _train(monkeypatch, document, path):
    trainer.write_recipe(path, document)
    monkeypatch.setattr(sys, 'argv', ['trainer', '--recipe', str(path)])
    monkeypatch.setattr(trainer, 'Metrics', Mock())
    monkeypatch.setattr(trainer, 'set_oom_score_adj', lambda *_: None)
    trainer.main()
    return torch.load(Path(document['out']) / 'model.pt', map_location='cpu', weights_only=False)


def test_distill_zero_matches_reference_weights(tmp_path, monkeypatch):
    _deterministic()
    data = tmp_path / 'data'
    fixture(data, ['a', 'b'])
    pack(data, options(data, ['a', 'b'], cells=True), tmp_path / 'cache')
    reference = _train(monkeypatch, _recipe(data, tmp_path / 'reference'), tmp_path / 'reference.json')
    # A path that cannot be opened proves the disabled path really skips the teacher read.
    disabled = _train(monkeypatch,
                      _recipe(data, tmp_path / 'disabled', distill_weight=0.0,
                              distill_from=str(tmp_path / 'never-loaded.pt')),
                      tmp_path / 'disabled.json')
    for name, value in reference['state_dict'].items():
        assert torch.equal(value.view(torch.uint8), disabled['state_dict'][name].view(torch.uint8)), name


def test_distill_from_initial_student_starts_at_zero(tmp_path, monkeypatch):
    _deterministic()
    data = tmp_path / 'data'
    fixture(data, ['a', 'b'])
    # The production B shape: identity card head, meanmax + row pool. Its masked packet
    # logits are what make a careless KL implementation produce NaN.
    pack(data, options(data, ['a', 'b'], cells=True, packet_head='identity'), tmp_path / 'cache')
    shape = dict(packet_head='identity', pool='meanmax', row_pool=True)
    baseline = _train(monkeypatch, _recipe(data, tmp_path / 'baseline', **shape), tmp_path / 'baseline.json')
    student = tmp_path / 'baseline' / 'model.pt'
    distilled = _train(monkeypatch,
                       _recipe(data, tmp_path / 'distilled', init=str(student),
                               distill_from=str(student), distill_weight=1.0, max_steps=1, **shape),
                       tmp_path / 'distilled.json')
    first = distilled['history'][0]
    assert first['distill_loss'] < 1e-6          # identical weights -> KL(incumbent || student) == 0
    assert first['bc_loss'] > 0                    # the BC signal is still the original labels
    assert first['train_loss'] >= first['bc_loss']


def test_distill_recipe_fields_stay_out_of_the_pack_key(tmp_path):
    b = load_recipe(ROOT / 'recipes/dw4i2_v2timer_clean_b.json')
    a = load_recipe(ROOT / 'recipes/dw4i2_v2timer_clean_a.json')
    assert b['distill_weight'] == 5.0 and b['distill_from'] is None
    assert a['distill_weight'] == 0 and a['distill_from'] is None
    assert training_row_options(a) == training_row_options(b)
    assert resolve_paths(dict(b, distill_from='artifacts/student/x/model.pt'))['distill_from'] == \
        str(ROOT / 'artifacts/student/x/model.pt')
    with pytest.raises(ValueError):
        validate(dict(b, distill_weight=-0.1))


def test_distill_teacher_architecture_mismatch_is_fatal(tmp_path):
    from pvz_rollout_policy import RolloutNet
    student = RolloutNet(width=4)
    teacher = RolloutNet(width=8)
    path = tmp_path / 'teacher.pt'
    torch.save({'state_dict': teacher.state_dict(), 'width': 8}, path)
    with pytest.raises(ValueError, match='architecture differs'):
        trainer.load_distill_teacher(student, path, 'cpu')
    with pytest.raises(ValueError, match='requires distill_from'):
        trainer.load_distill_teacher(student, None, 'cpu')


def test_distill_queue_freezes_incumbent_as_teacher(tmp_path):
    import curriculum_queue as queue
    incumbent = tmp_path / 'incumbent.pt'
    pipeline = queue.TrainingPipeline.__new__(queue.TrainingPipeline)
    pipeline.args = SimpleNamespace(
        recipe=ROOT / 'recipes/dw4i2_v2timer_clean_a.json',
        alternate_recipe=ROOT / 'recipes/dw4i2_v2timer_clean_b.json',
        state=tmp_path / 'state.json', log=tmp_path / 'log', frames=tmp_path / 'frames',
        b_from_incumbent=True, b_epochs=4, b_initial_epochs=8, a_restart=True,
        training_parallelism=1)
    pipeline.directory = tmp_path / 'rounds'
    pipeline.build_workers = 1
    pipeline.state = dict(round=52, log=[], training_jobs=[],
                          student=str(incumbent), student_round=101)
    pipeline._plan_build = Mock()
    pipeline._snapshot_selection = Mock()
    pipeline._seeded = Mock()
    pipeline.executor = Mock()
    pipeline.futures = []
    pipeline.enqueue()
    job = pipeline.state['training_jobs'][0]
    assert job['distill_weight'] == 5.0
    a_path, b_path = (Path(candidate['recipe']) for candidate in job['candidate_recipes'])
    a, b = load_recipe(a_path), load_recipe(b_path)
    assert b['init'] == str(incumbent) and b['distill_from'] == str(incumbent)
    assert b['epochs'] == 4 and b['distill_weight'] == 5.0
    assert a['distill_weight'] == 0 and a['distill_from'] is None
    assert training_row_options(a) == training_row_options(b)
    # The train-time lineage refresh keeps the same frozen teacher.
    assert pipeline._b_recipe(job)['distill_from'] == str(incumbent)
    assert queue.experiment_name(job) == 'task205-clean-b+b-from-incumbent+distill5'


def test_production_a_is_a_second_distilled_seed(tmp_path):
    import curriculum_queue as queue
    import qc_launch
    launch = qc_launch.command(SimpleNamespace(
        repo=tmp_path, worktree=ROOT, python=Path('/python'), review_share=None, logs=tmp_path,
        rounds=1, workers=1, recipe=None, alternate_recipe=None, b_init=None, b_initial_epochs=None,
        b_epochs=None, a_restart=False, executable=None))
    assert '--no-a-restart' in launch
    incumbent = tmp_path / 'incumbent.pt'
    pipeline = queue.TrainingPipeline.__new__(queue.TrainingPipeline)
    pipeline.args = SimpleNamespace(
        recipe=Path(launch[launch.index('--recipe') + 1]),
        alternate_recipe=Path(launch[launch.index('--alternate-recipe') + 1]),
        state=tmp_path / 'state.json', log=tmp_path / 'log', frames=tmp_path / 'frames',
        b_from_incumbent=True, b_epochs=4, b_initial_epochs=8, a_restart=False,
        training_parallelism=1)
    pipeline.directory = tmp_path / 'rounds'
    pipeline.build_workers = 1
    pipeline.state = dict(round=52, log=[], training_jobs=[],
                          student=str(incumbent), student_round=101)
    pipeline._plan_build = Mock()
    pipeline._snapshot_selection = Mock()
    pipeline._seeded = Mock()
    pipeline.executor = Mock()
    pipeline.futures = []
    pipeline.enqueue()
    job = pipeline.state['training_jobs'][0]
    a, b = pipeline._a_recipe(job), pipeline._b_recipe(job)
    for recipe in (a, b):
        assert recipe['init'] == recipe['distill_from'] == str(incumbent)
        assert recipe['epochs'] == 4 and recipe['distill_weight'] == 5.0
        assert recipe['optimizer']['lr'] == 0.0005
    assert job['training_seeds'] == [1, 2]
