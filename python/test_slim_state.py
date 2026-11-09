"""task258: compact recovery state without losing recipe, corpus or gate evidence."""
import copy
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / 'scripts')]
import curriculum_queue as q


def state_with_job(tmp_path):
    path = tmp_path / 'state.json'
    folder = tmp_path / 'state.pipeline/round_000001'
    recipe = dict(schema_version=1, kind='training', data=str(tmp_path / 'frames'),
                  labels=['base'], out=str(folder / 'student'), precompute_workers=3)
    job = dict(round=1, status='done', recipe=recipe, model=str(folder / 'student/model.pt'),
               promotion_approved=True, promotion_reasons=[])
    state = dict(round=1, cells={}, offsets={}, log=[], training_jobs=[job])
    return path, folder, state, job


@pytest.mark.parametrize('exists', [False, True])
def test_slim_state_embedded_recipe_migrates_once(tmp_path, exists):
    path, folder, state, job = state_with_job(tmp_path)
    recipe = copy.deepcopy(job['recipe'])
    if exists:
        folder.mkdir(parents=True)
        (folder / 'recipe.json').write_text(json.dumps(recipe))
    path.write_text(json.dumps(state))
    loaded = q.load_state(path)
    migrated = loaded['training_jobs'][0]
    assert q.read_json(q.job_recipe_path(migrated)) == recipe
    assert q.load_recipe(q.job_recipe_path(migrated), 'training')['precompute_workers'] == 3
    first_write = path.stat().st_mtime_ns
    recipe_write = q.job_recipe_path(migrated).stat().st_mtime_ns
    assert q.load_state(path) == loaded
    assert path.stat().st_mtime_ns == first_write
    assert q.job_recipe_path(migrated).stat().st_mtime_ns == recipe_write


def test_slim_state_recipe_conflict_preserves_both_copies(tmp_path):
    path, folder, state, job = state_with_job(tmp_path)
    folder.mkdir(parents=True)
    other = dict(job['recipe'], labels=['different'])
    (folder / 'recipe.json').write_text(json.dumps(other))
    path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match='embedded data retained'):
        q.load_state(path)
    assert q.read_json(path) == state
    assert q.read_json(folder / 'recipe.json') == other


@pytest.mark.parametrize('existing', ['same', 'missing', 'different'])
def test_slim_state_report_keeps_full_recovery_evidence(tmp_path, existing):
    path, folder, state, job = state_with_job(tmp_path)
    folder.mkdir(parents=True)
    (folder / 'recipe.json').write_text(json.dumps(job['recipe']))
    job['recipe'] = str(folder / 'recipe.json')
    report = dict(train_peak_anon_bytes=123, pack_peak_anon_bytes=60,
                  corpus_archives=['seed_1/base', 'seed_2/base'],
                  candidates=[dict(recipe=job['recipe'], status='trained')],
                  promotion=dict(version=q.GATE_VERSION, approved=True, reasons=[],
                      contract=dict(incumbent='initial.pt'), winner_family='B',
                      next_panel={'cell': {'mult_index': 2}}, paired=[{'candidate': True}]))
    job['timing'] = copy.deepcopy(report)
    complete = folder / 'complete.json'
    if existing != 'missing':
        complete.write_text(json.dumps(report if existing == 'same' else {'newer': True}))
    path.write_text(json.dumps(state))
    loaded = q.load_state(path)
    migrated = loaded['training_jobs'][0]
    assert 'corpus_archives' not in migrated['timing']
    assert q.job_report(migrated) == report
    assert migrated['timing']['promotion'] == {
        key: report['promotion'][key] for key in q.PROMOTION_STATE_FIELDS}
    if existing == 'different':
        assert q.read_json(complete) == {'newer': True}
    assert q.load_state(path) == loaded
    # Group-boundary saves must never reread or rewrite archived reports.
    with patch.object(q, 'read_json', side_effect=AssertionError('unexpected report read')):
        q.save_state(path, loaded)
        q.save_state(path, loaded)


def test_slim_state_log_retains_labels_and_student_round_cutoff(tmp_path):
    path, folder, state, job = state_with_job(tmp_path)
    folder.mkdir(parents=True)
    (folder / 'recipe.json').write_text(json.dumps(job['recipe']))
    job['recipe'] = str(folder / 'recipe.json')
    state['log'] = [dict(round=i, groups=[dict(label=f'group_{i}'),
                    dict(label=f'failed_{i}', status='failed')]) for i in range(1, 26)]
    before = q.pack_labels(state, 'base')
    old_student = q.pack_labels(state, 'base', through_round=3)
    path.write_text(json.dumps(state))
    loaded = q.load_state(path)
    history = sorted(folder.parent.glob('round_*/collection.json'))
    assert [q.read_json(entry) for entry in history] == state['log'][:5]
    assert len(loaded['log']) == q.STATE_LOG_ROUNDS
    assert q.pack_labels(loaded, 'base') == before
    assert q.pack_labels(loaded, 'base', through_round=3) == old_student
    written = [entry.stat().st_mtime_ns for entry in history]
    assert q.load_state(path) == loaded
    assert [entry.stat().st_mtime_ns for entry in history] == written
    # Crash after archiving but before the state replacement: reuse immutable rounds.
    path.write_text(json.dumps(state))
    resumed = q.load_state(path)
    assert q.pack_labels(resumed, 'base') == before
    assert [entry.stat().st_mtime_ns for entry in history] == written


def test_slim_state_pipeline_recovers_promoted_corpus(tmp_path):
    from test_curriculum_pipeline import setup
    args, state = setup(tmp_path)
    model = tmp_path / 'promoted.pt'
    model.write_bytes(b'model')
    job = dict(round=1, status='done', recipe=str(args.recipe), model=str(model),
               promotion_approved=True, timing=dict(corpus_archives=['seed_1/base'],
                   promotion=dict(approved=True, reasons=[])))
    state['training_jobs'] = [job]
    q.save_state(args.state, state)
    restored = q.load_state(args.state)
    pipeline = q.TrainingPipeline(args, restored)
    try:
        assert restored['student_round'] == 1
        assert restored['student_corpus_archives'] == ['seed_1/base']
        assert args.rollout_policy == model
        assert pipeline.futures == []
    finally:
        pipeline.close()
