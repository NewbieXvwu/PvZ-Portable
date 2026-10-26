"""Resume guards must preserve finished evidence and reject mixed evaluations."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import task93_validation as queue


def test_memory_admission_uses_measured_same_recipe_peak_with_margin(tmp_path, monkeypatch):
    monkeypatch.setattr(queue, 'DATA', tmp_path)
    assert queue.training_memory_requirement() == 16*1024**3
    proof = tmp_path/'seed_1/trained.json'
    proof.parent.mkdir()
    proof.write_text(json.dumps({'observed_peak_rss_bytes': 12118224896}))
    assert queue.training_memory_requirement() == (12118224896*115+99)//100
    proof.write_text(json.dumps({'observed_peak_rss_bytes': 1}))
    assert queue.training_memory_requirement() == 12*1024**3


def test_finished_games_are_reused_only_with_matching_model_and_task(tmp_path):
    task = dict(task_id='example', level=7, deck=[0, 1, 2, 3, 4, 5], seeds=[1, 2])
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'tasks': [task]}))
    model = tmp_path/'model.pt'
    config = dict(manifest=str(manifest), task_id='example', arms='fast', dagger_beta='0',
                  executable='/test/pvz', rollout_policy=str(model))
    (tmp_path/'config.json').write_text(json.dumps(config))
    games = tmp_path/'games.jsonl'
    rows = [dict(seed=s, deck=task['deck'], arm='fast', terminal=True, won=True) for s in (1, 2)]
    games.write_text('\n'.join(map(json.dumps, rows)))
    assert queue.complete_evaluation(games, task, model)
    with pytest.raises(ValueError, match='model differs'):
        queue.complete_evaluation(games, task, tmp_path/'different.pt')
    games.write_text(json.dumps(rows[0]))
    assert not queue.complete_evaluation(games, task, model)
    games.write_text('\n'.join(map(json.dumps, rows+[rows[0]])))
    with pytest.raises(ValueError, match='duplicate seed'):
        queue.complete_evaluation(games, task, model)
    games.write_text(json.dumps({**rows[0], 'terminal': False}))
    with pytest.raises(ValueError, match='not a terminal'):
        queue.complete_evaluation(games, task, model)


def test_qc_pending_requires_a_live_controller_and_valid_observation(tmp_path, monkeypatch):
    state = tmp_path/'qc.json'
    monkeypatch.setattr(queue, 'QC_STATE', state)
    monkeypatch.setattr(queue, 'qc_controllers', lambda: [123])
    state.write_text(json.dumps({'training_jobs': [{'status': 'queued'}]}))
    assert queue.qc_pending() is True
    state.write_text(json.dumps({'training_jobs': [{'status': 'done'}, {'status': 'failed'}]}))
    assert queue.qc_pending() is False
    state.write_text('{')
    assert queue.qc_pending() is None
    state.write_text('null')
    assert queue.qc_pending() is None
    monkeypatch.setattr(queue, 'qc_controllers', lambda: [])
    state.write_text(json.dumps({'training_jobs': [{'status': 'queued'}]}))
    assert queue.qc_pending() is False


def test_only_priority_preemption_retries_the_same_recipe(tmp_path, monkeypatch):
    monkeypatch.setattr(queue, 'DATA', tmp_path)
    folder = tmp_path/'seed_2'
    (folder/'student').mkdir(parents=True)
    (folder/'recipe.json').write_text('{}')
    calls = []

    def run(command, name, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            raise queue.QCPreempted('priority yield')
        (folder/'student/model.pt').write_bytes(b'test fixture')
        (folder/'student/metrics.json').write_text(json.dumps({'history': [{}]*12}))
        return 999999999

    monkeypatch.setattr(queue, 'run', run)
    queue.train_seed(2)
    assert len(calls) == 2 and calls[0] == calls[1]
    assert queue.complete_model(2)


def test_real_training_error_does_not_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(queue, 'DATA', tmp_path)
    calls = []

    def fail(*args, **kwargs):
        calls.append(args)
        raise RuntimeError('real loader failure')

    monkeypatch.setattr(queue, 'run', fail)
    with pytest.raises(RuntimeError, match='real loader failure'):
        queue.train_seed(2)
    assert len(calls) == 1


def test_uncertified_weights_are_never_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(queue, 'DATA', tmp_path)
    model = tmp_path/'seed_2/student/model.pt'
    model.parent.mkdir(parents=True)
    model.write_bytes(b'preserve for verification')
    with pytest.raises(ValueError, match='requires verification'):
        queue.train_seed(2)
    assert model.read_bytes() == b'preserve for verification'
