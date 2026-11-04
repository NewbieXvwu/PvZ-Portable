"""Small paired-report and fail-open monitor tests, without games or training."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'python')]
import promotion_gate as gate
from tactical_exam import check_split, paired_report


def item(name='a', skill='fog|pogo', archive='/exam/a', seeds=None):
    return dict(id=name, skill=skill, seeds=seeds or [242000000, 242000001],
                task=dict(drill_start=dict(archive=archive, decisions=[20], index=0,
                                          targets=[dict(id=7)])))


def test_exam_pairs_use_item_and_seed_and_report_each_skill():
    items = [item(), item('b', 'roof|ladder')]
    rows = [dict(status='ok', item=i['id'], model=m, seed=s, won=won)
            for i in items for m, wins in [('incumbent', [True, False]), ('B', [False, True]),
                                           ('A', [True, True])]
            for s, won in zip(i['seeds'], wins)]
    models = {'A': 'a.pt', 'B': 'b.pt', 'incumbent': 'r112.pt'}
    result = paired_report(list(reversed(rows)), items, models)
    assert result['status'] == 'ok' and result['veto'] is False
    for skill in ('fog|pogo', 'roof|ladder'):
        cell = result['comparisons']['B-incumbent']['skills'][skill]
        assert (cell['pairs'], cell['candidate_wins'], cell['incumbent_wins']) == (2, 1, 1)
        assert (cell['gain'], cell['harm'], cell['delta_rate']) == (1, 1, 0)
    assert result['comparisons']['A-B']['overall']['delta_rate'] == .5
    # Missing/error games must never become losses or be paired with another item.
    index = next(i for i, r in enumerate(rows) if r['item'] == 'b' and r['model'] == 'B'
                 and r['seed'] == 242000001)
    rows[index] = dict(status='error', item='b', model='B', seed=242000001, error='broken archive')
    result = paired_report(rows, items, models)
    assert result['status'] == 'partial'
    cell = result['comparisons']['B-incumbent']['skills']['roof|ladder']
    assert cell['pairs'] == 1 and cell['missing_pairs'] == 1 and len(result['errors']) == 1


def test_exam_split_excludes_training_archives_and_future_seed_blocks():
    training = dict(tasks=[dict(drill_start=dict(archive='/train/a', future_seed_start=53000000,
                                                future_seed_count=256))])
    check_split(dict(items=[item()]), training)
    with pytest.raises(ValueError, match='archive is used by training'):
        check_split(dict(items=[item(archive='/train/a')]), training)
    with pytest.raises(ValueError, match='training exam seed'):
        check_split(dict(items=[item(seeds=[53000001])]), training)


def test_exam_prefix_cache_replays_once_and_restores_before_each_future(monkeypatch):
    import drill_start
    class Env:
        def snapshot(self):
            return 7

        def restore(self, snapshot):
            assert snapshot == 7
            return {'tick': 500}

        def release_snapshot(self, snapshot):
            pass
    calls = []
    prefix = [{'tick': 500}]
    monkeypatch.setattr(drill_start, '_restore_prefix',
                        lambda *a, **kw: (calls.append(a) or {'tick': 500}, prefix))
    env, cache = Env(), {}
    for _ in range(3):
        obs, history = drill_start.restore_prefix(env, Path('/a'), 20, [], {}, cache=cache)
        assert obs['tick'] == 500 and history is prefix
    assert len(calls) == 1
    drill_start.restore_prefix(env, Path('/b'), 20, [], {}, cache=cache)
    assert len(calls) == 2


def test_exam_selection_requires_paired_calibration_and_stratifies():
    from exam_select import select
    items = [item('a'), item('b', 'roof|ladder'), item('c'), item('d')]
    for entry in items:
        entry['seeds'] = [242100000, 242100001, 242100002]
    rows = [dict(status='ok', item=i['id'], model=side, seed=seed, won=won)
            for i in items for side, wins in [('teacher', [True, False, False]),
                                              ('incumbent', [False, False, False])]
            for seed, won in zip(i['seeds'], wins)]
    # c is too easy for both arms, d has an incomplete teacher series.
    for row in rows:
        if row['item'] == 'c':
            row['won'] = True
    rows = [row for row in rows if not (row['item'] == 'd' and row['model'] == 'teacher')]
    result = select(dict(items=items), rows, dict(tasks=[]), limit=2)
    assert [i['id'] for i in result['items']] == ['a', 'b']
    assert result['items'][0]['seeds'] == [242200000]
    assert result['excluded'] == dict(no_observed_discrimination=1, incomplete_calibration=1)
    assert result['status'] == 'insufficient_calibration'


@pytest.mark.parametrize('mode', ['crash', 'timeout', 'missing_summary'])
def test_exam_monitor_failure_is_isolated(tmp_path, monkeypatch, mode):
    args = SimpleNamespace(out=tmp_path, value=Path('/value'), executable=Path('/engine'),
                           resource_dir=Path('/resources'), eval_units=4, exam_bank=Path('/bank'))
    class Process:
        pid = 123

        def wait(self, timeout=None):
            if timeout and mode == 'timeout':
                raise subprocess.TimeoutExpired('exam', timeout)
            return 1 if mode == 'crash' else 0

    killed = []
    monkeypatch.setattr(gate.subprocess, 'Popen', lambda *a, **kw: Process())
    monkeypatch.setattr(gate.os, 'killpg', lambda *a: killed.append(a))
    result = gate.exam_monitor(args, dict(B='/b', incumbent='/r112'))
    assert result['status'] == 'skipped' and result['veto'] is False
    assert bool(killed) == (mode == 'timeout')


def test_exam_regression_never_changes_gate_decision(tmp_path, monkeypatch):
    from test_promotion_evaluation import arguments
    args = arguments(tmp_path)
    # Both candidates beat the incumbent on every scored gate game.
    args.candidates['B'] = args.candidates['A']
    def unit(args, lock, manifest, task, side, model):
        return side, task['task_id'], {
            seed: dict(won=side != 'incumbent', final_wave=20, wave_count=20)
            for seed in task['seeds']}
    monkeypatch.setattr(gate, '_run_unit', unit)
    monkeypatch.setattr(gate, 'exam_monitor', lambda *a: dict(status='ok', veto=False,
        comparisons={'B-incumbent': {'overall': {'delta_rate': -1.0}}}))
    result = gate.evaluate(args)
    assert result['approved'] is True and result['winner_family'] == 'A'
    assert result['reasons'] == []
    assert result['monitor']['tactical_exam']['comparisons']['B-incumbent']['overall']['delta_rate'] == -1
    assert all('tactical_exam' not in side['monitor'] for side in result['candidates'].values())
    assert gate.valid_certificate(result, result['contract'])


def test_exam_monitor_survives_round_ab_report(tmp_path, monkeypatch):
    from test_curriculum_pipeline import setup, certificate
    import curriculum_queue as queue
    args, state = setup(tmp_path)
    pipeline = queue.TrainingPipeline(args, state)
    try:
        job = dict(round=1, recipe=str(args.recipe), model=str(tmp_path / 'student.pt'))
        pipeline._seeded(job)
        decision = certificate(pipeline, job, approved=False)
        exam = dict(status='partial', veto=False, comparisons={})
        decision['monitor']['tactical_exam'] = exam
        def command(cmd, logfile):
            out = Path(cmd[cmd.index('--out') + 1])
            out.mkdir(parents=True)
            (out / 'decision.json').write_text(json.dumps(decision))
        monkeypatch.setattr(pipeline, '_command', command)
        result = pipeline._evaluate(job)
        report = json.loads((tmp_path / 'ab_result.json').read_text())
        assert result['promotion']['approved'] is False
        assert report['monitor']['tactical_exam'] == exam
    finally:
        pipeline.executor.shutdown()
