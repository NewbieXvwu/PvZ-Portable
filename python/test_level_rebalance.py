"""task119/task121: level shares and fixed opening allocations, without game replay."""
import json
import tempfile
from pathlib import Path
import sys

import torch
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]
from pvz_training_rows import SourceRowCounts, balance_levels
from make_rebalance_panel import make_panel, LEVELS
import curriculum_queue as queue


def test_inverse_rows_and_effective_weight_cap_leave_validation_unchanged(tmp_path):
    counts = SourceRowCounts(balance=True)
    weights, validation = [], []
    for level, rows in ((7, 100), (19, 10), (23, 10), (32, 10), (37, 10)):
        archive = tmp_path / f'seed_{level}' / 'teacher'
        archive.mkdir(parents=True)
        (archive / 'meta.json').write_text(json.dumps(dict(context=dict(level=level))))
        counts.observe(archive / 'frames.npz', list(range(rows + 1)))
        weights.extend([100.0 if level == 7 else 1.0] * rows + [3.0])
        validation.extend([False] * rows + [True])
    weights = torch.tensor(weights)
    validation = torch.tensor(validation)
    before = weights.clone()
    report = counts.rebalance(weights, validation, .25)
    assert torch.equal(weights[validation], before[validation])
    assert abs(float(weights[~validation].sum() - before[~validation].sum())) < .01
    assert report['levels'][7]['effective_share'] == .25
    for level, start, stop in counts.spans:
        share = float(weights[start:stop][~validation[start:stop]].sum() / weights[~validation].sum())
        assert share <= .250001
        assert abs(share - report['levels'][level]['effective_share']) < 1e-6
    assert report['levels'][7]['inverse_rows'] < report['levels'][19]['inverse_rows']


def test_opening_cells_keep_decks_fixed_beta_and_resume_within_budget():
    tasks = {task['task_id']: task for task in make_panel()['tasks']}
    state = dict(round=0, offsets={}, cells={})
    queue.reconcile_cells(state, tasks, {}, {})
    assert sorted(cell['level'] for cell in state['cells'].values()) == list(LEVELS)
    for cell in state['cells'].values():
        cell['windows'] = {'1.0': dict(recent=[1] * 12, wins=12, games=12)}
    manifests = {task_id: Path('/opening.json') for task_id in tasks}
    plan, _ = queue.plan_round(state, tasks, {}, [1., 1.25, 1.5], manifests, 768, 48)
    assert sum(item['games'] for item in plan) == 192
    for level in LEVELS:
        pair = [item for item in plan if tasks[item['task_id']]['level'] == level]
        assert [(item['games'], item['beta'], item['offset'], item['multiplier']) for item in pair] == [
            (12, 0., 0, 1.), (12, .5, 12, 1.)]
        assert all('demo_archive' not in item for item in pair)
    assert all(cell['mult_index'] == 0 for cell in state['cells'].values())
    for level in (32, 37):
        task = next(task for task in tasks.values() if task['level'] == level)
        assert 31 in task['deck'] and 16 in task['deck'] and task['fog_observation'] == 'human_v2'
    first = plan[0]
    state['offsets'][first['task_id']] = 12
    state['completed_groups'] = [dict(first, status='done')]
    resumed, _ = queue.plan_round(state, tasks, {}, [1., 1.25, 1.5], manifests, 768, 48)
    same = [item for item in resumed if item['base'] == first['base']]
    assert [(item['offset'], item['beta']) for item in same] == [(12, .5)]
    assert sum(item['games'] for item in resumed) + 12 == 192


@pytest.mark.parametrize('stop_at_boundary', [False, True])
def test_last_group_stop_does_not_enqueue_a_new_training_round(tmp_path, monkeypatch, stop_at_boundary):
    from types import SimpleNamespace
    calls = []
    args = SimpleNamespace(state=tmp_path / 'state.json', log=tmp_path / 'log',
                           stop_flag=tmp_path / 'stop',
                           pipeline=SimpleNamespace(poll=lambda: None, enqueue=lambda: calls.append('enqueue')))
    state = dict(round=0, log=[], cells={}, offsets={})
    def collect(*_):
        if stop_at_boundary:
            args.stop_flag.touch()
    monkeypatch.setattr(queue, '_run_group', collect)
    assert queue.run_round(args, state, [dict(base='a', label='a', games=12)], [])
    assert state['round'] == 1
    assert calls == ([] if stop_at_boundary else ['enqueue'])

FIXTURE = ROOT / 'python/fixtures_level_shares_round21.json'


def round21_levels():
    """Level rows and pre-balance weights recorded by the round 21 trainer."""
    document = json.loads(FIXTURE.read_text())
    return {int(level): dict(rows=record['rows'], original_weight=record['original_weight'])
            for level, record in document['levels'].items()}, document


def task119_reference(levels, cap):
    """The exact task119 share computation, frozen here as the compatibility reference."""
    total_rows = sum(record['rows'] for record in levels.values())
    total_weight = sum(record['original_weight'] for record in levels.values())
    for record in levels.values():
        record['inverse_rows'] = total_rows / (len(levels) * record['rows'])
    masses = {level: record['original_weight'] * record['inverse_rows']
              for level, record in levels.items()}
    shares = {}
    while masses:
        remaining = 1 - sum(shares.values())
        total = sum(masses.values())
        capped = [level for level, mass in masses.items() if remaining * mass / total > cap]
        if not capped:
            shares.update({level: remaining * mass / total for level, mass in masses.items()})
            break
        for level in capped:
            shares[level] = cap
            del masses[level]
    for level, record in levels.items():
        record.update(effective_share=shares[level],
                      multiplier=total_weight * shares[level] / record['original_weight'])
    return levels


def test_default_parameters_reproduce_round21_shares_bit_for_bit():
    levels, document = round21_levels()
    reference = {int(level): record for level, record in
                 task119_reference(json.loads(json.dumps(levels)), document['cap']).items()}
    totals = balance_levels(levels, document['cap'])
    assert len(levels) == 46
    for level, record in reference.items():
        assert levels[level]['inverse_rows'] == record['inverse_rows']
        assert levels[level]['effective_share'] == record['effective_share']
        assert levels[level]['multiplier'] == record['multiplier']
        # The recorded level_shares.json is the ground truth, not only our own reference.
        assert levels[level]['multiplier'] == document['levels'][str(level)]['multiplier']
        assert levels[level]['effective_share'] == document['levels'][str(level)]['effective_share']
    assert totals['allocated_weight'] == totals['total_weight']
    assert totals['allocated_weight_ratio'] == 1.0
    assert 'multiplier_capped_from' not in levels[7]


def test_tempered_power_and_multiplier_ceiling_on_round21_rows():
    levels, document = round21_levels()
    totals = balance_levels(levels, document['cap'], power=.5, max_multiplier=20)
    raw = {level: record['original_weight'] / totals['total_weight'] for level, record in levels.items()}
    assert abs(raw[7] - .58908) < 1e-4            # L7 carries 58.9% of the unweighted mass
    assert levels[7]['multiplier'] < 1             # and is still pushed down, not up
    assert 19 < 100 * levels[7]['effective_share'] < 25
    assert levels[40]['multiplier'] == 20          # capped from 61.3
    assert levels[40]['multiplier_capped_from'] > 60
    assert max(record['multiplier'] for record in levels.values()) == 20
    assert totals['allocated_weight'] < totals['total_weight']
    assert abs(totals['allocated_weight_ratio'] - .99594) < 1e-4
    assert abs(sum(record['multiplier'] * record['original_weight']
                   for record in levels.values()) - totals['allocated_weight']) < 1e-3
    for record in levels.values():
        assert record['effective_share'] <= document['cap'] + 1e-9
    flat = {level: dict(rows=record['rows'], original_weight=record['original_weight'])
            for level, record in levels.items()}
    balance_levels(flat, document['cap'], power=0)
    # power=0 stops balancing, so only the 25% share cap still acts: L7 alone exceeds it,
    # and everything else is scaled by the same factor to refill the freed 75%.
    assert flat[7]['effective_share'] == document['cap']
    assert max(record['effective_share'] for record in flat.values()) == document['cap']
    others = {level: record['multiplier'] for level, record in flat.items() if level != 7}
    assert max(others.values()) - min(others.values()) < 1e-12
    assert abs(min(others.values()) - (1 - document['cap']) / (1 - raw[7])) < 1e-9


def test_rebalance_leaves_validation_rows_and_total_weight_untouched_by_default():
    counts = SourceRowCounts(balance=True)
    weights, validation = [], []
    for level, rows in ((7, 100), (19, 10), (23, 10), (32, 10), (37, 10)):
        archive = Path(tempfile.mkdtemp()) / f'seed_{level}' / 'teacher'
        archive.mkdir(parents=True)
        (archive / 'meta.json').write_text(json.dumps(dict(context=dict(level=level))))
        counts.observe(archive / 'frames.npz', list(range(rows + 1)))
        weights.extend([100.0 if level == 7 else 1.0] * rows + [3.0])
        validation.extend([False] * rows + [True])
    weights = torch.tensor(weights)
    validation = torch.tensor(validation)
    before = weights.clone()
    report = counts.rebalance(weights, validation, .25)
    assert torch.equal(weights[validation], before[validation])
    assert abs(float(weights[~validation].sum() - before[~validation].sum())) < .01
    assert (report['power'], report['max_multiplier']) == (1.0, None)
    assert report['allocated_weight'] == report['total_weight']

