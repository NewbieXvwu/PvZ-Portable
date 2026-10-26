"""task119: level shares and fixed opening allocations, without game replay."""
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]
from pvz_training_rows import SourceRowCounts
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
