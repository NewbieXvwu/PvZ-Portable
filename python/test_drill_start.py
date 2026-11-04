"""task235 directed tests: drill manifest, short horizon, criterion, cap and row weight.

Deliberately small: no trainer, no game, no full pytest file.  The production
collection -> pack -> row-read round trip is run separately on the Mac (task235 report).
"""
import json
import sys
import tempfile
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]

from curriculum_queue import load_demo_start_cells, plan_round
from drill_manifest import task_for
from drill_start import horizon_reached, judge
from pvz_recipe import training_row_options, validate
from pvz_training_rows import DemoRowWeights

VALUES = [1.0, 1.25, 1.5]
NAMES = {}


def drill_cell(decision=364, level=31, horizon=1200):
    return dict(family='drill_start', level=level, mult_index=0, last_round=0, windows={},
                ladder=[decision], task_id='drill235_l31_jack_test', archive='/archives/drill',
                seeds=list(range(53000000, 53000256)), drill=True, horizon=horizon,
                criterion=dict(threats_neutralized=True), targets=[dict(id=7)])


def test_load_drill_manifest_carries_horizon_criterion_and_targets():
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / 'archive'
        archive.mkdir()
        document = {'tasks': [{
            'task_id': 'drill235_l31_jack_test', 'level': 31, 'seeds': [42],
            'deck': [1, 2, 3, 4, 5, 6],
            'drill_start': {'archive': str(archive), 'decisions': [364], 'index': 0,
                            'future_seed_start': 53000000, 'future_seed_count': 4,
                            'horizon': 900,
                            'criterion': {'threats_neutralized': True},
                            'targets': [{'id': 7, 'name': 'jack'}]}}]}
        path = Path(tmp) / 'tasks.json'
        path.write_text(json.dumps(document))
        cells, manifests = load_demo_start_cells(path)
        cell = cells['drill235_l31_jack_test']
        assert cell['family'] == 'drill_start' and cell['drill'] is True
        assert cell['horizon'] == 900 and cell['ladder'] == [364]
        assert cell['targets'] == [{'id': 7, 'name': 'jack'}]
        assert cell['seeds'] == [53000000, 53000001, 53000002, 53000003]
        assert manifests == {'drill235_l31_jack_test': path}
        # A drill without targets is not a scorable cell.
        document['tasks'][0]['drill_start']['targets'] = []
        path.write_text(json.dumps(document))
        with pytest.raises(ValueError):
            load_demo_start_cells(path)


def test_short_horizon_is_measured_from_the_restored_root_not_zero():
    assert not horizon_reached({'tick': 1000}, root_tick=900, horizon=1200)
    assert horizon_reached({'tick': 2100}, root_tick=900, horizon=1200)
    assert horizon_reached({'tick': 2100}, root_tick=900, horizon=0)
    assert not horizon_reached({'tick': 2100}, root_tick=900, horizon=None)


class _Env:
    def __init__(self, zombies):
        self.zombies = zombies

    def privileged_state(self):
        return {'zombies': self.zombies}


def test_local_success_criterion_is_mechanical():
    target = dict(id=7, on_board=True, row=2, type=15, x=300.0, health=200)
    obs = {'tick': 1300, 'result': 0}
    verdict = judge(_Env([target]), obs, [{'id': 7}], 3, 3, horizon=1200, root_tick=100)
    assert not verdict['success'] and verdict['targets_alive'] == 1
    verdict = judge(_Env([]), obs, [{'id': 7}], 3, 3, horizon=1200, root_tick=100)
    assert verdict['success'] and verdict['tick_delta'] == 1200
    verdict = judge(_Env([]), obs, [{'id': 7}], 3, 2, horizon=1200, root_tick=100)
    assert not verdict['success'] and verdict['mower_used'] == 1
    verdict = judge(_Env([]), {'tick': 1300, 'result': 2}, [{'id': 7}], 3, 3,
                    horizon=1200, root_tick=100)
    assert not verdict['success'] and verdict['house_breach'] is True


def test_drill_budget_cap_limits_games_per_round():
    # 20 drill cells, a 200-game round, group=5, cap=5% => at most 10 drill games.
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / 'archive'
        archive.mkdir()
        (archive / 'meta.json').write_text(json.dumps({'verified': False, 'task': {'seed': 42}}))
        cells = {}
        for index in range(20):
            cell = drill_cell(decision=300 + index, level=31)
            cell['task_id'] = f'drill235_l31_jack_{index}'
            cell['archive'] = str(archive)
            cells[cell['task_id']] = cell
        state = dict(round=0, offsets={}, cells=cells)
        tasks = {base: {'seeds': [42]} for base in cells}
        manifests = {base: Path(tmp) / 'task235_drills.json' for base in cells}
        plan, _ = plan_round(state, tasks, NAMES, VALUES, manifests, budget=200, per_group=20,
                             drill_share=0.05, drill_games=5)
        drill_games = [item['games'] for item in plan if item['base'] in cells]
        assert sum(drill_games) <= 10
        assert drill_games and all(games <= 5 for games in drill_games)


def test_drill_rows_get_the_existing_load_time_weight_and_no_new_pack_key():
    weights = torch.ones(6)
    plan = DemoRowWeights()
    plan.observe(Path('/frames/seed_1/c1drill_fast_search/frames.npz'),
                 [0, 1, 2, 3, 4, 5],
                 dict(drill={'horizon': 1200}, label_source='teacher', demo_archive='/archive'))
    plan.apply(weights)
    assert weights.tolist() == [3.0] * 6
    assert plan.rows['drill'] == 6
    assert plan.metadata()['drill_row_weight'] == 3.0
    # The pack contract (and therefore the shared A/B pack key) is untouched.
    base = dict(schema_version=1, kind='training', data='/data', out='/out', labels=['a'])
    assert training_row_options(validate(dict(base))) == training_row_options(
        validate(dict(base, drill_row_weight=4.0)))
    for value in (float('nan'), float('inf'), -1, True, '3'):
        with pytest.raises(ValueError):
            validate(dict(base, drill_row_weight=value))


def test_registered_drill_manifest_covers_every_fog_level():
    path = ROOT / 'configs/task235_drills.json'
    cells, manifests = load_demo_start_cells(path)
    assert len(cells) >= 60
    assert {cell['level'] for cell in cells.values()} == {31, 32, 33, 34, 36, 37, 38, 39, 40}
    for task_id, cell in cells.items():
        assert cell['family'] == 'drill_start' and cell['drill'] is True
        assert cell['horizon'] == 1200 and len(cell['ladder']) == 1
        assert cell['targets'] and cell['seeds']
        assert Path(cell['archive']).is_absolute()
        assert manifests[task_id] == path


def test_manifest_task_reproduces_the_archive_environment():
    meta = {'task': {'id': 'rd193_l31', 'seed': 200020618, 'level': 31,
                     'deck': [1, 8, 16, 10, 42, 30, 14], 'zombie_count_multiplier': 1.5,
                     'first_aid_owned': False},
            'environment_task': {'level': 31, 'fog_observation': 'human_v2',
                                 'wave_cap': 20}}
    drill = {'id': 'framesv2:a:r10', 'level': 31, 'decision': 11, 'terrain': 'fog',
             'horizon': 1200, 'criterion': {'threats_neutralized': True},
             'targets': [{'id': 7, 'name': 'jack'}], 'trigger': {'name': 'jack'}}
    task = task_for(drill, meta, Path('/archive'), future_seed_start=53000000,
                    future_seed_count=256, horizon=None)
    assert task['level'] == 31 and task['seeds'] == [200020618]
    assert task['fog_observation'] == 'human_v2'
    assert task['drill_start']['archive'] == '/archive'
    assert task['drill_start']['horizon'] == 1200
    assert task['drill_start']['future_seed_start'] == 53000000
