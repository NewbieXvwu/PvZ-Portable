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
import drill_mine
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


def test_reference_terminal_score_uses_frozen_native_policy_and_restores_horizon():
    class ReferenceEnv:
        def __init__(self):
            self.calls = []

        def snapshot(self):
            return 7

        def load_rollout_net(self, path):
            self.calls.append(('load', path))

        def rollout(self, action, until, **kwargs):
            assert action == {'type': 'wait', 'ticks': 60} and until == 0
            assert kwargs['student_history'] == 'history'
            self._last_response = {'mowers_left': 1}
            return dict(tick=40000, result=1, terminal=True, wave=20, wave_count=20)

        def rollout_predict(self, previous_tick, previous_plant, **kwargs):
            assert previous_tick == 8940 and previous_plant is True
            return {'action': {'type': 'wait', 'ticks': 60}}

        def restore(self, snapshot):
            self.calls.append(('restore', snapshot))

        def release_snapshot(self, snapshot):
            self.calls.append(('release', snapshot))

    env = ReferenceEnv()
    obs = dict(tick=9000, result=0, terminal=False)
    verdict = judge(env, obs, [], 4, 3, horizon=6000, root_tick=3000,
                    reference_policy='frozen.bin', actor_policy='teacher.bin', student_history='history',
                    previous_tick=8940, previous_plant=True)
    assert verdict['success'] and verdict['mower_used'] == 3
    assert verdict['final_tick'] == obs['tick'] == 9000
    assert verdict['reference_final_tick'] == 40000 and verdict['scoring'] == 'reference_terminal'
    assert env.calls == [('load', 'frozen.bin'), ('restore', 7), ('release', 7), ('load', 'teacher.bin')]


def test_family_mining_merges_upgrade_chains_at_one_root_and_starts_before_counter(tmp_path, monkeypatch):
    import drill_mine
    archive = tmp_path / 'episode'
    archive.mkdir()
    (archive / 'meta.json').write_text('{}')
    frames = [dict(tick=tick, scene={'level': 22}, packets=[[0, 1], [1, 41], [2, 7], [3, 40]],
                   plants=[[0, 0, 1]], zombies=([[0, 7, 790, 0, 100, True]] if tick == 6000 else []))
              for tick in [0, 1200, 3000, 4200, 6000]]
    monkeypatch.setattr(drill_mine, 'frame_path', lambda path: path)
    monkeypatch.setattr(drill_mine, 'read_frames', lambda path: frames)
    drills = drill_mine.mine_families(dict(path=archive, archive='episode', source_root='main'))
    upgrade = [d for d in drills if d['family'] == 'upgrade']
    prep = [d for d in drills if d['family'] == 'preparation']
    assert len(upgrade) == 1 and len(upgrade[0]['trigger']['chains']) == 2
    assert len(prep) == 1 and prep[0]['root_tick'] == 4200
    assert all(d['scoring'] == 'reference_terminal' and d['targets'] == [] for d in drills)


def test_reference_manifest_loads_without_rescue_targets(tmp_path):
    meta = dict(task=dict(seed=42, level=21, deck=[0, 1, 2, 3, 16, 43]))
    drill = dict(id='pool_chain', family='upgrade', level=21, decision=50, terrain='pool',
                 horizon=6000, scoring='reference_terminal', criterion={'scoring': 'reference_terminal'},
                 reference_policy='/frozen/r112/model.pt', reference_sha256='frozen', targets=[])
    task = task_for(drill, meta, Path('/archive'), future_seed_start=54000000,
                    future_seed_count=256, horizon=None)
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps({'tasks': [task]}))
    cells, _ = load_demo_start_cells(path)
    assert cells[task['task_id']]['family'] == 'drill_start'
    assert cells[task['task_id']]['horizon'] == 6000
    assert task['drill_start']['reference_policy'] == '/frozen/r112/model.pt'
    assert not cells[task['task_id']]['targets']


def test_drill_calibration_uses_teacher_results_only(tmp_path, monkeypatch):
    import drill_select
    bank = tmp_path / 'bank.json'
    results = tmp_path / 'results.jsonl'
    out = tmp_path / 'selected.json'
    bank.write_text(json.dumps({'drills': [dict(id=key, level=21) for key in ['zero', 'mixed']]}))
    rows = [dict(status='ok', drill_id=key, arm=arm, seed=seed, success=win)
            for key, teacher, student in [('zero', [0, 0, 0], [1, 1, 1]),
                                          ('mixed', [1, 0, 0], [0, 0, 0])]
            for arm, wins in [('teacher', teacher), ('student', student)]
            for seed, win in enumerate(wins)]
    results.write_text('\n'.join(json.dumps(row) for row in rows))
    monkeypatch.setattr(sys, 'argv', ['drill_select', '--bank', str(bank), '--results', str(results),
                                     '--out', str(out)])
    drill_select.main()
    selected = json.loads(out.read_text())['drills']
    assert [d['id'] for d in selected] == ['mixed']
    assert selected[0]['teacher']['games'] == 3


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


@pytest.mark.parametrize('include_human', [False, True])
def test_mined_drill_ids_distinguish_seeds_with_the_same_archive_name(tmp_path, monkeypatch,
                                                                   include_human):
    frames = [dict(tick=index * 60, scene={'level': 31}, lanes=[0],
                   plants=[[0, 1, 1]], zombies=[[0, 18, 250., 0, 200, True, 0, 0, 0, 7]],
                   mowers=[[0, 1]]) for index in range(12)]
    monkeypatch.setattr(drill_mine, 'read_frames', lambda _path: frames)
    sequence = dict(root_frame=0, decision=1, row=0, removed_type=1,
                    replanted={'type': 1}, complete=True)
    monkeypatch.setattr(drill_mine, 'mine_human_sequences', lambda *_args: [sequence.copy()])
    mined = []
    for seed in (1, 2):
        relative = f'seed_{seed}/same_label'
        archive = tmp_path / relative
        archive.mkdir(parents=True)
        (archive / 'meta.json').write_text(json.dumps({'label_source': 'human'}))
        drills, _, _ = drill_mine.mine_archive(
            dict(path=str(archive), source_root='framesv2', archive=relative),
            max_per_archive=1, horizon=1200, lookback=0, min_gap=0,
            include_human=include_human)
        assert len(drills) == (2 if include_human else 1)
        mined.extend(drills)
    assert len({drill['id'] for drill in mined}) == len(mined)
    tasks = [task_for(drill, {'task': {'seed': 42, 'deck': [1, 8, 16, 10, 42, 30]}},
                      Path(drill['archive']), future_seed_start=53000000,
                      future_seed_count=256, horizon=None) for drill in mined]
    assert len({task['task_id'] for task in tasks}) == len(tasks)
