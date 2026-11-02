"""task222: the v8 gate panel, its generator and fresh calibration."""
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'python')]

import make_gate_v8_panel as maker
import pvz_promotion as promotion
import pvz_tasks
import curriculum_queue as queue
import promotion_gate as gate
from types import SimpleNamespace

PANEL = ROOT / 'configs/gate_v8_panel.json'
V5_PANEL = ROOT / 'configs/gate_v5_panel.json'


def panel():
    return json.loads(PANEL.read_text())


def test_registered_panel_covers_zones_and_uses_user_decks_first():
    document = panel()
    assert document['panel_version'] == 'v8'
    assert document['zone_counts'] == dict(day=8, night=8, pool=9, fog=8, roof=8, heldout=2)
    user = [task for task in document['tasks'] if task.get('deck_source') == 'user']
    assert len(user) == 8
    l29 = [task['deck'] for task in user if task['level'] == 29]
    assert l29 == [[1, 16, 35, 10, 42, 39, 44, 30, 21, 17],
                   [1, 7, 40, 22, 23, 19, 16, 17, 21, 46]]
    for task in document['tasks']:
        if task.get('deck_source') == 'task208':
            assert tuple(task['deck']) == pvz_tasks.GOOD_DECKS[task['level']][0]
        if task.get('deck'):
            pvz_tasks.profile_for_deck(tuple(task['deck']))
            assert pvz_tasks.terrain_covered(task['level'], task['deck'])


def test_generator_picks_up_a_new_user_recording(tmp_path):
    meta = tmp_path / 'user_l8_deadbeef' / 'seed_1' / 'user_l8_deadbeef' / 'meta.json'
    meta.parent.mkdir(parents=True)
    meta.write_text(json.dumps({'task': {'level': 8, 'deck': [0, 1, 3, 4, 5, 20], 'seed': 1}}))
    generated = maker.build(tmp_path, V5_PANEL, None)
    tasks = [task for task in generated['tasks'] if task['level'] == 8]
    assert len(tasks) == 1 and tasks[0]['deck_source'] == 'user'
    assert tasks[0]['deck'] == [0, 1, 3, 4, 5, 20]
    assert generated['qualification'][tasks[0]['task_id']]['recording_seed'] == 1


def test_v8_calibration_and_freeze_use_the_panel_multipliers():
    document = panel()
    calibration = promotion.initial_panel(document)
    assert calibration['task222_l16_night_user']['multiplier'] == \
        document['initial_multipliers']['16']
    assert calibration['task222_l49_roof_user']['low_win'] is True
    total = promotion.policy(document)['pair_count']
    block = (100_900_000, 100_900_000 + total)
    manifest = promotion.freeze_panel(
        document, block, calibration, json.loads((ROOT / 'configs/human_demo_tasks.json').read_text()))
    assert manifest['panel_version'] == 'v8-task222'
    assert sum(len(task['seeds']) for task in manifest['tasks']) == total
    by_id = {task['task_id']: task for task in manifest['tasks']}
    for task in document['tasks']:
        name = task['task_id']
        if name in document['monitor']:
            assert len(by_id[name]['seeds']) == document['monitor_seeds_per_task']
        else:
            assert by_id[name]['zombie_count_multiplier'] == calibration[name]['multiplier']
    for name in document['monitor']:
        assert manifest['groups'][name] == 'heldout'
    roof = [task for task in manifest['tasks'] if manifest['groups'][task['task_id']] == 'roof']
    assert roof and all(task['endpoint'] == 'waves' for task in roof)


def test_reconcile_gate_panel_resets_v5_and_keeps_same_version_progress():
    document = panel()
    state = dict(gate_panel={'task101_l1_day_d7': {'multiplier': 3.0}}, gate_panel_version='v5')
    queue.reconcile_gate_panel(state, document)
    assert state['gate_panel_version'] == 'v8'
    assert 'task101_l1_day_d7' not in state['gate_panel']
    key = 'task222_l16_night_user'
    state['gate_panel'][key]['multiplier'] = 3.0
    extended = copy.deepcopy(document)
    extra = copy.deepcopy(next(task for task in document['tasks'] if task['level'] == 8))
    extra.update(task_id='task222_l08_day_extra', deck=[0, 1, 3, 4, 5, 20], deck_source='task208',
                 pairs=45)
    extended['tasks'].append(extra)
    extended['groups'][extra['task_id']] = 'day'
    queue.reconcile_gate_panel(state, extended)
    assert state['gate_panel_version'] == 'v8'
    assert state['gate_panel'][key]['multiplier'] == 3.0
    assert extra['task_id'] in state['gate_panel']


def test_v8_sequential_evaluate_certificate(tmp_path, monkeypatch):
    document = panel()
    settings = promotion.initial_panel(document)
    args = SimpleNamespace(
        candidate=tmp_path / 'published.pt', incumbent=tmp_path / 'incumbent.pt',
        candidates={'A': str(tmp_path / 'A.pt'), 'B': str(tmp_path / 'B.pt')},
        calibration=settings, source_panel=PANEL, executable=tmp_path / 'engine',
        value=tmp_path / 'value.pt', resource_dir=tmp_path / 'resources', workers=1,
        eval_units=2, out=tmp_path / 'eval', seed_ledger=tmp_path / 'ledger.json',
        seed_block=None, native_games=False)
    args.incumbent.write_bytes(b'inc')
    calls = []

    def run(args, lock, manifest_path, task, side, model):
        calls.append((side, task['task_id'], len(task['seeds'])))
        wins = {'incumbent': 5, 'A': 7, 'B': 9}[side]
        return side, task['task_id'], {
            seed: dict(won=index % 12 < wins, final_wave=20 if index % 12 < wins else 5,
                       wave_count=20) for index, seed in enumerate(task['seeds'])}

    monkeypatch.setattr(gate, '_run_unit', run)
    result = gate.evaluate(args)
    assert result['sequential']['look'] == 1
    assert result['sequential']['adventure_pairs'] == 492
    assert result['overall']['pairs'] == 492
    assert gate.valid_certificate(result, result['contract'])
    assert {side for side, _, _ in calls} == {'incumbent', 'A', 'B'}


def _pair_rows(task_id, zone, count, endpoint, candidate_wins, incumbent_wins, waves=(10, 10)):
    rows = []
    for index in range(count):
        candidate = index < candidate_wins
        incumbent = index < incumbent_wins
        if endpoint == 'waves':
            candidate_wave, incumbent_wave = waves
        else:
            candidate_wave = 20 if candidate else 0
            incumbent_wave = 20 if incumbent else 0
        rows.append(dict(task_id=task_id, zone=zone, seed=index, candidate=candidate,
                         incumbent=incumbent, candidate_wave=candidate_wave,
                         incumbent_wave=incumbent_wave, wave_count=20, endpoint=endpoint))
    return rows


def test_next_panel_follows_the_promoted_side():
    manifest = dict(groups={'fog_task': 'fog', 'night_task': 'night'},
                    tasks=[dict(task_id='fog_task', gate_key='fog_task', seeds=[0, 1]),
                           dict(task_id='night_task', gate_key='night_task', seeds=[0, 1])])
    calibration = {'fog_task': dict(multiplier=1., low_win=True),
                   'night_task': dict(multiplier=1., low_win=False)}
    outcomes = {
        'A': {'fog_task': {0: {'won': True}, 1: {'won': False}},
              'night_task': {0: {'won': True}, 1: {'won': False}}},
        'B': {'fog_task': {0: {'won': False}, 1: {'won': False}},
              'night_task': {0: {'won': False}, 1: {'won': False}}},
        'incumbent': {'fog_task': {0: {'won': False}, 1: {'won': False}},
                      'night_task': {0: {'won': True}, 1: {'won': True}}}}
    promoted = promotion.next_panel(calibration, manifest, outcomes, 'A')
    stale = promotion.next_panel(calibration, manifest, outcomes)
    assert promoted['fog_task']['low_win'] is False
    assert promoted['fog_task']['multiplier'] == 1.
    assert promoted['fog_task']['rates']['A'] == .5
    assert stale['fog_task']['low_win'] is True
    assert stale['fog_task']['multiplier'] == .75


def test_recalibrate_panel_reuses_stored_winner_rates():
    pre = {'fog_task': dict(multiplier=1., low_win=True)}
    post = {'fog_task': dict(multiplier=.75, low_win=True,
                             rates={'A': .5, 'B': .1, 'incumbent': 0.})}
    fixed = promotion.recalibrate_panel(pre, post, 'A')
    assert fixed['fog_task']['low_win'] is False
    assert fixed['fog_task']['multiplier'] == 1.


def test_startup_recalibration_follows_current_student():
    pre = {'fog_task': dict(multiplier=1., low_win=True)}
    state = dict(student_round=95, gate_panel={
                    'fog_task': dict(multiplier=.75, low_win=True,
                                     rates={'A': .5, 'B': .1, 'incumbent': 0.})},
                training_jobs=[dict(round=95, promotion_approved=True, timing={'promotion': {
                    'winner_family': 'A', 'contract': {'version': 8, 'calibration': pre}}})])
    queue._recalibrate_current_incumbent(state, {'panel_version': 'v8'})
    assert state['gate_panel_student_round'] == 95
    assert state['gate_panel']['fog_task']['low_win'] is False
    assert state['gate_panel']['fog_task']['multiplier'] == 1.


def test_win_guard_vetoes_a_significant_loss_even_when_waves_are_flat():
    pairs = []
    pairs += _pair_rows('day_task', 'day', 96, 'win', 96, 88)
    pairs += _pair_rows('night_task', 'night', 96, 'win', 96, 96)
    pairs += _pair_rows('pool_task', 'pool', 96, 'win', 96, 96)
    pairs += _pair_rows('fog_wave', 'fog', 84, 'waves', 84, 84)
    pairs += _pair_rows('fog_win', 'fog', 12, 'win', 0, 8)
    manifest = dict(monitor=[], groups={'day_task': 'day', 'night_task': 'night',
                                        'pool_task': 'pool', 'fog_wave': 'fog', 'fog_win': 'fog'})
    result = promotion.decide(pairs, manifest, 1)
    assert result['zones']['fog']['win_guard_veto'] is True
    assert result['zones']['fog']['endpoint_used'] == 'win'
    assert 'significant_win_regression:fog' in result['reasons']
    assert result['approved'] is False


def test_win_guard_leaves_waves_alone_when_wins_are_too_rare():
    pairs = []
    pairs += _pair_rows('fog_wave', 'fog', 84, 'waves', 84, 84)
    pairs += _pair_rows('fog_win', 'fog', 12, 'win', 0, 3)
    manifest = dict(monitor=[], groups={'fog_wave': 'fog', 'fog_win': 'fog'})
    result = promotion.decide(pairs, manifest, 1)
    assert result['zones']['fog']['win_guard_veto'] is False
    assert result['zones']['fog']['endpoint_used'] == 'waves'
    assert 'significant_win_regression:fog' not in result['reasons']
