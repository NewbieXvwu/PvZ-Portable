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
