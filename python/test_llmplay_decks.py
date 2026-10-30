"""task186 的任务逻辑测试；不启动引擎，不重放存档。"""
import itertools
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch, Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
_TEMP = tempfile.TemporaryDirectory(prefix='llmplay-decks-test-')
os.environ['LLMPLAY_OUT'] = str(Path(_TEMP.name) / 'initial-out')
os.environ['PVZ_LLMPLAY_SESSION'] = str(Path(_TEMP.name) / 'budget.json')
from llmplay import decks, workflow, delivery, Game, budget
from llmplay.game import load_tasks, task_spec
from llmplay.build_package import task_manifest
from pvz_tasks import player_deck_levels


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv('LLMPLAY_SHARD', '0')
    monkeypatch.setenv('LLMPLAY_SHARDS', '1')
    monkeypatch.setenv('LLMPLAY_OUT', str(tmp_path / 'out'))
    budget.reset_session(tmp_path / 'budget.json')


def entry(level, cards, index=0, *, won=True, result=None, task_id=None, **kw):
    return dict(task_id=task_id or f'llmplay_decks_l{level}_m100', level=level,
                seed=decks.seed_at(level, index), deck=list(cards), won=won,
                result=(1 if won else 2) if result is None else result,
                valid=True, abandoned=False, archive=f'archives/l{level}_{index}',
                outcome={'final_tick': 100}, note='', **kw)


D1 = [0, 1, 2, 3, 4, 6]
D2 = [0, 1, 2, 3, 7, 17]


def test_decks_tasks_and_shards_share_original_engine_rules(monkeypatch):
    tasks = load_tasks()
    expected = {n for n in range(1, 50) if n % 5}
    assert set(player_deck_levels()) == expected
    assert {tasks[key]['level'] for key in decks.config()['order']} == expected
    assert all(tasks[key]['zombie_count_multiplier'] == 1.0 and not tasks[key].get('wave_cap')
               for key in decks.config()['order'])
    assert {t['task_id'] for t in task_manifest()['tasks']} >= set(decks.config()['order'])
    for count in (3, 7):
        assigned = []
        for index in range(count):
            decks.configure_shard(index, count)
            assigned.append(set(decks.assigned_tasks()))
        assert set.union(*assigned) == set(decks.config()['order'])
        assert sum(map(len, assigned)) == 40
        if count == 3:
            assert {decks.region(tasks[key]['level']) for key in assigned[0]} == {'day', 'night'}
            assert {decks.region(tasks[key]['level']) for key in assigned[2]} == {'roof'}
    spec, cards, _ = task_spec(tasks['llmplay_decks_native_l50_m100'])
    assert spec.level == 50 and spec.zombie_count_multiplier == 1.0
    assert cards == [] and spec.conveyor_start == 'original'


def test_decks_one_win_registers_and_two_card_difference_applies_to_history():
    history = entry(1, D1, task_id='llmplay3_a_l2_m100')
    doc = workflow.derive([history], 'run')
    info = doc['levels']['llmplay_decks_l1_m100']
    assert info['victory_decks'] == 1
    assert workflow.card_diff(D1, D2) == 2
    seed = info['available_seeds'][0]
    with pytest.raises(ValueError, match='至少差 2'):
        workflow.check(doc, 'llmplay_decks_l1_m100', D1[::-1], seed=seed)
    with pytest.raises(ValueError, match='至少差 2'):
        workflow.check(doc, 'llmplay_decks_l1_m100', [0, 1, 2, 3, 4, 7], seed=seed)
    assert workflow.check(doc, 'llmplay_decks_l1_m100', D2, seed=seed)['workflow_layer'] == 'decks'


def test_decks_holdout_signature_rejected_on_other_level():
    held = json.loads((ROOT / 'configs/deploy17_deck_holdouts.json').read_text())['tasks'][0]['deck']
    doc = workflow.derive([], 'run')
    with pytest.raises(ValueError, match='留出'):
        workflow.check(doc, 'llmplay_decks_l4_m100', held[::-1], seed=decks.seed_at(4, 0))
    with pytest.raises(ValueError, match='黑名单'):
        workflow.check(doc, 'llmplay_decks_l1_m100', [0, 1, 2, 3, 4, 38], seed=decks.seed_at(1, 0))


def test_decks_three_losses_reason_stored_in_third_archive_note():
    task_id = 'llmplay_decks_l1_m100'
    failures = [entry(1, D1, n, won=False) for n in range(2)]
    doc = workflow.derive(failures, 'run')
    candidate = next(item for item in decks.items(doc) if item['task_id'] == task_id)
    assert candidate['action'] == 'retry_deck' and candidate['losses'] == 2
    assert candidate['seed'] == decks.seed_at(1, 2)
    workflow.check(doc, task_id, D1, seed=candidate['seed'])
    with patch.object(workflow, '_progress', return_value=doc):
        with pytest.raises(ValueError, match='第三次败局'):
            workflow.archive_requirements(task_id, layer='decks', deck=D1, result=2, reason='经济与主力')
        workflow.archive_requirements(task_id, layer='decks', deck=D1, result=2,
                                      reason='经济与主力', note='后半场群伤不足')
    failures.append(dict(entry(1, D1, 2, won=False), note='后半场群伤不足'))
    doc = workflow.derive(failures, 'run')
    assert '后半场群伤不足' in workflow.report(doc)
    with pytest.raises(ValueError, match='三败'):
        workflow.check(doc, task_id, D1, seed=decks.seed_at(1, 3))
    workflow.check(doc, task_id, D2, seed=decks.seed_at(1, 3))


def test_decks_duplicate_exports_partial_and_abandon_do_not_inflate_losses():
    partial = entry(1, D1, won=False, result=0)
    lost = entry(1, D1, won=False)
    abandoned = dict(entry(1, D1, 1, won=False, result=0), abandoned=True)
    doc = workflow.derive([partial, lost, lost, abandoned], 'run')
    group = doc['levels']['llmplay_decks_l1_m100']['decks'][workflow.deck_key(D1)]
    assert group['losses'] == 1
    assert doc['levels']['llmplay_decks_l1_m100']['pending'] is None
    won = entry(1, D1)
    doc = workflow.derive([partial, won, won], 'run')
    assert doc['levels']['llmplay_decks_l1_m100']['victory_decks'] == 1


def test_decks_size_quotas_and_coverage_are_distinct_card_sets():
    task_id = 'llmplay_decks_l1_m100'
    small = [entry(1, cards, n) for n, cards in enumerate((D1, D2, [0, 1, 2, 4, 7, 17, 23]))]
    doc = workflow.derive(small, 'run')
    assert doc['levels'][task_id]['required_lengths'] == [8]
    with pytest.raises(ValueError, match='大小'):
        workflow.check(doc, task_id, [1, 3, 6, 17, 23, 29], seed=decks.seed_at(1, 3))
    # 同一组跨关获胜只算一组覆盖；两个签名都带向日葵则计两组。
    doc = workflow.derive([entry(1, D1), entry(2, D1), entry(2, D2, 1)], 'run')
    assert doc['coverage']['day']['counts']['1'] == 2
    assert doc['coverage']['day']['counts']['4'] == 1
    assert '1' not in doc['coverage']['day']['missing']


def test_decks_goal_grows_and_seed_windows_extend_without_materializing_seeds():
    # 基础目标和大小配额已满，仍然有下一项并提高目标。
    pool = [0, 2, 3, 4, 6, 7, 17, 18, 20, 22, 23, 26, 28, 29, 30, 32, 34, 36, 39]
    cards = []
    for length, count in ((6, 3), (8, 3), (9, 6)):
        cards.extend([[1] + list(combo) for combo in itertools.islice(itertools.combinations(pool, length - 1), count)])
    # 支线已完成（此夹具没有分配支线），本片覆盖完成后主线继续加4组。
    with patch.object(decks, 'assigned_tasks', side_effect=lambda native=False: [] if native else ['llmplay_decks_l1_m100']), \
            patch.dict(decks.config(), coverage_per_card=0):
        doc = workflow.derive([entry(1, deck, n) for n, deck in enumerate(cards)], 'run')
    info = doc['levels']['llmplay_decks_l1_m100']
    assert info['baseline_complete'] and info['goal'] == 16
    assert workflow.next_task(doc, claim=False)['item'] is not None
    assert doc['complete'] is False
    last = decks.config()['seed_window'] - 1
    doc = workflow.derive([entry(1, D1, last, won=False)], 'run')
    next_seed = doc['levels']['llmplay_decks_l1_m100']['available_seeds'][0]
    assert next_seed == decks.seed_at(1, last + 1)
    assert next_seed > decks.seed_at(50, 0)
    assert decks.seed_index(1, next_seed) == last + 1
    assert decks.seed_index(2, next_seed) is None


def test_decks_claims_progress_and_resume_preserve_registry(tmp_path):
    archive = tmp_path / 'episode'
    archive.mkdir()
    meta = dict(task=dict(id='llmplay_decks_l1_m100', level=1, seed=decks.seed_at(1, 0),
                          deck=D1, zombie_count_multiplier=1.0),
                outcome=dict(result=1, won=True, final_tick=100),
                deck_reason='向日葵经济、豌豆主力、坚果防守、樱桃应急、草地无专用卡',
                workflow_layer='decks')
    (archive / 'meta.json').write_text(json.dumps(meta))
    for filename in ('ticks.jsonl', 'frames.jsonl'):
        (archive / filename).write_text('')
    delivery.record_archive(archive)
    saved = tmp_path / 'saved.zip'
    saved.write_bytes((delivery.output_dir() / 'out.zip').read_bytes())
    first = delivery.next_task()['item']
    second = delivery.next_task()['item']
    assert (first['task_id'], first['seed']) != (second['task_id'], second['seed'])
    old_out = os.environ['LLMPLAY_OUT']
    os.environ['LLMPLAY_OUT'] = str(tmp_path / 'restored')
    try:
        with patch('llmplay.game.PvZEnv', side_effect=AssertionError('must not start engine')):
            delivery.resume(saved)
        assert delivery.progress()['levels']['llmplay_decks_l1_m100']['victory_decks'] == 1
    finally:
        os.environ['LLMPLAY_OUT'] = old_out


def test_decks_restoring_claim_does_not_let_search_branch_release_parent(tmp_path):
    task_id = 'llmplay_decks_l1_m100'
    primary = Game(task_id, seed=decks.seed_at(1, 0), deck=D1, reason='经济与主力')
    saved = tmp_path / 'game.json'
    primary.save(saved)
    with patch.object(primary, '_start'):
        child = primary.branch()
    assert not child._claimed
    child.close()
    assert workflow._slot(task_id, primary.seed) in workflow.held()
    primary.close()
    restored = Game.load(saved)
    assert restored._claimed
    restored.close()
    assert workflow._slot(task_id, primary.seed) not in workflow.held()


def test_decks_metadata_only_carry_survives_out_zip_resume(tmp_path):
    source = tmp_path / 'carry.json'
    prior = workflow.derive([entry(1, D1)], 'carry-run')
    source.write_text(json.dumps(prior))
    delivery.carry(source)
    assert delivery.progress()['archives'][0]['carried']
    saved = tmp_path / 'carry-out.zip'
    saved.write_bytes((delivery.output_dir() / 'out.zip').read_bytes())
    previous_out = os.environ['LLMPLAY_OUT']
    os.environ['LLMPLAY_OUT'] = str(tmp_path / 'resumed-carry')
    try:
        with patch('llmplay.game.PvZEnv', side_effect=AssertionError('must not replay')):
            delivery.resume(saved)
        assert delivery.progress()['levels']['llmplay_decks_l1_m100']['victory_decks'] == 1
    finally:
        os.environ['LLMPLAY_OUT'] = previous_out


def native_entry(level, index=0, won=True):
    return dict(entry(level, [], index, won=won),
                task_id=f'llmplay_decks_native_l{level}_m100',
                seed=decks.seed_at(level, index, native=True), conveyor_start='original',
                fog_observation='human_v2' if level == 40 else None)


def test_decks_fog_specs_and_unknown_history_do_not_count_as_public_wins():
    tasks = load_tasks()
    fog_ids = [key for key in decks.config()['order'] + decks.config()['native_order']
               if decks.region(tasks[key]['level']) == 'fog']
    assert {tasks[key]['level'] for key in fog_ids} == {31, 32, 33, 34, 36, 37, 38, 39, 40}
    for key in fog_ids:
        cards = [] if tasks[key].get('conveyor_start') else [1, 8, 9, 10, 16, 25]
        assert task_spec(tasks[key], deck=cards)[0].fog_observation == 'human_v2'
    old = entry(31, [1, 8, 9, 10, 16, 25])
    doc = workflow.derive([old], 'run')
    info = doc['levels'][old['task_id']]
    assert info['victory_decks'] == 0 and info['registered_decks']
    doc = workflow.derive([dict(old, fog_observation='human_v2')], 'run')
    assert doc['levels'][old['task_id']]['victory_decks'] == 1


def test_decks_native_is_lower_priority_and_eight_distinct_seeds_complete():
    doc = workflow.derive([], 'run')
    assert workflow.next_task(doc, claim=False)['item']['layer'] == 'decks'
    claimed = {workflow._slot(item['task_id'], item['seed']): {} for item in decks.items(doc)}
    with patch.object(workflow, '_read_claims', return_value=claimed):
        item = workflow.next_task(doc, claim=False)['item']
        assert item['layer'] == 'native' and item['action'] == 'native_play'
        workflow.check(doc, item['task_id'], [], seed=item['seed'])
        with pytest.raises(ValueError, match='不能自选'):
            workflow.check(doc, item['task_id'], D1, seed=item['seed'])
    with pytest.raises(ValueError, match='主线优先'):
        workflow.check(doc, item['task_id'], [], seed=item['seed'])
    repeated = [native_entry(10)] * 8
    assert workflow.derive(repeated, 'run')['native']['llmplay_decks_native_l10_m100']['wins'] == 1
    doc = workflow.derive([native_entry(10, n) for n in range(8)], 'run')
    assert doc['native']['llmplay_decks_native_l10_m100']['complete']
    assert all(item['level'] != 10 for item in decks.native_items(doc))
    assert '原生传送带支线' in workflow.report(doc)


def test_decks_native_shards_and_seed_windows_are_separate():
    tasks = load_tasks()
    sets = []
    for index in range(3):
        decks.configure_shard(index, 3)
        sets.append({tasks[key]['level'] for key in decks.assigned_tasks(native=True)})
    assert sets == [{5, 10, 20}, {25, 30, 40}, {45, 50}]
    for level in (5, 40, 50):
        for index in (0, decks.config()['seed_window']):
            seed = decks.seed_at(level, index, native=True)
            assert decks.seed_index(level, seed, native=True) == index
            assert decks.seed_index(level, seed) is None


def test_decks_fog_observe_summary_and_disappearance_remain_unknown():
    from llmplay import scene
    obs = dict(level=31, tick=1200, wave=1, wave_count=10, sun=50, fog=True,
               grid=[[1] * 9 for _ in range(6)], fog_amount=[[0] * 6 + [200] * 3 for _ in range(6)],
               plants=[dict(type=15, row=1, col=7, health=-1, max_health=-1, state=-1,
                            state_countdown=-1, asleep=True, wake_up_counter=987, squished=False)],
               zombies=[], packets=[], defenses=[], result=0, terminal=False)
    g = Game('llmplay_decks_l31_m100', deck=[1, 8, 9, 10, 16, 25], reason='雾关试卡', _gate=False)
    g.env = Mock(episode={})
    g.env.observe.return_value = obs
    g.obs = obs
    with patch.object(g, '_start'):
        text = g.observe()
        summary = g.summary()
    assert '看不见' in text and 'Doo?' in text
    assert '-1/-1' not in text and '987' not in text
    assert summary['zombies'] is None and summary['visible_zombies'] == 0
    assert '看不见' in summary['unseen_plants'][0]['status']
    assert summary['lanes'][1]['zombies'] is None
    g._note_events({'type': 'wait'}, True, {}, {1: (0, 1, 600, True, False)}, {}, before_wave=1)
    assert any(event['kind'] == 'unseen' for event in g.events)
    assert not any(event['kind'] == 'kill' for event in g.events)
    g.env = None
    g.close()


def test_decks_export_records_distinct_seed_attempts_and_restores_them(tmp_path):
    failures = [entry(1, D1, n, won=False) for n in range(2)]
    # 包括放弃/中途尝试；重复导出和相同种子不重复计数。
    assert decks.attempt_seeds(failures + failures, failures[0]['task_id'], D1, decks.seed_at(1, 2)) == [
        decks.seed_at(1, n) for n in range(3)]
    g = Game('llmplay_decks_l1_m100', seed=decks.seed_at(1, 2), deck=D1, reason='经济主力与防守', _gate=False)
    g.env = SimpleNamespace(episode={})
    g.obs = dict(result=1, tick=100, wave=10, wave_count=10, terminal=True)
    g.initial_obs = {}
    archive = tmp_path / 'winning'
    with patch.object(delivery, '_read_progress', return_value={'archives': failures}):
        g.export_archive(archive)
    meta = json.loads((archive / 'meta.json').read_text())
    assert meta['deck_seed_attempts'] == 3
    assert meta['deck_attempt_seeds'] == [decks.seed_at(1, n) for n in range(3)]
    restored = Game.load_archive(archive)
    assert restored.deck_attempt_seeds == meta['deck_attempt_seeds']
    g.env = None
    g.close()
    restored.close()


def test_decks_native_archive_uses_original_cards_without_fixed_deck_index(tmp_path):
    g = Game('llmplay_decks_native_l40_m100', seed=decks.seed_at(40, 0, native=True), _gate=False)
    g.env = SimpleNamespace(episode={})
    g.obs = dict(result=1, tick=100, wave=10, wave_count=10, terminal=True)
    g.initial_obs = {}
    g.ops = [dict(action={'type': 'plant', 'packet': 7, 'row': 0, 'col': 0}, tick=0, end_tick=0, ok=True)]
    archive = tmp_path / 'native-win'
    g.export_archive(archive)
    meta = json.loads((archive / 'meta.json').read_text())
    assert meta['task']['deck'] == [] and meta['opening'] == []
    assert meta['environment_task']['conveyor_start'] == 'original'
    assert meta['environment_task']['fog_observation'] == 'human_v2'
    assert delivery.progress()['native']['llmplay_decks_native_l40_m100']['wins'] == 1
    g.env = None
    g.close()
