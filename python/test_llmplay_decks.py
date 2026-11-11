"""task186 的任务逻辑测试；不启动引擎，不重放存档。"""
import itertools
import json
import os
from pathlib import Path
import sys
import tempfile
from contextlib import contextmanager
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


@contextmanager
def mainline_only():
    """派发顺序是 中盘接回 > 前沿倍率 > 支线 > 主线；单测主线校验闸时把前三轨清空。"""
    def assign(*, native=False, frontier=False):
        return [] if (native or frontier) else decks.config()['order']
    with patch.object(decks, 'assigned_tasks', side_effect=assign), \
            patch.object(decks, 'drill_catalog', return_value=[]), \
            patch.object(decks, 'scenario_catalog', return_value=[]), \
            patch.object(decks, 'resume_catalog', return_value=[]), \
            patch.object(decks, 'upgrade_target', return_value=0):
        yield


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
    # 别的任务族（L2）里的同卡组胜利，同样要给 L1 记账：登记去重跨任务族生效。
    history = dict(entry(1, D1), task_id='llmplay_decks_l2_m100')
    with mainline_only():
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
    with mainline_only():
        doc = workflow.derive([], 'run')
        with pytest.raises(ValueError, match='留出'):
            workflow.check(doc, 'llmplay_decks_l4_m100', held[::-1], seed=decks.seed_at(4, 0))
        with pytest.raises(ValueError, match='黑名单'):
            workflow.check(doc, 'llmplay_decks_l1_m100', [0, 1, 2, 3, 4, 38], seed=decks.seed_at(1, 0))


def test_decks_three_losses_reason_stored_in_third_archive_note():
    task_id = 'llmplay_decks_l1_m100'
    failures = [entry(1, D1, n, won=False) for n in range(2)]
    with mainline_only():
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
    with mainline_only():
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
    # 支线已完成（此夹具没有分配支线/重放），本片覆盖完成后主线继续加2组。
    with patch.object(decks, 'assigned_tasks',
                      side_effect=lambda native=False, frontier=False:
                      [] if native or frontier else ['llmplay_decks_l1_m100']), \
            patch.object(decks, 'resume_catalog', return_value=[]), \
            patch.object(decks, 'upgrade_target', return_value=0), \
            patch.dict(decks.config(), coverage_per_card=0):
        doc = workflow.derive([entry(1, deck, n) for n, deck in enumerate(cards)], 'run')
    info = doc['levels']['llmplay_decks_l1_m100']
    assert info['baseline_complete'] and info['goal'] == 14
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
    saved.write_bytes(delivery.export_bundle()[0].read_bytes())
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
    with mainline_only():
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


@pytest.mark.parametrize('layer', ['resume', 'drill', 'scenario'])
def test_task265_loaded_claim_renews_past_ttl_and_releases(tmp_path, monkeypatch, layer):
    from llmplay import next_task
    from pvz_frame_store import write_frames
    clock = [1000.0]
    monkeypatch.setattr(workflow, 'time', SimpleNamespace(time=lambda: clock[0]))
    monkeypatch.setenv('LLMPLAY_CLAIM_HOLD', '10')
    task_id = 'task265_' + layer
    archive = tmp_path / layer
    archive.mkdir()
    base = Game('llmplay_decks_l1_m100', seed=7, deck=D1, reason='经济', _gate=False)
    block = dict(id=task_id, future_seed=101, prefix_operations=0)
    meta = dict(task=dict(id=base.task_id if layer == 'resume' else task_id,
                          level=1, seed=7, deck=D1),
                environment_task=base._environment_task(), workflow_layer=layer,
                outcome=dict(final_tick=0))
    if layer == 'resume':
        meta['resume_id'] = task_id
    else:
        meta[layer] = block
    (archive / 'meta.json').write_text(json.dumps(meta))
    (archive / 'ticks.jsonl').write_text('')
    write_frames(archive / 'frames.npz', [], cache=False)
    item = dict(id=task_id, task_id=task_id, archive=str(archive), level=1,
                terrain='day', seed=7, deck=D1, future_seeds=[101, 102])
    with patch.object(decks, 'resume_catalog', return_value=[item] if layer == 'resume' else []), \
            patch.object(decks, 'drill_catalog', return_value=[item] if layer == 'drill' else []), \
            patch.object(decks, 'scenario_catalog', return_value=[item] if layer == 'scenario' else []), \
            patch.object(decks, 'assigned_tasks', side_effect=lambda native=False, frontier=False:
                         [] if native or frontier else decks.config()['order']), \
            patch.object(decks, 'upgrade_target', return_value=0):
        # 留住第一个未来，验证读第二个种子时不会续错题面默认值或原局 seed。
        first = next_task(ttl=10)['item']
        claimed = next_task(ttl=10)['item'] if layer != 'resume' else first
        key = workflow._slot(claimed['task_id'], claimed['seed'])
        token = workflow.held()[key]['token']
        assert workflow.held()[key]['expires'] == 1010
        game = Game.load_archive(claimed['path'])
        game.env = SimpleNamespace(close=Mock())
        game.obs = dict(wave=1, result=0)

        def step(action):
            game.tick += int(action.get('ticks') or 0)
            return True

        try:
            with patch.object(game, '_step_raw', side_effect=step), \
                    patch.object(game, '_note_events'), patch.object(game, '_maybe_frame'), \
                    patch.object(game, '_persist_best'):
                if layer != 'resume':
                    assert not game._claimed
                    game.reseed(claimed['future_seed'])
                assert game._claimed == (claimed['task_id'], claimed['seed'])
                for now in (1006, 1012):
                    clock[0] = now
                    game.wait(60)  # 同一波推进，不能依赖换波才续期。
                assert workflow.held()[key]['token'] == token
                other = next_task(ttl=10)['item']
                assert (other['task_id'], other.get('seed')) != (claimed['task_id'], claimed['seed'])
                saved = tmp_path / 'saved.json'
                game.save(saved)
                game.close()
                assert key not in workflow.held()
                restored = Game.load(saved)
                assert restored._claimed == (claimed['task_id'], claimed['seed'])
                restored.close()
                child = Game.load_archive(claimed['path'], _internal=True)
                child._hold_claim()
                assert not child._claimed
                child.close()
        finally:
            game.close()
            base.close()


def test_decks_metadata_only_carry_survives_out_zip_resume(tmp_path):
    source = tmp_path / 'carry.json'
    prior = workflow.derive([entry(1, D1)], 'carry-run')
    source.write_text(json.dumps(prior))
    delivery.carry(source)
    assert delivery.progress()['archives'][0]['carried']
    saved = tmp_path / 'carry-out.zip'
    saved.write_bytes(delivery.export_bundle()[0].read_bytes())
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


def test_decks_native_is_dispatched_after_frontier_until_sixteen_distinct_seed_wins():
    no_frontier = dict(side_effect=lambda native=False, frontier=False:
                       [] if frontier else (decks.config()["native_order"] if native else decks.config()["order"]))
    with patch.object(decks, 'assigned_tasks', **no_frontier), \
            patch.object(decks, 'drill_catalog', return_value=[]), \
            patch.object(decks, 'scenario_catalog', return_value=[]), \
            patch.object(decks, 'resume_catalog', return_value=[]), \
            patch.object(decks, 'upgrade_target', return_value=0):
        doc = workflow.derive([], 'run')
        assert doc['active_frontier'] == [] and doc['frontier']
        assert workflow.next_task(doc, claim=False)['item']['layer'] == 'native'
        with pytest.raises(ValueError, match='支线优先'):
            workflow.check(doc, doc['active_tasks'][0], D1,
                           seed=doc['levels'][doc['active_tasks'][0]]['available_seeds'][0])
        first_native = decks.native_items(doc)[0]
        workflow.check(doc, first_native['task_id'], [], seed=first_native['seed'])
        with pytest.raises(ValueError, match='不能自选'):
            workflow.check(doc, first_native['task_id'], D1, seed=first_native['seed'])
        claimed = {workflow._slot(item['task_id'], item['seed']): {} for item in decks.native_items(doc)}
        with patch.object(workflow, '_read_claims', return_value=claimed):
            item = workflow.next_task(doc, claim=False)['item']
            assert item['layer'] == 'decks' and item['action'] in ('new_deck', 'retry_deck', 'resume')
            workflow.check(doc, item['task_id'], item['deck'] or D1, seed=item['seed'])
    repeated = [native_entry(10)] * 8
    assert workflow.derive(repeated, 'run')['native']['llmplay_decks_native_l10_m100']['wins'] == 1
    doc = workflow.derive([native_entry(10, n) for n in range(16)], 'run')
    assert doc['native']['llmplay_decks_native_l10_m100']['complete']
    assert all(item['level'] != 10 for item in decks.native_items(doc))
    assert '原生传送带支线' in workflow.report(doc)


def test_decks_frontier_rotates_registered_decks_on_fresh_seeds_first():
    night = 'llmplay_decks_l11_m100'
    frontier = 'llmplay_frontier_l11_m200'

    def game(deck, index, **kw):
        return dict(entry(11, deck, index, **kw), seed=decks.seed_at(11, index))
    entries = [game(D1, 0), game(D2, 1)] + [game(D1, n) for n in range(2, 5)]
    with patch.object(decks, 'drill_catalog', return_value=[]), \
            patch.object(decks, 'scenario_catalog', return_value=[]), \
            patch.object(decks, 'resume_catalog', return_value=[]):
        doc = workflow.derive(entries, 'run')
    assert frontier in doc['active_frontier']
    info = doc['frontier'][frontier]
    assert info['multiplier'] == 2.0 and info['wins'] == 0 and len(info['decks']) == 2
    # task263 给 L30/40/45/50 的尾关原生项单独优先级；这里只验证普通 native 排在 frontier 后。
    with patch.dict(decks.config(), native_targets={}):
        item = workflow.next_task(doc, claim=False)['item']
        assert item['layer'] == 'frontier' and item['action'] == 'frontier_deck' and item['task_id'] == frontier
        assert item['multiplier'] == 2.0 and item['deck'] == D1 and item['seed'] == decks.frontier_seed_at(frontier, 0)
        workflow.check(doc, frontier, D1, seed=item['seed'])
        with pytest.raises(ValueError, match='已登记的胜利卡组'):
            workflow.check(doc, frontier, [0, 1, 2, 3, 4, 7], seed=item['seed'])
        with pytest.raises(ValueError, match='新种子'):
            workflow.check(doc, frontier, D1, seed=decks.seed_at(11, 0))
        with pytest.raises(ValueError, match='前沿倍率优先'):
            workflow.check(doc, decks.native_items(doc)[0]['task_id'], [],
                           seed=decks.native_items(doc)[0]['seed'])
    won = [dict(entry(11, D1, 100 + n, task_id=frontier), level=11,
                seed=decks.frontier_seed_at(frontier, n), zombie_count_multiplier=2.0) for n in range(2)]
    doc = workflow.derive(entries + won, 'run')
    assert doc['frontier'][frontier]['wins'] == 2
    assert decks.frontier_items(doc)[0]['deck'] == D1   # 2 胜后轮换回第一套
    assert decks.frontier_items(doc)[0]['seed'] == decks.frontier_seed_at(frontier, 2)
    done = [dict(entry(11, D1, 200 + n, task_id=frontier), level=11,
                 seed=decks.frontier_seed_at(frontier, n), zombie_count_multiplier=2.0) for n in range(24)]
    doc = workflow.derive(entries + done, 'run')
    assert doc['frontier'][frontier]['complete']
    assert all(row['task_id'] != frontier for row in decks.frontier_items(doc))


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
    with patch('llmplay.game.load_tasks', side_effect=AssertionError('archive owns its task definition')):
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


def test_compact_export_removes_finished_run_without_full_bundle():
    root = delivery.output_dir()
    g = Game('llmplay_decks_l1_m100', seed=decks.seed_at(1, 0), deck=D1, reason='经济主力与防守', _gate=False)
    g.env = SimpleNamespace(episode={})
    g.obs = dict(result=1, tick=100, wave=10, wave_count=10, terminal=True)
    g.initial_obs = {}
    old = root / '.delivery' / 'old'
    old.mkdir()
    (old / 'out.zip').write_bytes(b'obsolete')
    (root / 'out.zip').write_bytes(b'obsolete')
    (root / '.out.zip.partial').write_bytes(b'obsolete')
    run = root / 'player_finished'
    with patch.dict(sys.modules, torch=None):
        stored = g.export_archive(run)
    assert not run.exists() and stored.parent == root / 'archives'
    assert (stored / 'frames.npz').is_file() and not (stored / 'frames.jsonl').exists()
    from pvz_frame_store import read_frames
    assert read_frames(stored / 'frames.npz') == g.frames
    restored = Game.load_archive(stored)
    assert restored.frames == g.frames
    restored.close()
    assert not (root / 'out.zip').exists() and not (root / '.out.zip.partial').exists()
    assert not delivery.bundle_dir().exists()
    assert not list((root / '.delivery').iterdir()) and not old.exists()
    assert all((root / name).is_file() and not (root / name).is_symlink()
               for name in ('progress.json', 'report.md', 'budget.json', 'decks.json'))
    assert json.loads((root / 'decks.json').read_text())['llmplay_decks_l1_m100']
    g.env = None
    g.close()


def _compact_archive(tmp_path, seed=0):
    archive = tmp_path / ('episode-%d' % seed)
    archive.mkdir()
    meta = dict(task=dict(id='llmplay_decks_l1_m100', level=1, seed=decks.seed_at(1, seed),
                          deck=D1, zombie_count_multiplier=1.0),
                outcome=dict(result=1, won=True, final_tick=100), workflow_layer='decks')
    (archive / 'meta.json').write_text(json.dumps(meta))
    (archive / 'ticks.jsonl').write_text('{}\n')
    from pvz_frame_store import write_frames
    write_frames(archive / 'frames.npz', [{'noise': os.urandom(10000).hex()}], cache=False)
    delivery.record_archive(archive)
    return archive


def test_compact_bundle_whitelist_parts_replacement_and_failed_export(tmp_path, monkeypatch):
    import io
    import zipfile
    _compact_archive(tmp_path)
    root = delivery.output_dir()
    (root / 'policy.py').write_text('def policy(g): g.wait(30)\n')
    (root / 'run.log').write_text('large log')
    (root / 'active.json').write_text('{}')
    (root / '__pycache__').mkdir()
    (root / '__pycache__' / 'cached.py').write_text('discard')
    monkeypatch.setattr(delivery, 'BUNDLE_PART_BYTES', 4096)
    parts = delivery.export_bundle()
    assert len(parts) > 1 and all(p.stat().st_size <= 4096 for p in parts)
    assert parts[0].parent == root.with_name(root.name + '_exports')
    with zipfile.ZipFile(io.BytesIO(b''.join(p.read_bytes() for p in parts))) as bundle:
        names = bundle.namelist()
        assert {'out/progress.json', 'out/decks.json', 'out/policy.py'} <= set(names)
        assert all(name in ('out/progress.json', 'out/decks.json', 'out/policy.py')
                   or name.startswith('out/archives/') for name in names)
        assert all(bundle.getinfo(name).compress_type == zipfile.ZIP_STORED for name in names if name.endswith('.npz'))
        assert json.loads(bundle.read('out/progress.json')) == delivery._read_progress(root)
    before = [p.read_bytes() for p in parts]
    with patch.object(delivery.zipfile.ZipFile, 'write', side_effect=OSError('disk full')):
        with pytest.raises(OSError, match='disk full'):
            delivery.export_bundle()
    assert [p.read_bytes() for p in parts] == before
    monkeypatch.setattr(delivery, 'BUNDLE_PART_BYTES', 99_000_000)
    assert len(delivery.export_bundle()) == 1
    assert len(list(parts[0].parent.iterdir())) == 1


def test_compact_bundle_snapshot_does_not_block_new_games_and_resumes_parts(tmp_path, monkeypatch):
    import io
    import zipfile
    task_id = 'llmplay_decks_l1_m100'
    budget.session().note_ticks(task_id, 123, 'search')
    _compact_archive(tmp_path)
    root = delivery.output_dir()
    (root / 'policy.py').write_text('def policy(g): g.wait(30)\n')
    old = delivery._read_progress(root)
    write = zipfile.ZipFile.write
    published = []

    def publish_during_compression(bundle, filename, *args, **kwargs):
        if not published:
            published.append(True)
            # Would deadlock if export_bundle held the per-game publication lock.
            with (root / '.delivery.lock').open() as lock:
                budget.fcntl.flock(lock, budget.fcntl.LOCK_EX | budget.fcntl.LOCK_NB)
            _compact_archive(tmp_path, 1)
        return write(bundle, filename, *args, **kwargs)

    monkeypatch.setattr(delivery, 'BUNDLE_PART_BYTES', 4096)
    with patch.object(zipfile.ZipFile, 'write', publish_during_compression):
        parts = delivery.export_bundle()
    with zipfile.ZipFile(io.BytesIO(b''.join(p.read_bytes() for p in parts))) as bundle:
        assert json.loads(bundle.read('out/progress.json')) == old
    assert len(delivery._read_progress(root)['archives']) == 2
    monkeypatch.setenv('LLMPLAY_OUT', str(tmp_path / 'resumed'))
    budget.reset_session(tmp_path / 'resumed-budget.json')
    with patch('llmplay.game.PvZEnv', side_effect=AssertionError('no replay')):
        document = delivery.resume(parts[0])
    assert document['archives'] == old['archives']
    assert delivery.progress()['levels']['llmplay_decks_l1_m100']['victory_decks'] == 1
    assert budget.session().level(task_id)['search_ticks'] == 123
    assert (delivery.output_dir() / 'policy.py').read_text() == 'def policy(g): g.wait(30)\n'


def test_decks_roster_uses_seed_wave_table_instead_of_first_allowed_wave(tmp_path):
    import importlib
    module = importlib.import_module('llmplay.roster')
    module._query.cache_clear()
    env = Mock()
    env.reset.return_value = (dict(wave_count=20, zombies=[], loadout_context={'zombie_roster': [0, 16]}), {})
    env.privileged_state.return_value = {
        'hidden': {'zombies_in_wave': [[0], [], [], [], [], [], [16], [16]],
                   'rand_seed': 123456, 'rand_state_hex': 'do-not-return'},
        'zombies': [{'id': 99, 'x': 777, 'row': 4}]}
    context = Mock()
    context.__enter__ = Mock(return_value=env)
    context.__exit__ = Mock(return_value=False)
    before = delivery.progress()['archives']
    with patch.object(module, 'PvZEnv', return_value=context), \
            patch.object(module, '_ensure_executable'), patch.object(module, 'app_data', return_value=tmp_path):
        result = module.roster('llmplay_decks_l33_m100', seed=186032123)
        same = module.roster('llmplay_decks_l33_m100', seed=186032123)
        assert env.reset.call_count == 1  # 仅当前进程缓存，不建磁盘中间文件。
        selected = {zombie['type']: zombie for zombie in result['zombies']}
        assert selected[16]['first_wave'] == 7  # 静态mFirstAllowedWave是10，不能当本局首次波次。
        assert selected[16]['name'] == '气球僵尸'
        assert result['seed'] == 186032123
        spec = env.reset.call_args.kwargs['task']
        assert spec.level == 33 and spec.fog_observation == 'human_v2'
        assert spec.seed == result['seed']
        assert 'rand_seed' not in repr(result) and 'do-not-return' not in repr(result)
        result['zombies'].clear()
        assert module.roster('llmplay_decks_l33_m100', seed=186032123) == same
        env.step.assert_not_called()
        env.wait.assert_not_called()
    assert delivery.progress()['archives'] == before
    module._query.cache_clear()


def test_decks_roster_marks_allowed_and_summoned_threats_without_inventing_waves():
    from llmplay.roster import _summary
    spec, _, _ = task_spec(load_tasks()['llmplay_decks_l19_m100'], deck=D1)
    obs = dict(wave_count=20, zombies=[], loadout_context={'zombie_roster': [0, 6, 8]})
    result = _summary('llmplay_decks_l19_m100', spec, obs, {'hidden': {'zombies_in_wave': [[0], [8]]}})
    zombies = {item['type']: item for item in result['zombies']}
    assert zombies[8]['first_wave'] == 2
    assert zombies[9]['first_wave'] is None and zombies[9]['sources'] == ['summon']
    assert zombies[6]['first_wave'] is None and zombies[6]['sources'] == ['allowed']


def test_decks_roster_native_boss_has_stage_based_threats():
    from llmplay.roster import _summary
    spec, _, _ = task_spec(load_tasks()['llmplay_decks_native_l50_m100'])
    result = _summary('llmplay_decks_native_l50_m100', spec, dict(wave_count=0),
                      {'hidden': {'zombies_in_wave': []}})
    zombies = {item['type']: item for item in result['zombies']}
    assert zombies[25]['sources'] == ['boss']
    assert zombies[23]['sources'] == ['boss_summon']  # 巨人，来自源码gBossZombieList
    assert zombies[20]['sources'] == ['boss_summon']  # 蹦极，来自BossBungeeSpawn
    assert zombies[24]['sources'] == ['summon']      # 巨人投出小鬼
    assert all(item['first_wave'] is None for item in zombies.values())


def test_decks_roster_public_export_and_default_seed():
    import importlib
    import llmplay
    module = importlib.import_module('llmplay.roster')
    task_id = 'llmplay_decks_l7_m100'
    with patch.object(module, '_query', return_value={'zombies': []}) as query:
        llmplay.roster(task_id)
        query.assert_called_once_with(task_id, load_tasks()[task_id]['seeds'][0])
    with pytest.raises(ValueError, match='没有任务'):
        llmplay.roster('not-a-task')


def test_decks_resume_items_follow_priority_native_and_are_one_shot():
    catalog = [dict(id='r86a_l17_s1', archive='r86a_l17_s1', level=17, terrain='night',
                    seed=1, deck=D1, multiplier=1.0, base_task_id='task101_l17_night_n10',
                    breach_tick=9000, cutoff_tick=4500, source_arm='r86a'),
               dict(id='r86a_l21_s2', archive='r86a_l21_s2', level=21, terrain='pool',
                    seed=2, deck=D2, multiplier=1.0, base_task_id='task101_l21_pool_p9k',
                    breach_tick=12000, cutoff_tick=7500, source_arm='r86a')]
    with patch.object(decks, 'drill_catalog', return_value=[]), \
            patch.object(decks, 'scenario_catalog', return_value=[]), \
            patch.object(decks, 'resume_catalog', return_value=catalog), \
            patch.object(workflow, '_read_claims', return_value={}) as claims:
        doc = workflow.derive([], 'run')
        assert doc['resume']['total'] == 2 and doc['resume']['remaining'] == 2
        item = workflow.next_task(doc, claim=False)['item']
        assert item['layer'] == 'native' and str(item['level']) in decks.config()['native_targets']
        claims.return_value = {workflow._slot(row['task_id'], row['seed']): {}
                               for row in decks.native_items(doc)
                               if str(row['level']) in decks.config()['native_targets']}
        item = workflow.next_task(doc, claim=False)['item']
        assert item['layer'] == 'resume' and item['task_id'] == 'r86a_l17_s1'
        assert item['action'] == 'resume_archive' and item['path'].endswith('resume/r86a_l17_s1')
        with pytest.raises(ValueError, match='load_archive'):
            workflow.check(doc, 'r86a_l17_s1', D1, seed=1)
        played = dict(entry(17, D1, 0), resume_id='r86a_l17_s1',
                      task_id='task101_l17_night_n10')
        doc = workflow.derive([played], 'run')
        assert doc['resume']['played'] == 1 and doc['resume']['remaining'] == 1
        claims.return_value.update({workflow._slot(row['task_id'], row['seed']): {}
                                    for row in decks.native_items(doc)
                                    if str(row['level']) in decks.config()['native_targets']})
        item = workflow.next_task(doc, claim=False)['item']
        assert item['task_id'] == 'r86a_l21_s2'


def test_decks_upgrade_requirement_applies_until_session_target():
    night = 'llmplay_decks_l11_m100'
    def assign(*, native=False, frontier=False):
        return [] if (native or frontier) else decks.config()['order']
    with patch.object(decks, 'assigned_tasks', side_effect=assign), \
            patch.object(decks, 'resume_catalog', return_value=[]):
        doc = workflow.derive([], 'run')
        assert doc['upgrades']['night']['missing'] == 24
        item = [row for row in decks.items(doc) if row['task_id'] == night][0]
        assert item['upgrade_required'] and item['upgrade_cards'] == [42]
        with pytest.raises(ValueError, match='升级植物'):
            workflow.check(doc, night, D1, seed=item['seed'])
        upgraded = [42, 10, 9, 8, 3, 4, 17]
        workflow.check(doc, night, upgraded, seed=item['seed'])


def test_decks_resume_fresh_priority_beats_terrain_order():
    catalog = [
        dict(id='old_pool', archive='old_pool', level=21, terrain='pool', seed=1, deck=D1,
             multiplier=1.0, priority=1, cutoff_tick=100),
        dict(id='fresh_roof', archive='fresh_roof', level=41, terrain='roof', seed=2, deck=D1,
             multiplier=1.0, priority=0, cutoff_tick=900),
        dict(id='fresh_night', archive='fresh_night', level=11, terrain='night', seed=3, deck=D1,
             multiplier=1.0, priority=0, cutoff_tick=500),
    ]
    with patch.object(decks, 'drill_catalog', return_value=[]), \
            patch.object(decks, 'scenario_catalog', return_value=[]), \
            patch.object(decks, 'resume_catalog', return_value=catalog):
        doc = workflow.derive([], 'run')
        item = workflow.next_task(doc, claim=False)['item']
    # 先新鲜批次，再按地形（夜先于屋顶）；旧档即使地形更靠前也排在所有新鲜项之后。
    assert item['task_id'] == 'fresh_night'
    assert [row['id'] for row in doc['resume']['items']][:3] == ['fresh_night', 'fresh_roof', 'old_pool']


def test_task263_focus_resume_and_native_tail_precede_remaining_scenarios():
    drill = dict(task_id='drill263', archive='drill/drill263', level=39, terrain='fog',
                 source_seed=7, deck=D1, targets=[dict(id=9)], future_seeds=[101], priority=0)
    scenario = dict(task_id='scenario263', archive='scenario/scenario263', level=39,
                    terrain='fog', source_seed=8, deck=D1, future_seeds=[201])
    resumes = [dict(id=name, archive=name, level=22, terrain='pool', seed=seed, deck=D1,
                    multiplier=0.5, source_arm=arm, priority=priority)
               for name, seed, arm, priority in [('focus263', 301, 'r160', -1),
                                                  ('old263', 302, 'r94', 0)]]
    with patch.object(decks, 'drill_catalog', return_value=[drill]), \
            patch.object(decks, 'scenario_catalog', return_value=[scenario]), \
            patch.object(decks, 'resume_catalog', return_value=resumes):
        doc = workflow.derive([native_entry(50, n) for n in range(17)], 'run')
        boss = doc['native']['llmplay_decks_native_l50_m100']
        assert boss['wins'] == 17 and boss['target'] == 32 and not boss['complete']
        assert doc['native']['llmplay_decks_native_l10_m100']['target'] == 16
        claimed = {}
        for expected in ['drill263', 'focus263', 'llmplay_decks_native_l50_m100',
                         'scenario263', 'old263']:
            with patch.object(workflow, '_read_claims', return_value=claimed):
                item = workflow.next_task(doc, claim=False)['item']
                assert item['task_id'] == expected
                ordered, _ = workflow._ordered(doc)
                assert workflow._first_free(ordered, claimed)['task_id'] == expected
                if item['layer'] == 'native':
                    workflow.check(doc, item['task_id'], [], seed=item['seed'])
            done = [item]
            if item['layer'] == 'native':
                done = [row for row in decks.ordered_items(doc)
                        if row['layer'] == 'native' and row['level'] in [30, 40, 45, 50]]
            claimed.update({workflow._slot(row['task_id'], row['seed']): {} for row in done})


@contextmanager
def _task248_catalog():
    item = dict(task_id='drill248', archive='drill/drill248', level=1, terrain='day',
                source_seed=7, deck=D1, targets=[dict(id=9)], future_seeds=[101, 102, 103])
    with patch.object(decks, 'drill_catalog', return_value=[item]), \
            patch.object(decks, 'scenario_catalog', return_value=[]), \
            patch.object(decks, 'resume_catalog', return_value=[]):
        yield item


def _task248_finish(base, item):
    archive = base / ('episode-%s' % item['future_seed'])
    archive.mkdir()
    meta = dict(task=dict(id=item['task_id'], level=1, seed=7, deck=D1),
                outcome=dict(result=0, won=False), workflow_layer='drill',
                drill=dict(id=item['task_id'], future_seed=item['future_seed'], success=False))
    (archive / 'meta.json').write_text(json.dumps(meta))
    (archive / 'ticks.jsonl').write_text('')
    (archive / 'frames.jsonl').write_text('{}\n')
    delivery.record_archive(archive)
    workflow.release(item['task_id'], item['seed'])


def _task248_process(out, session_path, base, connection, stale):
    os.environ['LLMPLAY_OUT'] = out
    budget.reset_session(session_path)
    with _task248_catalog():
        snapshot = delivery.progress()
        connection.send('ready')
        connection.recv()
        item = (workflow.next_task(snapshot) if stale else delivery.next_task())['item']
        connection.send(item['future_seed'])
        if stale:
            connection.recv()
        _task248_finish(Path(base), item)
        delivery.export_bundle()
        connection.send('exported')
        if not stale:
            connection.recv()
            item = delivery.next_task()['item']
            connection.send(item['future_seed'])
            _task248_finish(Path(base), item)
            delivery.export_bundle()
    connection.close()


def test_task248_concurrent_claim_finish_export_uses_fresh_progress(tmp_path):
    """进程 A 留旧快照，B 完成并导出后 A 再认领，两者继续并发完成和导出。"""
    import multiprocessing
    context = multiprocessing.get_context('fork')
    with _task248_catalog():
        delivery.ensure_run_id()
        connections, processes = [], []
        for stale in (True, False):
            parent, child = context.Pipe()
            process = context.Process(target=_task248_process,
                                      args=(str(delivery.output_dir()), str(tmp_path / 'budget.json'),
                                            str(tmp_path), child, stale))
            process.start()
            child.close()
            connections.append(parent)
            processes.append(process)
        a, b = connections

        def receive(connection):
            assert connection.poll(10), '子进程超时'
            return connection.recv()

        try:
            assert receive(a) == receive(b) == 'ready'
            b.send('claim')
            first = receive(b)
            assert receive(b) == 'exported'
            a.send('claim with old snapshot')
            second = receive(a)
            assert second != first
            b.send('continue')
            third = receive(b)
            a.send('finish')
            assert receive(a) == 'exported'
            assert len({first, second, third}) == 3
            for process in processes:
                process.join(10)
                assert process.exitcode == 0
            document = delivery.progress()
            assert document['drill']['played'] == 3 and document['drill']['remaining'] == 0
            assert len(document['archives']) == len(list((delivery.output_dir() / 'archives').iterdir())) == 3
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                process.join(5)
            for connection in connections:
                connection.close()


def test_task248_stale_publish_resume_and_legacy_symlink_preserve_history(tmp_path):
    with _task248_catalog() as item:
        delivery.ensure_run_id()
        root = delivery.output_dir()
        legacy = root / '.delivery' / 'old.json'
        legacy.write_text((root / 'progress.json').read_text())
        (root / 'progress.json').unlink()
        (root / 'progress.json').symlink_to(legacy)
        stale = delivery.progress()
        _task248_finish(tmp_path, dict(item, seed=101, future_seed=101))
        with delivery._lock(root):
            delivery._publish(root, stale['archives'], budget.session())
        saved = tmp_path / 'old-out'
        saved.mkdir()
        old = dict(delivery.progress())
        old['archives'] = [dict(old['archives'][0], valid=False, won=False, carried=True)]
        (saved / 'progress.json').write_text(json.dumps(old))
        delivery.resume(saved)
        from llmplay.build_package import merge_carry
        merged = merge_carry(delivery.progress(), old)
        assert len(merged['archives']) == 1 and merged['archives'][0]['valid']
        assert not (root / 'progress.json').is_symlink()
        assert delivery.progress()['drill']['played'] == 1
        assert delivery.progress()['archives'][0]['valid']


def test_task248_invalid_drill_root_skips_queue_and_survives_resume(tmp_path):
    from llmplay import InvalidDrillError
    with _task248_catalog() as item:
        g = Game('llmplay_decks_l1_m100', seed=7, deck=D1, reason='气球', _gate=False)
        g.task_id = item['task_id']
        g.drill_meta = dict(item, prefix_operations=0, root_tick=0, horizon=60)
        env = SimpleNamespace(close=Mock(), privileged_state=lambda: {'zombies': []})

        def reset():
            g.obs = dict(tick=0, result=0, defenses=[[0, 1]])

        try:
            with patch('llmplay.game.PvZEnv', return_value=env), \
                    patch('llmplay.resources.app_data', return_value=tmp_path), \
                    patch.object(g, '_reset_engine', reset):
                with pytest.raises(InvalidDrillError, match='已跳过'):
                    g._start()
            assert env.close.call_count == 1
            doc = delivery.progress()
            assert doc['drill']['played'] == 3 and doc['drill']['remaining'] == 0
            assert all(e['invalid_start'] and not e['valid'] for e in doc['archives'])
            assert delivery.next_task()['item']['layer'] != 'drill'
            parts = delivery.export_bundle()
            os.environ['LLMPLAY_OUT'] = str(tmp_path / 'restored-invalid')
            delivery.resume(parts[0])
            assert delivery.progress()['drill']['remaining'] == 0
        finally:
            g.close()


@pytest.mark.parametrize('present,abandoned', [(False, False), (True, False), (True, True)])
def test_task248_import_short_captures_targets_at_prefix_boundary(tmp_path, present, abandoned):
    import import_llm_run as importer
    from pvz_frame_store import write_frames
    g = Game('llmplay_decks_l1_m100', seed=7, deck=D1, reason='气球', _gate=False)
    archive = tmp_path / 'drill'
    archive.mkdir()
    block = dict(id='drill248', targets=[dict(id=9)], root_tick=60, prefix_operations=1,
                 mowers_start=1, horizon=60, success=present and not abandoned)
    meta = dict(task=dict(id='drill248', level=1, seed=7, deck=D1),
                environment_task=g._environment_task(), workflow_layer='drill', drill=block,
                outcome=dict(final_tick=120), operations=2, abandoned=abandoned)
    (archive / 'meta.json').write_text(json.dumps(meta))
    ops = [dict(action={'type': 'wait', 'ticks': 60}, tick=t, end_tick=t + 60, ok=True)
           for t in (0, 60)]
    (archive / 'ticks.jsonl').write_text(''.join(json.dumps(op) + '\n' for op in ops))
    write_frames(archive / 'frames.npz', [], cache=False)

    class Env:
        def __init__(self, **kwargs):
            self.tick = 0

        def begin_demo(self): pass
        def set_lean_observation(self, value): pass
        def close(self): pass
        def observe(self):
            return dict(tick=self.tick, wave=1, sun=50, result=0, defenses=[dict(row=0, state=1)])
        def reset(self, **kwargs):
            self.tick = 0
            return self.observe(), {}
        def privileged_state(self):
            return dict(zombies=[dict(id=9, row=0, type=16, x=300, on_board=True)]
                        if present and self.tick == 60 else [])
        def step(self, action):
            self.tick += action['ticks']
            return self.observe(), 0, False, False, dict(events={}, ok=True)

    try:
        with patch('llmplay.game.PvZEnv', Env), patch('llmplay.resources.app_data', return_value=tmp_path):
            result = importer.verify_short(archive, tmp_path, tmp_path / 'engine')
        assert result['result'] == (present and not abandoned)
        verdict = json.loads((archive / 'meta.json').read_text())['drill']
        assert verdict['invalid_start'] == (not present)
    finally:
        g.close()


def test_task248_prefix_wait_uses_recorded_elapsed_time_and_matches_root(tmp_path):
    from llmplay.make_drill_bank import write_prefix, check_drill_prefix
    from pvz_frame_store import write_frames
    source, target = tmp_path / 'source', tmp_path / 'target'
    source.mkdir()
    g = Game('llmplay_decks_l1_m100', seed=7, deck=D1, reason='气球', _gate=False)
    meta = dict(task=dict(seed=7), context=dict(level=1, multiplier=1.25))
    (source / 'meta.json').write_text(json.dumps(meta))
    # WAIT(60) 在 tick 37 遇事件返回，下一次等待在 tick 97 结束。
    write_frames(source / 'frames.npz', [dict(tick=tick, zombies=[], mowers=[],
                 packets=[[i, card] for i, card in enumerate(D1)],
                 action=None if tick == 0 else dict(type='wait', ticks=60))
                 for tick in (0, 37, 97)], cache=False)
    write_prefix(source, target, task_id='drill248', layer='drill',
                 block=dict(targets=[dict(id=9, type=16, row=0, x=300.0)]),
                 knowledge='气球', horizon=60, future_seeds=[101], source={})
    ops = [json.loads(line) for line in (target / 'ticks.jsonl').read_text().splitlines()]
    assert sum(op['action']['ticks'] for op in ops) == 97
    assert ops[0]['action']['ticks'] == 37
    restored = json.loads((target / 'meta.json').read_text())
    assert restored['task']['deck'] == D1 and restored['task']['level'] == 1
    assert restored['environment_task']['zombie_count_multiplier'] == 1.25
    fake = SimpleNamespace(_start=Mock(), tick=97, terminal=False, ops=ops, close=Mock(),
                           zombies=lambda: [dict(id=90, type=16, row=0, x=300.0)],
                           _mower_count=lambda: 0,
                           env=SimpleNamespace(privileged_state=lambda: dict(tick=97, wave=1, sun=50,
                                                                              zombies=[])))
    try:
        with patch.object(Game, 'load_archive', return_value=fake):
            checked = check_drill_prefix(target)
        assert checked['status'] == 'ok' and checked['repaired'] == 1
        assert checked['targets'][0]['id'] == 90
        fake.zombies = lambda: []
        with patch.object(Game, 'load_archive', return_value=fake):
            assert check_drill_prefix(target)['status'] == 'rejected'
    finally:
        g.close()


@contextmanager
def _task266_resume(tmp_path):
    from pvz_frame_store import write_frames
    archive = tmp_path / 'prefix'
    archive.mkdir()
    base = Game('llmplay_decks_l1_m100', seed=7, deck=D1, reason='经济', _gate=False)
    item = dict(id='resume266', archive=str(archive), seed=7, level=1, deck=D1,
                cutoff_tick=240, multiplier=1.0)
    meta = dict(task=dict(id=base.task_id, level=1, seed=7, deck=D1),
                environment_task=base._environment_task(), resume_id=item['id'],
                resume=dict(cutoff_tick=240), workflow_layer='resume', outcome=dict(final_tick=240))
    (archive / 'meta.json').write_text(json.dumps(meta))
    ops = [dict(action=dict(type='wait', ticks=240), tick=0, end_tick=240, ok=True)]
    (archive / 'ticks.jsonl').write_text(json.dumps(ops[0]) + '\n')
    write_frames(archive / 'frames.npz', [dict(tick=240, operation=1)], cache=False)

    class Env:
        episode = {}
        def __init__(self, **kwargs): self.tick = 0
        def begin_demo(self): pass
        def set_lean_observation(self, value): pass
        def close(self): pass
        def observe(self):
            return dict(tick=self.tick, wave=1, sun=50, result=0, terminal=False)
        def reset(self, **kwargs):
            self.tick = 0
            return self.observe(), {}
        def step(self, action):
            self.tick += action.get('ticks', 0)
            return self.observe(), 0, False, False, dict(events={}, ok=True)

    with patch.object(decks, 'resume_catalog', return_value=[item]), \
            patch.object(decks, 'drill_catalog', return_value=[]), \
            patch.object(decks, 'scenario_catalog', return_value=[]), \
            patch('llmplay.game.PvZEnv', Env), \
            patch('llmplay.resources.app_data', return_value=tmp_path):
        try:
            yield item
        finally:
            base.close()


def test_task266_rewind_child_claim_prefix_and_carry(tmp_path, monkeypatch):
    with _task266_resume(tmp_path) as original:
        item = workflow.next_task(delivery.progress(), claim=False)['resume']
        game = Game.load_archive(item['path'], strategy='补经济')
        child = None
        try:
            game.wait(60)
            for invalid in (-1, 240, 300, 1.5):
                with pytest.raises(ValueError, match='to_tick'):
                    game.rewind_resume(reason='第3波缺经济', to_tick=invalid)
            child = game.rewind_resume(reason='第3波缺经济', to_tick=95)
            assert child.resume_id == 'resume266__rewind1'
            assert child.tick == 95 and child.ops[-1]['end_tick'] == 95
            assert child.ops[-1]['action']['ticks'] == 95
            assert child._claimed == (child.resume_id, 7)
            doc = delivery.progress()
            queued = workflow.next_task(doc, claim=False)['resume']
            # 占位期间其他子代理跳过子项。
            assert queued is None
            assert decks.ordered_items(doc)[0]['task_id'] == child.resume_id
            assert doc['archives'][-1]['abandoned']
            assert doc['resume']['rewound'] == 1
            child._start()
            assert child.tick == 95 and len(child.frames) == 1 and child.frames[0]['tick'] == 95
            assert child.resume_source['prefix_operations'] == 1
            child.wait(60)
            saved = tmp_path / 'child.save'
            child.save(saved)
            child.close()
            child = Game.load(saved)
            child._start()
            assert child.tick == 155 and child.frames[0]['tick'] == 95
            child.close()
            # 结论层 carry 不带旧存档，也能从包内原前缀重建子项。
            prior = delivery.output_dir() / 'progress.json'
            monkeypatch.setenv('LLMPLAY_OUT', str(tmp_path / 'carried'))
            delivery.carry(prior)
            again = delivery.next_task()['item']
            assert again['task_id'] == 'resume266__rewind1'
            restored = Game.load_archive(again['path'], resume_item=again['resume_item'], strategy='补经济')
            try:
                assert restored.tick == 95
                restored.abandon(reason='第3波仍缺阳光')
            finally:
                restored.close()
            latest = delivery.output_dir() / 'progress.json'
            monkeypatch.setenv('LLMPLAY_OUT', str(tmp_path / 'carried-again'))
            delivery.carry(latest)
            assert delivery.next_task()['item']['task_id'] != again['task_id']
        finally:
            game.close()
            if child:
                child.close()


@pytest.mark.parametrize('opening', [False, True])
def test_task266_rewind_limit_structural_and_diagnostics(tmp_path, opening):
    with _task266_resume(tmp_path) as item:
        game = Game.load_archive(item['archive'], strategy='补经济')
        try:
            for number in range(1, 2 if opening else 4):
                game = game.rewind_resume(reason='第%d波阳光不足' % number,
                                          to_tick=0 if opening else None)
                assert game.resume_source['rewind_count'] == number
                assert game.tick == (0 if opening else 240 // (2 ** number))
            last = game
            assert game.rewind_resume(reason='开局仍救不回，检查卡组与倍率') is None
            doc = delivery.progress()
            assert doc['resume']['remaining'] == 0 and doc['resume']['structural'] == 1
            assert doc['resume']['structural_ids'] == [item['id']]
            record = doc['archives'][-1]['resume']
            assert record['structural'] and record['original_resume_id'] == item['id']
            assert len(record['rewind_chain']) == (1 if opening else 3)
            assert record['diagnosis'] == '开局仍救不回，检查卡组与倍率'
            last.close()
        finally:
            game.close()


def test_task269_imitater_deck_export_derive_carry_and_import(tmp_path, monkeypatch):
    """[48, target] must survive export -> derive -> carry -> importer as one pair."""
    import import_llm_run
    from pvz_frame_store import write_frames
    deck = [1, 33, 41, [48, 1], 39, 44, 37, 4, 17, 30]
    archive = tmp_path / 'imitater-win'
    archive.mkdir()
    meta = dict(schema='episode_archive_v2', schema_version=2,
                task=dict(id='llmplay_decks_l47_m100', level=47, seed=decks.seed_at(47, 0),
                          deck=deck, zombie_count_multiplier=1.0),
                outcome=dict(result=1, won=True, final_tick=100, final_wave=40),
                deck_reason='模仿者复制向日葵，屋顶花盆与投手覆盖地形',
                workflow_layer='decks')
    (archive / 'meta.json').write_text(json.dumps(meta))
    (archive / 'ticks.jsonl').write_text('')
    write_frames(archive / 'frames.npz', [], cache=False)

    delivery.record_archive(archive)
    document = delivery.progress()
    info = document['levels']['llmplay_decks_l47_m100']
    assert info['victory_decks'] == 1
    assert workflow.deck_key(deck) in info['winning_decks']
    assert document['upgrades']['roof']['wins'] >= 1
    assert document['coverage']['roof']['counts']['1'] == 1

    parts = delivery.export_bundle()
    root = delivery.output_dir()
    carry_source = tmp_path / 'carry.json'
    carry_source.write_text(json.dumps(delivery._read_progress(root)))

    monkeypatch.setenv('LLMPLAY_OUT', str(tmp_path / 'carried'))
    budget.reset_session(tmp_path / 'carried-budget.json')
    delivery.carry(carry_source)
    carried = delivery.progress()
    assert carried['levels']['llmplay_decks_l47_m100']['victory_decks'] == 1

    monkeypatch.setenv('LLMPLAY_OUT', str(tmp_path / 'resumed'))
    budget.reset_session(tmp_path / 'resumed-budget.json')
    with patch('llmplay.game.PvZEnv', side_effect=AssertionError('must not replay')):
        delivery.resume(parts[0])
    resumed = delivery.progress()
    assert resumed['levels']['llmplay_decks_l47_m100']['victory_decks'] == 1

    # The importer keeps the Imitater pair intact instead of flattening or hashing it.
    assert import_llm_run._deck_cards(meta['task']['deck']) == deck
    entry = import_llm_run.family_entry(meta['task']['id'], meta, None)
    assert entry['deck'] == deck
    assert archive in import_llm_run.discover(tmp_path)


def test_task269_claimed_native_item_is_accepted_after_target_fills(tmp_path):
    task_id = 'llmplay_decks_native_l50_m100'
    before = [native_entry(50, n) for n in range(31)]
    old = workflow.derive(before, 'run')
    assert old['native'][task_id]['wins'] == 31 and old['native'][task_id]['target'] == 32
    items = [row for row in decks.native_items(old) if row['task_id'] == task_id]
    assert len(items) == 1  # one win left, so at most one outstanding claim
    item = items[0]

    after = workflow.derive([native_entry(50, n) for n in range(32)], 'run')
    assert after['native'][task_id]['complete']
    only_native = [(decks.native_items(after), 'native'), (decks.items(after), 'main')]
    with patch.object(decks, 'queues', return_value=only_native):
        with pytest.raises(ValueError, match='达到 32 胜'):
            workflow.check(after, task_id, [], seed=item['seed'])

    claims = {workflow._slot(task_id, item['seed']): dict(task_id=task_id, seed=item['seed'])}
    with patch.object(decks, 'queues', return_value=only_native), \
            patch.object(workflow, '_read_claims', return_value=claims):
        assert workflow.check(after, task_id, [], seed=item['seed'])['workflow_layer'] == 'native'
        assert not [row for row in decks.native_items(after) if row['task_id'] == task_id]

    # An earlier queue becoming free must not revoke a slot that was already issued.
    with patch.object(decks, 'drill_catalog', return_value=[dict(
            task_id='earlier_drill', archive='drill/earlier', level=1, terrain='day',
            seed=1, deck=D1, targets=[], future_seeds=[101])]), \
            patch.object(decks, 'scenario_catalog', return_value=[]), \
            patch.object(decks, 'resume_catalog', return_value=[]), \
            patch.object(decks, 'assigned_tasks',
                         side_effect=lambda native=False, frontier=False:
                         [] if frontier else (decks.config()['native_order'] if native
                                              else decks.config()['order'])), \
            patch.object(decks, 'upgrade_target', return_value=0), \
            patch.object(workflow, '_read_claims', return_value=claims):
        blocked = workflow.derive(before, 'run')
        assert workflow.next_task(blocked, claim=False)['item']['layer'] == 'drill'
        assert workflow.check(blocked, task_id, [], seed=item['seed'])['workflow_layer'] == 'native'
        with pytest.raises(ValueError, match='必刷题优先'):
            workflow.check(blocked, 'llmplay_decks_l1_m100', D1,
                           seed=blocked['levels']['llmplay_decks_l1_m100']['available_seeds'][0])
