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
    """派发顺序是 弱地形重放 > 传送带支线 > 主线新卡组；单测主线校验闸时把前两轨清空。"""
    def assign(*, native=False, replay=False):
        return [] if (native or replay) else decks.config()['order']
    with patch.object(decks, 'assigned_tasks', side_effect=assign):
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
    with patch.object(decks, 'assigned_tasks', side_effect=lambda native=False, replay=False: [] if native or replay else ['llmplay_decks_l1_m100']), \
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


def test_decks_native_is_dispatched_after_replay_until_sixteen_distinct_seed_wins():
    no_replay = dict(side_effect=lambda native=False, replay=False:
                     [] if replay else (decks.config()["native_order"] if native else decks.config()["order"]))
    with patch.object(decks, 'assigned_tasks', **no_replay):
        doc = workflow.derive([], 'run')
        assert doc['active_replay'] == []
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


def test_decks_replay_rotates_registered_decks_on_fresh_seeds_first():
    night = 'llmplay_decks_l11_m100'
    def game(deck, index, **kw):
        return dict(entry(11, deck, index, **kw), seed=decks.seed_at(11, index))
    entries = [game(D1, 0), game(D2, 1)]
    entries += [game(D1, n) for n in range(2, 26)]
    doc = workflow.derive(entries, 'run')
    assert night in doc['active_replay']
    info = doc['replay'][night]
    assert info['wins'] == 26 and info['goal'] == 48 and len(info['decks']) == 2
    item = workflow.next_task(doc, claim=False)['item']
    assert item['layer'] == 'replay' and item['action'] == 'replay_deck' and item['task_id'] == night
    assert item['deck'] == D2 and item['seed'] == decks.seed_at(11, 34)  # D2 只用过一次，先轮换到它
    workflow.check(doc, night, D2, seed=item['seed'])           # 已登记卡组可复用，不触发差 2 张闸
    workflow.check(doc, night, D1, seed=item['seed'])
    with pytest.raises(ValueError, match='已登记的胜利卡组'):
        workflow.check(doc, night, [0, 1, 2, 3, 4, 7], seed=item['seed'])
    with pytest.raises(ValueError, match='新种子'):
        workflow.check(doc, night, D1, seed=decks.seed_at(11, 0))
    with pytest.raises(ValueError, match='重放优先'):
        workflow.check(doc, decks.native_items(doc)[0]['task_id'], [],
                       seed=decks.native_items(doc)[0]['seed'])
    grown = entries + [game(D1, n) for n in range(26, 38)]
    doc = workflow.derive(grown, 'run')
    assert decks.replay_items(doc)[0]['deck'] == D2
    assert decks.replay_items(doc)[0]['seed'] == decks.seed_at(11, 46)
    doc = workflow.derive([game(deck, n)
                           for n, deck in enumerate([D1, D2] * 25)], 'run')
    assert doc['replay'][night]['goal'] == 72                   # 50 胜后加量到 24×3


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
    generations = [p for p in (root / '.delivery').iterdir() if not p.is_symlink()]
    assert len(generations) == 1 and not old.exists()
    assert set(p.name for p in generations[0].iterdir()) == {'progress.json', 'report.md', 'budget.json', 'decks.json'}
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
