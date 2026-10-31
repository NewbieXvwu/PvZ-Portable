"""task193: small metadata/planner/import cases and opt-in one-group rehearsal."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'python')]
import curriculum_queue as queue
import pvz_live_curriculum as live


def archive(root, identity, level=1, deck=(0, 1, 2, 3, 4, 7), multiplier=1.):
    path = root / 'sha256' / identity
    path.mkdir(parents=True)
    meta = dict(task=dict(id='gpt_' + identity, level=level, seed=123, deck=list(deck),
                          zombie_count_multiplier=multiplier),
                environment_task={}, verified=True, frame_count=101, outcome=dict(result=1))
    (path / 'meta.json').write_text(json.dumps(meta))
    # Metadata-only publication must never decode this deliberately invalid trajectory.
    (path / 'frames.npz').write_bytes(b'not read by live pool')
    return dict(archive_sha256=identity, archive=str(path), verify='通过', result=1, frames=101)


def publish(root, rows):
    (root / 'merged_progress.json').write_text(json.dumps(dict(archives=rows)))


def state():
    return dict(round=0, cells={}, offsets={}, log=[])


def test_deploy22_next_round_live_pool_dedup_mixing_and_global_holdout(tmp_path, monkeypatch):
    held = (0, 1, 2, 3, 4, 5)
    monkeypatch.setattr(live, 'heldout_signatures', lambda *a: {held})
    first = archive(tmp_path, 'a')
    duplicate = archive(tmp_path, 'b', deck=(7, 4, 3, 2, 1, 0))
    excluded = archive(tmp_path, 'c', level=2, deck=held)
    hard = archive(tmp_path, 'd', level=3, multiplier=2.)
    publish(tmp_path, [first, duplicate, excluded, hard])
    tasks = {'old': dict(task_id='old', level=1, deck=[0, 1, 2, 3, 5, 7], random_deck_cell=True)}
    saved = state()
    catalog = live.refresh(saved, tasks, tmp_path, tmp_path, [], {})
    assert len(catalog['rd193_l1_train']['options']) == 2  # existing pool + one deduplicated GPT deck
    assert 'rd193_l2_train' not in catalog and 'rd193_l3_train' not in catalog
    assert not any(item['kind'] == 'demo' for item in catalog.values())
    new = archive(tmp_path, 'e', level=4)
    publish(tmp_path, [first, duplicate, excluded, hard, new])
    refreshed = live.refresh(saved, tasks, tmp_path, tmp_path, [], {})
    assert 'rd193_l4_train' in refreshed


def test_deploy22_demo_group_rotation_preserves_registered_cell_and_relative_progress(tmp_path, monkeypatch):
    monkeypatch.setattr(live, 'heldout_signatures', lambda *a: set())
    a, b = archive(tmp_path, 'a', level=3), archive(tmp_path, 'b', level=3)
    unknown = archive(tmp_path, 'u', level=41, deck=(1, 33, 39, 44, 17, 3))
    publish(tmp_path, [a, b, unknown])
    saved = state()
    old = dict(archive=str(tmp_path / 'copied/sha256/a'), family='demo_start', level=3,
        task_id='old', ladder=[90, 73, 40], mult_index=1, last_round=5,
        windows={'73': dict(recent=[1, 0], wins=1, games=2)}, seeds=[11, 12])
    saved['cells']['old'] = copy.deepcopy(old)
    catalog = live.refresh(saved, {}, tmp_path, tmp_path, [], {3: 0.})
    base = live.group_id(3, (0, 1, 2, 3, 4, 7))
    assert len(catalog[base]['options']) == 2
    assert live.group_id(41, (1, 33, 39, 44, 17, 3)) not in catalog  # no hand-written roof trigger
    task, chosen = live.materialize(base, saved['cells'][base], catalog, 1, saved)
    assert chosen['adaptation_base'] == 'old' and chosen['demo_decision'] == 73
    saved['cells'][base].update(archive_cursor=1, mult_index=1)
    _, chosen = live.materialize(base, saved['cells'][base], catalog, 2, saved)
    assert chosen['demo_archive'] == b['archive'] and chosen['demo_decision'] == 80
    live.refresh(saved, {}, tmp_path, tmp_path, [], {3: 0.})
    assert saved['cells'][base]['mult_index'] == 1 and saved['cells'][base]['archive_cursor'] == 1
    assert saved['cells']['old']['windows'] == old['windows']
    assert saved['cells']['old']['ladder'] == old['ladder']
    assert saved['cells']['old']['live193_grouped']


def test_deploy22_cold_evidence_excludes_student_mix_and_demo_outcomes():
    tasks = {'day': dict(level=4), 'roof': dict(level=41)}
    groups = [dict(task_id='day', multiplier=1., beta=beta, wins=0, games=8) for beta in (0., .5, 1.)]
    groups += [dict(task_id='roof', multiplier=1., beta=.5, wins=0, games=100),
               dict(task_id='roof', multiplier=1., beta=1., wins=8, games=8, demo_archive='/demo')]
    saved = dict(log=[dict(groups=groups)])
    assert live.cold_teacher_rates(saved, tasks) == {4: 0.}


def test_deploy22_live_refresh_prefers_actual_teacher_over_bootstrap(tmp_path, monkeypatch):
    rates = tmp_path/'teacher_rates.json'
    rates.write_text(json.dumps(dict(levels={'4': .8, '34': 0.})))
    args = SimpleNamespace(gpt_import_dir=tmp_path, gpt_archive_root=tmp_path,
        teacher_rates=rates, gpt_cold_rates=None, log=tmp_path/'log')
    saved = dict(cells={}, log=[dict(groups=[dict(task_id='day', multiplier=1., beta=1., wins=0, games=8)])])
    captured = {}
    def refresh(*args):
        captured.update(args[-1])
        return {}
    monkeypatch.setattr(live, 'refresh', refresh)
    queue.refresh_live_curriculum(args, saved, {'day': dict(level=4)})
    assert captured == {4: 0., 34: 0.}


def test_deploy22_planner_quotas_and_seed_offsets_without_manifest(tmp_path):
    saved = state(); catalog = {}
    for level in range(1, 9):
        base = f'rd193_l{level}_train'
        saved['cells'][base] = dict(level=level, family='gpt_live', last_round=0, live193=True,
            random_deck_cell=True, windows={}, mult_index=2, ladder=list(live.LADDER), seeds=[])
        task = dict(task_id=base, level=level, deck=[0, 1, 2, 3, 4, 7], zombie_count_multiplier=1.)
        catalog[base] = dict(kind='random', options=[task])
        base = 'demo' + str(level)
        saved['cells'][base] = dict(level=level, family='gpt_live', last_round=0, live193=True,
            archive='/archive', windows={}, mult_index=0, ladder=list(live.DEMO_LADDER), seeds=[])
        catalog[base] = dict(kind='demo', options=[dict(identity=str(level), archive='/archive',
            task=dict(task, task_id='gpt_demo_' + str(level), seeds=[1]), frames=101, legacy_base=None)])
    ledger = tmp_path / 'seeds.json'
    plan, _ = queue.plan_round(saved, {}, {}, [1., 1.25], {'placeholder': ROOT / 'configs/gate_v5_panel.json'},
        768, 48, start_seed_ledger=ledger, random_deck_share=.25, demo_start_share=.25, live_catalog=catalog)
    assert sum(p['games'] for p in plan if 'demo_archive' not in p) == 192
    assert sum(p['games'] for p in plan if 'demo_archive' in p) == 192
    seed_sets = [set(p.get('future_seeds') or p['task_definition']['seeds']) for p in plan]
    assert len(set.union(*seed_sets)) == sum(map(len, seed_sets))
    assert all('task_definition' in p and p['offset_key'] == p['base'] for p in plan)
    # Whole-game live cells carry only this group's window; the cell table keeps accumulating.
    whole = [p for p in plan if 'demo_archive' not in p]
    assert all(p['seed_offset'] == 0 and len(p['task_definition']['seeds']) == p['games']
               for p in whole)
    first = whole[0]; base = first['base']
    saved['offsets'][base] = first['games']
    saved['cells'][base]['mult_index'] = 3
    next_plan, _ = queue.plan_round(saved, {}, {}, [1., 1.25], {'placeholder': ROOT / 'configs/gate_v5_panel.json'},
        768, 48, start_seed_ledger=ledger, random_deck_share=.25, demo_start_share=.25, live_catalog=catalog)
    again = next(p for p in next_plan if p['base'] == base)
    assert again['offset'] == first['games'] and again['seed_offset'] == 0
    assert set(again['task_definition']['seeds']).isdisjoint(first['task_definition']['seeds'])
    review = next(p for p in next_plan if p.get('review_of') == again['label'])
    assert review['games'] == 12 and review['multiplier'] == 1.
    assert review['task_definition']['deck'] == again['task_definition']['deck']
    assert review['offset_key'] == again['offset_key'] == base
    assert set(review['task_definition']['seeds']).isdisjoint(again['task_definition']['seeds'])


def test_deploy24_live193_seed_renewal_failure_skips_only_that_group(tmp_path, monkeypatch):
    saved = state(); catalog = {}
    for level in (1, 2):
        base = f'rd193_l{level}_train'
        saved['cells'][base] = dict(level=level, family='gpt_live', last_round=0, live193=True,
            random_deck_cell=True, windows={}, mult_index=2, ladder=list(live.LADDER), seeds=[])
        catalog[base] = dict(kind='random', options=[dict(
            task_id=base, level=level, deck=[0, 1, 2, 3, 4, 7], zombie_count_multiplier=1.)])
    exhausted = 'rd193_l1_train'

    def renew(cell, ledger, size=256):
        if cell is saved['cells'][exhausted]:
            raise queue.SeedPoolExhausted('test pool exhausted')
        cell['seeds'] = cell['seeds'] + list(range(200000000, 200000000 + size))
        return [200000000, 200000000 + size]

    monkeypatch.setattr(queue, 'renew_start_seeds', renew)
    plan, changes = queue.plan_round(saved, {}, {}, [1., 1.25],
        {'placeholder': ROOT / 'configs/gate_v5_panel.json'}, 768, 48,
        start_seed_ledger=tmp_path / 'seeds.json',
        random_deck_share=.25, demo_start_share=.25, live_catalog=catalog)
    assert exhausted not in {item['base'] for item in plan}
    assert any(exhausted in change and 'failed' in change for change in changes)
    assert plan and all(len(item['task_definition']['seeds']) == item['games'] for item in plan)


@pytest.mark.skipif(__import__('os').environ.get('PVZ_DEPLOY22_E2E') != '1', reason='one-time desktop rehearsal')
def test_deploy22_e2e_one_real_group(monkeypatch):
    import subprocess
    import tempfile
    from pvz_frame_store import frame_paths
    home = Path.home()
    library = home/'PvZ-search/.claude/worktrees/llmimport'
    production = json.loads((home/'pvz-logs/curriculum_state.json').read_text())
    model = Path(production['student'])
    original_run = subprocess.run
    with tempfile.TemporaryDirectory(prefix='task193-once-') as temporary:
        folder = Path(temporary)
        saved = state()
        catalog = live.refresh(saved, {}, library/'artifacts/task105/import', library,
            [ROOT/'configs/deploy17_deck_holdouts.json'], {})
        # A single real GPT cold-start deck; the round still goes through the production planner.
        base = min(catalog, key=lambda key: saved['cells'][key]['level'])
        level = saved['cells'][base]['level']
        catalog = {base: catalog[base]}
        saved['cells'] = {base: saved['cells'][base]}
        plan, changes = queue.plan_round(saved, {}, {}, [1., 1.25],
            {'inline': ROOT/'configs/gate_v5_panel.json'}, 2, 2,
            start_seed_ledger=folder/'seeds.json', random_deck_share=1.,
            demo_start_share=0., live_catalog=catalog)
        args = SimpleNamespace(state=folder/'state.json', log=folder/'curriculum.log',
            stop_flag=folder/'stop', python=Path(sys.executable), workers=1,
            frames=folder/'frames', out=folder/'out', rollout_policy=model,
            static_value=home/'PvZ-search/artifacts/hard/value_branch1/model.pt',
            value=home/'PvZ-search/artifacts/value_net/v5/model.pt',
            executable=home/'PvZ-search/artifacts/production/deploy20/pvz-portable',
            resource_dir=home/'.cache/pvz-research-resources', native_games=False,
            obs_v3=True, beta_threshold=.1, starved_beta=.5, probe_interval=3)
        def short_collection(command, **kwargs):
            # Same collector/engine/models/teacher labels; truncate games and rollout budget only.
            entry = "import sys;sys.path.insert(0,sys.argv.pop(1));import search_play;s=search_play;s.MAX_ACTIONS=8;s.main()"
            command = [command[0], '-c', entry, str(ROOT/'scripts'), *command[2:]]
            command[command.index('--verify-samples')+1] = '1'
            command += ['--samples', '1', '--horizon', '1', '--no-halving']
            result = original_run(command, **kwargs, capture_output=True, text=True, timeout=180)
            assert result.returncode == 0, result.stderr[-6000:]
            return result
        monkeypatch.setattr(queue.subprocess, 'run', short_collection)
        assert queue.run_round(args, saved, plan, changes)
        groups = saved['log'][-1]['groups']
        assert len(groups) == 1 and groups[0]['status'] == 'done' and groups[0]['valid_games'] == 2
        assert saved['offsets'][base] == 2
        paths = frame_paths(args.frames)
        assert len(paths) == 2
        for path in paths:
            meta = json.loads((path.parent/'meta.json').read_text())
            assert meta['task']['level'] == level and meta['task']['deck']
