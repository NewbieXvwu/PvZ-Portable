"""Pure-function tests for the task78 export/training/evaluation plumbing."""
import sys
from pathlib import Path
import unittest
import json
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'python'), str(ROOT/'scripts')]
import task78_ab as ab
import task78_export as export


class StopSetTest(unittest.TestCase):
    def test_every_fog_group_retains_the_three_preregistered_horizons(self):
        for group in ('fog_l31_human','fog_demo_l31','fog_demo_l32',
                      'fog_demo_l33_cactus','fog_demo_l33_cattail'):
            self.assertEqual(export.collection_horizons(dict(group=group)), (2500,4000,6000))
        self.assertEqual(export.collection_horizons(dict(group='roof_l41')), (2500,))
        self.assertEqual(export.collection_horizons(dict(group='pool_p9k'), old=True), (2500,))

    def test_region_floor_adds_hash_smallest_archive(self):
        archives = [dict(group='roof_l41', world=dict(seed=s, future_seed=None)) for s in range(10)]
        stops, added = export.stop_set(archives, old=False)
        roof = [i for i, a in enumerate(archives)]
        self.assertGreaterEqual(sum(stops[i] for i in roof), 2)
        self.assertTrue(all(isinstance(v, bool) for v in stops.values()))
        # Every archive the floor added must be flagged as a stopping archive.
        for index in added.get('roof', []):
            self.assertTrue(stops[index])

    def test_old_manifest_uses_the_v5_seed_split(self):
        archives = [dict(group='day_d7', world=dict(seed=s, future_seed=None)) for s in (1, 2, 3)]
        stops, added = export.stop_set(archives, old=True)
        self.assertEqual(added, {})
        self.assertEqual(stops, {i: export.v5_stop(a['world']['seed']) for i, a in enumerate(archives)})


class StatisticsTest(unittest.TestCase):
    def test_clopper_bounds_hit_the_extremes(self):
        self.assertEqual(ab.clopper_lower(0, 32), 0.0)
        self.assertEqual(ab.clopper_upper(32, 32), 1.0)
        self.assertGreater(ab.clopper_lower(30, 32), ab.clopper_lower(20, 32))
        self.assertLess(ab.clopper_upper(2, 32), ab.clopper_upper(12, 32))

    def test_mcnemar_is_symmetric_and_one_sided_evidence_is_small(self):
        self.assertAlmostEqual(ab.mcnemar(0, 0), 1.0)
        self.assertAlmostEqual(ab.mcnemar(3, 4), ab.mcnemar(4, 3))
        self.assertLess(ab.mcnemar(0, 8), 0.05)
        self.assertGreater(ab.mcnemar(5, 5), 0.5)


class CellsTest(unittest.TestCase):
    def test_cells_use_disjoint_reserved_seeds(self):
        cells = ab.cells()
        seen = set()
        for cell in cells:
            seeds=cell.get('future_seeds') or cell['seeds']
            self.assertEqual(len(seeds), cell['pairs'])
            self.assertFalse(seen & set(seeds))
            seen |= set(seeds)
        self.assertLessEqual(max(seen), 78101999)
        self.assertGreaterEqual(min(seen), ab.SEED_START)
        self.assertEqual(sum(c['pairs'] for c in cells), 512)

    def test_human_fog_and_all_three_takeover_points_are_in_the_formal_panel(self):
        cells=ab.cells()
        cold=next(c for c in cells if c['cell_id']=='fog_l31_snow_m100')
        self.assertEqual(cold['fog_observation'],'human')
        demos=[c for c in cells if 'future_seeds' in c]
        self.assertEqual([c['demo_decision'] for c in demos],[839,762,686])
        self.assertEqual(len({c['cell_id'] for c in cells}),15)


@pytest.fixture
def label_fixture(tmp_path, monkeypatch):
    from value_teacher_feasibility import group_summary
    labels = tmp_path/'labels'
    job = labels/'archive_00/h2500'
    root = job/'root_3'
    root.mkdir(parents=True)
    monkeypatch.setattr(export, 'paths', lambda _: (tmp_path/'replay.json', labels))
    monkeypatch.setattr(export, 'sha', lambda _: 'frozen')
    row = dict(archive='/archive', frames_path='/archive/frames.npz',
               frames_sha256='frozen', meta_sha256='frozen', roots=[dict(frame=3,tick=100,wave=1)])
    identity = dict(source_manifest_sha256='frozen', source=row, horizon=2500,
                    executable_sha256='engine', teacher_source_sha256='frozen',
                    label_channel_sha256='frozen', models={k:dict(path='/'+k,sha256='frozen')
                    for k in ('value','rollout_policy','static_value')})
    action = dict(type='wait',ticks=60)
    income = dict(start_tick=0,production=[])
    root_record = dict(candidates=[action],root_frame=dict(tick=100,wave=1),
                       root_state_sha256='root', decision_seed=123,common_future=123000,
                       sun_income_history=income)
    record = dict(candidate=0,action=action,common_future=123000,continuation_seed_base=5123,
                  history_checks=dict(frame=True,engine_state=True,income=True,income_start=True),
                  leaf_frame=dict(tick=110,terminal=False),sun_income_history=income,
                  outcome=dict(terminal=True,won=False,result=2,decisions=1,tick=130),
                  continuation_cpu_seconds=1,replay_cpu_seconds=.1)
    documents = {job/'complete.json': dict(identity=identity,roots=[dict(frame=3)]),
                 root/'root.json':root_record, root/'candidate_0.json':record,
                 root/'summary.json':group_summary([record])}
    for path, value in documents.items():
        path.write_text(json.dumps(value))
    (root/'candidate_0_teacher.jsonl').write_text(json.dumps(dict(decision=0,tick=110))+'\n')
    replay = dict(executable_sha256='engine',results=[dict(archive='/archive',
                  roots=[dict(frame=3,engine_sha256='root')])])
    return row,replay,root


def test_export_audits_complete_raw_label_chain(label_fixture):
    row,replay,_ = label_fixture
    rows,hashes = export.audit_labels(0,2500,row,Path('/sources.json'),replay)
    assert len(rows) == 1 and len(hashes) == 4


@pytest.mark.parametrize('corruption', ['missing_candidate','bad_history','missing_trace','changed_root'])
def test_export_rejects_corruption_even_with_complete_marker(label_fixture, corruption):
    row,replay,root = label_fixture
    path = root/'candidate_0.json'
    if corruption == 'missing_candidate':
        path.unlink()
    elif corruption == 'bad_history':
        record=json.loads(path.read_text());record['history_checks']['engine_state']=False
        path.write_text(json.dumps(record))
    elif corruption == 'missing_trace':
        (root/'candidate_0_teacher.jsonl').write_text('')
    else:
        record=json.loads((root/'root.json').read_text());record['root_state_sha256']='changed'
        (root/'root.json').write_text(json.dumps(record))
    with pytest.raises(ValueError):
        export.audit_labels(0,2500,row,Path('/sources.json'),replay)


def test_truth_bootstrap_reports_no_eligible_pairs_as_insufficient():
    from task78_truth import cluster_bootstrap
    result=cluster_bootstrap([dict(truth_pair=None)], ['v5','v6'])
    assert result['worlds']==0 and result['lower'] is None and result['upper'] is None
    assert result['status']=='insufficient_evidence'


def test_truth_bootstrap_keeps_same_future_decisions_in_one_cluster():
    from task78_truth import cluster_bootstrap
    rows=[dict(task='fog',seed=101,decision=i,
               truth_pair=dict(better='plant',worse='wait'),
               values=dict(v5=dict(plant=.1,wait=.2),v6=dict(plant=.2,wait=.1)))
          for i in range(4)]
    result=cluster_bootstrap(rows,['v5','v6'],draws=100)
    assert result['worlds']==1 and result['lower']==1 and result['upper']==1


def test_truth_report_rejects_a_failed_collection(tmp_path):
    from task78_truth import report
    path=tmp_path/'failed.json'
    path.write_text(json.dumps(dict(complete=True,passed=False,results=[])))
    with pytest.raises(ValueError,match='finish successfully'):
        report(path,{'v5':'/old','v6':'/new'})


def test_demonstration_pairing_uses_future_not_repeated_base_seed(tmp_path):
    (tmp_path/'games.jsonl').write_text('\n'.join(json.dumps(dict(seed=20261033,future_seed=s,
        terminal=True,won=bool(s%2))) for s in [78100000,78100001]))
    wins,missing=ab.load_arm(tmp_path,{78100000,78100001},use_future=True)
    assert missing==[] and wins=={78100000:False,78100001:True}


@pytest.mark.parametrize('overrides',[dict(terminal=False),dict(won='false')])
def test_pairing_rejects_invalid_game_results(tmp_path,overrides):
    (tmp_path/'games.jsonl').write_text(json.dumps(dict(dict(seed=1,terminal=True,won=False),**overrides)))
    with pytest.raises(ValueError):
        ab.load_arm(tmp_path,{1})


def test_victory_gates_do_not_accept_a_missing_demo_cell():
    results={c['cell_id']:dict(complete=True,net=1) for c in ab.cells()}
    del results['fog_l33_cactus_d686']
    pooled={t:dict(conservative_lower=0,net=10,mcnemar_p=.001) for t in ('day','night','pool','fog','roof')}
    gates=ab.game_gates(results,pooled)
    assert gates['fog']==dict(status='incomplete',passed=None)
    assert gates['day']['passed'] is True


def test_victory_gates_enforce_legacy_cell_floor_despite_pooled_noninferiority():
    results={c['cell_id']:dict(complete=True,net=1) for c in ab.cells()}
    results['pool_l21_p9k']['net']=-3
    pooled={t:dict(conservative_lower=0,net=10,mcnemar_p=.001) for t in ('day','night','pool','fog','roof')}
    assert ab.game_gates(results,pooled)['pool']['passed'] is False


def test_demo_run_command_preserves_base_seed_and_uses_independent_futures(tmp_path):
    cell=next(c for c in ab.cells() if c['cell_id']=='fog_l33_cactus_d762')
    plan=dict(models={k:dict(path='/models/'+k) for k in ('v5','v6','student','static')},
              executable=dict(path='/engine'),demos={cell['task_id']:dict(path='/demo')})
    args,out=ab.command(cell,'v6',plan,tmp_path)
    assert args[args.index('--seed-list')+1]=='20261033'
    assert args[args.index('--demo-decision')+1]=='762'
    assert args[args.index('--future-seeds')+1]==','.join(map(str,range(78100032,78100064)))
    assert args[args.index('--workers')+1]=='2'
    assert out==tmp_path/cell['cell_id']/'v4'/cell['task_id']


def test_the_two_paired_commands_change_only_the_leaf_and_output(tmp_path):
    cell=ab.cells()[0]
    plan=dict(models={k:dict(path='/models/'+k) for k in ('v5','v6','student','static')},
              executable=dict(path='/engine'))
    old,_=ab.command(cell,'v5',plan,tmp_path)
    new,_=ab.command(cell,'v6',plan,tmp_path)
    changed=[i for i,(a,b) in enumerate(zip(old,new)) if a!=b]
    assert changed==[old.index('--out')+1,old.index('--value')+1]


def test_frozen_panel_rejects_retrospective_seed_edit(tmp_path,monkeypatch):
    root=tmp_path/'run';root.mkdir()
    path=root/'tasks.json';path.write_text(json.dumps(dict(cells=[dict(seed=1)])))
    docs=tmp_path/'MEMORY/docs';docs.mkdir(parents=True)
    (docs/'task78-ab-plan.json').write_text(json.dumps(dict(sha256=ab.digest(path))))
    monkeypatch.setattr(ab,'ROOT',tmp_path)
    assert ab.frozen_panel(root)==[dict(seed=1)]
    path.write_text(json.dumps(dict(cells=[dict(seed=2)])))
    with pytest.raises(ValueError,match='preregistration'):
        ab.frozen_panel(root)


def test_diagnostic_mixture_preserves_the_preregistered_old_new_split():
    import task78_train as train
    assert sum(train.MIX_DIAGNOSTIC.values())==512
    assert train.MIX_DIAGNOSTIC['old_episode']==sum(v for k,v in train.MIX.items() if k.startswith('old_'))
    for name in ('fog','roof'):
        assert train.MIX_DIAGNOSTIC[name]==train.MIX[name]
    assert set(train.MIX_DIAGNOSTIC)=={'old_episode','fog','roof'}


def test_reconstructed_corpus_requires_a_marked_substitute_cache(tmp_path,monkeypatch):
    import numpy as np
    import torch
    import task78_train as train
    init=tmp_path/'model.pt'
    torch.save(dict(state_dict={}),init)
    episodes=tmp_path/'episodes';episodes.mkdir()
    (episodes/'episodes.json').write_text(json.dumps(dict(rows=2)))
    monkeypatch.setattr(train,'BASELINE_SHA',train.sha(init))
    with pytest.raises(ValueError,match='substitute'):
        train.reconstructed_sources(init,episodes)
    monkeypatch.setattr(train,'load_episodes',lambda directory: (
        torch.zeros(2,1),torch.zeros(2,1),torch.zeros(2),
        np.array([11,12]),np.array([True,False]),['day_d7','day_d7']))
    (episodes/'episodes.json').write_text(json.dumps(dict(rows=2,substitute_for='x')))
    sources,initial,provenance=train.reconstructed_sources(init,episodes)
    assert set(sources)=={'old_episode'} and provenance['preregistered_corpus'] is False
    assert (len(sources['old_episode'].fit),len(sources['old_episode'].val))==(1,1)


def test_terminal_panel_sources_cannot_overwrite_training_labels(tmp_path):
    from task78_collect import paths
    manifest=tmp_path/'roof_truth/k1/sources.json'
    replay,labels=paths(manifest)
    assert replay==manifest.with_name('replay.json')
    assert labels==manifest.parent/'labels'
    assert labels!=ROOT/'artifacts/task78/labels'


if __name__ == '__main__':
    unittest.main()
