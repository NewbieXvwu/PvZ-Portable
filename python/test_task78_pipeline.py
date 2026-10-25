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
    def test_only_the_human_fog_group_retains_three_horizons(self):
        self.assertEqual(export.collection_horizons(dict(group='fog_l31_human')), (2500,4000,6000))
        for group in ('fog_demo_l31', 'fog_demo_l32', 'fog_demo_l33_cactus',
                      'fog_demo_l33_cattail', 'roof_l41'):
            self.assertEqual(export.collection_horizons(dict(group=group)), (2500,))
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
            self.assertEqual(len(cell['seeds']), cell['pairs'])
            self.assertFalse(seen & set(cell['seeds']))
            seen |= set(cell['seeds'])
        self.assertLessEqual(max(seen), ab.SEED_START + 1999)
        self.assertGreaterEqual(min(seen), ab.SEED_START)
        self.assertEqual(sum(c['pairs'] for c in cells), 416)


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


if __name__ == '__main__':
    unittest.main()
