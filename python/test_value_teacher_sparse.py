import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import value_teacher_sparse as sparse
from value_teacher_feasibility import digest,validate_existing_candidate,write_json


class SparseTeacherTest(unittest.TestCase):
    def test_midpoint_quantiles_use_actual_order_without_terminal_outcomes(self):
        roots=[dict(decision=i*7,won=i%2) for i in range(195)]
        selected=sparse.spaced_roots(roots)
        self.assertEqual([r['decision']//7 for r in selected],[(2*i+1)*195//24 for i in range(12)])
        self.assertEqual(len({r['decision'] for r in selected}),12)
        for r in roots:r['won']=not r['won']
        self.assertEqual([r['decision'] for r in sparse.spaced_roots(roots)], [r['decision'] for r in selected])
        self.assertEqual(sparse.spaced_roots(roots[:3]),roots[:3])
        self.assertEqual(sparse.spaced_roots([]),[])

    def test_resume_rejects_changed_leaf_or_missing_terminal_label(self):
        expected=dict(leaf_state_sha256='leaf',leaf_history=dict(previous_frame=dict(plants=[(1,2,3)])),
                      common_future=33,history_checks=dict(frame=True))
        previous=json.loads(json.dumps(expected));previous['outcome']=dict(terminal=True,won=True,result=1)
        validate_existing_candidate(previous,expected)
        previous['leaf_state_sha256']='another'
        with self.assertRaises(ValueError):validate_existing_candidate(previous,expected)
        previous['leaf_state_sha256']='leaf';previous['outcome']['terminal']=False
        with self.assertRaises(ValueError):validate_existing_candidate(previous,expected)

    def test_disk_gate_counts_owned_files_without_shared_symlinks(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);out=r/'out';out.mkdir();shared=r/'shared';shared.write_bytes(b'x'*100)
            (out/'own').write_bytes(b'1234');(out/'link').symlink_to(shared)
            self.assertEqual(sparse.artifact_bytes(out),4)

    def test_cpu_budget_accumulates_attempts_and_filters_bootstrap(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d)
            for seed,phase,cost in [(1,'roots_old',2),(1,'roots_new',3),(1,'labels',5),(2,'labels',100)]:
                write_json(r/f'seed_{seed}'/'usage'/f'{phase}.json',dict(cpu_seconds=cost))
            write_json(r/'runs/a/controller_usage.json',dict(seeds=[1],cpu_seconds=1))
            write_json(r/'runs/b/controller_usage.json',dict(seeds=[1,2],cpu_seconds=10))
            self.assertEqual(sparse.used_cpu(r),121)
            self.assertEqual(sparse.used_cpu(r,{1}),11)

    def test_selection_must_match_completed_trajectory_and_frozen_teacher(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);(r/'root_trajectory.jsonl').write_text('{}\n')
            roots=[dict(decision=i*3) for i in range(40)]
            write_json(r/'trajectory_result.json',dict(search_roots=roots,teacher_sha256={'value':'fixed'}))
            selected=dict(seed=1,roots=sparse.spaced_roots(roots),trajectory_sha256=digest(r/'root_trajectory.jsonl'))
            write_json(r/'selection.json',selected)
            with patch.object(sparse,'_PLAN',dict(roots_per_seed=12,sha256={'value':'fixed'})):
                sparse.verify_selection(r,1)
                selected['roots'][0]={'decision':999};write_json(r/'selection.json',selected)
                with self.assertRaises(ValueError):sparse.verify_selection(r,1)

    def test_plan_rejects_duplicate_bootstrap_and_excess_workers(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);f=r/'input';f.write_text('frozen');keys=('value','static_value','rollout_policy','executable','manifest')
            p=dict(seeds=list(range(64)),bootstrap_seeds=list(range(8)),held_out_seeds=list(range(8,16)),
                roots_per_seed=12,workers=4,teacher={k:str(f) for k in keys},sha256={k:digest(f) for k in keys},
                search_source_sha256=digest(ROOT/'scripts/search_play.py'),out=str(r/'out'),artifact_root=str(r/'owned'))
            path=r/'plan.json';write_json(path,p);sparse.load_plan(path)
            p['bootstrap_seeds']=[1]*8;write_json(path,p)
            with self.assertRaises(ValueError):sparse.load_plan(path)
            p['bootstrap_seeds']=list(range(8));p['workers']=5;write_json(path,p)
            with self.assertRaises(ValueError):sparse.load_plan(path)

    def test_bootstrap_enforces_pair_cpu_and_disk_limits_together(self):
        with tempfile.TemporaryDirectory() as d:
            p=dict(out=d,roots_per_seed=12,bootstrap_min_pairs=125,cpu_budget_seconds=60*3600,
                   artifact_budget_bytes=5_000_000_000,artifact_root=d)
            def groups(directory,seed):
                n,w=(8,4) if seed<7 else (14,1)
                return [dict(labels=n,wins=w,mixed=True,effective_pairs=w*(n-w),nonterminal_pairs=w*(n-w),
                             nonterminal_labels=n,nonterminal_wins=w)]
            with patch.object(sparse,'audit_seed',side_effect=groups),patch.object(sparse,'used_cpu',return_value=7.5*3600), \
                 patch.object(sparse,'artifact_bytes',return_value=5_000_000_000),contextlib.redirect_stdout(io.StringIO()):
                self.assertTrue(sparse.report(p,list(range(8)),gate=True)['approved_to_continue'])
                p['bootstrap_min_pairs']=126
                self.assertFalse(sparse.report(p,list(range(8)),gate=True)['approved_to_continue'])
                p['bootstrap_min_pairs']=125;p['cpu_budget_seconds']=60*3600-1
                self.assertFalse(sparse.report(p,list(range(8)),gate=True)['approved_to_continue'])
                p['cpu_budget_seconds']=60*3600;p['artifact_budget_bytes']=4_999_999_999
                self.assertFalse(sparse.report(p,list(range(8)),gate=True)['approved_to_continue'])


if __name__=='__main__':unittest.main()
