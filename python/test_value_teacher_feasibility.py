import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from value_teacher_feasibility import group_summary, leaf_checks, load_plan, digest


class TeacherFeasibilityTest(unittest.TestCase):
    def test_terminal_leaf_state_hash_is_recorded_not_failed(self):
        live = dict(terminal=False)
        self.assertFalse(all(leaf_checks(True, False, True, True, live).values()))
        terminal = leaf_checks(True, False, True, True, dict(terminal=True))
        self.assertTrue(all(terminal.values()))
        self.assertIn('terminal_private_state', terminal)
        # A terminal leaf whose public frame differs is still a hard failure.
        self.assertFalse(all(leaf_checks(False, False, True, True, dict(terminal=True)).values()))
        # Non-terminal leaves keep the bit-exact requirement.
        self.assertFalse(all(leaf_checks(True, False, True, True, dict(terminal=False)).values()))

    def test_effective_pairs_count_only_opposite_terminal_outcomes(self):
        rows = [dict(outcome=dict(terminal=True, won=x), continuation_cpu_seconds=2,
                     replay_cpu_seconds=.5) for x in (True, True, False, False, False)]
        result = group_summary(rows)
        self.assertEqual(result['effective_pairs'], 6)
        self.assertEqual(result['total_pairs'], 10)
        self.assertTrue(result['mixed'])
        self.assertEqual(result['continuation_cpu_seconds'], 10)
        self.assertEqual(result['replay_cpu_seconds'], 2.5)
        self.assertFalse(group_summary(rows[:2])['mixed'])
        self.assertTrue(group_summary(rows[2:])['all_loss'])

    def test_missing_terminal_labels_are_not_counted_as_losses(self):
        with self.assertRaises(ValueError):
            group_summary([])
        with self.assertRaises(ValueError):
            group_summary([dict(outcome=dict(terminal=False, won=False))])

    def test_changed_inputs_and_process_budget_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            r = Path(d); model = r/'input'; model.write_text('frozen')
            keys = ('value','static_value','rollout_policy','executable','manifest')
            p = dict(workers=4, seeds=[1,2,3,4], out=str(r/'out'),
                     teacher={k:str(model) for k in keys}, sha256={k:digest(model) for k in keys})
            plan = r/'plan.json'; plan.write_text(json.dumps(p))
            self.assertEqual(load_plan(plan)['seeds'], [1,2,3,4])
            model.write_text('changed')
            with self.assertRaises(ValueError): load_plan(plan)
            model.write_text('frozen'); p['workers']=5; plan.write_text(json.dumps(p))
            with self.assertRaises(ValueError): load_plan(plan)
            p['workers']=4; p['seeds']=[1,1]; plan.write_text(json.dumps(p))
            with self.assertRaises(ValueError): load_plan(plan)


class TeacherReportTest(unittest.TestCase):
    def setUp(self):
        from value_teacher_feasibility import write_json
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.write = write_json
        model=self.root/'input'; model.write_text('frozen')
        keys=('value','static_value','rollout_policy','executable','manifest')
        self.out=self.root/'run'; self.seed=2000064
        self.plan=self.root/'plan.json'
        write_json(self.plan,dict(workers=1,seeds=[self.seed],out=str(self.out),waves=[3],
            teacher={k:str(model) for k in keys},sha256={k:digest(model) for k in keys}))
        d=self.out/f'seed_{self.seed}'; self.group=d/'wave_3'
        write_json(self.out/'completed.json',dict(seeds=[self.seed]))
        write_json(d/'result.json',dict(seed=self.seed,terminal=True,won=True,searched=100,
            missing_wave_targets=[],total_cpu_seconds=10,wall_seconds=12,final_frame=dict(wave=20)))
        (d/'root_trajectory.jsonl').write_text('{}\n')
        actions=[dict(type='wait'),dict(type='plant')]; base=self.seed*10000+50
        write_json(self.group/'root.json',dict(decision_seed=base,common_future=base*1000%2**32,
            candidates=actions,root_frame=dict(tick=6000),
            sun_income_history=dict(start_tick=0,production=[[5900,25]])))
        rows=[]
        for i,won in enumerate([True,False]):
            row=dict(candidate=i,action=actions[i],common_future=base*1000%2**32,
                continuation_seed_base=base+5000,
                history_checks={k:True for k in ('frame','engine_state','income','income_start')},
                leaf_frame=dict(terminal=False,tick=8500),
                sun_income_history=dict(start_tick=0,production=[[5900,25],[8000,25]]),
                outcome=dict(won=won,terminal=True,result=int(won),decisions=1),
                continuation_cpu_seconds=3,replay_cpu_seconds=.5)
            rows.append(row); write_json(self.group/f'candidate_{i}.json',row)
            (self.group/f'candidate_{i}_teacher.jsonl').write_text('{"decision":0}\n')
        write_json(self.group/'summary.json',group_summary(rows))
        samples=[dict(time=t,logical_cpus=20,qc_cpu_seconds_by_pid={'3':t},own_engines=1,qc_engines=16) for t in (1,11)]
        (self.out/'resources.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in samples))

    def test_report_recomputes_signal_and_includes_cpu_overhead(self):
        from value_teacher_feasibility_report import report
        r=report([self.plan])
        self.assertEqual(r['effective_pairs'],1)
        self.assertEqual(r['nonterminal_effective_pairs'],1)
        self.assertEqual(r['mixed_fraction'],1)
        self.assertEqual(r['cpu_seconds']['per_effective_ranking_pair_total'],10)
        self.assertEqual(r['resources']['qc_mean_cpu_cores_lower_bound'],1)
        self.assertAlmostEqual(r['expansion_64']['same_three_roots_single_future']['cpu_hours'],640/3600)

    def test_report_rejects_missing_candidate_and_incomplete_history_checks(self):
        from value_teacher_feasibility_report import report
        f=self.group/'candidate_1.json'; row=json.loads(f.read_text())
        row['history_checks']={'frame':True};self.write(f,row)
        with self.assertRaises(ValueError):report([self.plan])
        f.unlink()
        with self.assertRaises(ValueError):report([self.plan])

    def test_report_rejects_changed_seed_even_with_same_count(self):
        from value_teacher_feasibility_report import report
        self.write(self.out/'completed.json',dict(seeds=[self.seed+1]))
        with self.assertRaises(ValueError):report([self.plan])

    def test_terminal_leaf_does_not_create_value_net_ranking_signal(self):
        from value_teacher_feasibility_report import report
        f=self.group/'candidate_1.json';row=json.loads(f.read_text())
        row['leaf_frame']['terminal']=True;self.write(f,row)
        r=report([self.plan])
        self.assertEqual(r['effective_pairs'],1)
        self.assertEqual(r['nonterminal_effective_pairs'],0)
        self.assertIsNone(r['cpu_seconds']['per_nonterminal_effective_pair_total'])

    def test_income_context_requires_ordered_events_before_leaf(self):
        from value_teacher_feasibility_report import report
        f=self.group/'candidate_1.json';row=json.loads(f.read_text())
        row['sun_income_history']['production'].append([8501,25]);self.write(f,row)
        with self.assertRaises(ValueError):report([self.plan])


if __name__ == '__main__': unittest.main()
