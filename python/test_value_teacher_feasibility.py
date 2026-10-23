import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from value_teacher_feasibility import group_summary, load_plan, digest


class TeacherFeasibilityTest(unittest.TestCase):
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


if __name__ == '__main__': unittest.main()
