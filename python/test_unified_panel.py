"""Task23 panel membership, original conveyors and inherited holdout semantics."""
import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from unified_panel import generate


class UnifiedPanelTest(unittest.TestCase):
    def test_all_required_families_and_original_modes_without_mutating_sources(self):
        def task(name, level, deck):
            return dict(task_id=name, level=level, terrain='day' if level<11 else 'night', deck=deck,
                        seeds=list(range(30000,30256)), zombie_count_multiplier=1.0, wave_cap=None,
                        sun_start=50, preplanted=[], playthrough=2)
        day = dict(tasks=[task(f'day_l{level}_eval_d7',level,[0,1,2,3,4,5,7]) for level in (1,3,4,6,7,8,9,10)])
        perturb = dict(tasks=[dict(task(f'day_l{level}_eval_d7p_holdout{i}',level,[0,1,2,3,4,5,6]),
                                   split='heldout_deck', holdout_semantics='archived initial set')
                              for level in (7,8,9,10) for i in (1,2,3)])
        night = dict(tasks=[task(f'night_l{level}_eval_n9',level,[0,1,2,3,4,5,7,8,9])
                            for level in (11,12,13,14,16,17,18,19,20)])
        previous = copy.deepcopy((day,perturb,night)); result=generate(day,perturb,night)
        self.assertEqual(previous,(day,perturb,night))
        self.assertEqual(len(result['tasks']),29)
        from collections import Counter
        self.assertEqual(Counter(result['groups'].values()),dict(day=8,heldout=12,night=9))
        by_id={t['task_id']:t for t in result['tasks']}
        for name in ('day_l10_orig_eval','night_l20_eval_n9_orig'):
            self.assertEqual(by_id[name]['deck'],[])
            self.assertEqual(by_id[name]['conveyor_start'],'original')
        for held in perturb['tasks']:
            self.assertEqual(by_id[held['task_id']],held)
        self.assertNotIn('day_l10_eval_d7',by_id)
        self.assertNotIn('night_l20_eval_n9',by_id)
        self.assertNotIn('night_l15_eval_n9',by_id)


if __name__=='__main__': unittest.main()
