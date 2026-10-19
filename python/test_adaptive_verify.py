from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from collections import Counter
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from adaptive_verify import cpu_pair
import search_play


class AdaptiveVerificationTest(unittest.TestCase):
    def run_pair(self, scores, *, gap=.05, totals=None):
        calls, batches = [], []
        args = SimpleNamespace(samples=3, horizon=2500, margin=.02, adaptive_verify_gap=gap)
        def rollout(env, obs, action, until, root, seed):
            self.assertEqual((until, root), (2600, 7))
            calls.append((action, seed))
            return scores[action][seed % 1000 - 3]
        def values(frames):
            batches.append([f['value'] for f in frames])
            return [f['value'] for f in frames]
        result = cpu_pair(None, {'tick':100}, 7, ['policy','challenger'], (0,1),
                          [.4,.5] if totals is None else totals, [3,3], args, 123, 9,
                          rollout=rollout, terminal_score=lambda x:x.get('terminal_value'),
                          frame=lambda x:x.copy(), values=values)
        return result, calls, batches

    def test_strong_override_uses_three_futures_per_leg(self):
        result, calls, _ = self.run_pair({'policy':[{'terminal_value':.4}]*9,
                                         'challenger':[{'terminal_value':1.}]*9})
        self.assertTrue(result['early_stop'])
        self.assertEqual(result['samples'],3)
        self.assertEqual(len(calls),6)
        self.assertGreater(result['means'][1]-result['means'][0],.02)

    def test_strong_reversal_can_cancel_override_early(self):
        result, calls, _ = self.run_pair({'policy':[{'terminal_value':1.}]*9,
                                         'challenger':[{'terminal_value':0.}]*9})
        self.assertTrue(result['early_stop'])
        self.assertEqual(len(calls),6)
        self.assertLess(result['means'][1]-result['means'][0],.02)

    def test_ambiguous_initial_pair_finishes_original_futures(self):
        result, calls, _ = self.run_pair({'policy':[{'terminal_value':.5}]*9,
                                         'challenger':[{'terminal_value':.5}]*3+[{'terminal_value':.1}]*6})
        self.assertFalse(result['early_stop'])
        self.assertEqual(result['samples'],9)
        self.assertEqual(len(calls),18)
        for action in ['policy','challenger']:
            self.assertEqual([s for a,s in calls if a==action],list(range(123003,123012)))
        self.assertEqual(result['means'],[(.4*3+sum([.5]*9))/12,
                                          (.5*3+sum([.5]*3+[.1]*6))/12])

    def test_teacher_opt_in_records_actual_budget_and_cancels_override(self):
        args = SimpleNamespace(samples=3, horizon=2500, margin=.02, verify_samples=9,
                               adaptive_verify_gap=.05)
        wait = dict(search_play.NOOP)
        plant = {'type':'plant','packet':0,'row':0,'col':0}
        stats = {'overrides':0,'override_kinds':Counter()}
        with patch.dict(search_play._W,{},clear=True), patch.object(search_play,'_rollout_policy',
                side_effect=[{'rollout_result':1.}]*3+[{'rollout_result':0.}]*3) as rollout:
            chosen = search_play._verified_choice(None,{'tick':100},7,[wait,plant],
                [.4,.5],[3,3],1,0,wait,args,123,stats)
        self.assertEqual(chosen,wait)
        self.assertEqual(rollout.call_count,6)
        self.assertEqual(stats['adaptive_verify_saved_lanes'],12)
        self.assertEqual(stats['last_verification']['samples'],3)
        self.assertEqual(stats['last_verification']['requested_samples'],9)
        self.assertTrue(stats['last_verification']['adaptive_early_stop'])
        self.assertFalse(stats['last_verification']['after_override'])

    def test_fallback_recomputes_full_terminal_first_batches_and_sum(self):
        leaves=[{'value':.2},{'terminal_value':1.},{'value':.4}]*3
        result, calls, batches = self.run_pair({'policy':leaves,'challenger':leaves},gap=.99)
        self.assertFalse(result['early_stop'])
        self.assertEqual([len(b) for b in batches],[2,2,6,6])
        added=[1.,1.,1.]+[.2,.4]*3
        self.assertEqual(result['means'],[(.4*3+sum(added))/12,(.5*3+sum(added))/12])
        self.assertEqual(len(calls),18)


if __name__ == '__main__':
    unittest.main()
