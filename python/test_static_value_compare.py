import unittest
from static_value_compare import key,pool_actions
from static_value_compare_stats import clustered,ordering_credit,point_counts


class CommonPoolTest(unittest.TestCase):
    def test_pool_contains_both_teacher_sets_and_is_legal_and_unique(self):
        actions=[dict(type='plant',packet=p,row=0,col=c) for p in range(2) for c in range(6)]
        wait={'type':'wait','ticks':60}
        scores={'mac':[(1-i/100,a) for i,a in enumerate(actions)],
                'desktop':[(i/100,a) for i,a in enumerate(actions)]}
        arms={'mac':{'candidates':[actions[0],actions[6],wait]},
              'desktop':{'candidates':[actions[5],actions[-1],wait]}}
        result=pool_actions(scores,arms,actions,20261020)
        self.assertEqual(result,pool_actions(scores,arms,actions,20261020))
        keys={key(a) for a in result}
        self.assertEqual(len(keys),len(result))
        self.assertTrue(all(key(a) in {key(x) for x in actions+[wait]} for a in result))
        self.assertTrue(all(key(a) in keys for v in arms.values() for a in v['candidates']))
        # Two model choices each, then one random as yet uncovered cell per card.
        self.assertEqual(len(result),11)

    def test_wait_identity_ignores_timeout_spelling_only_for_fixed_teacher_waits(self):
        self.assertEqual(key({'type':'wait','ticks':60}),key({'type':'wait','ticks':60,'until':'timeout'}))


class PairedStatisticsTest(unittest.TestCase):
    def test_ties_do_not_invent_truth_or_award_an_order(self):
        self.assertIsNone(ordering_credit(1.,0.))
        self.assertEqual(ordering_credit(0.,.1),.5)
        self.assertEqual(ordering_credit(-.2,.1),0.)

    def test_known_ordering_and_heavy_label_differences_cluster_by_game(self):
        a=dict(type='plant',packet=0,row=0,col=0)
        b=dict(type='plant',packet=0,row=0,col=1)
        wait=dict(type='wait',ticks=60)
        arms={'mac':{'chosen':b,'candidates':[b,wait,a],'continuations':12},
              'desktop':{'chosen':a,'candidates':[a,wait,b],'continuations':12}}
        arms['mac_noise']={**arms['mac']}
        arms['desktop_noise']={**arms['desktop'],'chosen':b}
        row={'actions':[a,b,wait],'truth_scores':[[.8]*64,[.3]*64,[.2]*64],
             'static_scores':{'mac':[.2,.7,.1],'desktop':[.8,.3,.1]},
             'normal_arms':arms,'heavy_label':a,'policy':wait,
             'truth_continuations':192,'heavy_extra_continuations':36}
        counts=point_counts(row)
        self.assertEqual(counts['within_mac'],0)
        self.assertEqual(counts['within_desktop'],1)
        self.assertEqual(counts['all_mac'],2)
        self.assertEqual(counts['all_desktop'],3)
        self.assertAlmostEqual(counts['different_top_desktop_minus_mac_sum'],.5)
        result=clustered([counts,counts])
        self.assertEqual(result['games'],2)
        metrics=result['metrics']
        self.assertEqual(metrics['within_desktop_minus_mac']['game_cluster_ci95'],[1,1])
        self.assertEqual(metrics['heavy_agreement_desktop_minus_mac']['game_cluster_ci95'],[1,1])
        self.assertEqual(metrics['reseed_agreement_desktop_minus_mac']['game_cluster_ci95'],[-1,-1])

    def test_missing_pair_information_is_not_scored_as_a_tie_or_success(self):
        result=clustered([{'points':1,'mac_heavy':1,'desktop_heavy':0}])
        self.assertIsNone(result['metrics']['within_desktop_minus_mac']['estimate'])
        self.assertEqual(result['metrics']['heavy_agreement_desktop_minus_mac']['estimate'],-1)


if __name__=='__main__':unittest.main()
