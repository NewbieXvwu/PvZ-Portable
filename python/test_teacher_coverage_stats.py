import unittest
from teacher_coverage_stats import cluster_metrics, wilson


class CoverageStatisticsTest(unittest.TestCase):
    def test_clustered_cost_ratio_and_agreement_on_known_games(self):
        wait={'type':'wait'};plant={'type':'plant','packet':0,'row':0,'col':0}
        rows=[]
        for seed in (1,1,2):
            arms={name:{'chosen':wait,'actual_continuations':10,
                        'worker_and_simulator_cpu_seconds':.2}
                  for name in ('old','old_noise','new2','new2_noise','new4','new4_noise')}
            for name in ('new2','new2_noise','new4','new4_noise'):
                arms[name].update(chosen=plant,actual_continuations=12)
            rows.append({'seed':seed,'arms':arms})
        result=cluster_metrics(rows)
        self.assertEqual(result['points'],3)
        self.assertEqual(result['games'],2)
        metrics=result['metrics']
        self.assertEqual(metrics['old_noise']['game_cluster_ci95'],[1,1])
        self.assertEqual(metrics['new2_agreement']['estimate'],0)
        self.assertAlmostEqual(metrics['new2_count_increase']['estimate'],.2)
        for v in metrics['new2_count_increase']['game_cluster_ci95']:self.assertAlmostEqual(v,.2)
        self.assertEqual(metrics['new2_agreement_minus_old_noise']['game_cluster_ci95'],[-1,-1])

    def test_wilson_reports_uncertainty_at_boundary(self):
        lo=wilson(0,100);hi=wilson(100,100)
        self.assertAlmostEqual(lo['wilson95'][0],0)
        self.assertGreater(lo['wilson95'][1],0)
        self.assertLess(hi['wilson95'][0],1)
        self.assertAlmostEqual(hi['wilson95'][1],1)


if __name__=='__main__':unittest.main()
