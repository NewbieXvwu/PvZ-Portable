import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('qc_metrics',ROOT/'scripts/qc_metrics_throughput.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)


class TelemetryEvidenceTest(unittest.TestCase):
    def test_long_missing_interval_is_not_zero_usage_or_observed_time(self):
        rows=[dict(timestamp=t,mem_available_bytes=10*m.GIB,tree_cpu_percent=100) for t in (0,30,300,330)]
        d=m.weighted_window(rows,0,330)
        self.assertEqual(d['observed_seconds'],60)
        self.assertAlmostEqual(d['coverage'],60/330)
        self.assertEqual(d['tree_cpu_mean_percent'],100)

    def test_cpu_uses_ending_sample_and_window_is_clipped(self):
        rows=[dict(timestamp=0,mem_available_bytes=8*m.GIB,tree_cpu_percent=0),
              dict(timestamp=30,mem_available_bytes=10*m.GIB,tree_cpu_percent=200)]
        d=m.weighted_window(rows,10,20)
        self.assertEqual(d['observed_seconds'],10)
        self.assertEqual(d['tree_cpu_mean_percent'],200)
        self.assertEqual(d['mem_available_mean_gib'],8)

    def test_worker_prediction_accounts_for_rounded_log_not_invented_precision(self):
        d=m.worker_formula(dict(memory='18.4',per_worker='0.78',upper='16',workers='15'),15)
        self.assertTrue(d['consistent']);self.assertLessEqual(d['predicted_min'],15)
        self.assertGreaterEqual(d['predicted_max'],15)
        self.assertFalse(m.worker_formula(None,8)['available'])

    def test_isolated_controller_is_not_production_evidence(self):
        row=dict(argv=['python','/x/curriculum_queue.py','--state','/tmp/smoke_state.json'])
        with self.assertRaisesRegex(ValueError,'nonproduction state'):m.verify_production(row)

    def test_background_unconsumed_headroom_reduces_worker_budget(self):
        meta=dict(memory='10.1',per_worker='0.86',upper='16',workers='2',background='1.64')
        d=m.worker_formula(meta,2)
        self.assertTrue(d['consistent']);self.assertEqual(d['background_headroom_gib'],1.64)
        self.assertFalse(m.worker_formula({**meta,'background':None},2)['consistent'])

    def test_collection_rss_peak_is_not_misattributed_to_the_following_pack(self):
        rows=[dict(timestamp=0,rss_bytes=12*m.GIB,children_rss_bytes=0,mem_available_bytes=8*m.GIB,tree_cpu_percent=1600),
              dict(timestamp=30,rss_bytes=m.GIB,children_rss_bytes=0,mem_available_bytes=20*m.GIB,tree_cpu_percent=100),
              dict(timestamp=60,rss_bytes=m.GIB,children_rss_bytes=0,mem_available_bytes=20*m.GIB,tree_cpu_percent=100)]
        d=m.weighted_window(rows,5,60)
        self.assertEqual(d['sampled_tree_rss_peak_gib'],1)
        self.assertEqual(d['mem_available_min_gib'],20)

    def test_unavailable_resource_is_not_reported_as_zero(self):
        rows=[dict(timestamp=t,mem_available_bytes=None,tree_cpu_percent=None) for t in (0,30)]
        d=m.weighted_window(rows,0,30)
        self.assertIsNone(d['sampled_tree_rss_peak_gib'])
        self.assertIsNone(d['mem_available_mean_gib']);self.assertIsNone(d['tree_cpu_mean_percent'])
        self.assertEqual(d['memory_coverage'],0);self.assertEqual(d['cpu_coverage'],0)


if __name__=='__main__':unittest.main()
