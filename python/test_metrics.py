"""Exercise real sampling and the time-based warning/retention/report contracts."""
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from pvz_metrics import ActiveWindow, Metrics, summarize


class MetricsTests(unittest.TestCase):
    def test_live_sample_and_rates(self):
        with tempfile.TemporaryDirectory() as folder:
            metrics = Metrics('unit-test', directory=folder)
            metrics.update(step=0, rows=0)
            first = metrics.sample()
            metrics.update(step=2, rows=4096)
            last = metrics.sample()
            self.assertEqual(last['pid'], os.getpid())
            self.assertGreater(last['rss_bytes'], 0)
            self.assertGreater(last['mem_available_bytes'], 0)
            self.assertIn('some', last['pressure_memory'])
            self.assertGreater(last['rates']['rows_per_second'], 0)
            self.assertGreaterEqual(last['io']['read_bytes'], first['io']['read_bytes'])
            self.assertEqual(len(metrics.path.read_text().splitlines()), 2)

    def test_warning_requires_ten_minutes_and_known_gpu(self):
        window = ActiveWindow()
        for now in range(0, 600, 30):
            self.assertFalse(window.observe(now, 0)[2])
        self.assertTrue(window.observe(600, 0)[2])
        self.assertFalse(window.observe(630, 0)[2])
        window.observe(660, None)
        self.assertFalse(window.observe(690, 0)[2])
        window = ActiveWindow()
        for now in range(0, 630, 30):
            result = window.observe(now, 90)
        self.assertEqual(result, (1.0, 600, False))

    def test_rotation_retention_and_thread_shutdown(self):
        with tempfile.TemporaryDirectory() as folder:
            old = Path(folder) / 'old-123.jsonl'
            old.write_text('{}\n')
            os.utime(old, (time.time() - 8 * 86400,) * 2)
            metrics = Metrics('rotate', directory=folder, interval=.01)
            with patch('pvz_metrics.gpu_sample', return_value=None):
                metrics.sample()
                self.assertFalse(old.exists())
                metrics._day = '2000-01-01'
                metrics.sample()
                self.assertTrue(Path(folder, f'rotate-{os.getpid()}.2000-01-01.jsonl').exists())
                metrics.start()
                metrics.close()
            self.assertFalse(metrics._thread.is_alive())
            for line in metrics.path.read_text().splitlines():
                json.loads(line)

    def test_summary_weights_seconds_and_skips_unknown_gpu(self):
        def row(t, util, read, write, available):
            return dict(event='sample', monotonic_s=t, gpu=None if util is None else [dict(utilization_percent=util)],
                        io=dict(read_bytes=read, write_bytes=write), mem_available_bytes=available,
                        rss_bytes=10, children_rss_bytes=20)
        rows = [row(0, 0, 10, 20, 100), row(30, 90, 30, 50, 80),
                row(40, 0, 40, 60, 90), row(70, None, 50, 70, 70)]
        result = summarize(rows)
        self.assertEqual(result['training_effective_ratio'], .75)
        self.assertEqual(result['gpu_observed_seconds'], 40)
        self.assertEqual(result['disk_read_bytes'], 40)
        self.assertEqual(result['disk_write_bytes'], 50)
        self.assertEqual(result['minimum_mem_available_bytes'], 70)
        self.assertEqual(result['peak_tree_rss_bytes'], 30)
        self.assertIsNone(summarize([rows[-1]])['training_effective_ratio'])


if __name__ == '__main__':
    unittest.main()
