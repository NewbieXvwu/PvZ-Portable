"""Regression gates for task35's performance evidence; no simulator is needed."""
import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import bench_engine as bench


def report():
    return {"families": {"day": {"rollouts": 6, "replay_mismatches": 0,
            "rollout_digests": [{"frame": i, "digest": str(i)} for i in range(6)]}},
            "total": {"ticks_per_cpu_second": 100.0}}


class EngineBenchGateTest(unittest.TestCase):
    def test_checks_later_rollouts_and_statistics(self):
        left, right = report(), report()
        right["families"]["day"]["rollout_digests"][5]["digest"] = "changed"
        with self.assertRaisesRegex(RuntimeError, "state/statistics differ"):
            bench._require_identical(left, right)

    def test_rejects_truncated_evidence_and_replay_mismatch(self):
        for truncated in (False, True):
            bad = report()
            if truncated:
                bad["families"]["day"]["rollout_digests"].pop()
            else:
                bad["families"]["day"]["replay_mismatches"] = 1
            with self.assertRaisesRegex(RuntimeError, "incomplete or invalid"):
                bench._require_identical(report(), bad)

    def test_digest_includes_logical_stats_but_ignores_timers(self):
        stats = dict(estimated_result=-1, decisions=1, mowers_left=5, planted=[0] * 10,
                     net_calls=1, ticks=60, substeps=[[60, 0]], simulation_seconds=1)
        baseline = bench._rollout_digest({"tick": 60, "wave": 1}, stats)
        stats["simulation_seconds"] = 999
        self.assertEqual(baseline, bench._rollout_digest({"tick": 60, "wave": 1}, stats))
        for key in bench.ROLLOUT_STAT_KEYS:
            changed = copy.deepcopy(stats)
            changed[key] = None
            self.assertNotEqual(baseline, bench._rollout_digest({"tick": 60, "wave": 1}, changed))
        self.assertNotEqual(baseline, bench._rollout_digest({"tick": 60, "wave": 1, "terminal": True}, stats))

    def test_rejects_empty_evidence(self):
        with self.assertRaisesRegex(RuntimeError, "empty"):
            bench._require_identical({"families": {}}, {"families": {}})
        empty = report()
        empty["families"]["day"].update(rollouts=0, rollout_digests=[])
        with self.assertRaisesRegex(RuntimeError, "incomplete or invalid"):
            bench._require_identical(empty, empty)

    def test_compare_saves_summary_and_keeps_rounds_separate(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = argparse.Namespace(executable=root / "a", compare=root / "b", rounds=2,
                                      out=root / "comparison.json")
            outputs = []
            def fake_bench(namespace):
                outputs.append(namespace.out)
                result = report()
                result["total"]["ticks_per_cpu_second"] = 200 if namespace.executable == args.compare else 100
                return result
            with patch.object(bench, "bench", side_effect=fake_bench):
                result = bench.compare(args)
            self.assertEqual(result, json.loads(args.out.read_text()))
            self.assertEqual(result["ratio_median"], 2)
            self.assertEqual(len(set(outputs)), 4)
            self.assertEqual([p.name for p in outputs],
                             ["round_1_a.json", "round_1_b.json", "round_2_b.json", "round_2_a.json"])

    def test_failed_compare_does_not_emit_success_summary(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = argparse.Namespace(executable=root / "a", compare=root / "b", rounds=1,
                                      out=root / "comparison.json")
            bad = report()
            bad["families"]["day"]["rollout_digests"][-1]["digest"] = "different"
            with patch.object(bench, "bench", side_effect=[report(), bad]):
                with self.assertRaises(RuntimeError):
                    bench.compare(args)
            self.assertFalse(args.out.exists())

    def test_search_reads_games_and_enforces_worker_limit(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = argparse.Namespace(out=root, manifest=root / "manifest.json", task_id="day",
                                      seed_offset=0, games=2, executable=root / "engine", workers=2)
            rows = [dict(seed=i, won=True, seconds=2, decisions=10) for i in range(2)]
            (root / "games.jsonl").write_text("\n".join(map(json.dumps, rows)))
            with patch.object(bench.subprocess, "run") as run:
                result = bench.search(args)
                self.assertEqual(result["games"], 2)
                self.assertEqual(result["milliseconds_per_decision"], 200)
                args.workers = 3
                with self.assertRaises(ValueError):
                    bench.search(args)
                self.assertEqual(run.call_count, 1)
            args.workers = 1
            args.games = 8
            with patch.object(bench.subprocess, "run"), self.assertRaisesRegex(RuntimeError, "incomplete"):
                bench.search(args)


if __name__ == "__main__":
    unittest.main()
