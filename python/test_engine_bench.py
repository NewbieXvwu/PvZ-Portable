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
    def test_replay_checks_v2_state_and_rejects_early_terminal(self):
        from unittest.mock import Mock
        rows = [{}, {'action': {'type': 'wait', 'ticks': 60}, 'tick': 60, 'v2': {'wave_timer': 10}},
                {'action': {'type': 'wait', 'ticks': 60}, 'tick': 120, 'v2': {'wave_timer': 9}}]
        spec = {'deck': [0, 1, 2, 3, 4, 5]}
        for early in (False, True):
            env = Mock()
            env.step.return_value = ({'tick': 60, 'v2': {'wave_timer': 10 if early else 11}},
                                     0, early, False, {})
            with patch.object(bench, '_task_spec'), patch.object(bench, '_frame', side_effect=lambda v: v):
                _, errors = bench.replay_episode(env, spec, rows, [2] if early else [1])
            self.assertTrue(errors)
            self.assertEqual(errors[0]['key'], 'terminal_before_last_frame' if early else 'v2')

    def test_batch_prototype_requires_all_roots_and_identical_leaves(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/prototypes'))
        import bench_batch_rollout as batch
        metrics = dict(process_cpu_seconds=2, network_cpu_seconds=1,
                       simulation_cpu_seconds=0.5, snapshot_cpu_seconds=0.2, feature_cpu_seconds=0.1)
        root = {'arms': {arm: dict(metrics) for arm in ('native', '1', '9')},
                'serial_batch_identical': True, 'native_batch_identical': True}
        data = {'roots': [root]}
        with self.assertRaisesRegex(RuntimeError, 'incomplete'):
            batch.summarize(data, 64)
        summary = batch.summarize(data, 1)
        self.assertEqual(summary['components']['native']['snapshot_cpu_seconds'], 0.2)
        self.assertEqual(summary['network_speedup'], 1)
        batch.require_identical(data)
        root['native_batch_identical'] = False
        with self.assertRaisesRegex(RuntimeError, 'statistics differ'):
            batch.require_identical(data)

    def test_horizon_starts_at_restored_snapshot_not_archive_post_action(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps({'families': {'day': [
                {'archive': str(root / 'archive'), 'decisions': [1]}]}}))
            response = dict(ticks=2500, decisions=42, estimated_result=-1, mowers_left=5,
                            net_calls=42, planted_slots=[0] * 10, planted_types={}, substeps=[[2600, 0]],
                            simulation_seconds=1, network_seconds=1)
            from unittest.mock import Mock
            env = Mock()
            env._process = SimpleNamespace(pid=123)
            env._last_response = response
            env.restore.return_value = {'tick': 100}
            env.rollout.return_value = {'tick': 2600}
            args = argparse.Namespace(manifest=manifest, executable=root / 'engine', sim_skip=61,
                                      horizon=2500, rollouts=1, seed_base=0, max_actions=1000,
                                      out=root / 'bench.json')
            with patch.object(bench, '_export_student'), patch.object(bench, '_open_env', return_value=env), \
                    patch.object(bench, '_archive_rows', return_value=[{'tick': 100}, {'tick': 160}]), \
                    patch.object(bench, 'replay_episode', return_value=({1: 1}, [])), \
                    patch.object(bench, '_process_cpu', side_effect=[0, 1]), patch.object(bench, '_arm'):
                result = bench.bench(args)
            self.assertEqual(env.rollout.call_args.args[1], 2600)
            self.assertEqual(result['benchmark_version'], 2)

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
        stats = dict(estimated_result=-1, decisions=1, mowers_left=5, planted_slots=[0] * 10, planted_types={},
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
                self.assertIsNone(result['mean_network_seconds'])
                self.assertIsNone(result['mean_simulation_seconds'])
                args.workers = 3
                with self.assertRaises(ValueError):
                    bench.search(args)
                self.assertEqual(run.call_count, 1)
            args.workers = 1
            args.games = 8
            with patch.object(bench.subprocess, "run"), self.assertRaisesRegex(RuntimeError, "incomplete"):
                bench.search(args)

    def test_missing_game_timers_are_not_reported_as_zero(self):
        rows = [dict(seed=i, seconds=2, decisions=10) for i in range(2)]
        rows[0]['network_seconds'] = 1
        self.assertIsNone(bench.summarize_games(rows, 2)['mean_network_seconds'])
        for row in rows:
            row['network_seconds'] = 0
        self.assertEqual(bench.summarize_games(rows, 2)['mean_network_seconds'], 0)

    def test_games_pairs_the_two_binaries_and_reports_cpu_per_game(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = argparse.Namespace(out=root, manifest=root / "manifest.json", task=["day"],
                                      seed_offset=0, games=2, workers=2,
                                      executable=root / "new", compare=root / "old")
            rows = [dict(seed=i, won=True, seconds=9, decisions=10) for i in range(2)]
            spent = [{"cpu_seconds": 20.0, "wall_seconds": 40.0}, {"cpu_seconds": 40.0, "wall_seconds": 80.0}]
            with patch.object(bench, "run_games", side_effect=[(rows, spent[0]), (rows, spent[1])]) as run:
                report = bench.games(args)
            self.assertEqual([call.args[1] for call in run.call_args_list], ["day", "day"])
            self.assertEqual(report["tasks"][0]["cpu_ratio_old_over_new"], 2.0)
            self.assertEqual(report["tasks"][0]["new"]["cpu_seconds_per_game"], 10.0)
            self.assertEqual(report["tasks"][0]["differing_seeds"], [])
            self.assertTrue((root / "engine_games.json").exists())

    def test_games_flags_seeds_whose_outcome_differs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = argparse.Namespace(out=root, manifest=root / "manifest.json", task=["day"],
                                      seed_offset=0, games=2, workers=2,
                                      executable=root / "new", compare=root / "old")
            new_rows = [dict(seed=1, won=True, seconds=9, decisions=10),
                        dict(seed=2, won=False, seconds=9, decisions=10)]
            old_rows = [dict(new_rows[0]), dict(new_rows[1], won=True)]
            spent = {"cpu_seconds": 10.0, "wall_seconds": 10.0}
            with patch.object(bench, "run_games", side_effect=[(new_rows, spent), (old_rows, spent)]):
                report = bench.games(args)
            self.assertEqual(report["tasks"][0]["differing_seeds"], [2])


if __name__ == "__main__":
    unittest.main()
