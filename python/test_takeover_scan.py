#!/usr/bin/env python3
"""task91：接手扫描运行器的纯决策逻辑（不跑引擎）。

覆盖 build_ladder 的阶梯规则与 AdaptiveScan 的取点顺序（4 等距 → 二分 → 补档）。
"""

import json
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import takeover_scan as scan
import import_llm_run as importer
from compare_same_block import compare_document

from takeover_scan import (AdaptiveScan, build_ladder, resolve_target_task, update_manifest,
                           update_test_expectations)


def grid12() -> list:
    return [1, 91, 182, 272, 363, 453, 544, 634, 725, 815, 906, 996]


def drive(scanner: AdaptiveScan, rate_fn) -> int:
    """用合成胜率驱动老师臂循环，返回实测点数。"""
    count = 0
    while True:
        decision = scanner.next_teacher_point()
        if decision is None:
            return count
        wins, games = rate_fn(decision)
        scanner.observe(decision, 'teacher', {'wins': wins, 'games': games, 'rows': []})
        count += 1


class ResumeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.args = scan.parser().parse_args(['run', str(self.root / 'jobs.json'), '--out', str(self.root / 'out')])
        self.job = dict(task_id='t', seed=42, archive=str(self.root / 'archive'),
                        decision=100, arm='teacher', future_seeds=[20, 21])
        self.entry = dict(task_id='t', seed=42, archive=self.job['archive'], decisions=[100],
                          future_seeds=[20, 21], jobs=[self.job])
        (self.root / 'jobs.json').write_text(json.dumps(dict(schema='llm_scan_jobs_v1', archives=[self.entry])))

    def row(self, future):
        return dict(seed=42, future_seed=future, arm='fast_search', won=True, final_wave=30)

    def test_resume_only_missing_requested_futures_and_filters_old_rows(self):
        out = scan.job_out(self.args.out, self.job)
        out.mkdir(parents=True)
        raw = out / 'games.jsonl'
        raw.write_text('\n'.join(json.dumps(self.row(f)) for f in (10, 20)) + '\n')
        def run(command, **kwargs):
            self.assertEqual(command[command.index('--future-seeds') + 1], '21')
            with raw.open('a') as stream:
                stream.write(json.dumps(self.row(21)) + '\n')
            return type('Result', (), dict(returncode=0, stdout=''))()
        with patch.object(scan, 'validate_job_context'), patch.object(scan.subprocess, 'run', side_effect=run) as execute:
            rows = scan.run_job(self.job, self.args)
            self.assertEqual([r['future_seed'] for r in rows], [20, 21])
            self.assertEqual(scan.run_job(self.job, self.args), rows)
            self.assertEqual(execute.call_count, 1)

    def test_equal_row_counts_with_different_futures_are_not_a_checkpoint(self):
        store = {'100': {'teacher': scan.point_summary([self.row(10), self.row(11)])}}
        self.assertEqual(scan.cached_points_for_futures(store, [20, 21]), {})
        path = self.args.out / 'results/t_s42.json'
        scan.write_json(path, dict(task_id='t', seed=42, archive=self.job['archive'], points=store))
        self.args.futures, self.args.teacher_only = 2, True
        with patch.object(scan, 'run_point', return_value=scan.point_summary([self.row(20), self.row(21)])) as run:
            scan.cmd_run(self.args)
        run.assert_called_once()
        result = json.loads(path.read_text())
        self.assertEqual(result['future_seeds'], [20, 21])
        self.assertEqual(result['points']['100']['teacher']['games'], 2)

    def test_archive_path_is_part_of_checkpoint_identity(self):
        path = self.args.out / 'results/t_s42.json'
        scan.write_json(path, dict(task_id='t', seed=42, archive='/another/archive', points={}))
        with patch.object(scan, 'run_point') as run:
            with self.assertRaisesRegex(ValueError, '另一个存档'):
                scan.cmd_run(self.args)
        run.assert_not_called()

    def test_complete_scan_is_skipped_when_futures_match(self):
        self.args = scan.parser().parse_args(['adaptive', str(self.root / 'jobs.json'),
                                              '--out', str(self.root / 'out'), '--skip-complete'])
        path = self.args.out / 'results/t_s42.json'
        scan.write_json(path, dict(task_id='t', seed=42, archive=self.job['archive'],
                                   complete=True, future_seeds=[20, 21], points={}))
        with patch.object(scan, 'run_point') as run:
            scan.cmd_adaptive(self.args)
        run.assert_not_called()

    def test_complete_scan_with_other_futures_reruns(self):
        self.args = scan.parser().parse_args(['adaptive', str(self.root / 'jobs.json'),
                                              '--out', str(self.root / 'out'), '--skip-complete'])
        path = self.args.out / 'results/t_s42.json'
        scan.write_json(path, dict(task_id='t', seed=42, archive=self.job['archive'],
                                   complete=True, future_seeds=[10, 11], points={}))
        with patch.object(scan, 'run_point', return_value=dict(wins=0, games=2, rows=[])) as run:
            scan.cmd_adaptive(self.args)
        run.assert_called()

    def test_fresh_adaptive_creates_results_directory(self):
        self.args = scan.parser().parse_args(['adaptive', str(self.root / 'jobs.json'), '--out', str(self.root / 'fresh')])
        with patch.object(scan, 'run_point', return_value=dict(wins=0, games=2, rows=[])):
            scan.cmd_adaptive(self.args)
        self.assertTrue((self.args.out / 'results/t_s42.json').is_file())

    def test_input_fingerprint_rejects_changed_weights(self):
        archive = Path(self.job['archive']); archive.mkdir()
        for name in ('meta.json', 'ticks.jsonl'):
            (archive / name).write_text('{}')
        for key in ('student', 'value', 'static_value', 'executable'):
            path = self.root / key; path.write_bytes(b'original')
            setattr(self.args, key, path)
        self.args.manifest = self.root / 'tasks.json'
        self.args.manifest.write_text(json.dumps({'tasks': [{'task_id': 't', 'seeds': [42]}]}))
        out = self.root / 'context'; out.mkdir()
        scan.validate_job_context(self.job, self.args, out, False)
        self.args.student.write_bytes(b'changed weights')
        with self.assertRaisesRegex(ValueError, '搜索输入已变化'):
            scan.validate_job_context(self.job, self.args, out, True)


class BuildLadderTest(unittest.TestCase):
    def test_ladder_uses_full_line_as_top(self):
        ladder = build_ladder({100: (4, 8), 200: (6, 8), 300: (8, 8), 400: (5, 8), 500: (8, 8)})
        self.assertEqual(ladder['ladder'], [500, 400, 300, 200, 100])
        self.assertEqual(ladder['top'], 500)
        self.assertEqual(ladder['l50'], 100)
        self.assertTrue(ladder['top_is_full'])

    def test_passing_above_top_is_excluded(self):
        # 400 的 5/8 在最晚 8/8（300）之上：不如稳定点，不入阶梯；4/8 恰好过线。
        ladder = build_ladder({100: (4, 8), 200: (6, 8), 300: (8, 8), 400: (5, 8)})
        self.assertEqual(ladder['ladder'], [300, 200, 100])
        self.assertEqual(ladder['top'], 300)

    def test_no_full_line_takes_latest_passing(self):
        ladder = build_ladder({100: (6, 8), 200: (7, 8)})
        self.assertEqual(ladder['ladder'], [200, 100])
        self.assertEqual(ladder['top'], 200)
        self.assertFalse(ladder['top_is_full'])

    def test_all_below_half_returns_none(self):
        self.assertIsNone(build_ladder({100: (3, 8), 200: (3, 8)}))

    def test_failed_rungs_are_skipped(self):
        # 唯一 8/8 在 100；300 的 6/8 在顶档之上，不入阶梯（顶档取最晚稳定点）。
        ladder = build_ladder({100: (8, 8), 200: (3, 8), 300: (6, 8)})
        self.assertEqual(ladder['ladder'], [100])


class AdaptiveScanTest(unittest.TestCase):
    def test_equidistant_indices(self):
        self.assertEqual(AdaptiveScan(grid12()).equidistant, [0, 4, 7, 11])

    def test_monotonic_step_finds_l50_by_bisection(self):
        grid = grid12()
        scanner = AdaptiveScan(grid)
        tested = []

        def rate(decision):
            tested.append(decision)
            index = grid.index(decision)
            return (8, 8) if index >= 5 else (0, 8)

        count = drive(scanner, rate)
        state = scanner.finish()
        self.assertEqual(state['l50'], grid[5])
        self.assertEqual(state['top'], grid[11])
        self.assertEqual(state['ladder']['l50'], grid[5])
        # 等距 4 点 + 二分 2 点 + 补档 3 点（idx5..11 里未测的），远少于 12 点全跑。
        self.assertLess(count, 12)
        self.assertTrue(all(grid.index(d) >= 5 for d in state['ladder']['ladder']))
        self.assertEqual(state['student_points'], [grid[5], grid[11]])

    def test_all_fail_reports_no_ladder(self):
        scanner = AdaptiveScan(grid12())
        count = drive(scanner, lambda d: (0, 8))
        state = scanner.finish()
        self.assertLessEqual(count, 9)
        self.assertIsNone(state['ladder'])
        self.assertIsNone(state['l50'])
        self.assertEqual(state['student_points'], [])

    def test_all_pass_starts_at_first_rung(self):
        grid = grid12()
        scanner = AdaptiveScan(grid)
        count = drive(scanner, lambda d: (8, 8))
        state = scanner.finish()
        self.assertLessEqual(count, 9)
        self.assertEqual(state['l50'], grid[0])
        self.assertEqual(state['top'], grid[11])

    def test_nonmonotonic_island_from_real_l46_curve(self):
        grid = grid12()
        wins = [0, 0, 0, 0, 0, 0, 6, 3, 5, 8, 8, 8]
        scanner = AdaptiveScan(grid)
        count = drive(scanner, lambda d: (wins[grid.index(d)], 8))
        self.assertEqual(scanner.finish()['l50'], grid[6])
        self.assertLess((count + len(scanner.finish()['student_points'])) * 8, 96)

    def test_single_unmeasured_point_from_real_human_l49_curve(self):
        # d943/d1213 both fail, but their only unmeasured point d1078 passes.
        # Ignoring gaps of two grid intervals returned d1347, two grid positions late.
        grid = [1, 136, 270, 405, 540, 674, 809, 943, 1078, 1213, 1347, 1482]
        wins = [0, 0, 0, 0, 0, 1, 2, 3, 4, 3, 4, 8]
        scanner = AdaptiveScan(grid)
        count = drive(scanner, lambda d: (wins[grid.index(d)], 8))
        state = scanner.finish()
        self.assertEqual(state['l50'], 1078)
        self.assertEqual(state['top'], 1482)
        self.assertEqual(list(scanner.wins)[:4], [1, 540, 943, 1482])
        self.assertLessEqual(count, 9)
        self.assertLess((count + len(state['student_points'])) * 8, 96)

    def test_budget_is_below_half_of_full_scan_for_all_binary_curves(self):
        grid = grid12()
        for bits in range(1 << len(grid)):
            scanner = AdaptiveScan(grid)
            count = drive(scanner, lambda d: (8 if bits & (1 << grid.index(d)) else 0, 8))
            self.assertLess((count + len(scanner.finish()['student_points'])) * 8, 96)

    def test_partial_or_inconsistent_results_cannot_register(self):
        with self.assertRaises(ValueError):
            scan.validate_result({'complete': False})
        data = dict(wins=2, games=2, rows=[dict(future_seed=10, won=True), dict(future_seed=11, won=False)])
        with self.assertRaises(ValueError):
            scan.validate_result(dict(future_seeds=[10, 11], points={'1': {'teacher': data}}))

    def test_non_monotonic_l48_shape(self):
        grid = grid12()
        scanner = AdaptiveScan(grid)

        def rate(decision):
            index = grid.index(decision)
            if index < 4:
                return (0, 8)
            if index < 8:
                return (6, 8)
            if index < 11:
                return (3, 8)   # L48 型：中段跌回
            return (8, 8)

        drive(scanner, rate)
        state = scanner.finish()
        self.assertEqual(state['l50'], grid[4])
        self.assertEqual(state['top'], grid[11])
        # 中段跌回的档（3/8）不入阶梯。
        ladder = state['ladder']['ladder']
        self.assertNotIn(grid[8], ladder)
        self.assertIn(grid[4], ladder)


class BacktestEvidenceTest(unittest.TestCase):
    def fixture(self):
        grid, futures = grid12(), [10, 11]
        def summary():
            return dict(wins=2, games=2, rows=[dict(future_seed=f, won=True, final_wave=30) for f in futures])
        points = {str(d): {'teacher': summary()} for d in grid}
        order = [grid[i] for i in (0, 4, 7, 11)]
        measured = {str(d): {'teacher': summary()} for d in order}
        for d in (grid[0], grid[-1]):
            measured[str(d)]['student'] = summary()
        trace = dict(complete=True, algorithm='coarse4_guard_budget9_v1', grid=grid,
                     future_seeds=futures, teacher_decisions=order, points=measured,
                     l50=grid[0], scan_games=12)
        entry = dict(task_id='t', seed=42, decisions=grid, future_seeds=futures)
        return entry, dict(complete=True, points=points, future_seeds=futures, adaptive_trace=trace)

    def test_full_grid_and_independent_trace_are_verified(self):
        entry, result = self.fixture()
        comparison = compare_document(entry, result)
        self.assertTrue(comparison['passed'])
        self.assertEqual(comparison['delta'], 0)

    def test_partial_full_grid_is_not_accepted_as_a_baseline(self):
        entry, result = self.fixture()
        del result['points'][str(entry['decisions'][1])]
        with self.assertRaisesRegex(ValueError, '全量网格缺少'):
            compare_document(entry, result)

    def test_different_future_block_is_not_a_same_block_comparison(self):
        entry, result = self.fixture()
        result['adaptive_trace']['future_seeds'] = [20, 21]
        with self.assertRaisesRegex(ValueError, '不是同一块'):
            compare_document(entry, result)


class RegisterHelpersTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)

    def test_update_manifest_writes_and_is_idempotent(self):
        path = self.tmp / 'manifest.json'
        path.write_text(json.dumps({'tasks': [{'task_id': 't1', 'seeds': [1]},
                                              {'task_id': 't2', 'seeds': [2]}]}), encoding='utf-8')
        spec = dict(archive='a', decisions=[300, 200], index=0,
                    future_seed_start=86000000, future_seed_count=256)
        self.assertTrue(update_manifest(path, 't1', spec))
        self.assertFalse(update_manifest(path, 't1', spec))
        document = json.loads(path.read_text(encoding='utf-8'))
        self.assertEqual(document['tasks'][0]['demo_start'], spec)
        self.assertNotIn('demo_start', document['tasks'][1])

    def test_update_manifest_rejects_unknown_task(self):
        path = self.tmp / 'manifest.json'
        path.write_text(json.dumps({'tasks': []}), encoding='utf-8')
        with self.assertRaises(KeyError):
            update_manifest(path, 'nope', dict(decisions=[1]))

    def test_update_manifest_does_not_replace_an_existing_archive(self):
        path = self.tmp / 'manifest.json'
        old = dict(archive='old', decisions=[3, 2])
        path.write_text(json.dumps({'tasks': [{'task_id': 't1', 'demo_start': old}]}))
        before = path.read_bytes()
        with self.assertRaises(ValueError):
            update_manifest(path, 't1', dict(archive='other', decisions=[3, 2]))
        self.assertEqual(path.read_bytes(), before)

    def test_update_test_expectations_inserts_missing_ids(self):
        path = self.tmp / 'test_x.py'
        path.write_text("self.assertEqual(set(cells), {\n                                      'a',\n                                      'b',\n                                      })\n", encoding='utf-8')
        self.assertTrue(update_test_expectations(path, ['c']))
        self.assertIn("'c',", path.read_text(encoding='utf-8'))
        self.assertFalse(update_test_expectations(path, ['c']))

    def test_expectation_id_elsewhere_in_file_does_not_count(self):
        path = self.tmp / 'test_x.py'
        path.write_text("other = 'c'\nself.assertEqual(set(cells), {'a',\n})\n")
        self.assertTrue(update_test_expectations(path, ['c']))
        self.assertEqual(path.read_text().count("'c'"), 2)

    def test_actual_curriculum_template_updates_only_the_registered_manifest_test(self):
        path = self.tmp / 'test_curriculum_queue.py'
        original = (ROOT / 'python/test_curriculum_queue.py').read_text()
        path.write_text(original)
        boundary = original.index('    def test_registered_manifest_start_cells(')
        self.assertTrue(update_test_expectations(path, ['task91_new_cell']))
        changed = path.read_text()
        self.assertEqual(changed[:boundary], original[:boundary])
        self.assertIn("self.assertEqual(set(cells), {self.BASE})", changed)
        self.assertFalse(update_test_expectations(path, ['task91_new_cell']))


class SharedSeedLedgerTest(unittest.TestCase):
    def test_import_and_register_share_the_same_default_ledger(self):
        args = scan.parser().parse_args(['register', '/results'])
        self.assertEqual(args.ledger, importer.DEFAULT_SEED_LEDGER)
        self.assertEqual(args.scan_ledger, importer.DEFAULT_SEED_LEDGER)

    def test_scan_and_registration_blocks_do_not_overlap_and_retry_reuses_block(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / 'seeds.json'
            def row(name):
                return dict(task_id=name, base_task_id=name, seed=1, archive=name,
                            level=46, frames=100, grade='half')
            with patch.object(importer, 'occupied_seed_ranges', return_value=[]), \
                 patch.object(scan, 'occupied_seed_ranges', return_value=[]), \
                 patch.object(scan, 'LEGACY_REGISTER_LEDGER', Path(directory) / 'legacy.json'):
                importer.scan_plan([row('first')], {}, 84_000_000, ledger_path=ledger)
                registry = scan.RegisterLedger(ledger, ledger)
                first = registry.allocate(84_000_000, key='register-a')
                self.assertEqual(registry.allocate(84_000_000, key='register-a'), first)
                importer.scan_plan([row('second')], {}, 84_000_000, ledger_path=ledger)
            claims = list(json.loads(ledger.read_text())['reservations'].values())
            self.assertEqual(len(claims), 3)
            self.assertEqual(sorted(x['count'] for x in claims), [96, 96, 256])
            for i, a in enumerate(claims):
                for b in claims[i + 1:]:
                    self.assertTrue(a['start'] + a['count'] <= b['start'] or
                                    b['start'] + b['count'] <= a['start'])


class RegisterNewLlmFamilyTest(unittest.TestCase):
    def fixture(self, directory, won=True):
        root = Path(directory)
        base = dict(task_id='base', level=49, seeds=[42], deck=[1, 33, 39, 44, 15, 3],
                    zombie_count_multiplier=2.0, preplanted=[{'type': 33, 'row': 0, 'col': 0}],
                    demo_start=dict(archive='human', decisions=[9], future_seed_start=100,
                                    future_seed_count=256, index=0))
        manifest, design, tests, ledger = [root / name for name in
                                          ('tasks.json', 'design.md', 'test_cells.py', 'ledger.json')]
        manifest.write_text(json.dumps(dict(tasks=[base])))
        design.write_text('### 14.34 Existing tasks\n')
        tests.write_text("self.assertEqual(set(cells), {'base'})\n")
        result = root / 'scan.json'
        rows = [dict(future_seed=future, won=won) for future in (10, 11)]
        result.write_text(json.dumps(dict(schema=scan.RESULT_SCHEMA, task_id='base', seed=42,
                                         archive='llm', complete=True, future_seeds=[10, 11],
                                         points={'8': {'teacher': dict(wins=2 if won else 0,
                                                                       games=2, rows=rows)}})))
        args = scan.parser().parse_args(['register', str(result), '--manifest', str(manifest),
                                        '--design', str(design), '--tests', str(tests),
                                        '--ledger', str(ledger), '--scan-ledger', str(ledger),
                                        '--out', str(root / 'out')])
        return args, base

    def test_missing_llm_family_dry_run_registration_and_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            args, base = self.fixture(directory)
            files = [args.manifest, args.design, args.tests]
            before = {path: path.read_bytes() for path in files}
            with patch.object(scan, 'occupied_seed_ranges', return_value=[]), \
                    patch.object(scan, 'LEGACY_REGISTER_LEDGER', Path(directory) / 'legacy.json'), \
                    patch('builtins.print'):
                args.dry_run = True
                scan.cmd_register(args)
                self.assertEqual({path: path.read_bytes() for path in files}, before)
                self.assertFalse(args.ledger.exists())
                args.dry_run = False
                scan.cmd_register(args)
                tasks = json.loads(args.manifest.read_text())['tasks']
                self.assertEqual(tasks[0], base)
                new = tasks[1]
                self.assertEqual(new['task_id'], 'base_llm')
                for key in ('level', 'seeds', 'deck', 'preplanted', 'zombie_count_multiplier'):
                    self.assertEqual(new[key], base[key])
                self.assertEqual(new['demo_start']['archive'], 'llm')
                self.assertEqual(new['demo_start']['decisions'], [8])
                self.assertIn('`base_llm`', args.design.read_text())
                self.assertIn("'base_llm'", args.tests.read_text())
                report = json.loads((args.out / 'register_report.json').read_text())
                self.assertEqual(report['registered'], 1)
                self.assertTrue(report['entries'][0]['created_family'])
                files.append(args.ledger)
                registered = {path: path.read_bytes() for path in files}
                scan.cmd_register(args)
                self.assertEqual({path: path.read_bytes() for path in files}, registered)

    def test_failing_archive_does_not_create_a_family_or_reserve_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _ = self.fixture(directory, won=False)
            files = [args.manifest, args.design, args.tests]
            before = {path: path.read_bytes() for path in files}
            with patch.object(scan, 'occupied_seed_ranges', return_value=[]), patch('builtins.print'):
                scan.cmd_register(args)
            self.assertEqual({path: path.read_bytes() for path in files}, before)
            self.assertFalse(args.ledger.exists())
            self.assertEqual(json.loads((args.out / 'register_report.json').read_text())['registered'], 0)


class ResolveTargetTaskTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.path = self.tmp / 'manifest.json'
        self.path.write_text(json.dumps({'tasks': [
            {'task_id': 'base', 'demo_start': {'archive': 'artifacts/human_demo/raw/x'}},
            {'task_id': 'base_llm', 'demo_start': {'archive': 'artifacts/llm_demo/raw/x'}},
            {'task_id': 'fresh', 'seeds': [1]},
        ]}), encoding='utf-8')

    def test_archive_match_wins(self):
        task_id, existing = resolve_target_task(
            dict(task_id='base', archive='artifacts/llm_demo/raw/x'), self.path)
        self.assertEqual(task_id, 'base_llm')
        self.assertEqual(existing['archive'], 'artifacts/llm_demo/raw/x')

    def test_base_without_demo_start_is_used(self):
        task_id, existing = resolve_target_task(
            dict(task_id='fresh', archive='artifacts/llm_demo/raw/y'), self.path)
        self.assertEqual(task_id, 'fresh')
        self.assertIsNone(existing)

    def test_conflict_raises(self):
        with self.assertRaises(ValueError):
            resolve_target_task(
                dict(task_id='base', archive='artifacts/llm_demo/raw/z'), self.path)

    def test_base_occupied_falls_back_to_llm(self):
        # task87 先例：base 族已被人类示范占用时，LLM 存档落到 base_llm。
        self.path.write_text(json.dumps({'tasks': [
            {'task_id': 'base', 'demo_start': {'archive': 'artifacts/human_demo/raw/x'}},
            {'task_id': 'base_llm', 'seeds': [1]},
        ]}), encoding='utf-8')
        task_id, existing = resolve_target_task(
            dict(task_id='base', archive='artifacts/llm_demo/raw/x'), self.path)
        self.assertEqual(task_id, 'base_llm')
        self.assertIsNone(existing)

    def test_missing_llm_family_is_proposed_from_an_existing_base(self):
        self.path.write_text(json.dumps({'tasks': [
            {'task_id': 'base', 'demo_start': {'archive': 'human'}}]}))
        self.assertEqual(resolve_target_task(dict(task_id='base', archive='llm'), self.path),
                         ('base_llm', None))

    def test_unknown_task_raises(self):
        with self.assertRaises(KeyError):
            resolve_target_task(
                dict(task_id='nope', archive='artifacts/llm_demo/raw/w'), self.path)


if __name__ == '__main__':
    unittest.main()
