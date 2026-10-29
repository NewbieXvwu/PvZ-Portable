"""task89 存档一键入库的纯逻辑测试：不需要引擎。"""
import json
import contextlib
import io
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
from unittest.mock import Mock
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'python'))

import import_llm_run as importer  # noqa: E402


def _meta(task_id='demo_roof_46_doom_m200', seed=20261046, operations=120,
          won=True, wins=None, k=8, level=46, deck=(1, 33, 39, 44, 15, 3, 2, 17, 4, 35),
          multiplier=2.0, manual_patch=None):
    meta = dict(
        schema='episode_archive_v1', label_source='llm_gpt_pro',
        task=dict(id=task_id, level=level, seed=seed, deck=list(deck), policy='llm',
                  playthrough=2, zombie_count_multiplier=multiplier, wave_cap=None,
                  preplanted=[]),
        environment_task=dict(level=level, seed=seed, playthrough=2,
                              profile=dict(seed_slot_count=10, imitater_owned=False,
                                           first_aid_owned=False, pool_cleaner_owned=False,
                                           roof_cleaner_owned=False, rake_charges=0,
                                           owned_upgrade_plants=[44]),
                              forced_seeds=[], loadout_mode='fixed',
                              zombie_count_multiplier=multiplier, wave_cap=None, preplanted=[]),
        outcome=dict(result=1 if won else 0, won=won, final_wave=30, wave_count=30,
                     final_tick=operations, actions=operations),
        operations=operations, frame_count=operations + 1,
        initial_state_hash='deadbeef', verified=False,
    )
    if wins is not None:
        meta['robustness'] = dict(k=k, passed=wins >= 5, checkpoints={
            '10': dict(wins=wins, k=k, reached=True)})
    if manual_patch is not None:
        meta['manual_patch'] = manual_patch
    return meta


def _write_archive(root, name, meta):
    archive = root / name
    archive.mkdir(parents=True)
    (archive / 'meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                       encoding='utf-8')
    with (archive / 'ticks.jsonl').open('w', encoding='utf-8') as stream:
        for index in range(int(meta['operations'])):
            stream.write(json.dumps(dict(action=dict(type='wait', ticks=1), tick=index,
                                         end_tick=index + 1, ok=True,
                                         state_hash='x')) + '\n')
    with (archive / 'frames.jsonl').open('w', encoding='utf-8') as stream:
        for _ in range(int(meta['frame_count'])):
            stream.write('{}\n')
    return archive


class SafeExtractTest(unittest.TestCase):
    def test_rejects_parent_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / 'bad.zip'
            with zipfile.ZipFile(bundle, 'w') as zip_file:
                zip_file.writestr('../escape.json', '{}')
            with self.assertRaises(ValueError):
                importer.safe_extract(bundle, Path(directory) / 'out')
            self.assertFalse((Path(directory) / 'escape.json').exists())

    def test_rejects_absolute_path(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / 'bad.zip'
            with zipfile.ZipFile(bundle, 'w') as zip_file:
                zip_file.writestr('/tmp/absolute.json', '{}')
            with self.assertRaises(ValueError):
                importer.safe_extract(bundle, Path(directory) / 'out')

    def test_rejects_windows_parent_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / 'bad.zip'
            with zipfile.ZipFile(bundle, 'w') as zip_file:
                zip_file.writestr('a\\..\\escape.json', '{}')
            with self.assertRaises(ValueError):
                importer.safe_extract(bundle, Path(directory) / 'out')

    def test_extracts_normal_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / 'ok.zip'
            with zipfile.ZipFile(bundle, 'w') as zip_file:
                zip_file.writestr('out/a/meta.json', '{}')
            destination = Path(directory) / 'out'
            importer.safe_extract(bundle, destination)
            self.assertTrue((destination / 'out/a/meta.json').is_file())


class MetaTest(unittest.TestCase):
    def test_verification_explicitly_uses_timeline_without_state_hashes(self):
        verify = Mock(return_value={'verified': True})
        with patch.dict(sys.modules, human_demo=SimpleNamespace(verify=verify)):
            importer.verify_copy(Path('/archive'), Path('/resources'), Path('/engine'))
        verify.assert_called_once_with(Path('/archive'), Path('/resources'), Path('/engine'), strict_hashes=False)

    def test_conveyor_empty_deck_is_valid_but_fixed_empty_deck_is_not(self):
        with tempfile.TemporaryDirectory() as directory:
            meta = _meta(task_id='demo_conv_30_orig_m100', level=30, deck=(), multiplier=1.0)
            meta['task']['conveyor_start'] = 'original'
            archive = _write_archive(Path(directory), 'a', meta)
            self.assertEqual(importer.archive_problems(meta, archive), [])
            del meta['task']['conveyor_start']
            self.assertTrue(importer.archive_problems(meta, archive))

    def test_rejects_unsafe_id_and_inconsistent_outcome_or_multiplier(self):
        with tempfile.TemporaryDirectory() as directory:
            meta = _meta(task_id='../../outside')
            meta['outcome']['won'] = False
            meta['environment_task']['zombie_count_multiplier'] = 3.0
            archive = _write_archive(Path(directory), 'a', meta)
            problems = importer.archive_problems(meta, archive)
            self.assertEqual(len(problems), 3, problems)

    def test_detects_level_mismatch_between_id_and_meta(self):
        meta = _meta(task_id='demo_roof_46_doom_m200', level=48)
        with tempfile.TemporaryDirectory() as directory:
            archive = _write_archive(Path(directory), 'a', meta)
            problems = importer.archive_problems(meta, archive)
        self.assertTrue(any('编码的是 L46' in problem for problem in problems), problems)

    def test_detects_operations_mismatch(self):
        meta = _meta(operations=120)
        with tempfile.TemporaryDirectory() as directory:
            archive = _write_archive(Path(directory), 'a', meta)
            with (archive / 'ticks.jsonl').open('a', encoding='utf-8') as stream:
                stream.write('{}\n')
            problems = importer.archive_problems(meta, archive)
        self.assertTrue(any('operations=120' in problem for problem in problems), problems)

    def test_detects_task_environment_seed_mismatch(self):
        meta = _meta(seed=20261046)
        meta['environment_task']['seed'] = 20261047
        with tempfile.TemporaryDirectory() as directory:
            archive = _write_archive(Path(directory), 'a', meta)
            problems = importer.archive_problems(meta, archive)
        self.assertTrue(any('task.seed' in problem for problem in problems), problems)


class GradeTest(unittest.TestCase):
    def test_full_half_loss(self):
        self.assertEqual(importer.grade(_meta(won=True, wins=5)), 'full')
        self.assertEqual(importer.grade(_meta(won=True, wins=4)), 'half')
        self.assertEqual(importer.grade(_meta(won=True, wins=None)), 'half')
        self.assertEqual(importer.grade(_meta(won=False, wins=8)), 'loss')
        self.assertEqual(importer.grade(_meta(won=True, wins=5, k=5)), 'half')
        self.assertEqual(importer.grade(_meta(won=True, wins=9, k=8)), 'half')


class FamilyTest(unittest.TestCase):
    def setUp(self):
        self.tasks = {'demo_roof_46_doom_m200': dict(
            task_id='demo_roof_46_doom_m200', level=46, seeds=[20261046],
            zombie_count_multiplier=2.0, deck=[1, 33, 39, 44, 15, 3, 2, 17, 4, 35])}

    def test_same_seed_keeps_family(self):
        task_id, new, problems = importer.resolve_family(_meta(seed=20261046), self.tasks)
        self.assertEqual(task_id, 'demo_roof_46_doom_m200')
        self.assertFalse(new)
        self.assertEqual(problems, [])

    def test_foreign_seed_gets_suffix(self):
        task_id, new, problems = importer.resolve_family(_meta(seed=20261047), self.tasks)
        self.assertEqual(task_id, 'demo_roof_46_doom_m200_s20261047')
        self.assertTrue(new)
        self.assertEqual(problems, [])

    def test_foreign_seed_family_already_registered_is_not_new(self):
        tasks = dict(self.tasks, **{'demo_roof_46_doom_m200_s20261047': dict(
            task_id='demo_roof_46_doom_m200_s20261047', level=46, seeds=[20261047],
            zombie_count_multiplier=2.0, deck=[1, 33, 39, 44, 15, 3, 2, 17, 4, 35])})
        task_id, new, problems = importer.resolve_family(_meta(seed=20261047), tasks)
        self.assertEqual(task_id, 'demo_roof_46_doom_m200_s20261047')
        self.assertFalse(new)
        self.assertEqual(problems, [])

    def test_unknown_base_registers_itself(self):
        task_id, new, problems = importer.resolve_family(_meta(task_id='demo_roof_45_doom_m200',
                                                              level=45), self.tasks)
        self.assertEqual(task_id, 'demo_roof_45_doom_m200')
        self.assertTrue(new)
        self.assertEqual(problems, [])

    def test_deck_mismatch_is_refused(self):
        meta = _meta(seed=20261047, deck=(0, 1, 2, 3, 4, 5))
        _, _, problems = importer.resolve_family(meta, self.tasks)
        self.assertTrue(problems)

    def test_registered_family_cannot_change_multiplier(self):
        _, _, problems = importer.resolve_family(_meta(multiplier=3.0), self.tasks)
        self.assertTrue(problems)


class ScanTest(unittest.TestCase):
    def row(self, name):
        return dict(grade='half', archive=name, task_id=name, base_task_id=name,
                    seed=123, frames=100, level=46)

    def test_checks_each_strided_block(self):
        occupied = [(84_001_000, 84_001_095)]
        with patch.object(importer, 'occupied_seed_ranges', return_value=occupied):
            plan = importer.scan_plan([self.row('a'), self.row('b')], {}, 84_000_000)
        starts = [entry['future_seed_start'] for entry in plan['archives']]
        self.assertEqual(starts, [84_000_000, 85_001_000])

    def test_ledger_prevents_reuse_across_imports_and_preserves_reruns(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / 'scan_seed_ledger.json'
            with patch.object(importer, 'occupied_seed_ranges', return_value=[]):
                first = importer.scan_plan([self.row('a')], {}, 84_000_000, ledger_path=ledger)
                second = importer.scan_plan([self.row('b')], {}, 84_000_000, ledger_path=ledger)
                repeat = importer.scan_plan([self.row('a')], {}, 84_000_000, ledger_path=ledger)
            self.assertEqual(first['archives'][0]['future_seed_start'], 84_000_000)
            self.assertEqual(second['archives'][0]['future_seed_start'], 85_000_000)
            self.assertEqual(repeat['archives'][0]['future_seed_start'], 84_000_000)
            self.assertTrue(repeat['archives'][0]['reused_reservation'])

    def test_nested_seed_inventory_and_uint32_limit(self):
        seeds = set()
        importer._config_seed_points(dict(tasks={'a': [100, 101]}, seeds=[dict(seeds=[200])],
                                         future_seed_start=300, future_seed_count=4), seeds)
        self.assertTrue({100, 101, 200, 300, 301, 302, 303} <= seeds)
        with self.assertRaises(ValueError):
            importer.allocate_future_block([], 2**32 - 1, 96)
        ranges = importer.occupied_seed_ranges(ROOT)
        self.assertTrue(any(a <= 100_000_000 < b for a, b in ranges))

    def test_points_cover_tail_excluded(self):
        points = importer.scan_decisions(1000)
        self.assertEqual(len(points), 12)
        self.assertEqual(points[0], 1)
        self.assertEqual(points[-1], 940)
        self.assertEqual(points, sorted(set(points)))

    def test_renewal_pool_is_never_allocated(self):
        # task114 续领池 [200000000,201000000)：生产 qc 从这里切起点格未来块，
        # 导入侧落进去的格子在部署时会被拒（2026-10-29 实测）。
        low, high = importer.RESERVED_SEED_POOLS[0]
        ranges = importer.occupied_seed_ranges(ROOT)
        self.assertTrue(any(start <= low and high - 1 <= stop for start, stop in ranges))
        start = importer.allocate_future_block(ranges, low, 256)
        self.assertFalse(low <= start < high)
        self.assertGreaterEqual(start, high)
        # 池内整段都被当作占用：从池首起分配必须跳到池尾或更后。
        self.assertNotEqual(importer.allocate_future_block([(low, high - 1)], low, 256), low)

    def test_future_block_avoids_occupied_ranges(self):
        ranges = [(83_000_000, 83_000_307), (84_000_000, 84_000_095)]
        self.assertEqual(importer.allocate_future_block(ranges, 84_000_000, 96), 85_000_000)

    def test_seed_audit_finds_registered_blocks(self):
        ranges = importer.occupied_seed_ranges(ROOT)
        self.assertTrue(any(start <= 81_000_000 <= stop for start, stop in ranges))
        self.assertTrue(any(start <= 83_000_000 <= stop for start, stop in ranges))
        self.assertFalse(any(start <= 84_000_000 <= stop for start, stop in ranges))


class RegisterTest(unittest.TestCase):
    def test_multiple_new_families_are_valid_json_and_registered_in_design(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'tasks.json'
            path.write_text('{\n  "tasks": [\n    {"task_id":"old"}\n  ]\n}\n')
            entries = [dict(task_id=name, level=46, seeds=[seed],
                            zombie_count_multiplier=2.0, deck=[1, 33, 39, 44, 15, 3])
                       for name, seed in [('new_a', 123), ('new_b', 456)]]
            importer.register_tasks(path, entries)
            self.assertEqual(set(importer.load_manifest(path)), {'old', 'new_a', 'new_b'})
            design = root / 'DESIGN.md'
            importer.register_design(design, entries)
            original = design.read_text()
            importer.register_design(design, entries)
            self.assertEqual(design.read_text(), original)
            self.assertIn('`new_a`', original)
            self.assertIn('`new_b`', original)

    def test_append_preserves_existing_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'tasks.json'
            original = ('{\n  "tasks": [\n'
                        '    {"task_id":"a","level":46,"seeds":[1],"zombie_count_multiplier":2.0,"deck":[1]}\n'
                        '  ]\n}\n')
            path.write_text(original, encoding='utf-8')
            importer.register_tasks(path, [dict(task_id='b', level=46, seeds=[2],
                                                zombie_count_multiplier=2.0, deck=[1])])
            text = path.read_text(encoding='utf-8')
            self.assertIn('"task_id":"a"', text)
            self.assertIn('"task_id":"b"', text)
            self.assertTrue(text.endswith('  ]\n}\n'))
            self.assertEqual(len(importer.load_manifest(path)), 2)


class EndToEndTest(unittest.TestCase):
    def test_minimal_command_saves_scan_plan_and_reports_all_layers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            _write_archive(source, 'archive', _meta(wins=6))
            manifest = root / 'tasks.json'
            manifest.write_text('{"tasks":[]}')
            args = SimpleNamespace(input=source, dest=root / 'dest', manifest=manifest,
                                   layers=importer.DEFAULT_LAYERS, resource_dir=importer.DEFAULT_RESOURCES,
                                   executable=None, no_verify=True, future_base=84_000_000,
                                   report=None, scan_out=None, seed_ledger=root / 'seeds.json')
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(importer.main_with(args), 0)
            plans = list(args.dest.glob('scan_jobs_*.json'))
            self.assertEqual(len(plans), 1)
            plan = json.loads(plans[0].read_text())
            self.assertEqual(len(plan['archives'][0]['jobs']), 24)
            self.assertIn('192 条续走', output.getvalue())
            for layer in 'ABCDE':
                self.assertIn(f'| {layer} |', output.getvalue())
    def test_same_actions_with_changed_robustness_are_not_silently_deduplicated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = _write_archive(root, 'old', _meta(wins=4))
            new = _write_archive(root, 'new', _meta(wins=8))
            target, _ = importer.import_archive(old, root / 'dest', 'safe_id', 123)
            self.assertEqual(importer.import_archive(new, root / 'dest', 'safe_id', 123)[1], 'conflict')
            meta = json.loads((target / 'meta.json').read_text())
            meta['verified'] = True
            meta['hash_verification'] = 'tick/ok/outcome only'
            (target / 'meta.json').write_text(json.dumps(meta))
            self.assertEqual(importer.import_archive(old, root / 'dest', 'safe_id', 123)[1], 'duplicate')
    def test_input_directory_cannot_be_the_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            with self.assertRaisesRegex(ValueError, '输入只读'):
                importer.run(source, source / 'dest', source / 'tasks.json',
                             importer.DEFAULT_LAYERS, importer.DEFAULT_RESOURCES, None, False, 84_000_000)
    def test_imports_only_wins_and_registers_foreign_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'input'
            source.mkdir()
            _write_archive(source, 'win_same', _meta(seed=20261046, wins=6))
            _write_archive(source, 'win_foreign', _meta(seed=20261047, wins=2))
            _write_archive(source, 'loss', _meta(seed=20261048, won=False, operations=40))
            manifest = root / 'tasks.json'
            manifest.write_text(json.dumps({'tasks': [dict(
                task_id='demo_roof_46_doom_m200', level=46, seeds=[20261046],
                zombie_count_multiplier=2.0, deck=[1, 33, 39, 44, 15, 3, 2, 17, 4, 35])]},
                ensure_ascii=False), encoding='utf-8')
            dest = root / 'dest'
            result = importer.run(source, dest, manifest, importer.DEFAULT_LAYERS,
                                  importer.DEFAULT_RESOURCES, None, False, 84_000_000)
            grades = {row['seed']: row['grade'] for row in result['rows']}
            self.assertEqual(grades[20261046], 'full')
            self.assertEqual(grades[20261047], 'half')
            self.assertEqual(grades[20261048], 'loss')
            self.assertTrue((dest / 'seed_20261046/demo_roof_46_doom_m200/meta.json').is_file())
            self.assertTrue((dest / 'seed_20261047/demo_roof_46_doom_m200_s20261047/meta.json').is_file())
            self.assertFalse((dest / 'seed_20261048').exists())
            tasks = importer.load_manifest(manifest)
            self.assertIn('demo_roof_46_doom_m200_s20261047', tasks)
            self.assertEqual(tasks['demo_roof_46_doom_m200_s20261047']['seeds'], [20261047])
            self.assertEqual(tasks['demo_roof_46_doom_m200_s20261047']['deck'],
                             [1, 33, 39, 44, 15, 3, 2, 17, 4, 35])
            self.assertIn('`demo_roof_46_doom_m200_s20261047`',
                          (root / 'DESIGN.md').read_text())
            self.assertEqual(len(result['plan']['archives']), 2)
            self.assertEqual(sum(len(entry['jobs']) for entry in result['plan']['archives']), 48)
            starts = [entry['future_seed_start'] for entry in result['plan']['archives']]
            self.assertEqual(len(starts), len(set(starts)))

    def test_rerun_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'input'
            source.mkdir()
            _write_archive(source, 'win', _meta(seed=20261047, wins=6))
            manifest = root / 'tasks.json'
            manifest.write_text(json.dumps({'tasks': [dict(
                task_id='demo_roof_46_doom_m200', level=46, seeds=[20261046],
                zombie_count_multiplier=2.0, deck=[1, 33, 39, 44, 15, 3, 2, 17, 4, 35])]},
                ensure_ascii=False), encoding='utf-8')
            dest = root / 'dest'
            importer.run(source, dest, manifest, importer.DEFAULT_LAYERS,
                         importer.DEFAULT_RESOURCES, None, False, 84_000_000)
            importer.run(source, dest, manifest, importer.DEFAULT_LAYERS,
                         importer.DEFAULT_RESOURCES, None, False, 84_000_000)
            tasks = importer.load_manifest(manifest)
            self.assertEqual([task_id for task_id in tasks].count(
                'demo_roof_46_doom_m200_s20261047'), 1)


if __name__ == '__main__':
    unittest.main()
