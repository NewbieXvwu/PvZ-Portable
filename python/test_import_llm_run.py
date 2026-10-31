"""task89 存档一键入库的纯逻辑测试：不需要引擎。"""
import json
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


def _meta(task_id='demo_roof_46_doom_m200', seed=20261046, operations=6,
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
        verified=False,
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
                                         end_tick=index + 1, ok=True)) + '\n')
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
    def test_verification_replays_the_copied_archive(self):
        verify = Mock(return_value={'verified': True})
        with patch.dict(sys.modules, human_demo=SimpleNamespace(verify=verify)):
            importer.verify_copy(Path('/archive'), Path('/resources'), Path('/engine'))
        verify.assert_called_once_with(Path('/archive'), Path('/resources'), Path('/engine'))

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








if __name__ == '__main__':
    unittest.main()
