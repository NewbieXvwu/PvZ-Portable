"""SHA import must preserve variants, reuse exact receipts and reject corrupt storage."""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from llm_import_sha import archive_sha, run_sha
from python.test_import_llm_run import _meta, _write_archive
from deploy_task105 import merge_tasks


class ShaImportTest(unittest.TestCase):
    def test_interrupted_copy_leaves_no_partial_sha_target_and_can_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            archive = _write_archive(source, 'a', _meta(wins=5))
            manifest = root / 'configs/tasks.json'
            manifest.parent.mkdir()
            manifest.write_text(json.dumps({'tasks': []}))
            layers = root / 'layers.json'
            layers.write_text(json.dumps({'layers': {}}))
            args = (source, root / 'dest', manifest, layers, root, root / 'engine',
                    84_000_000, root / 'ledger.json')
            copyfile = shutil.copyfile
            calls = []

            def interrupted(source_path, destination):
                calls.append(source_path)
                if len(calls) == 2:
                    raise OSError('interrupted archive copy')
                return copyfile(source_path, destination)

            with patch('llm_import_sha.verification_context', return_value={'engine': 'one'}), \
                 patch('llm_import_sha.legacy.verify_copy', return_value={'verified': True}) as verify:
                with patch('llm_import_sha.shutil.copyfile', side_effect=interrupted):
                    with self.assertRaisesRegex(OSError, 'interrupted archive copy'):
                        run_sha(*args)
                self.assertFalse((root / 'dest/sha256' / archive_sha(archive)).exists())
                self.assertEqual(list((root / 'dest/sha256').iterdir()), [])
                self.assertEqual(verify.call_count, 0)
                result = run_sha(*args)
                self.assertEqual(verify.call_count, 1)
                self.assertEqual(result['rows'][0]['verify'], '通过')

    def test_deployment_merge_preserves_current_tasks_and_rejects_conflicts(self):
        current = {'tasks': [{'task_id': 'live', 'demo_start': {'index': 3}}], 'new_field': True}
        addition = {'task_id': 'new', 'demo_start': {'index': 0}}
        merged = merge_tasks(current, [addition])
        self.assertEqual(merged['tasks'][0], current['tasks'][0])
        self.assertEqual(len(current['tasks']), 1)
        self.assertTrue(merged['new_field'])
        self.assertEqual(merge_tasks(merged, [addition]), merged)
        with self.assertRaisesRegex(ValueError, 'conflict'):
            merge_tasks(merged, [{'task_id': 'new', 'demo_start': {'index': 4}}])

    def test_verifier_fields_are_ignored_but_robustness_and_tape_are_not(self):
        with tempfile.TemporaryDirectory() as directory:
            a = _write_archive(Path(directory), 'a', _meta())
            sha = archive_sha(a)
            meta = json.loads((a / 'meta.json').read_text())
            meta.update(verified=True, hash_verification='tick/ok/outcome')
            (a / 'meta.json').write_text(json.dumps(meta))
            self.assertEqual(archive_sha(a), sha)
            meta['robustness'] = {'k': 8}
            (a / 'meta.json').write_text(json.dumps(meta))
            self.assertNotEqual(archive_sha(a), sha)
            sha = archive_sha(a)
            with (a / 'ticks.jsonl').open('a') as stream:
                stream.write('{}\n')
            self.assertNotEqual(archive_sha(a), sha)

    def test_incremental_variants_failures_engine_change_and_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            _write_archive(source, 'a', _meta(wins=5))
            _write_archive(source, 'b', _meta(wins=6))
            manifest = root / 'configs/tasks.json'
            manifest.parent.mkdir()
            manifest.write_text(json.dumps({'tasks': []}))
            layers = root / 'layers.json'
            layers.write_text(json.dumps({'layers': {}}))
            args = (source, root / 'dest', manifest, layers, root, root / 'engine',
                    84_000_000, root / 'ledger.json')
            with patch('llm_import_sha.verification_context', return_value={'engine': 'one'}), \
                 patch('llm_import_sha.legacy.verify_copy', return_value={'verified': True}) as verify:
                first = run_sha(*args)
                before = manifest.read_bytes()
                self.assertEqual(verify.call_count, 2)
                self.assertEqual(len(first['registered']), 2)
                self.assertEqual(len({r['archive'] for r in first['rows']}), 2)
                second = run_sha(*args)
                self.assertEqual(verify.call_count, 2)
                self.assertFalse(second['registered'])
                self.assertEqual(manifest.read_bytes(), before)
                self.assertEqual([a['future_seed_start'] for a in first['plan']['archives']],
                                 [a['future_seed_start'] for a in second['plan']['archives']])
                _write_archive(source, 'c', _meta(wins=7))
                verify.side_effect = RuntimeError('timeline mismatch')
                third = run_sha(*args)
                self.assertEqual(verify.call_count, 3)
                self.assertEqual(third['rows'][2]['grade'], 'loss')
                self.assertIn('timeline mismatch', third['rows'][2]['note'])
                run_sha(*args)
                self.assertEqual(verify.call_count, 3)  # failed receipts also resume
            with patch('llm_import_sha.verification_context', return_value={'engine': 'two'}), \
                 patch('llm_import_sha.legacy.verify_copy', return_value={'verified': True}) as verify:
                run_sha(*args)
                self.assertEqual(verify.call_count, 3)
            target = root / 'dest/sha256' / archive_sha(source / 'a') / 'ticks.jsonl'
            target.write_text('corrupt')
            with patch('llm_import_sha.verification_context', return_value={'engine': 'two'}):
                with self.assertRaisesRegex(ValueError, 'SHA不符'):
                    run_sha(*args)


if __name__ == '__main__':
    unittest.main()
