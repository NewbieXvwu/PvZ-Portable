"""SHA import must preserve variants, reuse exact receipts and reject corrupt storage."""
import json
import shutil
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from import_llm_run import archive_sha, run as production_run_sha
from python.test_import_llm_run import _meta, _write_archive


def run_sha(*args, **kwargs):
    return production_run_sha(*args, executor_factory=ThreadPoolExecutor, **kwargs)


class ShaImportTest(unittest.TestCase):
    def test_cached_jsonl_receipt_survives_task32_inplace_npz_conversion(self):
        from pvz_frame_store import write_frames, read_frames
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = _write_archive(root / 'source', 'a', _meta(wins=5))
            manifest = root / 'configs/tasks.json'
            manifest.parent.mkdir()
            manifest.write_text(json.dumps({'tasks': []}))
            args = (root / 'source', root / 'dest', manifest, root, root / 'engine')
            with patch('import_llm_run.verification_context', return_value={}), \
                 patch('import_llm_run.verify_copy', return_value={'verified': True}):
                first = run_sha(*args)
            stored = root / 'dest/sha256' / archive_sha(archive)
            write_frames(stored / 'frames.npz', read_frames(archive / 'frames.jsonl'), cache=False)
            shutil.copyfile(archive / 'frames.jsonl', stored / 'frames.jsonl')
            with patch('import_llm_run.verify_copy', side_effect=AssertionError('no repeat replay')):
                # Also accept the converter's brief state between rename and JSONL unlink.
                transition = run_sha(*args)
                self.assertTrue(transition['rows'][0]['verification_reused'])
                self.assertFalse((stored / 'frames.jsonl').exists())
                second = run_sha(*args)
            self.assertTrue(second['rows'][0]['verification_reused'])
            self.assertEqual(first['rows'][0]['task_id'], second['rows'][0]['task_id'])
            self.assertEqual(second['rows'][0]['verify'], '通过')

    def test_single_verifier_can_share_the_three_game_budget_with_scanning(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(3):
                _write_archive(root / 'source', str(index), _meta(wins=5 + index))
            manifest = root / 'configs/tasks.json'
            manifest.parent.mkdir()
            manifest.write_text(json.dumps({'tasks': []}))
            with patch('import_llm_run.verification_context', return_value={}), \
                 patch('import_llm_run.verify_copy', return_value={'verified': True}) as verify:
                pools = []
                def executor(**kwargs):
                    pools.append(kwargs['max_workers'])
                    return ThreadPoolExecutor(**kwargs)
                production_run_sha(root / 'source', root / 'dest', manifest,
                                   root, root / 'engine',
                                   executor_factory=executor, verify_workers=1)
            self.assertEqual(pools, [1])
            self.assertEqual(verify.call_count, 3)

    def test_first_verification_uses_two_workers_and_cached_sha_never_replays(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            for index in range(3):
                _write_archive(source, str(index), _meta(wins=5 + index))
            manifest = root / 'configs/tasks.json'
            manifest.parent.mkdir()
            manifest.write_text(json.dumps({'tasks': []}))
            args = (source, root / 'dest', manifest, root, root / 'engine')
            lock, barrier = threading.Lock(), threading.Barrier(2)
            active, peak, calls = 0, 0, 0

            def verify(*unused):
                nonlocal active, peak, calls
                with lock:
                    calls += 1
                    first_pair = calls <= 2
                    active += 1
                    peak = max(peak, active)
                if first_pair:
                    barrier.wait(timeout=1)
                time.sleep(.01)
                with lock:
                    active -= 1
                return {'verified': True}

            with patch('import_llm_run.verification_context', return_value={'engine': 'one'}), \
                 patch('import_llm_run.verify_copy', side_effect=verify):
                run_sha(*args)
            self.assertEqual((calls, peak), (3, 2))
            with patch('import_llm_run.verification_context', side_effect=AssertionError('cached provenance is enough')), \
                 patch('import_llm_run.verify_copy', side_effect=AssertionError('must not replay')):
                result = run_sha(*args)
            self.assertTrue(all(row['verification_reused'] for row in result['rows']))
    def test_interrupted_copy_leaves_no_partial_sha_target_and_can_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            archive = _write_archive(source, 'a', _meta(wins=5))
            manifest = root / 'configs/tasks.json'
            manifest.parent.mkdir()
            manifest.write_text(json.dumps({'tasks': []}))
            args = (source, root / 'dest', manifest, root, root / 'engine')
            copyfile = shutil.copyfile
            def interrupted(source_path, destination):
                copyfile(source_path, destination)
                raise OSError('interrupted archive copy')

            with patch('import_llm_run.verification_context', return_value={'engine': 'one'}), \
                 patch('import_llm_run.verify_copy', return_value={'verified': True}) as verify:
                with patch('pvz_episode_archive.shutil.copyfile', side_effect=interrupted):
                    with self.assertRaisesRegex(OSError, 'interrupted archive copy'):
                        run_sha(*args)
                self.assertFalse((root / 'dest/sha256' / archive_sha(archive)).exists())
                self.assertEqual(list((root / 'dest/sha256').iterdir()), [])
                self.assertEqual(verify.call_count, 0)
                result = run_sha(*args)
                self.assertEqual(verify.call_count, 1)
                self.assertEqual(result['rows'][0]['verify'], '通过')


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
            args = (source, root / 'dest', manifest, root, root / 'engine')
            with patch('import_llm_run.verification_context', return_value={'engine': 'one'}), \
                 patch('import_llm_run.verify_copy', return_value={'verified': True}) as verify:
                first = run_sha(*args)
                before = manifest.read_bytes()
                self.assertEqual(verify.call_count, 2)
                self.assertEqual(len({r['archive'] for r in first['rows']}), 2)
                second = run_sha(*args)
                self.assertEqual(verify.call_count, 2)
                self.assertEqual(manifest.read_bytes(), before)
                self.assertEqual(len(json.loads((root / 'dest/merged_progress.json').read_text())['archives']), 2)
                _write_archive(source, 'c', _meta(wins=7))
                verify.side_effect = RuntimeError('timeline mismatch')
                third = run_sha(*args)
                self.assertEqual(verify.call_count, 3)
                self.assertEqual(third['rows'][2]['grade'], 'loss')
                self.assertIn('timeline mismatch', third['rows'][2]['note'])
                run_sha(*args)
                self.assertEqual(verify.call_count, 3)  # failed receipts also resume
            with patch('import_llm_run.verification_context', return_value={'engine': 'two'}), \
                 patch('import_llm_run.verify_copy', return_value={'verified': True}) as verify:
                run_sha(*args)
                self.assertEqual(verify.call_count, 0)
            target = root / 'dest/sha256' / archive_sha(source / 'a') / 'ticks.jsonl'
            target.write_text('corrupt')
            with patch('import_llm_run.verification_context', return_value={'engine': 'two'}):
                with self.assertRaisesRegex(ValueError, '大小不符'):
                    run_sha(*args)


if __name__ == '__main__':
    unittest.main()
