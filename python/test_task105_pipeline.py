"""Resume from result JSONs; incomplete packages cannot stop the supervisor."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import task105_pipeline as pipeline
import task105_supervise as supervisor
import task105_import_batch as batch
import task105_results as results


class ResumeTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.out = Path(tmp.name)
        self.entry = dict(task_id='task', seed=1, archive='sha256/abc', future_seeds=list(range(8)))
        self.result = dict(complete=True, archive='sha256/abc', future_seeds=list(range(8)),
                           ladder=dict(ladder=[90, 50]))
        self.tasks = dict(task=dict(demo_start=dict(archive='sha256/abc', decisions=[90, 50])))

    def saved(self):
        path = self.out / 'scan/results/task_s1.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.result))
        return pipeline.scan_finished(self.entry, self.out, self.tasks)

    def test_registered_scan_skips_both_commands(self):
        self.assertTrue(self.saved())

    def test_completed_scan_without_registration_still_needs_register(self):
        self.tasks = {}
        self.assertFalse(self.saved())

    def test_no_passing_ladder_is_a_terminal_scan(self):
        self.result['ladder'] = None
        self.tasks = {}
        self.assertTrue(self.saved())

    def test_partial_or_different_context_cannot_be_skipped(self):
        for key, value in [('complete', False), ('archive', 'different'), ('future_seeds', [99])]:
            with self.subTest(key=key):
                old = self.result[key]
                self.result[key] = value
                self.assertFalse(self.saved())
                self.result[key] = old

    def test_legacy_complete_requires_complete_package(self):
        status = self.out / 'pipeline_status.json'
        status.write_text(json.dumps(dict(state='complete')))
        with patch.object(supervisor, 'OUT', self.out), patch.object(supervisor, 'STATUS', status):
            for complete in (False, True):
                (self.out / 'delivery_summary.json').write_text(json.dumps(dict(complete=complete)))
                self.assertEqual(supervisor.state(), 'complete' if complete else 'incomplete')

    def test_input_receipt_hashes_only_when_input_changes(self):
        source = self.out / 'run.zip'
        source.write_bytes(b'first')
        with patch.object(pipeline, 'file_sha', return_value='sha-first') as digest:
            pipeline.record_input(source, self.out)
            pipeline.record_input(source, self.out)
            digest.assert_called_once_with(source)
        source.write_bytes(b'updated-run')
        with patch.object(pipeline, 'file_sha', return_value='sha-updated') as digest:
            pipeline.record_input(source, self.out)
            digest.assert_called_once_with(source)
        self.assertEqual(json.loads((self.out / 'input_receipt.json').read_text())['sha256'], 'sha-updated')

    def test_skip_import_uses_union_and_never_launches_import(self):
        out = self.out / 'artifacts/task105'
        (out / 'import').mkdir(parents=True)
        (out / 'import/merged_progress.json').write_text(json.dumps({'archives': [{}, {}]}))
        (out / 'import/import_progress.json').write_text(json.dumps({'archives': [{}]}))
        (out / 'scan_jobs.json').write_text(json.dumps({'archives': []}))
        (self.out / 'configs').mkdir()
        (self.out / 'configs/human_demo_tasks.json').write_text(json.dumps({'tasks': []}))
        with patch.object(pipeline, 'ROOT', self.out), \
             patch.object(pipeline, 'file_sha', return_value=pipeline.ENGINE_SHA), \
             patch.object(sys, 'argv', ['pipeline', 'unused.zip', '--skip-import']), \
             patch.object(pipeline.subprocess, 'run') as run:
            run.return_value.returncode = 0
            pipeline.main()
        self.assertTrue(all('task105_results.py' in call.args[0][1] for call in run.call_args_list))
        self.assertEqual(json.loads((out / 'pipeline_status.json').read_text())['imported'], 2)

    def test_package_union_preserves_old_scan_context_and_sha_variants(self):
        old = dict(task_id='t_abc', archive='sha256/abc', future_seeds=[10, 11])
        variant = dict(task_id='t_def', archive='sha256/def', future_seeds=[20, 21])
        rows, plan = batch.merge([dict(archive_sha256='abc')],
            dict(schema='jobs', archives=[old]),
            [dict(archives=[dict(archive_sha256='abc'), dict(archive_sha256='def')],
                  plan=dict(archives=[{**old, 'reused_reservation': True}, variant]))])
        self.assertEqual(len(rows['archives']), 2)
        self.assertEqual(plan['archives'], [old, variant])
        with self.assertRaisesRegex(ValueError, 'conflicting saved scan context'):
            batch.merge([], dict(schema='jobs', archives=[old]),
                        [dict(archives=[], plan=dict(archives=[{**old, 'future_seeds': [30]}]))])

    def test_controller_addendum_survives_generated_result_replacement(self):
        text = '# task\n\n## 结果\nold results\n\n## 追加输入 2\nnew inputs\n'
        prefix = results.report_prefix(text)
        self.assertIn('## 追加输入 2\nnew inputs', prefix)
        self.assertNotIn('old results', prefix)
        self.assertEqual(results.report_prefix(prefix + '## 结果\nnew results'), prefix)

    def test_staged_import_publication_preserves_scans_and_deduplicates_copies(self):
        out = self.out / 'artifacts/task105'
        (out / 'import').mkdir(parents=True)
        (self.out / 'configs').mkdir()
        source = self.out / 'llmplay3/out'
        source.mkdir(parents=True)
        manifest = self.out / 'configs/human_demo_tasks.json'
        existing = dict(task_id='old', level=41, seeds=[1], zombie_count_multiplier=1.,
                        deck=[1, 2], demo_start=dict(archive='sha/old', decisions=[90]))
        added = dict(task_id='new', level=42, seeds=[2], zombie_count_multiplier=1., deck=[3, 4])
        manifest.write_text(json.dumps(dict(tasks=[existing])))
        staged = self.out / 'staged.json'
        staged.write_text(json.dumps(dict(tasks=[{k: v for k, v in existing.items()
                                                 if k != 'demo_start'}, added])))
        old_job = dict(task_id='old', archive='sha/old', future_seeds=[10])
        new_job = dict(task_id='new', archive='sha/new', future_seeds=[20])
        (out / 'scan_jobs.json').write_text(json.dumps(dict(schema='jobs', archives=[old_job])))
        old_row = dict(archive_sha256='old', verify='通过', archive='sha/old')
        new_row = dict(archive_sha256='new', verify='通过', archive='sha/new')
        (out / 'import/merged_progress.json').write_text(json.dumps(dict(archives=[old_row])))
        old_package = dict(identity=dict(path='old.zip'), archives=1)
        (out / 'input_packages.json').write_text(json.dumps(dict(packages=[old_package])))
        plan_path = out / 'staged_plan.json'
        plan_path.write_text(json.dumps(dict(schema='jobs', archives=[old_job, new_job])))
        index = out / 'index.json'
        index.write_text(json.dumps(dict(input=str(source), scan_plan=str(plan_path),
                         archives=[old_row, new_row, {**new_row, 'archive': ''}])))
        with patch.object(batch, 'ROOT', self.out), patch.object(batch, 'OUT', out):
            for _ in range(2):
                progress, plan = batch.append_completed_import(source, index, staged)
        self.assertEqual(plan['archives'], [old_job, new_job])
        self.assertEqual(progress['archives'], [old_row, new_row])
        self.assertEqual(json.loads(manifest.read_text())['tasks'], [existing, added])
        packages = json.loads((out / 'input_packages.json').read_text())['packages']
        self.assertEqual(len(packages), 2)
        self.assertEqual(packages[-1]['source_copies'], 3)
        self.assertEqual(packages[-1]['archives'], 2)
        self.assertEqual(json.loads((out / 'pipeline_status.json').read_text())['state'], 'incomplete')

    def test_directory_identity_notices_nested_archive_changes_without_content_reads(self):
        archive = self.out / 'archives/one'
        archive.mkdir(parents=True)
        tape = archive / 'ticks.jsonl'
        tape.write_bytes(b'one')
        first = batch.identity(self.out)
        tape.write_bytes(b'longer tape')
        with patch.object(Path, 'read_bytes', side_effect=AssertionError('no content hashing')):
            second = batch.identity(self.out)
        self.assertNotEqual(first['inventory_sha256'], second['inventory_sha256'])
        self.assertEqual(second['size'], len(b'longer tape'))

    def test_cached_package_is_not_imported_again_before_union_resume(self):
        out = self.out / 'batch'
        (out / 'import').mkdir(parents=True)
        (out / 'import_packages').mkdir()
        source = self.out / 'one.zip'
        source.write_bytes(b'input')
        old = dict(task_id='t_abc', archive='sha256/abc', future_seeds=[10])
        package = dict(identity=batch.identity(source), archives=[dict(archive_sha256='abc', verify='通过')],
                       plan=dict(schema='jobs', archives=[old]))
        (out / 'import_packages/one.json').write_text(json.dumps(package))
        (out / 'import/progress_resume00099.json').write_text(json.dumps({'archives': package['archives']}))
        (out / 'scan_jobs_resume00099.json').write_text(json.dumps(package['plan']))
        with patch.object(batch, 'OUT', out), \
             patch.object(sys, 'argv', ['batch', str(source)]), \
             patch.object(batch.subprocess, 'run') as run, \
             patch.object(batch.os, 'execve') as resume:
            batch.main()
        self.assertEqual(run.call_count, 1)
        self.assertIn('task105_results.py', run.call_args.args[0][1])
        self.assertEqual(json.loads((out / 'scan_jobs.json').read_text())['archives'], [old])
        self.assertIn('--skip-import', resume.call_args.args[1])
