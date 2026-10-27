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
