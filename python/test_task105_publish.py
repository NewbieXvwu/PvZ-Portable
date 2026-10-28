"""A failed background push must not duplicate a scan commit or collect other files."""
import contextlib
import fcntl
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import task105_publish as publisher


class PublicationTest(unittest.TestCase):
    def test_push_retry_is_idempotent_and_leaves_other_changes_unstaged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'repo'
            remote = Path(directory) / 'remote.git'
            root.mkdir()

            def run(*args):
                return subprocess.check_output(['git', '-C', str(root), *args],
                                               text=True, stderr=subprocess.STDOUT).strip()

            run('init', '--initial-branch=llmimport')
            run('config', 'user.name', 'Task105 Test')
            run('config', 'user.email', 'test@example.invalid')
            subprocess.run(['git', 'init', '--bare', str(remote)], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            run('remote', 'add', 'origin', str(remote))
            for name in publisher.FILES:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({'scans': []}) if name == publisher.EVIDENCE else '{}\n')
            (root / '.gitignore').write_text('artifacts/\n')
            (root / 'unrelated.py').write_text('before\n')
            run('add', '.')
            run('commit', '-m', 'baseline')
            run('push', '-u', 'origin', 'llmimport')
            (root / 'unrelated.py').write_text('keep unstaged\n')
            out = root / 'artifacts/task105'
            out.mkdir(parents=True)
            (out / 'pipeline_status.json').write_text(json.dumps({'stage': 'adaptive_next'}))
            evidence = dict(scans=[dict(task_id='completed', registered=False, reason='no ladder')],
                            scanned=1, scan_expected=2, registered=0, deployed=0)
            (root / publisher.EVIDENCE).write_text(json.dumps(evidence))
            original_git = publisher.git

            def offline_push(*args):
                if args[0] == 'push':
                    with (out / 'delivery.lock').open('a') as handle:
                        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    raise subprocess.CalledProcessError(1, args, 'offline')
                return original_git(*args)

            with patch.object(publisher, 'ROOT', root), patch.object(publisher, 'OUT', out), \
                 contextlib.redirect_stdout(io.StringIO()):
                with patch.object(publisher, 'git', side_effect=offline_push):
                    with self.assertRaises(subprocess.CalledProcessError):
                        publisher.publish()
                commit = run('rev-parse', 'HEAD')
                self.assertEqual(run('rev-list', '--count', 'origin/llmimport..HEAD'), '1')
                publisher.publish()
                publisher.publish()
                self.assertEqual(run('rev-parse', 'HEAD'), commit)
                self.assertEqual(run('rev-list', '--count', 'origin/llmimport..HEAD'), '0')
                self.assertIn('M unrelated.py', run('status', '--short'))
                self.assertEqual(run('ls-files', 'artifacts'), '')
                daily = next((root / 'MEMORY').glob('20??-??-??.md'))
                self.assertEqual(daily.read_text().count('task105扫描完成：`completed`'), 1)
                run('add', 'unrelated.py')
                with self.assertRaisesRegex(ValueError, 'staged changes'):
                    publisher.publish()
                self.assertEqual(run('diff', '--cached', '--name-only'), 'unrelated.py')


if __name__ == '__main__':
    unittest.main()
