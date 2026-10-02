"""HF transfers preserve raw evidence and refuse incomplete reference chains."""
import contextlib
import fcntl
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import hf_sync


class RecordingApi:
    def __init__(self, destination):
        self.destination = destination
        self.uploads = []

    def create_repo(self, **kwargs):
        pass

    def upload_folder(self, **kwargs):
        self.uploads.append(kwargs)
        # Materialize downloads: copying follows file links, never copies links.
        shutil.copytree(kwargs["folder_path"], self.destination, symlinks=False)


class EvidenceIntegrityTests(unittest.TestCase):
    def setUp(self):
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        self.base = Path(holder.name)
        self.research = self.base / "research"
        self.run = self.research / "run_x"
        self.models = self.run / "runs/run_1"
        self.models.mkdir(parents=True)
        self.download = self.base / "download"
        self.api = RecordingApi(self.download)
        for target, value in [("RESEARCH_DIR", self.research),
                              ("_require_hf", lambda: (lambda **kwargs: self.api, None))]:
            p = patch.object(hf_sync, target, value)
            p.start()
            self.addCleanup(p.stop)

    def file(self, path, contents):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
        return path

    def resume(self, checkpoint):
        pointer = {"checkpoint": str(checkpoint.relative_to(self.run)),
                   "sha256": hf_sync._sha256(checkpoint)}
        (self.run / "resume.json").write_text(json.dumps(pointer))

    def cli(self, *args):
        with patch.object(sys, "argv", ["hf_sync.py", "--repo", "someone/private", *map(str, args)]), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return hf_sync.main()

    def test_rollouts_logs_window_and_resume_survive_verified_download(self):
        for update in range(12):
            self.file(self.models / f"update_{update:06d}_trained_{update}.pt", f"weights{update}".encode())
        boundary = self.file(self.models / "update_000012_boundary_100.pt", b"boundary-state")
        self.resume(boundary)
        shard = self.file(self.models / ".seed_jobs/update_1/hash/seed_0.npz", b"raw-shard")
        log = self.file(self.base / "run.log", b"failure and final evaluation\n")
        self.cli("push", "run_x", "--include-rollouts", "--include-trained-window", "--log", log)
        self.assertEqual(self.cli("verify", self.download), 0)
        manifest = json.loads((self.download / "MANIFEST.json").read_text())
        self.assertEqual(len([p for p in manifest["files"] if "_trained_" in p]), 8)
        self.assertIn(str(boundary.relative_to(self.run)), manifest["files"])
        self.assertEqual((self.download / shard.relative_to(self.run)).read_bytes(), b"raw-shard")
        self.assertEqual((self.download / "logs/run.log").read_bytes(), log.read_bytes())
        self.assertEqual(boundary.read_bytes(), b"boundary-state")
        self.assertTrue(shard.exists())

    def test_explicit_evidence_branch_is_bound_to_upload_and_manifest(self):
        checkpoint = self.file(self.models/'update_000001_boundary_1.pt', b'weights')
        self.resume(checkpoint)
        self.cli('--revision', 'evidence-run-x', 'push', 'run_x')
        self.assertEqual(self.api.uploads[0]['revision'], 'evidence-run-x')
        manifest=json.loads((self.download/'MANIFEST.json').read_text())
        self.assertEqual(manifest['hf_revision'], 'evidence-run-x')
        self.assertIn(str(checkpoint.relative_to(self.run)),manifest['files'])
        self.assertEqual(self.cli('verify',self.download),0)

    def test_explicit_branch_applies_to_whole_failed_evidence_tree(self):
        self.file(self.run/'report.json',b'{"gate_result":"fail"}')
        self.file(self.run/'failed.log',b'original failure')
        self.cli('--revision','evidence-failed','push-evidence',self.run,'--name','failed')
        self.assertEqual(self.api.uploads[0]['revision'],'evidence-failed')
        self.assertEqual((self.download/'failed.log').read_bytes(),b'original failure')

    def test_read_commands_respect_explicit_branch(self):
        listings,downloads=[],[]
        self.api.list_repo_files=lambda **kwargs: listings.append(kwargs) or ['run_x/MANIFEST.json']
        with patch.object(hf_sync,'_require_hf',lambda:(lambda **kwargs:self.api,lambda **kwargs:downloads.append(kwargs))):
            self.cli('--revision','evidence-run-x','ls')
            self.cli('--revision','evidence-run-x','pull','run_x','--dest',self.base/'pulled')
            self.cli('--revision','evidence-run-x','pull-all','--dest',self.base/'all')
        self.assertEqual(listings[0]['revision'],'evidence-run-x')
        self.assertEqual([item['revision'] for item in downloads],['evidence-run-x','evidence-run-x'])

    def test_missing_or_wrong_resume_hash_stops_before_upload(self):
        checkpoint = self.file(self.models / "update_000001_boundary_1.pt", b"weights")
        self.resume(checkpoint)
        checkpoint.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.cli("push", "run_x")
        checkpoint.unlink()
        with self.assertRaisesRegex(ValueError, "target missing"):
            self.cli("push", "run_x")
        self.assertEqual(self.api.uploads, [])

    def test_dry_run_needs_no_sdk_credentials_and_checks_selected_logs(self):
        self.file(self.models / "update_000001_boundary_1.pt", b"weights")
        with patch.object(hf_sync, "_require_hf", side_effect=AssertionError("SDK must not run")):
            self.assertEqual(self.cli("push", "run_x", "--dry-run"), 0)
            with self.assertRaisesRegex(FileNotFoundError, "selected log missing"):
                self.cli("push", "run_x", "--dry-run", "--log", self.base / "absent.log")

    def test_active_writer_is_not_interrupted_or_uploaded(self):
        lock = self.run / ".execution.lock"
        with lock.open("wb") as writer:
            fcntl.flock(writer, fcntl.LOCK_EX)
            with self.assertRaisesRegex(RuntimeError, "active run"):
                self.cli("push", "run_x")
            self.assertEqual(self.api.uploads, [])

    def test_full_archive_upload_verification_and_damage_rejection(self):
        source = self.base / "archive"
        payload = self.file(source / "failed/shard.npz", b"corrupt-original-preserved")
        log = self.file(source / "failed.log", b"original failure\n")
        inventory = {str(p.relative_to(source)): {"bytes": p.stat().st_size, "sha256": hf_sync._sha256(p)}
                     for p in (payload, log)}
        (source / "archive_manifest.json").write_text(json.dumps({"files": inventory}))
        self.cli("push-evidence", source, "--name", "archives/failure")
        self.assertEqual(self.cli("verify", self.download), 0)
        self.assertEqual((self.download / "failed.log").read_bytes(), log.read_bytes())
        (self.download / "failed/shard.npz").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "integrity failure"):
            self.cli("verify", self.download)
        payload.unlink()
        with self.assertRaisesRegex(ValueError, "reference chain broken"):
            self.cli("push-evidence", source, "--name", "archives/failure", "--dry-run")


if __name__ == "__main__":
    unittest.main()
