"""A storage revision may authorize one source hash, never experiment drift."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import research_comparison_summary as summary


class StorageRevisionTests(unittest.TestCase):
    def setUp(self):
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        self.root = Path(holder.name)
        self.addCleanup(patch.stopall)
        patch.object(summary, "ROOT", self.root).start()
        self.old = {"experiment_id": "candidate", "ppo": {"learning_rate": .0001},
                    "sampling": {"task_ids": ["one", "two"]}, "prerequisites": ["old-gate"]}
        self.new = copy.deepcopy(self.old)
        self.new["prerequisites"] = ["new-gate"]
        self.write("old.json", self.old)
        self.write("new.json", self.new)
        original = {"order": [{"config": "old.json", "output_dir": "run", "log": "run.log"}]}
        self.write("original.json", original)
        audit = {"gate_result": "pass", "remaining_executable_ast_exact": True, "trained_keep": 8,
                 "old_source_sha256": "old-hash", "new_source_sha256": "new-hash"}
        self.write("audit.json", audit)
        self.queue = {"order": [{"config": "new.json", "output_dir": "run", "log": "run.log"}],
                      "storage_revision": {"audit": "audit.json", "original_queue": "original.json",
                          "audit_sha256": self.digest("audit.json"),
                          "original_queue_sha256": self.digest("original.json"),
                          "old_gate": "old-gate", "new_gate": "new-gate", "candidates": ["candidate"]}}

    def write(self, name, data):
        (self.root / name).write_text(json.dumps(data))

    def digest(self, name):
        return hashlib.sha256((self.root / name).read_bytes()).hexdigest()

    def test_accepts_only_prerequisite_substitution(self):
        self.assertEqual(summary.storage_revision(self.queue)["trained_keep"], 8)

    def test_rejects_learning_rate_change(self):
        self.new["ppo"]["learning_rate"] = .001
        self.write("new.json", self.new)
        with self.assertRaises(ValueError):
            summary.storage_revision(self.queue)

    def test_rejects_task_reordering(self):
        self.new["sampling"]["task_ids"].reverse()
        self.write("new.json", self.new)
        with self.assertRaises(ValueError):
            summary.storage_revision(self.queue)

    def test_rejects_audit_change(self):
        (self.root / "audit.json").write_text("{}")
        with self.assertRaises(ValueError):
            summary.storage_revision(self.queue)

    def test_rejects_unknown_source_and_keeps_other_fingerprints(self):
        audit = summary.storage_revision(self.queue)
        good = {"python/pvz_research.py": "new-hash", "simulator": "native-hash"}
        actual = summary.comparable_fingerprints(good, audit)
        self.assertEqual(actual, {"python/pvz_research.py": "old-hash", "simulator": "native-hash"})
        self.assertEqual(good["python/pvz_research.py"], "new-hash")
        with self.assertRaises(ValueError):
            summary.comparable_fingerprints({"python/pvz_research.py": "third-hash"}, audit)


if __name__ == "__main__":
    unittest.main()
