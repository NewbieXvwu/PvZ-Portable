"""Reject changed reference configs and failed predecessors before GPU work."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import research_observation_interrupt_audit as audit


class ReferenceProtocolTests(unittest.TestCase):
    def test_reference_model_and_source_are_frozen(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "config.json"
            config.write_text('{}')
            model = {"width": 256, "layers": 6, "heads": 8}
            protocol = {"helper_sha256": audit.sha256_file(Path(audit.__file__)),
                        "required_fingerprints": {"config.json": audit.sha256_file(config)},
                        "configs": ["config.json"], "model": model}
            path = root / "protocol.json"
            path.write_text(json.dumps(protocol))
            with patch.object(audit, "ROOT", root):
                self.assertEqual(audit.read_protocol(path, [config], [{"model": model}]), protocol)
                with self.assertRaises(ValueError):
                    audit.read_protocol(path, [config], [{"model": {"width": 32}}])
                config.write_text('{"changed":true}')
                with self.assertRaises(ValueError):
                    audit.read_protocol(path, [config], [{"model": model}])

    def test_no_gpu_work_while_predecessor_running(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = root / "queue.json"
            state.write_text('{"status":"running"}')
            with self.assertRaises(RuntimeError):
                audit.wait_for_idle({"idle_states": {str(state): "matrix_budget_complete"}}, root, False)

    def test_successful_candidate_boundary_is_not_a_failed_matrix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = root / "queue.json"
            state.write_text('{"status":"process_finished","returncode":0}')
            def complete(_):
                state.write_text('{"status":"matrix_budget_complete"}')
            with patch.object(audit.time, "sleep", side_effect=complete):
                audit.wait_for_idle({"idle_states": {str(state): "matrix_budget_complete"}}, root, True)

    def test_predecessor_failure_stops_waiting_without_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = root / "queue.json"
            for data in ({"status": "process_finished", "returncode": 1}, {"status": "failed"}):
                state.write_text(json.dumps(data))
                with self.assertRaises(RuntimeError):
                    audit.wait_for_idle({"idle_states": {str(state): "complete"}}, root, True)


if __name__ == "__main__":
    unittest.main()
