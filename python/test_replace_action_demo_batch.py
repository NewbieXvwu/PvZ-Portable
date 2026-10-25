"""Scheduling guards for demonstration validation; no game process is started here."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import replace_action_demo_batch as batch


class DemonstrationBatchGuardsTest(unittest.TestCase):
    def test_busy_game_slots_refuse_the_batch_before_outputs_or_workers_are_created(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan = root / 'plan.json'
            plan.write_text(json.dumps(dict(archives=[{}] * 4, N=420, intervals=[48, 420])))
            with patch.object(batch, 'task_game_pids', return_value=[123]), \
                 patch.object(batch, 'ProcessPoolExecutor') as pool:
                with self.assertRaisesRegex(RuntimeError, 'slots are not free'):
                    batch.run(plan, root / 'output', root / 'resources', root / 'engine')
            pool.assert_not_called()
            self.assertFalse((root / 'output').exists())

    def test_a_guessed_horizon_cannot_replace_the_frozen_observed_maximum(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan = root / 'plan.json'
            plan.write_text(json.dumps(dict(archives=[{}] * 4, N=120, intervals=[48, 420])))
            with self.assertRaisesRegex(ValueError, 'frozen four-demo interval plan'):
                batch.run(plan, root / 'output', root / 'resources', root / 'engine')


if __name__ == '__main__':
    unittest.main()
