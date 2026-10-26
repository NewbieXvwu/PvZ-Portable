import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'python'))

import importlib.util

_spec = importlib.util.spec_from_file_location('task104_panel_eval', ROOT / 'scripts/task104_panel_eval.py')
panel_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(panel_eval)


class PanelManifestTest(unittest.TestCase):
    def test_panel_becomes_search_play_manifest(self):
        panel = {'tasks': [
            {'task_id': 'roof_l42', 'level': 42, 'terrain': 'roof', 'deck': [1, 33],
             'seeds': [104000000, 104000001], 'playthrough': 2, 'zombie_count_multiplier': 2.0,
             'wave_cap': None, 'preplanted': [], 'copied_from': 'demo_roof_42_melon_m200'},
            {'task_id': 'roof_l43', 'level': 43, 'terrain': 'roof', 'deck': [1, 34],
             'seeds': [104000100], 'playthrough': 2, 'zombie_count_multiplier': 2.0,
             'wave_cap': None, 'preplanted': []},
        ]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'manifest.json'
            document = panel_eval.manifest_for(panel, path, None)
            written = json.loads(path.read_text())
        self.assertEqual(written, document)
        self.assertEqual([task['task_id'] for task in document['tasks']], ['roof_l42', 'roof_l43'])
        self.assertEqual(document['tasks'][0]['seeds'], [104000000, 104000001])
        # panel-only bookkeeping must not leak into the manifest search_play reads
        self.assertNotIn('terrain', document['tasks'][0])
        self.assertNotIn('copied_from', document['tasks'][0])

    def test_level_filter_and_empty_selection(self):
        panel = {'tasks': [{'task_id': 'a', 'level': 42, 'deck': [1], 'seeds': [7]},
                           {'task_id': 'b', 'level': 43, 'deck': [1], 'seeds': [8]}]}
        with tempfile.TemporaryDirectory() as directory:
            document = panel_eval.manifest_for(panel, Path(directory) / 'm.json', [43])
            self.assertEqual([task['task_id'] for task in document['tasks']], ['b'])
            with self.assertRaises(ValueError):
                panel_eval.manifest_for(panel, Path(directory) / 'n.json', [99])
            with self.assertRaises(ValueError):
                panel_eval.manifest_for({'tasks': [{'task_id': 'c', 'level': 1, 'deck': [1]}]},
                                        Path(directory) / 'o.json', None)


if __name__ == '__main__':
    unittest.main()
