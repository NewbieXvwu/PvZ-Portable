"""task128: the committed gate-v5 panel must be exactly what the generator derives.

The panel is hashed into every gate contract, so a hand edit or a drifted
source would silently change what the gate measures. Regeneration is the check.
"""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / 'configs/gate_v5_panel.json'


def test_committed_panel_regenerates_byte_identically():
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/make_gate_v5_panel.py'), '--check'],
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr


def test_panel_is_the_adventure_set_with_all_loss_cells_excluded():
    document = json.loads(PANEL.read_text())
    assert document['panel_version'] == 'v5'
    adventure = [task for task in document['tasks'] if task['task_id'] not in document['monitor']]
    assert len(adventure) == 36 and len(document['monitor']) == 2
    assert len({task['level'] for task in adventure}) == 36
    assert all(task['zombie_count_multiplier'] == 1.0 and task['wave_cap'] is None for task in adventure)
    # Every excluded level is either the boss or a roof level with no recorded win.
    assert sorted(entry['level'] for entry in document['excluded']) == [41, 42, 43, 44, 46, 47, 48, 49, 50]
    assert all(entry['student_wins'] == 0 for entry in document['excluded'])
    # L30/L34/L39 are in on the teacher's wins, not on a student win; L30 and L40
    # are native conveyor levels, so they legitimately carry an empty task deck.
    assert {task['level'] for task in adventure if task['deck'] == []} == {30, 40}
    assert {task['level'] for task in adventure
            if document['qualification'][task['task_id']]['student_wins'] == 0} == {30, 34, 39}
    assert document['zone_counts'] == {'day': 9, 'fog': 9, 'heldout': 2, 'night': 9, 'pool': 9}
    assert all(record['zone'] != 'roof' for record in document['qualification'].values())
