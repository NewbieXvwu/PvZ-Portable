"""task128: the committed gate-v5 panel must be exactly what the generator derives.

The panel is hashed into every gate contract, so a hand edit or a drifted
source would silently change what the gate measures. Regeneration is the check.
"""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]
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


def test_gate_and_selection_keep_scoreboard_game_semantics(tmp_path):
    from pvz_promotion import freeze_panel
    from pvz_selection_refresh import selection_templates, make_plan, SELECTION_POOLS, SELECTION_BLOCK_SIZE
    from render_episode import training_task_kwargs

    source = json.loads(PANEL.read_text())
    gate = freeze_panel(source, version=5, seed_block=(100_000_000, 100_001_920))
    model = tmp_path / 'model.pt'
    model.write_bytes(b'incumbent')
    start = SELECTION_POOLS[0][0]
    selection = make_plan(selection_templates(ROOT / 'configs/student_selection_holdout.json', PANEL),
                          (start, start + SELECTION_BLOCK_SIZE), model)
    definitions = json.loads((ROOT / 'configs/adventure_scoreboard.json').read_text())['tasks']
    for panel in (gate, selection):
        table = {task['task_id']: task for task in panel['tasks']}
        for definition in definitions:
            if definition['task_id'] not in table:
                continue
            task = table[definition['task_id']]
            assert all(task[key] == value for key, value in definition.items())
            # Compare the actual reset arguments, including fog and conveyor
            # modes, at an identical seed. No game replay is needed.
            assert training_task_kwargs(task, 42) == training_task_kwargs(definition, 42)
