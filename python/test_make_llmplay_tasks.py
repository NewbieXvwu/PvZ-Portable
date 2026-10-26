"""Derived tasks retain terrain/deck, never a source demonstration start."""
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts' / 'llmplay'))
from make_llmplay_tasks import NEW_FAMILIES, build_missing


def test_derived_tasks_do_not_inherit_demo_start_or_seed():
    known = {task['task_id']: task for task in
             json.loads((ROOT / 'configs/human_demo_tasks.json').read_text())['tasks']}
    for task_id, (source_id, _) in NEW_FAMILIES.items():
        if source_id is not None:
            known.pop(task_id, None)
    for source_id, _ in NEW_FAMILIES.values():
        if source_id is not None:
            known[source_id]['demo_start'] = dict(archive='/source', decisions=[100], future_seeds=[900])
    before = copy.deepcopy(known)
    derived = build_missing(known)
    assert derived
    for task in derived:
        source_id, overrides = NEW_FAMILIES[task['task_id']]
        if source_id is not None:
            assert 'demo_start' not in task and 'seeds' not in task
            assert task['deck'] == known[source_id]['deck']
            assert all(task[key] == value for key, value in overrides.items())
    assert known == before
