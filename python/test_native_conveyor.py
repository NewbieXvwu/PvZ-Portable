"""task75 native conveyor family contracts."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'python')]

from make_native_conveyor_panel import build_tasks
from pvz_tasks import conveyor_belt_level, deck_for_seed


def test_native_conveyor_manifest_preserves_engine_dealing():
    tasks = build_tasks()
    assert len(tasks) == 6
    seeds = set()
    for task in tasks:
        assert conveyor_belt_level(task['level'])
        assert task['deck'] == [] and task['conveyor_start'] == 'original'
        assert task['wave_cap'] is None and task['zombie_count_multiplier'] == 1
        assert task['playthrough'] == 2 and task['preplanted'] == []
        assert not seeds.intersection(task['seeds'])
        seeds.update(task['seeds'])
        assert deck_for_seed(task, task['seeds'][0]) == ()
        assert len(task['seeds']) == (32 if task['task_id'].endswith('_eval') else 8192)
        assert task.get('fog_observation') == ('storm' if task['level'] == 40 else None)
