"""Original day tasks, generated in memory; seed blocks retain their production identity."""
from pathlib import Path

from pvz_tasks import deck_for_level

DAY_ORIG_MANIFEST = Path('builtin:day_orig')


def is_day_original_source(source):
    # Old queued jobs can still contain the retired manifest name.
    name = str(source)
    return name == str(DAY_ORIG_MANIFEST) or name.endswith((
        '/configs/panels/day_orig_panel.json',
        '/.claude/' 'worktrees/conveyor/artifacts/stage4/day_panel.json'))


def day_original_tasks():
    tasks = {}
    for level in (1, 2, 3, 4, 6, 7, 8, 9, 10):
        task_id = f'day_l{level}_eval'
        tasks[task_id] = dict(task_id=task_id, terrain='day', level=level,
                              deck=list(deck_for_level(level)),
                              zombie_count_multiplier=1.0, wave_cap=None,
                              sun_start=50, preplanted=[], playthrough=2,
                              seeds=list(range(30000, 31024)))
    for split, start, count in (('eval', 30000, 1024), ('train', 1880000, 8192)):
        task_id = f'day_l10_orig_{split}'
        tasks[task_id] = dict(task_id=task_id, terrain='day', level=10,
                              deck=list(deck_for_level(10, conveyor_start='original')),
                              conveyor_start='original', zombie_count_multiplier=1.0,
                              wave_cap=None, sun_start=50, preplanted=[], playthrough=2,
                              seeds=list(range(start, start + count)))
    return tasks
