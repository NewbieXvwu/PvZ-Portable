"""task92: a fixed, independent playing panel before the unchanged promotion gate."""
from __future__ import annotations

import copy

TRAINING_SEEDS = (1, 2)
SELECTION_SEED_BLOCK = (120_000_000, 120_006_000)
SELECTION_GAMES = 32
SELECTION_TASKS = (
    ('day', 'day_l7_eval_d7'),
    ('heldout', 'day_l7_eval_d7p_holdout1'),
    ('night', 'night_l12_eval_n10'),
    ('pool', 'pool_l21_eval_p9k'),
    ('fog', 'fog_user_l31_snow'),
    ('roof', 'roof_l41_train_r7'),
)


def seed_spans(document):
    """Conservatively reserve each declared list's full span, including demo futures."""
    if isinstance(document, dict):
        for key in ('seeds', 'future_seeds'):
            values = document.get(key)
            if isinstance(values, list) and values and all(type(seed) is int for seed in values):
                yield min(values), max(values) + 1
        start, count = document.get('future_seed_start'), document.get('future_seed_count')
        if type(start) is int and type(count) is int and count > 0:
            yield start, start + count
        for value in document.values():
            yield from seed_spans(value)
    elif isinstance(document, list):
        for value in document:
            yield from seed_spans(value)


def assert_disjoint(panel, reserved):
    seeds = [seed for task in panel['tasks'] for seed in task['seeds']]
    if len(seeds) != len(set(seeds)):
        raise ValueError('selection panel repeats a seed')
    for low, high in reserved:
        if any(low <= seed < high for seed in seeds):
            raise ValueError(f'selection seeds overlap registered interval [{low}, {high})')


def make_panel(source):
    """Keep registered task semantics; only replace the seed split."""
    table = {task['task_id']: task for task in source['tasks']}
    if 'fog_user_l31_snow' not in table:
        from make_fog_user_panel import tasks
        table.update((task['task_id'], task) for task in tasks())
    if 'roof_l41_train_r7' not in table:
        from make_roof_panel import ROOF
        from pvz_tasks import deck_for_level
        from zone_panel import task_document
        task = task_document(ROOF, 41, deck_for_level(41, family='roof_r7'),
                             'roof_l41_train_r7', [46_000_000])
        table[task['task_id']] = task
    tasks = []
    for index, (zone, name) in enumerate(SELECTION_TASKS):
        task = copy.deepcopy(table[name])
        first = SELECTION_SEED_BLOCK[0] + index * 1000
        task['seeds'] = list(range(first, first + SELECTION_GAMES))
        task.pop('demo_start', None)
        tasks.append(task)
    return dict(schema_version=1, panel_version='selection-task92-v1',
                purpose='independent student selection; total wins, greedy fast arm',
                groups=dict((name, zone) for zone, name in SELECTION_TASKS), tasks=tasks)
