"""Existing identity and unified task families, generated in memory."""
import json
from pathlib import Path

from day_orig_tasks import day_original_tasks, is_day_original_source
from make_day_panel import D7_DECK


HOLDOUT_DECKS = {
    7: ((0, 1, 2, 27, 5, 32, 3), (3, 7, 29, 4, 5, 1, 0), (5, 3, 36, 7, 4, 2, 0)),
    8: ((3, 22, 4, 2, 7, 5, 32), (1, 2, 34, 4, 3, 39, 7), (0, 28, 7, 1, 17, 5, 3)),
    9: ((0, 3, 1, 7, 37, 2, 4), (7, 2, 5, 38, 4, 1, 3), (4, 1, 2, 5, 21, 7, 0)),
    10: ((0, 20, 4, 5, 3, 2, 7), (7, 18, 21, 5, 4, 2, 3), (7, 22, 2, 3, 4, 27, 1)),
}
PERTURB_WARNINGS = {8: 1, 9: 1, 10: 1, 11: 4, 12: 1, 13: 1, 14: 1, 15: 1,
                    16: 2, 19: 2, 24: 3, 25: 8, 31: 1, 33: 16}


def _task(name, terrain, level, deck, start=30000, count=1024, **extra):
    return dict(task_id=name, terrain=terrain, level=level, deck=list(deck),
                zombie_count_multiplier=1.0, wave_cap=None, sun_start=50,
                preplanted=[], playthrough=2, seeds=list(range(start, start + count)), **extra)


def day_identity_tasks():
    heldout = [_task(f'day_l{level}_eval_d7p_holdout{index}', 'day', level, deck, count=256,
                     split='heldout_deck',
                     holdout_semantics='unordered plant set excluded from every perturbation training family')
               for level, decks in HOLDOUT_DECKS.items() for index, deck in enumerate(decks, 1)]
    excluded = sorted(tuple(sorted(task['deck'])) for task in heldout)
    tasks = []
    for level in HOLDOUT_DECKS:
        options = dict(protocol_version=4, ok=True, level=level, playthrough=2,
                       available_plants=list(range(40)),
                       not_recommended=[[plant, PERTURB_WARNINGS.get(plant, 0)] for plant in range(40)],
                       warning_scope='Board::SeedNotRecommendedForLevel; raw per-card flags; interactive OnStartButton not invoked')
        tasks.append(_task(f'day_l{level}_train_d7p', 'day', level, D7_DECK,
                           start=1840000 + (level - 7) * 10000, count=8192,
                           deck_perturb=dict(swaps=[1, 2], pool=[p for p in range(40) if p not in PERTURB_WARNINGS],
                                             engine_options=options,
                                             excluded_decks=[list(deck) for deck in excluded])))
    return dict(schema_version=1, purpose='task19 identity-head collection; new task families only',
                warning_scope='raw engine per-card recommendation flags; interactive whole-deck OnStartButton checks not called',
                tasks=tasks + heldout)


def generate(version='v2'):
    tasks = [_task(f'day_l{level}_eval_d7', 'day', level, D7_DECK) for level in (1, 3, 4, 6, 7, 8, 9)]
    tasks.append(_task('day_l10_orig_eval', 'day', 10, (), conveyor_start='original'))
    groups = {task['task_id']: 'day' for task in tasks}
    for task in day_identity_tasks()['tasks'][4:]:
        tasks.append(task)
        groups[task['task_id']] = 'heldout'
    suffix = 'n9' if version == 'v1' else 'n10'
    deck = (0, 1, 2, 3, 4, 5, 7, 8, 9) + (() if version == 'v1' else (11,))
    for level in (11, 12, 13, 14, 16, 17, 18, 19):
        tasks.append(_task(f'night_l{level}_eval_{suffix}', 'night', level, deck))
    tasks.append(_task(f'night_l20_eval_{suffix}_orig', 'night', 20, (), conveyor_start='original'))
    groups.update({task['task_id']: 'night' for task in tasks if task['terrain'] == 'night'})
    if version == 'v2':
        for level in (21, 22, 23, 24, 26):
            task = _task(f'pool_l{level}_eval_p9k', 'pool', level, (0, 1, 2, 3, 4, 5, 7, 16, 19))
            tasks.append(task)
            groups[task['task_id']] = 'pool'
    result = dict(schema_version=1, purpose='task23 unified student panel', groups=groups, tasks=tasks,
                  semantics='L10 holdouts retain archived initial decks and native subsequent conveyor; original L10/L20 start empty')
    if version == 'v2':
        result.update(panel_version='v2', purpose='task27 unified student panel: night n10 and pool p9k')
    return result


def read_panel(source):
    name = Path(source).name.removeprefix('builtin:')
    if name in ('day_identity', 'day_identity.json', 'day_panel_identity.json'):
        return day_identity_tasks()
    if name in ('unified_v1', 'unified_v1.json', 'unified_manifest.json'):
        return generate('v1')
    if name in ('unified_v2', 'unified_v2.json'):
        return generate('v2')
    if is_day_original_source(source):
        return dict(tasks=list(day_original_tasks().values()))
    return json.loads(Path(source).read_text())
