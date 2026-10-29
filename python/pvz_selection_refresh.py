"""Per-round selection plans from the gate-v5 adventure panel and a separate seed ledger.

task128: the body of the selection panel is the same adventure set the gate
measures (original 1.0x, scoreboard decks) on independent seeds, so the
candidate that gets picked is the one that plays the game's own levels best.
task137: the eight fixed v5 evaluation decks from configs/random_deck_cells_v5.json
are the guard against deck-specific overfitting; they are disjoint from the 16
training decks and never collected. The task126 unseen-deck cells stay as evidence
only: their wins are recorded in the round report but never enter the selection
total (`score_only`).

The old 30%-80% window screening is gone with the body it calibrated: selection
no longer reads the collector's windows, so a window whose games cannot be
attributed to the incumbent can no longer silently drop a cell.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

SELECTION_POOLS = ((130_000_000, 140_000_000),)
SELECTION_BLOCK_SIZE = 512
SELECTION_LEDGER = Path.home() / 'pvz-logs/selection_seed_ledger.json'
# task137: the three deck_random families were themselves curriculum material, so a
# candidate could be scored on decks it had been trained on. Selection now uses the
# fixed v5 evaluation decks from the task137 registry: same rule, disjoint from the
# 16 training decks and never collected.
RANDOM_DECK_REGISTRY = 'configs/random_deck_cells_v5.json'
PANEL_VERSION = 'selection-task137-v5'
GATE_PANEL_VERSION = 'v5'


def random_deck_tasks(root):
    """task137 evaluation decks -> selection cells (seeds assigned per round)."""
    from pvz_tasks import profile_for_deck, terrain_covered
    registry = json.loads((Path(root) / RANDOM_DECK_REGISTRY).read_text())
    training = {tuple(sorted(task['deck'])) for task in registry['tasks'] if task['split'] == 'train'}
    tasks, seen = [], set()
    for entry in registry['tasks']:
        if entry['split'] != 'eval':
            continue
        deck = tuple(entry['deck'])
        signature = tuple(sorted(deck))
        if signature in training or signature in seen:
            raise ValueError(f'random-deck evaluation set is not disjoint: {entry["task_id"]}')
        if not 6 <= len(deck) <= 10 or len(set(deck)) != len(deck):
            raise ValueError(f'invalid random-deck evaluation deck: {entry["task_id"]}')
        if not terrain_covered(entry['level'], deck):
            raise ValueError(f'random-deck evaluation deck misses terrain: {entry["task_id"]}')
        if profile_for_deck(deck).seed_slot_count != len(deck):
            raise ValueError(f'random-deck evaluation profile slot count differs: {entry["task_id"]}')
        task = dict(task_id=entry['task_id'], level=entry['level'], terrain=entry['terrain'],
                    deck=list(deck), playthrough=2, zombie_count_multiplier=1.0, seeds=[],
                    selection_group='random_deck')
        if entry['terrain'] == 'fog':
            task['fog_observation'] = 'human_v2'
        tasks.append(task)
        seen.add(signature)
    if not tasks:
        raise ValueError('random-deck registry has no evaluation decks')
    return tasks


def selection_templates(holdout_manifest=None, promotion_panel=None):
    """Adventure body + random families + score-only unseen decks, frozen at enqueue."""
    root = Path(__file__).resolve().parents[1]
    if promotion_panel is None:
        raise ValueError('selection requires the gate-v5 adventure panel (--promotion-panel)')
    document = json.loads(Path(promotion_panel).read_text())
    if document.get('panel_version') != GATE_PANEL_VERSION:
        raise ValueError(f'selection requires the registered gate-v5 panel, got '
                         f'{document.get("panel_version")!r}')
    monitor = set(document.get('monitor') or ())
    tasks = []
    for task in document['tasks']:
        if task['task_id'] in monitor:
            continue
        task = copy.deepcopy(task)
        task['seeds'] = []            # assigned per round from the selection ledger
        task['selection_group'] = 'adventure'
        tasks.append(task)
    tasks.extend(random_deck_tasks(root))
    templates = dict(schema_version=1, panel_version=PANEL_VERSION, tasks=tasks,
                     purpose='adventure levels of the gate-v5 panel on independent seeds, plus the eight '
                             'task137 v5 evaluation decks (never trained); unseen decks are score-only')
    if holdout_manifest is None:
        return templates
    from pvz_tasks import profile_for_deck, terrain_covered
    extra = json.loads(Path(holdout_manifest).read_text())['tasks']
    registry = json.loads((root / RANDOM_DECK_REGISTRY).read_text())
    training_decks = {tuple(sorted(task['deck'])) for task in registry['tasks']
                      if task['split'] == 'train'}
    names = {task['task_id'] for task in tasks}
    gate_names = {task['task_id'] for task in document['tasks']}
    gate_decks = {tuple(sorted(task['deck'])) for task in document['tasks']}
    signatures = set()
    for task in extra:
        deck = task['deck']
        signature = tuple(sorted(deck))
        if (task['task_id'] in names | gate_names or signature in gate_decks | signatures
                or task.get('selection_group') != 'unseen_deck'
                or task.get('seeds') or 'deck_random' in task or 'deck_perturb' in task
                or not 6 <= len(deck) <= 10 or len(set(deck)) != len(deck)
                or not terrain_covered(task['level'], deck)):
            raise ValueError('selection holdout must be a distinct, terrain-covered fixed card set without seeds')
        if profile_for_deck(tuple(deck)).seed_slot_count != len(deck):
            raise ValueError('selection holdout profile slot count differs from deck')
        if signature in training_decks:
            raise ValueError('selection holdout reuses a task137 training deck')
        names.add(task['task_id'])
        signatures.add(signature)
        task['score_only'] = True
        tasks.append(task)
    return templates


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_plan(templates, block, incumbent):
    """Pure planning. No engine, ledger write, score or training is performed here."""
    tasks = [copy.deepcopy(task) for task in templates['tasks']]
    if not tasks:
        raise ValueError('selection panel is empty')
    current = templates['panel_version'] == PANEL_VERSION
    budget = SELECTION_BLOCK_SIZE if current else 384
    count = 8 if current else budget // len(tasks)
    if count < 1 or block[1] - block[0] != budget:
        raise ValueError('selection tasks exceed the seed budget or block size is invalid')
    cursor = block[0]
    for task in tasks:
        games = 16 if (current and task.get('selection_group') == 'adventure'
                       and task['terrain'] == 'day') else count
        task['seeds'] = list(range(cursor, cursor + games))
        cursor += games
        if 'deck_random' in task:
            task['splits'] = dict(selection=task['seeds'])
    if cursor > block[1]:
        raise ValueError('selection tasks exceed the seed budget')
    return dict(schema_version=1, panel_version=templates['panel_version'], tasks=tasks,
                incumbent=str(incumbent), incumbent_sha256=sha(incumbent),
                seed_block=list(block), games_per_cell=count,
                games_per_task={task['task_id']: len(task['seeds']) for task in tasks},
                score_only=[task['task_id'] for task in tasks if task.get('score_only')],
                selection_holdout_tasks=[task['task_id'] for task in tasks
                                         if task.get('selection_group') == 'unseen_deck'],
                adventure_tasks=[task['task_id'] for task in tasks
                                 if task.get('selection_group') == 'adventure'],
                purpose=('adventure levels of the gate-v5 panel on independent seeds; '
                         + ('ordinary day levels take 16 games, other tasks 8; ' if current else 'equal games per task; ')
                         + 'unseen-deck cells are recorded but never scored'))


def validate_random_tasks(panel, env):
    """Freeze fresh engine-approved receipts using task93's exact sampler and filters."""
    from dataclasses import replace
    from pvz_constants import upgrade_plants
    from pvz_env import TaskSpec
    from pvz_tasks import (deck_for_level, profile_for_deck, random_deck_covered,
                           random_deck_draws)
    for task in panel['tasks']:
        if 'deck_random' not in task:
            continue
        base = deck_for_level(task['level'])
        capability = replace(profile_for_deck(base), owned_upgrade_plants=tuple(sorted(upgrade_plants())))
        options = task['deck_random']['engine_options']
        env.reset(deck=base, task=TaskSpec(level=task['level'], seed=task['seeds'][0],
                                          playthrough=2, profile=capability))
        actual = env.deck_options()
        recommended = dict(actual['not_recommended'])
        pool = sorted(card for card in env.deck_pool()['crazy_dave_pool']
                      if 0 <= card < 40 and not recommended.get(card, 1))
        if pool != options['available_plants'] or actual['not_recommended'] != options['not_recommended']:
            raise ValueError('selection random family engine pool/warnings changed')
        for seed in task['seeds']:
            env.reset(deck=base, task=TaskSpec(level=task['level'], seed=seed, playthrough=2, profile=capability))
            for attempt, deck in random_deck_draws(task, seed):
                if not random_deck_covered(task, deck) or env.deck_warnings(deck)['warnings']:
                    continue
                try:
                    env.reset(deck=deck, task=TaskSpec(level=task['level'], seed=seed,
                                                     playthrough=2, profile=profile_for_deck(deck)))
                except RuntimeError:
                    continue
                if env.deck_warnings(deck)['warnings']:
                    continue
                task['deck_random']['validated_decks'][str(seed)] = dict(attempt=attempt, deck=list(deck))
                break
            else:
                raise ValueError(f'no engine-approved random deck: {task["task_id"]}/{seed}')
    return panel
