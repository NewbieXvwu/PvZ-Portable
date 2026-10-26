"""Per-round selection plans from incumbent-only windows and a separate seed ledger."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

SELECTION_POOLS = ((130_000_000, 140_000_000),)
SELECTION_BLOCK_SIZE = 384
SELECTION_LEDGER = Path.home() / 'pvz-logs/selection_seed_ledger.json'
RANDOM_TASKS = ('day_l7_train_rand', 'night_l12_train_rand', 'pool_l21_train_rand')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def current_cells(state, incumbent, runs):
    """Screen the current policy's beta-zero suffix; never attribute older windows to it."""
    incumbent = str(Path(incumbent).resolve())
    sources, cells, excluded = {}, [], []
    for base, cell in sorted(state['cells'].items()):
        if cell.get('archive') or cell.get('ladder') and cell.get('kind') == 'demo_start':
            excluded.append(dict(base=base, reason='demonstration decision axis is not a multiplier cell'))
            continue
        for multiplier, bucket in sorted(cell.get('windows', {}).items(), key=lambda item: float(item[0])):
            outcomes, last = [], None
            for rnd in state.get('log', []):
                for group in rnd['groups']:
                    if (group['base'] != base or float(group['multiplier']) != float(multiplier)
                            or group.get('beta', 0) != 0 or group.get('student') != incumbent):
                        continue
                    folder = Path(runs) / f'{group["label"]}_{group["games"]}'
                    config, games = folder / 'config.json', folder / 'games.jsonl'
                    settings = json.loads(config.read_text())
                    if (str(Path(settings['rollout_policy']).resolve()) != incumbent
                            or float(settings['dagger_beta']) != 0 or settings['task_id'] != group['task_id']):
                        raise ValueError(f'selection window policy/config mismatch: {config}')
                    rows = [json.loads(line) for line in games.read_text().splitlines() if line.strip()]
                    if len(rows) != group['games'] or any(type(row.get('won')) is not bool
                                                        or row.get('terminal') is not True for row in rows):
                        raise ValueError(f'incomplete selection window group: {games}')
                    outcomes.extend(int(row['won']) for row in rows)
                    last = group
                    sources[str(config)], sources[str(games)] = sha(config), sha(games)
            recent = bucket.get('recent', [])
            outcomes = outcomes[-len(recent):] if recent else []
            if outcomes and outcomes != recent[-len(outcomes):]:
                # The window holds games the log cannot attribute (e.g. groups from before a restart);
                # a mixed window is not evidence about the incumbent, so this cell sits out the panel.
                excluded.append(dict(base=base, multiplier=float(multiplier),
                                     reason='current-policy records do not match window suffix'))
                continue
            if outcomes and .3 <= sum(outcomes) / len(outcomes) <= .8:
                cells.append(dict(base=base, multiplier=float(multiplier), wins=sum(outcomes),
                                  games=len(outcomes), rate=sum(outcomes) / len(outcomes),
                                  task_id=last['task_id'], manifest=last['manifest']))
    return cells, sources, excluded


def make_plan(state, incumbent, runs, templates, source_documents, block):
    """Pure planning. No engine, ledger write, score or training is performed here."""
    cells, sources, excluded = current_cells(state, incumbent, runs)
    count = SELECTION_BLOCK_SIZE // (len(cells) + len(RANDOM_TASKS))
    if count < 1 or block[1] - block[0] != SELECTION_BLOCK_SIZE:
        raise ValueError('selection cells exceed the 384-game budget or block size is invalid')
    tasks = []
    for cell in cells:
        document = source_documents[cell['manifest']]
        task = copy.deepcopy(next(task for task in document['tasks'] if task['task_id'] == cell['task_id']))
        if task.get('demo_start'):
            raise ValueError('demonstration start cannot be replayed as a multiplier selection cell')
        tasks.append(task)
    table = {task['task_id']: task for task in templates['tasks']}
    for name in RANDOM_TASKS:
        task = copy.deepcopy(table[name])
        task['deck_random']['validated_decks'] = {}
        tasks.append(task)
    for index, task in enumerate(tasks):
        start = block[0] + index * count
        task['seeds'] = list(range(start, start + count))
        if 'deck_random' in task:
            task['splits'] = dict(selection=task['seeds'])
    return dict(schema_version=1, panel_version='selection-task92c-v2-refresh',
                tasks=tasks, cells=cells, window_sources_sha256=sources, excluded=excluded,
                incumbent=str(incumbent), incumbent_sha256=sha(incumbent),
                seed_block=list(block), games_per_cell=count,
                purpose='30%-80% current-policy multiplier windows plus three required random families; equal games')


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
