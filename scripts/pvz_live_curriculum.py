"""task193: metadata-only live GPT deck pool and grouped demonstration handoff.

A round reads the existing import indexes/meta once (a few MB, normally seconds
on local WSL storage). No trajectory decoding, replay, scan, or derived manifest.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random

from import_llm_run import family_entry
from make_random_deck_cells import LADDER, REGISTRY, heldout_signatures
from pvz_tasks import terrain_covered

DEMO_LADDER = [90, 80, 70, 60, 50, 40, 30, 20, 10, 1]


def signature(deck):
    return tuple(sorted(deck))


def group_id(level, deck):
    digest = hashlib.sha256(repr(signature(deck)).encode()).hexdigest()[:12]
    return f'gpt193_l{level}_{digest}'


def imported_wins(directory, archive_root):
    rows = {}
    for name in ('merged_progress.json',):
        path = directory / name
        if path.exists():
            for row in json.loads(path.read_text()).get('archives', []):
                key = row.get('archive_sha256') or row.get('archive')
                if row.get('archive'):
                    rows[key] = row
    result = []
    for key, row in sorted(rows.items()):
        if row.get('verify') != '通过' or row.get('result') != 1 or row.get('manual_patch'):
            continue
        path = Path(row['archive'])
        path = path if path.is_absolute() else archive_root / path
        try:
            meta = json.loads((path / 'meta.json').read_text())
        except FileNotFoundError:
            continue  # The import index can retain a previously pruned archive.
        if meta.get('manual_patch') or not meta.get('verified') or meta.get('outcome', {}).get('result') != 1:
            continue
        task = family_entry(meta['task']['id'] + '_live193', meta, None)
        result.append(dict(identity=key, archive=str(path), task=task,
                           frames=int(meta.get('frame_count') or row.get('frames') or 0)))
    return result


def cold_teacher_rates(state, tasks):
    """Only actual cold-start all-teacher outcomes, never beta=0/.5 student/mix wins."""
    counts = {}
    for entry in state.get('log', []):
        for group in entry.get('groups', []):
            if group.get('demo_archive') or group.get('multiplier') != 1 or group.get('beta') != 1:
                continue
            task = group.get('task_definition') or tasks.get(group['task_id'], {})
            if not task.get('level') or task.get('demo_start'):
                continue
            if 'wins' in group and group.get('valid_games', group.get('games', 0)):
                wins, games = counts.setdefault(task['level'], [0, 0])
                counts[task['level']] = [wins + group['wins'], games + group.get('valid_games', group['games'])]
    return {level: wins / games for level, (wins, games) in counts.items() if games}


def refresh(state, tasks, import_dir, archive_root, heldout_panels, cold_rates):
    """Return this round's ephemeral catalog; persist only scheduling/adaptation state."""
    registry = json.loads(REGISTRY.read_text())
    heldout = heldout_signatures(registry, heldout_panels)
    wins = imported_wins(import_dir, archive_root)
    legacy, legacy_sha = {}, {}
    for base, cell in state['cells'].items():
        cell.pop('live193_grouped', None)
        if cell.get('archive') and not cell.get('live193'):
            path = Path(cell['archive'])
            legacy.setdefault(str(path.resolve()), []).append(base)
            if path.parent.name == 'sha256':
                legacy_sha.setdefault(path.name, []).append(base)
    pools, demos = {}, {}
    for task in tasks.values():
        if task.get('random_deck_cell') and not task.get('live193') and task.get('deck'):
            deck = signature(task['deck'])
            if deck not in heldout:
                pools.setdefault(task['level'], {}).setdefault(deck, dict(task, zombie_count_multiplier=1.))
    for entry in wins:
        task = entry['task']
        level, deck = task['level'], signature(task['deck'])
        if deck in heldout:
            continue
        # Whole games use original 1x wins only; demonstration starts reproduce their own environment.
        if (deck and task['zombie_count_multiplier'] == 1 and not task.get('wave_cap')
                and not task.get('preplanted') and not task.get('conveyor_start')
                and terrain_covered(level, deck)):
            pools.setdefault(level, {}).setdefault(deck, task)
        old = sorted(set(legacy.get(str(Path(entry['archive']).resolve()), []) +
                         legacy_sha.get(entry['identity'], [])))
        if entry['frames'] > 1 and (old or cold_rates.get(level, 1.) < .3):
            member = dict(entry, legacy_base=old[0] if old else None)
            demos.setdefault(group_id(level, deck), []).append(member)
            for base in old:
                state['cells'][base]['live193_grouped'] = True
    catalog = {}
    for level, options in sorted(pools.items()):
        base = f'rd193_l{level}_train'
        catalog[base] = dict(kind='random', options=[options[key] for key in sorted(options)])
        state['cells'].setdefault(base, dict(family='gpt_live', level=level, random_deck_cell=True,
            live193=True, last_round=0, mult_index=LADDER.index(1.), ladder=list(LADDER), windows={}, seeds=[]))
    for base, members in sorted(demos.items()):
        catalog[base] = dict(kind='demo', options=sorted(members, key=lambda e: e['identity']))
        state['cells'].setdefault(base, dict(family='gpt_live', level=members[0]['task']['level'],
            live193=True, archive=members[0]['archive'], last_round=0, mult_index=0,
            ladder=list(DEMO_LADDER), windows={}, seeds=[], archive_cursor=0))
    return catalog


def materialize(base, cell, catalog, round_index, state):
    entry = catalog[base]
    if entry['kind'] == 'random':
        rng = random.Random(f'task193:{round_index}:{base}')
        task = dict(rng.choice(entry['options']), task_id=base)
        task.update(zombie_count_multiplier=cell['ladder'][cell['mult_index']], live193=True)
        return task, {}
    member = entry['options'][cell.get('archive_cursor', 0) % len(entry['options'])]
    legacy = member.get('legacy_base')
    if legacy:
        old = state['cells'][legacy]
        decision = old['ladder'][old['mult_index']]
    else:
        fraction = cell['ladder'][cell['mult_index']] / 100
        decision = max(1, min(member['frames'] - 1, round((member['frames'] - 1) * fraction)))
    return dict(member['task']), dict(demo_archive=member['archive'], demo_decision=int(decision),
                                      adaptation_base=legacy or base,
                                      adaptation_value=decision if legacy else cell['ladder'][cell['mult_index']])
