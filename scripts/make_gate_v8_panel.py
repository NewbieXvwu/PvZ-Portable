#!/usr/bin/env python3
"""task222: rebuild configs/gate_v8_panel.json from the user's own deck picks.

The user's in-game deck choices are the authoritative benchmark.  Every
recording under the human frames directory contributes one task for its level
(one task per distinct deck, so two L29 recordings become two pool tasks).
Levels without a user recording use the first task208 good deck
(scripts/pvz_tasks.GOOD_DECKS), never a scoreboard deck and never filtered by
teacher win rate.  Re-running this script after new recordings is the whole
update path; no per-level hand editing.

Only structural rules are enforced here: 6..10 distinct cards, every card in
the legal 0..47 range, and terrain coverage (lily pad on pool/fog, flower pot on
roof).  Teacher acceptance is deliberately not an inclusion filter (user
2026-11-01: teacher defects are to be fixed, not avoided).
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'python')]

import pvz_constants  # noqa: E402
import pvz_tasks  # noqa: E402
from deck_rules import region_for_level  # noqa: E402
from pvz_promotion import (INITIAL_MULTIPLIERS, LOW_LEVELS, MONITOR_SEEDS_PER_TASK,  # noqa: E402
                           MULTIPLIERS, SEEDS_PER_TASK)

DEFAULT_USER_ROOT = Path.home() / 'PvZ-search/artifacts/student/frames_v2/human'
DEFAULT_V5_PANEL = ROOT / 'configs/gate_v5_panel.json'
DEFAULT_OUT = ROOT / 'configs/gate_v8_panel.json'
ZONE_ORDER = ('day', 'night', 'pool', 'fog', 'roof')
MONITOR_TASKS = ('day_l7_eval_d7p_holdout1', 'day_l9_eval_d7p_holdout2')
# Mirror the v5 seed assignment: the first 18 adventure tasks take 46 pairs and
# the rest 45, so the first sequential look is 12 pairs on every task.  The
# first-look adventure count stays close to the 456-pair v5 gate.
BONUS_PAIRS = 18


def _valid_level(level: int) -> str:
    """Zone for a selectable adventure level (raises for conveyor/special)."""
    return region_for_level(level)


def _reading(path: Path) -> dict:
    document = json.loads(path.read_text())
    return document.get('task', document)


def _deck(value):
    if not isinstance(value, list) or any(type(card) is not int for card in value):
        return None
    return tuple(value)


def _check_deck(level: int, deck, where: str) -> None:
    if not 6 <= len(deck) <= 10:
        raise ValueError(f'{where}: deck length {len(deck)} outside 6..10')
    if len(set(deck)) != len(deck):
        raise ValueError(f'{where}: duplicate card in deck {deck}')
    if any(not 0 <= card < 48 for card in deck):
        raise ValueError(f'{where}: card outside 0..47 in deck {deck}')
    if not pvz_tasks.terrain_covered(level, deck):
        raise ValueError(f'{where}: deck {deck} does not cover level {level} terrain '
                         f'({pvz_constants.background_for_level(level)})')
    profile = pvz_tasks.profile_for_deck(deck)
    if profile.seed_slot_count != len(deck):
        raise ValueError(f'{where}: profile slot count differs from deck length')


def user_recordings(root: Path) -> list[dict]:
    """All user_l* recordings with a level and a deck, one row per recording."""
    rows = []
    if not root.is_dir():
        return rows
    for meta in sorted(root.glob('**/meta.json')):
        if not any(part.startswith('user_l') for part in meta.parts):
            continue
        task = _reading(meta)
        level, deck = task.get('level'), _deck(task.get('deck'))
        if type(level) is not int or deck is None:
            continue
        try:
            zone = _valid_level(level)
        except ValueError as error:
            raise ValueError(f'{meta}: {error}') from error
        _check_deck(level, deck, str(meta))
        rows.append(dict(level=level, zone=zone, deck=deck, seed=task.get('seed'),
                         path=str(meta.relative_to(root))))
    return rows


def warm_start(path: Path | None) -> dict[int, dict]:
    """Existing staircase settings keyed by level; keeps multipliers, not outcomes."""
    if path is None:
        return {}
    document = json.loads(path.read_text())
    result = {}
    for key, setting in (document.get('gate_panel') or document).items():
        if not isinstance(setting, dict):
            continue
        match = re.search(r'_l(\d+)_', key)
        level = int(key) if str(key).isdigit() else int(match.group(1)) if match else None
        if level is None:
            continue
        multiplier = setting.get('multiplier')
        if multiplier in MULTIPLIERS:
            result[level] = dict(multiplier=float(multiplier),
                                 low_win=bool(setting.get('low_win')))
    return result


def calibrated_from_summary(path: Path, base: dict[int, float]) -> tuple[dict[int, float], set[int]]:
    """One standard staircase step from an incumbent baseline summary.

    Rates are aggregated per level (a level with two recorded decks gets one
    multiplier).  >=90% raises a step, <=10% lowers one, otherwise unchanged.
    Low-win levels (<=20% or roof) switch to the wave endpoint, exactly like
    next_panel does after a gate round.
    """
    document = json.loads(path.read_text())
    totals: dict[int, list[int]] = {}
    zones: dict[int, str] = {}
    for task in document.get('tasks', []):
        if task.get('zone') == 'heldout':
            continue
        level = int(task['level'])
        bucket = totals.setdefault(level, [0, 0])
        bucket[0] += int(task['wins'])
        bucket[1] += int(task['games'])
        zones[level] = task['zone']
    result = dict(base)
    low = set()
    for level, (wins, games) in totals.items():
        rate = wins / games
        index = MULTIPLIERS.index(result.get(level, 1.))
        direction = 1 if rate >= .9 else -1 if rate <= .1 else 0
        result[level] = MULTIPLIERS[max(0, min(len(MULTIPLIERS) - 1, index + direction))]
        if rate <= .2 or zones[level] == 'roof':
            low.add(level)
    return result, low


def monitor_definitions(path: Path) -> list[dict]:
    source = json.loads(path.read_text())
    table = {task['task_id']: task for task in source['tasks']}
    missing = [name for name in MONITOR_TASKS if name not in table]
    if missing:
        raise ValueError(f'monitor cells missing from {path}: {missing}')
    result = []
    for name in MONITOR_TASKS:
        task = copy.deepcopy(table[name])
        task['pairs'] = MONITOR_SEEDS_PER_TASK
        task['monitor_only'] = True
        result.append(task)
    return result


def build(user_root: Path, v5_panel: Path, warm_state: Path | None,
          calibration_summary: Path | None = None, base_panel: Path | None = None) -> dict:
    recordings = user_recordings(user_root)
    warm = warm_start(warm_state)
    by_level: dict[int, list[dict]] = {}
    for row in recordings:
        by_level.setdefault(row['level'], []).append(row)

    levels = set(pvz_tasks.GOOD_DECKS) | set(by_level)
    entries = []
    for level in sorted(levels):
        zone = _valid_level(level)
        user = by_level.get(level, [])
        distinct, seen = [], set()
        for row in user:
            if row['deck'] not in seen:
                seen.add(row['deck'])
                distinct.append(row)
        if distinct:
            for index, row in enumerate(distinct, 1):
                entries.append(dict(level=level, zone=zone, deck=row['deck'], source='user',
                                    source_index=index, source_count=len(distinct),
                                    recording=row))
        else:
            deck = pvz_tasks.GOOD_DECKS[level][0]
            _check_deck(level, deck, f'task208 L{level}')
            entries.append(dict(level=level, zone=zone, deck=deck, source='task208',
                                source_index=1, source_count=1, recording=None))

    entries.sort(key=lambda row: (ZONE_ORDER.index(row['zone']), row['level'], row['source_index']))
    tasks, groups, qualification = [], {}, {}
    for index, row in enumerate(entries):
        if row['source'] == 'user':
            suffix = 'user' if row['source_count'] == 1 else f'user{row["source_index"]}'
        else:
            suffix = 'good'
        task_id = f'task222_l{row["level"]:02d}_{row["zone"]}_{suffix}'
        if task_id in groups:
            raise ValueError(f'duplicate task id {task_id}')
        pairs = SEEDS_PER_TASK + 1 if index < BONUS_PAIRS else SEEDS_PER_TASK
        task = dict(task_id=task_id, level=row['level'], deck=list(row['deck']),
                    zombie_count_multiplier=1.0, terrain=row['zone'], wave_cap=None,
                    playthrough=2, preplanted=[], pairs=pairs, deck_source=row['source'])
        tasks.append(task)
        groups[task_id] = row['zone']
        qualification[task_id] = dict(
            zone=row['zone'], level=row['level'], deck=list(row['deck']), pairs=pairs,
            deck_source=row['source'], user_deck=row['source'] == 'user',
            task208_rank=row['source_index'] if row['source'] == 'task208' else None,
            teacher_evidence=('user recording, not filtered by teacher win rate'
                              if row['source'] == 'user'
                              else 'MEMORY/docs/task208-good-decks.md first (starred) deck'),
            **({'recording_seed': row['recording']['seed'],
                'recording': row['recording']['path']} if row['source'] == 'user' else {}))

    monitors = monitor_definitions(v5_panel)
    for task in monitors:
        if task['task_id'] in groups:
            raise ValueError(f'monitor task collides with adventure task: {task["task_id"]}')
        groups[task['task_id']] = 'heldout'
        qualification[task['task_id']] = dict(zone='heldout', level=task['level'],
                                              deck=list(task.get('deck') or ()),
                                              pairs=MONITOR_SEEDS_PER_TASK,
                                              deck_source='registered heldout', user_deck=False,
                                              monitor_only=True)
        tasks.append(task)

    initial, initial_low = {}, []
    base_panel_document = json.loads(base_panel.read_text()) if base_panel else None
    base_low = set((base_panel_document or {}).get('initial_low_win') or ())
    known = set(int(level) for level in (base_panel_document or {}).get('initial_multipliers', {}))
    for entry in entries:
        level = entry['level']
        setting = warm.get(level)
        if base_panel_document is not None:
            multiplier = float(base_panel_document.get('initial_multipliers', {}).get(
                str(level), INITIAL_MULTIPLIERS.get(level, 1.0)))
        else:
            multiplier = setting['multiplier'] if setting else INITIAL_MULTIPLIERS.get(level, 1.0)
        if multiplier not in MULTIPLIERS:
            raise ValueError(f'L{level}: multiplier {multiplier} outside {MULTIPLIERS}')
        initial[str(level)] = multiplier
        if base_panel_document is not None and level in known:
            low = level in base_low
        else:
            low = entry['zone'] == 'roof' or level in LOW_LEVELS or bool(setting and setting['low_win'])
        if low:
            initial_low.append(level)
    initial_low = sorted(set(initial_low))
    if calibration_summary is not None:
        base = {int(level): value for level, value in initial.items()}
        adjusted, low = calibrated_from_summary(calibration_summary, base)
        initial = {str(level): value for level, value in adjusted.items()}
        initial_low = sorted(low)

    zone_counts = {zone: sum(1 for row in entries if row['zone'] == zone) for zone in ZONE_ORDER}
    zone_counts['heldout'] = len(monitors)
    panel = dict(
        schema_version=1, panel_version='v8', status='registered',
        purpose=('task222 gate v8: the user\'s own in-game deck picks are the benchmark; '
                 'task208 first good decks fill levels without a user recording. Every terrain '
                 'zone is represented; no deck is filtered by teacher win rate.'),
        rule=('one task per user recording (distinct deck), otherwise the level\'s first task208 '
              'good deck; structural checks only (6-10 distinct cards, terrain coverage); '
              'multipliers come from the existing staircase, never hand-tuned per level'),
        sources=dict(**{
            'scripts/pvz_tasks.py': 'GOOD_DECKS / USER_DECKS deck source (task208)',
            'MEMORY/docs/task208-good-decks.md': 'task208 per-level deck and teacher evidence',
            'user recordings': 'desktop ~/PvZ-search/artifacts/student/frames_v2/human/user_l* '
                               '(meta.json task.level/task.deck)',
            'configs/gate_v5_panel.json': 'registered heldout monitor definitions (decks unchanged)',
        }),
        groups=groups, monitor=list(MONITOR_TASKS), zone_counts=zone_counts,
        seeds_per_task=SEEDS_PER_TASK, monitor_seeds_per_task=MONITOR_SEEDS_PER_TASK,
        bonus_pairs=BONUS_PAIRS, first_look_pairs=sum(
            math.ceil((SEEDS_PER_TASK + (1 if index < BONUS_PAIRS else 0)) * .25)
            for index in range(len(entries))) + MONITOR_SEEDS_PER_TASK * len(monitors),
        initial_multipliers=initial, initial_low_win=initial_low,
        initial_multiplier_source=('staircase-calibrated on the task222 incumbent baseline; '
                                   'see MEMORY/docs/task222-gate-v8.md'),
        user_recordings=[dict(level=row['level'], deck=list(row['deck']), seed=row['seed'],
                              path=row['path']) for row in recordings],
        excluded=[dict(level=level, reason='conveyor or special level: no deck choice')
                  for level in (5, 10, 15, 20, 25, 30, 35, 40, 45, 50)],
        qualification=qualification, tasks=tasks)
    return panel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--user-root', type=Path, default=DEFAULT_USER_ROOT)
    parser.add_argument('--v5-panel', type=Path, default=DEFAULT_V5_PANEL)
    parser.add_argument('--warm-state', type=Path,
                        help='curriculum_state.json or a {task: setting} snapshot for initial multipliers')
    parser.add_argument('--calibrate-from', type=Path,
                        help='incumbent baseline summary.json; apply one standard staircase step')
    parser.add_argument('--base-panel', type=Path,
                        help='use this panel\'s initial_multipliers as the staircase base')
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--check', action='store_true', help='fail unless --out already matches')
    args = parser.parse_args()
    panel = build(args.user_root, args.v5_panel, args.warm_state, args.calibrate_from,
                  args.base_panel)
    text = json.dumps(panel, indent=1, ensure_ascii=False) + '\n'
    if args.check:
        current = args.out.read_text() if args.out.exists() else ''
        if current != text:
            print(f'{args.out} is stale; rerun without --check', file=sys.stderr)
            raise SystemExit(1)
        print(f'{args.out}: up to date ({len(panel["tasks"])} tasks)')
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(f'{args.out}: {len(panel["tasks"])} tasks, zones {panel["zone_counts"]}, '
          f'first look {panel["first_look_pairs"]} pairs')


if __name__ == '__main__':
    main()
