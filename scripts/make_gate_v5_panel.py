#!/usr/bin/env python3
"""task128: freeze the gate-v5 source panel from the adventure scoreboard.

Gate v5 stops scoring synthetic curriculum cells and measures the adventure
tasks the game actually plays: original 1.0x difficulty with the scoreboard
decks (`configs/adventure_scoreboard.json`, frozen by task103). The level set is
the subset where the incumbent reference can win at all -- "all-loss cells have
no discriminative power" -- measured as

    r11 student wins >= 1 on the 32-seed scoreboard  OR
    teacher wins >= 1 in the task115 weak-level arm.

Sources (SHA-pinned below, so the panel is reproducible from git):

- `configs/adventure_scoreboard.json`   task definitions, 45 levels, 1.0x.
- `MEMORY/docs/adventure-scoreboard-r11-32.json`  per-level r11 student wins
  (45 levels x 32 seeds, Mac engine 6f7d2c3f, task103).
- `MEMORY/docs/task115-teacher-arm-weak-levels.md`  teacher arm, 16 seeds per
  weak level. The roof block (L41-44, L46-49) is only reported as an aggregate
  "0-1/16" and is therefore *not* attributable per level; those levels are
  recorded as unqualified, which is also what the rule above gives them once
  the aggregate is read as "the teacher loses there too" (task115 files them
  under `老师也输`). L50 (Boss) is 0/16 teacher and 0/32 student -> excluded.

The two heldout cells are carried over field-for-field from the registered v3
source panel (`artifacts/task73c/panel_v3_source.json`); v5 keeps running them
for monitoring only, never for the veto or the overall rate.

Run:  python3 scripts/make_gate_v5_panel.py [--check]
`--check` regenerates in memory and fails when `configs/gate_v5_panel.json`
is stale, without writing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]

SCOREBOARD = 'configs/adventure_scoreboard.json'
STUDENT_RESULTS = 'MEMORY/docs/adventure-scoreboard-r11-32.json'
TEACHER_DOC = 'MEMORY/docs/task115-teacher-arm-weak-levels.md'
OUTPUT = 'configs/gate_v5_panel.json'
SCOREBOARD_SHA256 = '5213bf9fef5a22f676ba4a9f90cf9d73c6a0d36d5393b6efe07392e9b07946cc'
STUDENT_RESULTS_SHA256 = 'de0c68c16c5f7582c8962916bc2334cd6f2e8e5877f5bb4f0ab0c48ae641b288'

# task115 teacher arm, 16 seeds per measured weak level (None = not measured).
TEACHER_WINS = {19: 16, 23: 15, 24: 16, 26: 13, 27: 9, 28: 16, 29: 13, 30: 1,
                32: 13, 33: 5, 34: 1, 37: 16, 39: 3, 40: 4, 50: 0}
TEACHER_GAMES = 16
# task115 reports the roof block only in aggregate ("0-1/16" across L41-44, L46-49),
# so no per-level teacher count exists. Kept as an explicit reason, not a guess.
UNMEASURED_BLOCKS = {41: 'task115 reports the roof block 41-44/46-49 only in aggregate (0-1/16)',
                     42: 'task115 reports the roof block 41-44/46-49 only in aggregate (0-1/16)',
                     43: 'task115 reports the roof block 41-44/46-49 only in aggregate (0-1/16)',
                     44: 'task115 reports the roof block 41-44/46-49 only in aggregate (0-1/16)',
                     46: 'task115 reports the roof block 41-44/46-49 only in aggregate (0-1/16)',
                     47: 'task115 reports the roof block 41-44/46-49 only in aggregate (0-1/16)',
                     48: 'task115 reports the roof block 41-44/46-49 only in aggregate (0-1/16)',
                     49: 'task115 reports the roof block 41-44/46-49 only in aggregate (0-1/16)'}

# Registered v3 heldout cells, copied field-for-field (artifacts/task73c/panel_v3_source.json).
HOLDOUT_TASKS = (
    dict(task_id='day_l7_eval_d7p_holdout1', terrain='day', level=7,
         deck=[0, 1, 2, 27, 5, 32, 3], zombie_count_multiplier=1.0, wave_cap=None,
         sun_start=50, preplanted=[], playthrough=2, split='heldout_deck',
         holdout_semantics='unordered plant set excluded from every perturbation training family'),
    dict(task_id='day_l9_eval_d7p_holdout2', terrain='day', level=9,
         deck=[7, 2, 5, 38, 4, 1, 3], zombie_count_multiplier=1.0, wave_cap=None,
         sun_start=50, preplanted=[], playthrough=2, split='heldout_deck',
         holdout_semantics='unordered plant set excluded from every perturbation training family'),
)

ZONE_BY_BACKGROUND = {'BACKGROUND_1_DAY': 'day', 'BACKGROUND_2_NIGHT': 'night',
                      'BACKGROUND_3_POOL': 'pool', 'BACKGROUND_4_FOG': 'fog',
                      'BACKGROUND_5_ROOF': 'roof', 'BACKGROUND_6_BOSS': 'roof'}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def zone_of(level: int) -> str:
    """Board::PickBackground zone, resolved from source (AGENTS §8), not level//10."""
    from pvz_constants import background_for_level
    return ZONE_BY_BACKGROUND[background_for_level(level)]


def build() -> dict:
    scoreboard = ROOT / SCOREBOARD
    results = ROOT / STUDENT_RESULTS
    if sha(scoreboard) != SCOREBOARD_SHA256:
        raise SystemExit(f'{SCOREBOARD} changed; re-freeze the v5 panel deliberately')
    if sha(results) != STUDENT_RESULTS_SHA256:
        raise SystemExit(f'{STUDENT_RESULTS} changed; re-freeze the v5 panel deliberately')
    definitions = json.loads(scoreboard.read_text())['tasks']
    student = {level['level']: level.get('student') or {}
               for level in json.loads(results.read_text())['levels']}
    from pvz_tasks import profile_for_deck, terrain_covered

    tasks, groups, qualification, excluded = [], {}, {}, []
    for definition in definitions:
        level = int(definition['level'])
        wins, games = int(student.get(level, {}).get('wins') or 0), int(student.get(level, {}).get('games') or 0)
        teacher = TEACHER_WINS.get(level)
        zone = zone_of(level)
        record = dict(zone=zone, student_wins=wins, student_games=games,
                      teacher_wins=teacher, teacher_games=TEACHER_GAMES if teacher is not None else None)
        if level in UNMEASURED_BLOCKS:
            record['note'] = UNMEASURED_BLOCKS[level]
        if wins < 1 and not (teacher and teacher >= 1):
            excluded.append(dict(level=level, task_id=definition['task_id'], **record,
                                 reason='no recorded win for the r11 student or the teacher'))
            continue
        deck = list(definition['deck'])
        # An empty deck is a native conveyor level: the engine deals its own belt,
        # so there is no task deck to check against the terrain (AGENTS §8).
        if deck and not terrain_covered(level, deck):
            raise SystemExit(f'scoreboard deck does not cover level {level} terrain (AGENTS §8)')
        if deck and profile_for_deck(tuple(deck)).seed_slot_count != len(deck):
            raise SystemExit(f'profile slot count differs from the deck at level {level}')
        task = dict(task_id=definition['task_id'], terrain=zone, level=level, deck=deck,
                    zombie_count_multiplier=float(definition['zombie_count_multiplier']),
                    wave_cap=None, playthrough=2, preplanted=[])
        if float(task['zombie_count_multiplier']) != 1.0:
            raise SystemExit(f'promotion needs original difficulty, level {level} is '
                             f'{task["zombie_count_multiplier"]}')
        tasks.append(task)
        groups[task['task_id']] = zone
        qualification[task['task_id']] = record

    tasks.sort(key=lambda task: task['level'])
    for task in HOLDOUT_TASKS:
        task = dict(task)
        tasks.append(task)
        groups[task['task_id']] = 'heldout'
        qualification[task['task_id']] = dict(zone='heldout', monitor_only=True)
    zones = sorted({groups[task['task_id']] for task in tasks})
    if len(tasks) != 38 or zones != ['day', 'fog', 'heldout', 'night', 'pool']:
        raise SystemExit(f'unexpected v5 panel: {len(tasks)} tasks, zones {zones}')
    counts = {zone: sum(1 for task in tasks if groups[task['task_id']] == zone) for zone in zones}
    if counts != {'day': 9, 'fog': 9, 'heldout': 2, 'night': 9, 'pool': 9}:
        raise SystemExit(f'unexpected zone counts: {counts}')
    return dict(schema_version=1, panel_version='v5',
                purpose='task128 gate-v5 source panel: the adventure levels the game actually plays '
                        '(original 1.0x, scoreboard decks) grouped by Board::PickBackground zone, plus the '
                        'two registered heldout cells for monitoring only. Seeds are assigned per round from '
                        'the out-of-git gate seed ledger; the seed lists below are never used.',
                rule='include a level when the r11 student won >= 1 of its 32 scoreboard seeds or the teacher '
                     'won >= 1 of its 16 task115 seeds; all-loss cells have no discriminative power',
                sources={SCOREBOARD: SCOREBOARD_SHA256, STUDENT_RESULTS: STUDENT_RESULTS_SHA256,
                         TEACHER_DOC: sha(ROOT / TEACHER_DOC)},
                groups=groups, monitor=[task['task_id'] for task in HOLDOUT_TASKS],
                zone_counts=counts, qualification=qualification, excluded=excluded, tasks=tasks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='verify the committed panel is current')
    args = parser.parse_args()
    document = build()
    text = json.dumps(document, indent=1) + '\n'
    target = ROOT / OUTPUT
    if args.check:
        if not target.exists() or target.read_text() != text:
            raise SystemExit(f'{OUTPUT} is stale; rerun without --check')
        print(f'{OUTPUT} is current: {len(document["tasks"])} tasks, {document["zone_counts"]}')
        return
    target.write_text(text)
    print(f'wrote {OUTPUT}: {len(document["tasks"])} tasks, {document["zone_counts"]}, '
          f'{len(document["excluded"])} excluded levels')


if __name__ == '__main__':
    main()
