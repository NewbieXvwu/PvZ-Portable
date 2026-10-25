#!/usr/bin/env python3
"""task70: fog-zone user-deck family under the human_v2 fog observation.

Why this family exists
----------------------
task66 recalibrated fog visibility by rendered opacity (tau=0.925, DESIGN section 14.25).
The user's annotations showed the old "cell value > 0" (human) rule hides about half of
the zombies a human can see, so the student observed less than the teacher. The existing
`fog_user_l31_snow` family (task68) is trained under `human`; per AGENTS.md section 8 rule 4
the observation change is a new family, so this module defines `fog_user_v2`: same deck,
level and multiplier ladder as task68, `fog_observation: human_v2`, and a fresh seed block.

Seeds
-----
Training `49_000_000 + 8192` and evaluation `49_100_000 + 32` are new blocks, disjoint from
every preregistered block: the legacy multiplier bands (5M + 700k bands), the task45 fog
blocks (39M-42M), the task68 fog_user block (43M-43.7M), the roof blocks (44M/45M/46M and
the 48M multiplier band) and the demo / task67 seed ranges (20261031-34, 20261101-32).
`make_multiplier_tasks` derives the six multiplier variants from 49_200_000 + 100_000 x band.

The manifest is local only (artifacts/ is gitignored). Regenerate it before regenerating
`artifacts/stage4/multiplier_panel.json`:

  python3 scripts/make_fog_user_v2_panel.py
  python3 scripts/make_multiplier_tasks.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]

from zone_panel import Zone, check_decks, task_document  # noqa: E402

LEVEL = 31
# task68 panel order (unchanged): sun-shroom, puff-shroom, fume-shroom, lily pad, tall nut,
# squash, tangle kelp, cherry bomb, wallnut, snow pea. Ten cards = SEEDBANK_MAX; no upgrade
# plant, so no owned_upgrade_plants entry is needed (AGENTS.md section 8).
USER_DECK = (9, 8, 10, 16, 23, 17, 19, 2, 3, 5)

TRAIN_SEED_BASE = 49_000_000
TRAIN_SEEDS = 8192
EVAL_SEED_BASE = 49_100_000
EVAL_SEEDS = 32

ZONE = Zone(
    key='fog_user_v2', task_prefix='fog_user_v2_l', first_level=LEVEL, last_level=LEVEL,
    terrain='fog', default_out=ROOT / 'artifacts/task70/fog_user_v2_panel.json',
    purpose='task70: fog zone user-deck family under the human_v2 fog observation (L31)',
    special_notes={LEVEL: 'task68 user deck; fog_observation human_v2 (task66 tau=0.925)'},
)


def tasks() -> list[dict]:
    """Base training task + the evaluation sibling, both with the task68 deck under human_v2."""
    problems = check_decks(ZONE, LEVEL, {'fog_user_v2': USER_DECK})
    if problems:
        raise RuntimeError('deck is not admissible (AGENTS.md section 8):\n' + '\n'.join(problems))
    train = task_document(ZONE, LEVEL, USER_DECK, 'fog_user_v2_l31_snow',
                          list(range(TRAIN_SEED_BASE, TRAIN_SEED_BASE + TRAIN_SEEDS)))
    train['fog_observation'] = 'human_v2'
    evaluation = task_document(ZONE, LEVEL, USER_DECK, 'fog_user_v2_l31_snow_eval',
                               list(range(EVAL_SEED_BASE, EVAL_SEED_BASE + EVAL_SEEDS)))
    evaluation['fog_observation'] = 'human_v2'
    for task in (train, evaluation):
        assert task['level'] == LEVEL and task['terrain'] == 'fog'
        assert task['deck'] == list(USER_DECK) and task['zombie_count_multiplier'] == 1.0
        assert task['fog_observation'] == 'human_v2'
    assert not set(train['seeds']) & set(evaluation['seeds'])
    return [train, evaluation]


def document() -> dict:
    return {
        'schema_version': 1,
        'split': 'stage4',
        'purpose': ZONE.purpose,
        'note': ('task70 new family; task68 fog_user and the task45 fog families unchanged. '
                 'Deck and level come from the task68 panel; only the fog observation moves to '
                 'human_v2 (task66, DESIGN section 14.25). fog_observation is a family '
                 'property, not a switch.'),
        'source': 'MEMORY/docs/task70-engine-redeploy.md; MEMORY/docs/fog-user-decks.md',
        'seed_blocks': {'train': [TRAIN_SEED_BASE, TRAIN_SEED_BASE + TRAIN_SEEDS - 1],
                        'eval': [EVAL_SEED_BASE, EVAL_SEED_BASE + EVAL_SEEDS - 1],
                        'multiplier_base': 49_200_000},
        'tasks': tasks(),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', type=Path, default=ZONE.default_out)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    doc = document()
    if args.dry_run:
        print(json.dumps(doc, ensure_ascii=False, indent=1))
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=1) + '\n')
    print(f'wrote {args.out}: ' + ', '.join(f"{t['task_id']} {len(t['seeds'])} seeds" for t in doc['tasks']))


if __name__ == '__main__':
    main()
