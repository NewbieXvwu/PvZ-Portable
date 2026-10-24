#!/usr/bin/env python3
"""task68: fog-zone user-deck family `fog_user_l31_snow` (level 31, human fog observation).

Why this family exists
----------------------
task67 measured twelve cold-start decks on the fog zone (L31-L34, 32 paired seeds each).
Only `fog_user_l31_snow` crossed 50%: teacher 25/32 (78.1%), against 0/32 for the qc
`fog_l*_train_f8` deck on the same seeds (paired McNemar p<1e-6). The difference is one
card: snow pea 5 replaces plantern 25, and the teacher stops replanting planterns
(4.3/game with the old deck vs 2.7 sun-shrooms; with the snow deck 21.6 sun-shrooms,
24.9 puff-shrooms, 8.3 fume-shrooms). Evidence: MEMORY/docs/fog-user-decks.md.

This module only writes the *task definition* (deck, level, seeds, fog observation); no
playing rule, no card ranking. Deck order is the task67 panel's order.

Seeds
-----
Training `43_000_000 + 8192` and evaluation `43_700_000 + 32` are new blocks, disjoint from
the demo seeds (20261031-20261034), the task67 panel (20261101-20261132), the task45 fog
blocks (39M/40M/41M/42M), roof (44M/45M/46M/48M) and the multiplier bands (5M/700k).
`make_multiplier_tasks` derives the six multiplier variants from 43_100_000 + 100_000 x band,
so the base task keeps its own 43_000_000 block.

The manifest is local only (artifacts/ is gitignored). Regenerate it before regenerating
`artifacts/stage4/multiplier_panel.json`:

  python3 scripts/make_fog_user_panel.py
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
# task67 panel order (sun-shroom, puff-shroom, fume-shroom, lily pad, tall nut, squash,
# tangle kelp, cherry bomb, wallnut, snow pea). Ten cards = SEEDBANK_MAX; no upgrade plant,
# so no owned_upgrade_plants entry is needed (AGENTS.md section 8).
USER_DECK = (9, 8, 10, 16, 23, 17, 19, 2, 3, 5)

TRAIN_SEED_BASE = 43_000_000
TRAIN_SEEDS = 8192
EVAL_SEED_BASE = 43_700_000
EVAL_SEEDS = 32

ZONE = Zone(
    key='fog_user', task_prefix='fog_user_l', first_level=LEVEL, last_level=LEVEL, terrain='fog',
    default_out=ROOT / 'artifacts/task68/fog_user_panel.json',
    purpose='task68: fog zone user-deck family (L31, human fog observation)',
    special_notes={LEVEL: 'user deck from the task67 panel; teacher 25/32 cold start'},
)


def tasks() -> list[dict]:
    """Base training task + the evaluation sibling, both with the task67 deck."""
    problems = check_decks(ZONE, LEVEL, {'fog_user': USER_DECK})
    if problems:
        raise RuntimeError('deck is not admissible (AGENTS.md section 8):\n' + '\n'.join(problems))
    train = task_document(ZONE, LEVEL, USER_DECK, 'fog_user_l31_snow',
                          list(range(TRAIN_SEED_BASE, TRAIN_SEED_BASE + TRAIN_SEEDS)))
    train['fog_observation'] = 'human'
    evaluation = task_document(ZONE, LEVEL, USER_DECK, 'fog_user_l31_snow_eval',
                               list(range(EVAL_SEED_BASE, EVAL_SEED_BASE + EVAL_SEEDS)))
    evaluation['fog_observation'] = 'human'
    for task in (train, evaluation):
        assert task['level'] == LEVEL and task['terrain'] == 'fog'
        assert task['deck'] == list(USER_DECK) and task['zombie_count_multiplier'] == 1.0
        assert task['fog_observation'] == 'human'
    assert not set(train['seeds']) & set(evaluation['seeds'])
    return [train, evaluation]


def document() -> dict:
    return {
        'schema_version': 1,
        'split': 'stage4',
        'purpose': ZONE.purpose,
        'note': ('task68 new family; task45 fog families unchanged. Deck and level come from the '
                 'task67 cold-start panel (MEMORY/docs/fog-user-decks.md); no playing rule and no '
                 'card ranking is encoded here. fog_observation is a family property, not a switch.'),
        'source': 'MEMORY/docs/fog-user-decks.md (task67: teacher 25/32, control 0/32)',
        'seed_blocks': {'train': [TRAIN_SEED_BASE, TRAIN_SEED_BASE + TRAIN_SEEDS - 1],
                        'eval': [EVAL_SEED_BASE, EVAL_SEED_BASE + EVAL_SEEDS - 1],
                        'multiplier_base': 43_100_000},
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
