#!/usr/bin/env python3
"""task79b: selected human_v2 L36-L38 fog curriculum base tasks."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]

from zone_panel import Zone, check_decks, task_document  # noqa: E402

ZONE = Zone(key='fog79b', task_prefix='fog79b_l', first_level=36, last_level=38,
            terrain='fog', default_out=ROOT / 'artifacts/task79b/fog79b_panel.json',
            purpose='task79b: selected human_v2 fog curriculum bases')
BASE_SEED = 79_100_000
SEEDS_PER_LEVEL = 8192
DECKS = {
    36: (9, 8, 10, 16, 23, 17, 19, 2, 5, 28),
    37: (9, 8, 10, 16, 23, 17, 19, 2, 5, 28),
    38: (9, 8, 10, 16, 23, 17, 19, 2, 5, 31),
}


def tasks() -> list[dict]:
    result = []
    for level, deck in DECKS.items():
        problems = check_decks(ZONE, level, {'best': deck})
        if problems:
            raise RuntimeError('\n'.join(problems))
        task = task_document(ZONE, level, deck, f'fog79b_l{level}_best',
                             list(range(BASE_SEED + (level - 36) * SEEDS_PER_LEVEL,
                                        BASE_SEED + (level - 35) * SEEDS_PER_LEVEL)))
        task['fog_observation'] = 'human_v2'
        result.append(task)
    return result


def document() -> dict:
    return {'schema_version': 1, 'split': 'stage4', 'purpose': ZONE.purpose,
            'seed_blocks': {'train': [BASE_SEED, BASE_SEED + 3 * SEEDS_PER_LEVEL - 1],
                            'multiplier_base': 79_200_000}, 'tasks': tasks()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=ZONE.default_out)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    value = document()
    if args.dry_run:
        print(json.dumps(value, ensure_ascii=False, indent=1))
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(value, indent=1) + '\n')
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
