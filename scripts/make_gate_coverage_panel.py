#!/usr/bin/env python3
"""task247: 门禁弱项的整局训练任务。卡组与倍率生成各沿用唯一现有实现。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]

from pvz_tasks import GOOD_DECKS, USER_DECKS, USER_DECK_ALTERNATES
from zone_panel import Zone, check_decks, task_document

DEFAULT_OUT = ROOT / 'artifacts/task247/gate_coverage_panel.json'


def tasks() -> list[dict]:
    specs = [
        ('pool_user_l22_user', 22, USER_DECKS[22], 247_000_000, 'pool'),
        ('pool_user_l29_user1', 29, USER_DECKS[29], 248_000_000, 'pool'),
        ('pool_user_l29_user2', 29, USER_DECK_ALTERNATES[29][0], 249_000_000, 'pool'),
    ]
    specs += [(f'fog_good_l{level}_train', level, GOOD_DECKS[level][0],
               250_000_000 + (level - 31) * 2048, 'fog') for level in (31, 32, 34, 36)]
    result = []
    for name, level, deck, start, terrain in specs:
        zone = Zone(key=terrain, task_prefix='', first_level=level, last_level=level,
                    terrain=terrain, default_out=DEFAULT_OUT, purpose='task247 gate coverage')
        problems = check_decks(zone, level, {name: deck})
        if problems:
            raise ValueError('\n'.join(problems))
        result.append(task_document(zone, level, deck, name, list(range(start, start + 2048))))
    return result


def document() -> dict:
    return dict(schema_version=1, purpose='task247: whole-game coverage of gate v8 weak decks',
                note='Decks from pvz_tasks; variants from make_multiplier_tasks; existing families unchanged.',
                tasks=tasks())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    doc = document()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=1) + '\n')
    print(f'wrote {args.out}: {len(doc["tasks"])} tasks')


if __name__ == '__main__':
    main()
