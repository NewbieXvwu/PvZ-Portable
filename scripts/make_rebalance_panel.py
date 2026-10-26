"""task119 fixed 1.0x opening cells, matching the task101/103 scoreboard decks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LEVELS = (19, 23, 24, 26, 28, 29, 32, 37)
SEED_START = 210_000_000
SEED_STRIDE = 100_000
SEEDS_PER_TASK = 8192


def make_panel():
    tasks = []
    for index, level in enumerate(LEVELS):
        deck = ([0, 1, 2, 3, 4, 5, 7, 8, 9, 11] if level == 19 else
                [9, 8, 10, 16, 23, 17, 19, 2, 31, 3] if level == 32 else
                [9, 8, 10, 16, 23, 17, 19, 2, 5, 31] if level == 37 else
                [0, 1, 2, 3, 4, 5, 7, 16, 19])
        start = SEED_START + index * SEED_STRIDE
        task = dict(task_id=f'opening119_l{level}_magnet' if level in (32, 37)
                    else f'opening119_l{level}_scoreboard', level=level, deck=deck,
                    zombie_count_multiplier=1.0, opening_specialist=True,
                    seeds=list(range(start, start + SEEDS_PER_TASK)))
        if level in (32, 37):
            task['fog_observation'] = 'human_v2'
        tasks.append(task)
    return dict(tasks=tasks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(make_panel(), separators=(',', ':')) + '\n')


if __name__ == '__main__':
    main()
