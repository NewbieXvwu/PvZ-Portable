"""Read-only task45 inventory from the running engine; waiting is a reset/termination probe."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]

from pvz_env import PvZEnv, TaskSpec
from pvz_tasks import deck_for_level, profile_for_deck
import pvz_constants


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--resource-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    definitions = {d['enum']: d for d in pvz_constants.zombie_defs()}
    rows = []
    with PvZEnv(args.resource_dir, executable=args.executable) as env:
        for level in range(31, 41):
            deck = deck_for_level(level)
            obs, _ = env.reset(deck=deck, task=TaskSpec(
                level=level, seed=450031, profile=profile_for_deck(deck)))
            row = dict(level=level, background=pvz_constants.background_for_level(level),
                       deck=list(deck), warnings=env.deck_warnings(deck), pool=env.deck_pool(),
                       roster=[definitions[k] for k in obs['loadout_context']['zombie_roster']],
                       initial_fog=obs['fog'])
            # No playing policy: let an untouched board reach a terminal state.
            while not obs['terminal'] and obs['tick'] < 240000:
                obs, *_ = env.step({'type': 'wait', 'ticks': 1000})
            row.update(terminal=obs['terminal'], result=obs['result'],
                       final_tick=obs['tick'], final_wave=obs['wave'], wave_count=obs['wave_count'])
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
