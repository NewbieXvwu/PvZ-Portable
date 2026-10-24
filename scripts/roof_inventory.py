"""task46: query roof boards and reach terminal with waits (no playing policy)."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]
from pvz_env import PvZEnv, TaskSpec
from pvz_tasks import deck_for_level, profile_for_deck


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, default=ROOT / 'build/pvz-portable')
    parser.add_argument('--resource-dir', type=Path, default=Path.home() / '.cache/pvz-research-resources')
    parser.add_argument('--out', type=Path, default=ROOT / 'artifacts/task46/inventory.json')
    args = parser.parse_args()
    os.environ['PVZ_NO_POPUP'] = '1'
    rows = []
    with PvZEnv(args.resource_dir, executable=args.executable) as env:
        for level in range(41, 50):
            deck = deck_for_level(level)
            obs, _ = env.reset(deck=deck, task=TaskSpec(level=level, seed=46000000 + level,
                                                       profile=profile_for_deck(deck)))
            row = dict(level=level, deck=list(deck), night=obs['night'], roof=obs['roof'],
                       wave_count=obs['wave_count'],
                       loadout_context=obs.get('loadout_context'),
                       pots=[[p['row'], p['col']] for p in obs['plants'] if p['type'] == 33],
                       legal_actions=obs['legal_actions'], options=env.deck_options(),
                       pool=env.deck_pool(), warnings=env.deck_warnings(deck),
                       scheduled_zombie_types=sorted({z for wave in env.privileged_state()['hidden']['zombies_in_wave']
                                                      for z in wave}))
            snap = env.snapshot()
            action = next(a for a in obs['legal_actions']['plants'] if a['packet'] == deck.index(1))
            planted, _, _, _, info = env.step({'type': 'plant', **action})
            row['sunflower_on_pot_probe'] = dict(action=action, ok=info['ok'],
                types_at_cell=sorted(p['type'] for p in planted['plants']
                                     if p['row'] == action['row'] and p['col'] == action['col']))
            obs = env.restore(snap)
            env.release_snapshot(snap)
            seen = set()
            while not obs['terminal'] and obs['tick'] < 200000:
                obs, *_ = env.step({'type': 'wait', 'ticks': 1000})
                seen.update(z['type'] for z in obs['zombies'])
            row.update(terminal=obs['terminal'], result=obs['result'], final_tick=obs['tick'],
                       final_wave=obs['wave'], observed_zombie_types=sorted(seen))
            if not obs['terminal']:
                raise RuntimeError(f'L{level} did not reach terminal')
            rows.append(row)
            print(f"L{level}: pots={len(row['pots'])} warnings={row['warnings']['warnings']} "
                  f"roster={row['scheduled_zombie_types']} terminal={obs['terminal']} tick={obs['tick']}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(executable_sha256=hashlib.sha256(args.executable.read_bytes()).hexdigest(),
                                        method='wait-only terminal smoke; not a policy baseline', levels=rows), indent=1) + '\n')


if __name__ == '__main__':
    main()
