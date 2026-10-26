#!/usr/bin/env python3
"""task95: L50 boss observation gate.

Runs 4 wait-driven episodes on boss_l50_orig_eval seeds (85,000,000..85,000,003) per
engine.  Checks the new engine exposes the task85 boss fields (public obs 'boss' block
with the documented keys; privileged zombies carry 'summon_counter') and that the
terminal outcome (result / tick / wave_count) is identical to the deploy14 engine,
which does not read those fields -- the boss channels must not change the simulation.

Usage:
  python3 scripts/task95_boss_l50_gate.py \
      --old <deploy14> --new <deploy15> --out artifacts/task95/boss_l50.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]

from pvz_env import PvZEnv, TaskSpec, profile_for_deck  # noqa: E402

RESOURCE = Path('/home/newbiexvwu/.cache/pvz-research-resources')
SEEDS = [85_000_000, 85_000_001, 85_000_002, 85_000_003]
BOSS_KEYS = {'active', 'stage', 'head', 'action', 'target_row', 'target_col', 'projectile'}


def run_engine(executable: Path) -> list[dict]:
    rows = []
    with PvZEnv(RESOURCE, executable=executable) as env:
        for seed in SEEDS:
            obs, _info = env.reset(deck=(), task=TaskSpec(
                level=50, seed=seed, conveyor_start='original',
                profile=profile_for_deck(())))
            assert 'boss' in obs, f'seed {seed}: no boss block in public observation'
            boss_reset = dict(obs['boss'])
            assert BOSS_KEYS <= set(boss_reset), boss_reset.keys()
            assert boss_reset['active'], f'seed {seed}: boss not active at reset'
            steps = 0
            while not obs['terminal'] and steps < 400:
                obs, *_ = env.step({'type': 'wait', 'ticks': 1000})
                steps += 1
            assert obs['terminal'], f'seed {seed}: no terminal after {steps} waits'
            priv = env.privileged_state()
            zombies = priv.get('zombies', [])
            summon = sorted({z.get('summon_counter') for z in zombies
                             if 'summon_counter' in z})
            rows.append({
                'seed': seed,
                'boss_reset': boss_reset,
                'boss_terminal': dict(obs.get('boss', {})),
                'result': obs['result'],
                'tick': obs['tick'],
                'wave_count': obs['wave_count'],
                'state_hash': obs.get('state_hash'),
                'summon_counter_values': summon,
                'has_summon_counter': bool(summon),
            })
            print(f"  seed {seed}: result={obs['result']} tick={obs['tick']} "
                  f"waves={obs['wave_count']} boss_keys={sorted(boss_reset)} "
                  f"summon_counter={bool(summon)}", flush=True)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old', type=Path, required=True)
    p.add_argument('--new', type=Path, required=True)
    p.add_argument('--out', type=Path, default=ROOT / 'artifacts/task95/boss_l50.json')
    args = p.parse_args()
    evidence = {}
    for tag, executable in (('old', args.old), ('new', args.new)):
        if not executable.is_file():
            raise FileNotFoundError(executable)
        print(f'== {tag}: {executable}', flush=True)
        evidence[tag] = run_engine(executable)
    checks = {
        'new_boss_fields_present': all(r['boss_reset'].keys() >= BOSS_KEYS
                                       for r in evidence['new']),
        'new_summon_counter_present': all(r['has_summon_counter'] for r in evidence['new']),
        'terminal_match': all(o['result'] == n['result'] and o['tick'] == n['tick']
                              and o['wave_count'] == n['wave_count']
                              for o, n in zip(evidence['old'], evidence['new'])),
    }
    evidence['checks'] = checks
    evidence['pass'] = all(checks.values())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(evidence, ensure_ascii=False, indent=1) + '\n')
    print('BOSS L50', 'PASS' if evidence['pass'] else 'FAIL', checks)
    sys.exit(0 if evidence['pass'] else 1)


if __name__ == '__main__':
    main()
