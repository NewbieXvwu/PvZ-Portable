#!/usr/bin/env python3
"""task88 step 2 support: prove the engine keeps a 9-digit seed intact.

Resets the same level at a few seeds and hashes the observation chain:
  - the same seed twice must agree (determinism),
  - a 9-digit seed must differ from its 16-bit residue (no uint16 truncation),
  - adjacent 9-digit seeds must differ (no rounding at the top of the range).
Read-only: one game process at a time, PVZ_NO_POPUP=1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from pvz_env import PvZEnv


def digest(observation):
    return hashlib.sha256(json.dumps(observation, sort_keys=True, default=str).encode()).hexdigest()[:16]


def chain(env, seed, level, ticks, steps):
    observation, _ = env.reset(level=level, seed=seed)
    echoed = env.privileged_state()['hidden']['rand_seed']
    hashes = [digest(observation)]
    for _ in range(steps):
        observation, _, done, truncated, _ = env.step(dict(type="wait", ticks=ticks))
        hashes.append(digest(observation))
        if done or truncated:
            break
    return dict(seed=seed, observed_rand_seed=echoed, chain=hashes)


def measure(env, seed, level, ticks, steps):
    # Keep both samples: setdefault previously discarded the second reset and
    # compared the first sample with itself, making determinism a tautology.
    seeds = (seed, seed, seed % 65536, seed % 256, seed + 1,
             100_999_999, 65535, 65536, (1 << 32) - 1)
    runs = [chain(env, value, level, ticks, steps) for value in seeds]
    chains = {run['seed']: run['chain'] for run in runs[1:]}
    return dict(
        level=level, ticks_per_step=ticks, steps=steps, runs=runs,
        seeds=list(dict.fromkeys(seeds)), chains={str(k): v for k, v in chains.items()},
        deterministic=runs[0]['chain'] == runs[1]['chain'],
        exact_uint32_echo=all(run['observed_rand_seed'] == run['seed'] for run in runs),
        differs_from_uint16_residue=runs[0]['chain'] != chains[seed % 65536],
        differs_from_uint8_residue=runs[0]['chain'] != chains[seed % 256],
        differs_from_next_seed=runs[0]['chain'] != chains[seed + 1],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--resource-dir", type=Path, required=True)
    parser.add_argument("--level", type=int, default=7)
    parser.add_argument("--seed", type=int, default=100_000_000)
    parser.add_argument("--ticks", type=int, default=100)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--out", type=Path, default=ROOT / "MEMORY/docs/gate-seed-width-check.json")
    args = parser.parse_args()
    if not 65536 <= args.seed < (1 << 32) - 1 or args.steps < 0 or args.ticks < 0:
        parser.error('seed must be in [65536, 2**32-2]; ticks/steps must be nonnegative')
    os.environ["PVZ_NO_POPUP"] = "1"
    with PvZEnv(executable=args.executable, resource_dir=args.resource_dir) as env:
        result = measure(env, args.seed, args.level, args.ticks, args.steps)
    result.update(executable=str(args.executable.resolve()),
                  executable_sha256=hashlib.sha256(args.executable.read_bytes()).hexdigest())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key not in ('chains', 'runs')}, indent=1))
    return 0 if all(result[key] for key in (
        'deterministic', 'exact_uint32_echo', 'differs_from_uint16_residue',
        'differs_from_uint8_residue', 'differs_from_next_seed')) else 1


if __name__ == "__main__":
    sys.exit(main())
