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
    hashes = [digest(observation)]
    for _ in range(steps):
        observation, _, done, truncated, _ = env.step(dict(type="wait", ticks=ticks))
        hashes.append(digest(observation))
        if done or truncated:
            break
    return hashes


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
    os.environ["PVZ_NO_POPUP"] = "1"
    seeds = (args.seed, args.seed, args.seed % 65536, args.seed % 256, args.seed + 1)
    chains = {}
    with PvZEnv(executable=args.executable, resource_dir=args.resource_dir) as env:
        for seed in seeds:
            chains.setdefault(seed, chain(env, seed, args.level, args.ticks, args.steps))
    result = dict(
        level=args.level, ticks_per_step=args.ticks, steps=args.steps,
        seeds=list(dict.fromkeys(seeds)), chains={str(k): v for k, v in chains.items()},
        deterministic=chains[args.seed] == chains[args.seed],
        differs_from_uint16_residue=chains[args.seed] != chains[args.seed % 65536],
        differs_from_uint8_residue=chains[args.seed] != chains[args.seed % 256],
        differs_from_next_seed=chains[args.seed] != chains[args.seed + 1],
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({key: result[key] for key in result if key != "chains"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
