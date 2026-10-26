#!/usr/bin/env python3
"""task95: state_hash determinism probe (task82 effect).

For each engine binary: 8 fresh processes reset the same task and record state_hash
after reset and after two short waits; then one process repeats reset 4 times.
task82 expects the candidate (deploy15) engine to produce a single hash across
processes and across same-process resets; the pre-task82 engine is run as a control
and is expected to show multiple hashes (the reset dual-state line).

Usage:
  python3 scripts/task95_state_hash_probe.py --engine old:/path/pvz-portable \
      --engine new:/path/pvz-portable --out artifacts/task95/state_hash_probe.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]

from pvz_env import PlayerProfileContext, TaskSpec  # noqa: E402
from pvz_event_env import EventWaitEnv  # noqa: E402

RESOURCE = Path('/home/newbiexvwu/.cache/pvz-research-resources')
TASK = TaskSpec(level=7, seed=30000, profile=PlayerProfileContext(),
                preplanted=tuple((0, row, 0) for row in range(5)))
DECK = [0, 1, 2, 3, 4, 5]


def fresh_process_hashes(executable: Path, processes: int = 8) -> list[list[str]]:
    hashes = []
    for _ in range(processes):
        with EventWaitEnv(RESOURCE, executable=executable) as env:
            env.begin_demo()
            env.reset(deck=DECK, task=TASK)
            row = [env.state_hash()]
            for ticks in (60, 600):
                env.step({'type': 'wait', 'ticks': ticks})
                row.append(env.state_hash())
            hashes.append(row)
    return hashes


def same_process_hashes(executable: Path, resets: int = 4) -> list[str]:
    hashes = []
    with EventWaitEnv(RESOURCE, executable=executable) as env:
        for _ in range(resets):
            env.begin_demo()
            env.reset(deck=DECK, task=TASK)
            env.step({'type': 'wait', 'ticks': 60})
            hashes.append(env.state_hash())
    return hashes


def probe(executable: Path) -> dict:
    multi = fresh_process_hashes(executable)
    same = same_process_hashes(executable)
    return {
        'executable': str(executable),
        'multi_process': {'hashes': multi, 'distinct': len({tuple(r) for r in multi})},
        'same_process': {'hashes': same, 'distinct': len(set(same))},
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine', action='append', required=True,
                   help='tag:/path/to/pvz-portable (repeatable)')
    p.add_argument('--out', type=Path, default=ROOT / 'artifacts/task95/state_hash_probe.json')
    args = p.parse_args()
    evidence = {}
    for spec in args.engine:
        tag, _, path = spec.partition(':')
        executable = Path(path)
        if not executable.is_file():
            raise FileNotFoundError(executable)
        print(f'== probing {tag}: {executable}', flush=True)
        evidence[tag] = probe(executable)
        print(f"   multi-process distinct={evidence[tag]['multi_process']['distinct']}  "
              f"same-process distinct={evidence[tag]['same_process']['distinct']}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(evidence, ensure_ascii=False, indent=1) + '\n')
    print(args.out)


if __name__ == '__main__':
    main()
