"""task79 open item: round-11 vs dw4i_s1 pairing on L37, same 16 seeds.

The Mac batch could not separate "round 11 is stronger" from "the desktop build behaves
differently" because the round-11 weights only exist on the desktop.  This script is the one
command that closes that gap: run it on the desktop (or anywhere the round-11 checkpoint is
readable) and it reproduces task79 control A exactly, except the proposal policy is round 11.

  A (recorded, Mac):  dw4i_s1  4/16 on seeds 30560..30575, human_v2, magnet deck
  E (this script):    round 11 ?/16 on the same seeds, deck, observation and value nets

If E is ~15/16 the gap is the student; if E is ~4/16 it is the engine build.

Usage:
  python3 scripts/task79_r11_pair.py \
      --student ~/pvz-logs/curriculum_state.pipeline/round_000011/student/model.pt \
      --executable <repo>/build/pvz-portable --resource-dir <resources> \
      --value <repo>/artifacts/value_net/v5/model.pt \
      --static-value <repo>/artifacts/hard/value_branch1/model.pt \
      --out artifacts/task79/control/E_r11
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R11_SHA = '1cf82f5e6cf7994ca961d91e878d4ac65a4563eea2be50a1e96a9787c22dfd4c'
SEEDS = list(range(30560, 30576))
DECK = [9, 8, 10, 16, 23, 17, 19, 2, 5, 31]
TASK_ID = 'fog79_l37_ctrlE_r11'


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--student', type=Path, required=True)
    p.add_argument('--executable', type=Path, required=True)
    p.add_argument('--resource-dir', type=Path, required=True)
    p.add_argument('--value', type=Path, required=True)
    p.add_argument('--static-value', type=Path, required=True)
    p.add_argument('--out', type=Path, default=ROOT / 'artifacts/task79/control/E_r11')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seeds', type=int, default=len(SEEDS))
    p.add_argument('--level', type=int, default=37)
    p.add_argument('--seed-start', type=int, default=SEEDS[0])
    p.add_argument('--deck', default=','.join(str(card) for card in DECK),
                   help='comma separated seed types; default is the L37 magnet deck')
    p.add_argument('--task-id', default=TASK_ID)
    args = p.parse_args()
    deck = [int(card) for card in args.deck.split(',')]
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))

    digest = sha(args.student)
    if digest != R11_SHA:
        raise SystemExit(f'round 11 sha mismatch: {digest} != {R11_SHA}')
    for name in ('executable', 'value', 'static_value'):
        if not getattr(args, name).exists():
            raise SystemExit(f'{name} missing: {getattr(args, name)}')

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = args.out / 'manifest.json'
    manifest.write_text(json.dumps(dict(tasks=[dict(
        task_id=args.task_id, level=args.level, deck=deck, seeds=seeds,
        zombie_count_multiplier=1.0, fog_observation='human_v2')]), indent=1) + '\n')

    command = [sys.executable, str(ROOT / 'scripts/search_play.py'),
               '--manifest', str(manifest), '--task-id', args.task_id,
               '--executable', str(args.executable), '--resource-dir', str(args.resource_dir),
               '--arms', 'fast_search', '--dagger-beta', '1', '--verify-samples', '9',
               '--student-candidates', '2', '--workers', str(args.workers),
               '--seeds', str(args.seeds), '--out', str(args.out),
               '--rollout-policy', str(args.student), '--value', str(args.value),
               '--static-value', str(args.static_value)]
    with (args.out / 'run.log').open('a') as stream:
        subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT,
                       env={**os.environ, 'PVZ_NO_POPUP': '1', 'OMP_NUM_THREADS': '1',
                            'OPENBLAS_NUM_THREADS': '1'})
    rows = [json.loads(line) for line in (args.out / 'games.jsonl').read_text().splitlines()]
    wins = sum(row['won'] for row in rows)
    print(f'{args.task_id} {wins} / {len(rows)}  (dw4i_s1 on the same seeds: see control A)', flush=True)
    (args.out / 'result.json').write_text(json.dumps(dict(
        task_id=args.task_id, level=args.level, deck=deck,
        student=str(args.student), student_sha256=digest,
        seeds=[row['seed'] for row in rows], wins=wins, games=len(rows)), indent=1) + '\n')


if __name__ == '__main__':
    main()
