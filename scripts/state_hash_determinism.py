#!/usr/bin/env python3
"""Independent-process reset/tape probe; runs one game process at a time (also on Mac)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]
from human_demo import reset_archive
from pvz_event_env import EventWaitEnv


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def segments(path):
    """Split STATE_BYTES into Save4 chunks, RNG and individual snapshot counters."""
    raw = path.read_bytes()
    assert raw[:7] == b'PVZSB1\n'
    size, = struct.unpack_from('<Q', raw, 7)
    board = raw[15:15 + size]
    assert board[:10] == b'PVZP_SAVE4'
    result = {'board_header': board[:24]}
    pos = 24
    while pos < len(board):
        kind, length = struct.unpack_from('<II', board, pos)
        result[f'chunk_{kind}'] = board[pos + 8:pos + 8 + length]
        pos += 8 + length
    pos = 15 + size
    size, = struct.unpack_from('<Q', raw, pos)
    pos += 8
    result['rng'] = raw[pos:pos + size]
    pos += size
    for name in ('appRandSeed', 'randSeed', 'appCounter', 'zombiesKilled', 'plantsEaten',
                 'sunProduced', 'triggeredLawnMowers', 'gameScene', 'boardResult'):
        result[name] = raw[pos:pos + 4]
        pos += 4
    assert pos == len(raw)
    return {name: {'sha256': hashlib.sha256(data).hexdigest(), 'size': len(data),
                   **({'hex': data.hex()} if len(data) <= 8 else {})}
            for name, data in result.items()}


def run(args):
    os.environ['PVZ_NO_POPUP'] = '1'
    args.out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((args.archive / 'meta.json').read_text())
    report = {'executable': str(args.executable),
              'executable_sha256': hashlib.sha256(args.executable.read_bytes()).hexdigest(),
              'archive': str(args.archive), 'tape_sha256': hashlib.sha256(
                  (args.archive / 'ticks.jsonl').read_bytes()).hexdigest(), 'runs': []}
    for repeat in range(args.repeats):
        trace = args.out / f'trace_{repeat}.jsonl'
        with EventWaitEnv(args.resource_dir, executable=args.executable) as env, trace.open('w') as output:
            obs = reset_archive(env, meta)
            initial_hash = env.state_hash()
            dump = args.out / f'reset_{repeat}.bin'
            assert env._command(f'STATE_BYTES {json.dumps(str(dump))}')['ok']
            item = {'pid': env._process.pid, 'initial_hash': initial_hash, 'segments': segments(dump)}
            output.write(json.dumps({'operation': 0, 'hash': initial_hash, 'observation': digest(obs)}) + '\n')
            logical = hashlib.sha256()
            first_archive_hash_diff = 0 if initial_hash != meta['initial_state_hash'] else None
            operations = 0
            with (args.archive / 'ticks.jsonl').open() as tape:
                for operation, line in enumerate(tape, 1):
                    if args.operations and operation > args.operations:
                        break
                    entry = json.loads(line)
                    before = obs['tick']
                    obs, _, _, _, info = env.step(entry['action'])
                    state_hash = env._last_response.get('state_hash') or env.state_hash()
                    assert (before, obs['tick'], info['ok']) == (entry['tick'], entry['end_tick'], entry['ok']), operation
                    if first_archive_hash_diff is None and state_hash != entry['state_hash']:
                        first_archive_hash_diff = operation
                    # Privileged view covers hidden zombies too; compare every operation,
                    # stronger than checking only archived decision frames.
                    state = env.privileged_state() if args.privileged else obs
                    record = {'operation': operation, 'hash': state_hash,
                              'observation': digest(state), 'events': info['events']}
                    logical.update(json.dumps({k: v for k, v in record.items() if k != 'hash'},
                                              sort_keys=True).encode())
                    output.write(json.dumps(record, separators=(',', ':')) + '\n')
                    operations = operation
                    env.episode['operations'].clear()
            item.update(operations=operations, logical_sha256=logical.hexdigest(),
                        first_archive_hash_diff=first_archive_hash_diff,
                        outcome={k: obs.get(k) for k in ('tick', 'wave', 'result', 'terminal')})
            if not args.operations:
                outcome = meta['outcome']
                assert (obs['tick'], obs['wave'], obs['result']) == (
                    outcome['final_tick'], outcome['final_wave'], outcome['result'])
            report['runs'].append(item)
            (args.out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps({k: v for k, v in item.items() if k != 'segments'}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, default=ROOT / 'build/pvz-portable')
    parser.add_argument('--resource-dir', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=8)
    parser.add_argument('--operations', type=int, default=2000,
                        help='0: entire tape; -1: reset only; default: first 2000 operations')
    parser.add_argument('--privileged', action='store_true')
    args = parser.parse_args()
    for name in ('executable', 'resource_dir', 'archive', 'out'):
        if not getattr(args, name).is_absolute():
            parser.error(f'--{name.replace("_", "-")} must be absolute')
    if args.repeats < 1 or args.operations < -1:
        parser.error('invalid repeat/operation count')
    run(args)


if __name__ == '__main__':
    main()
