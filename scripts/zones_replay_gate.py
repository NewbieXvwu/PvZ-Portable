"""Compare full observations on immutable non-fog action tapes against a deployed engine.

The historical tape hashes are retained as provenance, while the current deployed
binary is the comparison reference. Each wait is expanded to individual game ticks.
Only one engine runs at a time; candidate mismatches fail immediately.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from pvz_env import PlayerProfileContext, PvZEnv, TaskSpec


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def replay(executable, resources, tape, reference=None):
    with gzip.open(tape, 'rt') as stream:
        data = json.load(stream)
    spec = dict(data['task'])
    spec['profile'] = PlayerProfileContext(**spec['profile'])
    spec['preplanted'] = tuple(tuple(p) for p in spec['preplanted'])
    hashes = []
    historical_differences = 0
    with PvZEnv(resources, executable=executable) as env:
        observation, _ = env.reset(deck=data['deck'], task=TaskSpec(**spec))

        def check():
            nonlocal historical_differences
            index = len(hashes)
            value = digest(observation)
            if reference is not None:
                if index >= len(reference) or value != reference[index]:
                    raise AssertionError(f'{tape.name}: frame {index}, tick {observation["tick"]} differs')
            if index >= len(data['hashes']) or value != data['hashes'][index]:
                historical_differences += 1
            hashes.append(value)

        check()
        for operation in data['operations']:
            if observation['terminal']:
                raise AssertionError(f'{tape.name}: terminal before the action tape ended')
            steps = operation['ticks'] if operation['type'] == 'wait' else 1
            action = dict(type='wait', ticks=1) if operation['type'] == 'wait' else operation
            for _ in range(steps):
                observation, *_ = env.step(action)
                check()
                if observation['terminal']:
                    break
        assert observation['terminal'], f'{tape.name}: incomplete game'
        assert len(hashes) == len(data['hashes']), f'{tape.name}: historical frame count changed'
        if reference is not None:
            assert len(hashes) == len(reference), f'{tape.name}: reference frame count changed'
        final = {key: observation[key] for key in ('tick', 'wave', 'result')}
        assert final == data['final'], (tape.name, final, data['final'])
    return hashes, dict(level=spec['level'], frames=len(hashes), final=final,
                        terminal=True, differences=0,
                        observation_stream_sha256=digest(hashes),
                        historical_digest_differences=historical_differences,
                        tape_sha256=hashlib.sha256(tape.read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('reference', 'verify'))
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--resource-dir', type=Path, required=True)
    parser.add_argument('--tapes', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report = dict(mode=args.mode, executable=str(args.executable),
                  executable_sha256=hashlib.sha256(args.executable.read_bytes()).hexdigest(), episodes=[])
    for level in (7, 12, 21, 10):
        path = args.out / f'reference-l{level}.json.gz'
        reference = None
        if args.mode == 'verify':
            with gzip.open(path, 'rt') as stream:
                reference = json.load(stream)
        hashes, result = replay(args.executable, args.resource_dir, args.tapes / f'l{level}.json.gz', reference)
        if args.mode == 'reference':
            with gzip.open(path, 'wt') as stream:
                json.dump(hashes, stream)
        report['episodes'].append(result)
        print(json.dumps(result), flush=True)
    (args.out / f'{args.mode}.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
