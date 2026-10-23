"""Archive four complete non-fog student games and check every tick's full observation.

The baseline engine creates immutable action tapes and SHA256 per frame; the candidate
replays those tapes without consulting the student. Large tapes stay in artifacts/task45.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]
from pvz_env import PvZEnv, TaskSpec, PlayerProfileContext
from pvz_tasks import deck_for_level, profile_for_deck


def digest(obs):
    return hashlib.sha256(json.dumps(obs, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def run(args, level):
    tape = args.out / f'l{level}.json.gz'
    if args.mode == 'record':
        import torch
        from pvz_rollout_policy import act, load_student
        from pvz_student_observation import StudentHistory
        from pvz_value_net import observation_context
        from render_episode import _frame
        torch.set_num_threads(1)
        net = load_student(args.student)
        history = StudentHistory(net.obs) if net.obs != 'v1' else None
        deck = deck_for_level(level)
        task = TaskSpec(level=level, seed=450000 + level, profile=profile_for_deck(deck))
        data = dict(task=asdict(task), deck=deck, operations=[], hashes=[])
    else:
        with gzip.open(tape, 'rt') as stream:
            data = json.load(stream)
        td = dict(data['task'])
        td['profile'] = PlayerProfileContext(**td['profile'])
        td['preplanted'] = tuple(tuple(p) for p in td['preplanted'])
        task, deck = TaskSpec(**td), data['deck']
    with PvZEnv(args.resource_dir, executable=args.executable) as env:
        obs, _ = env.reset(deck=deck, task=task)
        index = 0

        def check():
            nonlocal index
            value = digest(obs)
            if args.mode == 'record':
                data['hashes'].append(value)
            elif index >= len(data['hashes']) or data['hashes'][index] != value:
                raise AssertionError(f'L{level} frame {index}, tick {obs["tick"]} differs')
            index += 1

        check()
        if args.mode == 'record':
            context = observation_context(obs)
            previous, action = None, None
            while not obs['terminal']:
                if obs['tick'] > 180000:
                    raise AssertionError('whole-game tape did not terminate')
                frame = _frame(obs)
                frame['action'] = action
                action = act(net, frame, previous, obs['legal_actions']['plants'],
                             context=context, history=history)
                previous = frame
                primitive = {k: v for k, v in action.items() if k != 'until'}
                data['operations'].append(primitive)
                if primitive['type'] == 'wait':
                    for _ in range(primitive['ticks']):
                        obs, *_ = env.step(dict(type='wait', ticks=1))
                        check()
                        if obs['terminal']:
                            break
                else:
                    obs, *_ = env.step(primitive)
                    check()
            data['final'] = dict(tick=obs['tick'], wave=obs['wave'], result=obs['result'])
            with gzip.open(tape, 'wt') as stream:
                json.dump(data, stream)
        else:
            for action in data['operations']:
                if action['type'] == 'wait':
                    for _ in range(action['ticks']):
                        obs, *_ = env.step(dict(type='wait', ticks=1))
                        check()
                        if obs['terminal']:
                            break
                else:
                    obs, *_ = env.step(action)
                    check()
            assert obs['terminal'] and index == len(data['hashes'])
        return dict(level=level, frames=index, final=data['final'], differences=0,
                    executable_sha256=hashlib.sha256(args.executable.read_bytes()).hexdigest(),
                    tape_sha256=hashlib.sha256(tape.read_bytes()).hexdigest())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['record', 'verify'])
    p.add_argument('--executable', type=Path, required=True)
    p.add_argument('--resource-dir', type=Path, required=True)
    p.add_argument('--student', type=Path)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for level in (7, 12, 21, 10):
        rows.append(run(args, level))
        print(json.dumps(rows[-1]), flush=True)
    (args.out / f'{args.mode}.json').write_text(json.dumps(rows, indent=2) + '\n')


if __name__ == '__main__':
    main()
