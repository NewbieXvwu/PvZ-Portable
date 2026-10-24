"""Record a short windowed demonstration from a scripted action list, then verify it.

Regression harness for task63.  The demo recorder runs the engine with a window and the
verifier replays it headless; both must agree tick for tick, including the ticks that run a
particle FIELD_SHAKE.  The plan below is deliberately small -- a few puff-shrooms, two shovels,
a few hundred ticks -- because it only has to prove the two paths agree, not to win the level.

    python3 scripts/human_demo_autoplay.py --task demo_fog_31_fume
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]

from human_demo import Recorder, verify
from pvz_env import TaskSpec
from pvz_event_env import EventWaitEnv
from pvz_tasks import profile_for_deck
from render_episode import load_training_task, training_task_kwargs

# Puff-shroom costs no sun, so the plan works on any level without waiting for an economy.
LAND_ROWS = (0, 1, 4, 5)


def plan(deck, columns=2, rows=2):
    packet = next((index for index, seed in enumerate(deck) if seed == 8), 0)  # SEED_PUFFSHROOM
    cells = [(col, row) for row in LAND_ROWS[:rows] for col in range(columns)]
    actions = [dict(type='plant', packet=packet, col=col, row=row) for col, row in cells]
    actions.append(dict(type='wait', ticks=90))
    actions += [dict(type='shovel', col=col, row=row) for col, row in cells[:2]]
    actions.append(dict(type='wait', ticks=180))
    return actions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resource-dir', type=Path,
                        default=Path('/Users/newbiexvwu/Downloads/Plants_Vs_Zombies_V1.2.0.1073_EN'))
    parser.add_argument('--executable', type=Path, default=ROOT / 'build/pvz-portable')
    parser.add_argument('--manifest', type=Path, default=ROOT / 'configs/human_demo_tasks.json')
    parser.add_argument('--task', default='demo_fog_31_fume')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--out', type=Path, default=ROOT / 'artifacts/human_demo/autoplay')
    args = parser.parse_args()

    spec = load_training_task(args.manifest, args.task)
    kw = training_task_kwargs(spec, args.seed)
    deck, seed = kw['deck'], kw['seed']
    task = TaskSpec(level=kw['level'], seed=seed, playthrough=2,
                    profile=profile_for_deck(deck, kw['first_aid_owned']),
                    **{key: value for key, value in kw['task_extra'].items() if value is not None})
    archive = args.out / f'seed_{seed}' / args.task
    with EventWaitEnv(args.resource_dir, executable=args.executable, headless=False) as env:
        recorder = Recorder(env, archive, args.task, deck, task, 'task63 自动有窗口回归')
        for action in plan(deck):
            if recorder.observation['terminal']:
                break
            recorder.step(action)
        recorder.close()
    report = verify(archive, args.resource_dir, args.executable)
    print(json.dumps(report, ensure_ascii=False))
    print(f'存档：{archive}')
    if not report['verified']:
        raise SystemExit('自动有窗口录制未能通过校验')


if __name__ == '__main__':
    main()
