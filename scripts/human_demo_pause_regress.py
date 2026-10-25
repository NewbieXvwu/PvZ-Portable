"""Record a short windowed demonstration that exercises the task80 operation pause.

The plan: pause (Esc), plant two puff-shrooms on the frozen tick, confirm nothing on the
tape advanced, resume, then end the recording the same way a window close does -- and the
archive must verify.  The key presses go through DEMO_INPUT, which pushes real SDL events
into the queue HumanInput::Poll reads, so the script drives the handler a human key would.

    python3 scripts/human_demo_pause_regress.py --task demo_fog_31_fume
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resource-dir', type=Path,
                        default=Path('/Users/newbiexvwu/Downloads/Plants_Vs_Zombies_V1.2.0.1073_EN'))
    parser.add_argument('--executable', type=Path, default=ROOT / 'build/pvz-portable')
    parser.add_argument('--manifest', type=Path, default=ROOT / 'configs/human_demo_tasks.json')
    parser.add_argument('--task', default='demo_fog_31_fume')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--out', type=Path, default=ROOT / 'artifacts/human_demo/pause_regress')
    args = parser.parse_args()

    spec = load_training_task(args.manifest, args.task)
    kw = training_task_kwargs(spec, args.seed)
    deck, seed = kw['deck'], kw['seed']
    task = TaskSpec(level=kw['level'], seed=seed, playthrough=2,
                    profile=profile_for_deck(deck, kw['first_aid_owned']),
                    **{key: value for key, value in kw['task_extra'].items() if value is not None})
    archive = args.out / f'seed_{seed}' / args.task
    with EventWaitEnv(args.resource_dir, executable=args.executable, headless=False) as env:
        recorder = Recorder(env, archive, args.task, deck, task, 'task80 操作暂停回归')
        tick0 = recorder.observation['tick']

        # Pause: the poll must report it and offer a zero-tick wait, never a close.
        env.demo_key('escape')
        event = env.human_input()
        assert event['paused'] and not event['closed'], event
        assert event['action'] == {'type': 'wait', 'ticks': 0}, event
        recorder.step(event['action'])
        assert recorder.observation['tick'] == tick0, 'a paused poll must not move time'
        for _ in range(3):                    # idle paused polls keep coming back the same
            event = env.human_input()
            assert event['paused'] and event['action'] == {'type': 'wait', 'ticks': 0}, event

        # Plant two while frozen: both land on the same tick, as consecutive actions.
        packet = next((index for index, seed_type in enumerate(deck) if seed_type == 8), 0)
        for row in LAND_ROWS[:2]:
            recorder.step(dict(type='plant', packet=packet, col=0, row=row))
        entries = [json.loads(line) for line in
                   (archive / 'ticks.jsonl').read_text().splitlines()[-2:]]
        assert all(entry['action']['type'] == 'plant' for entry in entries), entries
        assert all(entry['tick'] == tick0 for entry in entries), 'paused plants must share a tick'
        assert recorder.observation['tick'] == tick0, 'planting while paused must not move time'

        # Resume: time offers ticks again.
        env.demo_key('escape')
        event = env.human_input()
        assert not event['paused'] and not event['closed'], event
        assert event['action'] == {'type': 'wait', 'ticks': 1}, event
        recorder.step(dict(type='wait', ticks=90))
        assert recorder.observation['tick'] > tick0

        # End the recording the way a window close does.
        env.demo_key('close')
        event = env.human_input()
        assert event['closed'] and not event['paused'], event
        recorder.close()
    report = verify(archive, args.resource_dir, args.executable)
    print(json.dumps(report, ensure_ascii=False))
    print(f'存档：{archive}')
    if not report['verified']:
        raise SystemExit('暂停回归录制未能通过校验')


if __name__ == '__main__':
    main()
