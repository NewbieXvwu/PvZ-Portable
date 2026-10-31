"""Record native-window demonstrations, replay actions, and export training archives."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]

from pvz_env import PlayerProfileContext, TaskSpec
from pvz_event_env import EventWaitEnv
from pvz_frame_store import frame_path, read_frames, write_frames
from pvz_episode_archive import read_meta, write_meta
from pvz_tasks import profile_for_deck
from render_episode import _frame, load_training_task, training_task_kwargs


def reset_archive(env, meta, *, fog_observation=None):
    """Restore the archive's initial board.

    task70 B6: fog_observation is presentation only, so a caller may override the
    recorded value with the manifest's: demo_start cells under human_v2 must not inherit the
    recording-time x-ray just because the demo task predates the fog families.  The
    board itself is untouched.
    """
    config = dict(meta['environment_task'])
    if fog_observation is not None:
        config['fog_observation'] = fog_observation
    config['profile'] = PlayerProfileContext(**config['profile'])
    config['preplanted'] = tuple(tuple(plant) for plant in config.get('preplanted', ()))
    env.begin_demo()
    return env.reset(deck=meta['task']['deck'], task=TaskSpec(**config))[0]


class Recorder:
    def __init__(self, env, archive: Path, task_id: str, deck, task: TaskSpec, note: str = '',
                 label_source: str = 'human'):
        self.env, self.archive = env, archive
        self.label_source = label_source
        env.begin_demo()
        self.observation, _ = env.reset(deck=deck, task=task)
        options = env.deck_options()
        warnings = env.deck_warnings(deck)['warnings']
        original_conveyor = task.conveyor_start == 'original' and not deck
        if set(deck) - set(options['available_plants']) or (warnings and not original_conveyor):
            raise ValueError(f'deck contains locked plants or engine warnings: {warnings}')
        archive.mkdir(parents=True, exist_ok=False)
        self.operations = 0
        self.events = {}
        self.last_tick = self.observation['tick']
        self.last_operation = 0
        self.frame_count = 0
        self.meta = dict(label_source=label_source, note=note,
                         task=dict(id=task_id, level=task.level, seed=task.seed, deck=list(deck),
                                   policy='human', playthrough=task.playthrough,
                                   zombie_count_multiplier=task.zombie_count_multiplier,
                                   wave_cap=task.wave_cap, preplanted=list(task.preplanted)),
                         environment_task=asdict(task),
                         context=dict(level=task.level, multiplier=task.zombie_count_multiplier,
                                      roster=self.observation['loadout_context']['zombie_roster']),
                         sun_collection='environment_auto', verified=False)
        self.tape = (archive / 'ticks.jsonl').open('w', encoding='utf-8', buffering=1)
        self.frames = []
        self._frame(None)
        self._save_meta()

    def _frame(self, action, rejected=False):
        frame = _frame(self.observation)
        frame.update(action=action, action_rejected=rejected, label_source=self.label_source, operation=self.operations)
        if action and action['type'] not in ('plant', 'wait'):
            frame['input_action'] = frame['action']
            frame['action'] = None
        self.frames.append(frame)
        self.frame_count += 1
        self.last_tick = self.observation['tick']
        self.last_operation = self.operations

    def _save_meta(self):
        obs = self.observation
        result = int(obs.get('result') or 0)
        self.meta.update(frame_count=self.frame_count, operations=self.operations, totals=self.events,
                         outcome=dict(result=result, won=result == 1, reason='通关' if result == 1 else
                                      '僵尸进屋' if obs['terminal'] else '中途结束',
                                      final_wave=obs['wave'], wave_count=obs['wave_count'],
                                      final_tick=obs['tick'], actions=self.frame_count - 1,
                                      plants_eaten=self.events.get('plants_eaten', 0),
                                      plants_eaten_free=self.events.get('plants_eaten_free', 0)))
        self.meta = write_meta(self.archive, self.meta)
        write_frames(self.archive / 'frames.npz', self.frames, cache=False)

    def step(self, action):
        if action['type'] == 'wait' and 'until' not in action:
            for _ in range(action['ticks']):
                if self.observation['terminal']:
                    break
                self._step(dict(type='wait', ticks=1))
        else:
            if self.observation['tick'] != self.last_tick:
                self._frame(dict(type='wait', ticks=self.observation['tick'] - self.last_tick))
            self._step(action)
        return self.observation

    def _step(self, action):
        before = self.observation['tick']
        self.observation, _, _, _, info = self.env.step(action)
        self.last_info = info
        self.operations += 1
        self.tape.write(json.dumps(dict(action=action, tick=before, end_tick=self.observation['tick'],
                                        ok=info['ok']),
                                   separators=(',', ':')) + '\n')
        for name, value in info['events'].items():
            self.events[name] = self.events.get(name, 0) + value
        if action['type'] != 'wait':
            self._frame(action, rejected=not info['ok'])
        elif self.observation['tick'] - self.last_tick >= 60 or self.observation['terminal']:
            self._frame(dict(type='wait', ticks=self.observation['tick'] - self.last_tick))

    def close(self):
        if self.last_operation != self.operations:
            self._frame(dict(type='wait', ticks=self.observation['tick'] - self.last_tick))
        self.tape.close()
        self._save_meta()


def replay(env, archive: Path, decision: int | None = None, overrides: dict | None = None,
           save: Path | None = None, *, fog_observation: str | None = None):
    meta = read_meta(archive)
    frames = read_frames(frame_path(archive))
    if decision is not None and not 1 <= decision <= len(frames):
        raise ValueError(f'decision must be between 1 and {len(frames)}')
    stop = frames[decision - 1]['operation'] if decision is not None else meta['operations']
    obs = reset_archive(env, meta, fog_observation=fog_observation)
    recorder = None
    if save is not None:
        config = dict(meta['environment_task'])
        config['profile'] = PlayerProfileContext(**config['profile'])
        config['preplanted'] = tuple(tuple(plant) for plant in config.get('preplanted', ()))
        recorder = Recorder(env, save, meta['task']['id'], meta['task']['deck'], TaskSpec(**config),
                            meta.get('note', ''), label_source='counterfactual')
        obs = recorder.observation
    replacements = {frames[index - 1]['operation'] + 1: (frames[index]['operation'], action)
                    for index, action in (overrides or {}).items()}
    diverged = False
    rejected = False
    events = {}
    changed_frames = [_frame(obs)] if overrides else None
    frame_operations = {frame['operation'] for frame in frames}
    mowers = sum(defense['state'] == 1 for defense in obs.get('defenses') or [])
    mowers_left = mowers
    skip_until = 0
    with (archive / 'ticks.jsonl').open() as stream:
        for operation, line in enumerate(stream, 1):
            if operation > stop:
                break
            if operation <= skip_until:
                continue
            entry = json.loads(line)
            if not diverged and obs['tick'] != entry['tick']:
                raise RuntimeError(f'demonstration tick differs at operation {operation}')
            action = entry['action']
            if operation in replacements:
                skip_until, action = replacements[operation]
            diverged |= operation in replacements
            if recorder is None:
                obs, _, _, _, info = env.step(action)
            else:
                before_events = dict(recorder.events)
                obs = recorder.step(action)
                info = dict(ok=recorder.last_info['ok'], events={name: value - before_events.get(name, 0)
                                                               for name, value in recorder.events.items()})
            rejected |= operation in replacements and not info['ok']
            for name, value in info['events'].items():
                events[name] = events.get(name, 0) + value
            if not obs['terminal']:
                mowers_left = sum(defense['state'] == 1 for defense in obs.get('defenses') or [])
            if changed_frames is not None and (operation in replacements or operation in frame_operations or obs['terminal']):
                frame = _frame(obs)
                frame.update(action=action if action['type'] in ('plant', 'wait') else None,
                             action_rejected=not info['ok'], label_source='counterfactual')
                changed_frames.append(frame)
            if not diverged and (info['ok'] != entry['ok'] or obs['tick'] != entry['end_tick']):
                raise RuntimeError(f'demonstration action timeline differs at operation {operation}, tick {obs["tick"]}')
            if obs['terminal'] and diverged:
                break
    if decision is None and not diverged:
        outcome = meta['outcome']
        if (obs['tick'], obs['wave'], obs['result']) != (outcome['final_tick'], outcome['final_wave'], outcome['result']):
            raise RuntimeError('demonstration terminal outcome differs')
    if overrides:
        if recorder is not None:
            recorder.meta['source_archive'] = str(archive.resolve())
            recorder.meta['overrides'] = overrides
            recorder.close()
        meta = {**meta, 'label_source': 'counterfactual', 'totals': events,
                'outcome': dict(won=obs['result'] == 1, result=obs['result'], final_wave=obs['wave'],
                                wave_count=obs['wave_count'], final_tick=obs['tick'],
                                reason='通关' if obs['result'] == 1 else '僵尸进屋' if obs['terminal'] else '中途结束',
                                mowers_used=mowers - mowers_left, plants_eaten=events.get('plants_eaten', 0),
                                plants_eaten_free=events.get('plants_eaten_free', 0),
                                override_rejected=rejected)}
        return obs, meta, changed_frames
    return obs, meta, frames[:decision] if decision is not None else frames


def verify(archive: Path, resource_dir, executable=None):
    with EventWaitEnv(resource_dir, executable=executable) as env:
        obs, meta, _ = replay(env, archive)
    meta['verified'] = True
    write_meta(archive, meta)
    return dict(verified=True, operations=meta['operations'], tick=obs['tick'], wave=obs['wave'], result=obs['result'])


def export(archive: Path, data: Path):
    meta = read_meta(archive)
    if not meta.get('verified'):
        raise ValueError('verify the demonstration before exporting training data')
    target = data / f'seed_{meta["task"]["seed"]}' / meta['task']['id']
    target.mkdir(parents=True, exist_ok=False)
    write_frames(target / 'frames.npz', read_frames(frame_path(archive)))
    write_meta(target, meta)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resource-dir', type=Path, default=Path.home() / '.cache/pvz-research-resources')
    parser.add_argument('--executable', type=Path, default=ROOT / 'build/pvz-portable')
    commands = parser.add_subparsers(dest='command', required=True)
    record = commands.add_parser('record')
    record.add_argument('--manifest', type=Path, default=ROOT / 'configs/human_demo_tasks.json')
    record.add_argument('--task', required=True)
    record.add_argument('--seed', type=int)
    record.add_argument('--note', default='')
    record.add_argument('--out', type=Path, default=ROOT / 'artifacts/human_demo/raw')
    check = commands.add_parser('verify')
    check.add_argument('--archive', type=Path, required=True)
    convert = commands.add_parser('export')
    convert.add_argument('--archive', type=Path, required=True)
    convert.add_argument('--out', type=Path, default=ROOT / 'artifacts/human_demo/frames_v2')
    args = parser.parse_args()
    if args.command == 'verify':
        print(json.dumps(verify(args.archive, args.resource_dir, args.executable)))
        return
    if args.command == 'export':
        print(export(args.archive, args.out))
        return
    spec = load_training_task(args.manifest, args.task)
    kw = training_task_kwargs(spec, args.seed)
    deck, seed = kw['deck'], kw['seed']
    task = TaskSpec(level=kw['level'], seed=seed, playthrough=2,
                    profile=profile_for_deck(deck, kw['first_aid_owned']),
                    **{key: value for key, value in kw['task_extra'].items() if value is not None})
    archive = args.out / f'seed_{seed}' / args.task
    with EventWaitEnv(args.resource_dir, executable=args.executable, headless=False) as env:
        recorder = Recorder(env, archive, args.task, deck, task, args.note)
        print(f'录制中：{archive}。Esc 或空格暂停/继续（暂停时可以慢慢种、铲、收集），'
              f'右键取消选卡，关闭窗口结束录制。', flush=True)
        try:
            while not recorder.observation['terminal']:
                event = env.human_input()
                if event['closed']:
                    break
                # task80: while paused the poll yields wait ticks:0 -- step nothing, so game
                # time stands still; but a plant/shovel/collect is still accepted and runs at
                # the frozen tick.  Several paused actions therefore land on one tick.
                if event['paused'] and event['action']['type'] == 'wait':
                    time.sleep(.03)
                    continue
                recorder.step(event['action'])
        finally:
            recorder.close()
    print(json.dumps(verify(archive, args.resource_dir, args.executable)))
    print(f'存档：{archive}')


if __name__ == '__main__':
    main()
