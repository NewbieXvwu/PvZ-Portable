"""Task58: replay human archives, verify observations, isolate privileged labels."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]
from pvz_event_env import EventWaitEnv
from pvz_env import TaskSpec
from pvz_frame_store import read_frames
from pvz_student_observation import StudentHistory
from pvz_tasks import profile_for_deck
from pvz_value_net import context_vector
from render_episode import _frame

SOURCE = Path('/home/newbiexvwu/PvZ-search/.claude/worktrees/fog/artifacts/task45')
OUT = ROOT / 'artifacts/task58'
ENGINE = Path('/home/newbiexvwu/PvZ-search/.claude/worktrees/deploy3/build/pvz-portable')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def inventory():
    entries, seen = [], {}
    paths = sorted(SOURCE.rglob('frames.npz')) + sorted(SOURCE.rglob('frames.jsonl'))
    for path in paths:
        meta = json.loads((path.parent / 'meta.json').read_text())
        task = meta['task']
        if task.get('fog_observation') != 'human':
            continue
        frames = read_frames(path)
        identity = [task['level'], task['seed'], task['deck'],
                    [f.get('action') for f in frames]]
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        if key in seen:
            seen[key]['duplicates'].append(str(path))
            continue
        seed = task['seed']
        split = ('test' if seed < 42000000 or seed % 100 >= 6 else
                 'validation' if seed % 100 == 5 else 'train')
        item = dict(id=key[:16], source=str(path), source_sha256=digest(path),
                    meta_sha256=digest(path.parent / 'meta.json'), task=task,
                    context=meta.get('context'), split=split, frames=len(frames), duplicates=[])
        entries.append(item)
        seen[key] = item
    # Same seed in alternate decks or trajectories must never cross sets.
    groups = {}
    for entry in entries:
        key = (entry['task']['level'], entry['task']['seed'])
        assert groups.setdefault(key, entry['split']) == entry['split']
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / 'inventory.json', entries)
    return entries


def labels(raw):
    """Ground truth only; physical fog uses the engine's clamped 9-column grid."""
    target = np.zeros((35, 6, 9), np.uint8)
    for z in raw['zombies']:
        if not z['on_board'] or not 0 <= z['row'] < 6 or not 40 <= z['x'] < 760:
            continue
        col = int((z['x'] - 40) // 80)
        target[0, z['row'], col] = 1
        target[1 + min(z['type'], 33), z['row'], col] = 1
    return target


def encode_public(history, frame, previous, context):
    """This function deliberately has no privileged argument."""
    b, s = history.features(frame, previous, context)
    return b.numpy(), s.numpy()


def replay(entry, env):
    path = Path(entry['source'])
    assert digest(path) == entry['source_sha256']
    assert digest(path.parent / 'meta.json') == entry['meta_sha256']
    frames = read_frames(path)
    task = entry['task']
    # Use archived deck verbatim; never regenerate a perturbed historical deck.
    deck = tuple(task['deck'])
    extra = {k: task[k] for k in ('wave_cap', 'preplanted', 'zombie_count_multiplier',
                                  'playthrough') if task.get(k) is not None}
    obs, _ = env.reset(deck=deck, task=TaskSpec(level=task['level'], seed=task['seed'],
        profile=profile_for_deck(deck, task.get('first_aid_owned', False)),
        fog_observation='human', **extra))
    context = entry['context'] or {'multiplier': task.get('zombie_count_multiplier', 1),
        'roster': obs['loadout_context']['zombie_roster']}
    cv = context_vector(context['multiplier'], context['roster'])
    histories = [StudentHistory('v2fog'), StudentHistory('v2fog')]
    buffers = {k: [] for k in ('human_board', 'human_scalar', 'xray_board', 'xray_scalar',
                               'target', 'fog', 'age', 'tick', 'dt')}
    last_visible = np.full((6, 9), -1, np.int32)
    previous = [None, None]
    blover_pending, cleared = False, False
    events = []
    for i, archived in enumerate(frames):
        info = None
        if i:
            obs, _, _, _, info = env.step(archived['action'])
            assert info['ok'] == (not archived.get('action_rejected', False))
            a = archived['action']
            if info['ok'] and a['type'] == 'plant' and deck[a['packet']] == 27:
                blover_pending = True
        public = json.loads(json.dumps(_frame(obs)))
        # Metadata / labels in an archive are not observation fields.
        for key in public:
            if public[key] != archived.get(key):
                raise AssertionError(f"{entry['id']} frame {i} tick {obs['tick']}: {key} differs")
        raw = env.privileged_state()
        assert raw['tick'] == public['tick']
        truth_frame = _frame(raw)
        truth_frame['sun_income_rate'] = public['sun_income_rate']
        public['action'] = truth_frame['action'] = archived.get('action')
        fog = np.asarray(public['fog_amount']) > 0
        tick = public['tick']
        last_visible[~fog] = tick
        age = np.where(last_visible >= 0, tick - last_visible, -1)
        if blover_pending and not fog.any():
            cleared = True
        elif blover_pending and cleared and fog.any():
            events.append(dict(index=i, tick=tick))
            blover_pending, cleared = False, False
        for j, (name, frame) in enumerate(zip(('human', 'xray'), (public, truth_frame))):
            b, s = encode_public(histories[j], frame, previous[j], cv)
            buffers[f'{name}_board'].append(b)
            buffers[f'{name}_scalar'].append(s)
            previous[j] = frame
        for name, value in dict(target=labels(raw), fog=fog, age=age, tick=tick,
                                dt=tick - (frames[i-1]['tick'] if i else tick)).items():
            buffers[name].append(value)
    target = OUT / 'episodes' / f"{entry['id']}.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **{k: np.asarray(v) for k, v in buffers.items()})
    result = dict(id=entry['id'], split=entry['split'], level=task['level'], seed=task['seed'],
        source_sha256=entry['source_sha256'], pack_sha256=digest(target),
        frames=len(frames), checked_public_frames=len(frames), mismatches=0,
        hidden_cells=int(np.sum(buffers['fog'])),
        hidden_occupied=int(np.sum(np.asarray(buffers['fog']) * np.asarray(buffers['target'])[:, 0])),
        returns=events, board_shape=list(b.shape), scalar_shape=list(s.shape))
    write_json(target.with_suffix('.json'), result)
    return result


def collect(limit=None):
    os.environ['PVZ_NO_POPUP'] = '1'
    entries = inventory()
    results = []
    # One live game process across the entire serial collection.
    with EventWaitEnv('/home/newbiexvwu/.cache/pvz-research-resources', executable=ENGINE) as env:
        for entry in entries[:limit]:
            pack = OUT / 'episodes' / f"{entry['id']}.npz"
            meta = pack.with_suffix('.json')
            if pack.exists() and meta.exists():
                result = json.loads(meta.read_text())
                assert digest(pack) == result['pack_sha256']
                assert result['source_sha256'] == entry['source_sha256']
            else:
                result = replay(entry, env)
            results.append(result)
            print(json.dumps(result), flush=True)
    write_json(OUT / 'corpus.json', dict(engine=str(ENGINE), engine_sha256=digest(ENGINE),
        inventory_sha256=digest(OUT / 'inventory.json'), episodes=results))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['inventory', 'collect'])
    p.add_argument('--limit', type=int)
    args = p.parse_args()
    inventory() if args.stage == 'inventory' else collect(args.limit)
