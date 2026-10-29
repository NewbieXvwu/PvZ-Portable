"""task171 / E8: sample retained r42 rows and repeat the current teacher search.

Archives, packs and production are read-only. Outputs are local audit observations.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
from dataclasses import fields
import itertools
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]
import numpy as np


def category(label):
    if label.startswith('z'):
        return 'history'
    if 'rd137' in label:
        return 'random'
    if 'roof' in label:
        return 'roof'
    if 'fog' in label:
        return 'fog'
    return 'other'


def spread(items, count):
    return [items[int(i)] for i in np.linspace(0, len(items) - 1, min(count, len(items)))]


def remote_copy(requests, host):
    """Stream only requested files over the existing SSH connection, without remote writes."""
    requests = {src: str(dst) for src, dst in requests.items() if not Path(dst).exists()}
    if not requests:
        return
    code = ('import base64,json\nfrom pathlib import Path\n'
            f'paths={list(requests)!r}\n'
            'for name in paths:\n'
            ' p=Path(name)\n'
            ' if p.exists():\n'
            '  print(json.dumps([name,base64.b64encode(p.read_bytes()).decode()]),flush=True)\n')
    proc = subprocess.Popen(['ssh', '-o', 'ConnectTimeout=20', host,
                             'wsl', '-d', 'Ubuntu', '-e', 'python3', '-'],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    proc.stdin.write(code)
    proc.stdin.close()
    copied = 0
    for line in proc.stdout:
        src, data = json.loads(line)
        dst = Path(requests[src])
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(base64.b64decode(data))
        copied += 1
        if copied % 20 == 0:
            print(json.dumps({'copied_files': copied, 'requested': len(requests)}), flush=True)
    if proc.wait():
        raise RuntimeError('SSH read interrupted; completed local files can be reused')


def archive_path(source, args):
    for root in (args.archives, args.out / 'frames'):
        p = root / source
        if p.exists():
            return p
        if p.with_suffix('.jsonl').exists():
            return p.with_suffix('.jsonl')
    return args.out / 'frames' / source


def demo_path(remote, args):
    return args.out / 'desktop' / remote.lstrip('/')


def prepare(args):
    from pvz_frame_store import read_frames
    with np.load(args.pack) as stored:
        meta = json.loads(str(stored['meta']))
        segments = dict(zip(meta['labels'], stored['segments'].tolist()))
    groups = defaultdict(list)
    for label in segments:
        groups[category(label)].append(label)
    chosen = []
    for cat, n in [('history', 12), ('roof', 10), ('fog', 10), ('random', 10), ('other', 8)]:
        labels = sorted(groups[cat])
        if cat in ('roof', 'fog'):
            labels = spread([s for s in labels if 'demo' not in s], n - 2) + spread(
                [s for s in labels if 'demo' in s], 2)
        else:
            labels = spread(labels, n)
        if cat == 'history':
            labels += [s for s in sorted(groups[cat]) if s not in labels][:2]
        chosen.extend(labels)
    rng = random.Random(171)
    segment_rows = []
    requests = {}
    for label in chosen:
        name = segments[label]
        with np.load(args.pack.parent / name) as stored:
            sources = stored['source'].tolist()
            archive = stored['source_archive']
            decisions = stored['source_row']
            weights = stored['weight']
        ids = list(range(len(sources)))
        rng.shuffle(ids)
        for a in ids[:4]:
            src = sources[a]
            path = archive_path(src, args)
            rows = np.flatnonzero(archive == a)
            segment_rows.append((label, name, src, rows, decisions[rows], weights[rows]))
            if not path.exists():
                for filename in ('meta.json', 'frames.npz'):
                    remote = args.desktop_frames.rstrip('/') + '/' + str(Path(src).parent / filename)
                    requests[remote] = args.out / 'frames' / Path(src).parent / filename
    remote_copy(requests, args.host)
    pools = defaultdict(list)
    info = {}
    demos = {}
    for label, segment, src, rows, decisions, weights in segment_rows:
        path = archive_path(src, args)
        m = json.loads((path.parent / 'meta.json').read_text())
        frames = read_frames(path)
        task = m['task']
        level = task.get('level', m.get('context', {}).get('level'))
        deck = task.get('deck', [p[1] for p in frames[0]['packets']])
        version = task.get('checkpoint', task.get('policy', 'unknown'))
        version = re.sub(r'^.*?(round_\d+|v\d+)/(?:student/)?model.pt$', r'\1', version)
        terrain = ('night' if level == 35 else 'day' if level <= 10 else
                   'night' if level <= 20 else 'pool' if level <= 30 else
                   'fog' if level <= 40 else 'roof')
        signature = ','.join(map(str, sorted(deck)))
        bucket = f'{label}|{version}|L{level}:{terrain}|{signature}'
        info[bucket] = dict(label=label, teacher=version, level=level, terrain=terrain,
                            deck=deck, category=category(label))
        for row, decision, weight in zip(rows, decisions, weights):
            i = int(decision)
            pools[bucket].append(dict(bucket=bucket, source=src, segment=segment,
                                      packed_row=int(row), decision=i, weight=float(weight),
                                      old=frames[i].get('label') or frames[i].get('action')))
        if m.get('demo_archive'):
            remote = m['demo_archive']
            for filename in ('meta.json', 'frames.npz', 'frames.jsonl', 'ticks.jsonl'):
                demos[remote + '/' + filename] = demo_path(remote, args) / filename
    remote_copy(demos, args.host)
    # Each output line is an actual selected pack row, with its bucket description.
    with (args.out / 'points.jsonl').open('w') as stream:
        for bucket in sorted(pools):
            pool = pools[bucket]
            if len(pool) < args.rows:
                print(json.dumps({'undersized_bucket': bucket, 'rows': len(pool)}), flush=True)
                continue
            for point in rng.sample(pool, args.rows):
                stream.write(json.dumps({**point, **info[bucket]}) + '\n')
    print(json.dumps({'selected_buckets': sum(len(v) >= args.rows for v in pools.values()),
                      'selected_rows': sum(args.rows for v in pools.values() if len(v) >= args.rows)}), flush=True)


def normalized(action, frame, *, lane=False):
    if action['type'] == 'plant':
        packet = frame['packets'][action['packet']]
        return ('plant', packet[1], action['row'], None if lane else action['col'])
    if action['type'] == 'wait':
        return ('wait', action.get('ticks', 60), action.get('until', 'timeout'))
    return (action['type'], action.get('row'), action.get('col'))


def reset_game(env, frames, meta, args):
    from pvz_env import TaskSpec, profile_for_deck
    from pvz_student_observation import StudentHistory
    import search_play as teacher
    task = meta['task']
    state = {}
    teacher._W['demo_actions'] = []
    if meta.get('demo_archive'):
        from human_demo import reset_archive
        demo = demo_path(meta['demo_archive'], args)
        dm = json.loads((demo / 'meta.json').read_text())
        from pvz_frame_store import frame_path, read_frames
        prefix = read_frames(frame_path(demo))[:meta['demo_decision']]
        fog = task.get('fog_observation')
        obs = reset_archive(env, dm, fog_observation=fog)
        stop = prefix[-1]['operation']
        with (demo / 'ticks.jsonl').open() as stream:
            for operation, line in enumerate(stream, 1):
                if operation > stop:
                    break
                obs, *_ = env.step(json.loads(line)['action'])
        # The directory's future seed is recorded separately from the demonstration reset seed.
        future = int(frames_source_seed(meta, args))
        env.reseed(future)
        state['frame'] = prefix[-2] if len(prefix) > 1 else None
        state['action'] = prefix[-1].get('action')
        state['history'] = StudentHistory(teacher._W['fast'].obs)
        for frame in prefix[:-1]:
            state['history'].extra(frame)
        all_demo = read_frames(frame_path(demo))
        teacher._W['demo_actions'] = [(a['tick'], b['action']) for a, b in zip(all_demo, all_demo[1:])
                                      if (b.get('action') or {}).get('type') == 'plant']
    else:
        deck = task.get('deck', [p[1] for p in frames[0]['packets']])
        config = {f.name: task[f.name] for f in fields(TaskSpec) if f.name in task and f.name != 'profile'}
        config.setdefault('level', meta['context']['level'])
        config.setdefault('zombie_count_multiplier', meta.get('context', {}).get('multiplier', 1.0))
        # Old task metadata omitted the limits; their compact scene retained them.
        scene = frames[0].get('scene') or {}
        config.setdefault('wave_cap', scene.get('wave_cap'))
        config['preplanted'] = tuple(tuple(p) for p in config.get('preplanted', ()))
        obs, _ = env.reset(deck=deck, task=TaskSpec(
            profile=profile_for_deck(deck, task.get('first_aid_owned', False)), **config))
    from pvz_value_net import observation_context
    teacher._W['context'] = observation_context(obs)
    return obs, state


def frames_source_seed(meta, args):
    from pvz_frame_store import archive_seed
    return archive_seed(args.current_source.parts[0])


def outcome_key(env, root, action):
    """Observable action result on the same RNG, used only for disputed labels."""
    from render_episode import _frame
    env.restore_fast(root)
    obs, _, _, _, info = env.step(action)
    if not info['ok']:
        return None
    # Instant plants may disappear; compare after the same 100 ticks, including economy/cooldowns.
    if not obs['terminal']:
        obs, *_ = env.step({'type': 'wait', 'ticks': 100})
    result = _frame(obs)
    for field in ('action', 'action_rejected', 'label', 'targets'):
        result.pop(field, None)
    result['packets'] = sorted([p[1:] for p in result['packets']])
    return json.dumps(result, sort_keys=True)


def run(args):
    import search_play as teacher
    from pvz_event_env import EventWaitEnv
    from pvz_frame_store import read_frames
    points = [json.loads(line) for line in (args.out / 'points.jsonl').read_text().splitlines()]
    dest = args.out / 'results.jsonl'
    done = set()
    if dest.exists():
        done = {(p['source'], p['decision']) for p in map(json.loads, dest.read_text().splitlines())}
    by_archive = defaultdict(list)
    for point in points:
        if (point['source'], point['decision']) not in done:
            by_archive[point['source']].append(point)
    keep, value_bins = teacher._export_values(str(args.value), str(args.static_value))
    teacher._init_worker(value_bins[0], str(args.student), 61, value_bins[1])
    search_args = teacher.parser().parse_args([
        '--rollout-policy', str(args.student), '--value', str(args.value),
        '--static-value', str(args.static_value), '--manifest', 'unused', '--task-id', 'audit',
        '--out', str(args.out), '--verify-samples', '9', '--student-candidates', '2',
        '--save-candidates', '1'])
    env = EventWaitEnv(str(args.resource_dir), executable=str(args.executable), headless=True)
    teacher._W['env'] = env
    completed = 0
    started = time.time()
    try:
        env.load_rollout_net(teacher._W['rollout_file'])
        env.load_value_net(value_bins[0])
        env.load_value_net(value_bins[1], static=True)
        with dest.open('a', buffering=1) as stream:
            for source, group in by_archive.items():
                path = archive_path(source, args)
                frames = read_frames(path)
                meta = json.loads((path.parent / 'meta.json').read_text())
                args.current_source = Path(source)
                obs, state = reset_game(env, frames, meta, args)
                selected = {p['decision']: p for p in group}
                last_search = -10**9
                for i in range(1, max(selected) + 1):
                    # Feed all archived prefix frames to the current student's history.
                    proposal = teacher._fast_action(obs, state)
                    teacher._W['student_history'] = state.get('history')
                    if i in selected:
                        point = selected[i]
                        root = env.snapshot_fast()
                        choices = []
                        details = []
                        t0 = time.time()
                        try:
                            for repeat in range(args.repeats):
                                env.restore_fast(root)
                                stats = dict(searched=0, overrides=0, override_kinds=Counter(),
                                             last_search_tick=last_search)
                                seed = (171000003 + frames_source_seed(meta, args) * 10000 + i * 37
                                        + repeat * 10000019) % 2**32
                                choices.append(teacher._search(env, obs, proposal, search_args, seed, stats))
                                details.append(dict(searched=stats['searched'],
                                                    best=stats.get('last_best'),
                                                    targets=stats.get('last_targets')))
                            before = frames[i - 1]
                            actions = [point['old'], *choices]
                            unique = {normalized(a, before): a for a in actions}
                            outcomes = {key: outcome_key(env, root, a) for key, a in unique.items()} if len(unique) > 1 else {}
                            pairs = list(itertools.combinations(range(len(actions)), 2))
                            same_result = [(a, b) for a, b in pairs
                                           if normalized(actions[a], before) != normalized(actions[b], before)
                                           and outcomes.get(normalized(actions[a], before)) is not None
                                           and outcomes[normalized(actions[a], before)] == outcomes[normalized(actions[b], before)]]
                            result = {**point, 'tick': obs['tick'], 'choices': choices, 'searches': details,
                                      'same_result_pairs': same_result,
                                      'seconds': time.time() - t0}
                            stream.write(json.dumps(result) + '\n')
                            completed += 1
                        finally:
                            env.restore_fast(root)
                            env.release_snapshot(root)
                        if completed % 10 == 0 or completed == 1:
                            print(json.dumps({'done': len(done) + completed, 'total': len(points),
                                              'seconds': round(time.time() - started, 1),
                                              'last_seconds': round(result['seconds'], 2),
                                              'source': source}), flush=True)
                        if args.limit and completed >= args.limit:
                            return
                    if i == max(selected):
                        break
                    state['action'] = frames[i]['action']
                    if frames[i].get('targets'):
                        saved = env.snapshot_fast()
                        env.restore_fast(saved)
                        env.release_snapshot(saved)
                        last_search = obs['tick']
                    obs, *_, info = env.step(frames[i]['action'])
                    if not info['ok']:
                        obs, *_ = env.step({'type': 'wait', 'ticks': 60})
    finally:
        env.close()
        keep.cleanup()


def summarize(args):
    from pvz_frame_store import read_frames
    buckets = defaultdict(Counter)
    current_source, frames = None, None
    dest = args.out / 'results.jsonl'
    if not dest.exists():
        return
    for line in dest.read_text().splitlines():
        p = json.loads(line)
        if p['source'] != current_source:
            current_source = p['source']
            frames = read_frames(archive_path(current_source, args))
        before = frames[p['decision'] - 1]
        count = buckets[p['bucket']]
        choices = p['choices']
        old = normalized(p['old'], before)
        keys = [normalized(a, before) for a in choices]
        count['rows'] += 1
        count['old_agree'] += old == keys[0]
        count['old_comparisons'] += len(keys)
        count['old_agree_all'] += sum(old == key for key in keys)
        for a, b in itertools.combinations(range(len(keys)), 2):
            count['repeat_pairs'] += 1
            count['repeat_agree'] += keys[a] == keys[b]
        actions = [p['old'], *choices]
        all_keys = [old, *keys]
        disputes = [(a, b) for a, b in itertools.combinations(range(len(actions)), 2)
                    if all_keys[a] != all_keys[b]]
        count['disputed_rows'] += bool(disputes)
        count['same_result_noise_rows'] += bool(p['same_result_pairs'])
        count['same_lane_different_col_rows'] += any(
            normalized(actions[a], before, lane=True) == normalized(actions[b], before, lane=True)
            and actions[a]['type'] == 'plant' for a, b in disputes)
        count['slot_only_rows'] += any(actions[a] != actions[b] and all_keys[a] == all_keys[b]
                                      and actions[a]['type'] == 'plant'
                                      for a, b in itertools.combinations(range(len(actions)), 2))
        count['searched_calls'] += sum(s['searched'] for s in p['searches'])
    result = {}
    for bucket, c in buckets.items():
        n = c['rows']
        result[bucket] = {**c, 'old_top1_rate': c['old_agree'] / n,
                          'repeat_top1_rate': c['repeat_agree'] / c['repeat_pairs'],
                          'same_result_noise_rate': c['same_result_noise_rows'] / n,
                          'same_lane_different_col_rate': c['same_lane_different_col_rows'] / n,
                          'E8_downweight': c['old_agree'] / n < .7,
                          'E8_consider_soft': c['same_result_noise_rows'] / n > .2}
    (args.out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'rows': sum(c['rows'] for c in buckets.values()), 'buckets': len(buckets)}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('prepare', 'run', 'summarize'))
    p.add_argument('--pack', type=Path, default=Path.home() / 'pvz-bench-data/r42pack/packs/b3163458970ae486.npz')
    p.add_argument('--archives', type=Path, default=Path.home() / 'PvZAgent/artifacts/hard_mac/frames')
    p.add_argument('--out', type=Path, default=Path.home() / 'pvz-bench-data/task171')
    p.add_argument('--host', default='newbiexvwu@100.120.155.85')
    p.add_argument('--desktop-frames', default='/home/newbiexvwu/PvZ-search/artifacts/student/frames_v2')
    p.add_argument('--rows', type=int, default=200)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--student', type=Path, default=Path.home() / 'pvz-bench-data/r36/model.pt')
    p.add_argument('--value', type=Path, default=Path.home() / 'pvz-bench-data/task171/teacher/value.pt')
    p.add_argument('--static-value', type=Path, default=Path.home() / 'pvz-bench-data/task171/teacher/static.pt')
    p.add_argument('--executable', type=Path, default=Path.home() / 'PvZAgent/build/pvz-portable')
    p.add_argument('--resource-dir', type=Path,
                   default=Path.home() / 'Downloads/Plants_Vs_Zombies_V1.2.0.1073_EN')
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    globals()[args.mode](args)


if __name__ == '__main__':
    main()
