"""Validate the four frozen task69 demonstrations and export native-legal replacement labels.

This batch needs four free game slots. The task runner schedules it after A/B, so it never
holds up the paired manager. Original HF archives stay unchanged; derived tapes/NPZ stay ignored.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def task_game_pids():
    active = []
    for entry in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            arguments = entry.read_bytes().split(b'\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if not arguments or not arguments[0]:
            continue
        binary = Path(arguments[0].decode(errors='replace'))
        if binary.is_relative_to(ROOT) and binary.name in ('pvz-portable', 'baseline-engine', 'treatment-engine'):
            active.append(int(entry.parent.name))
    return active


def validate_one(job):
    item, plan, plan_hash, out, resource, executable = job
    import torch
    torch.set_num_threads(1)
    from human_demo import export, rebaseline, replacement_pairs
    from pvz_frame_store import cache_episode, frame_bytes, frame_path, read_frames
    import numpy as np

    source = Path(item['source'])
    for name, expected in item['source_hashes'].items():
        if digest(source / name) != expected:
            raise RuntimeError(f'original demonstration changed: {source / name}')
    result_file = out / 'results' / (item['name'] + '.json')
    if result_file.exists():
        existing = json.loads(result_file.read_text())
        if (existing['N_plan_sha256'] != plan_hash or existing['engine_sha256'] != digest(executable)
                or existing['tool_sha256'] != digest(ROOT / 'scripts/human_demo.py')
                or existing['batch_tool_sha256'] != digest(__file__)):
            raise RuntimeError('existing validation belongs to another plan or engine')
        for key in ('derived_tape', 'training_frames'):
            if digest(existing[key]['path']) != existing[key]['sha256']:
                raise RuntimeError(f'existing validation file changed: {key}')
        return existing

    log_path = out / 'logs' / (item['name'] + '.log')
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a', buffering=1) as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        derived = out / 'validated' / item['name']
        if derived.exists():
            raise RuntimeError(f'incomplete prior validation retained for inspection: {derived}')
        print('REBASELINE', item['name'], flush=True)
        verification = rebaseline(source, derived, resource, executable)
        derived_meta = json.loads((derived / 'meta.json').read_text())
        source_frames = read_frames(frame_path(source))
        if digest(frame_path(derived)) != item['source_hashes'][frame_path(source).name]:
            raise RuntimeError('rebaseline changed the original public frames')
        if derived_meta['rebaseline']['checked_public_frames'] != len(source_frames):
            raise RuntimeError('not every original public frame was checked')
        pairs, distribution = replacement_pairs(source_frames)
        if distribution['intervals'] != item['distribution']['intervals']:
            raise RuntimeError('observed intervals differ from the frozen plan')
        if any(pair['interval'] > plan['N'] for pair in pairs):
            raise RuntimeError('pair exceeds the frozen N')
        print('EXPORT', item['name'], flush=True)
        target = export(derived, out / 'training', resource, executable)
        meta_path = target / 'meta.json'
        meta = json.loads(meta_path.read_text())
        pairing = meta['replacement_pairing']
        pairing['observed_archive_max_interval'] = pairing['N']
        pairing.update(N=plan['N'], N_rule=plan['rule'], N_plan_sha256=plan_hash)
        write_json(meta_path, meta)
        frames_path = frame_path(target)
        frames = read_frames(frames_path)
        labelled = {i: frame['label'] for i, frame in enumerate(frames)
                    if (frame.get('label') or {}).get('type') == 'replace'}
        if len(labelled) != pairing['labelled']:
            raise RuntimeError('metadata label count differs from archived labels')
        for index, (old, new) in enumerate(zip(source_frames, frames)):
            without_label = {key: value for key, value in new.items() if key != 'label'}
            if old != without_label:
                raise RuntimeError(f'original frame/action changed at frame {index}')
        if len(source_frames) != len(frames):
            raise RuntimeError('frame count changed during export')
        cache = cache_episode(frames_path, out / 'cache', 'v2')
        with np.load(cache, allow_pickle=False) as encoded:
            rows = {int(index): row for row, index in enumerate(encoded['rows'])}
            for index, action in labelled.items():
                row = rows[index]
                if not (encoded['plant'][row] == 1 and encoded['packet'][row] == action['packet'] and
                        encoded['cell'][row] == action['row'] * 9 + action['col']):
                    raise RuntimeError(f'training cache label differs at frame {index}')
            cache_shape = dict(boards=list(encoded['boards'].shape), scalars=list(encoded['scalars'].shape))
        by_frame = {pair['shovel_frame']: pair for pair in item['pairs']}
        transitions = Counter()
        for index in labelled:
            pair = by_frame[index]
            transitions[str(pair['removed_types']) + ' -> ' + str(pair['new_type'])] += 1
        invalid = [dict(frame=index, **{key: by_frame[index][key] for key in
                    ('interval', 'action', 'removed_types', 'new_type', 'sun_at_shovel', 'packet_at_shovel')})
                   for index in pairing['illegal_at_shovel']]
        result = dict(name=item['name'], task=item['task'], N=plan['N'], N_plan_sha256=plan_hash,
            engine_sha256=digest(executable), tool_sha256=digest(ROOT / 'scripts/human_demo.py'),
            batch_tool_sha256=digest(__file__),
            source_hashes=item['source_hashes'], distribution=distribution, native_pairing=pairing,
            invalid_at_shovel=invalid, labelled_transitions=dict(transitions),
            verified_replay=verification, checked_public_frames=len(source_frames),
            rebaseline=derived_meta['rebaseline'], training_cache_shape=cache_shape,
            derived_tape=dict(path=str(derived / 'ticks.jsonl'), sha256=digest(derived / 'ticks.jsonl')),
            training_frames=dict(path=str(frames_path), sha256=digest(frames_path)))
        for name, expected in item['source_hashes'].items():
            if digest(source / name) != expected:
                raise RuntimeError('original input changed during validation')
        write_json(result_file, result)
        print('DONE', item['name'], 'labels', pairing['labelled'], flush=True)
        return result


def run(plan_path, out, resource, executable):
    plan = json.loads(plan_path.read_text())
    if len(plan['archives']) != 4 or plan['N'] != max(plan['intervals']):
        raise ValueError('requires the frozen four-demo interval plan')
    active = task_game_pids()
    if active:
        raise RuntimeError(f'four demo slots are not free: {active}')
    out.mkdir(parents=True, exist_ok=True)
    plan_hash = digest(plan_path)
    jobs = [(item, plan, plan_hash, out, resource, executable) for item in plan['archives']]
    results = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(validate_one, job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print('VALIDATED', result['name'], result['native_pairing']['labelled'], flush=True)
    results.sort(key=lambda result: result['name'])
    transitions = Counter()
    for result in results:
        transitions.update(result['labelled_transitions'])
    summary = dict(N=plan['N'], N_rule=plan['rule'], N_plan_sha256=plan_hash, source=plan['source'],
        archives=results, shovel_attempts=plan['shovel_attempts'],
        successful_shovels=plan['successful_shovels'], rejected_shovels=plan['rejected_shovels'],
        paired=sum(result['distribution']['paired'] for result in results),
        unmatched_successful_shovels=sum(len(result['distribution']['unmatched_shovels']) for result in results),
        labelled=sum(result['native_pairing']['labelled'] for result in results),
        illegal_at_shovel=sum(len(result['native_pairing']['illegal_at_shovel']) for result in results),
        input_board_mismatch=sum(len(result['native_pairing']['input_board_mismatch']) for result in results),
        labelled_transitions=dict(transitions),
        checked_public_frames=sum(result['checked_public_frames'] for result in results),
        verified_operations=sum(result['verified_replay']['operations'] for result in results),
        originals_unchanged=True, labels_native_validated=True)
    write_json(out / 'human-pairing-results.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('resource', type=Path)
    parser.add_argument('executable', type=Path)
    args = parser.parse_args()
    run(args.plan.resolve(), args.out.resolve(), args.resource.resolve(), args.executable.resolve())


if __name__ == '__main__':
    main()
