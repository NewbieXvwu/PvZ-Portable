"""Task78-only native-trace student history; frozen teacher search and outcomes unchanged."""
from collections import Counter
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import time

import search_play as teacher
from value_teacher_feasibility import (cpu, history_record, write_json, leaf_checks,
    finish_teacher, validate_existing_candidate, group_summary)
from task78_native_history import POLICY, recover, context_digest


def pack_trace(path):
    """Losslessly retain the numerical input trace without another replay/readback."""
    path = Path(path)
    target = path.with_suffix(path.suffix + '.gz')
    temporary = target.with_suffix(target.suffix + '.tmp')
    with path.open('rb') as source, temporary.open('wb') as destination:
        with gzip.GzipFile(filename='', fileobj=destination, mode='wb', compresslevel=1, mtime=0) as packed:
            shutil.copyfileobj(source, packed)
    temporary.replace(target)
    path.unlink()
    return dict(trace_file=target.name,
                packed_trace_sha256=hashlib.sha256(target.read_bytes()).hexdigest())


def sample_root(env, obs, state, args, decision_seed, stats, directory):
    """Retain native leaves and recover their student inputs from native TRACE."""
    from render_episode import _frame
    directory.mkdir(parents=True, exist_ok=True)
    root = env.snapshot()
    root_hash = env._debug_state_sha256()
    baseline = dict(teacher._W)
    original = teacher._rollout_policy
    captured = []
    common = (decision_seed * 1000) % 2**32

    def retain(environment, current, first, until_tick, parent=None, seed=None):
        if seed != common:
            return original(environment, current, first, until_tick, parent, seed)
        trace = directory / f'native_candidate_{len(captured)}.jsonl'
        history = teacher._W.get('student_history') if teacher._W['fast'].obs != 'v1' else None
        leaf = environment.rollout(first, until_tick, sim_skip=teacher._W.get('rollout_sim_skip', 0),
            snapshot_id=parent, seed=seed, student_history=history, trace_path=trace)
        response = dict(environment._last_response)
        metrics = teacher._W.setdefault('rollout_metrics', {})
        for name in ('decisions', 'simulation_seconds', 'network_seconds'):
            metrics[name] = metrics.get(name, 0) + response[name]
        leaf['value'] = environment.leaf_values(income=leaf['sun_income_rate'])[0]
        if seed == common:
            captured.append(dict(action=deepcopy(first), leaf=deepcopy(leaf),
                snapshot=environment.snapshot(), state_hash=environment._debug_state_sha256(),
                income=list(environment._sun_production_history),
                income_start=environment._sun_history_start_tick, trace=trace,
                response={k: response[k] for k in ('decisions', 'substeps', 'planted_slots')}))
        return leaf

    started = cpu(env)
    try:
        teacher._rollout_policy = retain
        proposed = state['action']
        chosen = teacher._search(env, obs, proposed, args, decision_seed, stats)
    finally:
        teacher._rollout_policy = original
    search_cpu = cpu(env) - started
    targets = [a for a, _ in stats['last_targets']]
    key = lambda a: json.dumps(a, sort_keys=True)
    if Counter(map(key, targets)) != Counter(key(x['action']) for x in captured):
        raise RuntimeError('captured first-future leaves do not cover the actual candidate set exactly')
    # Search mutates caches, not state/history; preserve post-search caches for the root trajectory.
    post_search = dict(teacher._W)
    directory.mkdir(parents=True, exist_ok=True)
    root_record = dict(decision_seed=decision_seed, common_future=common,
        root_frame=_frame(obs), root_state_sha256=root_hash, history=history_record(state),
        sun_income_history=dict(start_tick=env._sun_history_start_tick,
                                production=list(env._sun_production_history)),
        context=teacher._W['context'].tolist(), candidates=targets, chosen=chosen,
        search_cpu_seconds=search_cpu)
    root_path = directory / 'root.json'
    if root_path.exists():
        previous = json.loads(root_path.read_text())
        for key in root_record.keys() - {'search_cpu_seconds'}:
            if previous[key] != json.loads(json.dumps(root_record[key])):
                raise ValueError(f'existing root differs on {key}: {root_path}')
    write_json(root_path, root_record)
    rows = []
    try:
        for index, item in enumerate(captured):
            if teacher._W.get('diagnostic_progress'):
                teacher._W['diagnostic_progress']()
            before = cpu(env)
            teacher._W.clear(); teacher._W.update(baseline)
            teacher._W['student_history'] = deepcopy(state.get('history'))
            net = teacher._W['fast']
            recovered, proof = recover(item['trace'], state, _frame(obs), item['leaf'],
                item['response'], teacher._W.get('context') if net.context else None,
                net.obs, identity_packet=net.packet_head == 'identity')
            restored = env.restore(item['snapshot'])
            checks = leaf_checks(_frame(restored) == _frame(item['leaf']),
                env._debug_state_sha256() == item['state_hash'],
                list(env._sun_production_history) == item['income'],
                env._sun_history_start_tick == item['income_start'], item['leaf'])
            checks['native_history'] = True
            if not all(checks.values()):
                write_json(directory / f'candidate_{index}_mismatch.json',
                    dict(checks=checks, native=_frame(item['leaf']), restored=_frame(restored)))
                raise RuntimeError(f'leaf history reconstruction mismatch: {checks}')
            proof.update(pack_trace(item['trace']))
            replay_cpu = cpu(env) - before
            teacher._W.clear(); teacher._W.update(baseline)
            # Preserve the same search parameters while giving every candidate the same
            # continuation search-noise schedule (indexed by continuation decision ordinal).
            seed_base = decision_seed + 5000
            initial_history = history_record(recovered)
            leaf_frame = _frame(item['leaf'])
            candidate_path = directory / f'candidate_{index}.json'
            if candidate_path.exists():
                previous = json.loads(candidate_path.read_text())
                if context_digest(previous['leaf_history']) != proof['context_sha256']:
                    raise ValueError('existing candidate student input history differs')
                old_checks = previous['history_checks']
                if not all(old_checks.get(k) is True for k in ('frame', 'engine_state', 'income', 'income_start')):
                    raise ValueError('existing candidate has incomplete leaf checks')
                expected = dict(candidate=index, action=item['action'], common_future=common,
                    continuation_seed_base=seed_base, leaf_frame=leaf_frame,
                    leaf_state_sha256=item['state_hash'], leaf_history=previous['leaf_history'],
                    sun_income_history=dict(start_tick=item['income_start'], production=item['income']),
                    history_checks=old_checks)
                validate_existing_candidate(previous, expected)
                previous.update(history_checks=checks, history_source=POLICY, native_history=proof)
                write_json(candidate_path, previous)
                rows.append(previous)
                continue
            begin = cpu(env); wall = time.monotonic()
            outcome = finish_teacher(env, item['leaf'], recovered, args, seed_base,
                                     directory / f'candidate_{index}_teacher.jsonl')
            row = dict(candidate=index, action=item['action'], common_future=common,
                continuation_seed_base=seed_base, leaf_frame=leaf_frame,
                leaf_state_sha256=item['state_hash'], leaf_history=initial_history,
                sun_income_history=dict(start_tick=item['income_start'], production=item['income']),
                history_checks=checks, history_source=POLICY, native_history=proof,
                outcome=outcome, replay_cpu_seconds=replay_cpu,
                continuation_cpu_seconds=cpu(env)-begin,
                continuation_wall_seconds=time.monotonic()-wall)
            write_json(candidate_path, row)
            rows.append(row)
            print(json.dumps(dict(seed=decision_seed//10000, root=decision_seed%10000,
                candidate=index, won=outcome['won'], cpu=row['continuation_cpu_seconds'])), flush=True)
        summary = group_summary(rows)
        summary.update(decision_seed=decision_seed, tick=obs['tick'], wave=obs['wave'],
                       candidates=len(rows), search_cpu_seconds=search_cpu)
        write_json(directory / 'summary.json', summary)
        return chosen, summary
    finally:
        teacher._W.clear(); teacher._W.update(post_search)
        env.restore(root)
        for item in captured:
            env.release_snapshot(item['snapshot'])
        env.release_snapshot(root)
        if env._debug_state_sha256() != root_hash:
            raise RuntimeError('root state changed by label collection')
