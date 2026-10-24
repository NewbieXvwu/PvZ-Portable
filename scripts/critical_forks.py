"""Archive decisions -> exhaustive legal candidates -> frozen search teacher terminals.

One collector for every terrain. Plans pin inputs, archived reset context and one-based
decision lists; terminal labels never participate in root selection. No production switches.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import fcntl
import gzip
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]

PLAN = RUN = STOP = None
TEACHER = None
LAST = 0


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=1, allow_nan=False) + '\n')
    tmp.replace(path)


def key(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def physical(frame):
    annotations = {'action', 'label', 'targets', 'verification', 'action_rejected', 'eaten'}
    return {k: v for k, v in frame.items() if k not in annotations}


def all_candidates(obs):
    plants = [dict(type='plant', packet=a['packet'], row=a['row'], col=a['col'])
              for a in obs['legal_actions'].get('plants', [])]
    plants.sort(key=lambda a: (a['packet'], a['row'], a['col']))
    shovels = [dict(type='shovel', row=a['row'] if isinstance(a, dict) else a[1],
                    col=a['col'] if isinstance(a, dict) else a[0])
               for a in obs['legal_actions'].get('shovels', [])]
    shovels.sort(key=lambda a: (a['row'], a['col']))
    actions = [dict(type='wait', ticks=60, until='timeout')] + plants + shovels
    if len({key(a) for a in actions}) != len(actions):
        raise ValueError('duplicate legal action')
    return actions


def select_points(frames, params):
    """Archive-only selection; retain every eligibility reason before applying the quota."""
    from render_episode import _mower_fired
    reasons = {}
    breaches = [i for i in range(1, len(frames)) if _mower_fired(frames[i-1], frames[i])]
    if frames[-1].get('terminal') and frames[-1].get('result') == 2:
        breaches.append(len(frames)-1)
    for breach in sorted(set(breaches)):
        for i in range(max(1, breach-params['pre_breach_decisions']+1), breach+1):
            if not frames[i-1].get('terminal'):
                reasons.setdefault(i, []).append(dict(kind='before_breach', breach=breach))
    flat = []
    for i in range(1, len(frames)):
        vals = [a[3] for a in frames[i].get('targets', []) if math.isfinite(a[3])]
        if (len(vals) >= 2 and max(vals)-min(vals) < params['flat_spread_threshold']
                and frames[-1]['tick']-frames[i-1]['tick'] <= params['flat_window_before_loss_ticks']):
            flat.append((i, max(vals)-min(vals)))
    for i, spread in flat[-params['flat_points_per_game']:] if params['flat_points_per_game'] else []:
        reasons.setdefault(i, []).append(dict(kind='flat_archived_leaf', spread=spread))
    # Reserve both requested mechanisms before applying the cap; newest points cost less.
    before = [i for i in sorted(reasons) if any(r['kind']=='before_breach' for r in reasons[i])]
    flat_selected = [i for i in sorted(reasons) if any(r['kind']=='flat_archived_leaf' for r in reasons[i])]
    chosen = list(dict.fromkeys(before[-params['pre_breach_decisions']:] + flat_selected))
    chosen = sorted(chosen, reverse=True)[:params['max_points_per_game']]
    return [dict(decision=i, tick=frames[i-1]['tick'], reasons=reasons[i]) for i in sorted(chosen)]


def future_seed(root_id, stage, k, search=False):
    text = f'critical-forks-v1:{root_id}:{stage}:{k}:{"search" if search else "world"}'
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], 'little')


def history_record(state):
    h = state.get('history')
    return dict(previous_frame=state.get('frame'), previous_action=state.get('action'),
                history=None if h is None else dict(obs=h.obs, wave=h.wave, wave_tick=h.wave_tick,
                            plants=[[*k, v] for k, v in sorted(h.plants.items())]))


def cpu(env=None):
    a = resource.getrusage(resource.RUSAGE_SELF); b = resource.getrusage(resource.RUSAGE_CHILDREN)
    total = a.ru_utime+a.ru_stime+b.ru_utime+b.ru_stime
    if env is not None and env._process is not None and env._process.poll() is None:
        fields = Path(f'/proc/{env._process.pid}/stat').read_text().rsplit(')', 1)[1].split()
        total += (int(fields[11])+int(fields[12])) / os.sysconf('SC_CLK_TCK')
    return total


def usage(env=None, force=False):
    global LAST
    now = time.monotonic()
    if force or now-LAST >= 2:
        save(RUN/f'usage-{os.getpid()}.json', dict(cpu_seconds=cpu(env), pid=os.getpid(),
             engine_pid=env._process.pid if env and env._process else None, time=time.time()))
        LAST = now
    if STOP is not None and STOP.is_set():
        raise RuntimeError('collection stopped; unfinished branches are not losses')


def artifact_bytes(out):
    return sum(p.stat().st_size for p in Path(out).rglob('*') if p.is_file() and not p.is_symlink())


def spent_cpu(out):
    total = 0.
    for run in (Path(out)/'runs').glob('*'):
        receipts = sum(load(p)['cpu_seconds'] for p in run.glob('usage-*.json'))
        if (run/'complete.json').exists():
            receipts = max(receipts, load(run/'complete.json').get('cpu_seconds', 0.))
        total += receipts
    return total


def check_plan(filename):
    if not Path(filename).is_absolute():
        raise ValueError('absolute plan required')
    plan = load(filename)
    if not 1 <= plan['workers'] <= 4:
        raise ValueError('one to four workers required')
    for name, record in plan['inputs'].items():
        if not Path(record['path']).is_absolute() or sha(record['path']) != record['sha256']:
            raise ValueError('changed/relative input: '+name)
    if not Path(plan['out']).is_absolute():
        raise ValueError('absolute output required')
    for job in plan['jobs']:
        if not Path(job['archive']).is_absolute():
            raise ValueError('absolute archive required')
    return plan


def init_worker(plan, run, stop):
    global PLAN, RUN, STOP, TEACHER
    PLAN, RUN, STOP = plan, Path(run), stop
    import search_play
    TEACHER = search_play
    TEACHER._init_worker(plan['_runtime_values'][0], plan['teacher']['rollout_policy'],
                        plan['teacher']['rollout_sim_skip'], plan['_runtime_values'][1])
    usage(force=True)


def teacher_args(job):
    args = TEACHER.parser().parse_args([
        '--rollout-policy', PLAN['teacher']['rollout_policy'], '--value', PLAN['teacher']['value'],
        '--static-value', PLAN['teacher']['static_value'], '--manifest', job['manifest'],
        '--task-id', job['task_id'], '--out', PLAN['out'], '--workers', '1',
        '--per-card', str(PLAN['teacher']['per_card']),
        '--student-candidates', str(PLAN['teacher']['student_candidates']),
        '--samples', str(PLAN['teacher']['samples']), '--horizon', str(PLAN['teacher']['horizon']),
        '--verify-samples', str(PLAN['teacher']['verify_samples']),
        '--min-gap', str(PLAN['teacher']['min_gap']), '--rollout-sim-skip', str(PLAN['teacher']['rollout_sim_skip']),
        '--executable', PLAN['teacher']['executable'], '--resource-dir', PLAN['teacher']['resource_dir']])
    return args


def replay(job):
    from pvz_env import TaskSpec
    from pvz_event_env import EventWaitEnv
    from pvz_tasks import profile_for_deck
    from pvz_frame_store import frame_path, read_frames
    from render_episode import _frame
    from pvz_value_net import observation_context
    archive = Path(job['archive']); path = frame_path(archive)
    if sha(path) != job['frames_sha256'] or sha(archive/'meta.json') != job['meta_sha256']:
        raise ValueError('archive changed')
    frames = read_frames(path); spec = deepcopy(job['reset']); deck = tuple(spec.pop('deck'))
    first_aid = spec.pop('first_aid_owned', False)
    if list(deck) != [p[1] for p in frames[0]['packets']]:
        raise ValueError('archived deck differs from frozen reset')
    spec['preplanted'] = tuple(tuple(p) for p in spec.get('preplanted', []))
    env = TEACHER._W['env'] = EventWaitEnv(PLAN['teacher']['resource_dir'],
            executable=PLAN['teacher']['executable'], headless=True)
    env.load_rollout_net(TEACHER._W['rollout_file'])
    env.load_value_net(TEACHER._W['value_files'][0])
    env.load_value_net(TEACHER._W['value_files'][1], static=True)
    obs, _ = env.reset(deck=deck, task=TaskSpec(profile=profile_for_deck(deck, first_aid), **spec))
    TEACHER._W['context'] = observation_context(obs); state = {}; last_search = -10**9
    for i in range(job['decision']):
        usage(env)
        actual, expected = physical(_frame(obs)), physical(frames[i])
        if key(actual) != key(expected):
            fields = [k for k in actual.keys()|expected.keys() if actual.get(k) != expected.get(k)]
            save(Path(PLAN['out'])/'failures'/f'{job["id"]}.json', dict(frame=i, fields=fields,
                 actual=actual, expected=expected))
            raise ValueError(f'archive replay differs at frame {i}: {fields}')
        proposed = TEACHER._fast_action(obs, state)
        TEACHER._W['student_history'] = state.get('history')
        if i == job['decision']-1:
            if job.get('student_executed', False) and proposed != frames[i+1]['action']:
                raise ValueError('archived student proposal differs')
            return env, obs, state, teacher_args(job), frames, last_search
        action = frames[i+1]['action']; state['action'] = action
        if frames[i+1].get('targets'):
            # The real teacher snapshots/restores at these exact search decisions.
            root = env.snapshot_fast(); env.restore(root); env.release_snapshot(root)
            last_search = obs['tick']
        obs, *_, info = env.step(action)
        rejected = not info.get('ok')
        if rejected != bool(frames[i+1].get('action_rejected')):
            raise ValueError('prefix acceptance differs')
        if rejected:
            obs, *_ = env.step(dict(type='wait', ticks=60))
    raise ValueError('invalid decision')


def audit_search(env, obs, state, args, seed, last_search):
    """Instrument the single formal teacher; preserve root, history and caches afterward."""
    trace = {}; original_static = TEACHER._static_scores; original_verified = TEACHER._verified_choice
    baseline = dict(TEACHER._W); before = env._debug_state_sha256()
    income = list(env._sun_production_history); start = env._sun_history_start_tick
    def static(*pos, **kw):
        result = original_static(*pos, **kw)
        trace['static_scores'] = [dict(action=a, value=float(v)) for v, a in result]
        return result
    def verified(*pos, **kw):
        trace['candidates'] = deepcopy(pos[3]); trace['halving_counts'] = list(pos[5])
        trace['pre_verification_values'] = [float(v) if math.isfinite(v) else None for v in pos[4]]
        chosen = original_verified(*pos, **kw)
        trace['final_values'] = [float(v) if math.isfinite(v) else None for v in pos[4]]
        return chosen
    stats = dict(searched=0, overrides=0, override_kinds=Counter(), last_search_tick=last_search)
    try:
        TEACHER._static_scores = static; TEACHER._verified_choice = verified
        chosen = TEACHER._search(env, obs, state['action'], args, seed, stats)
        trace.update(chosen=chosen, searched=bool(stats['searched']), margin=TEACHER.MARGIN, audit_schema=1)
        by_packet = {}
        for entry in sorted(trace.get('static_scores', []), key=lambda e:-e['value']):
            by_packet.setdefault(entry['action']['packet'], []).append(entry['action'])
        trace['static_selected'] = [a for values in by_packet.values() for a in values[:args.per_card]]
        trace['student_added'] = sorted(stats.get('last_student_keys', set()))
        trace['halving_pruned'] = [a for a, n, v in zip(trace.get('candidates', []),
             trace.get('halving_counts', []), trace.get('pre_verification_values', []))
             if n < args.samples or v is None]
        if env._debug_state_sha256() != before or income != list(env._sun_production_history) or start != env._sun_history_start_tick:
            raise ValueError('instrumented search changed root')
        return trace
    finally:
        TEACHER._static_scores = original_static; TEACHER._verified_choice = original_verified
        TEACHER._W.clear(); TEACHER._W.update(baseline)


def identity(job, action, index, stage, k):
    return dict(root=job['id'], archive_sha256=job['frames_sha256'], decision=job['decision'],
                action=action, candidate=index, stage=stage, k=k,
                future=future_seed(job['id'], stage, k), search_future=future_seed(job['id'], stage, k, True),
                inputs=PLAN['inputs'])


def validate(row, expected, trace):
    if row['identity'] != expected or row['outcome']['terminal'] is not True:
        raise ValueError('incompatible or incomplete terminal label')
    o = row['outcome']
    if o['result'] not in (1, 2) or o['won'] != (o['result'] == 1) or not 0 <= o['mowers_used'] <= 6:
        raise ValueError('invalid terminal result')
    if sha(trace) != row['trace_sha256'] or not all(row['checks'].values()):
        raise ValueError('failed trace/root checks')
    with gzip.open(trace, 'rt') as f:
        records = [json.loads(line) for line in f]
    if [r['decision'] for r in records] != list(range(o['decisions'])):
        raise ValueError('incomplete teacher trace')
    return row


def finish_teacher(env, obs, state, args, seed, trace, initial_mowers, mowers_left, leaf_tick):
    """Same decision loop as task43, with strict action acceptance and true terminals."""
    from render_episode import _frame
    stats = dict(searched=0, overrides=0, override_kinds=Counter()); leaf = None
    ready = lambda o: sum(d['state'] == 1 for d in o.get('defenses', []))
    left = mowers_left
    with gzip.open(trace, 'wt', compresslevel=1) as f:
        for decision in range(TEACHER.MAX_ACTIONS):
            usage(env)
            if obs['terminal']:
                outcome = dict(won=obs['result'] == 1, terminal=True, result=obs['result'],
                    wave=obs['wave'], tick=obs['tick'], mowers_used=initial_mowers-left, decisions=decision)
                return outcome, leaf or dict(frame=_frame(obs), value=float(outcome['won']))
            if leaf is None and obs['tick'] >= leaf_tick:
                leaf = dict(frame=_frame(obs), value=env.leaf_values(income=obs.get('sun_income_rate') or 0.0)[0])
            proposed = TEACHER._fast_action(obs, state)
            TEACHER._W['student_history'] = state.get('history')
            chosen = TEACHER._search(env, obs, proposed, args, (seed+decision) % 2**32, stats)
            f.write(json.dumps(dict(decision=decision, tick=obs['tick'], student=proposed, chosen=chosen))+'\n')
            state['action'] = chosen; obs, *_, info = env.step(chosen)
            if not info.get('ok'):
                raise ValueError('frozen teacher rejected action')
            if not obs['terminal']:
                left = ready(obs)
    raise RuntimeError('teacher action cap before terminal; no loss label written')


def collect_job(job):
    from render_episode import _frame
    env = None; folder = Path(PLAN['out'])/'roots'/job['id']; folder.mkdir(parents=True, exist_ok=True)
    try:
        env, obs, state, args, frames, last_search = replay(job)
        actions = all_candidates(obs); root = env.snapshot(); engine_hash = env._debug_state_sha256()
        income = list(env._sun_production_history); income_start = env._sun_history_start_tick
        baseline = dict(TEACHER._W)
        stable_job = {k: v for k, v in job.items() if k not in ('stage', 'samples', 'indices', 'inspect_only')}
        record = dict(job=stable_job, candidates=actions, student_action=state['action'], frame=_frame(obs),
             history=history_record(state), context=TEACHER._W['context'].tolist(),
             engine_sha256=engine_hash, income=income, income_start=income_start)
        path = folder/'root.json'
        if path.exists() and load(path) != json.loads(json.dumps(record)):
            raise ValueError('root changed')
        save(path, record)
        if not (folder/'search.json').exists() or 'audit_schema' not in load(folder/'search.json'):
            save(folder/'search.json', audit_search(env, obs, state, args,
                 job['reset']['seed']*10000+job['decision']-1, last_search))
        if job.get('inspect_only'):
            return dict(id=job['id'], candidates=len(actions), inspected=True)
        stage = job.get('stage', 'screen'); samples = job.get('samples', PLAN['screen_futures'])
        indices = job.get('indices', list(range(len(actions)))); rows = []
        initial_mowers = sum(s == 1 for _, s in frames[0]['mowers'])
        for index in indices:
            action = actions[index]
            for k in range(samples):
                usage(env); expected = identity(job, action, index, stage, k)
                dest = folder/f'{stage}_{index}_{k}.json'; trace = dest.with_suffix('.jsonl.gz')
                if dest.exists():
                    rows.append(validate(load(dest), expected, trace)); continue
                started = cpu(env)
                env.restore(root); TEACHER._W.clear(); TEACHER._W.update(baseline)
                TEACHER._W['student_history'] = deepcopy(state.get('history'))
                if (env._debug_state_sha256() != engine_hash or list(env._sun_production_history) != income
                        or env._sun_history_start_tick != income_start):
                    raise ValueError('root restore/history mismatch')
                # Formal score-only rollouts omit the public leaf and Python income history.
                # Request the same native student continuation with its full diagnostic reply.
                short = env.rollout(action, obs['tick']+args.horizon,
                    sim_skip=PLAN['teacher']['rollout_sim_skip'], snapshot_id=root,
                    seed=expected['future'], student_history=TEACHER._W.get('student_history'))
                short_value = TEACHER._score_terminal_or_value(short)
                if short_value is None:
                    short_value = env.leaf_values(income=short['sun_income_rate'])[0]
                short_frame = _frame(short)
                env.restore(root); env.reseed(expected['future'])
                TEACHER._W.clear(); TEACHER._W.update(baseline)
                actual_state = deepcopy(state); actual_state['action'] = action
                left = sum(d['state'] == 1 for d in obs.get('defenses', []))
                after, *_, info = env.step(action)
                if not info.get('ok'):
                    raise ValueError('engine legal candidate rejected')
                post_action = _frame(after)
                if not after['terminal']:
                    left = sum(d['state'] == 1 for d in after.get('defenses', []))
                outcome, leaf = finish_teacher(env, after, actual_state, args, expected['search_future'],
                                              trace, initial_mowers, left, obs['tick']+args.horizon)
                row = dict(identity=expected, outcome=outcome, v4_short=float(short_value), short_leaf=short_frame,
                     post_action_frame=post_action,
                     teacher_leaf=leaf, terminal_engine_sha256=env._debug_state_sha256(), trace_sha256=sha(trace),
                     checks=dict(root=True, income=True, history=True, legal=True, terminal=True, strict_actions=True),
                     cpu_seconds=cpu(env)-started)
                validate(row, expected, trace); save(dest, row); rows.append(row); usage(env, True)
                print(json.dumps(dict(root=job['id'], stage=stage, candidate=index, k=k,
                     won=outcome['won'], wave=outcome['wave'], cpu=round(row['cpu_seconds'], 3))), flush=True)
        env.restore(root)
        if env._debug_state_sha256() != engine_hash:
            raise ValueError('root changed after labeling')
        env.release_snapshot(root)
        save(folder/f'{stage}-complete.json', dict(indices=indices, samples=samples, labels=len(rows)))
        return dict(id=job['id'], stage=stage, labels=len(rows))
    finally:
        TEACHER._close_worker_env(); usage(force=True)


def run(plan, jobs, phase):
    out = Path(plan['out']); out.mkdir(parents=True, exist_ok=True)
    with (out/'collector.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_dir = out/'runs'/f'{int(time.time())}-{phase}'; run_dir.mkdir(parents=True)
        own_start = cpu(); stop = mp.get_context('fork').Event()
        import search_play as teacher
        value_keep, values = teacher._export_values(plan['teacher']['value'], plan['teacher']['static_value'])
        teacher._load_nets(values[0], plan['teacher']['rollout_policy'], values[1])
        plan = dict(plan, _runtime_values=values)
        start_ticks = Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]
        save(run_dir/'controller.json', dict(pid=os.getpid(), start_ticks=start_ticks, phase=phase, jobs=len(jobs)))
        results = []
        with mp.get_context('fork').Pool(plan['workers'], initializer=init_worker,
                                       initargs=(plan, str(run_dir), stop)) as pool:
            pending = pool.imap_unordered(collect_job, jobs)
            try:
                while len(results) < len(jobs):
                    if artifact_bytes(out) > plan['artifact_budget_bytes']:
                        stop.set(); raise RuntimeError('5GB artifact limit reached')
                    if spent_cpu(out) > plan.get('cpu_budget_seconds', math.inf):
                        stop.set(); raise RuntimeError('CPU budget reached')
                    save(run_dir/'progress.json', dict(completed=len(results), jobs=len(jobs), cpu_seconds=spent_cpu(out)))
                    try:
                        row = pending.next(timeout=5)
                    except mp.TimeoutError:
                        continue
                    results.append(row); save(run_dir/'results.json', results)
                pool.close(); pool.join()
            except BaseException:
                stop.set(); pool.close(); pool.join(); raise
        save(run_dir/'complete.json', dict(results=results, cpu_seconds=cpu()-own_start))
        value_keep.cleanup()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('plan', type=Path); p.add_argument('stage', choices=['inspect', 'screen', 'confirm'])
    p.add_argument('--jobs', type=Path, help='absolute frozen subset; defaults to plan jobs')
    args = p.parse_args(); plan = check_plan(args.plan)
    jobs = load(args.jobs)['jobs'] if args.jobs else plan['jobs']
    if args.jobs and not args.jobs.is_absolute():
        raise ValueError('absolute subset path required')
    jobs = [dict(j, inspect_only=args.stage == 'inspect', stage=args.stage) for j in jobs]
    run(plan, jobs, args.stage)


if __name__ == '__main__':
    main()
