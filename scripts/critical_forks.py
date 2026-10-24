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


def pick_roots(eligible, count=3, end_tick=10000):
    """Outcome-blind temporal sampling shared by all terrains."""
    if not eligible:
        return []
    if count not in (1, 2, 3):
        raise ValueError('one to three temporal roots required')
    targets = [0, end_tick//2, end_tick] if count == 3 else [0, end_tick] if count == 2 else [end_tick//2]
    selected = []
    for tick in targets:
        best = min(eligible, key=lambda p: (abs(p['tick']-tick), p['decision']))
        if best not in selected:
            selected.append(best)
    return sorted(selected, key=lambda p: p['decision'])


def job_future(job, stage, k, search=False):
    legacy = job.get('legacy')
    if legacy:
        stage = stage + '-teacher-search' if search else stage
        text = f"{legacy['namespace']}:{legacy['cell']}:{job['decision']-1}:{stage}:{k}"
        return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], 'little')
    return future_seed(job['id'], stage, k, search)


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
            final = load(run/'complete.json')
            receipts = max(receipts, final.get('cpu_seconds', 0.), final.get('controller_children_cpu_seconds', 0.))
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


def open_game(job):
    from pvz_env import TaskSpec
    from pvz_event_env import EventWaitEnv
    from pvz_tasks import profile_for_deck
    from pvz_value_net import observation_context
    spec = deepcopy(job['reset']); deck = tuple(spec.pop('deck'))
    first_aid = spec.pop('first_aid_owned', False)
    spec['preplanted'] = tuple(tuple(p) for p in spec.get('preplanted', []))
    env = TEACHER._W['env'] = EventWaitEnv(PLAN['teacher']['resource_dir'],
            executable=PLAN['teacher']['executable'], headless=True)
    env.load_rollout_net(TEACHER._W['rollout_file'])
    env.load_value_net(TEACHER._W['value_files'][0])
    env.load_value_net(TEACHER._W['value_files'][1], static=True)
    obs, _ = env.reset(deck=deck, task=TaskSpec(profile=profile_for_deck(deck, first_aid), **spec))
    TEACHER._W['context'] = observation_context(obs)
    return env, obs


def replay(job):
    from pvz_frame_store import frame_path, read_frames
    from render_episode import _frame
    archive = Path(job['archive']); path = frame_path(archive)
    if sha(path) != job['frames_sha256'] or sha(archive/'meta.json') != job['meta_sha256']:
        raise ValueError('archive changed')
    frames = read_frames(path)
    if job['reset']['deck'] != [p[1] for p in frames[0]['packets']]:
        raise ValueError('archived deck differs from frozen reset')
    env, obs = open_game(job); state = {}; last_search = -10**9
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
            if job.get('legacy') or job.get('root_check'):
                legacy = job.get('legacy') or job['root_check']
                if env._debug_state_sha256() != legacy['engine_sha256']:
                    raise ValueError('legacy root engine identity differs')
                public = dict(frame=_frame(obs), student_history=history_record(state))
                if hashlib.sha256(key(public).encode()).hexdigest() != legacy['history_sha256']:
                    raise ValueError('legacy root public history differs')
                last_search = legacy.get('last_search_tick', -10**9)
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


def collect_trajectory(job):
    """Fresh student prefix and outcome-blind temporal roots; no terminal labels.

    Archived teacher prefixes continue to enter through replay/import-legacy. This
    stage supplies the old fog/roof opening-window capability for new panels.
    """
    from render_episode import _frame
    env = None
    try:
        env, obs = open_game(job); state = {}; args = teacher_args(job)
        folder=Path(PLAN['out'])/'trajectories'/job['id']
        path=folder/'frames.jsonl'
        if path.exists():
            raise ValueError('trajectory exists; use a fresh output plan')
        folder.mkdir(parents=True,exist_ok=True)
        window=PLAN.get('window_end_tick',10000); eligible=[]; frames=[dict(_frame(obs),action=None)]
        last_search=-10**9
        for decision in range(TEACHER.MAX_ACTIONS):
            if obs['terminal'] or obs['tick']>window:break
            usage(env)
            action=TEACHER._fast_action(obs,state)
            TEACHER._W['student_history']=state.get('history')
            scheduled=not(args.min_gap and action['type']!='plant' and obs['tick']-last_search<args.min_gap)
            if obs['legal_actions'].get('plants') and scheduled:
                check=dict(engine_sha256=env._debug_state_sha256(),
                    history_sha256=hashlib.sha256(key(dict(frame=_frame(obs),student_history=history_record(state))).encode()).hexdigest(),
                    last_search_tick=last_search)
                eligible.append(dict(decision=decision+1,tick=obs['tick'],root_check=check,
                                     candidate_count=len(all_candidates(obs))))
                last_search=obs['tick']
            state['action']=action;obs,*_,info=env.step(action)
            if not info.get('ok'):raise ValueError('student trajectory rejected action')
            frames.append(dict(_frame(obs),action=action))
        else:
            if not obs['terminal'] and obs['tick']<=window:
                raise RuntimeError('student trajectory reached action cap before window end')
        path.write_text(''.join(json.dumps(frame)+'\n' for frame in frames))
        save(folder/'meta.json',dict(reset=job['reset'],source_job=job['id'],prefix='student'))
        jobs=[]
        for point in pick_roots(eligible,PLAN.get('temporal_roots_per_game',3),window):
            fresh={k:v for k,v in job.items() if k not in ('legacy','indices','samples','inspect_only','stage')}
            fresh.update(id=f"{job['id']}-t{point['decision']}",archive=str(folder),
                frames_sha256=sha(path),meta_sha256=sha(folder/'meta.json'),decision=point['decision'],
                student_executed=True,root_check=point['root_check'])
            jobs.append(fresh)
        save(folder/'eligible.json',dict(eligible=eligible,jobs=jobs))
        return dict(id=job['id'],jobs=jobs,eligible=len(eligible))
    finally:
        TEACHER._close_worker_env();usage(force=True)


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
        trace['best_index'] = pos[6]; trace['student_index'] = pos[7]
        trace['pre_verification_values'] = [float(v) if math.isfinite(v) else None for v in pos[4]]
        chosen = original_verified(*pos, **kw)
        trace['final_values'] = [float(v) if math.isfinite(v) else None for v in pos[4]]
        return chosen
    stats = dict(searched=0, overrides=0, override_kinds=Counter(), last_search_tick=last_search)
    try:
        TEACHER._static_scores = static; TEACHER._verified_choice = verified
        chosen = TEACHER._search(env, obs, state['action'], args, seed, stats)
        trace.update(chosen=chosen, searched=bool(stats['searched']), margin=TEACHER.MARGIN, audit_schema=2,
                     required_halving_samples=args.samples, per_card=args.per_card,
                     verification=stats.get('last_verification'))
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
                future=job_future(job, stage, k), search_future=job_future(job, stage, k, True),
                inputs=PLAN['inputs'])


def validate(row, expected, trace):
    if row['identity'] != expected or row['outcome']['terminal'] is not True:
        raise ValueError('incompatible or incomplete terminal label')
    o = row['outcome']
    if o['result'] not in (1, 2) or o['won'] != (o['result'] == 1) or not 0 <= o['mowers_used'] <= 6:
        raise ValueError('invalid terminal result')
    required = {'root', 'income', 'history', 'legal', 'terminal', 'strict_actions'}
    if set(row['checks']) != required or any(v is not True for v in row['checks'].values()):
        raise ValueError('missing or failed exact root/history checks')
    if sha(trace) != row['trace_sha256']:
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
            chosen = TEACHER._search(env, obs, proposed, args, seed+decision, stats)
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
             engine_sha256=engine_hash, income=income, income_start=income_start,
             initial_mowers=sum(s == 1 for _, s in frames[0]['mowers']))
        path = folder/'root.json'
        if path.exists() and load(path) != json.loads(json.dumps(record)):
            raise ValueError('root changed')
        save(path, record)
        if not (folder/'search.json').exists() or load(folder/'search.json').get('audit_schema') != 2:
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
            pending = pool.imap_unordered(collect_trajectory if phase=='roots' else collect_job, jobs)
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
        if phase=='roots':
            save(out/'temporal-jobs.json',dict(jobs=[j for row in results for j in row['jobs']]))
        value_keep.cleanup()


def import_legacy(source, selection, out, executable, workers=2):
    """Freeze old fog/roof trajectories into ordinary archive jobs, without editing evidence.

    Only the input adapter knows the historical layout. Replay, candidate enumeration,
    audit and terminal continuation all use the same collector as new fork data.
    """
    from render_episode import load_training_task, training_task_kwargs
    for path in (source, selection, out, executable):
        if not Path(path).is_absolute():
            raise ValueError('absolute legacy paths required')
    out = Path(out)
    if out.exists():
        raise ValueError('legacy destination already exists; preserve frozen plans')
    original = load(source); paths = original['paths']; jobs = []
    inputs = {}
    def pin(name, path, expected=None):
        path = Path(path); digest = sha(path)
        if expected is not None and digest != expected:
            raise ValueError('changed legacy input: '+name)
        inputs[name] = dict(path=str(path), sha256=digest)
    pin('legacy_plan', source); pin('legacy_selection', selection)
    for name in ('rollout_policy', 'value', 'static_value', 'manifest'):
        pin(name, paths[name], original['sha256'][name])
    pin('executable', executable); pin('collector', Path(__file__))
    pin('search_source', ROOT/'scripts/search_play.py')
    import search_play
    from curriculum_queue import command_for
    from types import SimpleNamespace
    namespace = original['spec'].get('namespace', 'task45-E')
    for entry in load(selection)['jobs']:
        cell, point = entry['cell'], entry['point']
        trajectory_path = Path(source).parent/'episodes'/cell['id']/'trajectory.json'
        pin('trajectory_'+cell['id'], trajectory_path)
        trajectory = load(trajectory_path)
        if trajectory['cell'] != cell or point not in trajectory['eligible']:
            raise ValueError('legacy selection differs from frozen eligible trajectory')
        kw = training_task_kwargs(load_training_task(paths['manifest'], cell['task_id']), cell['seed'])
        reset = dict(level=kw['level'], seed=kw['seed'], deck=kw['deck'], playthrough=2,
                     **{k:v for k,v in kw['task_extra'].items() if v is not None})
        archive = out/'archives'/cell['id']; archive.mkdir(parents=True, exist_ok=True)
        frames = deepcopy(trajectory['frames'])
        for i, frame in enumerate(frames):
            frame['action'] = trajectory['tape'][i-1] if i else None
        frame_file = archive/'frames.jsonl'
        frame_file.write_text(''.join(json.dumps(f)+'\n' for f in frames))
        save(archive/'meta.json', dict(reset=reset, source=str(trajectory_path)))
        job = dict(id=f"{cell['id']}-d{point['decision']+1}", archive=str(archive),
            frames_sha256=sha(frame_file), meta_sha256=sha(archive/'meta.json'),
            reset=reset, decision=point['decision']+1, manifest=paths['manifest'], task_id=cell['task_id'],
            terrain='fog' if cell['level'] < 40 else 'roof', observation=cell.get('observation', 'human'),
            samples=entry.get('samples', 3),
            legacy=dict(namespace=namespace, cell=cell['id'], engine_sha256=point['state_sha256'],
                history_sha256=point['public_history_sha256'],
                last_search_tick=point.get('schedule_last_search_tick', -10**9)))
        # Retain exact source candidate identities and outcome-blind source selection.
        if 'candidate_indices' in entry:
            job['indices'] = entry['candidate_indices']
        jobs.append(job)
    if not jobs:
        raise ValueError('empty legacy selection')
    cfg = SimpleNamespace(python=Path(sys.executable), **{k:Path(v) for k,v in paths.items() if k!='manifest'},
                          workers=1, frames=out/'unused', out=out/'unused')
    args = search_play.parser().parse_args(command_for(cfg, Path(paths['manifest']), jobs[0]['task_id'], 0, 1, 'terminal')[2:])
    teacher = {k:getattr(args,k) for k in ('per_card','student_candidates','samples','horizon',
               'verify_samples','min_gap','rollout_sim_skip')}
    teacher.update({k:paths[k] for k in ('rollout_policy','value','static_value','resource_dir')}, executable=str(executable))
    plan = dict(out=str(out), workers=workers, inputs=inputs, teacher=teacher, jobs=jobs,
        screen_futures=3, confirm_futures=16, artifact_budget_bytes=5_000_000_000,
        protocol=dict(confirmation=dict(alpha=.05,selection='metric_bests',mean_gap=dict(won=1/3,wave=1.,tick=600.,mowers_used=.5))),
        legacy_source=dict(plan=str(source),selection=str(selection),source_hashes=original['sha256'],
            engine_replaced=original['sha256']['executable'] != sha(executable)))
    save(out/'plan.json', plan)
    return plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('plan', type=Path); p.add_argument('stage', choices=['roots', 'inspect', 'screen', 'confirm', 'import-legacy'])
    p.add_argument('--jobs', type=Path, help='absolute frozen subset; defaults to plan jobs')
    p.add_argument('--out', type=Path); p.add_argument('--executable', type=Path)
    p.add_argument('--workers', type=int, default=2)
    args = p.parse_args()
    if args.stage == 'import-legacy':
        if not all((args.jobs, args.out, args.executable)) or not 1 <= args.workers <= 2:
            p.error('import requires --jobs selection --out destination --executable engine and 1–2 workers')
        import_legacy(args.plan, args.jobs, args.out, args.executable, args.workers)
        return
    plan = check_plan(args.plan)
    jobs = load(args.jobs)['jobs'] if args.jobs else plan['jobs']
    if args.jobs and not args.jobs.is_absolute():
        raise ValueError('absolute subset path required')
    jobs = [dict(j, inspect_only=args.stage == 'inspect', stage=args.stage) for j in jobs]
    if args.stage=='roots':
        jobs=list({j['archive']:j for j in jobs}.values())
    run(plan, jobs, args.stage)


if __name__ == '__main__':
    main()
