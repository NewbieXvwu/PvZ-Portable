"""task43 round2 diagnostic: frozen teacher roots -> real search leaves -> teacher outcomes.

Run with one absolute plan JSON path. No production teacher switches or policy changes.
Each worker owns one engine, and validates reference history against its native leaf before use.
"""
from __future__ import annotations

from argparse import Namespace
from collections import Counter
from copy import deepcopy
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'python')]
import search_play as teacher


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=1, allow_nan=False) + '\n')
    tmp.replace(path)


def history_record(state):
    h = state.get('history')
    return dict(previous_frame=state.get('frame'), previous_action=state.get('action'),
                history=None if h is None else dict(obs=h.obs, wave=h.wave, wave_tick=h.wave_tick,
                    plants=[[*k, v] for k, v in sorted(h.plants.items())]))


def stats_new():
    return dict(searched=0, overrides=0, override_kinds=Counter())


def cpu(env=None):
    own = resource.getrusage(resource.RUSAGE_SELF)
    total = own.ru_utime + own.ru_stime
    if env is not None:
        # Worker owns exactly this live engine; avoid double-counting RUSAGE_CHILDREN.
        data = Path(f'/proc/{env._process.pid}/stat').read_text().rsplit(')', 1)[1].split()
        total += (int(data[11]) + int(data[12])) / os.sysconf('SC_CLK_TCK')
    return total


def process_sample():
    """Read proc only: count actual qc/own engines and measure qc CPU without touching sessions."""
    processes = {}
    for path in Path('/proc').glob('[0-9]*'):
        try:
            fields = (path/'stat').read_text().rsplit(')', 1)[1].split()
            command = (path/'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            processes[int(path.name)] = (int(fields[1]), fields[0],
                (int(fields[11])+int(fields[12]))/os.sysconf('SC_CLK_TCK'), command)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    def below(pid, roots):
        seen = set()
        while pid in processes and pid not in seen:
            if pid in roots: return True
            seen.add(pid); pid = processes[pid][0]
        return False
    qc = {pid for pid, (_,_,_,cmd) in processes.items()
          if '/worktrees/cq/scripts/curriculum_queue.py' in cmd and 'python' in cmd.split(' ')[0]}
    own = {pid for pid, (_,_,_,cmd) in processes.items()
           if str(ROOT/'scripts/value_teacher_feasibility.py') in cmd and 'python' in cmd.split(' ')[0]}
    qc_members = {str(pid): p[2] for pid,p in processes.items() if below(pid,qc)}
    return dict(time=time.time(), logical_cpus=os.cpu_count(), qc_controllers=sorted(qc),
        qc_cpu_seconds_by_pid=qc_members,
        qc_engines=sum('/pvz-portable -env ' in p[3] and below(pid,qc) for pid,p in processes.items()),
        own_engines=sum('/pvz-portable -env ' in p[3] and below(pid,own) for pid,p in processes.items()),
        load1=os.getloadavg()[0])


def action_step(env, action):
    obs, _, _, _, info = env.step(action)
    if not info.get('ok'):
        obs, *_ = env.step({'type': 'wait', 'ticks': 60})
    return obs


def finish_teacher(env, obs, state, args, seed_base, trace_path):
    """Same fast_action/search/step cycle and MAX_ACTIONS as search_play._play."""
    stats = stats_new()
    with trace_path.open('w') as trace:
        for index in range(teacher.MAX_ACTIONS):
            if obs.get('terminal'):
                return dict(won=obs.get('result') == 1, terminal=True, result=obs.get('result'),
                            tick=obs['tick'], wave=obs['wave'], decisions=index,
                            searched=stats['searched'], overrides=stats['overrides'])
            proposed = teacher._fast_action(obs, state)
            teacher._W['student_history'] = state.get('history')
            chosen = teacher._search(env, obs, proposed, args, seed_base + index, stats)
            trace.write(json.dumps(dict(decision=index, tick=obs['tick'], wave=obs['wave'],
                                       student=proposed, chosen=chosen)) + '\n')
            state['action'] = chosen
            obs = action_step(env, chosen)
    raise RuntimeError('teacher continuation reached action cap without a terminal outcome')


def sample_root(env, obs, state, args, decision_seed, stats, directory):
    """Instrument only this call, retaining actual k=0 leaves before the search restores root."""
    from render_episode import _frame
    root = env.snapshot()
    root_hash = env._debug_state_sha256()
    baseline = dict(teacher._W)
    original = teacher._rollout_policy
    captured = []
    common = (decision_seed * 1000) % 2**32

    def retain(environment, current, first, until_tick, parent=None, seed=None):
        leaf = original(environment, current, first, until_tick, parent, seed)
        if seed == common:
            captured.append(dict(action=deepcopy(first), leaf=deepcopy(leaf),
                snapshot=environment.snapshot(), state_hash=environment._debug_state_sha256(),
                income=list(environment._sun_production_history),
                income_start=environment._sun_history_start_tick))
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
    write_json(directory / 'root.json', dict(decision_seed=decision_seed, common_future=common,
        root_frame=_frame(obs), root_state_sha256=root_hash, history=history_record(state),
        context=teacher._W['context'].tolist(), candidates=targets, chosen=chosen,
        search_cpu_seconds=search_cpu))
    rows = []
    try:
        for index, item in enumerate(captured):
            before = cpu(env)
            teacher._W.clear(); teacher._W.update(baseline)
            teacher._W['student_history'] = deepcopy(state.get('history'))
            recovered = {}
            replay = teacher._python_rollout_reference(env, obs, item['action'],
                obs['tick'] + args.horizon, root, common, fast_state_out=recovered)
            checks = dict(frame=_frame(replay) == _frame(item['leaf']),
                engine_state=env._debug_state_sha256() == item['state_hash'],
                income=list(env._sun_production_history) == item['income'],
                income_start=env._sun_history_start_tick == item['income_start'])
            if not all(checks.values()):
                write_json(directory / f'candidate_{index}_mismatch.json',
                    dict(checks=checks, native=_frame(item['leaf']), replay=_frame(replay)))
                raise RuntimeError(f'leaf history reconstruction mismatch: {checks}')
            replay_cpu = cpu(env) - before
            env.restore(item['snapshot'])
            teacher._W.clear(); teacher._W.update(baseline)
            # Preserve the same search parameters while giving every candidate the same
            # continuation search-noise schedule (indexed by continuation decision ordinal).
            seed_base = decision_seed + 5000
            initial_history = history_record(recovered)
            leaf_frame = _frame(item['leaf'])
            begin = cpu(env); wall = time.monotonic()
            outcome = finish_teacher(env, item['leaf'], recovered, args, seed_base,
                                     directory / f'candidate_{index}_teacher.jsonl')
            row = dict(candidate=index, action=item['action'], common_future=common,
                continuation_seed_base=seed_base, leaf_frame=leaf_frame,
                leaf_state_sha256=item['state_hash'], leaf_history=initial_history,
                history_checks=checks, outcome=outcome, replay_cpu_seconds=replay_cpu,
                continuation_cpu_seconds=cpu(env)-begin,
                continuation_wall_seconds=time.monotonic()-wall)
            write_json(directory / f'candidate_{index}.json', row)
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


def group_summary(rows):
    if not rows or any(x['outcome'].get('terminal') is not True for x in rows):
        raise ValueError('complete terminal labels required')
    wins = sum(x['outcome']['won'] for x in rows)
    n = len(rows)
    return dict(labels=n, wins=wins, mixed=0 < wins < n, all_loss=wins == 0,
        effective_pairs=wins * (n-wins), total_pairs=n*(n-1)//2,
        continuation_cpu_seconds=sum(x['continuation_cpu_seconds'] for x in rows),
        replay_cpu_seconds=sum(x['replay_cpu_seconds'] for x in rows))


def run_seed(job):
    from render_episode import _frame, load_training_task, training_task_kwargs
    from pvz_env import TaskSpec
    from pvz_tasks import profile_for_deck
    from pvz_event_env import EventWaitEnv
    from pvz_value_net import observation_context
    plan, seed = job
    args = Namespace(**plan['teacher'])
    teacher._init_worker(args.value, args.rollout_policy, args.rollout_sim_skip, args.static_value)
    directory = Path(plan['out']) / f'seed_{seed}'
    if directory.exists():
        raise FileExistsError(f'refuse to overwrite existing seed run: {directory}')
    directory.mkdir(parents=True)
    env = None
    started = cpu(); wall = time.monotonic()
    try:
        env = teacher._W['env'] = EventWaitEnv(args.resource_dir, executable=args.executable, headless=True)
        env.load_rollout_net(teacher._W['rollout_file'])
        task_def = load_training_task(Path(args.manifest), args.task_id)
        kw = training_task_kwargs(task_def, seed)
        task = TaskSpec(level=kw['level'], seed=seed, playthrough=2,
            profile=profile_for_deck(tuple(kw['deck']), kw.get('first_aid_owned', False)),
            **{k:v for k,v in kw['task_extra'].items() if v is not None})
        obs, _ = env.reset(deck=tuple(kw['deck']), task=task)
        teacher._W['context'] = observation_context(obs)
        teacher._W['student_history'] = None
        state = {}; stats = stats_new(); groups = []; missing = list(plan['waves'])
        with (directory / 'root_trajectory.jsonl').open('w') as trace:
            for decision in range(teacher.MAX_ACTIONS):
                if obs.get('terminal'):
                    break
                proposed = teacher._fast_action(obs, state)
                teacher._W['student_history'] = state.get('history')
                eligible = bool((obs.get('legal_actions') or {}).get('plants')) and not (
                    args.min_gap and proposed['type'] != 'plant' and
                    obs['tick']-stats.get('last_search_tick', -10**9) < args.min_gap)
                if missing and obs['wave'] >= missing[0] and eligible:
                    target = missing.pop(0)
                    chosen, group = sample_root(env, obs, state, args, seed*10000+decision,
                        stats, directory / f'wave_{target}')
                    group['wave_target'] = target; groups.append(group)
                else:
                    chosen = teacher._search(env, obs, proposed, args, seed*10000+decision, stats)
                trace.write(json.dumps(dict(decision=decision, tick=obs['tick'], wave=obs['wave'],
                    student=proposed, chosen=chosen))+'\n'); trace.flush()
                state['action'] = chosen
                obs = action_step(env, chosen)
            else:
                raise RuntimeError('root teacher reached action cap')
        total_cpu = cpu(env)-started
        result = dict(seed=seed, terminal=obs.get('terminal'), won=obs.get('result') == 1,
            final_tick=obs['tick'], decisions=decision, searched=stats['searched'],
            overrides=stats['overrides'], missing_wave_targets=missing, groups=groups,
            total_cpu_seconds=total_cpu, wall_seconds=time.monotonic()-wall,
            final_frame=_frame(obs))
        write_json(directory / 'result.json', result)
        return result
    finally:
        teacher._close_worker_env()


def load_plan(path):
    if not Path(path).is_absolute():
        raise ValueError('absolute plan path required')
    p = json.loads(Path(path).read_text())
    if not 1 <= p['workers'] <= 4 or not 1 <= len(p['seeds']) <= 8:
        raise ValueError('at most four workers and eight root seeds')
    if len(set(p['seeds'])) != len(p['seeds']):
        raise ValueError('duplicate seeds')
    for name in ('value', 'static_value', 'rollout_policy', 'executable', 'manifest'):
        filename = p['teacher'][name]
        if not Path(filename).is_absolute() or digest(filename) != p['sha256'][name]:
            raise ValueError(f'input changed or relative path: {name}')
    if not Path(p['out']).is_absolute():
        raise ValueError('absolute output path required')
    return p


def main():
    plan = load_plan(sys.argv[1])
    out = Path(plan['out']); out.mkdir(parents=True, exist_ok=True)
    if (out/'run.json').exists():
        raise FileExistsError('run already exists; inspect live handle/results before any retry')
    write_json(out/'run.json', dict(pid=os.getpid(), plan=plan, started=time.time()))
    teacher._load_nets(plan['teacher']['value'], plan['teacher']['rollout_policy'],
                       plan['teacher']['static_value'])
    teacher._freeze_startup()
    results = []
    with mp.get_context('fork').Pool(plan['workers']) as pool:
        pending = pool.imap_unordered(run_seed, [(plan,s) for s in plan['seeds']])
        with (out/'resources.jsonl').open('w') as monitor:
            while len(results) < len(plan['seeds']):
                sample = process_sample()
                monitor.write(json.dumps(sample)+'\n'); monitor.flush()
                if sample['own_engines'] > 4:
                    raise RuntimeError('own engine count exceeded four')
                try:
                    result = pending.next(timeout=10)
                except mp.TimeoutError:
                    continue
                results.append(result)
                write_json(out/'progress.json', dict(seeds_completed=[r['seed'] for r in results]))
        pool.close(); pool.join()
    write_json(out/'completed.json', dict(seeds=sorted(r['seed'] for r in results), finished=time.time()))


if __name__ == '__main__':
    main()
