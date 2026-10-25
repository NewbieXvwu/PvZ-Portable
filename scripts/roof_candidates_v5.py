"""Isolated task76 diagnostics. Never imported by production collection code."""
from __future__ import annotations

import argparse
import copy
from collections import Counter
import contextlib
import fcntl
import gzip
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
MAIN = Path('/home/newbiexvwu/PvZ-search')
OUT = ROOT / 'artifacts/task76'
PLAN = ROOT / 'MEMORY/docs/task76-roof-candidates-v5-plan.json'
MAX_GAME_PROCESSES = 2
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'python')]
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['PVZ_NO_POPUP'] = '1'



def owned_game_processes():
    """Inspect only task76's frozen engine; other tasks consume no task76 slots."""
    games = []
    expected = str(OUT/'frozen/pvz-portable')
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():
            continue
        try:
            executable = os.readlink(process/'exe').removesuffix(' (deleted)')
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if executable == expected:
            games.append(dict(pid=int(process.name), executable=executable))
    return games


def require_game_capacity(workers):
    if not 1 <= workers <= MAX_GAME_PROCESSES:
        raise ValueError('master per-task memory limit: one or two game processes only')
    games = owned_game_processes()
    if len(games) + workers > MAX_GAME_PROCESSES:
        raise RuntimeError('task76 slots occupied; no additional task76 game started: '
                           + json.dumps(games))


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, document):
    path = Path(path)
    if not path.is_absolute() or not path.is_relative_to(ROOT):
        raise ValueError('task76 writes must stay in the absolute roofcand tree')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=1, allow_nan=False) + '\n')
    temporary.replace(path)


def freeze(source, destination):
    source, destination = Path(source), Path(destination)
    digest = sha(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha(destination) != digest:
            raise ValueError('frozen input differs: ' + str(destination))
    else:
        with source.open('rb') as src, destination.open('xb') as dst:
            while chunk := src.read(1 << 20):
                dst.write(chunk)
        destination.chmod(source.stat().st_mode & 0o777)
    if digest != sha(source) or digest != sha(destination):
        raise ValueError('input changed during freezing')
    return dict(source=str(source), path=str(destination), sha256=digest,
                bytes=destination.stat().st_size)


def prepare():
    if PLAN.exists():
        verify_plan()
        print('existing frozen plan verified; not refreezing the live student')
        return
    require_game_capacity(1)
    state = read('/home/newbiexvwu/pvz-logs/curriculum_state.json')
    sources = dict(engine=MAIN/'.claude/worktrees/cq/build4/pvz-portable',
                   student=Path(state['student']), value=MAIN/'artifacts/value_net/v5/model.pt',
                   static=MAIN/'artifacts/hard/value_branch1/model.pt')
    if (MAIN/'artifacts/value_net/v4/model.pt').exists():
        sources['v4_control'] = MAIN/'artifacts/value_net/v4/model.pt'
    inputs = {k: freeze(p, OUT/'frozen'/('pvz-portable' if k == 'engine' else k+'.pt'))
              for k, p in sources.items()}
    from make_roof_panel import ROOF, EVAL_START, EVAL_COUNT
    from zone_panel import task_document
    from pvz_tasks import deck_for_level, profile_for_deck
    from pvz_env import TaskSpec
    from pvz_event_env import EventWaitEnv
    tasks = [task_document(ROOF, level, deck_for_level(level, family='roof_r7'),
             f'roof_l{level}_eval_r7', list(range(EVAL_START, EVAL_START + EVAL_COUNT)))
             for level in range(41, 50)]
    checks = []
    resources = '/home/newbiexvwu/.cache/pvz-research-resources'
    with EventWaitEnv(resources, executable=inputs['engine']['path'], headless=True) as env:
        for task in tasks:
            deck = tuple(task['deck'])
            obs, _ = env.reset(deck=deck, task=TaskSpec(level=task['level'], seed=task['seeds'][0],
                                profile=profile_for_deck(deck), playthrough=2))
            options = env.deck_options(); warnings = env.deck_warnings(deck)['warnings']
            assert set(deck) <= set(options['available_plants']) and not warnings
            checks.append(dict(level=task['level'], warnings=warnings, deck=task['deck'],
                               starting_sun=obs['sun'], starting_pots=sum(p['type']==33 for p in obs['plants'])))
    manifest = OUT/'roof_panel.json'
    write(manifest, dict(schema_version=1, purpose='task76 copied registered r7 evaluation definitions', tasks=tasks))
    files = ['scripts/search_play.py', 'python/pvz_env.py', 'python/pvz_event_env.py',
             'python/pvz_rollout_policy.py', 'scripts/pvz_tasks.py', 'scripts/render_episode.py',
             'scripts/make_roof_panel.py', 'scripts/zone_panel.py']
    plan = dict(schema_version=1, task=76, inputs=inputs, manifest=str(manifest),
                manifest_sha256=sha(manifest), resource_dir=resources,
                student_round_at_freeze=state.get('student_round'),
                curriculum_round_at_freeze=state.get('round'),
                code_sha256={str(ROOT/p):sha(ROOT/p) for p in files},
                deck_checks=checks, workers=MAX_GAME_PROCESSES,
                teacher=dict(per_card=1, student_candidates=2, samples=3, verify_samples=9,
                             horizon=2500, min_gap=300, rollout_sim_skip=61, dagger_beta=1.0, margin=.02),
                initial_levels=[42,43], seeds_per_level=32,
                expansion='Every roof level in L42-L49 with task74 teacher win rate <50%; request authoritative task74 result; do not infer from v4.',
                arms=['production','all_plants','all_legal'],
                contrasts=dict(primary='all_plants vs production isolates omission of legal planting cells',
                               supplementary='all_legal additionally includes legal shovels; report action-space effects separately'),
                coverage='Record every game decision before action, every legal plant and shovel, production candidates, all halving rounds and verification; packet slots mapped to actual plant types.',
                terminal_probe=dict(futures_per_action=8, actions='Every legal plant and shovel plus canonical wait at each frozen probe root',
                    continuation='Frozen greedy student C++ ROLLOUT until_tick=0, matching task72b; explicitly distinct from task46 search-teacher terminal continuation',
                    selection='After paired panels finish, first two losing production seeds per required level; four time-spaced scheduled legal roots (0/one-third/two-thirds/latest) per loss; terminal outcomes never used to choose roots.',
                    leaf_futures=9, horizons=[2500,4000,6000], v4_control='Same roots/futures/h2500 where v4 exists; also report historical task46 53.9% as protocol-different reference.'),
                production_changes=False, protected_tree_writes=False,
                historical_v4_reference='MEMORY/docs/task46-roof-zone.md: won 53.9%, cluster95%47.1-60.9; different roots, model, two futures and teacher continuation')
    write(PLAN, plan)
    print(json.dumps(dict(frozen_student=sources['student'].as_posix(), student_round=plan['student_round_at_freeze'],
                         manifest=str(manifest), inputs=inputs, deck_checks=checks), ensure_ascii=False, indent=1))


def verify_plan():
    plan = read(PLAN)
    for name, item in plan['inputs'].items():
        if sha(item['path']) != item['sha256']:
            raise ValueError('frozen input changed: ' + name)
    if sha(plan['manifest']) != plan['manifest_sha256']:
        raise ValueError('frozen manifest changed')
    for path, digest in plan['code_sha256'].items():
        if sha(path) != digest:
            raise ValueError('production algorithm changed inside diagnostic checkout: '+path)
    return plan


def action_key(action):
    return json.dumps(action, sort_keys=True, separators=(',', ':'))


def all_actions(obs):
    plants = [dict(type='plant', packet=a['packet'], row=a['row'], col=a['col'])
              for a in obs['legal_actions'].get('plants', [])]
    plants.sort(key=lambda a: (a['packet'], a['row'], a['col']))
    shovels = [dict(type='shovel', row=a['row'] if isinstance(a, dict) else a[1],
                    col=a['col'] if isinstance(a, dict) else a[0])
               for a in obs['legal_actions'].get('shovels', [])]
    shovels.sort(key=lambda a: (a['row'], a['col']))
    return [dict(type='wait', ticks=60, until='timeout')] + plants + shovels


def augment_candidates(candidates, obs, arm):
    result = copy.deepcopy(candidates)
    if arm == 'production':
        return result
    if arm not in ('all_plants', 'all_legal'):
        raise ValueError('unknown diagnostic arm')
    present = {action_key(a) for a in result}
    for action in all_actions(obs):
        if arm == 'all_plants' and action['type'] == 'shovel':
            continue
        if action_key(action) not in present:
            result.append(action)
            present.add(action_key(action))
    return result


class PanelRecorder:
    """Per-decision observer; wrappers are confined to this diagnostic worker."""
    def __init__(self, path, arm):
        self.path, self.arm = Path(path), arm
        self.stream = None
        self.current = None
        self.searched = 0
        self.decisions = 0

    def before(self, env, obs, fast_state, decision):
        from render_episode import _frame
        self.current = dict(decision=decision, tick=obs['tick'], wave=obs['wave'],
            sun=obs['sun'], frame=_frame(obs), legal_actions=all_actions(obs),
            packets=obs['packets'], searched=False)

    def after(self, obs, action, policy, rejected, info):
        self.current.update(executed=action, policy=policy, rejected=rejected,
                            next_tick=obs['tick'], terminal=obs['terminal'])
        if rejected:
            raise ValueError('diagnostic teacher executed a rejected action')
        self.stream.write(json.dumps(self.current, separators=(',', ':'), allow_nan=False)+'\n')
        self.decisions += 1

    def __enter__(self):
        import search_play as teacher
        self.originals = {name: getattr(teacher, name) for name in
                          ('_static_scores', '_halving_choice', '_continuations', '_verified_choice')}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = gzip.open(self.path, 'wt', compresslevel=1)

        def static(*args, **kwargs):
            scores = self.originals['_static_scores'](*args, **kwargs)
            self.current['static_scores'] = [dict(action=a, value=float(v)) for v, a in scores]
            return scores

        def halving(env, obs, root, candidates, policy, policy_key, args, seed, stats):
            self.current.update(searched=True, production_candidates=copy.deepcopy(candidates),
                                decision_seed=seed, rounds=[])
            expanded = augment_candidates(candidates, obs, self.arm)
            self.current['evaluated_candidates'] = copy.deepcopy(expanded)
            self.searched += 1
            return self.originals['_halving_choice'](env, obs, root, expanded, policy,
                                                       policy_key, args, seed, stats)

        def continuations(env, obs, root, jobs, args, stats):
            handles = self.originals['_continuations'](env, obs, root, jobs, args, stats)
            values = teacher._leaf_values(handles)
            self.current['rounds'].append([dict(action=a, future=seed % 2**32, value=float(v))
                                          for (a, seed), v in zip(jobs, values)])
            return handles

        def verified(env, obs, root, candidates, totals, counts, best, policy_index,
                     policy, args, seed, stats, **kwargs):
            self.current['before_verification'] = dict(values=list(totals), counts=list(counts),
                                                       best=best, policy_index=policy_index)
            chosen = self.originals['_verified_choice'](env, obs, root, candidates, totals,
                         counts, best, policy_index, policy, args, seed, stats, **kwargs)
            self.current['after_verification_values'] = list(totals)
            verification = copy.deepcopy(stats.get('last_verification'))
            if verification:
                verification.pop('seconds', None)
            self.current['verification'] = verification
            return chosen

        teacher._static_scores, teacher._halving_choice = static, halving
        teacher._continuations, teacher._verified_choice = continuations, verified
        return self

    def __exit__(self, *error):
        import search_play as teacher
        for name, original in self.originals.items():
            setattr(teacher, name, original)
        self.stream.close()


def teacher_args(plan, level, frames=False):
    import search_play as teacher
    p = plan['inputs']
    argv = ['--rollout-policy', p['student']['path'], '--value', p['value']['path'],
            '--static-value', p['static']['path'], '--executable', p['engine']['path'],
            '--resource-dir', plan['resource_dir'], '--manifest', plan['manifest'],
            '--task-id', f'roof_l{level}_eval_r7', '--out', str(OUT/'panels'), '--workers', '1']
    for key in ('per_card','student_candidates','samples','verify_samples','horizon',
                'min_gap','rollout_sim_skip','dagger_beta'):
        argv += ['--'+key.replace('_','-'), str(plan['teacher'][key])]
    if frames:
        argv += ['--frames', str(OUT/'frames'), '--frames-label', f'production_l{level}']
    return teacher.parser().parse_args(argv)


def initialize_worker(plan):
    import search_play as teacher
    teacher._init_worker(plan['inputs']['value']['path'], plan['inputs']['student']['path'],
                         plan['teacher']['rollout_sim_skip'], plan['inputs']['static']['path'])


def preload(plan):
    import search_play as teacher
    keep, values = teacher._export_values(plan['inputs']['value']['path'], plan['inputs']['static']['path'])
    teacher._load_nets(values[0], plan['inputs']['student']['path'], values[1])
    teacher._freeze_startup()
    return keep


def semantic_game(row):
    return {k:v for k,v in row.items() if k != 'seconds'}


def play_job(job):
    import search_play as teacher
    plan, level, seed, arm = job['plan'], job['level'], job['seed'], job['arm']
    folder = OUT/'panels'/f'l{level}'/arm/str(seed)
    identity = dict(level=level, seed=seed, diagnostic_arm=arm,
                    input_sha256={k:v['sha256'] for k,v in plan['inputs'].items()},
                    manifest_sha256=plan['manifest_sha256'], teacher=plan['teacher'], schema=1)
    if (folder/'result.json').exists():
        result = read(folder/'result.json')
        if result['identity'] != identity or sha(folder/'decisions.jsonl.gz') != result['trace_sha256']:
            raise ValueError('incompatible completed game receipt')
        return result
    args = teacher_args(plan, level, frames=arm=='production')
    # Native allocation identifiers survive RESET and enter STATE_HASH. Fresh
    # processes make later archived-root replay compare the same hidden identity.
    teacher._close_worker_env()
    started = time.monotonic()
    with PanelRecorder(folder/'decisions.jsonl.gz', arm) as recorder:
        outcome = teacher._play(dict(args=args, seed=seed, arm='fast_search'), observer=recorder)
    if not outcome['terminal']:
        raise RuntimeError('game reached action limit; no loss label written')
    result = dict(identity=identity, outcome=outcome, trace_sha256=sha(folder/'decisions.jsonl.gz'),
                  recorded_decisions=recorder.decisions, recorded_searches=recorder.searched,
                  elapsed_seconds=time.monotonic()-started, worker_pid=os.getpid(),
                  driver_sha256_at_execution=sha(Path(__file__)))
    assert result['recorded_decisions'] == outcome['decisions']
    write(folder/'result.json', result)
    print(json.dumps(dict(level=level, seed=seed, arm=arm, won=outcome['won'],
                          seconds=round(result['elapsed_seconds'],2))), flush=True)
    return result


@contextlib.contextmanager
def controller_lock():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'controller.lock').open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def smoke():
    require_game_capacity(1)
    import search_play as teacher
    plan = verify_plan()
    keep = preload(plan)
    initialize_worker(plan)
    try:
        args = teacher_args(plan,42)
        plain = teacher._play(dict(args=args, seed=45000000, arm='fast_search'))
        reused_plain = teacher._play(dict(args=args, seed=45000000, arm='fast_search'))
        assert {k:v for k,v in semantic_game(plain).items() if k!='state_hash'} == {
            k:v for k,v in semantic_game(reused_plain).items() if k!='state_hash'}
        teacher._close_worker_env()
        folder = OUT/'smoke'
        with PanelRecorder(folder/'instrumented-production.jsonl.gz', 'production') as recorder:
            instrumented = teacher._play(dict(args=args, seed=45000000, arm='fast_search'), observer=recorder)
        write(folder/'parity-comparison.json',dict(plain=plain,instrumented=instrumented,
            reused_uninstrumented=reused_plain,
            differing_fields=[k for k in plain if semantic_game(plain).get(k)!=semantic_game(instrumented).get(k)]))
        assert semantic_game(plain) == semantic_game(instrumented), 'observer changed baseline game'
        for arm in ('all_plants','all_legal'):
            teacher._close_worker_env()
            with PanelRecorder(folder/f'{arm}.jsonl.gz',arm) as recorder:
                result = teacher._play(dict(args=args,seed=45000000,arm='fast_search'),observer=recorder)
            assert result['terminal']
            with gzip.open(folder/f'{arm}.jsonl.gz','rt') as f:
                for row in map(json.loads,f):
                    if row['searched']:
                        expected = {action_key(a) for a in row['legal_actions']
                                    if arm=='all_legal' or a['type']!='shovel'}
                        actual = {action_key(a) for a in row['evaluated_candidates']}
                        assert expected <= actual
        write(ROOT/'MEMORY/docs/task76-roof-smoke.json', dict(parity=True,
            plain=plain,instrumented=instrumented,expanded_arms_complete=True,
            reused_uninstrumented=reused_plain,
            native_reset_note='Process reuse changes native STATE_HASH without changing any other game outcome field; fresh-process plain/instrumented equality includes STATE_HASH.',
            inputs=plan['inputs'],driver_sha256=sha(Path(__file__))))
        print('SMOKE PASS: full production game identical; both expansions complete and terminal',flush=True)
    finally:
        teacher._close_worker_env()
        keep.cleanup()


def panels(levels, count, workers):
    plan = verify_plan()
    require_game_capacity(workers)
    if not 1 <= count <= 32:
        raise ValueError('at most 32 fixed evaluation seeds')
    jobs = [dict(plan=plan,level=level,seed=45000000+i,arm=arm)
            for i in range(count) for level in levels for arm in plan['arms']]
    with controller_lock():
        keep = preload(plan)
        try:
            with mp.get_context('fork').Pool(workers,initializer=initialize_worker,initargs=(plan,)) as pool:
                results = list(pool.imap_unordered(play_job,jobs,chunksize=1))
            write(OUT/'panel-batch-complete.json',dict(levels=levels,games_per_arm=count,
                workers=workers,completed=len(results),complete=True))
        finally:
            keep.cleanup()


def trace_rows(level, arm, seed):
    path = OUT/'panels'/f'l{level}'/arm/str(seed)/'decisions.jsonl.gz'
    with gzip.open(path, 'rt') as stream:
        return [json.loads(line) for line in stream]


def round_values(row):
    """Round-1 leaf value of every candidate: all of them, one common future seed."""
    if not row.get('rounds'):
        return {}
    return {action_key(entry['action']): float(entry['value']) for entry in row['rounds'][0]}


def shared_prefix(left, right):
    """Decisions before the first difference: same state, same action, hence the same root."""
    count = 0
    for a, b in zip(left, right):
        if a['frame'] != b['frame'] or action_key(a['executed']) != action_key(b['executed']):
            break
        count += 1
    return count


def binom_two_sided(k, n):
    if n == 0:
        return 1.0
    from math import comb
    tail = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def production_like(row):
    """What the production candidate rule would have offered at this state.

    Reconstructed from the trace: one best cell per affordable card (the static net's per-card
    argmax, per_card=1), the student's own action, and the wait. The student's two extra
    candidates are not recorded per decision, so this set is a subset of production's; a cell
    missing from it was *definitely* not offered by production, and a cell inside it probably was.
    """
    static = {}
    for item in row.get('static_scores') or []:
        static[action_key(item['action'])] = float(item['value'])
    best = {}
    for key, value in static.items():
        packet = json.loads(key)['packet']
        if packet not in best or value > best[packet][0]:
            best[packet] = (value, key)
    keys = {value[1] for value in best.values()}
    keys.add(action_key(row['policy']))
    return keys, static


def coverage(level, seed, arms):
    """Per-decision candidate coverage, reconstructed inside each arm's own trace."""
    prod = trace_rows(level, 'production', seed)
    result = {}
    for arm in arms:
        rows = trace_rows(level, arm, seed)
        prefix = shared_prefix(prod, rows)
        decisions = []
        for index, row in enumerate(rows):
            if not row.get('searched'):
                continue
            values = round_values(row)
            legal = [x for x in row['legal_actions'] if x['type'] == 'plant']
            offered, static = production_like(row)
            scored = [(values[action_key(x)], x) for x in legal if action_key(x) in values]
            inside = [(v, x) for v, x in scored if action_key(x) in offered]
            if not scored or not inside:
                continue
            best_all, best_action = max(scored, key=lambda item: item[0])
            best_inside = max(inside, key=lambda item: item[0])[0]
            packet_cells = [static[action_key(x)] for x in legal
                            if action_key(x) in static and x['packet'] == best_action['packet']]
            decisions.append(dict(
                decision=index, tick=row['tick'], wave=row['wave'], sun=row['sun'],
                legal_plants=len(legal), evaluated=len(row['evaluated_candidates']),
                production_offered=len(offered),
                omitted_best=action_key(best_action) not in offered,
                regret=best_all - best_inside,
                best_value=best_all, best_inside=best_inside, best_action=best_action,
                best_packet_cells=len(packet_cells),
                best_static_rank=(sorted(packet_cells, reverse=True).index(static[action_key(best_action)]) + 1
                                  if action_key(best_action) in static else None),
                chosen=row['executed'], chosen_in_production=action_key(row['executed']) in offered))
        result[arm] = dict(shared_prefix=prefix, searched=len(decisions), decisions=decisions)
    return result


def probe_targets(level, seed):
    """Four time-spaced searched decisions, chosen from the trace alone (never from outcomes)."""
    rows = trace_rows(level, 'production', seed)
    receipt = read(OUT/'panels'/f'l{level}'/'production'/str(seed)/'result.json')
    final_tick = receipt['outcome']['final_tick']
    searched = [i for i, row in enumerate(rows) if row.get('searched')]
    chosen = []
    for fraction in (0.0, 1/3, 2/3):
        tick = final_tick * fraction
        chosen.append(min(searched, key=lambda i: (abs(rows[i]['tick'] - tick), i)))
    chosen.append(searched[-1])
    return sorted(set(chosen))


def probe_seeds(level, count=2):
    losing = []
    for receipt_path in sorted((OUT/'panels'/f'l{level}'/'production').glob('*/result.json')):
        receipt = read(receipt_path)
        if not receipt['outcome']['won']:
            losing.append(int(receipt['identity']['seed']))
    return sorted(losing)[:count]


class TerminalRecorder:
    """Read-only probe at scheduled roots: every legal action, terminal futures and leaf futures."""

    def __init__(self, level, seed, targets, model, futures, leaf_futures, horizon):
        self.level, self.seed, self.targets = level, seed, set(targets)
        self.model, self.futures = model, futures
        self.leaf_futures, self.horizon = leaf_futures, horizon
        self.written = []

    def before(self, env, obs, fast_state, decision):
        if decision in self.targets:
            self.probe(env, obs, decision)

    def after(self, obs, action, policy, rejected, info):
        pass

    @staticmethod
    def future_seed(parts):
        return int(hashlib.sha256(':'.join(map(str, parts)).encode()).hexdigest()[:8], 16)

    def probe(self, env, obs, decision):
        import search_play as teacher
        root = env.snapshot_fast()
        try:
            records = []
            for action in all_actions(obs):
                key = action_key(action)
                terminal = [teacher._score_terminal_or_value(
                                teacher._rollout_policy(env, obs, action, 0, root,
                                    self.future_seed(('terminal', self.seed, decision, key, future))))
                            for future in range(self.futures)]
                leaves = [teacher._leaf_values([teacher._rollout_policy(
                              env, obs, action, obs['tick'] + self.horizon, root,
                              self.future_seed(('leaf', self.seed, decision, key, future)))])[0]
                          for future in range(self.leaf_futures)]
                records.append(dict(action=action, terminal=terminal,
                                    terminal_wins=sum(1 for v in terminal if v == 1.0),
                                    terminal_unfinished=sum(1 for v in terminal if v is None),
                                    leaf_values=leaves,
                                    leaf_mean=sum(leaves) / len(leaves)))
        finally:
            env.restore_fast(root)
            env.release_snapshot(root)
        path = OUT/'terminal'/f'l{self.level}'/str(self.seed)/f'root_{decision}_{self.model}.json'
        write(path, dict(schema_version=1, level=self.level, seed=self.seed, decision=decision,
                         tick=obs['tick'], wave=obs['wave'], sun=obs['sun'], model=self.model,
                         futures=self.futures, leaf_futures=self.leaf_futures,
                         horizon=self.horizon, actions=records))
        self.written.append(str(path))
        print(json.dumps(dict(level=self.level, seed=self.seed, decision=decision,
                              tick=obs['tick'], actions=len(records))), flush=True)


def terminal_job(job):
    import search_play as teacher
    plan, level, seed, model = job['plan'], job['level'], job['seed'], job['model']
    targets = probe_targets(level, seed)
    path_dir = OUT/'terminal'/f'l{level}'/str(seed)
    if all((path_dir/f'root_{d}_{model}.json').exists() for d in targets):
        return dict(level=level, seed=seed, model=model, skipped=True)
    args = teacher_args(plan, level)
    teacher._close_worker_env()
    recorder = TerminalRecorder(level, seed, targets, model, plan['terminal_probe']['futures_per_action'],
                                plan['terminal_probe']['leaf_futures'], 2500)
    outcome = teacher._play(dict(args=args, seed=seed, arm='fast_search'), observer=recorder)
    return dict(level=level, seed=seed, model=model, terminal=outcome['terminal'],
                won=outcome['won'], probes=len(recorder.written))


def initialize_value_worker(payload):
    plan, model = payload
    import search_play as teacher
    teacher._init_worker(plan['inputs'][model]['path'], plan['inputs']['student']['path'],
                         plan['teacher']['rollout_sim_skip'], plan['inputs']['static']['path'])


def terminal_probe(levels, count, workers):
    plan = verify_plan()
    if not 1 <= workers <= 4:
        raise ValueError('at most four engines for the terminal probe')
    jobs = []
    for level in levels:
        for seed in probe_seeds(level, count):
            for model in ('value', 'v4_control'):
                jobs.append(dict(plan=plan, level=level, seed=seed, model=model))
    if not jobs:
        raise SystemExit('no losing production seeds to probe')
    with controller_lock():
        keep = preload(plan)
        try:
            for model in ('value', 'v4_control'):
                subset = [job for job in jobs if job['model'] == model]
                with mp.get_context('fork').Pool(min(workers, len(subset)),
                        initializer=initialize_value_worker, initargs=((plan, model),)) as pool:
                    for result in pool.imap_unordered(terminal_job, subset, chunksize=1):
                        print(json.dumps(result), flush=True)
        finally:
            keep.cleanup()
    write(OUT/'terminal'/'complete.json', dict(levels=levels, seeds_per_level=count,
          models=['value', 'v4_control'], jobs=len(jobs), complete=True))


def spearman(pairs):
    n = len(pairs)
    if n < 3:
        return None
    def ranks(values):
        order = sorted(range(n), key=lambda i: values[i])
        out = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and values[order[j + 1]] == values[order[i]]:
                j += 1
            rank = (i + j) / 2 + 1
            for k in range(i, j + 1):
                out[order[k]] = rank
            i = j + 1
        return out
    rx, ry = ranks([p[0] for p in pairs]), ranks([p[1] for p in pairs])
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else None


def terminal_report(levels):
    """Leaf value against terminal truth at the same frozen roots (task72b protocol)."""
    summary = dict(schema_version=1, task=76, futures_per_action=None, leaf_horizon=2500,
                   historical_v4_reference='MEMORY/docs/task46-roof-zone.md 53.9% (different roots, model, two futures, teacher continuation)',
                   per_level={})
    for level in levels:
        per_model = {}
        for model in ('value', 'v4_control'):
            roots = sorted((OUT/'terminal'/f'l{level}').glob(f'*/root_*_{model}.json'))
            pairs, argmax_hits, argmax_total, calib, unfinished = [], 0, 0, [], 0
            for path in roots:
                document = read(path)
                summary['futures_per_action'] = document['futures']
                usable = [a for a in document['actions'] if a['terminal_unfinished'] == 0 and a['terminal']]
                unfinished += sum(a['terminal_unfinished'] for a in document['actions'])
                if len(usable) < 2:
                    continue
                truth = {action_key(a['action']): a['terminal_wins'] / len(a['terminal']) for a in usable}
                leaf = {action_key(a['action']): a['leaf_mean'] for a in usable}
                keys = sorted(truth)
                for i in range(len(keys)):
                    for j in range(i + 1, len(keys)):
                        a, b = keys[i], keys[j]
                        if truth[a] == truth[b]:
                            continue
                        pairs.append((leaf[a] - leaf[b], truth[a] - truth[b]))
                best_truth = max(keys, key=lambda k: (truth[k], -leaf[k]))
                best_leaf = max(keys, key=lambda k: (leaf[k], -truth[k]))
                argmax_hits += best_truth == best_leaf
                argmax_total += 1
                calib.append(dict(decision=document['decision'], tick=document['tick'],
                                  leaf_max=round(max(leaf.values()), 4),
                                  truth_max=round(max(truth.values()), 4),
                                  leaf_argmax=json.loads(best_leaf), truth_argmax=json.loads(best_truth),
                                  actions=len(keys)))
            agree = sum(1 for d, t in pairs if (d > 0) == (t > 0))
            per_model[model] = dict(roots=argmax_total, pairs=len(pairs),
                                    pairwise_accuracy=(agree / len(pairs)) if pairs else None,
                                    argmax_agreement=(argmax_hits / argmax_total) if argmax_total else None,
                                    spearman=spearman(pairs), unfinished_futures=unfinished,
                                    calibration=calib)
        summary['per_level'][level] = per_model
    write(ROOT/'MEMORY/docs/task76-roof-terminal-truth.json', summary)
    print(json.dumps({l: {m: {k: v for k, v in d.items() if k != 'calibration'}
                          for m, d in models.items()} for l, models in summary['per_level'].items()},
                     ensure_ascii=False, indent=1))
    return summary


def panel_report(levels):
    plan = read(PLAN)
    arms = plan['arms']
    summary = dict(schema_version=1, task=76, levels=list(levels), arms=arms,
                   seeds_per_level=plan['seeds_per_level'], teacher=plan['teacher'],
                   inputs={k: v['sha256'] for k, v in plan['inputs'].items()},
                   historical_v4_reference=plan['historical_v4_reference'])
    per_level = {}
    for level in levels:
        seeds = sorted(int(p.parent.name) for p in (OUT/'panels'/f'l{level}'/'production').glob('*/result.json'))
        outcomes = {arm: {} for arm in arms}
        for arm in arms:
            for seed in seeds:
                receipt = OUT/'panels'/f'l{level}'/arm/str(seed)/'result.json'
                if receipt.exists():
                    outcomes[arm][seed] = read(receipt)['outcome']
        wins = {arm: sum(1 for o in outcomes[arm].values() if o['won']) for arm in arms}
        profile = {}
        for arm in arms:
            games = list(outcomes[arm].values())
            if not games:
                continue
            waves = sorted(g['final_wave'] for g in games)
            ticks = sorted(g['final_tick'] for g in games)
            planted = {}
            for g in games:
                for kind, count in (g.get('planted_types') or {}).items():
                    planted[kind] = planted.get(kind, 0) + count / len(games)
            profile[arm] = dict(games=len(games), median_final_wave=waves[len(waves)//2],
                                max_final_wave=waves[-1], median_final_tick=ticks[len(ticks)//2],
                                mean_wave_count=sum(g['wave_count'] for g in games)/len(games),
                                mean_decisions=sum(g['decisions'] for g in games)/len(games),
                                mean_plants_per_game=round(sum(planted.values()), 2),
                                plants_per_game={k: round(v, 2) for k, v in sorted(planted.items())},
                                mowers_used=sum(g.get('mowers_used', 0) for g in games)/len(games))
        paired = {}
        for arm in arms[1:]:
            shared = sorted(set(outcomes['production']) & set(outcomes[arm]))
            up = sum(1 for s in shared if not outcomes['production'][s]['won'] and outcomes[arm][s]['won'])
            down = sum(1 for s in shared if outcomes['production'][s]['won'] and not outcomes[arm][s]['won'])
            paired[arm] = dict(seeds=len(shared), production_wins=sum(1 for s in shared if outcomes['production'][s]['won']),
                               arm_wins=sum(1 for s in shared if outcomes[arm][s]['won']),
                               gained=up, lost=down, mcnemar_p=binom_two_sided(min(up, down), up + down))
        detail = {}
        for seed in seeds:
            if all((OUT/'panels'/f'l{level}'/arm/str(seed)/'result.json').exists() for arm in arms):
                detail[seed] = coverage(level, seed, arms)
        flat = [d for seed in detail for arm in arms for d in detail[seed][arm]['decisions']]
        by_arm = {}
        for arm in arms:
            rows = [d for seed in detail for d in detail[seed][arm]['decisions']]
            if rows:
                by_arm[arm] = dict(decisions=len(rows),
                    omitted_best_rate=sum(d['omitted_best'] for d in rows) / len(rows),
                    mean_regret=sum(d['regret'] for d in rows) / len(rows),
                    regret_above_margin=sum(d['regret'] > plan['teacher']['margin'] for d in rows) / len(rows),
                    mean_legal_plants=sum(d['legal_plants'] for d in rows) / len(rows),
                    mean_production_offered=sum(d['production_offered'] for d in rows) / len(rows),
                    mean_evaluated=sum(d['evaluated'] for d in rows) / len(rows),
                    chosen_outside_production=sum(not d['chosen_in_production'] for d in rows) / len(rows),
                    mean_shared_prefix=sum(detail[s][arm]['shared_prefix'] for s in detail) / len(detail))
        per_level[level] = dict(seeds=len(seeds), completed={arm: len(outcomes[arm]) for arm in arms},
                                wins=wins, paired=paired, profile=profile, coverage=by_arm,
                                decision_records=detail)
    detail = {str(level): value.pop('decision_records') for level, value in per_level.items()}
    summary['per_level'] = per_level
    # Per-decision records are megabytes; only aggregates are small enough for git.
    write(OUT/'reports'/'task76-roof-candidates-v5-decisions.json', dict(schema_version=1, per_level=detail))
    write(ROOT/'MEMORY/docs/task76-roof-candidates-v5-report.json', summary)
    print(json.dumps({l: dict(seeds=v['seeds'], wins=v['wins'], paired=v['paired'], profile=v['profile'], coverage=v['coverage'])
                      for l, v in per_level.items()}, ensure_ascii=False, indent=1))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare','verify','smoke','panels','report','terminal','terminal-report'])
    parser.add_argument('--levels', default='42,43')
    parser.add_argument('--games',type=int,default=32)
    parser.add_argument('--workers',type=int,default=MAX_GAME_PROCESSES)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare()
    elif args.mode == 'verify':
        verify_plan(); print('frozen inputs, manifest and production source identities verified')
    elif args.mode == 'smoke':
        with controller_lock():
            smoke()
    elif args.mode == 'report':
        from roof_candidates_v5_analysis import panel_report as audited_panel_report
        audited_panel_report([int(v) for v in args.levels.split(',')])
    elif args.mode == 'terminal':
        from roof_candidates_v5_analysis import terminal_probe as audited_terminal_probe
        audited_terminal_probe([int(v) for v in args.levels.split(',')], args.workers)
    elif args.mode == 'terminal-report':
        from roof_candidates_v5_analysis import terminal_report as audited_terminal_report
        audited_terminal_report([int(v) for v in args.levels.split(',')])
    else:
        panels([int(v) for v in args.levels.split(',')],args.games,args.workers)


if __name__ == '__main__':
    main()
