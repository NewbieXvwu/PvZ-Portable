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
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'python')]
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['PVZ_NO_POPUP'] = '1'


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
                deck_checks=checks, workers=6,
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
    if not 1 <= workers <= 6 or not 1 <= count <= 32:
        raise ValueError('one to six engines; at most 32 fixed evaluation seeds')
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare','verify','smoke','panels'])
    parser.add_argument('--levels', default='42,43')
    parser.add_argument('--games',type=int,default=32)
    parser.add_argument('--workers',type=int,default=6)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare()
    elif args.mode == 'verify':
        verify_plan(); print('frozen inputs, manifest and production source identities verified')
    elif args.mode == 'smoke':
        with controller_lock():
            smoke()
    else:
        panels([int(v) for v in args.levels.split(',')],args.games,args.workers)


if __name__ == '__main__':
    main()
