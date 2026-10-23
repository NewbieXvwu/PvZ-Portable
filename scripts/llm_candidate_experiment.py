"""task52 preregistered A/B_low/B_novel study; never invoked by curriculum_queue.

Stages: prepare -> calibrate -> run -> summarize. Each stage checks frozen inputs.
Only this harness installs search_play._W['llm_proposer']. At most two owned games.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
MAIN = Path('/home/newbiexvwu/PvZ-search')
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]
from llm_candidates import GeminiCache, Triggers, atomic_json, canonical, digest

FAMILIES = ('day', 'roof', 'fog')
ARMS = ('A', 'B_low', 'B_novel')
SOURCES = ('scripts/llm_candidate_experiment.py', 'scripts/llm_candidates.py', 'scripts/search_play.py',
           'scripts/render_episode.py', 'scripts/episode_query.py', 'scripts/pvz_tasks.py',
           'python/pvz_env.py', 'python/pvz_event_env.py', 'python/pvz_value_net.py',
           'python/pvz_rollout_policy.py', 'python/pvz_student_observation.py', 'python/pvz_fog.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def task_definitions():
    from pvz_tasks import deck_for_level, terrain_covered
    day = next(t for t in read(MAIN / 'artifacts/stage4/day_panel.json')['tasks']
               if t['task_id'] == 'day_l7_eval_d7')
    tasks = [deepcopy(day)]
    for family, level in [('roof', 41), ('fog', 31)]:
        deck = deck_for_level(level, **({'family': 'roof_r7'} if family == 'roof' else {}))
        tasks.append(dict(task_id='roof_l41_eval_r7' if family == 'roof' else 'fog_l31_eval_f8',
                          terrain=family, level=level, deck=list(deck), zombie_count_multiplier=1.0,
                          wave_cap=None, sun_start=50, preplanted=[], playthrough=2,
                          **({'fog_observation': 'human'} if family == 'fog' else {})))
    for index, task in enumerate(tasks):
        assert terrain_covered(task['level'], task['deck'])
        task['seeds'] = (list(range(52000000+1000*index, 52000016+1000*index))
                         + list(range(52010000+1000*index, 52010002+1000*index)))
    return tasks


def prepare(out):
    tasks = task_definitions()
    manifest = out / 'manifest.json'
    definition = dict(schema_version=1, tasks=tasks, purpose='task52 paired comparison and disjoint calibration')
    if manifest.exists() and read(manifest) != definition:
        raise ValueError('manifest changed; use a new study directory')
    atomic_json(manifest, definition)
    model_paths = dict(student=MAIN/'artifacts/student/dw4i_s1/model.pt',
                       value=MAIN/'artifacts/value_net/v4/model.pt',
                       static=MAIN/'artifacts/hard/value_branch1/model.pt')
    config = dict(rollout_policy=str(model_paths['student']), value=str(model_paths['value']),
                  static_value=str(model_paths['static']), executable=str(ROOT/'build/pvz-portable'),
                  resource_dir='/home/newbiexvwu/.cache/pvz-research-resources',
                  manifest=str(manifest), out=str(out), per_card=1, student_candidates=2,
                  samples=3, verify_samples=9, halving=True, horizon=2500, min_gap=300,
                  rollout_sim_skip=61, dagger_beta=1.0, frames=None, leaf_probe=None)
    settings = dict(quantile=.25, max_calls=4, interval=1200, adjacent_rows=1, terminal_futures=2)
    paths = [ROOT / p for p in SOURCES] + list(model_paths.values()) + [ROOT/'build/pvz-portable', manifest]
    plan = dict(config=config, settings=settings, files={str(p): sha(p) for p in paths},
                tasks={name: task['task_id'] for name, task in zip(FAMILIES, tasks)},
                model='gemini-3.8-flash-high', endpoint='http://localhost:8317',
                preregistration_sha256=sha(ROOT/'MEMORY/docs/llm-candidates.md'))
    target = out / 'plan.json'
    if target.exists() and read(target) != plan:
        raise ValueError('frozen plan changed; use a new study directory')
    atomic_json(target, plan)
    return plan


def verify(plan):
    for filename, expected in plan['files'].items():
        if sha(filename) != expected:
            raise ValueError(f'frozen input changed: {filename}')


def init_worker(config, files):
    import search_play as t
    t._init_worker(files[0], config['rollout_policy'], config['rollout_sim_skip'], files[1])


class Proposer:
    def __init__(self, arm, plan, threshold, calibration=False):
        self.arm, self.calibration = arm, calibration
        self.settings = plan['settings']
        self.triggers = Triggers(threshold=threshold, **{k: self.settings[k] for k in
                                 ('max_calls', 'interval', 'adjacent_rows')})
        self.client = GeminiCache(ROOT/'artifacts/task52/cache', base_url=plan['endpoint'], model=plan['model'])
        self.events, self.leaves, self.current = [], [], None
        self.diagnostic_cpu, self.trigger_cpu = 0., 0.

    def __call__(self, env, obs, root, candidates, policy, args, decision_seed, stats):
        import search_play as t
        self.current = None
        tick = obs['tick']
        if not self.calibration and not self.triggers.budget_allows(tick):
            return []
        evidence = {}
        if self.calibration or self.arm == 'B_low':
            anchor = policy if policy['type'] == 'plant' else t.NOOP
            started = time.monotonic()
            jobs = [(anchor, decision_seed*1000+k) for k in range(args.samples)]
            # Diagnostic probes have their own stats so they cannot alter teacher counts.
            handles = t._continuations(env, obs, root, jobs, args, {})
            values = t._leaf_values(handles)
            self.trigger_cpu += time.monotonic()-started
            self.leaves.extend(values)
            evidence['policy_leaf_values'] = values
            if self.calibration or not self.triggers.low(values):
                return []
        elif self.arm == 'B_novel':
            if policy['type'] == 'plant':
                row = policy['row']
            else:
                top, _keys = t._student_topk_candidates(obs['legal_actions']['plants'], 1)
                row = top[0]['row'] if top else None
            types = self.triggers.novel(obs, row)
            if not types:
                return []
            evidence.update(reference_row=row, new_types=types)
        else:
            return []
        # A preprobe may leave the engine on a future leaf. Gemini gets obs (the original
        # public root), and all actual candidate continuations restore this same root.
        reply = self.client.propose(obs)
        self.triggers.called(tick, evidence.get('new_types', []))
        present = {canonical(a) for a in candidates}
        novel = [a for a in reply['actions'] if canonical(a) not in present]
        self.current = dict(source='llm', decision_seed=decision_seed, tick=tick, wave=obs['wave'],
                            trigger=self.arm, evidence=evidence, **reply, novel_actions=novel,
                            packet_types={str(p['index']): p['type'] for p in obs['packets']})
        self.events.append(self.current)
        return reply['actions']

    def chosen(self, env, obs, root, candidates, totals, policy_index, chosen, decision_seed):
        if self.current is None:
            return
        record = self.current
        key = canonical(chosen)
        record['chosen'] = chosen
        record['suggestion_selected'] = key in {canonical(a) for a in record['actions']}
        record['novel_selected'] = key in {canonical(a) for a in record['novel_actions']}
        record['scores'] = [[a, v if math.isfinite(v) else None] for a, v in zip(candidates, totals)]
        record['anchor_value'] = totals[policy_index]
        if not record['novel_selected']:
            return
        from pvz_cpu_usage import usage, process_usage, difference
        import search_play as t
        index = next(i for i, a in enumerate(candidates) if canonical(a) == key)
        record['selected_leaf_advantage'] = totals[index]-totals[policy_index]
        worker0, engine0 = usage(), process_usage(env._process.pid)
        started = time.monotonic()
        outcomes = []
        for k in range(self.settings['terminal_futures']):
            future = (decision_seed*1000 + 9000001 + k) % 2**32
            pair = {}
            for name, action in [('selected', chosen), ('anchor', candidates[policy_index])]:
                leaf = env.rollout(action, 0, snapshot_id=root, seed=future, max_actions=3000,
                                   sim_skip=t._W['rollout_sim_skip'],
                                   student_history=t._W.get('student_history'))
                pair[name] = dict(terminal=bool(leaf['terminal']), won=leaf.get('result') == 1,
                                  final_wave=leaf['wave'], final_tick=leaf['tick'])
            outcomes.append(dict(future=future, **pair))
        cpu = difference(usage(), worker0)['total_s'] + difference(process_usage(env._process.pid), engine0)['total_s']
        record.update(terminal_pairs=outcomes, diagnostic_cpu_s=cpu,
                      diagnostic_wall_s=time.monotonic()-started)
        self.diagnostic_cpu += cpu


def run_game(job, plan, thresholds):
    import search_play as t
    from pvz_cpu_usage import usage, process_usage, difference
    verify(plan)
    family, arm, seed, calibration = job
    out = Path(plan['config']['out'])
    dest = out / ('calibration' if calibration else 'games') / family / arm / str(seed)
    contract = digest(plan)
    result_path = dest/'result.json'
    if result_path.exists():
        result = read(result_path)
        trace = read(dest/'trace.json')
        if (result['contract'] == contract and result['terminal'] and len(trace) == result['decisions']
                and sha(dest/'trace.json') == result['trace_sha256']):
            return result
        raise ValueError(f'cannot resume inconsistent result: {dest}')
    t._close_worker_env()
    config = dict(plan['config'], task_id=plan['tasks'][family])
    args = SimpleNamespace(**config)
    proposer = Proposer(arm, plan, thresholds.get(family), calibration)
    proposer.client.context = dict(family=family, arm=arm, seed=seed, calibration=calibration)
    trace = []
    original_search, original_choice = t._search, t._verified_choice

    def search(env, obs, policy, settings, decision_seed, stats):
        proposer.triggers.observe(obs)
        before = stats['searched']
        chosen = original_search(env, obs, policy, settings, decision_seed, stats)
        trace.append(dict(tick=obs['tick'], wave=obs['wave'], root=digest(obs), policy=policy,
                          chosen=chosen, searched=stats['searched'] != before))
        return chosen

    def choice(env, obs, root, candidates, totals, counts, best, policy_index, policy, settings,
               decision_seed, stats, **kw):
        chosen = original_choice(env, obs, root, candidates, totals, counts, best, policy_index,
                                 policy, settings, decision_seed, stats, **kw)
        proposer.chosen(env, obs, root, candidates, totals, policy_index, chosen, decision_seed)
        return chosen

    t._search, t._verified_choice = search, choice
    if arm != 'A' or calibration:
        t._W['llm_proposer'] = proposer
    worker0, started = usage(), time.monotonic()
    try:
        result = t._play(dict(seed=seed, arm='fast_search', args=args))
        engine = process_usage(t._W['env']._process.pid)
        cpu = difference(usage(), worker0)['total_s'] + sum(engine.values())
        assert result['terminal'] and len(trace) == result['decisions']
        atomic_json(dest/'trace.json', trace)
        atomic_json(dest/'events.json', proposer.events)
        result.update(family=family, comparison_arm=arm, calibration=calibration, contract=contract,
                      trace_sha256=sha(dest/'trace.json'), events_sha256=sha(dest/'events.json'),
                      cpu_s_including_diagnostics=cpu, wall_s_including_diagnostics=time.monotonic()-started,
                      diagnostic_cpu_s=proposer.diagnostic_cpu, trigger_wall_s=proposer.trigger_cpu,
                      calibration_leaf_values=proposer.leaves if calibration else [],
                      llm_queries=len(proposer.events),
                      llm_cache_hits=sum(e['cache_hit'] for e in proposer.events))
        atomic_json(result_path, result)
        return result
    finally:
        t._search, t._verified_choice = original_search, original_choice
        t._W.pop('llm_proposer', None)
        t._close_worker_env()


def execute(plan, *, calibration=False, workers=2):
    import search_play as t
    verify(plan)
    out = Path(plan['config']['out'])
    if calibration:
        thresholds = {}
    else:
        calibrated = read(out/'calibration.json')
        if calibrated['contract'] != digest(plan):
            raise ValueError('calibration is from another frozen study')
        thresholds = calibrated['thresholds']
    jobs = [(family, arm, (52010000 if calibration else 52000000)+1000*i+j, calibration)
            for j in range(2 if calibration else 16) for i, family in enumerate(FAMILIES)
            for arm in (('A',) if calibration else ARMS)]
    keep, files = t._export_values(plan['config']['value'], plan['config']['static_value'])
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'),
                                 initializer=init_worker, initargs=(plan['config'], files)) as pool:
            futures = {pool.submit(run_game, job, plan, thresholds): job for job in jobs}
            results = []
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(canonical(dict(done=len(results), total=len(jobs), family=result['family'],
                                     arm=result['comparison_arm'], seed=result['seed'], won=result['won'],
                                     wave=result['final_wave'], llm_queries=result['llm_queries'])), flush=True)
    finally:
        keep.cleanup()
    if calibration:
        import numpy as np
        values = {f: [v for r in results if r['family'] == f for v in r['calibration_leaf_values']]
                  for f in FAMILIES}
        if any(not v for v in values.values()):
            raise ValueError('calibration has no policy leaves')
        atomic_json(out/'calibration.json', dict(contract=digest(plan),
                    quantile=plan['settings']['quantile'], counts={f:len(v) for f,v in values.items()},
                    thresholds={f:float(np.quantile(v, plan['settings']['quantile'])) for f,v in values.items()}))


def paired_summary(before, after):
    assert len(before) == len(after) == 16
    assert [r['seed'] for r in before] == [r['seed'] for r in after]
    wins = [int(b['won'])-int(a['won']) for a,b in zip(before,after)]
    waves = [b['final_wave']-a['final_wave'] for a,b in zip(before,after)]
    improved, regressed = wins.count(1), wins.count(-1)
    n = improved+regressed
    p = min(1., 2*sum(math.comb(n,k) for k in range(min(improved,regressed)+1))/2**n)
    return dict(games=16, before_wins=sum(r['won'] for r in before), after_wins=sum(r['won'] for r in after),
                net_wins=sum(wins), improved=improved, regressed=regressed, paired_binomial_p=p,
                wave_deltas=waves, mean_wave_delta=statistics.mean(waves),
                wave_delta_range=[min(waves),max(waves)])


def summarize(plan):
    verify(plan)
    out = Path(plan['config']['out'])
    report = dict(contract=digest(plan), status='complete', families={})
    manifest = {t['task_id']: t for t in read(plan['config']['manifest'])['tasks']}
    for i, family in enumerate(FAMILIES):
        arms, events = {}, {}
        for arm in ARMS:
            arms[arm], events[arm] = [], []
            for seed in range(52000000+1000*i,52000016+1000*i):
                d = out/'games'/family/arm/str(seed)
                result = read(d/'result.json')
                assert result['contract']==digest(plan) and result['terminal']
                assert sha(d/'trace.json') == result['trace_sha256']
                assert sha(d/'events.json') == result['events_sha256']
                assert len(read(d/'trace.json')) == result['decisions']
                arms[arm].append(result);events[arm].extend(read(d/'events.json'))
        comparisons = {}
        for arm in ARMS[1:]:
            e = events[arm]
            counts = Counter()
            tokens = Counter()
            actual = Counter()
            types = Counter()
            raw_types = Counter()
            for event in e:
                numeric = lambda d: {k:v for k,v in d.items() if type(v) in (int,float)}
                counts.update(event['counts']);tokens.update(numeric(event['usage']))
                for attempt in event['attempts']:
                    if attempt['ok']:
                        actual.update(numeric(attempt.get('usage', {})))
                for action in event['actions']:
                    types[int(event['packet_types'][str(action['packet'])])] += 1
                payload = event.get('payload')
                for a in (payload.get('actions', []) if isinstance(payload,dict) else []):
                    if isinstance(a,dict) and str(a.get('packet')) in event['packet_types']:
                        raw_types[int(event['packet_types'][str(a['packet'])])] += 1
            pairs = [p for event in e for p in event.get('terminal_pairs', [])]
            completed = [p for p in pairs if p['selected']['terminal'] and p['anchor']['terminal']]
            row = paired_summary(arms['A'],arms[arm])
            row.update(queries=len(e), queries_per_game=len(e)/16, counts=dict(counts),
                       cache_hits=sum(x['cache_hit'] for x in e),
                       actual_api_attempts=sum(len(x['attempts']) for x in e),
                       logical_tokens=dict(tokens), actual_tokens_from_completed_games=dict(actual),
                       monetary_cost=None, monetary_cost_reason='gateway tariff not provided',
                       selected_queries=sum(x['suggestion_selected'] for x in e),
                       selected_novel_queries=sum(x['novel_selected'] for x in e),
                       novel_proposals=sum(len(x['novel_actions']) for x in e),
                       selected_leaf_advantages=[x['selected_leaf_advantage'] for x in e if x['novel_selected']],
                       terminal_diagnostic_pairs=len(pairs), terminal_diagnostic_complete=len(completed),
                       terminal_diagnostic_net_wins=sum(int(p['selected']['won'])-int(p['anchor']['won']) for p in completed),
                       terminal_diagnostic_wave_deltas=[p['selected']['final_wave']-p['anchor']['final_wave'] for p in completed],
                       legal_proposal_plant_types=dict(types), raw_resolvable_proposal_plant_types=dict(raw_types),
                       search_points=sum(r['searched'] for r in arms[arm]),
                       cpu_s_including_diagnostics=sum(r['cpu_s_including_diagnostics'] for r in arms[arm]),
                       diagnostic_cpu_s=sum(r['diagnostic_cpu_s'] for r in arms[arm]))
            ratio = lambda n,d: n/d if d else None
            row.update(selected_query_rate=ratio(row['selected_queries'],len(e)),
                       selected_novel_query_rate=ratio(row['selected_novel_queries'],len(e)),
                       selected_novel_proposal_rate=ratio(row['selected_novel_queries'],row['novel_proposals']),
                       selected_legal_proposal_rate=ratio(row['selected_queries'],counts['valid']),
                       selected_search_rate=ratio(row['selected_queries'],row['search_points']),
                       illegal_rate=ratio(counts['illegal']+counts['malformed']+counts['excess'],counts['returned']),
                       legal_clear_fog_proportion=ratio(sum(types[k] for k in (22,25,27)),counts['valid']),
                       legal_counter_proportions={str(k):ratio(types[k],counts['valid']) for k in (31,26,27,23)},
                       raw_resolvable_clear_fog_proportion=ratio(sum(raw_types[k] for k in (22,25,27)),counts['returned']),
                       raw_resolvable_counter_proportions={str(k):ratio(raw_types[k],counts['returned']) for k in (31,26,27,23)})
            comparisons[arm] = row
        report['families'][family] = dict(task=manifest[plan['tasks'][family]], comparisons=comparisons)
    ledger = ROOT/'artifacts/task52/cache/requests.jsonl'
    entries = [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
    report['api_ledger_including_retries_and_interrupted_games'] = dict(
        attempts=len(entries), failed_attempts=sum(not e['ok'] for e in entries),
        tokens=dict(sum((Counter({k:v for k,v in e.get('usage',{}).items() if type(v) in (int,float)})
                        for e in entries), Counter())))
    atomic_json(out/'summary.json', report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['prepare','calibrate','run','summarize'])
    p.add_argument('--out', type=Path, default=ROOT/'artifacts/task52/study-v1')
    p.add_argument('--workers', type=int, default=2, choices=(1,2))
    args = p.parse_args()
    if args.stage == 'prepare':
        plan = prepare(args.out)
        print(canonical(dict(out=str(args.out), contract=digest(plan))), flush=True)
    else:
        plan = read(args.out/'plan.json')
        if args.stage in ('calibrate','run'):
            import fcntl
            with (args.out/'runner.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                owned = []
                for proc in Path('/proc').iterdir():
                    if not proc.name.isdigit():
                        continue
                    try:
                        if (proc/'exe').resolve() == ROOT/'build/pvz-portable':
                            owned.append(proc.name)
                    except OSError:
                        continue
                if owned:
                    raise RuntimeError('owned game still running; inspect it before restarting')
                execute(plan, calibration=args.stage=='calibrate', workers=args.workers)
        else:
            print(canonical(summarize(plan)))


if __name__ == '__main__':
    main()
