"""Strict audit/report for task43 round2; pass absolute pilot/remaining plan JSON paths."""
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys

from value_teacher_feasibility import digest, group_summary, load_plan, write_json


def report(paths):
    plans = [load_plan(p) for p in paths]
    seeds = [s for p in plans for s in p['seeds']]
    if len(seeds) != len(set(seeds)) or len(seeds) > 8:
        raise ValueError('duplicate or excess planned seeds')
    if any(p['teacher'] != plans[0]['teacher'] or p['sha256'] != plans[0]['sha256']
           or p['waves'] != plans[0]['waves'] for p in plans):
        raise ValueError('incompatible plans')
    groups = []; results = []; labels = []; hashes = {}; resources = []
    by_wave = defaultdict(list)
    for p, filename in zip(plans, paths):
        hashes[str(filename)] = digest(filename)
        out = Path(p['out'])
        completed = json.loads((out/'completed.json').read_text())
        if Counter(completed['seeds']) != Counter(p['seeds']):
            raise ValueError('controller did not complete exactly planned seeds')
        actual = {int(x.name.removeprefix('seed_')) for x in out.glob('seed_*') if x.is_dir()}
        if actual != set(p['seeds']): raise ValueError('unexpected/missing seed directory')
        for seed in p['seeds']:
            d = out/f'seed_{seed}'; result = json.loads((d/'result.json').read_text())
            if result['seed'] != seed or result['terminal'] is not True:
                raise ValueError('root not terminal or seed mismatch')
            if any(w <= result['final_frame']['wave'] for w in result['missing_wave_targets']):
                raise ValueError('missed target even though root reached its wave')
            expected_waves = set(p['waves']) - set(result['missing_wave_targets'])
            actual_waves = {int(x.name.removeprefix('wave_')) for x in d.glob('wave_*') if x.is_dir()}
            if actual_waves != expected_waves: raise ValueError('unexpected/missing root')
            results.append(result)
            for target in sorted(expected_waves):
                root_dir = d/f'wave_{target}'
                root = json.loads((root_dir/'root.json').read_text())
                candidates = root['candidates']
                if root['common_future'] != (root['decision_seed']*1000) % 2**32:
                    raise ValueError('root common future formula differs')
                if len({json.dumps(c,sort_keys=True) for c in candidates}) != len(candidates):
                    raise ValueError('duplicate candidate')
                files = sorted(root_dir.glob('candidate_[0-9]*.json'))
                if len(files) != len(candidates): raise ValueError('missing/excess candidate outcome')
                rows = [json.loads(f.read_text()) for f in files]
                if {x['candidate'] for x in rows} != set(range(len(candidates))):
                    raise ValueError('candidate indices do not exactly cover actual search set')
                for row in rows:
                    if row['action'] != candidates[row['candidate']]: raise ValueError('action differs')
                    if row['common_future'] != root['common_future']: raise ValueError('future differs')
                    if row['continuation_seed_base'] != root['decision_seed']+5000:
                        raise ValueError('continuation future schedule differs')
                    if set(row['history_checks']) != {'frame','engine_state','income','income_start'} or not all(
                            x is True for x in row['history_checks'].values()):
                        raise ValueError('history check failed')
                    if type(row['outcome']['won']) is not bool or row['outcome']['terminal'] is not True:
                        raise ValueError('invalid terminal label')
                    if row['outcome']['won'] != (row['outcome']['result'] == 1):
                        raise ValueError('win/result differs')
                    if any(not math.isfinite(row[k]) or row[k] < 0
                           for k in ('continuation_cpu_seconds','replay_cpu_seconds')):
                        raise ValueError('invalid CPU measurement')
                    trace = root_dir/f"candidate_{row['candidate']}_teacher.jsonl"
                    actions = [json.loads(line) for line in trace.read_text().splitlines()]
                    if [x['decision'] for x in actions] != list(range(row['outcome']['decisions'])):
                        raise ValueError('continuation trace missing or repeated decisions')
                    hashes[str(trace)] = digest(trace)
                summary = group_summary(rows)
                saved = json.loads((root_dir/'summary.json').read_text())
                if any(saved[k] != v for k,v in summary.items()):
                    raise ValueError('group summary differs from raw outcomes')
                if root['decision_seed']//10000 != seed: raise ValueError('root seed differs')
                summary.update(seed=seed,wave_target=target,root_tick=root['root_frame']['tick'])
                groups.append(summary); by_wave[target].append(summary); labels.extend(rows)
                for f in [root_dir/'root.json',root_dir/'summary.json',*files]: hashes[str(f)]=digest(f)
            for f in [d/'result.json',d/'root_trajectory.jsonl']: hashes[str(f)]=digest(f)
        samples = [json.loads(x) for x in (out/'resources.jsonl').read_text().splitlines()]
        resources.append(samples)
    total_cpu = sum(x['total_cpu_seconds'] for x in results)
    continuation_cpu = sum(x['continuation_cpu_seconds'] for x in labels)
    replay_cpu = sum(x['replay_cpu_seconds'] for x in labels)
    effective = sum(g['effective_pairs'] for g in groups)
    n_groups = len(groups); n_labels = len(labels); n_seeds = len(seeds)
    if not n_groups or not n_labels: raise ValueError('no completed groups')
    mixed = sum(g['mixed'] for g in groups)
    qc_core_seconds = qc_time = 0
    for samples in resources:
        for a,b in zip(samples,samples[1:]):
            if b['time'] <= a['time']: raise ValueError('resource timestamps not ordered')
            qc_time += b['time']-a['time']
            ac,bc = a['qc_cpu_seconds_by_pid'],b['qc_cpu_seconds_by_pid']
            qc_core_seconds += sum(max(0,bc[k]-ac[k]) for k in ac.keys() & bc.keys())
    all_samples = [x for xs in resources for x in xs]
    max_own = max(x['own_engines'] for x in all_samples)
    if max_own > 4: raise ValueError('four-engine budget exceeded')
    cores = min(x['logical_cpus'] for x in all_samples)
    # qc reserves sixteen workers. Actual samples may be during its pack/wait phase.
    reserved_qc = 16
    available = min(4,max(0,cores-reserved_qc))
    per_lane_util = total_cpu/sum(r['wall_seconds'] for r in results)
    observed_capacity = available*min(1,per_lane_util)
    overhead_per_seed = (total_cpu-continuation_cpu-replay_cpu)/n_seeds
    cost_per_group = (continuation_cpu+replay_cpu)/n_groups
    searched_per_seed = sum(r['searched'] for r in results)/n_seeds
    def estimate(groups_per_seed, futures):
        hours = 64*(overhead_per_seed+groups_per_seed*cost_per_group*futures)/3600
        return dict(cpu_hours=hours,
            elapsed_hours_at_four_reserved_cores=hours/available if available else None,
            elapsed_hours_at_observed_lane_utilization=hours/observed_capacity if observed_capacity else None,
            candidate_terminal_labels=64*groups_per_seed*n_labels/n_groups*futures,
            effective_pairs_linear_projection=64*groups_per_seed*effective/n_groups*futures)
    return dict(status='complete',seeds=sorted(seeds),root_games_won=sum(r['won'] for r in results),
        root_games=n_seeds,root_groups=n_groups,missing_targets={str(r['seed']):r['missing_wave_targets'] for r in results},
        mixed_groups=mixed,mixed_fraction=mixed/n_groups,all_loss_groups=sum(g['all_loss'] for g in groups),
        labels=n_labels,wins=sum(x['outcome']['won'] for x in labels),
        label_win_fraction=sum(x['outcome']['won'] for x in labels)/n_labels,
        effective_pairs=effective,total_candidate_pairs=sum(g['total_pairs'] for g in groups),
        cpu_seconds=dict(total=total_cpu,teacher_continuation=continuation_cpu,history_replay=replay_cpu,
            root_and_instrumentation=total_cpu-continuation_cpu-replay_cpu,
            per_candidate_terminal_pair_total=total_cpu/n_labels,
            per_effective_ranking_pair_total=total_cpu/effective if effective else None,
            per_effective_ranking_pair_continuation=continuation_cpu/effective if effective else None),
        mean_search_roots_per_game=searched_per_seed,groups=groups,
        per_wave={str(k):dict(groups=len(v),mixed=sum(x['mixed'] for x in v),
            labels=sum(x['labels'] for x in v),wins=sum(x['wins'] for x in v),
            effective_pairs=sum(x['effective_pairs'] for x in v),
            cpu_seconds=sum(x['continuation_cpu_seconds']+x['replay_cpu_seconds'] for x in v))
            for k,v in by_wave.items()},
        resources=dict(samples=len(all_samples),max_own_engines=max_own,logical_cpus=cores,
            qc_engines_min=min(x['qc_engines'] for x in all_samples),
            qc_engines_max=max(x['qc_engines'] for x in all_samples),
            qc_mean_cpu_cores_lower_bound=qc_core_seconds/qc_time if qc_time else None,
            qc_reserved_workers=reserved_qc,assumed_available_cores=available,
            mean_cpu_per_lane_wall_second=per_lane_util),
        expansion_64=dict(same_three_roots_single_future=estimate(n_groups/n_seeds,1),
            all_search_roots_single_future=estimate(searched_per_seed,1),
            all_search_roots_three_futures=estimate(searched_per_seed,3)),
        limitations=['One shared actual first-round future per root: observed win/loss contrasts, not precise win probabilities.',
            'Twelve intended roots in four independent games; candidate pairs and roots within a game are correlated.',
            'Expansion extrapolates stratified wave samples; continuation cost varies with phase and policy state.',
            'CPU excludes parent model preload/report and small worker-init costs; includes root/continuation Python and engine plus replay/check overhead.',
            'qc CPU cores from surviving PID deltas are a lower bound; reserved16/20 cores is an allocation scenario, not a promise of throughput.'],
        input_sha256=plans[0]['sha256'],teacher=plans[0]['teacher'],artifact_sha256=hashes)


if __name__ == '__main__':
    paths = [Path(x) for x in sys.argv[1:]]
    if not paths or not all(x.is_absolute() for x in paths): raise ValueError('absolute plan paths required')
    value = report(paths)
    output = paths[0].parent/'summary.json'
    write_json(output,value)
    print(json.dumps({k:v for k,v in value.items() if k not in ('artifact_sha256','groups','teacher')},indent=1))
