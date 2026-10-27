"""任务124：只调已有等待候选参数，记录探针漏斗与配对对局。"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import time

os.environ.setdefault('PVZ_NO_POPUP', '1')

import search_play as sp

ROOT = Path(__file__).resolve().parents[1]
PARAMETERS = {
    'A': dict(save_candidates=1, save_share=0.5, save_max_wait=1800, save_interval=600),
    'B': dict(save_candidates=4, save_share=0.05, save_max_wait=2400, save_interval=600),
}


def play(job):
    args = job['args']
    directory = args.out / f"{job['variant']}_l{job['level']}_{job['seed']}"
    directory.mkdir(parents=True, exist_ok=True)
    original_plan, original_probe = sp._save_plan, sp._save_probe
    context = {}
    counts = Counter()
    stats_ref = {}
    economy = dict(max_sun=0, spending=Counter(), first_spending=Counter())
    before = {}
    with (directory / 'probes.jsonl').open('w') as probes, (directory / 'timeline.jsonl').open('w') as timeline:
        def trace_probe(env, obs, root, packet, cost, search_args, decision_seed, until_tick):
            current = obs
            original_step = env.step
            def step(action):
                nonlocal current
                response = original_step(action)
                current = response[0]
                return response
            env.step = step
            try:
                plan = original_probe(env, obs, root, packet, cost, search_args, decision_seed, until_tick)
            finally:
                env.step = original_step
            legal = sum(entry['packet'] == packet for entry in
                        (current.get('legal_actions') or {}).get('plants') or [])
            reason = ('scored' if plan is not None else 'terminal' if current['terminal'] else
                      'unaffordable' if current['sun'] < cost else 'no_legal_cell' if not legal else 'no_static_score')
            counts[reason] += 1
            record = dict(tick=obs['tick'], sun=obs['sun'], packet=packet,
                          plant_type=obs['packets'][packet]['type'], cost=cost,
                          waited=current['tick'] - obs['tick'], end_sun=current['sun'],
                          end_terminal=current['terminal'], legal_cells=legal, reason=reason,
                          current_best=context['best'], plan=plan,
                          gap=None if plan is None else plan['value'] - context['best'])
            probes.write(json.dumps(record) + '\n')
            return plan

        def trace_plan(env, obs, root, candidates, totals, search_args, decision_seed, stats):
            context['best'] = max(float(value) for value in totals if math.isfinite(float(value)))
            stats_ref.clear()
            stats_ref.update(reference=stats)
            return original_plan(env, obs, root, candidates, totals, search_args, decision_seed, stats)

        class Economy:
            def before(self, env, obs, state, decision):
                before.clear()
                before.update(tick=obs['tick'], sun=obs['sun'], packets=obs['packets'])
                economy['max_sun'] = max(economy['max_sun'], obs['sun'])

            def after(self, obs, action, policy_choice, rejected, info):
                record = dict(tick=before['tick'], sun=before['sun'], action=action,
                              student_action=policy_choice, rejected=bool(rejected))
                if action['type'] == 'plant':
                    packet = before['packets'][action['packet']]
                    record.update(plant_type=packet['type'], cost=packet['cost'])
                    if not rejected:
                        economy['spending'][packet['type']] += packet['cost']
                        if before['tick'] < 6000:
                            economy['first_spending'][packet['type']] += packet['cost']
                timeline.write(json.dumps(record) + '\n')

        sp._save_plan, sp._save_probe = trace_plan, trace_probe
        try:
            result = sp._play(job, observer=Economy())
        finally:
            sp._save_plan, sp._save_probe = original_plan, original_probe
    stats = stats_ref.get('reference', {})
    result.update(level=job['level'], task_id=args.task_id, variant=job['variant'],
                  probes=dict(counts), economy=economy,
                  **{name: stats.get(name, 0) for name in ('save_probes', 'save_overrides', 'save_wait_ticks')})
    (directory / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def summarize(rows):
    summary = {}
    for level in sorted({row['level'] for row in rows}):
        paired = {variant: {row['seed']: row for row in rows if row['level'] == level
                            and row['variant'] == variant} for variant in PARAMETERS}
        seeds = sorted(set(paired['A']) & set(paired['B']))
        gained = sum(paired['B'][seed]['won'] and not paired['A'][seed]['won'] for seed in seeds)
        lost = sum(paired['A'][seed]['won'] and not paired['B'][seed]['won'] for seed in seeds)
        discordant = gained + lost
        probability = min(1.0, 2 * sum(math.comb(discordant, count) for count in range(min(gained, lost) + 1))
                          / 2 ** discordant) if discordant else 1.0
        row = dict(pairs=len(seeds), gained=gained, lost=lost, mcnemar_p=probability)
        for variant, games in paired.items():
            plants, probes = Counter(), Counter()
            for game in games.values():
                plants.update(game['planted_types'])
                probes.update(game['probes'])
            row[variant] = dict(games=len(games), wins=sum(game['won'] for game in games.values()),
                seconds=sum(game['seconds'] for game in games.values()), planted_types=dict(plants),
                probes=dict(probes), save_overrides=sum(game['save_overrides'] for game in games.values()),
                save_wait_ticks=sum(game['save_wait_ticks'] for game in games.values()))
        summary[str(level)] = row
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('pilot', 'ab'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, choices=range(1, 11), default=10)
    parser.add_argument('--source-config', type=Path, default=ROOT.parent /
        'blindspot/artifacts/task120/prod/base_task101_l42_roof_demo/config.json')
    options = parser.parse_args()
    options.out = options.out.resolve()
    options.out.mkdir(parents=True, exist_ok=True)
    config = json.loads(options.source_config.read_text())
    panel = json.loads((ROOT / 'configs/adventure_scoreboard.json').read_text())
    levels = (42,) if options.phase == 'pilot' else (42, 43, 44, 7)
    offset, count = (16, 1) if options.phase == 'pilot' else (32, 16)
    tasks = [dict(task, seeds=list(range(panel['seed_block'][0] + (task['level'] - 1) * panel['stride'] + offset,
                                      panel['seed_block'][0] + (task['level'] - 1) * panel['stride'] + offset + count)))
             for task in panel['tasks'] if task['level'] in levels]
    manifest = options.out / 'manifest.json'
    manifest.write_text(json.dumps(dict(schema_version=1, tasks=tasks), indent=2) + '\n')
    jobs = []
    for task in sorted(tasks, key=lambda entry: levels.index(entry['level'])):
        for seed in task['seeds']:
            for variant, parameters in PARAMETERS.items():
                flags = []
                for name in ('rollout_policy', 'value', 'static_value', 'resource_dir', 'executable',
                             'per_card', 'student_candidates', 'samples', 'horizon', 'verify_samples',
                             'min_gap', 'rollout_sim_skip'):
                    flags.extend(['--' + name.replace('_', '-'), str(config[name])])
                args = sp.parser().parse_args(flags + ['--manifest', str(manifest), '--task-id', task['task_id'],
                                                       '--out', str(options.out)])
                args.halving = str(config['halving']).lower() == 'true'
                vars(args).update(parameters)
                jobs.append(dict(args=args, seed=seed, arm='fast_search', variant=variant, level=task['level']))
    (options.out / 'config.json').write_text(json.dumps(dict(source_config=str(options.source_config),
        source=config, parameters=PARAMETERS, workers=options.workers, phase=options.phase), indent=2) + '\n')
    keep, values = sp._export_values(config['value'], config['static_value'])
    rows = []
    try:
        sp._load_nets(values[0], config['rollout_policy'], values[1])
        sp._freeze_startup()
        started = time.monotonic()
        with (options.out / 'games.jsonl').open('w') as stream, mp.get_context('fork').Pool(
                min(options.workers, len(jobs)), initializer=sp._init_worker,
                initargs=(values[0], config['rollout_policy'], args.rollout_sim_skip, values[1])) as pool:
            for completed, result in enumerate(pool.imap_unordered(play, jobs), 1):
                rows.append(result)
                stream.write(json.dumps(result) + '\n')
                stream.flush()
                print(f"{completed}/{len(jobs)} L{result['level']} {result['variant']} {result['seed']} "
                      f"won={result['won']} probes={result['probes']} {time.monotonic() - started:.1f}s", flush=True)
            pool.close()
            pool.join()
    finally:
        keep.cleanup()
        if sp._W.get('rollout_tmp'):
            sp._W['rollout_tmp'].cleanup()
    (options.out / 'summary.json').write_text(json.dumps(summarize(rows), indent=2) + '\n')


if __name__ == '__main__':
    main()
