#!/usr/bin/env python3
"""task104 预登记判据的机械实现：读两套面板的 summary.json，给出结论表与通过/未通过。

判据来自 `MEMORY/docs/task104-demo-opening-rows.md` 的「训练前预登记」，写在这里是为了
让结论可以重算，而不是手抄：

  1. 两个训练种子的屋顶三关合并胜率都提高（gain_s > 0，s ∈ {1,2}）
  2. 两种子平均净提高 >= 10 个百分点
  3. 任一屋顶关的两种子平均净下降 <= 1/32
  4. 非屋顶回归：两种子平均总体净下降 <= 3 个百分点，任一实际覆盖区净下降 <= 5 个百分点

四条全过才算"示范有效"，否则结论是"本配方未证实收益"（不调阈值、不部署）。

  python3 scripts/task104_verdict.py --roof artifacts/task104/eval/roof/summary.json \
      --regression artifacts/task104/eval/regression/summary.json --out artifacts/task104/verdict.json
"""
from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path

from value_widen_ab import paired

SEEDS = ('1', '2')
MIN_MEAN_GAIN = Fraction(10, 100)
MAX_LEVEL_DROP = Fraction(1, 32)
MAX_REGRESSION_DROP = Fraction(3, 100)
MAX_REGION_DROP = Fraction(5, 100)


def model_name(arm: str, seed: str) -> str:
    return f'{arm}_s{seed}'


def rate(games_won: int, games: int) -> Fraction:
    if not games:
        raise ValueError('panel arm has no games')
    return Fraction(games_won, games)


def collect(summary: dict, arm: str, seed: str) -> dict:
    name = model_name(arm, seed)
    if name not in summary['models']:
        raise KeyError(f'{name} missing from summary; have {sorted(summary["models"])}')
    return summary['models'][name]


def by_level(model: dict) -> dict[str, dict]:
    return {task['task_id']: task for task in model['tasks']}


def by_region(model: dict) -> dict[str, tuple[int, int]]:
    return {region: (bucket['won'], bucket['games']) for region, bucket in model['terrains'].items()}


def paired_tasks(summary: dict) -> list[dict]:
    rows = []
    for seed in SEEDS:
        a, b = by_level(collect(summary, 'a', seed)), by_level(collect(summary, 'b', seed))
        for task_id in sorted(set(a) | set(b)):
            if task_id not in a or task_id not in b:
                raise KeyError(f'task {task_id} missing from one arm')
            if a[task_id]['seeds'] != b[task_id]['seeds']:
                raise ValueError(f'task {task_id}: paired seeds differ')
            for task in (a[task_id], b[task_id]):
                if set(task['outcomes']) != {str(seed) for seed in task['seeds']}:
                    raise ValueError(f'task {task_id}: outcomes differ from frozen seeds')
            rows.append({'training_seed': int(seed), 'task_id': task_id,
                         'a': f'{a[task_id]["won"]}/{a[task_id]["games"]}',
                         'b': f'{b[task_id]["won"]}/{b[task_id]["games"]}',
                         **paired(a[task_id]['outcomes'], b[task_id]['outcomes'])})
    return rows


def compare(roof: dict, regression: dict) -> dict:
    rows, failures = [], []
    gains = []
    for seed in SEEDS:
        a = collect(roof, 'a', seed)
        b = collect(roof, 'b', seed)
        a_rate = rate(a['won'], a['games'])
        b_rate = rate(b['won'], b['games'])
        gain = b_rate - a_rate
        gains.append(gain)
        rows.append({'training_seed': int(seed), 'a': f'{a["won"]}/{a["games"]}',
                     'b': f'{b["won"]}/{b["games"]}',
                     'a_rate': float(a_rate), 'b_rate': float(b_rate), 'gain_pp': float(gain * 100)})
        if gain <= 0:
            failures.append(f'训练种子 {seed}：屋顶合并胜率没有提高（{float(gain * 100):+.1f} 个百分点）')
    mean_gain = sum(gains) / len(gains)
    if mean_gain < MIN_MEAN_GAIN:
        failures.append(f'两种子平均净提高 {float(mean_gain * 100):+.1f} 个百分点，低于 10')

    levels = paired_tasks(roof)
    per_level = {}
    for row in levels:
        per_level.setdefault(row['task_id'], []).append(Fraction(row['b'].split('/')[0]) / Fraction(row['b'].split('/')[1])
                                                        - Fraction(row['a'].split('/')[0]) / Fraction(row['a'].split('/')[1]))
    level_summary = []
    for task_id, values in sorted(per_level.items()):
        mean = sum(values) / len(values)
        level_summary.append({'task_id': task_id, 'mean_delta': float(mean)})
        if mean < -MAX_LEVEL_DROP:
            failures.append(f'{task_id}：两种子平均净下降 {float(-mean * 100):.1f} 个百分点，超过 1/32')

    regions, region_summary = [], []
    for seed in SEEDS:
        a, b = by_region(collect(regression, 'a', seed)), by_region(collect(regression, 'b', seed))
        for region in sorted(set(a) | set(b)):
            if region not in a or region not in b:
                raise KeyError(f'regression region {region} missing from one arm')
            regions.append((region, rate(*b[region]) - rate(*a[region])))
            region_summary.append({'training_seed': int(seed), 'region': region,
                                   'a': f'{a[region][0]}/{a[region][1]}',
                                   'b': f'{b[region][0]}/{b[region][1]}',
                                   'delta': float(rate(*b[region]) - rate(*a[region]))})
    overall = []
    for seed in SEEDS:
        a = collect(regression, 'a', seed)
        b = collect(regression, 'b', seed)
        overall.append(rate(b['won'], b['games']) - rate(a['won'], a['games']))
    overall_mean = sum(overall) / len(overall)
    if overall_mean < -MAX_REGRESSION_DROP:
        failures.append(f'非屋顶回归两种子平均净下降 {float(-overall_mean * 100):.1f} 个百分点，超过 3')
    region_means = {}
    for region, delta in regions:
        region_means.setdefault(region, []).append(delta)
    for region, values in sorted(region_means.items()):
        mean = sum(values) / len(values)
        if mean < -MAX_REGION_DROP:
            failures.append(f'回归区 {region}：两种子平均净下降 {float(-mean * 100):.1f} 个百分点，超过 5')

    return {'schema': 'task104_verdict_v1',
            'criteria': {'min_mean_gain_pp': float(MIN_MEAN_GAIN * 100),
                         'max_level_drop_games_of_32': float(MAX_LEVEL_DROP * 32),
                         'max_regression_drop_pp': float(MAX_REGRESSION_DROP * 100),
                         'max_region_drop_pp': float(MAX_REGION_DROP * 100)},
            'roof': rows, 'roof_levels': level_summary, 'roof_paired': levels,
            'roof_training_seed_ranges_pp': {
                key: [min(row[field] for row in rows) * factor, max(row[field] for row in rows) * factor]
                for key, field, factor in [('a', 'a_rate', 100), ('b', 'b_rate', 100), ('gain', 'gain_pp', 1)]},
            'regression': {'overall_pp': [float(x * 100) for x in overall],
                           'overall_mean_pp': float(overall_mean * 100),
                           'training_seed_gain_range_pp': [float(min(overall) * 100), float(max(overall) * 100)],
                           'paired': paired_tasks(regression),
                           'regions': region_summary,
                           'region_means_pp': {k: float(sum(v) / len(v) * 100) for k, v in region_means.items()}},
            'failures': failures,
            'verdict': 'demonstration rows help' if not failures else 'not proven; do not deploy'}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--roof', type=Path, required=True)
    p.add_argument('--regression', type=Path, required=True)
    p.add_argument('--out', type=Path)
    args = p.parse_args()
    result = compare(json.loads(args.roof.read_text()), json.loads(args.regression.read_text()))
    print(f'屋顶  A -> B（每行一个训练种子）')
    for row in result['roof']:
        print(f"  seed {row['training_seed']}: {row['a']} -> {row['b']}  {row['gain_pp']:+.1f} 个百分点")
    mean_gain_pp = sum(r['gain_pp'] for r in result['roof']) / len(result['roof'])
    print(f'两种子平均净提高 {mean_gain_pp:+.1f} 个百分点')
    print('非屋顶回归（总体，两种子）', [f'{v:+.1f}' for v in result['regression']['overall_pp']])
    print('分区', {k: round(v, 1) for k, v in result['regression']['region_means_pp'].items()})
    if result['failures']:
        print('未通过：')
        for failure in result['failures']:
            print('  -', failure)
    print('结论：', result['verdict'])
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1) + '\n')
        print('wrote', args.out)


if __name__ == '__main__':
    main()
