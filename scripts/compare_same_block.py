#!/usr/bin/env python3
"""task91 回测：同一存档同一未来种子块下，自适应取点的 50% 线 vs 全量 12 点。

用法：compare_same_block.py <results_dir> <scan_jobs.json> <task_id:seed>...
每份的结果 json 需同时含 adaptive 汇总（l50）与补齐后的 12 点 teacher 数据。
"""
import json
import sys
from pathlib import Path

from takeover_scan import AdaptiveScan, validate_result, validate_summary


def compare_document(entry: dict, document: dict) -> dict:
    """要求完整网格与独立自适应轨迹，且逐局未来与结果完全对应。"""
    trace = document.get('adaptive_trace') or {}
    grid = entry['decisions']
    futures = trace.get('future_seeds')
    if not trace.get('complete') or trace.get('grid') != grid:
        raise ValueError('缺少完整的自适应取点轨迹')
    if not futures or futures != entry['future_seeds'][:len(futures)]:
        raise ValueError('自适应未来与清单不是同一块')
    validate_result(document)
    points = document['points']
    full = {}
    for decision in grid:
        data = points.get(str(decision), {}).get('teacher')
        if data is None:
            raise ValueError(f'全量网格缺少老师点 {decision}')
        validate_summary(data, futures)
        full[decision] = data
    teacher = {int(d): bucket['teacher'] for d, bucket in trace.get('points', {}).items() if 'teacher' in bucket}
    order = trace.get('teacher_decisions') or []
    if len(order) != len(teacher) or set(order) != set(teacher) or len(order) > 9:
        raise ValueError('自适应取点轨迹不完整或超过预算')
    coarse = 5 if trace.get('algorithm', '').startswith('coarse5_') else 4
    initial = [grid[i] for i in AdaptiveScan._pick_equidistant(len(grid), coarse)]
    if order[:len(initial)] != initial:
        raise ValueError('自适应没有先跑等距点')
    for decision, data in teacher.items():
        validate_summary(data, futures)
        rows = lambda item: {r['future_seed']: r for r in item['rows']}
        if decision not in full or rows(data) != rows(full[decision]):
            raise ValueError('自适应与全量逐局证据不同')
    actual_games = 0
    for bucket in trace['points'].values():
        for data in bucket.values():
            validate_summary(data, futures)
            actual_games += data['games']
    if actual_games != trace.get('scan_games') or actual_games >= len(grid) * len(futures):
        raise ValueError('自适应成本记录不符或未低于全量两臂的一半')
    passing = lambda values: min((d for d, data in values.items() if data['wins'] * 2 >= data['games']), default=None)
    adaptive_l50, full_l50 = passing(teacher), passing(full)
    if adaptive_l50 != trace.get('l50'):
        raise ValueError('自适应L50与轨迹不一致')
    delta = (0 if adaptive_l50 is full_l50 is None else
             abs(grid.index(adaptive_l50) - grid.index(full_l50))
             if adaptive_l50 is not None and full_l50 is not None else None)
    return dict(task_id=entry['task_id'], seed=entry['seed'], adaptive_l50=adaptive_l50,
                full_l50=full_l50, delta=delta, teacher_points=len(teacher), scan_games=actual_games,
                full_two_arm_games=len(grid) * len(futures) * 2,
                passed=delta is not None and delta <= 1)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    results, plan = Path(argv[0]), json.loads(Path(argv[1]).read_text())
    entries = {(e['task_id'], e['seed']): e for e in plan['archives']}
    violations = 0
    for spec in argv[2:]:
        task_id, seed = spec.rsplit(':', 1)
        entry = entries[(task_id, int(seed))]
        document = json.loads((results / f'{task_id}_s{seed}.json').read_text())
        try:
            result = compare_document(entry, document)
            print(json.dumps(result, ensure_ascii=False))
            violations += not result['passed']
        except ValueError as error:
            print(f'{spec}: {error}')
            violations += 1
    return int(bool(violations))


if __name__ == '__main__':
    raise SystemExit(main())
