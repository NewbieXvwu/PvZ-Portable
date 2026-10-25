#!/usr/bin/env python3
"""task91 回测：同一存档同一未来种子块下，自适应取点的 50% 线 vs 全量 12 点。

用法：compare_same_block.py <results_dir> <scan_jobs.json> <task_id:seed>...
每份的结果 json 需同时含 adaptive 汇总（l50）与补齐后的 12 点 teacher 数据。
"""
import json
import sys
from pathlib import Path

results = Path(sys.argv[1])
plan = json.loads(Path(sys.argv[2]).read_text(encoding='utf-8'))
grids = {(e['task_id'], e['seed']): e['decisions'] for e in plan['archives']}

print(f"{'存档':44s} {'adaptive L50':>12s} {'12点 L50':>10s} {'档差':>4s}  判据")
violations = 0
for spec in sys.argv[3:]:
    task_id, seed = spec.rsplit(':', 1)
    seed = int(seed)
    document = json.loads((results / f'{task_id}_s{seed}.json').read_text(encoding='utf-8'))
    grid = grids[(task_id, seed)]
    points = document['points']
    full_l50 = None
    for index, decision in enumerate(grid):
        data = (points.get(str(decision)) or {}).get('teacher')
        if data and data['games'] and data['wins'] / data['games'] >= 0.5:
            full_l50 = (index, decision)
            break
    adaptive_l50 = document.get('l50')
    if full_l50 is None or adaptive_l50 is None:
        print(f'{task_id}:{seed}  数据不全（full={full_l50} adaptive={adaptive_l50}）')
        violations += 1
        continue
    delta = abs(full_l50[0] - grid.index(adaptive_l50))
    verdict = '通过' if delta <= 1 else '超差'
    violations += delta > 1
    print(f'{task_id}:{seed}  adaptive L50={adaptive_l50}  12点 L50={full_l50[1]}  档差={delta}  {verdict}')
print(f'超差份数：{violations}')
sys.exit(1 if violations else 0)
