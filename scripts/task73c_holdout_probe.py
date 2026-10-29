#!/usr/bin/env python3
"""task73c: pick the day_l9 holdout replacement by measuring the incumbent student.

day_l9_eval_d7p_holdout1 sits on the floor (0-2/192 on both sides in the round-13
gate), so it contributes nothing but 384 wasted episodes. The replacement must be
a registered day-family holdout whose incumbent win rate lands in 15-85%, verified
here on fresh seeds before it enters the v3 panel - not assumed.

Seeds come from the same out-of-git gate ledger as the gate itself (a probe is gate
work): one contiguous block, never reused. Evaluation parameters match the round-13
gate contract exactly (deploy4 engine, value v5, arms=fast, dagger-beta=0), so the
rates transfer to the gate. Re-running resumes missing (arm, seed) games only.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]
from promotion_gate import allocate_seed_block, read, write  # noqa: E402
from pvz_promotion import SEED_LEDGER_DEFAULT  # noqa: E402

# Every registered day-family holdout the round-13 gate could swap in for the
# floor day_l9 holdout, except day_l7 holdout1 (already measured by the task73
# retest at the ceiling, r11 28/32) and the floor day_l9 holdout1 itself.
# Measured result (2026-10-25): the whole group is strongly polarized on the
# incumbent - floors at 0-2% and ceilings at 87.5-100%, none inside 15-85%.
CANDIDATES = (
    'day_l7_eval_d7p_holdout2', 'day_l7_eval_d7p_holdout3',
    'day_l8_eval_d7p_holdout1', 'day_l8_eval_d7p_holdout2', 'day_l8_eval_d7p_holdout3',
    'day_l9_eval_d7p_holdout2', 'day_l9_eval_d7p_holdout3',
    'day_l10_eval_d7p_holdout1', 'day_l10_eval_d7p_holdout2', 'day_l10_eval_d7p_holdout3',
)
SEEDS_PER_CANDIDATE = 64


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-panel', type=Path,
                        default=(Path(__file__).resolve().parents[1] / 'configs/panels/unified_v2.json'))
    parser.add_argument('--incumbent', type=Path,
                        default=Path('/home/newbiexvwu/pvz-logs/curriculum_state.pipeline/round_000011/student/model.pt'))
    parser.add_argument('--executable', type=Path,
                        default=Path('/home/newbiexvwu/PvZ-search/.claude/worktrees/cq/build4/pvz-portable'))
    parser.add_argument('--value', type=Path,
                        default=Path('/home/newbiexvwu/PvZ-search/artifacts/value_net/v5/model.pt'))
    parser.add_argument('--resource-dir', type=Path, default=Path.home() / '.cache/pvz-research-resources')
    parser.add_argument('--python', type=Path, default=Path('/home/newbiexvwu/.venvs/ml/bin/python3'))
    parser.add_argument('--seed-ledger', type=Path, default=Path(SEED_LEDGER_DEFAULT))
    parser.add_argument('--out', type=Path, default=ROOT / 'artifacts/task73c/probe')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    for name in ('source_panel', 'incumbent', 'executable', 'value', 'resource_dir', 'seed_ledger'):
        if not getattr(args, name).is_absolute():
            parser.error(f'--{name.replace("_", "-")} must be absolute')
    if not args.incumbent.is_file():
        parser.error(f'incumbent model missing: {args.incumbent}')
    args.out.mkdir(parents=True, exist_ok=True)

    total = len(CANDIDATES) * SEEDS_PER_CANDIDATE
    start, end = allocate_seed_block(args.seed_ledger, total)
    print(f'seed ledger: probe block [{start}, {end}) from {args.seed_ledger}', flush=True)
    document = read(args.source_panel)
    table = {task['task_id']: task for task in document['tasks']}
    missing = [name for name in CANDIDATES if name not in table]
    if missing:
        parser.error(f'candidates not registered in {args.source_panel}: {missing}')
    tasks = []
    for index, name in enumerate(CANDIDATES):
        task = dict(table[name])
        task['seeds'] = list(range(start + index * SEEDS_PER_CANDIDATE,
                                   start + (index + 1) * SEEDS_PER_CANDIDATE))
        tasks.append(task)
    manifest = dict(schema_version=1, panel_version='v2',
                    groups={task['task_id']: 'heldout' for task in tasks}, tasks=tasks,
                    purpose=f'task73c holdout replacement probe; ledger seeds [{start}, {end}); '
                            f'incumbent {args.incumbent}')
    manifest_path = args.out / 'probe_manifest.json'
    write(manifest_path, manifest)

    started = time.time()
    summary = dict(incumbent=str(args.incumbent), executable=str(args.executable),
                   value=str(args.value), seed_block=[start, end],
                   seeds_per_candidate=SEEDS_PER_CANDIDATE, candidates={})
    for task in tasks:
        name = task['task_id']
        folder = args.out / name
        folder.mkdir(parents=True, exist_ok=True)
        games_path = folder / 'games.jsonl'
        complete = games_path.exists() and len(games_path.read_text().splitlines()) == SEEDS_PER_CANDIDATE
        if not complete:
            command = [str(args.python), str(ROOT / 'scripts/search_play.py'),
                       '--rollout-policy', str(args.incumbent), '--value', str(args.value),
                       '--manifest', str(manifest_path), '--task-id', name,
                       '--executable', str(args.executable), '--resource-dir', str(args.resource_dir),
                       '--seeds', str(SEEDS_PER_CANDIDATE), '--workers', str(args.workers),
                       '--arms', 'fast', '--dagger-beta', '0', '--out', str(folder)]
            with (folder / 'run.log').open('a') as log:
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                        env={**os.environ, 'PVZ_NO_POPUP': '1',
                                             'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1',
                                             'OPENBLAS_NUM_THREADS': '1'})
            if result.returncode:
                raise RuntimeError(f'{name}: exit {result.returncode}; see {folder / "run.log"}')
        rows = [json.loads(line) for line in games_path.read_text().splitlines() if line.strip()]
        if len(rows) != SEEDS_PER_CANDIDATE or {row['seed'] for row in rows} != set(task['seeds']):
            raise RuntimeError(f'{name}: incomplete probe games ({len(rows)})')
        wins = sum(1 for row in rows if row['won'])
        summary['candidates'][name] = dict(wins=wins, games=len(rows),
                                           rate=wins / len(rows))
        print(f'{name}: {wins}/{len(rows)}', flush=True)
    summary['elapsed_seconds'] = time.time() - started
    summary['in_band'] = [name for name, row in summary['candidates'].items()
                          if .15 <= row['rate'] <= .85]
    write(args.out / 'probe-summary.json', summary)
    print(json.dumps(summary['candidates'], indent=2), flush=True)


if __name__ == '__main__':
    main()
