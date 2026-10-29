#!/usr/bin/env python3
"""task73: promotion-gate power retest on fresh seeds (read-only wrt qc/cq/pipeline).

Re-runs the exact gate evaluation parameters (search_play, arms=fast, dagger-beta 0,
deploy3 executable, value_net/v5) on NEW seeds carved from the registered v2
evaluation pool [30000,31024) minus every seed consumed by a recorded evaluation,
which is also disjoint from all training seed tags (all >= 1600000).

Five students (round 2/5/8/10/11) x 8 panel tasks x 32 fresh seeds = 1280 episodes.
Pairwise comparisons reuse the same per-(task,seed) outcomes; statistics come from
pvz_promotion (the gate's own implementation).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]
from pvz_promotion import PANEL_TASKS, summarize, paired_interval

PIPELINE = Path('/home/newbiexvwu/pvz-logs/curriculum_state.pipeline')
FROZEN_MANIFEST = PIPELINE / 'round_000011/evaluations/c83bfc91077e9168e9b8/manifest.json'
UNIFIED = (Path(__file__).resolve().parents[1] / 'configs/panels/unified_v2.json')
EXECUTABLE = Path('/home/newbiexvwu/PvZ-search/artifacts/archive/deploy3/build/pvz-portable')
VALUE = Path('/home/newbiexvwu/PvZ-search/artifacts/value_net/v5/model.pt')
RESOURCE_DIR = Path('/home/newbiexvwu/.cache/pvz-research-resources')
STUDENTS = {
    'r2': PIPELINE / 'round_000002/student/model.pt',
    'r5': PIPELINE / 'round_000005/student/model.pt',
    'r8': PIPELINE / 'round_000008/student/model.pt',
    'r10': PIPELINE / 'round_000010/student/model.pt',
    'r11': PIPELINE / 'round_000011/student/model.pt',
}
SEEDS_PER_TASK = 32
POOL_BASE = 30256  # first pool seed not consumed by any recorded evaluation (scan: 30000-30255 used)
OUT = ROOT / 'artifacts/task73/retest'


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_manifest() -> Path:
    frozen = json.loads(FROZEN_MANIFEST.read_text())
    by_id = {t['task_id']: t for t in frozen['tasks']}
    tasks = []
    for index, (_, name) in enumerate(PANEL_TASKS):
        task = json.loads(json.dumps(by_id[name]))
        base = POOL_BASE + index * SEEDS_PER_TASK
        task['seeds'] = list(range(base, base + SEEDS_PER_TASK))
        tasks.append(task)
    manifest = dict(schema_version=1, panel_version='task73-retest',
                    groups=frozen['groups'], tasks=tasks,
                    purpose='task73 fresh-seed retest of the task49A promotion panel')
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'retest-manifest.json'
    temporary = path.with_suffix('.building')
    temporary.write_text(json.dumps(manifest, indent=1) + '\n')
    temporary.replace(path)
    return path


def write_provenance(manifest_path: Path) -> None:
    document = dict(
        pool_source=str(UNIFIED), pool_source_sha256=sha(UNIFIED),
        pool_window=[30000, 31024],
        frozen_panel_seeds=[30000, 30063],
        consumed_by_recorded_evaluations=[30000, 30255],
        fresh_window=[POOL_BASE, POOL_BASE + 8 * SEEDS_PER_TASK],
        training_disjoint_evidence=dict(
            frames_roots=['/home/newbiexvwu/PvZ-search/artifacts/student/frames_v2',
                          '/home/newbiexvwu/pvz-logs/curriculum_state.pipeline/frames'],
            seed_tags_scanned=157863, min_training_seed=1600000,
            training_seeds_inside_pool=0),
        assignment='task index i (PANEL_TASKS order) gets seeds '
                   f'[{POOL_BASE}+32i, {POOL_BASE}+32(i+1))',
        frozen_manifest_source=str(FROZEN_MANIFEST),
        frozen_manifest_sha256=sha(FROZEN_MANIFEST),
        retest_manifest_sha256=sha(manifest_path),
        executable=str(EXECUTABLE), executable_sha256=sha(EXECUTABLE),
        value=str(VALUE), value_sha256=sha(VALUE),
        students={k: dict(path=str(v), sha256=sha(v)) for k, v in STUDENTS.items()},
    )
    path = OUT / 'provenance.json'
    temporary = path.with_suffix('.building')
    temporary.write_text(json.dumps(document, indent=1) + '\n')
    temporary.replace(path)


def run_episode_block(manifest_path: Path, student: str, task_id: str, seeds: list[int]) -> None:
    folder = OUT / student / task_id
    folder.mkdir(parents=True, exist_ok=True)
    done = set()
    if (folder / 'games.jsonl').exists():
        for line in (folder / 'games.jsonl').read_text().splitlines():
            row = json.loads(line)
            done.add((row['arm'], row['seed']))
    missing = [s for s in seeds if ('fast', s) not in done]
    if not missing:
        return
    command = [sys.executable, str(ROOT / 'scripts/search_play.py'),
               '--rollout-policy', str(STUDENTS[student]), '--value', str(VALUE),
               '--manifest', str(manifest_path), '--task-id', task_id,
               '--seed-list', ','.join(map(str, seeds)),
               '--executable', str(EXECUTABLE), '--resource-dir', str(RESOURCE_DIR),
               '--arms', 'fast', '--dagger-beta', '0', '--workers', '5',
               '--out', str(folder)]
    started = time.time()
    with (folder / 'run.log').open('a') as log:
        log.write(f'=== {time.strftime("%F %T")} {len(missing)} games ===\n')
        log.flush()
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                env={**os.environ, 'PVZ_NO_POPUP': '1',
                                     'CUDA_VISIBLE_DEVICES': '',
                                     'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})
    if result.returncode:
        raise RuntimeError(f'{student} {task_id}: exit {result.returncode}; see {folder / "run.log"}')
    rows = [json.loads(line) for line in (folder / 'games.jsonl').read_text().splitlines()]
    got = {row['seed'] for row in rows if row['arm'] == 'fast'}
    if not set(seeds) <= got:
        raise RuntimeError(f'{student} {task_id}: missing seeds after run')
    timing_path = OUT / 'timing.jsonl'
    with timing_path.open('a') as tj:
        tj.write(json.dumps(dict(student=student, task_id=task_id,
                                 wall_seconds=time.time() - started,
                                 episodes=len(missing),
                                 engine_seconds=sum(row.get('seconds') or 0 for row in rows
                                                    if row['arm'] == 'fast' and row['seed'] in set(seeds)))) + '\n')
    print(f'{student} {task_id}: {len(missing)} episodes in {time.time() - started:.0f}s '
          f'({sum(1 for row in rows if row["arm"] == "fast" and row["seed"] in set(seeds) and row["won"])}/{len(seeds)} won)',
          flush=True)


def outcomes(student: str, task_id: str, seeds: list[int]) -> dict[int, bool]:
    rows = [json.loads(line) for line in (OUT / student / task_id / 'games.jsonl').read_text().splitlines()]
    return {row['seed']: row['won'] for row in rows if row['arm'] == 'fast' and row['seed'] in set(seeds)}


def analyse(manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    wins = {s: {t['task_id']: outcomes(s, t['task_id'], t['seeds']) for t in manifest['tasks']} for s in STUDENTS}
    results = {}
    names = list(STUDENTS)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pairs = []
            for task in manifest['tasks']:
                for seed in task['seeds']:
                    pairs.append(dict(task_id=task['task_id'], seed=seed,
                                      zone=manifest['groups'][task['task_id']],
                                      candidate=wins[a][task['task_id']][seed],
                                      incumbent=wins[b][task['task_id']][seed]))
            results[f'{a}_vs_{b}'] = dict(overall=summarize(pairs),
                                          zones={z: summarize([p for p in pairs if p['zone'] == z])
                                                 for z in sorted(set(manifest['groups'].values()))},
                                          by_task={t['task_id']: summarize(
                                              [p for p in pairs if p['task_id'] == t['task_id']])
                                              for t in manifest['tasks']})
    document = dict(results=results,
                    raw_wins={s: {t: {str(seed): w for seed, w in tasks.items()}
                                  for t, tasks in by_task.items()} for s, by_task in wins.items()})
    path = OUT / 'retest-results.json'
    temporary = path.with_suffix('.building')
    temporary.write_text(json.dumps(document, indent=1) + '\n')
    temporary.replace(path)
    return document


def main() -> None:
    manifest_path = build_manifest()
    write_provenance(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    for student in STUDENTS:
        for task in manifest['tasks']:
            run_episode_block(manifest_path, student, task['task_id'], task['seeds'])
    document = analyse(manifest_path)
    print(json.dumps({k: dict(pairs=v['overall']['pairs'],
                              delta=round(v['overall']['delta_rate'], 4),
                              ci=[round(x, 4) for x in v['overall']['ci95']])
                      for k, v in document['results'].items()}, indent=1), flush=True)


if __name__ == '__main__':
    main()
