#!/usr/bin/env python3
"""task74: teacher cold-start sweep on fog L36-39 (two decks) and roof L42-49 (roof_r7).

Production teacher parameters on the deploy4 engine: value_net/v5 leaves, desktop
static-value branch1, dagger beta=1 (teacher plays), verify-samples 9,
student-candidates 2, arms fast_search. 16 cells x 16 fresh seeds at 1.0x.
Seeds [30512, 30768): the registered eval pool's untouched remainder after the
task49A gates (30000-30063) and the task73 retest (30256-30511); training seed
tags are all >= 1600000 (disjointness audit: MEMORY/docs/gate-v2-seed-window-audit.json).
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
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'python')]
from pvz_tasks import deck_for_level

EXECUTABLE = Path('/home/newbiexvwu/PvZ-search/.claude/worktrees/cq/build4/pvz-portable')
VALUE = Path('/home/newbiexvwu/PvZ-search/artifacts/value_net/v5/model.pt')
STATIC_VALUE = Path('/home/newbiexvwu/PvZ-search/artifacts/hard/value_branch1/model.pt')
PROPOSAL = Path('/home/newbiexvwu/pvz-logs/curriculum_state.pipeline/round_000011/student/model.pt')
RESOURCE_DIR = Path('/home/newbiexvwu/.cache/pvz-research-resources')
SEED_BASE = 30512
SEEDS_PER_CELL = 16
WORKERS = 8  # concurrent game processes (task74 cap: 8)
OUT = ROOT / 'artifacts/task74/sweep'

QC_FOG_DECK = list(deck_for_level(36))                       # qc default deck_for_level (fog)
USER_FOG_DECK = [9, 8, 10, 16, 23, 17, 19, 2, 5, 31]         # 主控指定：阳光菇/小喷菇/大喷菇/睡莲/高坚果/窝瓜/缠绕海草/樱桃/寒冰射手/磁力菇
ROOF_R7_DECK = list(deck_for_level(41, family='roof_r7'))    # task46 roof_r7


def cells():
    result = []
    for level in range(36, 40):
        result.append(dict(task_id=f'fog_l{level}_qcdeck', terrain='fog', level=level, deck=QC_FOG_DECK,
                           fog_observation='human'))
        result.append(dict(task_id=f'fog_l{level}_userdeck', terrain='fog', level=level, deck=USER_FOG_DECK,
                           fog_observation='human'))
    for level in range(42, 50):
        result.append(dict(task_id=f'roof_l{level}_r7', terrain='roof', level=level, deck=ROOF_R7_DECK))
    return result


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_manifest() -> Path:
    tasks = []
    for index, cell in enumerate(cells()):
        base = SEED_BASE + index * SEEDS_PER_CELL
        tasks.append(dict(task_id=cell['task_id'], terrain=cell['terrain'], level=cell['level'],
                          deck=cell['deck'], zombie_count_multiplier=1.0, wave_cap=None,
                          sun_start=50, preplanted=[], playthrough=2,
                          seeds=list(range(base, base + SEEDS_PER_CELL)),
                          **({'fog_observation': cell['fog_observation']} if cell.get('fog_observation') else {})))
    manifest = dict(schema_version=1, panel_version='task74-cold-start',
                    purpose='task74 teacher cold-start sweep, 1.0x, 16 fresh seeds per cell',
                    seed_window=[SEED_BASE, SEED_BASE + len(tasks) * SEEDS_PER_CELL], tasks=tasks)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'sweep-manifest.json'
    temporary = path.with_suffix('.building')
    temporary.write_text(json.dumps(manifest, indent=1) + '\n')
    temporary.replace(path)
    return path


def write_provenance(manifest_path: Path) -> None:
    document = dict(
        window=[SEED_BASE, SEED_BASE + 16 * SEEDS_PER_CELL],
        disjoint_evidence=dict(
            consumed_before_task74=[[30000, 30255], [30256, 30511]],
            consumed_before_task74_by=['task49A gates + registered v2 evaluations', 'task73 retest (this worktree)'],
            training_frame_tags_min=1600000,
            audit='MEMORY/docs/gate-v2-seed-window-audit.json'),
        teacher_parameters=dict(executable=str(EXECUTABLE), value=str(VALUE), static_value=str(STATIC_VALUE),
                                proposal_policy=str(PROPOSAL), dagger_beta=1, verify_samples=9,
                                student_candidates=2, arms='fast_search', samples=3, horizon=2500,
                                multiplier=1.0, note='proposal policy is the current production student (round 11); '
                                                    'with dagger_beta=1 the search plays'),
        decks=dict(qc_fog=QC_FOG_DECK, user_fog=USER_FOG_DECK, roof_r7=ROOF_R7_DECK),
        manifest_sha256=sha(manifest_path),
        executable_sha256=sha(EXECUTABLE), value_sha256=sha(VALUE), static_value_sha256=sha(STATIC_VALUE),
        proposal_sha256=sha(PROPOSAL),
    )
    path = OUT / 'provenance.json'
    temporary = path.with_suffix('.building')
    temporary.write_text(json.dumps(document, indent=1) + '\n')
    temporary.replace(path)


def shard_jobs(manifest):
    jobs = []
    for task in manifest['tasks']:
        seeds = task['seeds']
        for shard in range(4):
            jobs.append((task['task_id'], seeds[shard * 4:(shard + 1) * 4]))
    return jobs


def run_one(manifest_path, task_id, seeds):
    folder = OUT / task_id
    folder.mkdir(parents=True, exist_ok=True)
    done = set()
    games = folder / 'games.jsonl'
    if games.exists():
        for line in games.read_text().splitlines():
            row = json.loads(line)
            done.add((row['arm'], row['seed']))
    todo = [s for s in seeds if ('fast_search', s) not in done]
    if not todo:
        return
    command = [sys.executable, str(ROOT / 'scripts/search_play.py'),
               '--rollout-policy', str(PROPOSAL), '--value', str(VALUE),
               '--static-value', str(STATIC_VALUE),
               '--manifest', str(manifest_path), '--task-id', task_id,
               '--seed-list', ','.join(map(str, todo)),
               '--executable', str(EXECUTABLE), '--resource-dir', str(RESOURCE_DIR),
               '--arms', 'fast_search', '--dagger-beta', '1',
               '--verify-samples', '9', '--student-candidates', '2',
               '--workers', '1', '--out', str(folder)]
    started = time.time()
    with (folder / 'run.log').open('a') as log:
        log.write(f'=== {time.strftime("%F %T")} {len(todo)} games ===\n')
        log.flush()
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                env={**os.environ, 'PVZ_NO_POPUP': '1',
                                     'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})
    if result.returncode:
        raise RuntimeError(f'{task_id} {todo}: exit {result.returncode}; see {folder / "run.log"}')


def main():
    manifest_path = build_manifest()
    write_provenance(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    jobs = shard_jobs(manifest)
    running = []
    index = 0
    started = time.time()
    while index < len(jobs) or running:
        while index < len(jobs) and len(running) < WORKERS:
            task_id, seeds = jobs[index]
            index += 1
            proc = subprocess.Popen([sys.executable, '-c',
                'import sys, json; sys.path.insert(0, %r); sys.path.insert(0, %r);'
                'from task74_cold_start_sweep import run_one; run_one(%r, %r, %r)'
                % (str(ROOT / 'scripts'), str(ROOT / 'python'), str(manifest_path), task_id, seeds)],
                cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            running.append((proc, task_id, seeds))
        done = []
        for entry in running:
            if entry[0].poll() is not None:
                if entry[0].returncode:
                    raise RuntimeError(f'{entry[1]} {entry[2]}: shard exit {entry[0].returncode}')
                done.append(entry)
        for entry in done:
            running.remove(entry)
            rows = [json.loads(l) for l in (OUT / entry[1] / 'games.jsonl').read_text().splitlines()]
            mine = [r for r in rows if r['seed'] in set(entry[2]) and r['arm'] == 'fast_search']
            print(f'{entry[1]} {entry[2]}: {sum(r["won"] for r in mine)}/{len(mine)} won '
                  f'({time.time() - started:.0f}s elapsed)', flush=True)
        if running:
            time.sleep(2)
    print(f'ALL DONE in {(time.time() - started) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
