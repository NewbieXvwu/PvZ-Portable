"""生成 task129 冻结卡组并复用搜索老师完成实测。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

from deck_rules import RULE_VERSION, generate_deck

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'configs/task129_decks_v5.json'
OUT = ROOT / 'artifacts/task129'
LEVELS = (7, 9, 14, 24, 26, 33, 37, 42, 48)
INPUTS = {
    'rollout-policy': Path('/Users/newbiexvwu/PvZAgent/.claude/worktrees/fog36/artifacts/task79/models/r11/model.pt'),
    'value': Path('/Users/newbiexvwu/PvZAgent/.claude/worktrees/demo/artifacts/hard/value_v5/model.pt'),
    'static-value': Path('/Users/newbiexvwu/PvZAgent/artifacts/hard/value_branch1/model.pt'),
    'executable': Path('/Users/newbiexvwu/PvZAgent/.claude/worktrees/sleepinput/build/pvz-portable'),
}
RESOURCE_DIR = '/Users/newbiexvwu/Downloads/Plants_Vs_Zombies_V1.2.0.1073_EN'


def prepare():
    rng = random.Random(129)
    tasks = []
    for level_index, level in enumerate(LEVELS):
        for deck_index in range(6):
            length = rng.randint(7, 10)
            draw_seed = rng.getrandbits(32)
            start = 129000000 + level_index * 1000 + deck_index * 100
            task = dict(task_id=f'deck129_v5_l{level:02d}_d{deck_index + 1:02d}',
                        level=level, deck=generate_deck(level, length, draw_seed),
                        draw_seed=draw_seed, zombie_count_multiplier=1.0,
                        seeds=list(range(start, start + 8)))
            if 31 <= level <= 39:
                task['fog_observation'] = 'human_v2'
            tasks.append(task)
    document = dict(schema_version=1, rule_version=RULE_VERSION, draw_seed=129,
                    seed_block=[129000000, 129009000], tasks=tasks)
    MANIFEST.write_text(json.dumps(document, indent=2) + '\n')
    provenance = dict(manifest_sha256=hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
                      inputs={name: dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                              for name, path in INPUTS.items()}, resource_dir=RESOURCE_DIR,
                      teacher=dict(dagger_beta=1, verify_samples=9, student_candidates=2))
    (ROOT / 'MEMORY/docs/task129-inputs.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(f'{len(tasks)} decks, {sum(len(task["seeds"]) for task in tasks)} games', flush=True)


def run_task(task):
    folder = OUT / task['task_id']
    folder.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(ROOT / 'scripts/search_play.py'),
               '--manifest', str(MANIFEST), '--task-id', task['task_id'],
               '--arms', 'fast_search', '--dagger-beta', '1',
               '--verify-samples', '9', '--student-candidates', '2',
               '--workers', '1', '--out', str(folder),
               '--frames', str(OUT / 'frames'), '--frames-label', task['task_id'],
               '--resource-dir', RESOURCE_DIR]
    for name, path in INPUTS.items():
        command.extend([f'--{name}', str(path)])
    started = time.monotonic()
    with (folder / 'run.log').open('a') as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                       env={**os.environ, 'PVZ_NO_POPUP': '1', 'OMP_NUM_THREADS': '1',
                            'OPENBLAS_NUM_THREADS': '1', 'CUDA_VISIBLE_DEVICES': ''}, check=True)
    rows = [json.loads(line) for line in (folder / 'games.jsonl').read_text().splitlines()]
    print(f'{task["task_id"]}: {sum(row["won"] for row in rows)}/{len(rows)} '
          f'({time.monotonic() - started:.0f}s)', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'run'))
    args = parser.parse_args()
    if args.phase == 'prepare':
        prepare()
    else:
        tasks = json.loads(MANIFEST.read_text())['tasks']
        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(run_task, tasks))
        print('TASK129_SWEEP_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
