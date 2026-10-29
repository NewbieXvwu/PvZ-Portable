"""task79 control runs: L37 attribution between observation, student and seed block.

Diagnostic only — the manifest lives under artifacts/ (ignored) and is never registered as a
task family.  Two controls, both on the L37 magnet deck:

A  task74's own seeds (30,560..30,575) under this branch's build and human_v2.  If A wins
   like task74 did, the desktop batch is reproducible here and the seed block is not the cause.
B  this branch's seeds (79,020,000..79,020,015) under the old `human` observation.  If B wins,
   the observation change is the cause; if B loses, it is the build/student/level instance.
C  L36 (a level this batch wins) on its own seeds under `human`, as the positive control that the
   observation switch alone does not sink a level.
D  A's seeds and observation with the newer Mac student dw5i_s1 (a106d1a7…).  Diagnostic only: if D
   recovers, the L37 gap is student capability, not deck or level.

Usage: python3 scripts/task79_control.py [A|B|both]
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/task79/control'
MANIFEST = OUT / 'manifest.json'
EXECUTABLE = ROOT / 'build/pvz-portable'
RESOURCE = Path('/Users/newbiexvwu/Downloads/Plants_Vs_Zombies_V1.2.0.1073_EN')
MODELS = {
    'rollout-policy': Path('/Users/newbiexvwu/PvZAgent/artifacts/archive/demo/student/dw4i_s1/model.pt'),
    'value': Path('/Users/newbiexvwu/PvZAgent/artifacts/archive/demo/hard/value_v5/model.pt'),
    'static-value': Path('/Users/newbiexvwu/PvZAgent/artifacts/hard/value_branch1/model.pt'),
}
DECK = [9, 8, 10, 16, 23, 17, 19, 2, 5, 31]
CONTROLS = {
    'A': dict(task_id='fog79_l37_ctrlA_task74seeds', level=37, deck=DECK,
              seeds=list(range(30560, 30576)), fog_observation='human_v2'),
    'B': dict(task_id='fog79_l37_ctrlB_humanobs', level=37, deck=DECK,
              seeds=list(range(79020000, 79020016)), fog_observation='human'),
    'C': dict(task_id='fog79_l36_ctrlC_humanobs', level=36, deck=DECK,
              seeds=list(range(79000000, 79000008)), fog_observation='human'),
    'D': dict(task_id='fog79_l37_ctrlD_dw5i', level=37, deck=DECK,
              seeds=list(range(30560, 30576)), fog_observation='human_v2',
              student='dw5i'),
}
STUDENTS = {
    'dw4i': MODELS['rollout-policy'],
    'dw5i': Path('/Users/newbiexvwu/PvZAgent/artifacts/student/dw5i_s1/model.pt'),
}


def manifest():
    tasks = []
    for control in CONTROLS.values():
        task = {k: v for k, v in control.items() if k != 'student'}
        task['zombie_count_multiplier'] = 1.0
        tasks.append(task)
    OUT.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(dict(tasks=tasks), indent=1) + '\n')


def run(key):
    control = CONTROLS[key]
    folder = OUT / control['task_id']
    folder.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(ROOT / 'scripts/search_play.py'),
               '--manifest', str(MANIFEST), '--task-id', control['task_id'],
               '--executable', str(EXECUTABLE), '--resource-dir', str(RESOURCE),
               '--arms', 'fast_search', '--dagger-beta', '1', '--verify-samples', '9',
               '--student-candidates', '2', '--workers', '4',
               '--seeds', str(len(control['seeds'])), '--out', str(folder)]
    for name, path in MODELS.items():
        command.extend(['--' + name, str(STUDENTS[control.get('student', 'dw4i')]) if name == 'rollout-policy' else str(path)])
    with (folder / 'run.log').open('a') as stream:
        subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT,
                       env={**os.environ, 'PVZ_NO_POPUP': '1', 'OMP_NUM_THREADS': '1',
                            'OPENBLAS_NUM_THREADS': '1'})
    rows = [json.loads(line) for line in (folder / 'games.jsonl').read_text().splitlines()]
    print(key, control['task_id'], sum(row['won'] for row in rows), '/', len(rows), flush=True)


def main():
    os.environ['PVZ_NO_POPUP'] = '1'
    manifest()
    which = sys.argv[1] if len(sys.argv) > 1 else 'both'
    for key in (['A', 'B', 'C', 'D'] if which == 'both' else [which]):
        run(key)


if __name__ == '__main__':
    main()
