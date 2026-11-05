"""task249：固定种子，调用生产入口逐项比较老师实验与采集。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from curriculum_queue import command_for
from make_gate_coverage_panel import tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'artifacts/task249')
    parser.add_argument('--workers', type=int, default=3, choices=(1, 2, 3))
    parser.add_argument('--groups', help='逗号分隔的组名，省略则全部运行')
    parser.add_argument('--driver-root', type=Path, required=True,
                        help='与只读引擎协议匹配的现有 search_play 工作树')
    args = parser.parse_args()
    directory = args.data.resolve()
    work = directory / 'work'
    work.mkdir(parents=True, exist_ok=True)
    models = directory / 'models'
    gate = next(t for t in json.loads((ROOT / 'configs/gate_v8_panel.json').read_text())['tasks']
                if t['task_id'] == 'task222_l22_pool_user')
    gate.update(seeds=list(range(243002000, 243002016)), pairs=16,
                zombie_count_multiplier=0.5)
    new = next(t for t in tasks() if t['level'] == 22)
    new.update(task_id='pool_user_l22_user_m050', zombie_count_multiplier=0.5,
               seeds=gate['seeds'], pairs=16)
    panel = directory / 'panel.json'
    panel.write_text(json.dumps(dict(tasks=[gate, new]), indent=1) + '\n')
    # 各组共用种子，单变量的配对关系登记在 task249 报告中。
    groups = [
        ('old_teacher', 'old', 'build-vn6', gate, False, 1, True),
        ('r119_teacher', 'r119', 'build-vn6', gate, False, 1, True),
        ('build2_teacher', 'r119', 'build2', gate, False, 1, True),
        ('new_task_teacher', 'r119', 'build2', new, False, 1, True),
        ('frames_only_teacher', 'r119', 'build2', new, True, 1, True),
        ('collection_teacher', 'r119', 'build2', new, True, 1, True),
        ('collection_student', 'r119', 'build2', new, True, 0, True),
        ('python_teacher', 'r119', 'build2', new, True, 1, False),
        ('old_collection_student', 'old', 'build2', new, True, 0, True),
        ('production_driver', 'r119', 'build2', new, True, 1, True),
        ('r117_teacher', 'r117', 'build2', new, True, 1, True),
        ('r117_collection_student', 'r117', 'build2', new, True, 0, True),
    ]
    selected = set(args.groups.split(',')) if args.groups else {g[0] for g in groups}
    environment = dict(os.environ, PVZ_NO_POPUP='1', OMP_NUM_THREADS='1',
                       OPENBLAS_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', TMPDIR=str(work))
    for name, student, build, task, frames, beta, native in groups:
        if name not in selected:
            continue
        options = SimpleNamespace(python=sys.executable, rollout_policy=models / f'{student}.pt',
            rollout_policy_gpt=models / 'gptbc.pt', value=models / 'value.pt',
            value_roof=models / 'roof.pt', static_value=models / 'static.pt',
            executable=Path.home() / f'pvz-mac-collect/{build}/pvz-portable',
            resource_dir=Path.home() / '.cache/pvz-research-resources',
            frames=directory / 'frames' / name, out=directory, workers=args.workers,
            native_games=native, obs_v3=frames and name != 'frames_only_teacher')
        command = command_for(options, panel, task['task_id'], 0, 16, name,
                              workers=args.workers, beta=beta)
        if name != 'production_driver':
            command[1] = str(args.driver_root.resolve() / 'scripts/search_play.py')
        if not frames:
            index = command.index('--frames')
            del command[index:index + 4]  # --frames PATH --frames-label LABEL
        print(name, json.dumps(command), flush=True)
        with (directory / f'{name}.log').open('a') as log:
            subprocess.run(command, env=environment, stdout=log, stderr=log, check=True)
        rows = [json.loads(line) for line in
                (directory / f'{name}_16/games.jsonl').read_text().splitlines()]
        print(name, 'wins', sum(r['won'] for r in rows), '/', len(rows), flush=True)


if __name__ == '__main__':
    main()
