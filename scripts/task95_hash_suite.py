#!/usr/bin/env python3
"""Check task95 hash determinism on the six actual A2 task definitions.

Run after the game gates have exited: this probe uses one simulator at a time.
The legacy L7 probe is retained separately. No policy or production defaults change.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'python')]

from pvz_env import TaskSpec
from pvz_event_env import EventWaitEnv
from pvz_tasks import profile_for_deck
from render_episode import load_training_task, training_task_kwargs
from task70_redeploy_gate import CELLS

RESOURCE = Path('/home/newbiexvwu/.cache/pvz-research-resources')
OUT = ROOT / 'artifacts/task95/state_hash_suite_build2.json'
ENGINES = {
    'old': ('deploy14-pvz-portable', '76ff36dc026f67641aefccd5f30f6cdae65eee14ee197dc24497413123a7874a'),
    'new': ('pvz-portable-build2', '83aed3bed2b74392b20367915d84908faf9384a986a115893baecdccf65f2029'),
}


def stable(multi: list[dict], same: list[dict]) -> bool:
    """Require complete samples and the same three points in both process modes."""
    if len(multi) != 8 or len(same) != 4:
        return False
    rows = multi + same
    if any(len(row.get('hashes', [])) != 3 or len(row.get('ticks', [])) != 3
           or not all(isinstance(value, str) and value for value in row['hashes'])
           for row in rows):
        return False
    return all(row == rows[0] for row in rows)


def reset_trajectory(env: EventWaitEnv, task: TaskSpec, deck: list[int]) -> dict:
    env.begin_demo()
    obs, _ = env.reset(deck=deck, task=task)
    result = {'hashes': [env.state_hash()], 'ticks': [obs['tick']]}
    for duration in (60, 600):
        obs, *_ = env.step({'type': 'wait', 'ticks': duration})
        result['hashes'].append(env.state_hash())
        result['ticks'].append(obs['tick'])
    return result


def case_spec(manifest: Path, task_id: str) -> tuple[TaskSpec, list[int]]:
    definition = load_training_task(manifest, task_id)
    kw = training_task_kwargs(definition, definition['seeds'][0])
    task = TaskSpec(level=kw['level'], seed=kw['seed'], playthrough=2,
                    profile=profile_for_deck(tuple(kw['deck']), kw['first_aid_owned']),
                    **{k: v for k, v in kw['task_extra'].items() if v is not None})
    return task, kw['deck']


def main() -> int:
    os.environ.update(PVZ_NO_POPUP='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    evidence = {'scope': 'six A2 tasks, first seed; reset and waits of 60 then 600 ticks',
                'engines': {}}
    for tag, (filename, expected_sha) in ENGINES.items():
        binary = ROOT / 'artifacts/task95/engine' / filename
        actual_sha = hashlib.sha256(binary.read_bytes()).hexdigest()
        if actual_sha != expected_sha:
            raise ValueError(f'{tag} archive SHA changed')
        report = {'executable': str(binary), 'sha256': actual_sha, 'cells': []}
        for cell, task_id, manifest in CELLS:
            task, deck = case_spec(manifest, task_id)
            multi = []
            for _ in range(8):
                with EventWaitEnv(RESOURCE, executable=binary) as env:
                    multi.append(reset_trajectory(env, task, deck))
            with EventWaitEnv(RESOURCE, executable=binary) as env:
                same = [reset_trajectory(env, task, deck) for _ in range(4)]
            item = {'cell': cell, 'task_id': task_id, 'deck': deck,
                    'environment_task': asdict(task), 'manifest': str(manifest),
                    'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
                    'multi_process': multi, 'same_process': same,
                    'multi_distinct': len({tuple(row['hashes']) for row in multi}),
                    'same_distinct': len({tuple(row['hashes']) for row in same}),
                    'stable': stable(multi, same)}
            report['cells'].append(item)
            print(tag, cell, 'multi distinct', item['multi_distinct'],
                  'same distinct', item['same_distinct'], 'stable', item['stable'], flush=True)
        evidence['engines'][tag] = report
        # Keep completed control/candidate evidence if a later case is interrupted.
        OUT.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + '\n')
    evidence['pass'] = all(cell['stable'] for cell in evidence['engines']['new']['cells'])
    OUT.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + '\n')
    print('SIX FAMILY HASH', 'PASS' if evidence['pass'] else 'FAIL', flush=True)
    return 0 if evidence['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
