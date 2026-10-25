"""Isolated task76 diagnostics. Never imported by production collection code."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
MAIN = Path('/home/newbiexvwu/PvZ-search')
OUT = ROOT / 'artifacts/task76'
PLAN = ROOT / 'MEMORY/docs/task76-roof-candidates-v5-plan.json'
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'python')]
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['PVZ_NO_POPUP'] = '1'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, document):
    path = Path(path)
    if not path.is_absolute() or not path.is_relative_to(ROOT):
        raise ValueError('task76 writes must stay in the absolute roofcand tree')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=1, allow_nan=False) + '\n')
    temporary.replace(path)


def freeze(source, destination):
    source, destination = Path(source), Path(destination)
    digest = sha(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha(destination) != digest:
            raise ValueError('frozen input differs: ' + str(destination))
    else:
        with source.open('rb') as src, destination.open('xb') as dst:
            while chunk := src.read(1 << 20):
                dst.write(chunk)
        destination.chmod(source.stat().st_mode & 0o777)
    if digest != sha(source) or digest != sha(destination):
        raise ValueError('input changed during freezing')
    return dict(source=str(source), path=str(destination), sha256=digest,
                bytes=destination.stat().st_size)


def prepare():
    if PLAN.exists():
        verify_plan()
        print('existing frozen plan verified; not refreezing the live student')
        return
    state = read('/home/newbiexvwu/pvz-logs/curriculum_state.json')
    sources = dict(engine=MAIN/'.claude/worktrees/cq/build4/pvz-portable',
                   student=Path(state['student']), value=MAIN/'artifacts/value_net/v5/model.pt',
                   static=MAIN/'artifacts/hard/value_branch1/model.pt')
    if (MAIN/'artifacts/value_net/v4/model.pt').exists():
        sources['v4_control'] = MAIN/'artifacts/value_net/v4/model.pt'
    inputs = {k: freeze(p, OUT/'frozen'/('pvz-portable' if k == 'engine' else k+'.pt'))
              for k, p in sources.items()}
    from make_roof_panel import ROOF, EVAL_START, EVAL_COUNT
    from zone_panel import task_document
    from pvz_tasks import deck_for_level, profile_for_deck
    from pvz_env import TaskSpec
    from pvz_event_env import EventWaitEnv
    tasks = [task_document(ROOF, level, deck_for_level(level, family='roof_r7'),
             f'roof_l{level}_eval_r7', list(range(EVAL_START, EVAL_START + EVAL_COUNT)))
             for level in range(41, 50)]
    checks = []
    resources = '/home/newbiexvwu/.cache/pvz-research-resources'
    with EventWaitEnv(resources, executable=inputs['engine']['path'], headless=True) as env:
        for task in tasks:
            deck = tuple(task['deck'])
            obs, _ = env.reset(deck=deck, task=TaskSpec(level=task['level'], seed=task['seeds'][0],
                                profile=profile_for_deck(deck), playthrough=2))
            options = env.deck_options(); warnings = env.deck_warnings(deck)['warnings']
            assert set(deck) <= set(options['available_plants']) and not warnings
            checks.append(dict(level=task['level'], warnings=warnings, deck=task['deck'],
                               starting_sun=obs['sun'], starting_pots=sum(p['type']==33 for p in obs['plants'])))
    manifest = OUT/'roof_panel.json'
    write(manifest, dict(schema_version=1, purpose='task76 copied registered r7 evaluation definitions', tasks=tasks))
    files = ['scripts/search_play.py', 'python/pvz_env.py', 'python/pvz_event_env.py',
             'python/pvz_rollout_policy.py', 'scripts/pvz_tasks.py', 'scripts/render_episode.py',
             'scripts/make_roof_panel.py', 'scripts/zone_panel.py']
    plan = dict(schema_version=1, task=76, inputs=inputs, manifest=str(manifest),
                manifest_sha256=sha(manifest), resource_dir=resources,
                student_round_at_freeze=state.get('student_round'),
                curriculum_round_at_freeze=state.get('round'),
                code_sha256={str(ROOT/p):sha(ROOT/p) for p in files},
                deck_checks=checks, workers=6,
                teacher=dict(per_card=1, student_candidates=2, samples=3, verify_samples=9,
                             horizon=2500, min_gap=300, rollout_sim_skip=61, dagger_beta=1.0, margin=.02),
                initial_levels=[42,43], seeds_per_level=32,
                expansion='Every roof level in L42-L49 with task74 teacher win rate <50%; request authoritative task74 result; do not infer from v4.',
                arms=['production','all_plants','all_legal'],
                contrasts=dict(primary='all_plants vs production isolates omission of legal planting cells',
                               supplementary='all_legal additionally includes legal shovels; report action-space effects separately'),
                coverage='Record every game decision before action, every legal plant and shovel, production candidates, all halving rounds and verification; packet slots mapped to actual plant types.',
                terminal_probe=dict(futures_per_action=8, actions='Every legal plant and shovel plus canonical wait at each frozen probe root',
                    continuation='Frozen greedy student C++ ROLLOUT until_tick=0, matching task72b; explicitly distinct from task46 search-teacher terminal continuation',
                    selection='After paired panels finish, first two losing production seeds per required level; four time-spaced scheduled legal roots (0/one-third/two-thirds/latest) per loss; terminal outcomes never used to choose roots.',
                    leaf_futures=9, horizons=[2500,4000,6000], v4_control='Same roots/futures/h2500 where v4 exists; also report historical task46 53.9% as protocol-different reference.'),
                production_changes=False, protected_tree_writes=False,
                historical_v4_reference='MEMORY/docs/task46-roof-zone.md: won 53.9%, cluster95%47.1-60.9; different roots, model, two futures and teacher continuation')
    write(PLAN, plan)
    print(json.dumps(dict(frozen_student=sources['student'].as_posix(), student_round=plan['student_round_at_freeze'],
                         manifest=str(manifest), inputs=inputs, deck_checks=checks), ensure_ascii=False, indent=1))


def verify_plan():
    plan = read(PLAN)
    for name, item in plan['inputs'].items():
        if sha(item['path']) != item['sha256']:
            raise ValueError('frozen input changed: ' + name)
    if sha(plan['manifest']) != plan['manifest_sha256']:
        raise ValueError('frozen manifest changed')
    for path, digest in plan['code_sha256'].items():
        if sha(path) != digest:
            raise ValueError('production algorithm changed inside diagnostic checkout: '+path)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare','verify'])
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare()
    else:
        verify_plan(); print('frozen inputs, manifest and production source identities verified')


if __name__ == '__main__':
    main()
