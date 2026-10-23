"""task46: reproducible roof deck screening and bounded, resumable search_play runs."""
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
MAIN = Path('/home/newbiexvwu/PvZ-search')
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]
from curriculum_queue import command_for
from perturb_family import candidate_decks
from pvz_env import PvZEnv, TaskSpec
from pvz_tasks import deck_for_level, profile_for_deck, terrain_covered

OUT = ROOT / 'artifacts/task46'


def settings():
    return SimpleNamespace(python=Path('/home/newbiexvwu/.venvs/ml/bin/python3'),
        rollout_policy=MAIN / 'artifacts/student/dw4i_s1/model.pt',
        static_value=MAIN / 'artifacts/hard/value_branch1/model.pt',
        value=MAIN / 'artifacts/value_net/v4/model.pt', executable=ROOT / 'build/pvz-portable',
        workers=1, resource_dir=Path('/home/newbiexvwu/.cache/pvz-research-resources'),
        frames=OUT / 'frames', out=OUT / 'runs')


def prepare():
    inventory = json.loads((OUT / 'inventory.json').read_text())['levels']
    pool = sorted(set.intersection(*(set(r['pool']['crazy_dave_pool']) for r in inventory)))
    options = inventory[0]['options']
    flags = dict(options['not_recommended'])
    pool = [p for p in pool if 0 <= p < 40 and flags[p] == 0]
    base = deck_for_level(41)
    candidates = list(candidate_decks(base, pool))
    random.Random(46).shuffle(candidates)
    selected = []
    args = settings()
    with PvZEnv(args.resource_dir, executable=args.executable) as env:
        for deck in candidates:
            if not terrain_covered(41, deck):
                continue
            admitted = True
            for level in range(41, 50):
                env.reset(deck=deck_for_level(level), task=TaskSpec(level=level, seed=44000000,
                    profile=profile_for_deck(deck_for_level(level))))
                if env.deck_warnings(deck)['warnings']:
                    admitted = False
                    break
            if admitted:
                selected.append(deck)
            if len(selected) == 16:
                break
    if len(selected) != 16:
        raise RuntimeError('not enough admissible neighboring decks')
    tasks = []
    for index, deck in enumerate([list(base)] + selected):
        tasks.append(dict(task_id=f'roof_pilot_c{index:02d}_l41', terrain='roof', level=41,
            deck=deck, zombie_count_multiplier=1.0, wave_cap=None, sun_start=50,
            preplanted=[], playthrough=2, seeds=[44000000, 44000001]))
    document = dict(schema_version=1, purpose='task46 deck selection, separate from evaluation',
        selection=dict(method='Random(46) shuffle of exact 1/2 swaps from engine common candidate pool; '
                              'first 16 terrain-covered decks with zero engine warnings on all nine levels',
                       pool=pool, candidates=len(candidates),
                       ranking='teacher wins, mean final_wave/wave_count, then candidate id'), tasks=tasks)
    path = OUT / 'pilot_manifest.json'
    path.write_text(json.dumps(document, indent=1) + '\n')
    print(json.dumps(document, ensure_ascii=False, indent=1))


def commands(manifest, count, arms):
    args = settings()
    document = json.loads(manifest.read_text())
    tasks = document['tasks']
    jobs = []
    for task in tasks:
        for arm in arms:
            label = task['task_id'] + '_' + arm
            command = command_for(args, manifest, task['task_id'], 0, count, label)
            # Evaluation needs final-board evidence, not training frame archives.
            index = command.index('--frames')
            del command[index:index + 4]
            command[command.index('--dagger-beta') + 1] = '1' if arm == 'teacher' else '0'
            command += ['--arms', 'fast_search' if arm == 'teacher' else 'fast']
            jobs.append(dict(label=label, command=command))
    return jobs


def prepare_followup():
    """Apply the frozen conditional plan only to completed low-win regular cells."""
    plan = json.loads((ROOT / 'MEMORY/docs/task46-roof-followup-plan.json').read_text())
    formal = json.loads((OUT / 'formal_manifest.json').read_text())['tasks']
    diagnostics, half, checks = [], [], []
    args = settings()
    with PvZEnv(args.resource_dir, executable=args.executable) as env:
        for level in range(41, 45):
            base = next(t for t in formal if t['task_id'] == f'roof_l{level}_eval_r7')
            path = OUT / f'runs/{base["task_id"]}_teacher_16/games.jsonl'
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            if (len(rows) != 16 or {row['seed'] for row in rows} != set(base['seeds'][:16])
                    or not all(row['terminal'] for row in rows)):
                raise ValueError(f'L{level}: follow-up needs the complete formal teacher cell')
            if sum(row['won'] for row in rows) / 16 >= 0.2:
                continue
            diagnostic = copy.deepcopy(base)
            diagnostic.update(task_id=f'roof_l{level}_eval_c05diag',
                              deck=plan['deck_recheck']['deck'], seeds=plan['deck_recheck']['seeds'])
            env.reset(deck=tuple(diagnostic['deck']), task=TaskSpec(level=level,
                seed=diagnostic['seeds'][0], profile=profile_for_deck(tuple(diagnostic['deck']))))
            warnings = env.deck_warnings(diagnostic['deck'])['warnings']
            available = env.deck_options()['available_plants']
            if warnings or not set(diagnostic['deck']) <= set(available):
                raise ValueError(f'L{level}: c05 fails actual deck checks')
            checks.append(dict(level=level, warnings=warnings, all_cards_unlocked=True))
            diagnostics.append(diagnostic)
            lowered = copy.deepcopy(base)
            lowered.update(task_id=base['task_id'] + '_m050', zombie_count_multiplier=0.5,
                           seeds=plan['half_multiplier']['seeds'])
            half.append(lowered)
    for name, tasks in [('deck_recheck_manifest', diagnostics), ('half_manifest', half)]:
        (OUT / f'{name}.json').write_text(json.dumps(dict(schema_version=1,
            preregistration='MEMORY/docs/task46-roof-followup-plan.json', tasks=tasks), indent=1) + '\n')
    (OUT / 'followup_deck_checks.json').write_text(json.dumps(checks, indent=1) + '\n')
    print(f'conditional follow-up levels: {[check["level"] for check in checks]}', flush=True)


def run_job(job):
    label = job['label']
    env = dict(os.environ, PVZ_NO_POPUP='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1')
    with (OUT / f'{label}.log').open('a') as log:
        result = subprocess.run(job['command'], stdout=log, stderr=log, env=env, cwd=ROOT)
    print(f'{label}: exit {result.returncode}', flush=True)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'prepare-followup', 'run'))
    parser.add_argument('--manifest', type=Path, default=OUT / 'pilot_manifest.json')
    parser.add_argument('--count', type=int, default=2)
    parser.add_argument('--arms', default='teacher')
    parser.add_argument('--workers', type=int, choices=range(1, 5), default=4)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.mode == 'prepare':
        prepare()
        return
    if args.mode == 'prepare-followup':
        prepare_followup()
        return
    jobs = commands(args.manifest, args.count, args.arms.split(','))
    (OUT / f'{args.manifest.stem}_commands.json').write_text(json.dumps(jobs, indent=1) + '\n')
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        codes = list(executor.map(run_job, jobs))
    if any(codes):
        raise SystemExit('some runs failed; inspect per-run logs and resume the same manifest')


if __name__ == '__main__':
    main()
