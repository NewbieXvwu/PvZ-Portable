"""Enqueue pinned task162 controls using the already frozen r42 corpus.

No data loading/replay/training. The expensive public-token construction is a resumable
preparation job after the CNN controls, not a synchronous admission precheck.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
from pvz_recipe import load_recipe, write_recipe
from gpu_window_runner import write_json


def enqueue(state, recipe_path, revision, code_root, python, manifest, replace=False):
    corpus = json.loads((state / 'corpus/corpus.json').read_text())
    base = load_recipe(recipe_path, 'training')
    if base['obs'] != 'v2timer' or corpus['rows'] != 23631235:
        raise ValueError('the decision study requires the frozen r42 v2timer corpus')
    split = state / 'decision-data/split'
    tokens = state / 'decision-data/tokens'
    mask = split / 'unseen.npy'
    focused = 'target_training_rows' in manifest
    desktop = manifest.get('desktop_arms', [arm['name'] for arm in manifest['arms']])
    budget_path = state / 'decision-data/throughput/ready.json'
    common = dict(revision=revision, code_root=str(code_root), host_gib=5, gpu_gib=0, slice_seconds=900,
                  kind='prepare', recipe=str(recipe_path))
    jobs = []
    prep_script = str(code_root / 'scripts/prepare_decision_data.py')
    jobs.append(dict(common, id='decision_split', priority=0, out=str(split), command=[python, prep_script, 'split',
        '--pack', corpus['pack'], '--data', base['data'], '--merge-dir', corpus['merge_dir'], '--out', str(split)]))
    jobs.append(dict(common, id='decision_tokens', priority=5 if focused else 60, out=str(tokens),
        depends_on=[str(split/'ready.json')] + ([str(budget_path)] if focused else []),
        command=[python, prep_script, 'tokens', '--sources', str(split), '--out', str(tokens)] +
                (['--budget', str(budget_path)] if focused else [])))
    recipes = state / 'decision-recipes'; recipes.mkdir(parents=True, exist_ok=True)
    for arm in manifest['arms']:
        for seed in manifest['seeds']:
            name = f'r42_{arm["name"]}_s{seed}'
            out = state / 'runs' / name
            path = recipes / (name + '.json')
            variant = dict(base, out=str(out), seed=seed, init=None, source_row_report=False,
                width=arm['width'], arch=arm['arch'], micro_batch_size=arm['micro_batch_size'], validation_batch_size=32,
                provenance=dict(source_path=str(recipe_path), source_revision=revision,
                    notes=f'task162 decision control; exact r42 rows; common whole-deck exclusion; token encoder {manifest["token_source_revision"]}'))
            if focused:
                variant.update(epochs=manifest['epochs'], batch_size=manifest['batch_size'])
                variant['provenance']['notes'] += '; two-day seed1 cohort; previous optimizer runs retired'
            if path.exists() and not replace:
                if load_recipe(path) != variant:
                    raise ValueError(f'cannot replace frozen recipe {path}')
            else:
                write_recipe(path, variant)
            flags = ['--pack', corpus['pack'], '--merge-dir', corpus['merge_dir'], '--deck-holdout', str(mask)]
            dependencies = [str(split/'ready.json')]
            if focused:
                dependencies += [str(budget_path)]
            if arm.get('tokens'):
                flags += ['--tokens', str(tokens)]
                dependencies += [str(tokens/'ready.json')]
            job = dict(id=name, priority=arm['priority'], revision=revision, code_root=str(code_root),
                arch=arm['arch'], seed=seed, recipe=str(path), out=str(out), host_gib=5, gpu_gib=4,
                depends_on=dependencies, slice_seconds=1800,
                deck_split=str(split/'ready.json'),
                command=[python, str(code_root/'scripts/train_rollout_policy.py'), '--recipe', str(path),
                         *flags, '--checkpoint-seconds', '30', '--skip-validation'],
                evaluate=[python, str(code_root/'scripts/eval_window_policy.py'), '--recipe', str(path),
                          '--out', str(out), '--batch-size', str(min(32, arm['micro_batch_size'])), *flags])
            jobs.append(job)
            if focused:
                budget_flags = ['--training-budget', str(budget_path)]
                job['command'] += ['--resident-training-rows', *budget_flags, '--budget-arm', arm['name']]
                job['evaluate'] += [*budget_flags, '--device', 'cuda', '--preview-rows', str(manifest['preview_rows'])]
                job['enabled'] = arm['name'] in desktop
                job['defer_evaluation'] = arm['name'] != 'cnn_entity' and len(desktop) > 1
                job['after_jobs'] = []
                if arm['name'] == 'cnn_entity' and len(desktop) > 1:
                    job['after_jobs'] = [f'r42_{name}_s1' for name in ('cnn64', 'v3_small')]
                    job['evaluation_peers'] = list(job['after_jobs'])
                    for peer in ('cnn64', 'v3_small'):
                        job['evaluate'] += ['--peer-out', str(state/'runs'/f'r42_{peer}_s1')]
    if focused:
        jobs.append(dict(common, id='decision_cleanup', priority=300, out=str(state/'decision-data/cleanup'),
            enabled=len(desktop) == len(manifest['arms']),
            after_jobs=[f'r42_{arm["name"]}_s1' for arm in manifest['arms']],
            command=[python, str(code_root/'scripts/cleanup_decision_cache.py'), '--state', str(state),
                     '--out', str(state/'decision-data/cleanup')]))
    # Original two examples have no exclusion split and are a different cohort.
    for name in ('r42_cnn64', 'r42_cnn128'):
        old = state / 'queue' / (name + '.json')
        if old.exists():
            document = json.loads(old.read_text())
            document.update(enabled=False, superseded_by=f'{name}_s1', reason='common whole-deck holdout and pinned code cohort')
            write_json(old, document)
    for job in jobs:
        path = state / 'queue' / (job['id'] + '.json')
        if path.exists() and not replace and json.loads(path.read_text()) != job:
            raise ValueError(f'cannot replace queued pinned job {path}')
        write_json(path, job)
    summary = dict(study=manifest['study'], revision=revision, token_source_revision=manifest['token_source_revision'],
        code_root=str(code_root), corpus=corpus, jobs=[{k: j[k] for k in ('id', 'priority', 'revision', 'out')} for j in jobs],
        handoff=str(state/'mac-handoff'), training_jobs=len(manifest['arms'])*len(manifest['seeds']))
    write_json(state/'decision-study.json', summary)
    return summary


def replan_two_day(state, recipe_path, revision, code_root, python, manifest):
    """Replace existing research queue/configuration; no new manifest or data scan."""
    budget_path = state/'decision-data/throughput/ready.json'
    old = json.loads(budget_path.read_text())
    report = json.loads((budget_path.parent/'report.json').read_text())
    fraction = manifest['target_training_rows'] / old['measured_training_rows']
    rates = {arm['name']: report['rates'][arm['name']]['rows_per_second'] for arm in manifest['arms']}
    # The measured cold end-to-end rate, rather than hot probe throughput, drives the deadline.
    window_hours = (len(rates)*manifest['target_training_rows']/manifest['measured_end_to_end_rows_per_second']/3600 + 2) * 1.25
    budget = dict(study=manifest['study'], epochs=manifest['epochs'], train_row_fraction=fraction,
        measured_training_rows=old['measured_training_rows'], target_training_rows=manifest['target_training_rows'],
        batch_size=manifest['batch_size'], micro_batches={arm['name']: arm['micro_batch_size'] for arm in manifest['arms']},
        sidecar_max_bytes=old['sidecar_max_bytes'], estimated_sidecar_bytes=(manifest['target_training_rows']+manifest['preview_rows'])*old['sparse_bytes_per_row_with_margin'],
        revision=revision, seed1_only=True, planning_window_fraction=manifest['planning_window_fraction'],
        estimated_window_hours=window_hours, estimated_preview_calendar_hours=window_hours/manifest['planning_window_fraction'],
        preview_rows=manifest['preview_rows'], heldout='unchanged; shared uniform preview then complete heldout',
        desktop_arms=manifest.get('desktop_arms', list(rates)),
        probe_rows_per_second=rates,
        sparse_bytes_per_row_with_margin=old['sparse_bytes_per_row_with_margin'],
        reason='three seed1 arms, one epoch and common global-row hash; deadline includes one hour tokens, half hour preview, half hour restart overhead, plus 25% margin')
    # Disable the old cohorts first, including seed2 and task163, so none can consume this deadline.
    for path in (state/'queue').glob('*.json'):
        if path.name == 'status.json':
            continue
        job = json.loads(path.read_text())
        job.update(enabled=False, reason='two-day three-arm seed1 cohort; other arms/seeds deferred')
        write_json(path, job)
    outputs = [state/'runs'/f'r42_{arm["name"]}_s1' for arm in manifest['arms']]
    outputs.append(state/'decision-data/tokens')
    for out in outputs:
        if out.exists() and old.get('study') != manifest['study']:
            out.rename(out.with_name(out.name+'_before_2day'))
    write_json(budget_path, budget)
    summary = enqueue(state, recipe_path, revision, code_root, python, manifest, replace=True)
    summary['budget'] = budget
    write_json(state/'decision-study.json', summary)
    return summary


def enqueue_card_attributes(state, recipe_path, revision, code_root, python):
    """task192: one opt-in arm; reuse the exact CNN64 rows, budget and prefix.

    Only the two required queue/recipe objects are written. Existing cohorts,
    manifests, masks, production configuration and the runner are untouched.
    """
    control = json.loads((state / 'queue/r42_cnn64_s1.json').read_text())
    name = 'cnn64_cardattr_s1'
    path = state / 'decision-recipes' / (name + '.json')
    out = state / 'runs' / name
    variant = load_recipe(recipe_path, 'training')
    variant.update(card_attributes=True, out=str(out))
    variant['provenance'] = dict(variant['provenance'], source_revision=revision)
    write_recipe(path, variant)
    job = dict(control, id=name, recipe=str(path), out=str(out), revision=revision, code_root=str(code_root),
               priority=control['priority'] + 1, enabled=True, after_jobs=['r42_cnn64_s1'], defer_evaluation=False)
    replacements = {control['recipe']: str(path), control['out']: str(out)}
    for field in ('command', 'evaluate'):
        command = [replacements.get(arg, arg) for arg in control[field]]
        command[0] = python
        command[1] = str(code_root / 'scripts' / Path(command[1]).name)
        job[field] = command
    job['evaluate'] += ['--preview-only', '--peer-out', control['out'], '--peers-inline']
    job.pop('evaluation_peers', None)
    write_json(state / 'queue' / (name + '.json'), job)
    return dict(job=name, revision=revision, after_jobs=job['after_jobs'], out=str(out))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--state', type=Path, required=True)
    p.add_argument('--recipe', type=Path, required=True)
    p.add_argument('--revision', required=True, help='committed full code revision to execute for every attempt')
    p.add_argument('--card-attributes', action='store_true', help='enqueue only task192 from the existing r42_cnn64_s1 recipe')
    p.add_argument('--replan-two-day', action='store_true', help='replace existing study jobs; stop the research service before invoking')
    args = p.parse_args()
    revision = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', args.revision+'^{commit}'], text=True).strip()
    code_root = args.state / 'code' / revision
    if not code_root.exists():
        code_root.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', '-C', str(ROOT), 'worktree', 'add', '--detach', str(code_root), revision], check=True)
    if args.card_attributes:
        print(json.dumps(enqueue_card_attributes(args.state.resolve(), args.recipe.resolve(),
                                                revision, code_root, sys.executable)))
        return
    manifest = json.loads((code_root/'configs/decision_r42.json').read_text())
    operation = replan_two_day if args.replan_two_day else enqueue
    summary = operation(args.state.resolve(), args.recipe.resolve(), revision, code_root, sys.executable, manifest)
    print(json.dumps(dict(training_jobs=summary['training_jobs'], revision=revision, handoff=summary['handoff'])))


if __name__ == '__main__':
    main()
