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


def enqueue(state, recipe_path, revision, code_root, python, manifest):
    corpus = json.loads((state / 'corpus/corpus.json').read_text())
    base = load_recipe(recipe_path, 'training')
    if base['obs'] != 'v2timer' or corpus['rows'] != 23631235:
        raise ValueError('the decision study requires the frozen r42 v2timer corpus')
    split = state / 'decision-data/split'
    tokens = state / 'decision-data/tokens'
    mask = split / 'unseen.npy'
    common = dict(revision=revision, code_root=str(code_root), host_gib=5, gpu_gib=0, slice_seconds=900,
                  kind='prepare', recipe=str(recipe_path))
    jobs = []
    prep_script = str(code_root / 'scripts/prepare_decision_data.py')
    jobs.append(dict(common, id='decision_split', priority=0, out=str(split), command=[python, prep_script, 'split',
        '--pack', corpus['pack'], '--data', base['data'], '--merge-dir', corpus['merge_dir'], '--out', str(split)]))
    jobs.append(dict(common, id='decision_tokens', priority=60, out=str(tokens), depends_on=[str(split/'ready.json')],
        command=[python, prep_script, 'tokens', '--sources', str(split), '--out', str(tokens)]))
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
            if path.exists():
                if load_recipe(path) != variant:
                    raise ValueError(f'cannot replace frozen recipe {path}')
            else:
                write_recipe(path, variant)
            flags = ['--pack', corpus['pack'], '--merge-dir', corpus['merge_dir'], '--deck-holdout', str(mask)]
            dependencies = [str(split/'ready.json')]
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
    # Original two examples have no exclusion split and are a different cohort.
    for name in ('r42_cnn64', 'r42_cnn128'):
        old = state / 'queue' / (name + '.json')
        if old.exists():
            document = json.loads(old.read_text())
            document.update(enabled=False, superseded_by=f'{name}_s1', reason='common whole-deck holdout and pinned code cohort')
            write_json(old, document)
    for job in jobs:
        path = state / 'queue' / (job['id'] + '.json')
        if path.exists() and json.loads(path.read_text()) != job:
            raise ValueError(f'cannot replace queued pinned job {path}')
        write_json(path, job)
    summary = dict(study=manifest['study'], revision=revision, token_source_revision=manifest['token_source_revision'],
        code_root=str(code_root), corpus=corpus, jobs=[{k: j[k] for k in ('id', 'priority', 'revision', 'out')} for j in jobs],
        handoff=str(state/'mac-handoff'), training_jobs=len(manifest['arms'])*len(manifest['seeds']))
    write_json(state/'decision-study.json', summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--state', type=Path, required=True)
    p.add_argument('--recipe', type=Path, required=True)
    p.add_argument('--revision', required=True, help='committed full code revision to execute for every attempt')
    args = p.parse_args()
    revision = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', args.revision+'^{commit}'], text=True).strip()
    code_root = args.state / 'code' / revision
    if not code_root.exists():
        code_root.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', '-C', str(ROOT), 'worktree', 'add', '--detach', str(code_root), revision], check=True)
    manifest = json.loads((code_root/'configs/decision_r42.json').read_text())
    summary = enqueue(args.state.resolve(), args.recipe.resolve(), revision, code_root, sys.executable, manifest)
    print(json.dumps(dict(training_jobs=summary['training_jobs'], revision=revision, handoff=summary['handoff'])))


if __name__ == '__main__':
    main()
