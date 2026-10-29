"""Prepare task162 jobs from task157's transferred corpus, without enqueuing/training.

Immutable numeric archives are hardlinked into independent train/heldout views.
Missing training packs are built once per input version; all architecture variants
reuse them. No archive replay or data audit is run by this preparation command.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from pvz_recipe import write_recipe
from pvz_training_pack import archive_fingerprint, pack_options
from pack_training_rows import pack


def write_json(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def prepare(source, state, matrix, python=sys.executable):
    source, state = Path(source).resolve(), Path(state).resolve()
    manifest = json.loads((source/'manifest.json').read_text())
    state.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(source/'manifest.json',state/'source_manifest.json')
    labels = {'train':set(),'heldout':set()}
    counts = Counter()
    for archive in manifest['archives']:
        split = archive['split']
        group = 'train' if split=='train' else 'heldout'
        counts[split]+=1
        relative = Path(archive['path']).relative_to('frames')
        labels[group].add(relative.parent.name)
        destination=state/'corpus'/group/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        for name in ('frames.npz','meta.json'):
            os.link((source/archive['path']).parent/name,destination.parent/name)
    packs = {}
    for obs in sorted({s['obs'] for s in matrix['structures']}):
        for group in ('train','heldout'):
            data=state/'corpus'/group
            options=pack_options(labels[group], wait_keep=matrix['wait_keep'] if group=='train' else 1,
                obs=obs,packet_head='identity',sources=archive_fingerprint(data,labels[group]),
                token_input='obs_v3' if obs=='v3' else 'compact')
            packs[obs,group]=pack(data,options,precompute_workers=1)
    jobs=[]
    for structure in matrix['structures']:
        for seed in matrix['seeds']:
            name=f"task163_{structure['name']}_s{seed}"
            out=state/'runs'/name
            obs=structure['obs']
            recipe=dict(schema_version=1,kind='training',data=str(state/'corpus/train'),
                labels=sorted(labels['train']),out=str(out),device='cuda',threads=1,precompute_workers=1,
                seed=seed,epochs=matrix['epochs'],wait_keep=matrix['wait_keep'],packet_head='identity',
                width=structure['width'],arch='student_v3' if obs=='v3' else 'cnn',obs=obs,
                token_input='obs_v3' if obs=='v3' else 'compact',network=structure['network'],
                pool=structure.get('pool','mean'),row_pool=structure.get('row_pool',False),
                batch_size=structure['batch_size'],validation_batch_size=32 if obs=='v3' else 256,
                validation=matrix['internal_validation'],lr_schedule='cosine',init=None,
                provenance=dict(source_revision=manifest['input_revision'],notes='task157 fixed manifest train only; seen/unseen/terrain samples exclusively CPU heldout; no GPU training launched by preparation'))
            path=state/'recipes'/(name+'.json')
            path.parent.mkdir(parents=True,exist_ok=True)
            write_recipe(path,recipe)
            job=dict(id=name,recipe=str(path),out=str(out),host_gib=matrix['host_gib'],gpu_gib=matrix['gpu_gib'],
                slice_seconds=matrix['slice_seconds'],
                command=[python,str(ROOT/'scripts/train_rollout_policy.py'),'--recipe',str(path),
                    '--pack',str(packs[obs,'train']),'--merge-dir',str(state/'merges'/obs/'train'),
                    '--checkpoint-seconds',str(matrix['checkpoint_seconds']),'--skip-validation'],
                evaluate=[python,str(ROOT/'scripts/eval_window_policy.py'),'--recipe',str(path),
                    '--pack',str(packs[obs,'heldout']),'--merge-dir',str(state/'merges'/obs/'heldout'),
                    '--out',str(out),'--eval-data',str(state/'corpus/heldout'),
                    '--manifest',str(state/'source_manifest.json'),'--batch-size','32' if obs=='v3' else '256'])
            write_json(state/'jobs'/(name+'.json'),job)
            jobs.append(name)
    result=dict(source=str(source),input_revision=manifest['input_revision'],split_games=dict(counts),jobs=jobs,
                queue=str(state/'jobs'),enqueued=False,
                packs={f'{obs}/{group}':str(path) for (obs,group),path in packs.items()},
                training_split='manifest train; common 10% seed holdout inside training corpus',
                external_evaluation='all manifest seen/unseen/terrain_sample rows, without wait sampling')
    write_json(state/'prepared.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True,help='downloaded prefix containing manifest.json and frames/')
    parser.add_argument('--state',type=Path,required=True,help='new isolated output directory (never the production queue)')
    parser.add_argument('--matrix',type=Path,default=ROOT/'configs/task163_student_v3_jobs.json')
    args=parser.parse_args()
    print(json.dumps(prepare(args.source,args.state,json.loads(args.matrix.read_text())),ensure_ascii=False))


if __name__=='__main__': main()
