"""Separate task78 formal evaluation; never alter the frozen diagnostic runtime.

Stages: tasks (register fresh identities), freeze, v5, v6, report.
No stage trains a model. freeze/run require the original-corpus candidate.
"""
from __future__ import annotations

from copy import deepcopy
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'python')]
import task78_ab as shared
from task78_data import sha

OUT=ROOT/'artifacts/task78/formal_ab'
OLD=ROOT/'artifacts/task78/ab'
REGISTRATION=ROOT/'MEMORY/docs/task78-formal-ab-plan.json'
CANDIDATE=ROOT/'artifacts/task78/formal_v6/model.pt'
OFFSET=512
HYPER=dict(lr=1e-4,weight_decay=1e-4,epochs=24,steps=256,batch_size=512,
           seed=0,patience=4,min_improvement=1e-4)


def identities(panel):
    return [s for cell in panel for s in (cell.get('future_seeds') or cell['seeds'])]


def declared_overlap(checkpoint,selected,name):
    if 'trained_on' in checkpoint:
        return sorted(set(selected)&set(checkpoint['trained_on']))
    if name=='student' and isinstance(checkpoint.get('labels'),str) and checkpoint['labels']:
        return None  # Label names alone do not prove a complete historical seed set.
    raise ValueError(f'{name}: required training identity metadata is missing')


def shifted_panel(panel):
    """Only change evaluation identities; keep tasks, decks and takeovers intact."""
    result=deepcopy(panel)
    for cell in result:
        key='future_seeds' if 'future_seeds' in cell else 'seeds'
        cell[key]=[s+OFFSET for s in cell[key]]
        lower=78100000 if key=='future_seeds' else 78000000
        if any(not lower<=s<lower+2000 for s in cell[key]):
            raise ValueError('formal identities exceed the original reserved blocks')
    fresh=identities(result)
    if len(set(fresh))!=len(fresh) or set(fresh)&set(identities(panel)):
        raise ValueError('formal identities overlap or duplicate the diagnostic panel')
    return result


def write_once(path,document):
    if path.exists() and json.loads(path.read_text())!=document:
        raise ValueError(f'frozen formal input changed: {path}')
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():path.write_text(json.dumps(document,indent=1)+'\n')


def prepare():
    """Register only a panel and future model location, without starting games."""
    if REGISTRATION.exists():
        existing=json.loads(REGISTRATION.read_text());frozen_panel()
        bindings={OLD/'tasks.json':existing['parent_tasks_sha256'],
                  OLD/'runtime.json':existing['parent_runtime_sha256'],
                  ROOT/'artifacts/task78/export.json':existing['new_export_sha256'],
                  ROOT/'MEMORY/docs/task78-v5-corpus-requirements.json':existing['original_corpus_requirements_sha256'],
                  ROOT/'scripts/task78_train.py':existing['training_source_sha256'],
                  OUT/'registration_seed_audit/seed_audit_summary.json':existing['broad_seed_audit_summary_sha256']}
        if any(sha(path)!=digest for path,digest in bindings.items()):
            raise ValueError('registered formal input changed; preserve it before amending')
        return existing
    import torch
    from task78_train import new_sources,MIX
    from task78_seed_audit import main as seed_audit
    previous=shared.frozen_panel(OLD)
    panel=shifted_panel(previous)
    if len(panel)!=15 or sum(c['pairs'] for c in panel)!=512:
        raise ValueError('the full fifteen-cell, 512-pair design is required')
    supplement=json.loads((ROOT/'MEMORY/docs/task78-demo-start-plan.json').read_text())
    supplement_ids={s for c in supplement['cells'] for s in c['future_seeds']}
    if set(identities(panel))&supplement_ids:
        raise ValueError('formal identities overlap the supplemental diagnostic')
    tasks={}
    for cell in panel:
        tasks.setdefault(cell['task_id'],{k:v for k,v in cell.items() if k not in
            ('cell_id','terrain','pairs','demo_decision','future_seeds')})
    document=dict(tasks=list(tasks.values()),cells=panel,
                  reserved=[[78000000,78002000],[78100000,78102000]])
    write_once(OUT/'tasks.json',document)
    sources,provenance=new_sources(ROOT/'artifacts/task78/data')
    requirements_path=ROOT/'MEMORY/docs/task78-v5-corpus-requirements.json'
    requirements=json.loads(requirements_path.read_text())
    splits={f'old_{name}':dict(fit=counts['fit'],val=counts['validation'])
            for name,counts in requirements['required_splits'].items()}
    splits.update({name:dict(fit=len(src.fit),val=len(src.val)) for name,src in sources.items()})
    template=json.loads((OLD/'runtime.json').read_text())
    overlaps={}
    selected=set(identities(panel))
    for name,model in template['models'].items():
        if sha(model['path'])!=model['sha256']:raise ValueError('diagnostic model identity changed')
        cp=torch.load(model['path'],map_location='cpu',weights_only=False,mmap=True)
        overlaps['diagnostic_v6' if name=='v6' else name]=declared_overlap(cp,selected,name)
    # The preceding broad seed audit covered both reserved blocks; check current
    # task78 source identities and every raw label future as well.
    source_path=ROOT/'artifacts/task78/sources.json'
    original=json.loads(source_path.read_text())
    used={v for a in original['archives'] for v in a['world'].values() if v is not None}
    used.update(((original.get('seed_base',78200000)+i*100+j)*1000)%2**32
                for i,a in enumerate(original['archives']) for j in range(len(a['roots'])))
    overlaps['new_region_worlds_and_label_futures']=sorted(selected&used)
    if any(overlaps.values()):raise ValueError('formal evaluation overlaps known training identities')
    seed_audit(OUT/'registration_seed_audit')
    audit_summary=OUT/'registration_seed_audit/seed_audit_summary.json'
    registration=dict(scope='formal_original_corpus_paired_panel',tasks_sha256=sha(OUT/'tasks.json'),
        parent_tasks_sha256=sha(OLD/'tasks.json'),parent_runtime_sha256=sha(OLD/'runtime.json'),
        candidate=str(CANDIDATE),training_log=str(CANDIDATE.with_name('train_log.json')),
        new_export_sha256=provenance['sha256'],mix=MIX,hyper=HYPER,source_splits=splits,
        original_corpus_requirements_sha256=sha(requirements_path),identity_offset=OFFSET,
        training_source_sha256=sha(ROOT/'scripts/task78_train.py'),
        cells=[dict(cell_id=c['cell_id'],pairs=c['pairs'],seed_role='future' if 'future_seeds' in c else 'environment',
                    first=(c.get('future_seeds') or c['seeds'])[0],last=(c.get('future_seeds') or c['seeds'])[-1]) for c in panel],
        pairs=512,model_training_identity_overlaps=overlaps,
        model_seed_metadata_missing=[name for name,value in overlaps.items() if value is None],
        broad_seed_audit_summary_sha256=sha(audit_summary),
        historical_audit='Local-only broad audit at registration; runtime reuses this snapshot and checks actual checkpoint training identities. Undisclosed remote experiments are not covered.',
        training_started=False,engine_processes_started=0,
        independent_truth='Existing frozen-v5 roof labels may be rescored with the formal candidate; no diagnostic-model scores are substituted.')
    write_once(REGISTRATION,registration)
    return registration


def frozen_panel():
    registration=json.loads(REGISTRATION.read_text())
    if sha(OUT/'tasks.json')!=registration['tasks_sha256']:
        raise ValueError('formal panel differs from its committed registration')
    return json.loads((OUT/'tasks.json').read_text())['cells']


def validate_training(checkpoint,training,registration):
    """An independent directory alone never upgrades a diagnostic checkpoint."""
    if (checkpoint.get('preregistered_corpus') is not True or
            checkpoint.get('mix')!=registration['mix'] or
            checkpoint.get('dropped_sources')!=[] or training.get('complete') is not True or
            training.get('mix')!=registration['mix'] or training.get('preregistered_corpus') is not True):
        raise ValueError('completed original-corpus training is required')
    original=checkpoint.get('input_provenance',{}).get('original',{})
    if original.get('preregistered_corpus',True) is not True or not original.get('descriptor') or not original.get('sha256'):
        raise ValueError('original corpus descriptor receipt is required')
    if checkpoint.get('input_provenance',{}).get('new',{}).get('sha256')!=registration['new_export_sha256']:
        raise ValueError('formal candidate must use the complete corrected 192-task export')
    for key,value in registration['hyper'].items():
        if checkpoint.get('hyper',{}).get(key)!=value:
            raise ValueError(f'formal training hyperparameter differs: {key}')
    if set(checkpoint.get('sources',{}))!=set(registration['source_splits']):
        raise ValueError('formal training source coverage differs')
    for name,expected in registration['source_splits'].items():
        if checkpoint.get('sources',{}).get(name)!=expected:
            raise ValueError(f'formal training source split differs: {name}')
        if {k:training.get('sources',{}).get(name,{}).get(k) for k in ('fit','val')}!=expected:
            raise ValueError(f'formal training log source split differs: {name}')
    device=checkpoint['hyper'].get('device','')
    wall=training.get('wall_seconds',0)
    memory=checkpoint.get('mem_available_gib',0)
    if (device not in ('cpu','cuda') and not device.startswith('cuda:') or
            not math.isfinite(memory) or memory<10 or not math.isfinite(wall) or wall<=0 or
            device.startswith('cuda') and wall>3600):
        raise ValueError('formal training device or GPU memory/time limit is unverified')


def registered_seed_audit(registration):
    """Reuse the sealed registration receipt; do not rescan unrelated experiments."""
    summary=OUT/'registration_seed_audit/seed_audit_summary.json'
    digest=sha(summary)
    if digest!=registration['broad_seed_audit_summary_sha256']:
        raise ValueError('registration seed audit summary changed')
    audit=json.loads(summary.read_text())
    if audit.get('passed') is not True:
        raise ValueError('registration seed audit did not pass')
    if sha(audit['evidence'])!=audit['evidence_sha256']:
        raise ValueError('registration seed audit evidence changed')
    return digest


def runtime_plan():
    import torch
    registration=json.loads(REGISTRATION.read_text());panel=frozen_panel()
    if not CANDIDATE.is_file():raise FileNotFoundError('formal candidate is missing; restore original v5 corpus and train first')
    checkpoint=torch.load(CANDIDATE,map_location='cpu',weights_only=False,mmap=True)
    log=CANDIDATE.with_name('train_log.json');training=json.loads(log.read_text())
    validate_training(checkpoint,training,registration)
    if sha(ROOT/'scripts/task78_train.py')!=registration['training_source_sha256']:
        raise ValueError('formal training source changed after registration')
    template=OLD/'runtime.json'
    if sha(template)!=registration['parent_runtime_sha256']:raise ValueError('parent runtime changed')
    result=json.loads(template.read_text())
    if sha(checkpoint['init'])!=result['models']['v5']['sha256']:
        raise ValueError('formal candidate did not initialize from the frozen v5')
    if sha(ROOT/'MEMORY/docs/task78-v5-corpus-requirements.json')!=registration['original_corpus_requirements_sha256']:
        raise ValueError('original corpus requirements changed')
    if any(sha(path)!=digest for path,digest in result['code'].items()):raise ValueError('frozen shared code changed')
    for demo in result['demos'].values():
        if any(sha(path)!=digest for path,digest in demo['files'].items()):raise ValueError('demonstration changed')
    result['models']['v6']=dict(path=str(CANDIDATE),sha256=sha(CANDIDATE))
    selected=set(identities(panel))
    missing_metadata=[]
    for name,item in result['models'].items():
        if sha(item['path'])!=item['sha256']:raise ValueError(f'{name} changed')
        cp=torch.load(item['path'],map_location='cpu',weights_only=False,mmap=True)
        overlap=declared_overlap(cp,selected,name)
        if overlap:raise ValueError(f'{name} has seen formal evaluation identities')
        if overlap is None:missing_metadata.append(name)
    if sha(result['executable']['path'])!=result['executable']['sha256']:raise ValueError('engine changed')
    original=checkpoint['input_provenance']['original']
    if sha(original['descriptor'])!=original['sha256']:raise ValueError('original corpus descriptor changed')
    if sha(ROOT/'artifacts/task78/export.json')!=registration['new_export_sha256']:raise ValueError('corrected corpus export changed')
    result.update(panel_sha256=sha(OUT/'tasks.json'),registration_sha256=sha(REGISTRATION),
                  training_log_sha256=sha(log),original_descriptor_sha256=original['sha256'],
                  model_seed_metadata_missing=missing_metadata)
    result['registration_seed_audit_summary_sha256']=registered_seed_audit(registration)
    result['historical_seed_audit_scope']='registration_snapshot'
    for name in ('task78_formal_ab.py','task78_train.py','task78_seed_audit.py'):
        path=ROOT/'scripts'/name;result['code'][str(path)]=sha(path)
    write_once(OUT/'runtime.json',result)
    return result


def run_arm(arm):
    from task78_roof_truth import no_games
    from task78_train import require_memory
    with (ROOT/'artifacts/task78/games-controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=runtime_plan()
        for cell in frozen_panel():
            no_games();require_memory();runtime_plan()
            args,output=shared.command(cell,arm,plan,OUT);output.mkdir(parents=True,exist_ok=True)
            write_once(output/'cell_run.json',dict(runtime_sha256=sha(OUT/'runtime.json'),
                command=args,cell_id=cell['cell_id'],arm=arm))
            with (output/'run.log').open('a') as stream:
                subprocess.run(args,stdout=stream,stderr=subprocess.STDOUT,check=True,
                    env=os.environ|dict(PVZ_NO_POPUP='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))


def report():
    panel=frozen_panel();results={};pooled={};plan=None
    for cell in panel:
        expected=set(cell.get('future_seeds') or cell['seeds']);arms=[]
        for arm,legacy in (('v5','v3'),('v6','v4')):
            output=OUT/cell['cell_id']/legacy/cell['task_id']
            rows,missing=shared.load_arm(output,expected,'future_seeds' in cell)
            if rows:
                if plan is None:plan=runtime_plan()
                args,_=shared.command(cell,arm,plan,OUT)
                if json.loads((output/'cell_run.json').read_text())!=dict(runtime_sha256=sha(OUT/'runtime.json'),command=args,cell_id=cell['cell_id'],arm=arm):
                    raise ValueError('formal rows belong to another runtime')
            arms.append((rows,missing))
        (old,missing_old),(new,missing_new)=arms
        if old is None or new is None:
            results[cell['cell_id']]=dict(terrain=cell['terrain'],pairs=0,complete=False,
                missing_old=missing_old,missing_new=missing_new);continue
        seeds=set(old)&set(new);gain=sum(new[s] and not old[s] for s in seeds);harm=sum(old[s] and not new[s] for s in seeds)
        entry=dict(terrain=cell['terrain'],pairs=len(seeds),old_wins=sum(old[s] for s in seeds),
                   new_wins=sum(new[s] for s in seeds),gain=gain,harm=harm,net=gain-harm,
                   mcnemar_p=shared.mcnemar(gain,harm),complete=len(seeds)==cell['pairs'] and not missing_old and not missing_new)
        results[cell['cell_id']]=entry
        acc=pooled.setdefault(cell['terrain'],dict(pairs=0,old_wins=0,new_wins=0,gain=0,harm=0))
        for key in acc:acc[key]+=entry[key]
    for acc in pooled.values():
        n=acc['pairs'];acc.update(net=acc['gain']-acc['harm'],mcnemar_p=shared.mcnemar(acc['gain'],acc['harm']),
            conservative_lower=shared.clopper_lower(acc['gain'],n)-shared.clopper_upper(acc['harm'],n) if n else None)
    gates=shared.game_gates(results,pooled,panel)
    result=dict(cells=results,terrain=pooled,gates=gates,complete=all(v['status']=='evaluated' for v in gates.values()),
                all_game_gates_pass=all(v['passed'] is True for v in gates.values()),diagnostic=False if plan else None,
                preregistered_recipe_met=True if plan else None,
                model_seed_metadata_missing=plan.get('model_seed_metadata_missing',[]) if plan else None,
                formal_candidate_verified=plan is not None,includes_terminal_and_stability_gates=False)
    (OUT/'paired.json').write_text(json.dumps(result,indent=1)+'\n');return result


if __name__=='__main__':
    if len(sys.argv)!=2 or sys.argv[1] not in ('tasks','freeze','v5','v6','report'):
        raise SystemExit('usage: task78_formal_ab.py tasks|freeze|v5|v6|report')
    stage=sys.argv[1]
    result=prepare() if stage=='tasks' else runtime_plan() if stage=='freeze' else report() if stage=='report' else run_arm(stage)
    if result is not None:print(json.dumps(result,indent=1))
