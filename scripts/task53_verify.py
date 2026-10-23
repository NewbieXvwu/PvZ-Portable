"""Independent delivery audit of actual task53 evidence, learner rows and the live v5 reader."""
from __future__ import annotations
import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'python')]
import critical_forks as f
import critical_forks_data as data


def verify(plan):
    import numpy as np
    import torch
    from pvz_frame_store import frame_path,read_frames
    from pvz_rollout_policy import load_student,act
    from pvz_student_observation import StudentHistory
    from task53_evaluate import summary
    from task53_repro import live_games
    out=Path(plan['out'])
    if not (out/'pipeline-complete.json').exists() or live_games():
        raise RuntimeError('finish all experiment games before final audit')
    groups,by_terrain=data.screen_summary(plan,plan['jobs']);confirmed=data.confirm_summary(plan)
    expected=f.load(ROOT/'MEMORY/docs/task53-critical-forks-results.json')
    assert by_terrain==expected['by_terrain']
    assert len(groups)==47 and sum(g['labels'] for g in groups)==9723
    assert sum(len(g['candidates']) for g in groups)==3241
    assert all(by_terrain[t]['games']==8 for t in ('day','night','pool'))
    registered=f.load(ROOT/'MEMORY/docs/task53-critical-forks-sampling-v2.json')
    assert registered['jobs']==plan['jobs'],'frozen point selection changed'
    for job in plan['jobs']:
        recorded=f.load(out/'roots'/job['id']/'root.json')
        stable={k:v for k,v in job.items() if k not in ('stage','samples','indices','inspect_only')}
        assert recorded['job']==stable
    for episode in plan['episodes']:
        archive=Path(episode['archive'])
        assert f.sha(frame_path(archive))==episode['frames_sha256']
        assert f.sha(archive/'meta.json')==episode['meta_sha256']
    selected=f.load(out/'confirmation.json')['jobs']
    seeds=[]
    for stage,jobs,count in [('screen',plan['jobs'],3),('confirm',selected,16)]:
        for j in jobs:
            for k in range(count):
                seeds.extend(f.future_seed(j['id'],stage,k,search) for search in (False,True))
    assert len(set(seeds))==len(seeds),'future seed collision'
    exported=f.load(out/'export.json');registration=f.load(out/'training-registration.json')
    prerepair=f.load(out/'frozen/training-registration-before-metadata-repair.json')
    assert registration['replay']==prerepair['replay'] and registration['evaluation']==prerepair['evaluation']
    assert {k:v for k,v in registration['recipe'].items() if k!='provenance'}=={
        k:v for k,v in prerepair['recipe'].items() if k!='provenance'}
    recipe=f.load(out/'training/recipe.json');base=registration['recipe']
    assert all(recipe[k]==v for k,v in base.items() if k not in ('labels','label_weight'))
    assert set(recipe['labels'])==set(exported['labels']) and recipe['label_weight']==exported['label_weight']
    source=f.load(out/'training/model/source_rows.json')
    assert source['rows']==3072+len(exported['student_corrections'])
    assert sum(v['rows'] for k,v in source['sources'].items() if k.startswith('replay_'))==3072
    train_corrections=sum(v['training_rows'] for k,v in source['sources'].items() if k.startswith('fork_'))
    assert train_corrections>0,'new corrections must participate in training, not only validation'
    checkpoint=torch.load(out/'training/model/model.pt',map_location='cpu',weights_only=False)
    original=torch.load(plan['teacher']['rollout_policy'],map_location='cpu',weights_only=False)
    effective=f.load(out/'training/recipe.effective.json');resolution=f.load(out/'training/step-resolution.json')
    assert {k:v for k,v in effective.items() if k!='max_steps'}=={k:v for k,v in recipe.items() if k!='max_steps'}
    from task53_pipeline import effective_step_cap
    assert effective['max_steps']==effective_step_cap(recipe,sum(v['training_rows'] for v in source['sources'].values()))
    assert resolution['registered_recipe_sha256']==f.sha(out/'training/recipe.json')
    assert resolution['effective_recipe_sha256']==f.sha(out/'training/recipe.effective.json')
    assert checkpoint['recipe_sha256']==resolution['effective_recipe_sha256']
    assert checkpoint['optimizer_steps']==effective['max_steps']
    assert checkpoint['optimizer_steps']>0
    changed=sum(not torch.equal(v,original['state_dict'][k]) for k,v in checkpoint['state_dict'].items()
                if k in original['state_dict'])
    assert changed>0,'checkpoint weights never changed'
    rows=[json.loads(line) for line in (out/'evaluation/games.jsonl').read_text().splitlines()]
    paired=summary(rows,registration['evaluation']);assert len(rows)==384
    saved=f.load(ROOT/'MEMORY/docs/task53-critical-forks-evaluation.json')
    assert all(paired[k]==saved[k] for k in ('overall','by_terrain','paired'))
    for row in rows:
        assert row['model_sha256']==saved['contract']['models'][row['arm']]['sha256']
    evaluation_seeds={j['reset']['seed'] for j in registration['evaluation']}
    assert len(evaluation_seeds)==192 and not evaluation_seeds&{e['reset']['seed'] for e in plan['episodes']}
    # Use the real read-only task43 parser, rather than a schema-shaped mock.
    reader_path=Path('/home/newbiexvwu/PvZ-search/.claude/worktrees/v5/scripts/train_value_widen.py')
    search_paths=list(sys.path)
    try:
        spec=importlib.util.spec_from_file_location('task53_readonly_v5_reader',reader_path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    finally:
        sys.path[:]=search_paths
    shards=[Path(v['path']) for v in exported['value_files']]
    assert all(f.sha(v['path'])==v['sha256'] for v in exported['value_files'])
    loaded=module.Decisions(shards,set(),'')
    indices=np.concatenate([loaded.fit[:2],loaded.val[:2]])
    boards,scalars,labels=loaded.batch(indices)
    assert len(boards)==len(scalars)==len(labels)==len(indices)
    assert sum(v['rows'] for v in exported['value_files'])==sum(len(p['won']) for p in loaded.parts)
    views={}
    by_id={j['id']:j for j in plan['jobs']}
    for terrain in ('day','night','pool'):
        directory=out/'value-data/by-terrain'/terrain;directory.mkdir(parents=True,exist_ok=True)
        paths=[p for p in shards if by_id[p.stem]['terrain']==terrain]
        for path in paths:
            link=directory/path.name
            if not link.exists():link.symlink_to(path)
            assert link.resolve()==path.resolve()
        views[terrain]=dict(directory=str(directory),shards=len(paths))
    pool_paths=[p for p in shards if by_id[p.stem]['terrain']=='pool']
    pool_held={int(s) for p in loaded.parts if p['config']['task_id'].startswith('pool_')
               for s in p['seed'].unique().tolist() if module.held(s)}
    control=None
    if not pool_held:
        # This extra file is a real parser-only validation control, never part of the
        # task53 export or student training. The new pool data remain training-only
        # under task43's pre-existing seed hash and need its existing holdout corpus.
        for p in sorted((Path('/home/newbiexvwu/PvZ-search/.claude/worktrees/v5/artifacts/task43/l21_dump')).glob('*.pt')):
            d=torch.load(p,map_location='cpu',weights_only=False,mmap=True)
            if d.get('config',{}).get('task_id','').startswith('pool_') and any(module.held(s) for s in d['seed'].unique().tolist()):
                control=dict(path=str(p),sha256=f.sha(p),format_check_only=True);break
        if control is None:raise ValueError('need an existing v5 held-out shard for pool-only parser check')
    pool_reader=module.Decisions(pool_paths+([Path(control['path'])] if control else []),set(),'pool_')
    pool_reader.batch(np.array([pool_reader.fit[0],pool_reader.val[0]]))
    # Check the root supervision path itself against the frozen student's recorded action.
    incumbent=load_student(plan['teacher']['rollout_policy']);candidate=load_student(out/'training/model/model.pt')
    matches=[]
    for correction in confirmed['corrections']:
        if not correction['student_representable']:continue
        job=correction['job'];root=f.load(out/'roots'/job['id']/'root.json')
        frames=read_frames(frame_path(Path(job['archive'])));i=job['decision']
        history=StudentHistory(incumbent.obs)
        for frame in frames[:i-1]:history.extra(frame)
        legal=[a for a in root['candidates'] if a['type']=='plant']
        frame,previous=frames[i-1],frames[i-2] if i>=2 else None
        old=act(incumbent,frame,previous,legal,context=np.asarray(root['context']),history=deepcopy(history))
        new=act(candidate,frame,previous,legal,context=np.asarray(root['context']),history=deepcopy(history))
        assert old==root['student_action'],'training frame/history differs from live student root'
        matches.append(dict(root=job['id'],label=correction['action'],incumbent=old,candidate=new,
                            candidate_matches_label=new==correction['action']))
    support={}
    for terrain in ('day','night','pool'):
        subset=[g for g in groups if g['job']['terrain']==terrain];support[terrain]={}
        for metric,gap in plan['protocol']['confirmation']['mean_gap'].items():
            support[terrain][metric]={}
            for label,nonterminal in [('all_pairs',False),('nonterminal_v4_only',True)]:
                usable=[g for g in subset if data.rank_counts(g,metric,gap,nonterminal)['pairs']]
                support[terrain][metric][label]=dict(informative_games=len({g['job']['archive'] for g in usable}),
                                                    informative_roots=len(usable))
    search_audits=[f.load(out/'roots'/j['id']/'search.json') for j in plan['jobs']]
    assert all({'static_scores','static_selected','candidates','halving_counts','halving_pruned'}<=s.keys()
               for s in search_audits if s['searched'])
    correction_search=[]
    for correction in confirmed['corrections']:
        s=f.load(out/'roots'/correction['root']/'search.json');action=correction['action']
        correction_search.append(dict(root=correction['root'],searched=s['searched'],
            static_selected=action in s.get('static_selected',[]) if s['searched'] else None,
            candidate=action in s.get('candidates',[]) if s['searched'] else None,
            halving_pruned=action in s.get('halving_pruned',[]) if s['searched'] else None,chosen=s['chosen']==action))
    value=dict(status='evidence_and_formats_verified',roots=47,episodes=24,candidates=3241,screen_labels=9723,
        confirm_labels=confirmed['labels'],all_future_seeds_unique=True,future_seed_count=len(seeds),
        exact_terminal_and_trace_coverage=True,frozen_inputs_unchanged=True,
        training=dict(old_rows=3072,all_new_rows=len(exported['student_corrections']),new_training_rows=train_corrections,
                      original_registration_settings_unchanged=True,metadata_only_revision_verified=True,
                      registered_step_cap_preserved=recipe['max_steps'],step_resolution=resolution,
                      optimizer_steps=checkpoint['optimizer_steps'],changed_parameter_tensors=changed,
                      root_input_and_incumbent_action_parity=matches),
        evaluation=dict(pairs=192,games=384,complete_and_model_hashes_verified=True),
        v4_agreement_support=support,
        formal_search_audits=dict(roots=len(search_audits),searched=sum(s['searched'] for s in search_audits),
            complete_search_fields_for_all_executed_searches=True),
        confirmed_correction_formal_search_audit=correction_search,
        v5_reader=dict(path=str(reader_path),sha256=f.sha(reader_path),shards=len(shards),
                       rows=sum(v['rows'] for v in exported['value_files']),
                       training_rows=len(loaded.fit),validation_rows=len(loaded.val),real_batch_loaded=True,
                       terrain_views=views,pool_only_batch_loaded=True,pool_export_held_out_seeds=sorted(pool_held),
                       parser_validation_control=control,
                       pool_split_note='Use new pool shards as additional training data alongside task43 existing held-out corpus; parser control is not exported or trained here.'),
        ranking_pairs=exported['ranking_pairs'],cpu_seconds_metered=f.spent_cpu(out),active_game_pids=[])
    value['source_cohorts']={t:sorted({e['source'] for e in plan['episodes'] if e['terrain']==t})
                              for t in ('day','night','pool')}
    value['mower_convention']='same as formal search_play: last nonterminal ready-mower count; winning cleanup does not count as consumed mowers'
    f.save(ROOT/'MEMORY/docs/task53-critical-forks-audit.json',value)
    print('VERIFIED',value['roots'],'roots;',value['screen_labels'],'screen labels;',train_corrections,
          'new training rows; live v5 batches loaded',flush=True)
    return value


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('plan',type=Path)
    a=p.parse_args();verify(f.check_plan(a.plan))
