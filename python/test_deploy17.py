"""task165: small allocation/recipe/pack fixtures; never start a production service."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import pytest
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'python')]
import curriculum_queue as q
import make_random_deck_cells as decks


def fixture(tmp_path):
    archive=tmp_path/'archive';archive.mkdir()
    (archive/'meta.json').write_text(json.dumps(dict(verified=True,task=dict(seed=42))))
    cells={};tasks={}
    for kind,n in [('opening',8),('demo',275),('rd',16),('mult',43)]:
        for i in range(n):
            name=f'{kind}_l9_{i:03d}'
            cells[name]=dict(level=9,mult_index=0,last_round=1,windows={},family='fixture')
            tasks[name]=dict(task_id=name,level=9,seeds=list(range(2048)))
            if kind=='opening': cells[name]['opening_specialist']=True
            if kind=='rd': cells[name]['random_deck_cell']=True
            if kind=='demo':
                cells[name].update(archive=str(archive),task_id=name,seeds=list(range(2048)),ladder=[100])
                tasks[name]['seeds']=[42]
    state=dict(round=47,cells=cells,offsets={},retired_cells={},log=[])
    return state,tasks,{k:tmp_path/'panel.json' for k in tasks}


def counts(state,groups):
    totals=dict(opening=0,demo=0,random=0,multiplier=0)
    for group in groups:
        cell=state['cells'].get(group['base'],state['retired_cells'].get(group['base'],{}))
        kind='opening' if cell.get('opening_specialist') else 'random' if cell.get('random_deck_cell') else 'demo' if cell.get('archive') else 'multiplier'
        totals[kind]+=group['games']
    return totals


def test_deploy17_allocation_and_partial_resume(tmp_path):
    original,tasks,manifests=fixture(tmp_path)
    def plan(state,**kw): return q.plan_round(state,tasks,{},[1.0],manifests,768,48,**kw)[0]
    legacy=plan(copy.deepcopy(original))
    assert counts(original,legacy)==dict(opening=192,demo=576,random=0,multiplier=0)
    state=copy.deepcopy(original)
    proposed=plan(state,demo_start_share=.25,random_deck_share=.25)
    assert counts(state,proposed)==dict(opening=192,demo=192,random=192,multiplier=192)
    complete=[proposed[0]]+[g for g in proposed if g['base'].startswith('rd_')][:4]+[g for g in proposed if g['base'].startswith('demo_')][:4]
    state['completed_groups']=complete
    for group in complete:
        state['offsets'][group['task_id']]=group['offset']+group['games']
        if not group['base'].startswith('opening'):state['cells'][group['base']]['last_round']=48
    demo=next(g['base'] for g in complete if g['base'].startswith('demo_'))
    state['retired_cells'][demo]=state['cells'].pop(demo)
    resumed=plan(state,demo_start_share=.25,random_deck_share=.25)
    assert counts(state,complete+resumed)==dict(opening=192,demo=192,random=192,multiplier=192)
    assert not {g['label'] for g in resumed}&{g['label'] for g in complete}


def test_deploy17_scarcity_never_borrows_demo_quota(tmp_path):
    state,tasks,manifests=fixture(tmp_path)
    state['cells']={k:v for k,v in state['cells'].items() if not k.startswith('rd_')}
    plan,_=q.plan_round(state,tasks,{},[1.0],manifests,768,48,demo_start_share=.25,random_deck_share=.25)
    totals=counts(state,plan)
    assert totals['demo']<=192 and totals['multiplier']>192 and sum(totals.values())<=768


def test_deploy17_recipe_and_frozen_pool(tmp_path):
    a=json.loads((ROOT/'recipes/dw4i2_v2timer.json').read_text())
    b=json.loads((ROOT/'recipes/dw4i2_v2timer_history10.json').read_text())
    assert {k for k in a if a[k]!=b[k]}=={'label_weight'}
    assert len(a['label_weight'])==33 and all(k.startswith('z') for k in a['label_weight'])
    assert set(a['label_weight'].values())=={1.5} and set(b['label_weight'].values())=={1.0}
    registry=json.loads(decks.REGISTRY.read_text())
    holdouts=[ROOT/'configs/deploy17_deck_holdouts.json']
    pool=decks.load_frozen_pool(ROOT/'configs/deploy17_random_deck_pool.json',registry,holdouts)
    training={tuple(sorted(t['deck'])) for t in pool+registry['tasks'] if t.get('split','train')=='train'}
    assert not training&decks.heldout_signatures(registry,holdouts)
    assert len(pool)>0 and any('task148' in t.get('also_sources',[]) for t in pool)
    duplicate=tmp_path/'bad.json';duplicate.write_text(json.dumps(dict(tasks=[pool[0],pool[0]])))
    with pytest.raises(ValueError):decks.load_frozen_pool(duplicate,registry,holdouts)


def test_deploy17_shared_pack_weights_match_physical_pack(tmp_path):
    import torch
    from test_incremental_pack import fixture as rows,options
    from pack_training_rows import pack
    import train_rollout_policy as trainer
    from pvz_training_rows import LabelWeightAdjustment
    data=tmp_path/'data';rows(data,['fresh','z_old'])
    a=pack(data,options(data,['fresh','z_old'],label_weight={'z_old':1.5}),tmp_path/'cache')
    adjust=LabelWeightAdjustment({'z_old':1.0},{'z_old':1.5})
    shared=trainer.read_pack(a,data,row_observer=adjust.observe)
    original=shared[5].clone();adjust.apply(shared[5])
    b=pack(data,options(data,['fresh','z_old'],label_weight={'z_old':1.0}),tmp_path/'cache')
    physical=trainer.read_pack(b,data)
    torch.testing.assert_close(shared[5],physical[5],rtol=0,atol=1e-6)
    assert not torch.equal(original,shared[5])
    torch.testing.assert_close(trainer.read_pack(a,data)[5],original,rtol=0,atol=0)
    assert torch.equal(shared[2],physical[2]) and torch.equal(shared[3],physical[3])


def test_deploy17_pipeline_accepts_label_only_alternate(tmp_path):
    import threading
    pipeline=q.TrainingPipeline.__new__(q.TrainingPipeline)
    pipeline.args=SimpleNamespace(recipe=ROOT/'recipes/dw4i2_v2timer.json',
        alternate_recipe=ROOT/'recipes/dw4i2_v2timer_history10.json',candidates_per_recipe=2,
        state=tmp_path/'state.json',log=tmp_path/'qc.log')
    pipeline.directory=tmp_path/'pipeline';pipeline.build_workers=1
    pipeline.state=dict(round=1,log=[],training_jobs=[])
    pipeline._plan_build=Mock();pipeline._snapshot_selection=Mock();pipeline._seeded=Mock()
    pipeline.executor=Mock();pipeline.futures=[]
    pipeline.enqueue()
    job=pipeline.state['training_jobs'][0]
    assert [r['recipe_family'] for r in job['candidate_recipes']]==['A','A','B','B']
    assert job['training_seeds']==[1,2,3,4]
