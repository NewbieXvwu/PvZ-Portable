from __future__ import annotations

import concurrent.futures
import hashlib
import importlib.util
import json
import pickle
import random
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'python'),str(ROOT/'scripts')]
OUT=Path.home()/'PvZAgent-gru-bc-level7-v1'
TEACHER=ROOT/'scripts/bc_pipeline/scripted_baseline_d1.py'
RESOURCE=str(Path.home()/'.cache/pvz-research-resources')
CHECKPOINT=Path.home()/'PvZAgent-bc-handoff/best_model_pipeline.pt'
TRAIN_FIRST=1_400_000
EPISODES=5_000
WORKERS=8
SHARD_SIZE=EPISODES//WORKERS
WAIT_CAP_PER_SHARD=150_000
TARGET_SAMPLES=1_000_000
VALIDATION_SAMPLES=20_000
MAX_ACTIONS=4_000
CONFIG={'layers':6,'width':256,'heads':8,'ff_width':1024,
        'gru_layers':2,'gru_width':256,'critic_width':256,'critic_layers':1,
        'input_flags':7,'wait_mode':'events','wait_mask':'progress_v1'}


def clean_action(raw):
    action={key:raw[key] for key in ('type','packet','row','col','ticks','until') if key in raw}
    if action['type']=='wait':
        action['until']='timeout'
    return action


def transition(obs, action, previous_action, elapsed, events, previous_wait_result,
               action_info, model_api, config):
    tensors,metadata=model_api.observation_tokens(obs,config['input_flags'])
    legal=model_api.policy_legal_summary(obs,config)
    if action['type']=='wait' and not legal['wait_condition_mask'][model_api.CONDITIONS.index('timeout')]:
        raise RuntimeError('timeout wait condition masked in demonstration')
    return {
        'tokens':model_api.pack_tokens(tensors,metadata),
        'wave':int(obs['wave']),
        'legal':legal,
        'previous_action':previous_action,
        'elapsed_since_previous_observation':int(elapsed),
        'events':events or {},
        'previous_wait_result':previous_wait_result,
        'action':action,
        'wait_result':action_info.get('wait_result'),
        'action_duration_ticks':int(action_info.get('ticks_advanced',0)),
    }


def legal_action(obs, action, model_api, config):
    raw=obs['legal_actions']
    try:
        model_api.validate_policy_action(config,action)
        if action['type']=='plant':
            return any(a['packet']==action['packet'] and a['row']==action['row']
                       and a['col']==action['col'] for a in raw['plants'])
        if action['type']=='shovel':
            cell=action['row']*9+action['col']
            summary=model_api.legal_summary(raw)
            return bool((summary['shovel_mask']>>cell)&1)
        return bool(raw.get('wait',True)) and action['ticks'] in model_api.WAIT_TICKS
    except (KeyError,ValueError):
        return False


def run_shard(index):
    import pvz_agent_model as model_api
    from pvz_event_env import EventWaitEnv
    from pvz_env import TaskSpec

    torch_checkpoint=None
    try:
        import torch
        torch_checkpoint=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
        if torch_checkpoint['config']!=CONFIG:
            raise RuntimeError('mainline checkpoint config changed')
    finally:
        del torch_checkpoint

    spec=importlib.util.spec_from_file_location(f'teacher_d1_shard_{index}',TEACHER)
    teacher=importlib.util.module_from_spec(spec); spec.loader.exec_module(teacher)
    deck=(0,1,2,3,4,5,7)
    env=EventWaitEnv(RESOURCE,headless=True)
    first=TRAIN_FIRST+index*SHARD_SIZE
    db_path=OUT/f'shard_{index:02d}.sqlite'
    db=sqlite3.connect(db_path)
    db.execute('PRAGMA journal_mode=OFF')
    db.execute('PRAGMA synchronous=OFF')
    db.execute('CREATE TABLE nonwait (id INTEGER PRIMARY KEY, episode_id INTEGER, step_id INTEGER, kind TEXT, packet INTEGER, plant_type INTEGER, payload BLOB)')
    db.execute('CREATE TABLE waits (id INTEGER PRIMARY KEY, episode_id INTEGER, step_id INTEGER, payload BLOB)')
    db.execute('BEGIN')
    rng=random.Random(42_000+first)
    counts=Counter()
    teacher_packets=Counter()
    legality=Counter()
    wins=truncated=decisions=wait_seen=wait_kept=nonwait_kept=0
    for offset in range(SHARD_SIZE):
        seed=first+offset
        task=TaskSpec(level=7,seed=seed,playthrough=2,
                      profile=teacher.profile_for_deck(deck))
        obs,_=env.reset(deck=deck,task=task)
        previous_action=None
        previous_wait_result=None
        elapsed=0
        events={}
        episode_actions=0
        while not obs['terminal'] and episode_actions<MAX_ACTIONS:
            raw=teacher.choose(obs)
            action=clean_action(raw)
            kind=action['type']
            counts[kind]+=1
            if kind=='plant':
                packet=action['packet']
                ptype=int(obs['packets'][packet]['type'])
                teacher_packets[str(packet)]+=1
            else:
                packet=None
                ptype=None
            is_legal=legal_action(obs,action,model_api,CONFIG)
            legality['valid' if is_legal else 'invalid']+=1
            keep=False
            wait_slot=None
            if kind!='wait':
                keep=True
            else:
                wait_seen+=1
                if wait_kept<WAIT_CAP_PER_SHARD:
                    keep=True
                    wait_kept+=1
                    wait_slot=wait_kept
                else:
                    choice=rng.randrange(wait_seen)
                    if choice<WAIT_CAP_PER_SHARD:
                        keep=True
                        wait_slot=choice+1
            obs_before=obs
            obs,_,done,_,info=env.step(action)
            if not info.get('ok'):
                raise RuntimeError(f'E1 teacher action rejected: seed={seed} action={action}')
            if keep:
                item=transition(obs_before,action,previous_action,elapsed,events,
                                previous_wait_result,info,model_api,CONFIG)
                item['episode_id']=seed
                item['step_id']=episode_actions
                payload=pickle.dumps(item,protocol=5)
                if kind=='wait':
                    if wait_slot<=wait_kept and wait_slot==wait_kept and wait_kept<=WAIT_CAP_PER_SHARD:
                        # New reservoir entry; a replacement uses an already occupied slot.
                        occupied=db.execute('SELECT 1 FROM waits WHERE id=?',(wait_slot,)).fetchone()
                        if occupied is None:
                            db.execute('INSERT INTO waits(id,episode_id,step_id,payload) VALUES(?,?,?,?)',
                                       (wait_slot,seed,episode_actions,payload))
                        else:
                            db.execute('UPDATE waits SET episode_id=?,step_id=?,payload=? WHERE id=?',
                                       (seed,episode_actions,payload,wait_slot))
                    else:
                        db.execute('UPDATE waits SET episode_id=?,step_id=?,payload=? WHERE id=?',
                                   (seed,episode_actions,payload,wait_slot))
                else:
                    db.execute('INSERT INTO nonwait(episode_id,step_id,kind,packet,plant_type,payload) VALUES(?,?,?,?,?,?)',
                               (seed,episode_actions,kind,packet,ptype,payload))
                    nonwait_kept+=1
                if (nonwait_kept+wait_kept)%5_000==0:
                    db.commit(); db.execute('BEGIN')
            previous_action=action
            previous_wait_result=info.get('wait_result')
            elapsed=int(info.get('ticks_advanced',0))
            events=info.get('events') or {}
            decisions+=1
            episode_actions+=1
            if done: break
        is_terminal=bool(obs['terminal'])
        truncated+=int(not is_terminal)
        wins+=int(int(obs.get('result',0))==1)
        if (offset+1)%125==0:
            print(f'shard={index} episodes={offset+1}/{SHARD_SIZE} wins={wins} decisions={decisions}',flush=True)
    db.commit()
    db.close()
    env.close()
    return {'shard':index,'seed_range':[first,first+SHARD_SIZE-1],
            'episodes':SHARD_SIZE,'wins':wins,'truncated':truncated,
            'decisions':decisions,'teacher_action_counts':dict(counts),
            'teacher_plant_packet_counts':dict(teacher_packets),
            'legal_actions':dict(legality),'nonwait_saved':nonwait_kept,
            'waits_seen':wait_seen,'wait_reservoir_saved':wait_kept,
            'db_path':str(db_path)}


def build_dataset(shards, config):
    candidates=[]
    nonwait_total=plant_count=shovel_count=0
    per_shard=[]
    for shard in shards:
        conn=sqlite3.connect(shard['db_path'])
        n=conn.execute('SELECT COUNT(*) FROM nonwait').fetchone()[0]
        w=conn.execute('SELECT COUNT(*) FROM waits').fetchone()[0]
        plant=shard['teacher_action_counts'].get('plant',0)
        shovel=shard['teacher_action_counts'].get('shovel',0)
        nonwait_total+=n; plant_count+=plant; shovel_count+=shovel
        per_shard.append((shard,n,w))
        candidates.extend((shard['shard'],i) for i in range(1,w+1))
        conn.close()
    wait_target=min(max(0,TARGET_SAMPLES-nonwait_total),10*plant_count,len(candidates))
    total=nonwait_total+wait_target
    if total<TARGET_SAMPLES:
        raise RuntimeError(f'5,000 demos cannot supply 1M rows with wait:plant<=10:1: '
                           f'nonwait={nonwait_total}, plant={plant_count}, waits={len(candidates)}')
    if wait_target>10*plant_count:
        raise RuntimeError('selected wait:plant ratio exceeds 10:1')
    selected=set(random.Random(81_022).sample(candidates,wait_target)) if wait_target else set()
    val_ids=set(random.Random(81_023).sample(range(1,total+1),min(VALIDATION_SAMPLES,total-1)))
    final_path=OUT/'scripted_level7_bc_samples_v2.sqlite'
    if final_path.exists(): final_path.unlink()
    out=sqlite3.connect(final_path)
    out.execute('PRAGMA journal_mode=OFF'); out.execute('PRAGMA synchronous=OFF')
    out.execute('CREATE TABLE samples (id INTEGER PRIMARY KEY, split INTEGER, episode_id INTEGER, step_id INTEGER, kind TEXT, packet INTEGER, plant_type INTEGER, payload BLOB)')
    sample_id=0
    dataset_counts=Counter(); dataset_packets=Counter(); dataset_plant_types=Counter()
    for shard,n,w in per_shard:
        src=sqlite3.connect(shard['db_path'])
        queries=[('SELECT episode_id,step_id,kind,packet,plant_type,payload FROM nonwait ORDER BY id',None)]
        wait_ids=sorted(i for si,i in selected if si==shard['shard'])
        for start in range(0,len(wait_ids),800):
            ids=wait_ids[start:start+800]
            if ids:
                q=','.join('?' for _ in ids)
                queries.append((f'SELECT episode_id,step_id,"wait",NULL,NULL,payload FROM waits WHERE id IN ({q})',ids))
        batch=[]
        for sql,args in queries:
            cursor=src.execute(sql,args or ())
            for episode_id,step_id,kind,packet,plant_type,payload in cursor:
                sample_id+=1
                split=int(sample_id in val_ids)
                batch.append((sample_id,split,episode_id,step_id,kind,packet,plant_type,payload))
                dataset_counts[kind]+=1
                if kind=='plant':
                    dataset_packets[str(packet)]+=1
                    dataset_plant_types[str(plant_type)]+=1
                if len(batch)>=2_000:
                    out.executemany('INSERT INTO samples VALUES(?,?,?,?,?,?,?,?)',batch)
                    batch.clear()
        if batch:
            out.executemany('INSERT INTO samples VALUES(?,?,?,?,?,?,?,?)',batch)
        src.close()
    out.commit()
    out.execute('CREATE INDEX split_id ON samples(split,id)')
    out.execute('CREATE INDEX sequence_order ON samples(split,episode_id,step_id)')
    out.commit()
    actual=out.execute('SELECT COUNT(*) FROM samples').fetchone()[0]
    # Sequence training must hold out complete episodes. A row-wise split would
    # place adjacent steps from the same episode in both train and validation.
    episode_counts=out.execute(
        'SELECT episode_id,COUNT(*) FROM samples GROUP BY episode_id ORDER BY episode_id'
    ).fetchall()
    random.Random(81_023).shuffle(episode_counts)
    validation_episode_ids=[]
    validation_count=0
    for episode_id,count in episode_counts:
        if validation_count >= VALIDATION_SAMPLES:
            break
        validation_episode_ids.append((episode_id,))
        validation_count+=count
    out.execute('UPDATE samples SET split=0')
    out.executemany('UPDATE samples SET split=1 WHERE episode_id=?',validation_episode_ids)
    out.commit()
    train_n=out.execute('SELECT COUNT(*) FROM samples WHERE split=0').fetchone()[0]
    val_n=out.execute('SELECT COUNT(*) FROM samples WHERE split=1').fetchone()[0]
    out.close()
    if actual<TARGET_SAMPLES or sample_id!=actual:
        raise RuntimeError(f'dataset row count mismatch: {actual} expected {total}')
    return {'path':str(final_path),'total':actual,'train':train_n,'validation':val_n,
            'teacher_plant_actions':plant_count,'teacher_shovel_actions':shovel_count,
            'sampled_wait_actions':wait_target,'wait_to_plant_ratio':wait_target/plant_count if plant_count else None,
            'action_counts':dict(dataset_counts),'plant_packet_counts':dict(dataset_packets),
            'plant_type_counts':dict(dataset_plant_types),'retained_all_nonwait':True,
            'wait_reservoir_candidates':len(candidates)}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    import torch
    ckpt=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    if ckpt['config']!=CONFIG:
        raise RuntimeError(f'checkpoint config differs: {ckpt["config"]}')
    print(json.dumps({'phase':'collection','config':CONFIG,'seeds':[TRAIN_FIRST,TRAIN_FIRST+EPISODES-1],
                      'wait_cap_per_shard':WAIT_CAP_PER_SHARD}),flush=True)
    started=time.time()
    args=list(range(WORKERS))
    shards=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures=[pool.submit(run_shard,i) for i in args]
        for i,f in enumerate(concurrent.futures.as_completed(futures),1):
            result=f.result(); shards.append(result)
            print(f'finished shard={result["shard"]} episodes={result["episodes"]} wins={result["wins"]} '
                  f'decisions={result["decisions"]} legal={result["legal_actions"]}',flush=True)
    shards.sort(key=lambda x:x['shard'])
    dataset=build_dataset(shards,CONFIG)
    for shard in shards:
        shard_path=Path(shard['db_path']).resolve()
        if shard_path.parent != OUT.resolve() or not shard_path.name.startswith('shard_'):
            raise RuntimeError(f'refusing to remove non-shard path: {shard_path}')
        shard_bytes=shard_path.stat().st_size
        shard_path.unlink()
        print(f"removed merged shard={shard_path.name} bytes={shard_bytes}",flush=True)
    payload={'schema_version':1,'simulator_sha256':hashlib.sha256((ROOT/'build/pvz-portable').read_bytes()).hexdigest(),
             'configuration':CONFIG,'episodes':sum(s['episodes'] for s in shards),
             'wins':sum(s['wins'] for s in shards),'truncated':sum(s['truncated'] for s in shards),
             'decisions':sum(s['decisions'] for s in shards),
             'teacher_action_counts':dict(sum((Counter(s['teacher_action_counts']) for s in shards),Counter())),
             'teacher_plant_packet_counts':dict(sum((Counter(s['teacher_plant_packet_counts']) for s in shards),Counter())),
             'teacher_model_legal_counts':dict(sum((Counter(s['legal_actions']) for s in shards),Counter())),
             'dataset':dataset,'shards':shards,
             'collection_seconds':round(time.time()-started,1)}
    report=OUT/'collection_manifest.json'
    report.write_text(json.dumps(payload,ensure_ascii=False,separators=(',',':'))+'\\n')
    print(json.dumps({'phase':'collection_done','report':str(report),'episodes':payload['episodes'],
                      'wins':payload['wins'],'decisions':payload['decisions'],
                      'teacher_action_counts':payload['teacher_action_counts'],'dataset':dataset,
                      'collection_seconds':payload['collection_seconds']},ensure_ascii=False),flush=True)


if __name__=='__main__': main()
