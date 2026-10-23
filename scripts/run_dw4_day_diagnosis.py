"""Task23 step 5: common-data losses, 64 beta0 games, width128 control and full panel."""
import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import random
import re
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'python')]
from run_dw4_study import OUT as BASE, read, write, sha, verify, record, guard, trainers
from pvz_frame_store import frame_bytes, read_frames

OUT = BASE/'day_diagnosis'
guard.OUT = OUT
MODELS = {'identity': Path(read(BASE/'study_plan.json')['reference_models']['identity']),
          'dw4i_s1': BASE/'dw4i_s1/model.pt', 'dw4i_s2': BASE/'dw4i_s2/model.pt'}

from pvz_recipe import materialize_recipe


def prepare():
    path = OUT/'plan.json'
    if path.exists(): return read(path)
    assert read(BASE/'study_complete.json')['complete']
    original = read(BASE/'study_plan.json'); verify(original)
    inventory = read(BASE/'dataset_inventory.json')
    old = sorted({r['seed'] for r in inventory if r['group'] == 'identity_base'})
    new = sorted({r['seed'] for r in inventory})
    # Actual trainer logs give these seed counts, ruling out zero-row seed discrepancies.
    assert len(old) == 35455 and len(new) == 46000
    for values in (old,new): random.Random(0).shuffle(values)
    common = set(old[:len(old)//10]) & set(new[:len(new)//10])
    selected = []; levels = Counter()
    for r in inventory:
        meta = read(Path(r['source'])/'meta.json'); level = meta.get('context',{}).get('level') or meta['task'].get('level')
        levels[level] += 1
        if r['group'] != 'identity_base' or r['seed'] not in common or level not in (7,8,9,10): continue
        source = Path(r['source'])/'frames.npz'
        first = json.loads(frame_bytes(source).splitlines()[0])
        deck = [p[1] for p in sorted(first['packets']) if p[1] >= 0]
        if deck != [0,1,2,3,4,5,7]: continue
        selected.append(dict(path=str(source), level=level, label=r['label'], seed=r['seed'], frames_sha256=r['frames_sha256']))
    # Include any existing early-day archives; currently training data only covers L7 onwards.
    extras = []
    main = Path('/home/newbiexvwu/PvZ-search/artifacts/student/frames_v2')
    for meta_path in main.glob('*/*/meta.json'):
        meta = read(meta_path); level = meta.get('context',{}).get('level') or meta['task'].get('level')
        source = meta_path.parent/'frames.npz'
        if level not in (1,2,3,4,5,6) or not source.exists(): continue
        seed = meta['task']['seed']
        if seed in set(new) and seed not in common: continue
        entry = dict(path=str(source), level=level, label=meta_path.parent.name, seed=seed, frames_sha256=sha(source))
        extras.append(entry)
    selected += extras
    write(OUT/'historical_selection.json', selected)
    recipe = materialize_recipe(ROOT/'recipes/dw4i.json', OUT/'width128.recipe.json',
                                data=BASE/'training_frames', out=OUT/'width128_s1',
                                cache_dir=OUT/'cache', width=128)
    command = [guard.PYTHON, str(ROOT/'scripts/train_rollout_policy.py'), '--recipe', str(recipe)]
    plan = dict(registered_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), original_plan_sha256=sha(BASE/'study_plan.json'),
                comparison_models={k:str(v) for k,v in MODELS.items()}, comparison_sha256={k:sha(v) for k,v in MODELS.items()},
                historical_selection_sha256=sha(OUT/'historical_selection.json'), historical_archives=len(selected),
                historical_archives_by_level=dict(Counter(r['level'] for r in selected)), existing_l1_l6_archives=len(extras),
                common_validation_seeds=sorted(common), old_seed_count=len(old), new_seed_count=len(new),
                frozen_archive_levels={str(k):v for k,v in levels.items()},
                read_cache='/home/newbiexvwu/PvZ-search/artifacts/student/v2_cache',
                collection_policy='dw4i_s1', collection_levels=[1,3,4,6], collection_seeds=list(range(30000,30016)),
                collection_teacher=dict(samples=3,horizon=2500,min_gap=300,margin=.02,per_card=1,halving=True,backend='cpp'),
                beta=0, workers=4, width128_training_command=command, capacity_seed=1, epochs=12,
                sources={name:sha(ROOT/name) for name in ('scripts/run_dw4_day_diagnosis.py','scripts/dw4_day_losses.py',
                         'scripts/dw4_forward_benchmark.py','scripts/report_dw4_day_diagnosis.py')},
                policy='Fixed s1 diagnostic sample, both original width64 seeds reported; no added labels or tuning for capacity control.')
    write(path,plan)
    target = ROOT/'MEMORY/docs/task23-day-diagnosis-plan.json'; write(target,plan)
    record(f'5诊断登记：旧/新共同留出seed的同一批d7档案{len(selected)}份（逐关计数见计划；L8仅2份），现有L1–L6档案{len(extras)}份。固定dw4i s1在L1/L3/L4/L6各前16个面板seed、beta0采老师标签，共64局，CPU搜索默认3样本/2500tick/300间隔；随后同一48816档案仅改width128、seed1原12epoch，跑完整29项并测相同原生状态单步成本。',target)
    return plan


def check(plan):
    assert sha(BASE/'study_plan.json') == plan['original_plan_sha256']
    verify(read(BASE/'study_plan.json'))
    assert sha(OUT/'historical_selection.json') == plan['historical_selection_sha256']
    for name,digest in plan['sources'].items(): assert sha(ROOT/name) == digest, name
    for name,path in plan['comparison_models'].items(): assert sha(Path(path)) == plan['comparison_sha256'][name]


def run_stage(command, tag, **options):
    result_path = OUT/(tag+'.result.json')
    if result_path.exists():
        assert read(result_path)['returncode'] == 0, 'Preserve and inspect failed attempt before resuming: '+tag
        assert read(OUT/(tag+'.command.json'))['command'] == command
        return read(result_path)
    process = OUT/(tag+'.process.json')
    assert not process.exists() or not guard.live(read(process)['pid']), 'Existing live stage: '+tag
    result = guard.run(command,tag,cwd=ROOT,**options)
    assert result['returncode'] == 0, result
    return result


def losses(plan, selection, tag, plan_path=None):
    command = [guard.PYTHON,str(ROOT/'scripts/dw4_day_losses.py'),'--plan',str(plan_path or OUT/'plan.json'),
               '--selection',str(selection),'--out',str(OUT/(tag+'.json'))]
    run_stage(command,tag)
    result = read(OUT/(tag+'.json'))
    assert result['selection_sha256'] == sha(selection)
    return result


def collect(plan):
    selected = []; totals = {}; panel = BASE/'panel_dw4i_s1'
    for level in plan['collection_levels']:
        tag = f'early_l{level}'; folder = OUT/tag; label = f'dw4diag_l{level}'
        command = [guard.PYTHON,str(ROOT/'scripts/search_play.py'),'--rollout-policy',str(MODELS['dw4i_s1']),
                   '--value','/home/newbiexvwu/PvZ-search/artifacts/value_net/v3/model.pt',
                   '--manifest',str(BASE/'unified_manifest.json'),'--task-id',f'day_l{level}_eval_d7',
                   '--out',str(folder),'--workers','4','--seeds','16','--arms','fast_search',
                   '--samples','3','--horizon','2500','--min-gap','300','--margin','.02','--per-card','1',
                   '--rollout-backend','cpp','--executable',str(ROOT/'build/pvz-portable'),
                   '--resource-dir','/home/newbiexvwu/.cache/pvz-research-resources',
                   '--dagger-beta','0','--frames',str(OUT/'frames'),'--frames-label',label]
        run_stage(command, tag)
        games = [json.loads(line) for line in (folder/'games.jsonl').read_text().splitlines()]
        before = {r['seed']:r for r in (json.loads(line) for line in (panel/f'day_l{level}_eval_d7/games.jsonl').read_text().splitlines())}
        assert len(games) == 16 and {r['seed'] for r in games} == set(plan['collection_seeds'])
        counts = Counter(games=16)
        for game in games:
            path = OUT/f'frames/seed_{game["seed"]}/{label}_fast_search/frames.npz'
            frames = read_frames(path)
            count = Counter(); previous = frames[0]
            for frame in frames[1:]:
                action, teacher = frame['action'],frame['label']
                def key(v): return (v['type'],v.get('packet'),v.get('row'),v.get('col')) if v['type'] != 'wait' else ('wait',)
                disagree = key(action) != key(teacher)
                count['decisions'] += 1; count['disagreements'] += disagree
                count['type_disagreements'] += action['type'] != teacher['type']
                count['teacher_plant'] += teacher['type'] == 'plant'
                count['student_plant'] += action['type'] == 'plant'
                count['rejected'] += bool(frame.get('action_rejected'))
                count['both_plant'] += action['type'] == teacher['type'] == 'plant'
                if action['type'] == teacher['type'] == 'plant':
                    count['packet_disagreements'] += action['packet'] != teacher['packet']
                    count['same_packet_plants'] += action['packet'] == teacher['packet']
                    if action['packet'] == teacher['packet']:
                        count['cell_disagreements'] += (action['row'],action['col']) != (teacher['row'],teacher['col'])
                previous = frame
            assert count['decisions'] == game['decisions'] and count['disagreements'] == game['overrides']
            counts.update(count); counts['searched'] += game['searched']; counts['won'] += game['won']
            counts['panel_outcome_matches'] += game['won'] == before[game['seed']]['won']
            selected.append(dict(path=str(path),level=level,label=path.parent.name,seed=game['seed'],frames_sha256=sha(path)))
        totals[str(level)] = dict(counts)
        record(f'5当前局面标签：L{level}的16局beta0完成，学生/老师分歧{counts["disagreements"]}/{counts["decisions"]}，搜索决策{counts["searched"]}次；与原面板胜负吻合{counts["panel_outcome_matches"]}/16。')
    assert len(selected) == 64
    write(OUT/'current_selection.json',selected); write(OUT/'disagreements.json',totals)


def train(plan):
    destination = OUT/'width128_s1/model.pt'
    if (OUT/'training128_complete.json').exists():
        assert sha(destination) == read(OUT/'training128_complete.json')['model_sha256']; return
    verify(read(BASE/'study_plan.json'),data=True)
    stable = None
    while True:
        foreign = trainers(); available = guard.available()
        if not foreign and available >= 14*1024*1024:
            if stable is None: stable = time.monotonic()
            if time.monotonic()-stable >= 30: break
        else: stable = None
        print(json.dumps(dict(waiting_for_gpu=True,foreign_trainers=foreign,available_kib=available)),flush=True); time.sleep(10)
    check(plan)
    resources = run_stage(plan['width128_training_command'],'training128',extra={'CUDA_VISIBLE_DEVICES':'0'},
                          pause_when=lambda owned: bool(set(trainers())-set(owned)))
    import torch
    net = torch.load(destination,map_location='cpu',weights_only=True)
    original = torch.load(MODELS['dw4i_s1'],map_location='cpu',weights_only=True)
    assert net['width'] == 128 and net['packet_head'] == 'identity' and net['row_pool']
    assert len(net['history']) == 12 and net['labels'] == original['labels']
    assert all(torch.isfinite(v).all() for v in net['state_dict'].values())
    for key in ('pool','context','obs','row_pool','packet_head','label_weight','packet_balance','aux_weight','aux_horizon',
                'soft_weight','spread_weight','spread_cap','lr_schedule','ema','cell_soft_weight'):
        assert net[key] == original[key],key
    # Embedding width follows --width by design; no independent embedding hyperparameter was changed.
    assert net['plant_embedding_dim'] == 128 and original['plant_embedding_dim'] == 64
    assert read(OUT/'width128_s1/source_rows.json') == read(BASE/'dw4i_s1/source_rows.json')
    check(plan)
    write(OUT/'training128_complete.json',dict(model_sha256=sha(destination),model_bytes=destination.stat().st_size,
                                             resources=resources,rows=13643901,epochs=12,seed=1))
    record(f'5容量训练完成：width128 seed1同冻结语料原12epoch结束，实际保留13643901行且来源/训练/验证行数与64宽s1完全一致；耗时{resources["wall_s"]:.1f}s，模型{destination.stat().st_size}字节，随后完整面板与原生单步计时。')


def upload_capacity():
    marker = OUT/'upload128.json'; model = OUT/'width128_s1/model.pt'; digest = sha(model)
    if marker.exists():
        assert read(marker)['model_sha256'] == digest; return
    from import_dw4_transfer import hf_environment
    from huggingface_hub import HfApi, hf_hub_download
    repo = hf_environment(); api = HfApi(); remote = 'transfer/dw4i_w128_s1_model.pt'
    info = api.model_info(repo,files_metadata=True)
    if any(entry.rfilename == remote for entry in info.siblings):
        existing = Path(hf_hub_download(repo,remote,revision=info.sha,local_dir=OUT/'hf_verify'))
        assert sha(existing) == digest, 'Different capacity model already exists; do not overwrite'
    else: api.upload_file(path_or_fileobj=str(model),path_in_repo=remote,repo_id=repo)
    info = api.model_info(repo,files_metadata=True)
    downloaded = Path(hf_hub_download(repo,remote,revision=info.sha,local_dir=OUT/'hf_verify'))
    assert sha(downloaded) == digest
    write(marker,dict(repo=repo,path=remote,revision=info.sha,model_sha256=digest,download_verified=True))
    record('5容量模型HF交付：transfer/dw4i_w128_s1_model.pt按固定远端revision回读SHA核对，检查点不入git。')


def main(phase):
    OUT.mkdir(parents=True,exist_ok=True); plan = prepare(); check(plan)
    if phase == 'prepare': return
    if not (OUT/'historical_losses.json').exists():
        losses(plan,OUT/'historical_selection.json','historical_losses')
        record('5旧分布损失完成：同一批共同留出档案的plant BCE、teacher-plant条件packet/cell CE及准确率已对三个模型逐行统计；不训练、不做模型专属重权，结果留在本树供容量对照后合报。')
    if not (OUT/'disagreements.json').exists(): collect(plan)
    losses(plan,OUT/'current_selection.json','current_losses')
    train(plan); check(plan); upload_capacity()
    capacity_plan = {**plan, 'comparison_models':{'width128':str(OUT/'width128_s1/model.pt')},
                     'comparison_sha256':{'width128':sha(OUT/'width128_s1/model.pt')}}
    write(OUT/'capacity_loss_plan.json',capacity_plan)
    losses(capacity_plan,OUT/'historical_selection.json','capacity_historical_losses',OUT/'capacity_loss_plan.json')
    losses(capacity_plan,OUT/'current_selection.json','capacity_current_losses',OUT/'capacity_loss_plan.json')
    command = [guard.PYTHON,str(ROOT/'scripts/unified_panel.py'),'--recipe',str(ROOT/'recipes/panel_v1.json'),'--rollout-policy',str(OUT/'width128_s1/model.pt'),
               '--out',str(OUT/'panel_width128_s1')]
    # unified_panel itself has the same 2/3GiB guard and full resume contracts.
    log = OUT/'panel_width128_s1.txt'
    if not (OUT/'panel128_complete.json').exists():
        with log.open('a') as stream: subprocess.run(command,cwd=ROOT,env=guard.ENV,stdout=stream,stderr=subprocess.STDOUT,check=True)
        panel = read(OUT/'panel_width128_s1/panel.json')
        assert panel['complete'] and len(panel['tasks']) == 29 and sum(r['games'] for r in panel['tasks']) == 2944
        write(OUT/'panel128_complete.json',dict(panel_sha256=sha(OUT/'panel_width128_s1/panel.json')))
        record('5容量面板完成：width128 seed1统一29项/2944局齐，未更改任务或挑选子集；随后核对单步计时并合报。')
    command = [guard.PYTHON,str(ROOT/'scripts/dw4_forward_benchmark.py'),'--narrow',str(MODELS['dw4i_s1']),
               '--wide',str(OUT/'width128_s1/model.pt'),'--out',str(OUT/'forward_benchmark.json')]
    run_stage(command,'forward_benchmark')
    command = [guard.PYTHON,str(ROOT/'scripts/report_dw4_day_diagnosis.py')]
    with (OUT/'report.txt').open('a') as stream: subprocess.run(command,cwd=ROOT,env=guard.ENV,stdout=stream,stderr=subprocess.STDOUT,check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('prepare','all'),required=True)
    main(parser.parse_args().phase)
