"""Task27's gated, frozen-corpus, width64 seed1 training and v1/v2 evaluation."""
import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path
import re
import subprocess
import time
from zoneinfo import ZoneInfo
from task27_common import ROOT, MAIN, OUT, PYTHON, guard, trainers, read, write, sha, record, run_stage
from import_task27_mac import refresh
from pvz_recipe import materialize_recipe

OLD = MAIN/'.claude/worktrees/dw4/artifacts/task23'


def completed(log):
    pattern = re.compile(r'\s(\S+) \(dw4i_s1, ([^,]+), offset (\d+), (\d+) workers\): (\d+)/(\d+)')
    return {m[1]+'_fast_search':dict(task=m[2],offset=int(m[3]),games=int(m[6]))
            for m in pattern.finditer(log)}


def round_labels(number):
    return ([f'n{n}t{number}_fast_search' for n in (12,14,17,18,19)] +
            [f'd{n}dag{number}d_fast_search' for n in (1,3,4,6)] +
            [f'p{n}dag{number+1}_fast_search' for n in (21,22,23,24,26,27,28,29)] +
            [f'd{n}p{number+6}d_fast_search' for n in (7,8,9,10)])


def wait_gate():
    while True:
        log = Path('/home/newbiexvwu/pvz-logs/queue103.log').read_text()
        done = completed(log)
        missing = sorted(set(round_labels(1)+round_labels(2))-done.keys())
        if not missing:
            assert 'd10p8d_fast_search' in done
            return
        print(dict(waiting_for='q103 full round2',missing=missing),flush=True)
        time.sleep(30)


def freeze():
    path = OUT/'study_plan.json'
    if path.exists(): return read(path)
    wait_gate()
    refresh()
    previous = read(ROOT/'MEMORY/docs/task23-study-plan.json')
    assert sha(OLD/'study_plan.json') == sha(ROOT/'MEMORY/docs/task23-study-plan.json')
    original = read(OLD/'dataset_inventory.json')
    assert sha(OLD/'dataset_inventory.json') == previous['inventory_sha256']
    assert len(original) == previous['archives'] == 48816
    logs = {q:Path(f'/home/newbiexvwu/pvz-logs/queue{q}.log').read_text() for q in (102,103)}
    assert 'QUEUE102 DONE' in logs[102]
    q102, q103 = (completed(logs[q]) for q in (102,103))
    full_rounds = [n for n in (1,2,3,4) if set(round_labels(n)) <= q103.keys()]
    assert 1 in full_rounds and 2 in full_rounds
    desktop = {**q102,**{label:q103[label] for n in full_rounds for label in round_labels(n)}}
    mac = read(OUT/'mac_import.json')
    mac_labels = {r['label']+'_fast_search' for r in mac['rounds']
                  if not r.get('already_present_old_round') and not r['label'].startswith(('d9dag','d10dag'))}
    # Old Mac daytime/high-difficulty rounds only expand the recipe if not imported before task27.
    mac_labels.update(mac['introduced_labels'])
    extra_labels = set(desktop) | mac_labels
    candidates = {r['archive']:(Path(r['source']), 'dw4i_frozen',r) for r in original}
    for source in (MAIN/'artifacts/student/frames_v2').glob('*/*/frames.npz'):
        if source.parent.name not in extra_labels or not (source.parent/'meta.json').is_file(): continue
        key='/'.join(source.parts[-3:-1])
        if key in candidates:
            assert sha(source) == candidates[key][2]['frames_sha256']
            continue
        group = 'q102' if source.parent.name in q102 else 'q103' if source.parent.name in desktop else 'mac_new'
        candidates[key] = (source.parent,group,None)
    counts = Counter(folder.name for folder,_,_ in candidates.values())
    for label,count in previous['archive_counts'].items(): assert counts[label] >= count, label
    for label,task in desktop.items(): assert counts[label] == task['games'], (label,counts[label],task)
    assert extra_labels <= counts.keys(), extra_labels-counts.keys()
    taskmaps = {}
    for manifest in (MAIN/'artifacts/stage4/night_panel.json',MAIN/'artifacts/stage4/day_panel.json',
                     MAIN/'artifacts/stage4/pool_panel.json',
                     MAIN/'.claude/worktrees/identity/artifacts/stage4/day_panel_identity.json'):
        taskmaps.update({t['task_id']:t for t in read(manifest)['tasks']})
    expected = {label:set(taskmaps[row['task']]['seeds'][row['offset']:row['offset']+row['games']])
                for label,row in desktop.items()}
    actual = {label:set() for label in desktop}
    held = {tuple(sorted(t['deck'])) for t in read(OUT/'unified_v2.json')['tasks'] if '_holdout' in t['task_id']}
    inventory = [];mirror = OUT/'training_frames';mirror.mkdir(exist_ok=True)
    for i,(key,(folder,group,parent)) in enumerate(sorted(candidates.items())):
        frame,meta = folder/'frames.npz',folder/'meta.json'
        hashes = dict(frames_sha256=sha(frame),meta_sha256=sha(meta))
        if parent:
            assert all(hashes[k] == parent[k] for k in hashes), str(folder)
        metadata = read(meta);seed=int(folder.parent.name.split('_',1)[1])
        assert metadata['task']['seed'] == seed
        assert not 30000 <= seed <= 31023, ('evaluation seed leaked',key)
        label=folder.name
        if label in actual: actual[label].add(seed)
        if metadata['task'].get('deck_perturb'):
            assert tuple(sorted(metadata['task']['deck'])) not in held, key
        row=dict(archive=key,source=str(folder),label=label,seed=seed,group=group,**hashes)
        inventory.append(row)
        target=mirror/key;target.mkdir(parents=True,exist_ok=True)
        for source in (frame,meta):
            link=target/source.name
            if link.exists(): assert link.is_symlink() and link.resolve() == source.resolve()
            else: link.symlink_to(source.resolve())
        if i%4000 == 0: print(dict(freezing=i,total=len(candidates)),flush=True)
    assert actual == expected
    write(OUT/'dataset_inventory.json',inventory)
    labels = previous['labels'] + sorted(set(counts)-set(previous['labels']))
    recipe = materialize_recipe(ROOT/'recipes/dw5i.json', OUT/'recipes/dw5i_s1.json',
                                data=str(mirror), labels=labels, out=str(OUT/'dw5i_s1'),
                                cache_dir=str(OUT/'cache'), source_row_report=True)
    command = [PYTHON, str(ROOT/'scripts/train_rollout_policy.py'), '--recipe', str(recipe)]
    source_paths = ['scripts/run_dw5_study.py','scripts/import_task27_mac.py','scripts/report_dw5_study.py',
                    'scripts/train_rollout_policy.py','python/pvz_frame_store.py','python/pvz_rollout_policy.py',
                    'python/pvz_training_rows.py','python/pvz_student_observation.py','scripts/check_capacity_native.py']
    from run_task27_pretraining import plan as evaluation_plan, verify as verify_evaluation
    evaluation = evaluation_plan();verify_evaluation(evaluation)
    data = dict(frozen_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                previous_plan_sha256=sha(OLD/'study_plan.json'),pretraining_plan_sha256=sha(OUT/'pretraining_plan.json'),
                inventory_sha256=sha(OUT/'dataset_inventory.json'),archives=len(inventory),labels=labels,
                archive_counts=dict(counts),group_archives=dict(Counter(r['group'] for r in inventory)),
                q103_full_rounds=full_rounds,q103_labels=sorted(set(desktop)-q102.keys()),q102_labels=sorted(q102),
                queue_logs_sha256={str(q):__import__('hashlib').sha256(text.encode()).hexdigest() for q,text in logs.items()},
                mac_import_sha256=sha(OUT/'mac_import.json'),mac_revision=mac['revision'],training_command=command,recipe_sha256=sha(recipe),
                sources={p:sha(ROOT/p) for p in source_paths},seed=1,width=64,epochs=12,
                policy='All requested sources; no win filtering, no best seed/epoch selection, no default promotion.')
    write(path,data)
    for q,text in logs.items(): (OUT/f'queue{q}_freeze.txt').write_text(text)
    target=ROOT/'MEMORY/docs/task27-study-plan.json';write(target,data)
    record(f'4语料冻结：q103完整第{full_rounds}轮（含第2轮d10p8d）已核对；原dw4i全部48816档案保留，加入q102全部/指定q103完整轮次/已上传Mac缺失轮次，共{len(inventory)}档案/{len(labels)}标签，逐文件SHA冻结。64宽seed1原12epoch配方，未加入评估局。',target)
    return data


def verify(data, files=False):
    from run_task27_pretraining import verify as verify_evaluation
    verify_evaluation(read(OUT/'pretraining_plan.json'))
    assert sha(OUT/'pretraining_plan.json') == data['pretraining_plan_sha256']
    assert sha(OUT/'dataset_inventory.json') == data['inventory_sha256']
    assert sha(OUT/'mac_import.json') == data['mac_import_sha256']
    for path,digest in data['sources'].items(): assert sha(ROOT/path) == digest, path
    if files:
        for row in read(OUT/'dataset_inventory.json'):
            for key,name in (('frames','frames.npz'),('meta','meta.json')):
                assert sha(OUT/'training_frames'/row['archive']/name) == row[key+'_sha256']


def train(data):
    marker=OUT/'training_complete.json'
    if marker.exists():
        assert read(marker)['model_sha256'] == sha(OUT/'dw5i_s1/model.pt');return
    verify(data,files=True)
    stable=None
    probe=subprocess.run(['pgrep','-f','train_rollout_policy'],stdout=subprocess.PIPE,text=True)
    print(dict(pgrep_train_rollout_policy=probe.stdout.split(),returncode=probe.returncode),flush=True)
    while True:
        busy=trainers();available=guard.available()
        if not busy and available >= 14*1024*1024:
            if stable is None: stable=time.monotonic()
            if time.monotonic()-stable >= 30:break
        else:stable=None
        print(dict(waiting_for='training admission',foreign_trainers=busy,available_kib=available),flush=True)
        time.sleep(10)
    result=run_stage(data['training_command'],'training_s1',extra={'CUDA_VISIBLE_DEVICES':'0'},
                     pause_when=lambda owned:bool(set(trainers())-set(owned)))
    import torch
    model=OUT/'dw5i_s1/model.pt';checkpoint=torch.load(model,map_location='cpu',weights_only=True)
    metrics=read(OUT/'dw5i_s1/metrics.json');rows=read(OUT/'dw5i_s1/source_rows.json')
    assert checkpoint['packet_head']=='identity' and checkpoint['row_pool']
    assert checkpoint['state_dict']['trunk.0.weight'].shape[0] == 64
    assert checkpoint['labels'].split(',') == data['labels']
    assert checkpoint['history'] == metrics['history'] and [h['epoch'] for h in checkpoint['history']] == list(range(1,13))
    assert all(torch.isfinite(v).all() for v in checkpoint['state_dict'].values())
    assert rows['rows'] == metrics['data_memory']['rows'] == sum(r['rows'] for r in rows['sources'].values())
    assert {label:r['archives'] for label,r in rows['sources'].items()} == data['archive_counts']
    verify(data)
    write(marker,dict(complete=True,model_sha256=sha(model),model_bytes=model.stat().st_size,
                     rows=rows['rows'],resources=result,source_rows_sha256=sha(OUT/'dw5i_s1/source_rows.json')))
    record(f'4训练完成：dw5i_s1宽64、identity+row_pool，原12epoch自然结束，{rows["rows"]}保留行，耗时{result["wall_s"]:.1f}s，最低可用{result["minimum_available_kib"]/1024**2:.2f}GiB；实际来源行数齐，模型随后HF交付与两版面板。')


def upload():
    marker=OUT/'upload.json';model=OUT/'dw5i_s1/model.pt';digest=sha(model)
    if marker.exists(): assert read(marker)['model_sha256']==digest;return
    from import_dw4_transfer import hf_environment
    from huggingface_hub import HfApi,hf_hub_download
    repo=hf_environment();api=HfApi();name='transfer/dw5i_s1_model.pt'
    info=api.model_info(repo,files_metadata=True)
    if any(e.rfilename==name for e in info.siblings):
        existing=Path(hf_hub_download(repo,name,revision=info.sha,local_dir=OUT/'hf_verify'))
        assert sha(existing)==digest,'different checkpoint already at requested name'
    else:api.upload_file(path_or_fileobj=str(model),path_in_repo=name,repo_id=repo)
    info=api.model_info(repo,files_metadata=True)
    downloaded=Path(hf_hub_download(repo,name,revision=info.sha,local_dir=OUT/'hf_verify'))
    assert sha(downloaded)==digest
    write(marker,dict(repo=repo,path=name,revision=info.sha,model_sha256=digest,download_verified=True))
    record('4模型交付：HF transfer/dw5i_s1_model.pt上传后固定revision回读SHA一致；模型不入git。')


def main():
    data=freeze();verify(data);train(data);upload()
    # Wait on the exact reference controller; never compete with its four game slots.
    while not (OUT/'pretraining_complete.json').exists():
        pid=read(OUT/'pretraining_controller.process.json')['pid']
        assert guard.live(pid),'reference controller stopped; inspect before resuming'
        print(dict(waiting_for='references',pid=pid),flush=True);time.sleep(30)
    command=[PYTHON,str(ROOT/'scripts/check_capacity_native.py'),'--model',str(OUT/'dw5i_s1/model.pt'),
             '--out',str(OUT/'native_parity.json'),'--manifest',str(OUT/'unified_v2.json'),
             '--tasks','day_l9_eval_d7,night_l12_eval_n10,day_l10_orig_eval,pool_l21_eval_p9k']
    run_stage(command,'native_parity');assert read(OUT/'native_parity.json')['passed']
    for version in ('v2','v1'):
        recipe = materialize_recipe(ROOT/f'recipes/panel_{version}.json', OUT/f'recipes/panel_dw5i_{version}.json',
                                    version=version, manifest=str(OUT/f'unified_{version}.json'),
                                    rollout_policy=str(OUT/'dw5i_s1/model.pt'), out=str(OUT/f'panel_dw5i_{version}'))
        command = [PYTHON, str(ROOT/'scripts/unified_panel.py'), '--recipe', str(recipe)]
        run_stage(command,'panel_dw5i_'+version);verify(data)
        marker=OUT/f'panel_{version}_complete.json'
        if not marker.exists():
            panel=read(OUT/f'panel_dw5i_{version}/panel.json')
            assert panel['complete'] and len(panel['tasks'])==(34 if version=='v2' else 29)
            write(marker,dict(complete=True,panel_sha256=sha(OUT/f'panel_dw5i_{version}/panel.json')))
            record(f'4学生{version}面板完成：{len(panel["tasks"])}任务自然结束；完整核验与对照报告随后合报。')
    command=[PYTHON,str(ROOT/'scripts/report_dw5_study.py')]
    with (OUT/'report.txt').open('a') as stream:
        subprocess.run(command,cwd=ROOT,env=guard.ENV,stdout=stream,stderr=subprocess.STDOUT,check=True)


if __name__=='__main__':main()
