"""Task23: freeze the requested data after q101 round3, train both seeds, upload and evaluate."""
import argparse
from collections import Counter
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
MAIN = Path('/home/newbiexvwu/PvZ-search')
OUT = ROOT/'artifacts/task23'
sys.path[:0] = [str(ROOT/'python'),str(ROOT/'scripts')]
import dagger_experiment_runner as guard
from run_identity_validation import descendants, trainers
from import_mac_labels import hf_environment, sha
from unified_panel import read, write

guard.OUT = OUT
guard.descendants = descendants
guard.ENV = {**os.environ,'PYTHONPATH':str(ROOT/'python')+':'+str(ROOT/'scripts'),
             'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MALLOC_ARENA_MAX':'2',
             'CUDA_VISIBLE_DEVICES':'','PVZ_EXECUTABLE':str(ROOT/'build/pvz-portable'),
             'PVZ_RESOURCE_DIR':'/home/newbiexvwu/.cache/pvz-research-resources'}


def record(text, *documents):
    today = datetime.now(ZoneInfo('Asia/Shanghai')).date().isoformat()
    book = ROOT/'MEMORY/docs/task23-dw4i-unified-panel.md'
    before, after = book.read_text().split('\n## 问题',1)
    book.write_text(before+'\n- '+today+'，'+text+'\n\n## 问题'+after)
    daily = ROOT/f'MEMORY/{today}.md'
    with daily.open('a') as stream: stream.write('- task23：'+text+'\n')
    assert subprocess.check_output(['git','-C',str(ROOT),'branch','--show-current']).strip()==b'dw4'
    assert not subprocess.check_output(['git','-C',str(ROOT),'diff','--cached','--name-only']).strip()
    paths = [book,daily,*documents]
    assert all(p.suffix not in ('.pt','.npz','.log') and p.stat().st_size<1_000_000 for p in paths)
    subprocess.run(['git','-C',str(ROOT),'add',*[str(p) for p in paths]],check=True)
    stats=subprocess.check_output(['git','-C',str(ROOT),'diff','--cached','--numstat'])
    assert all(line.split(b'\t')[0]!=b'-' for line in stats.splitlines())
    print(stats.decode()+subprocess.check_output(['git','-C',str(ROOT),'diff','--cached','--stat']).decode(),flush=True)
    tracked=subprocess.check_output(['git','-C',str(ROOT),'ls-files','-z','artifacts','logs'])
    ignored=subprocess.run(['git','-C',str(ROOT),'check-ignore','-z','--stdin','--no-index'],input=tracked,stdout=subprocess.PIPE)
    assert ignored.returncode in (0,1) and not ignored.stdout
    subprocess.run(['git','-C',str(ROOT),'commit','-m','task23: '+text.split('：')[0][:90]],check=True)
    subprocess.run(['git','-C',str(ROOT),'push','origin','dw4'],check=True)


def completed_queue_labels():
    text=Path('/home/newbiexvwu/pvz-logs/queue101.log').read_text()
    return set(re.findall(r'\s((?:n\d+dag\d+|d\d+p\d+d)) \(identity,[^\n]+: \d+/64\s*$',text,re.M))


def gate():
    required={f'n{level}dag{r}' for r in (1,2,3) for level in (11,12,13,14,16,17,18,19)}
    required|={f'd{level}p{r}d' for r in (1,2,3) for level in (7,8,9,10)}
    while True:
        missing=required-completed_queue_labels()
        path=OUT/'baseline_controller.process.json'
        baseline_busy=path.exists() and guard.live(read(path)['pid'])
        if not baseline_busy:
            for name in ('v34','identity'):
                panel=OUT/f'baseline_{name}/panel.json'
                assert panel.exists() and read(panel)['complete'] and len(read(panel)['tasks'])==29, 'baseline stopped incomplete; inspect its existing controller output'
        if not missing and not baseline_busy: return sorted(required)
        print(json.dumps(dict(waiting_for_q101=sorted(missing),baseline_controller_live=baseline_busy)),flush=True)
        time.sleep(30)


def source_group(label, base):
    if label in base:return 'identity_base'
    if re.fullmatch(r'd(?:7|8|9|10)dag2[1-8]_fast_search',label):return 'queue100'
    if re.fullmatch(r'd(?:9|10)dag4b_fast_search',label):return 'mac_v33'
    if re.fullmatch(r'd(?:9|10)dag5b_fast_search',label):return 'mac_v34'
    if re.fullmatch(r'd(?:7|8|9|10)p\d+m_fast_search',label):return 'perturb_mac'
    if re.fullmatch(r'd(?:7|8|9|10)p\d+d_fast_search',label):return 'perturb_desktop'
    if re.fullmatch(r'n\d+dag\d+_fast_search',label):return 'night'
    if label in ('d10orig1_fast_search','d10orig2_fast_search'):return 'original'


def freeze():
    path=OUT/'study_plan.json'
    if path.exists():return read(path)
    gate_labels=gate()
    checks=OUT/'training_report_checks.txt'
    assert 'Ran 5 tests' in checks.read_text() and checks.read_text().rstrip().endswith('OK')
    assert read(OUT/'training_yield_check.json')['paused_when_external_busy']
    with (OUT/'mac_refresh.txt').open('w') as log:
        subprocess.run([guard.PYTHON,str(ROOT/'scripts/import_mac_labels.py'),'--report',str(OUT/'mac_import.json')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    old=read(ROOT/'MEMORY/docs/task19-validation-amended-plan.json');base=old['labels']
    candidates={}
    for source in (MAIN/'artifacts/student/frames_v2',MAIN/'.claude/worktrees/conveyor/artifacts/task20/frames'):
        for frames in source.glob('*/*/frames.npz'):
            label=frames.parent.name
            group=source_group(label,base)
            if group is None or not (frames.parent/'meta.json').is_file():continue
            key='/'.join(frames.parts[-3:-1])
            if key in candidates:
                assert sha(frames)==sha(candidates[key][0])
                assert sha(frames.parent/'meta.json')==sha(candidates[key][0].parent/'meta.json')
            else:candidates[key]=(frames,group)
    counts=Counter(frames.parent.name for frames,_ in candidates.values())
    assert all(counts[label]==n for label,n in old['archive_counts'].items())
    assert counts['d10orig1_fast_search']==counts['d10orig2_fast_search']==128
    assert all(counts[label+'_fast_search']==64 for label in gate_labels)
    assert all(counts[label+'_fast_search']==640 for label in ('d9dag4b','d10dag4b','d9dag5b','d10dag5b'))
    held={tuple(sorted(t['deck'])) for t in read(OUT/'unified_manifest.json')['tasks'] if '_holdout' in t['task_id']}
    identity_hash=sha(MAIN/'.claude/worktrees/identity/artifacts/task19/identity_dw3_s1/model.pt')
    inventory=[];mirror=OUT/'training_frames';mirror.mkdir(exist_ok=True)
    for index,(key,(frames,group)) in enumerate(sorted(candidates.items())):
        meta=frames.parent/'meta.json';task=read(meta)['task'];seed=int(frames.parent.parent.name.split('_')[-1])
        assert task['seed']==seed
        if group.startswith('perturb'):
            assert tuple(sorted(task['deck'])) not in held
            assert task['checkpoint_sha256']==identity_hash
        if group=='original':assert task['deck']==[] and task['conveyor_start']=='original'
        row=dict(archive=key,source=str(frames.parent),label=frames.parent.name,group=group,seed=seed,
                 frames_sha256=sha(frames),meta_sha256=sha(meta))
        destination=mirror/key;destination.mkdir(parents=True,exist_ok=True)
        for original in (frames,meta):
            target=destination/original.name
            if target.exists():assert target.is_symlink() and target.resolve()==original.resolve()
            else:target.symlink_to(original.resolve())
        inventory.append(row)
        if index%2000==0:print(json.dumps(dict(freezing_archives=index,total=len(candidates))),flush=True)
    write(OUT/'dataset_inventory.json',inventory)
    labels=base+sorted(set(counts)-set(base))
    models=dict(identity=str(MAIN/'.claude/worktrees/identity/artifacts/task19/identity_dw3_s1/model.pt'),v34=str(MAIN/'artifacts/student/v34/model.pt'))
    sources=['scripts/run_dw4_study.py','scripts/report_dw4_study.py','scripts/import_mac_labels.py',
             'scripts/unified_panel.py','scripts/train_rollout_policy.py',
             'scripts/dagger_experiment_runner.py','scripts/run_identity_validation.py',
             'python/pvz_rollout_policy.py','python/pvz_frame_store.py','python/pvz_student_observation.py',
             'python/pvz_value_net.py','python/pvz_training_rows.py','scripts/search_play.py',
             'python/pvz_env.py','python/pvz_event_env.py','scripts/render_episode.py',
             'scripts/pvz_tasks.py','scripts/pvz_constants.py','src/EnvironmentRollout.cpp',
             'src/EnvironmentRollout.h','src/LawnApp.cpp','src/LawnApp.h','src/main.cpp']
    commands=[]
    for seed in (1,2):
        command=list(old['command']);command[1]=str(ROOT/'scripts/train_rollout_policy.py')
        changes={'--data':str(mirror),'--labels':','.join(labels),'--out':str(OUT/f'dw4i_s{seed}'),
                 '--cache-dir':str(OUT/'cache'),'--read-cache-dir':str(MAIN/'artifacts/student/v2_cache'),'--seed':str(seed)}
        for flag,value in changes.items():command[command.index(flag)+1]=value
        command+=['--source-row-report',str(OUT/f'dw4i_s{seed}/source_rows.json')]
        assert '--init' not in command and '--max-steps' not in command
        commands.append(command)
    plan=dict(frozen_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),q101_completed_labels=gate_labels,
              inventory_sha256=sha(OUT/'dataset_inventory.json'),archives=len(inventory),labels=labels,archive_counts=dict(counts),
              groups={label:source_group(label,base) for label in labels},training_commands=commands,
              sources={name:sha(ROOT/name) for name in sources},executable_sha256=sha(ROOT/'build/pvz-portable'),
              manifest_sha256=sha(OUT/'unified_manifest.json'),reference_models=models,reference_sha256={k:sha(Path(v)) for k,v in models.items()},
              mac_import=read(OUT/'mac_import.json'),missing_queue100=[f'd{l}dag{r}' for l in (7,8,9,10) for r in range(21,29) if not counts[f'd{l}dag{r}_fast_search']],
              training_seeds=[1,2],epochs=12,workers=4,panel_games_per_model=2944,
              engineering=dict(training_checks=5,training_checks_sha256=sha(checks),
                               panel_smoke_sha256=sha(ROOT/'MEMORY/docs/task23-panel-smoke.json'),
                               training_yield=read(OUT/'training_yield_check.json')),
              interpretation='one fixed corpus for both fresh seeds, final epoch; no model/default promotion')
    for name in ('v34','identity'):
        panel=read(OUT/f'baseline_{name}/panel.json')
        assert panel['model_sha256']==plan['reference_sha256'][name]
        assert panel['executable_sha256']==plan['executable_sha256'] and panel['manifest_sha256']==plan['manifest_sha256']
    write(path,plan);copy=ROOT/'MEMORY/docs/task23-study-plan.json';write(copy,plan)
    record(f'3数据冻结：queue101完整3轮已核对日志，合训{len(inventory)}档案/{len(labels)}标签，两个seed共享逐文件SHA清单；新增来源不丢输局、不择样本。Mac按冻结时已上传版本刷新，缺失轮次/标签在计划中明确列出；训练保留原12epoch/权重/row_pool/identity配方，仅增加行数统计。',copy)
    return plan


def verify(plan, data=False):
    assert sha(OUT/'unified_manifest.json')==plan['manifest_sha256']
    assert sha(ROOT/'build/pvz-portable')==plan['executable_sha256']
    for name,digest in plan['sources'].items():assert sha(ROOT/name)==digest,name
    for name,path in plan['reference_models'].items():assert sha(Path(path))==plan['reference_sha256'][name]
    assert sha(OUT/'dataset_inventory.json')==plan['inventory_sha256']
    if data:
        for row in read(OUT/'dataset_inventory.json'):
            for name in ('frames','meta'):
                path=OUT/'training_frames'/row['archive']/(name+('.npz' if name=='frames' else '.json'))
                assert sha(path)==row[name+'_sha256'],str(path)


def train(plan,seed):
    marker=OUT/f'training_s{seed}_complete.json'
    if marker.exists():
        assert sha(OUT/f'dw4i_s{seed}/model.pt')==read(marker)['model_sha256'];return
    verify(plan,data=True)
    stable=None
    while True:
        busy=trainers();available=guard.available()
        if not busy and available>=14*1024*1024:
            if stable is None:stable=time.monotonic()
            if time.monotonic()-stable>=30:break
        else:stable=None
        print(json.dumps(dict(training_seed_wait=seed,foreign_trainers=busy,available_kib=available)),flush=True);time.sleep(10)
    verify(plan)
    tag=f'training_s{seed}';process=OUT/(tag+'.process.json');result_path=OUT/(tag+'.result.json')
    assert not process.exists() or not guard.live(read(process)['pid']), 'existing live training'
    assert not result_path.exists(), 'inspect and preserve the previous attempt before retrying training'
    result=guard.run(plan['training_commands'][seed-1],tag,cwd=ROOT,extra={'CUDA_VISIBLE_DEVICES':'0'},
                     pause_when=lambda owned: bool(set(trainers())-set(owned)))
    assert result['returncode']==0,result
    import torch
    model=torch.load(OUT/f'dw4i_s{seed}/model.pt',map_location='cpu',weights_only=True)
    assert model['packet_head']=='identity' and model['row_pool'] and len(model['history'])==12
    assert model['labels'].split(',')==plan['labels']
    assert all(torch.isfinite(v).all() for v in model['state_dict'].values())
    rows=read(OUT/f'dw4i_s{seed}/source_rows.json');metrics=read(OUT/f'dw4i_s{seed}/metrics.json')
    assert rows['rows']==metrics['data_memory']['rows']==sum(r['rows'] for r in rows['sources'].values())
    assert {k:v['archives'] for k,v in rows['sources'].items()}==plan['archive_counts']
    verify(plan)
    write(marker,dict(model_sha256=sha(OUT/f'dw4i_s{seed}/model.pt'),resources=result,rows=rows['rows'],
                      source_rows_sha256=sha(OUT/f'dw4i_s{seed}/source_rows.json')))
    record(f'3种子{seed}训练完成：原配方12epoch自然结束，{rows["rows"]}实际保留行，各标签训练/验证行数已记录；模型SHA{read(marker)["model_sha256"]}。耗时{result["wall_s"]:.1f}s、子树RSS求和峰{result["peak_rss_kib"]}KiB（共享页会重复）、可用最低{result["minimum_available_kib"]}KiB，不将RSS求和当物理内存用量。')


def upload(seed):
    marker=OUT/f'upload_s{seed}.json'
    model=OUT/f'dw4i_s{seed}/model.pt';digest=sha(model)
    if marker.exists():assert read(marker)['model_sha256']==digest;return
    from huggingface_hub import HfApi,hf_hub_download
    repo=hf_environment();api=HfApi();path=f'transfer/dw4i_s{seed}_model.pt'
    info=api.model_info(repo,files_metadata=True);existing=next((e for e in info.siblings if e.rfilename==path),None)
    if existing:
        cached=Path(hf_hub_download(repo,path,revision=info.sha,local_dir=OUT/'uploaded_verification'))
        assert sha(cached)==digest,'different model already exists at requested remote name'
    else:api.upload_file(path_or_fileobj=str(model),path_in_repo=path,repo_id=repo)
    info=api.model_info(repo,files_metadata=True)
    cached=Path(hf_hub_download(repo,path,revision=info.sha,local_dir=OUT/'uploaded_verification'))
    assert sha(cached)==digest
    write(marker,dict(repo=repo,path=path,revision=info.sha,model_sha256=digest,download_verified=True))
    record(f'3种子{seed}模型HF交付：{path}上传后按远端revision回读，SHA与本地训练检查点一致；模型不入git。')


def evaluate(plan,seed):
    verify(plan)
    marker=OUT/f'panel_s{seed}_complete.json'
    if marker.exists():
        assert read(marker)['panel_sha256']==sha(OUT/f'panel_dw4i_s{seed}/panel.json');return
    command=[guard.PYTHON,str(ROOT/'scripts/unified_panel.py'),'--rollout-policy',str(OUT/f'dw4i_s{seed}/model.pt'),
             '--out',str(OUT/f'panel_dw4i_s{seed}')]
    with (OUT/f'panel_dw4i_s{seed}.txt').open('a') as log:
        subprocess.run(command,cwd=ROOT,env=guard.ENV,stdout=log,stderr=subprocess.STDOUT,check=True)
    panel=read(OUT/f'panel_dw4i_s{seed}/panel.json');assert panel['complete'] and len(panel['tasks'])==29
    assert sum(r['games'] for r in panel['tasks'])==2944
    verify(plan)
    record(f'4种子{seed}统一面板完成：29项、2944局均已写出，完整三组汇总与逐项对照将在两种子齐后核验。')
    write(marker,dict(panel_sha256=sha(OUT/f'panel_dw4i_s{seed}/panel.json')))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('all','freeze','train','evaluate'),required=True)
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True);plan=freeze();verify(plan)
    if args.phase in ('all','train'):
        for seed in (1,2):train(plan,seed);upload(seed)
    if args.phase in ('all','evaluate'):
        for seed in (1,2):evaluate(plan,seed)
        subprocess.run([guard.PYTHON,str(ROOT/'scripts/report_dw4_study.py')],cwd=ROOT,env=guard.ENV,check=True)
