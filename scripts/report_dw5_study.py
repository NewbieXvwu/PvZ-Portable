"""Audit every task27 deliverable against raw records, then write the unified comparison."""
from collections import Counter, defaultdict
import json
from pathlib import Path
from task27_common import ROOT, MAIN, OUT, read, write, sha, record
from run_dw5_study import verify
from unified_panel import summarize


def audit_panel(folder, version, model_hash):
    manifest=read(OUT/f'unified_{version}.json');panel=read(folder/'panel.json')
    assert panel['complete'] and panel['model_sha256']==model_hash
    assert panel['executable_sha256']==sha(ROOT/'build/pvz-portable')
    assert panel['manifest_sha256']==sha(OUT/f'unified_{version}.json')
    assert len(panel['tasks'])==(34 if version=='v2' else 29)
    for task in manifest['tasks']:
        name=task['task_id'];group=manifest['groups'][name];n=64 if group in ('heldout','pool') else 128
        actual=summarize(folder/name,task,task['seeds'][:n],plant_types=True);actual['group']=group
        assert actual==next(row for row in panel['tasks'] if row['task_id']==name)
        assert read(folder/'resources'/(name+'.result.json'))['returncode']==0
        config=read(folder/name/'config.json')
        assert config['arms']=='fast' and config['workers']=='4'
        contract=read(folder/'resources'/(name+'.contract.json'))
        assert contract['model_sha256']==model_hash and contract['executable_sha256']==panel['executable_sha256']
        assert contract['manifest_sha256']==panel['manifest_sha256'] and contract['seeds']==task['seeds'][:n]
    for group,stats in panel['groups'].items():
        part=[r for r in panel['tasks'] if r['group']==group]
        assert stats['games']==sum(r['games'] for r in part) and stats['won']==sum(r['won'] for r in part)
        assert abs(stats['mean_win_rate']-sum(r['win_rate'] for r in part)/len(part))<1e-12
        counts=Counter()
        for row in part:counts.update(row['planted_by_type_total'])
        assert dict(counts)==stats['planted_by_type_total']
    assert sum(t['games'] for t in panel['tasks'])==(3264 if version=='v2' else 2944)
    return panel


def main():
    plan=read(OUT/'study_plan.json');verify(plan,files=True)
    engineering=read(ROOT/'MEMORY/docs/task27-panel-engineering.json')
    assert engineering['complete'] and engineering['tests']==24 and engineering['v1_matches_task23']
    zones=read(ROOT/'MEMORY/docs/task27-zone-comparison.json')
    assert zones['passed'] and [c['tasks'] for c in zones['comparisons']]==[54,9,35]
    assert all(c['passed'] and not c['changed'] and c['task_order_equal'] for c in zones['comparisons'])
    # Verify shared manifests still have the exact contents that the comparison audited.
    for comparison in zones['comparisons']:
        assert sha(Path(comparison['current']))==comparison['current_sha256']
        assert sha(Path(comparison['generated']))==comparison['generated_sha256']
    training=read(OUT/'training_complete.json');model=OUT/'dw5i_s1/model.pt'
    assert training['complete'] and training['model_sha256']==sha(model)
    assert training['resources']==read(OUT/'training_s1.result.json') and training['resources']['returncode']==0
    assert read(OUT/'training_s1.command.json')['command']==plan['training_command']
    rows=read(OUT/'dw5i_s1/source_rows.json');metrics=read(OUT/'dw5i_s1/metrics.json')
    assert sha(OUT/'dw5i_s1/source_rows.json')==training['source_rows_sha256']
    assert rows['rows']==training['rows']==metrics['data_memory']['rows']
    assert {k:v['archives'] for k,v in rows['sources'].items()}==plan['archive_counts']
    import torch
    checkpoint=torch.load(model,map_location='cpu',weights_only=True)
    assert checkpoint['history']==metrics['history'] and [h['epoch'] for h in metrics['history']]==list(range(1,13))
    assert checkpoint['packet_head']=='identity' and checkpoint['row_pool']
    assert checkpoint['state_dict']['trunk.0.weight'].shape[0]==64
    assert checkpoint['labels'].split(',')==plan['labels']
    assert all(torch.isfinite(v).all() for v in checkpoint['state_dict'].values())
    upload=read(OUT/'upload.json')
    assert upload['download_verified'] and upload['model_sha256']==sha(model)
    assert upload['path']=='transfer/dw5i_s1_model.pt'
    from import_dw4_transfer import hf_environment
    from huggingface_hub import HfApi
    hf_environment()
    remote=next(e for e in HfApi().model_info(upload['repo'],revision=upload['revision'],files_metadata=True).siblings
                if e.rfilename==upload['path'])
    assert remote.lfs and remote.lfs.sha256==sha(model)
    native=read(OUT/'native_parity.json')
    assert native['passed'] and native['states']==256 and native['width']==64
    assert native['model_sha256']==sha(model) and native['executable_sha256']==sha(ROOT/'build/pvz-portable')
    assert read(OUT/'native_parity.result.json')['returncode']==0
    references=read(OUT/'pretraining_plan.json')
    panels={name:audit_panel(OUT/f'baseline_v2_{name}','v2',references['models_sha256'][name])
            for name in ('dw4i_s1','identity')}
    new={v:audit_panel(OUT/f'panel_dw5i_{v}',v,sha(model)) for v in ('v1','v2')}
    old_v1_path=MAIN/'.claude/worktrees/dw4/artifacts/task23/panel_dw4i_s1/panel.json'
    old_v1=read(old_v1_path);old_manifest=read(OUT/'unified_v1.json')
    assert old_v1['complete'] and old_v1['model_sha256']==references['models_sha256']['dw4i_s1']
    assert old_v1['manifest_sha256']==sha(OUT/'unified_v1.json')
    for task in old_manifest['tasks']:
        group=old_manifest['groups'][task['task_id']];n=64 if group=='heldout' else 128
        actual=summarize(old_v1_path.parent/task['task_id'],task,task['seeds'][:n]);actual['group']=group
        assert actual==next(r for r in old_v1['tasks'] if r['task_id']==task['task_id'])
    dead=read(ROOT/'MEMORY/docs/task27-dead-decks.json')
    assert set(dead)=={'day_l7_eval_d7p_holdout3','day_l8_eval_d7p_holdout1','day_l9_eval_d7p_holdout1'}
    for name,data in dead.items():
        folder=OUT/'dead_decks'/name;game_rows=[json.loads(l) for l in (folder/'games.jsonl').read_text().splitlines()]
        assert data['games']==len(game_rows)==16 and data['won']==sum(r['won'] for r in game_rows)
        assert sha(folder/'games.jsonl')==data['games_sha256']
        assert {r['seed'] for r in game_rows}==set(range(30000,30016))
        cfg=read(folder/'config.json')
        assert cfg['dagger_beta']=='1.0' and cfg['verify_samples']=='9' and cfg['student_candidates']=='2'
        assert cfg['static_value']==str(MAIN/'artifacts/hard/value_branch1/model.pt') and cfg['workers']=='4'
        assert read(OUT/('dead_'+name+'.result.json'))['returncode']==0
    source_groups=defaultdict(Counter)
    base_labels=set(read(ROOT/'MEMORY/docs/task23-study-plan.json')['labels'])
    for label,stats in rows['sources'].items():
        group='dw4i_labels' if label in base_labels else 'q102' if label in plan['q102_labels'] else 'q103' if label in plan['q103_labels'] else 'mac_new'
        source_groups[group].update({key:stats[key] for key in ('archives','rows','training_rows','validation_rows')})
    full=dict(complete=True,study_plan=plan,training=training,metrics=metrics,source_rows=rows,
              upload=upload,native_parity=native,v2_references=panels,dw5i_panels=new,
              historical_dw4i_v1=old_v1,dead_decks=dead)
    write(OUT/'results.json',full)
    compact=dict(complete=True,full_results_path=str(OUT/'results.json'),full_results_sha256=sha(OUT/'results.json'),
                 study_plan_sha256=sha(OUT/'study_plan.json'),training=training,upload=upload,
                 source_groups={k:dict(v) for k,v in source_groups.items()},
                 v2={name:dict(groups=p['groups'],tasks=[{k:t[k] for k in ('task_id','games','won','win_rate')} for t in p['tasks']])
                     for name,p in {**panels,'dw5i_s1':new['v2']}.items()},
                 v1={name:dict(groups=p['groups'],tasks=[{k:t[k] for k in ('task_id','games','won','win_rate')} for t in p['tasks']])
                     for name,p in {'dw4i_s1':old_v1,'dw5i_s1':new['v1']}.items()},
                 native_states=native['states'],dead={name:{k:d[k] for k in ('deck','games','won','loss_waves','nearest_terminal_zombie_types','conclusion')} for name,d in dead.items()},
                 caveats=['One fixed training seed and final epoch; no seed/epoch selection or default promotion.',
                          'V1 and V2 use different night decks; compare models within each version.',
                          'Source rows grouped by label; expanded old labels remain in dw4i_labels.',
                          'Zero wins in 16 teacher games flags structural weakness; not proof of impossibility.',
                          'Terminal nearest zombies describe surviving threats, not instrumented killers.',
                          'Historical dw4i v1 uses its original binary; additive telemetry checked against legacy outcomes.'])
    target=ROOT/'MEMORY/docs/task27-results.json';write(target,compact);assert target.stat().st_size<100_000
    lines=['## task27：面板v2与第二次合训dw5i','',
           'v1保持旧29任务；v2改为夜间n10九关并加泳池p9k五关，共34任务/3264局。只报告各区域任务平均胜率，不把版本之间的夜间变化当成模型收益。', '',
           '| v2 区域 | identity_dw3_s1 | dw4i_s1 | dw5i_s1 | dw5−dw4 |',
           '|---|---:|---:|---:|---:|']
    for group in ('day','heldout','night','pool'):
        stats=[p['groups'][group] for p in (panels['identity'],panels['dw4i_s1'],new['v2'])]
        lines.append('| '+group+' | '+' | '.join(f"{s['won']}/{s['games']} ({s['mean_win_rate']:.2%})" for s in stats)+f" | {100*(stats[2]['mean_win_rate']-stats[1]['mean_win_rate']):+.2f}pp |")
    lines += ['', '| v2 任务 | 旧identity | dw4i_s1 | dw5i_s1 |','|---|---:|---:|---:|']
    for row in new['v2']['tasks']:
        name=row['task_id'];results=[next(t for t in p['tasks'] if t['task_id']==name) for p in (panels['identity'],panels['dw4i_s1'])]+[row]
        lines.append('| '+name+' | '+' | '.join(f"{r['won']}/{r['games']}" for r in results)+' |')
    early={}
    for name,panel in {**panels,'dw5i_s1':new['v2']}.items():
        selected=[r for r in panel['tasks'] if r['task_id'] in {f'day_l{n}_eval_d7' for n in (1,3,4,6)}]
        early[name]=dict(won=sum(r['won'] for r in selected),games=sum(r['games'] for r in selected))
    lines += ['', '白天早关L1/3/4/6单独合计：'+'；'.join(f"{name} {s['won']}/{s['games']} ({s['won']/s['games']:.2%})" for name,s in early.items())+'。早关是否恢复，以这个预先指定的四关合计及逐关列判断。','',
              '| v1 区域（历史可比） | dw4i_s1 | dw5i_s1 |','|---|---:|---:|']
    for group in ('day','heldout','night'):
        lines.append('| '+group+' | '+' | '.join(f"{p['groups'][group]['won']}/{p['groups'][group]['games']} ({p['groups'][group]['mean_win_rate']:.2%})" for p in (old_v1,new['v1']))+' |')
    lines += ['', '| v1 任务 | dw4i_s1 | dw5i_s1 |','|---|---:|---:|']
    for row in new['v1']['tasks']:
        before=next(t for t in old_v1['tasks'] if t['task_id']==row['task_id'])
        lines.append(f"| {row['task_id']} | {before['won']}/{before['games']} | {row['won']}/{row['games']} |")
    lines += ['', '| 全零留出诊断 | 卡组 | 老师胜局 | 输局波数 |','|---|---|---:|---|']
    for name,d in dead.items():lines.append(f"| {name} | {d['deck']} | {d['won']}/16 | {d['loss_waves']} |")
    lines += ['', '老师参数照queue103的run，仅改beta=1实际执行搜索，每套前16评估seed。逐局最后存活僵尸类型和距离见诊断JSON；最近僵尸不是经埋点确认的致死者。0/16提示卡组结构性弱点，留出集合本次保持不变，由主控决定是否另开任务族替换。','',
              f"训练语料在q103完整{plan['q103_full_rounds']}轮后冻结：{plan['archives']}档案/{len(plan['labels'])}标签，原dw4i全部48816份逐文件哈希保持；q102全部、指定q103完整轮次、已上传未导入Mac档案只增不改导入。不选赢局，不纳入评估种子。",'',
              '| 实际来源（按标签） | 档案 | 保留行 | 训练行 | 验证行 |','|---|---:|---:|---:|---:|']
    for group,s in source_groups.items():lines.append(f"| {group} | {s['archives']} | {s['rows']} | {s['training_rows']} | {s['validation_rows']} |")
    resources=training['resources'];epoch_seconds=sum(h['seconds'] for h in metrics['history'])
    lines += ['', f"64宽identity+row_pool、seed1从头原12epoch；实际{rows['rows']}行，加载{metrics['load_seconds']:.1f}s、epoch训练+验证合计{epoch_seconds:.1f}s、总墙钟{resources['wall_s']:.1f}s。子树RSS求和峰{resources['peak_rss_kib']/1024**2:.2f}GiB（共享页可能重复），最低可用{resources['minimum_available_kib']/1024**2:.2f}GiB；模型{training['model_bytes']}字节。训练服从外部训练让卡与2/3GiB内存暂停守卫。",'',
              'C++与Python已在白天/夜间n10/原版传送带/泳池共256个真实状态核对，各头argmax与实际动作完全一致，并通过既定绝对+相对+实测浮点噪声容差。种植计数按每次成功动作前的真实类型累计；传送带没有用初始卡组翻译总槽计数。','',
              f"HF `{upload['path']}`，固定revision `{upload['revision']}` 已回读SHA一致。完整本地结果 `{OUT/'results.json'}`，SHA256 `{sha(OUT/'results.json')}`；模型、原始对局、档案和大JSON不入git。见[KB核验结果](task27-results.json)、[训练冻结计划](task27-study-plan.json)、[留出诊断](task27-dead-decks.json)。",'',
              '仅一个固定训练种子、使用第12epoch；不按最优seed/epoch挑选，不宣称总体非劣，不改默认/生产模型。q103和其他tmux队列未修改，主stage4清单只读。']
    report=ROOT/'MEMORY/docs/unified-panel.md';text=report.read_text().split('\n## task27：面板v2与第二次合训dw5i')[0].rstrip()
    report.write_text(text+'\n\n'+'\n'.join(lines)+'\n')
    book=ROOT/'MEMORY/docs/task27-dw5i.md';text=book.read_text()
    issue='- 三套全零留出老师诊断：'+ '；'.join(f"{n} {d['won']}/16，卡组{d['deck']}" for n,d in dead.items())+'。若老师全败，列为结构性弱卡组嫌疑，留出不自行替换，由主控决定。\n'
    if issue not in text:text=text.replace('## 问题\n','## 问题\n\n'+issue);book.write_text(text)
    verify(plan)
    record('4最终核验完成：两参考v2及dw5i的v2/v1共12736局逐任务原始seed/胜局/类型计数与全部退出码通过；48局beta1留出诊断、冻结语料/原12epoch/256状态原生对拍/HF固定版本交付均齐，统一报告已更新，不改默认。',target,report)
    write(OUT/'complete.json',dict(complete=True,results_sha256=sha(target),full_results_sha256=sha(OUT/'results.json')))


if __name__=='__main__':main()
