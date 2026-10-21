"""Audit task23's two trainings, HF delivery and all 11776 raw evaluation records."""
from collections import Counter
import json
from pathlib import Path
import torch

from run_dw4_study import ROOT, OUT, read, write, sha, verify, record
from unified_panel import summarize
from import_dw4_transfer import hf_environment


def main():
    plan=read(OUT/'study_plan.json');verify(plan,data=True)
    assert sha(OUT/'training_report_checks.txt')==plan['engineering']['training_checks_sha256']
    assert sha(ROOT/'MEMORY/docs/task23-panel-smoke.json')==plan['engineering']['panel_smoke_sha256']
    assert read(ROOT/'MEMORY/docs/task23-study-plan.json')==plan
    manifest=read(OUT/'unified_manifest.json')
    definitions={t['task_id']:t for t in manifest['tasks']}
    assert len(definitions)==29
    outputs={};training={}
    original=torch.load(plan['reference_models']['identity'],map_location='cpu',weights_only=True)
    recipe=('width','pool','label_weight','packet_balance','row_pool','aux_weight','aux_horizon','packet_head',
            'plant_embedding_dim','context','obs','soft_weight','spread_weight','spread_cap','lr_schedule','ema','cell_soft_weight')
    from huggingface_hub import HfApi,hf_hub_download
    hf_environment();api=HfApi()
    for seed in (1,2):
        folder=OUT/f'dw4i_s{seed}';model=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
        marker=read(OUT/f'training_s{seed}_complete.json');metrics=read(folder/'metrics.json');sources=read(folder/'source_rows.json')
        assert sha(folder/'model.pt')==marker['model_sha256']
        assert marker['source_rows_sha256']==sha(folder/'source_rows.json')
        assert model['labels'].split(',')==plan['labels']
        assert all(model[key]==original[key] for key in recipe)
        assert all(torch.isfinite(value).all() for value in model['state_dict'].values())
        assert model['history']==metrics['history'] and [r['epoch'] for r in metrics['history']]==list(range(1,13))
        assert all(torch.isfinite(torch.tensor(row[key])) for row in metrics['history'] for key in ('train_loss','val_loss'))
        assert sources['rows']==metrics['data_memory']['rows']==marker['rows']
        assert {k:v['archives'] for k,v in sources['sources'].items()}==plan['archive_counts']
        assert all(r['training_rows']+r['validation_rows']==r['rows'] for r in sources['sources'].values())
        assert read(OUT/f'training_s{seed}.command.json')['command']==plan['training_commands'][seed-1]
        assert read(OUT/f'training_s{seed}.result.json')==marker['resources'] and marker['resources']['returncode']==0
        uploaded=read(OUT/f'upload_s{seed}.json')
        assert uploaded['model_sha256']==marker['model_sha256'] and uploaded['download_verified']
        info=api.model_info(uploaded['repo'],revision=uploaded['revision'],files_metadata=True)
        remote=next(entry for entry in info.siblings if entry.rfilename==uploaded['path'])
        if remote.lfs:assert remote.lfs.sha256==marker['model_sha256']
        else:
            downloaded=Path(hf_hub_download(uploaded['repo'],uploaded['path'],revision=uploaded['revision'],local_dir=OUT/'final_hf_verification'))
            assert sha(downloaded)==marker['model_sha256']
        groups={}
        for label,rows in sources['sources'].items():
            group=plan['groups'][label];entry=groups.setdefault(group,Counter())
            entry.update(rows)
        training[f'dw4i_s{seed}']=dict(seed=seed,epochs=12,model_sha256=marker['model_sha256'],
                                     rows=sources['rows'],source_groups={g:dict(v) for g,v in groups.items()},
                                     source_labels=sources['sources'],final_epoch=metrics['history'][-1],
                                     resources=marker['resources'],upload=uploaded,recipe_matches=list(recipe))
    assert training['dw4i_s1']['source_labels']==training['dw4i_s2']['source_labels']
    models=dict(v34=(OUT/'baseline_v34',Path(plan['reference_models']['v34'])),
                identity=(OUT/'baseline_identity',Path(plan['reference_models']['identity'])),
                dw4i_s1=(OUT/'panel_dw4i_s1',OUT/'dw4i_s1/model.pt'),
                dw4i_s2=(OUT/'panel_dw4i_s2',OUT/'dw4i_s2/model.pt'))
    raw={}
    for name,(folder,checkpoint) in models.items():
        panel=read(folder/'panel.json')
        assert panel['complete'] and len(panel['tasks'])==29
        assert panel['model_sha256']==sha(checkpoint) and panel['manifest_sha256']==plan['manifest_sha256']
        assert panel['executable_sha256']==plan['executable_sha256']
        assert Counter(r['task_id'] for r in panel['tasks'])==Counter(definitions.keys())
        checked=[];raw[name]={}
        for row in panel['tasks']:
            task=definitions[row['task_id']];group=manifest['groups'][task['task_id']]
            count=64 if group=='heldout' else 128
            actual=summarize(folder/task['task_id'],task,task['seeds'][:count]);actual['group']=group
            assert actual==row
            config=read(folder/task['task_id']/'config.json')
            assert config['rollout_policy']==str(checkpoint) and config['workers']=='4'
            assert config['executable']==str(ROOT/'build/pvz-portable') and config['manifest']==str(OUT/'unified_manifest.json')
            games=[json.loads(v) for v in (folder/task['task_id']/'games.jsonl').read_text().splitlines()]
            raw[name][task['task_id']]={r['seed']:r for r in games}
            checked.append(actual)
        assert sum(r['games'] for r in checked)==2944
        for group,stats in panel['groups'].items():
            subset=[r for r in checked if r['group']==group]
            assert stats['tasks']==len(subset) and stats['games']==sum(r['games'] for r in subset)
            assert stats['won']==sum(r['won'] for r in subset)
            assert abs(stats['mean_win_rate']-sum(r['win_rate'] for r in subset)/len(subset))<1e-12
        outputs[name]=dict(model_sha256=sha(checkpoint),panel_sha256=sha(folder/'panel.json'),groups=panel['groups'],tasks=checked)
    comparisons={}
    for name in ('dw4i_s1','dw4i_s2'):
        changes=[]
        for task in definitions:
            before=raw['identity'][task];after=raw[name][task];assert set(before)==set(after)
            gain=sum(after[s]['won'] and not before[s]['won'] for s in before)
            loss=sum(before[s]['won'] and not after[s]['won'] for s in before)
            changes.append(dict(task_id=task,pilot_only_wins=gain,reference_only_wins=loss,
                                delta_wins=gain-loss,delta_pp=100*(gain-loss)/len(before)))
        comparisons[name]=dict(tasks=changes,groups={g:100*(outputs[name]['groups'][g]['mean_win_rate']-outputs['identity']['groups'][g]['mean_win_rate']) for g in ('day','heldout','night')})
    result=dict(complete=True,evaluated_games=11776,archives=plan['archives'],labels=len(plan['labels']),
                plan_sha256=sha(OUT/'study_plan.json'),inventory_sha256=plan['inventory_sha256'],training=training,
                models=outputs,comparisons_vs_identity=comparisons,
                limitations=['Two fixed training seeds, no best-seed/epoch selection or production promotion.',
                             'Many data sources are added together; not an attribution study of any individual source.',
                             'Additional data also changes the seed-based validation split and optimizer-step count.',
                             'v34 is a slot model; night and changed-deck results also reflect slot identity mismatch.',
                             'L10 holdouts test initial-deck generalization; later conveyor refills are not held out.',
                             'Training RSS peaks sum process RSS and may count shared pages repeatedly.'])
    destination=ROOT/'MEMORY/docs/task23-results.json';write(destination,result)
    lines=['# task23：统一面板与dw4i合训','',
           f"同一冻结数据{plan['archives']}档案/{len(plan['labels'])}标签，原identity+row_pool配方从头各训12epoch，seed1/2均完整报告。四模型同一29项面板，每模型2944局，共11776局。",'',
           '| 任务 | v34 | identity_dw3_s1 | dw4i s1 | dw4i s2 |','|---|---:|---:|---:|---:|']
    for task in manifest['tasks']:
        cells=[]
        for name in models:
            row=next(r for r in outputs[name]['tasks'] if r['task_id']==task['task_id'])
            cells.append(f"{row['won']}/{row['games']}")
        lines.append('| '+task['task_id']+' | '+' | '.join(cells)+' |')
    lines+=['','| 区域任务平均胜率 | v34 | identity_dw3_s1 | dw4i s1 | dw4i s2 |','|---|---:|---:|---:|---:|']
    for group in ('day','heldout','night'):
        lines.append('| '+group+' | '+' | '.join(f"{outputs[name]['groups'][group]['mean_win_rate']:.2%}" for name in models)+' |')
    lines+=['','相对旧identity的区域平均变化（pp）：']
    for name in comparisons:
        lines.append('- '+name+'：'+', '.join(f'{g} {v:+.2f}' for g,v in comparisons[name]['groups'].items()))
    lines+=['','| 训练 | 实际保留行 | 耗时秒 | 子树RSS求和峰GiB | 最低可用GiB |','|---|---:|---:|---:|---:|']
    for name,item in training.items():
        r=item['resources'];lines.append(f"| {name} | {item['rows']} | {r['wall_s']:.1f} | {r['peak_rss_kib']/1024**2:.2f} | {r['minimum_available_kib']/1024**2:.2f} |")
    lines+=['','各来源实际保留行（两个seed的来源/训练/验证计数相同）：','',
            '| 来源组 | 档案 | 总行 | 训练行 | 验证行 |','|---|---:|---:|---:|---:|']
    for name,counts in training['dw4i_s1']['source_groups'].items():
        lines.append(f"| {name} | {counts['archives']} | {counts['rows']} | {counts['training_rows']} | {counts['validation_rows']} |")
    lines+=['','完整逐标签行数、逐种子配对得失、模型/程序/数据SHA与HF交付版本见[核验结果](task23-results.json)。'
            '两个模型均上传HF transfer/dw4i_s{1,2}_model.pt，并回读核对内容。'
            'Mac与缺失标签按[冻结计划](task23-study-plan.json)记账，原始首批导入见[记录](task23-mac-import.json)。',
            '', 'L10三套留出保持task23指定的初始牌语义，后续仍由原生传送带补卡；原版L10/L20是10空槽开局。'
            'v34的夜间/换卡成绩混有slot错位，不能当作纯粹地形策略差异。'
            '此试验同时加入多种数据，不能归因到单一来源；只报告固定两种子，不宣称总体非劣，不晋升默认/生产模型。']
    report=ROOT/'MEMORY/docs/unified-panel.md';report.write_text('\n'.join(lines)+'\n')
    verify(plan)
    record('4最终核验完成：两个seed原12epoch、来源实际行数与完整四模型29项/11776局逐seed原始记录均重读通过，两个模型HF上传并回读SHA一致。夜间、白天/原版与留出逐项及区域变化已写统一报告；不挑seed/epoch，不改默认/生产模型。',destination,report)
    write(OUT/'study_complete.json',dict(complete=True,results_sha256=sha(destination)))


if __name__=='__main__':main()
