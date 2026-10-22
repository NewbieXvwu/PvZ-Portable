"""Validate task23 step 5's actual records and append the day regression diagnosis."""
from collections import Counter
import json
from pathlib import Path

from run_dw4_day_diagnosis import ROOT, BASE, OUT, read, write, sha, check, record
from unified_panel import summarize


def main():
    plan = read(OUT/'plan.json'); check(plan)
    training = read(OUT/'training128_complete.json')
    model = OUT/'width128_s1/model.pt'
    assert training['model_sha256'] == sha(model)
    assert training['resources'] == read(OUT/'training128.result.json') and training['resources']['returncode'] == 0
    assert read(OUT/'training128.command.json')['command'] == plan['width128_training_command']
    assert read(OUT/'width128_s1/source_rows.json') == read(BASE/'dw4i_s1/source_rows.json')
    import torch
    checkpoint = torch.load(model,map_location='cpu',weights_only=True)
    metrics = read(OUT/'width128_s1/metrics.json')
    assert checkpoint['history'] == metrics['history'] and [r['epoch'] for r in metrics['history']] == list(range(1,13))
    assert all(torch.isfinite(v).all() for v in checkpoint['state_dict'].values())
    assert metrics['data_memory']['rows'] == 13643901
    uploaded = read(OUT/'upload128.json')
    assert uploaded['download_verified'] and uploaded['model_sha256'] == sha(model)
    from import_dw4_transfer import hf_environment
    from huggingface_hub import HfApi
    hf_environment()
    remote = next(entry for entry in HfApi().model_info(uploaded['repo'],revision=uploaded['revision'],files_metadata=True).siblings
                  if entry.rfilename == uploaded['path'])
    assert remote.lfs and remote.lfs.sha256 == sha(model)
    panel = read(OUT/'panel_width128_s1/panel.json'); manifest = read(BASE/'unified_manifest.json')
    original_plan = read(BASE/'study_plan.json')
    assert panel['complete'] and panel['model_sha256'] == sha(model)
    assert panel['manifest_sha256'] == original_plan['manifest_sha256'] and panel['executable_sha256'] == original_plan['executable_sha256']
    assert Counter(t['task_id'] for t in panel['tasks']) == Counter(t['task_id'] for t in manifest['tasks'])
    for task in manifest['tasks']:
        group = manifest['groups'][task['task_id']]; size = 64 if group == 'heldout' else 128
        actual = summarize(OUT/'panel_width128_s1'/task['task_id'],task,task['seeds'][:size]); actual['group'] = group
        assert actual == next(row for row in panel['tasks'] if row['task_id'] == task['task_id'])
        assert read(OUT/'panel_width128_s1/resources'/(task['task_id']+'.result.json'))['returncode'] == 0
        assert read(OUT/'panel_width128_s1'/task['task_id']/'config.json')['workers'] == '4'
    assert sum(row['games'] for row in panel['tasks']) == 2944
    for group, stats in panel['groups'].items():
        rows = [row for row in panel['tasks'] if row['group'] == group]
        assert stats['games'] == sum(row['games'] for row in rows) and stats['won'] == sum(row['won'] for row in rows)
        assert abs(stats['mean_win_rate']-sum(row['win_rate'] for row in rows)/len(rows)) < 1e-12
    historical = read(OUT/'historical_losses.json'); current = read(OUT/'current_losses.json')
    wide_historical = read(OUT/'capacity_historical_losses.json'); wide_current = read(OUT/'capacity_current_losses.json')
    for data, selection, count, tag in ((historical,'historical_selection.json',plan['historical_archives'],'historical_losses'),
                                      (current,'current_selection.json',64,'current_losses'),
                                      (wide_historical,'historical_selection.json',plan['historical_archives'],'capacity_historical_losses'),
                                      (wide_current,'current_selection.json',64,'capacity_current_losses')):
        assert data['selection_sha256'] == sha(OUT/selection) and data['archives'] == count
        assert read(OUT/(tag+'.result.json'))['returncode'] == 0
    historical['models'].update(wide_historical['models']); current['models'].update(wide_current['models'])
    disagreements = read(OUT/'disagreements.json')
    assert set(disagreements) == {'1','3','4','6'} and all(v['games'] == 16 for v in disagreements.values())
    for level, counts in disagreements.items():
        assert read(OUT/f'early_l{level}.result.json')['returncode'] == 0
        config = read(OUT/f'early_l{level}/config.json')
        assert config['dagger_beta'] == '0.0' and config['rollout_policy'] == plan['comparison_models']['dw4i_s1']
        assert config['workers'] == '4'
        assert counts['decisions'] >= counts['searched'] >= counts['disagreements']
    timing = read(OUT/'forward_benchmark.json')
    assert timing['complete'] and timing['states'] == 24 and timing['widths']['64']['samples'] == timing['widths']['128']['samples'] == 576
    assert timing['widths']['128']['model_sha256'] == sha(model)
    assert timing['widths']['64']['model_sha256'] == plan['comparison_sha256']['dw4i_s1']
    assert read(OUT/'forward_benchmark.result.json')['returncode'] == 0
    old = read(BASE/'baseline_identity/panel.json'); narrow = read(BASE/'panel_dw4i_s1/panel.json'); narrow2 = read(BASE/'panel_dw4i_s2/panel.json')
    gain = 100*(panel['groups']['day']['mean_win_rate']-narrow['groups']['day']['mean_win_rate'])
    remaining = 100*(panel['groups']['day']['mean_win_rate']-old['groups']['day']['mean_win_rate'])
    if gain <= 0:
        conclusion = '同seed加宽未恢复白天，当前证据不支持仅加宽就能解决回退；结合早关训练数据缺口，更倾向先补当前学生的早关局面数据。'
    elif remaining < 0:
        conclusion = '同seed加宽部分恢复白天，支持容量/拟合存在贡献，但仍未恢复旧identity水平；不能把回退全部归因于容量，仍需早关数据对照。'
    else:
        conclusion = '同seed加宽恢复了本面板白天表现，支持容量/拟合存在贡献；能否同时保留夜间与留出收益，要结合完整面板及单步成本判断。'
    result = dict(complete=True,plan_sha256=sha(OUT/'plan.json'),historical_losses=historical,current_losses=current,
                  disagreements=disagreements,training128=training,upload128=uploaded,capacity_panel=panel,
                  forward_benchmark={k:v for k,v in timing.items() if k != 'records'},
                  day_gain_vs_same_seed64_pp=gain,day_delta_vs_old_identity_pp=remaining,conclusion=conclusion,
                  caveats=['Single width128 seed; no best-seed/epoch selection, no production promotion.',
                           'No L1/L3/L4/L6 label-added retraining intervention: missing-data diagnosis remains an inference.',
                           'Teacher is anchored search with dw4i_s1 rollout policy, finite horizon and min-gap; disagreement is not ground-truth regret.',
                           'Historical L8 has only two common-validation archives.',
                           'Hard-label evaluation losses are unweighted and conditional; not the weighted training objective.',
                           'Width also sets the default identity embedding dimension; no separate hyperparameter changed.',
                           'Native Predict timing includes public features and selection, excludes IPC and simulation; not end-to-end collection time.'])
    target = ROOT/'MEMORY/docs/task23-day-diagnosis.json'; write(target,result)
    lines = ['## 第5步：白天下降诊断','',conclusion,'',
             f"白天相对同seed64宽变化{gain:+.2f}pp，相对旧identity{remaining:+.2f}pp。只做一个128宽seed，不能排除随机优化差异；未做加早关标签再训的直接干预，数据缺口判断不是因果定论，也没有MoE必要性的直接证据。",'',
             f"冻结语料中L1–L6档案为0；检查现有主frames_v2后可用早关档案{plan['existing_l1_l6_archives']}份。历史比较使用旧/新共同留出的{plan['historical_archives']}份d7档案（L7/L8/L9/L10各424/2/13/13份），三模型完全相同行、全部有效行，无wait抽样。训练日志种子数35455/46000与清单一致，按训练器相同Random(0)划分求交集，避免新模型见过验证样本。",'',
             '| 历史验证来源 | 模型 | 档案 | 决策行 | plant BCE | packet CE | cell CE（老师卡） |',
             '|---|---|---:|---:|---:|---:|---:|']
    for group in ('level_7','level_8','level_9','level_10','all'):
        for name,data in historical['models'].items():
            if group not in data: continue
            row = data[group]
            lines.append(f"| {group} | {name} | {row['archives']} | {int(row['rows'])} | {row['plant_bce']:.4f} | {row['packet_ce']:.4f} | {row['cell_ce_teacher_packet']:.4f} |")
    lines += ['', 'plant为逐行无权BCE；packet/cell只在老师标记种植的行算CE，cell条件于老师卡。另记录正/负plant BCE与各头准确率，完整逐标签来源见JSON；这不是训练时带正类、卡种、spread与软标签重权的总loss。L8样本很少，不单独据其推断容量。','',
              '固定dw4i s1的L1/L3/L4/L6各16局（面板前16seed），beta=0学生实际走、dw4i推演+搜索打标签。搜索保持3样本、2500tick、最小300tick间隔与0.02锚定margin；该标签既非独立重老师，也非最优动作真值。','',
              '| 关卡 | 局数 | 分歧/全部决策 | 比例 | 搜索次数 | 分歧/搜索 | 与原面板胜负吻合 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for level,c in disagreements.items():
        lines.append(f"| L{level} | 16 | {c['disagreements']}/{c['decisions']} | {c['disagreements']/c['decisions']:.2%} | {c['searched']} | {c['disagreements']/max(1,c['searched']):.2%} | {c['panel_outcome_matches']}/16 |")
    lines += ['', '| 当前s1局面标签（64局合计） | plant BCE | packet CE | cell CE（老师卡） |', '|---|---:|---:|---:|']
    for name,data in current['models'].items():
        row = data['all']; lines.append(f"| {name} | {row['plant_bce']:.4f} | {row['packet_ce']:.4f} | {row['cell_ce_teacher_packet']:.4f} |")
    lines += ['', '当前局面是额外诊断标签，不曾加入128宽训练。两宽度训练仍使用同一48816档案、170标签、13643901实际保留行，seed1、原12epoch、损失权重/采样/优化器不变，仅CLI新增--width 128（identity嵌入维度随宽度默认变化）。','',
              '| 区域任务均值 | 旧identity | dw4i64 s1 | dw4i64 s2 | dw4i128 s1 |', '|---|---:|---:|---:|---:|']
    for group in ('day','heldout','night'):
        lines.append('| '+group+' | '+' | '.join(f"{p['groups'][group]['mean_win_rate']:.2%}" for p in (old,narrow,narrow2,panel))+' |')
    lines += ['', '| 统一面板任务 | dw4i64 s1 | dw4i128 s1 |', '|---|---:|---:|']
    for row in panel['tasks']:
        before = next(t for t in narrow['tasks'] if t['task_id'] == row['task_id'])
        lines.append(f"| {row['task_id']} | {before['won']}/{before['games']} | {row['won']}/{row['games']} |")
    resources = training['resources']; widths = timing['widths']
    lines += ['', f"128宽训练耗时{resources['wall_s']:.1f}s，子树RSS求和峰{resources['peak_rss_kib']/1024**2:.2f}GiB（共享页可能重复），最低可用{resources['minimum_available_kib']/1024**2:.2f}GiB；模型{training['model_bytes']}字节，64宽{widths['64']['model_bytes']}字节。",'',
              f"同一原生程序、单线程BLAS、同24个有合法种植的L9真实快照，ABBA/BAAB交替宽度，每状态每模型8次预热+24次计时，共各576次。C++内部network_seconds计时Predict（公共特征+网络+选动作），每次确认1个真实网络调用；IPC、加载、还原与模拟均在计时外。64/128宽中位数{widths['64']['median_us']:.2f}/{widths['128']['median_us']:.2f}微秒，倍率{timing['median_ratio_128_over_64']:.2f}；逐状态配对中位倍率{timing['paired_state_median_ratio']:.2f}。这决定老师续走中的学生调用成本，但不能把倍率直接当端到端采集耗时倍率。",'',
              '完整逐标签损失、逐头分歧、训练/面板/计时证据见[第5步核验结果](task23-day-diagnosis.json)，预登记见[固定计划](task23-day-diagnosis-plan.json)。容量模型在HF transfer/dw4i_w128_s1_model.pt，已按固定revision回读SHA核对。']
    report = ROOT/'MEMORY/docs/unified-panel.md'; text = report.read_text().split('\n## 第5步：白天下降诊断')[0].rstrip()
    report.write_text(text+'\n\n'+'\n'.join(lines)+'\n')
    check(plan)
    record(f'5诊断收尾：共同验证逐头损失、64局beta0老师分歧、同语料128宽12epoch及完整2944局、原生相同状态前向计时均完成并核验；白天相对同seed64宽{gain:+.2f}pp、相对旧identity{remaining:+.2f}pp，Predict中位成本{timing["median_ratio_128_over_64"]:.2f}倍。结论及限制已并入unified-panel.md，不挑seed、不改默认。',target,report)
    write(OUT/'complete.json',dict(complete=True,results_sha256=sha(target)))


if __name__ == '__main__': main()
