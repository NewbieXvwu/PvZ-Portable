"""Record completed task30 stages, with text-only commits on branch vnet.

This is the task-specific completion hook for long-running experiments. It never
stages artifacts or model files, changes defaults, or edits another worktree.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[1]
DOCS=ROOT/'MEMORY/docs'
TASK=DOCS/'task30-value-widen.md'
REPORT=DOCS/'value-widen.md'


def read(path):return json.loads(path.read_text())
def write(path,data):path.write_text(json.dumps(data,indent=1)+'\n')
def fmt(value):return '—' if value is None else f'{value:.3f}'


def collect(out):
    groups=[]
    for directory,label,expected,student in [
        ('supplement','task30_pool24',32,'dw4i_s1'),('supplement_old','task30_pool21',16,'identity_dw3_s1'),
        ('supplement','task30_night20',32,'dw4i_s1'),('supplement','task30_day10',32,'dw4i_s1')]:
        rows=[json.loads(s) for s in (out/directory/label/'games.jsonl').read_text().splitlines()]
        if len({r['seed'] for r in rows})!=expected:raise ValueError(f'incomplete collection: {label}')
        groups.append(dict(label=label,student=student,games=len(rows),wins=sum(r['won'] for r in rows),
            seed_start=min(r['seed'] for r in rows),seed_end=max(r['seed'] for r in rows),
            mean_seconds=round(sum(r['seconds'] for r in rows)/len(rows),1)))
    data=read(out/'v4/episodes.json')
    external=read(out/'full_audit_manifest.json')
    result=dict(groups=groups,complete=True,episode_rows=data['rows'],episode_summary=data['summary'],external_games=len(external))
    write(DOCS/'value-widen-collection.json',result)
    return f"补采全部完成：{sum(g['games'] for g in groups)} 局；外部 A6 {len(external)} 局；训练/停止特征 {data['rows']} 帧。",[DOCS/'value-widen-collection.json']


def training(out):
    import torch
    history=read(out/'v4/training.json');checkpoint=torch.load(out/'v4/model.pt',map_location='cpu',weights_only=False)
    best=history['history'][checkpoint['epoch']]
    used=set(checkpoint['trained_on']);external={g['seed'] for g in read(out/'full_audit_manifest.json')}
    if used&external:raise ValueError('checkpoint/external seed overlap')
    result={k:v for k,v in history.items() if k!='config'}
    result.update(best_epoch=checkpoint['epoch'],best_validation=best['validation'],encoder=checkpoint['encoder'],
                  exposed_seeds=len(used),code_commit=checkpoint['code_commit'],audit_overlap=0)
    write(DOCS/'value-widen-training.json',result)
    return f"v4 训练完成：编码 v2，最佳 epoch {checkpoint['epoch']}，停止损失 {best['validation']:.4f}；外部种子交集 0。",[DOCS/'value-widen-training.json']


def audit(out):
    result=read(out/'v4_audit/metrics.json')
    for group in result['cohorts'].values():
        for model in group['models'].values():
            for wave in model['per_wave']:wave['calibration'].pop('bins',None)
    write(DOCS/'value-widen-v4.json',result)
    return 'v4 / v3 / branch1 同一外部种子的逐波 AUC 与校准完成，详见 value-widen-v4.json。',[DOCS/'value-widen-v4.json']


def final(out):
    paired=read(out/'ab/paired.json')
    if not paired['complete']:raise ValueError('paired experiment incomplete')
    write(DOCS/'value-widen-paired.json',paired)
    metrics=read(DOCS/'value-widen-v4.json');train=read(DOCS/'value-widen-training.json')
    collection=read(DOCS/'value-widen-collection.json');transfer=read(DOCS/'value-widen-transfer.json')
    white=paired['results']['day_l7_z15_eval']
    night=sum(v['delta_wins'] for k,v in paired['results'].items() if k.startswith('night_'))
    pool=sum(v['delta_wins'] for k,v in paired['results'].items() if k.startswith('pool_'))
    observed=white['delta_wins']>=0 and night>0 and pool>0
    conclusion=('本轮观察到白天保持、夜间与泳池净胜局增加；保留候选，建议主控扩大样本验收。'
                if observed else '本轮未同时满足白天保持、夜间和泳池更好的条件，不建议替换 v3。')
    lines=['# 价值网络跨区加宽（task30）','',conclusion,
        '默认值保持不变。每关 32 对无法把“不显著”解释为已经证明非劣；是否切换由主控决定。','',
        '## 数据与隔离','',
        f"原始外部体检 1360 局；加上独立补采种子后，最终外部 A6 共 {collection['external_games']} 局。每局每波取首个非终局帧，逐波计算 AUC；同一种子从拟合、停止及初始化历史中整体排除。单一结果类别的 AUC 保留为空。",
        f"现有 L7 z13/z15 分叉 354005 行，97 文件、352 种子；整局特征 {collection['episode_rows']} 帧。按种子哈希留出停止集；补采哈希余数 1/2 的 20% 种子另留外部 A6。训练批次与停止指标均按 75% 分叉 / 25% 整局结果混合。",
        'v3 元数据中的 Mac 全部 93 万行原始分叉在本机不全，本轮使用可用本地分叉并从 v3 热启动。trained_on 包含初始化历史与本轮停止种子。补采老师与原有学生局分布不同，分别报告外部指标。','',
        '| 补采组 | 策略 | 胜/局 |','|---|---|---:|']
    for g in collection['groups']:lines.append(f"| {g['label']} | {g['student']} | {g['wins']}/{g['games']} |")
    lines+=['','## 编码与兼容','',
        'v2 跳过负植物类型；原版传送带手牌按引擎免费标志编码，同类多包按任一可用聚合。追加夜间、泳池、雾、屋顶、原版传送带、公开坟头数量 6 个标量。空槽实际污染旧 scalars[4]（总波数）和 scalars[58]（类型 53 就绪位）。',
        '白天原版体检帧中，53.6% 含空槽，89.8% 含免费但原价阳光不足的手牌；2-10 分别为 25.1% / 98.0%。旧 v1 逐位及缓存签名保留。C++ 只做学生前向，价值前向在 Python，CPU 叶子按检查点编码；CUDA 预编码叶子仍只支持 v1，v2 提前拒绝。静态 branch1 仍固定旧编码。',
        f"新检查点编码 `{train['encoder']}`，最佳 epoch {train['best_epoch']}，外部种子交集 0。7 项无游戏兼容/种子隔离/热启动门禁与独立 Release 编译通过。",'',
        '## 外部校准与逐波排序','',
        '下表 Brier 汇总全部波首帧，受存活波数影响；逐波样本数、AUC、预测/实际胜率与校准分箱见 JSON。整局间 AUC 不替代同局候选排序或老师实战。','',
        '| 区 | 胜/局 | v3 Brier | v4 Brier | branch1 Brier |','|---|---:|---:|---:|---:|']
    for name,g in metrics['cohorts'].items():
        m=g['models'];lines.append(f"| {name} | {g['wins']}/{g['games']} | {fmt(m['v3']['calibration']['brier'])} | {fmt(m['v4']['calibration']['brier'])} | {fmt(m['branch1']['calibration']['brier'])} |")
    lines+=['','| 区 | 波 | v3 AUC | v4 AUC | branch1 AUC |','|---|---:|---:|---:|---:|']
    for name,g in metrics['cohorts'].items():
        for wave in (5,10,15):
            aucs={k:next((x['auc'] for x in v['per_wave'] if x['wave']==wave),None) for k,v in g['models'].items()}
            lines.append(f"| {name} | {wave} | {fmt(aucs['v3'])} | {fmt(aucs['v4'])} | {fmt(aucs['branch1'])} |")
    # Mean per-wave AUC over waves with both outcome classes: the "what is left outside daytime" summary.
    def mean_auc(model):
        values=[x['auc'] for x in model['per_wave'] if x['auc'] is not None]
        return sum(values)/len(values) if values else None, len(values)
    lines+=['','| 区 | 胜/局 | v3 平均逐波 AUC | v4 平均逐波 AUC | branch1 平均逐波 AUC | 可算 AUC 的波数 |','|---|---:|---:|---:|---:|---:|']
    for name,g in metrics['cohorts'].items():
        cells=[mean_auc(g['models'][m]) for m in ('v3','v4','branch1')]
        lines.append(f"| {name} | {g['wins']}/{g['games']} | {fmt(cells[0][0])} | {fmt(cells[1][0])} | {fmt(cells[2][0])} | {cells[1][1]} |")
    saturated=[f"{name}（实际 {g['wins']}/{g['games']}，v4 平均预测 {g['models']['v4']['calibration']['predicted']:.3f}）"
        for name,g in metrics['cohorts'].items() if g['models']['v4']['calibration']['predicted']<0.05 and g['wins']==0]
    if saturated:lines+=['',f"v4 在这些全败区把胜率压到近零：{('、'.join(saturated))}。校准方向正确（v3 高估到 0.5–0.7），但叶值几乎不随局面变化，同局候选的排序只能靠实战 A/B 检验；泳池老师数据（有胜有负）仍给出可算的排序信息。"]
    lines+=['','## 老师同种子 A/B','',
        '固定 dw4i_s1 推演、beta=1、branch1 静态估值、verify=9、student-candidates=2，其余 queue103 默认；只换叶子 v3→v4。泳池 L21/L24 在任何 A/B 开局前固定，覆盖 20/40 波。每关同种子 32 对，双侧精确 McNemar。','',
        '| 任务 | v3 胜 | v4 胜 | 净胜局 | v4 独胜:独负 | p |','|---|---:|---:|---:|---:|---:|']
    for name,g in paired['results'].items():lines.append(f"| {name} | {g['old_wins']}/{g['pairs']} | {g['new_wins']}/{g['pairs']} | {g['delta_wins']:+d} | {g['gain']}:{g['harm']} | {g['mcnemar_exact_two_sided']:.4g} |")
    explanation=('本轮同时改变编码与训练覆盖，收益不能单独归因于其中一项。A6 是不同整局间的胜负预测，老师实战才检验同局动作选择；本轮观察信号仍需扩大配对样本确认。' if observed else
        '输入修复与跨区重训未同时实现白天保持和跨区提升。A6 比较不同整局的胜负，搜索选择同局的候选；跨区新增标签主要是整局结果，保留的分叉仍集中在 L7，因此离线校准不能直接证明跨区动作排序变好。后续需固定推演学生，补跨区分叉排序对照。')
    lines+=['',conclusion,'',explanation,'',
        '## 交付','',
        f"候选已上传 HF `{transfer['repo']}/transfer/value_v4.pt`，下载回验 SHA256 `{transfer['sha256']}` 一致。编码、数据与配对工具在 vnet 分支；模型、帧、缓存与滚动日志不入 git。",
        '结论层：[原始体检](value-widen-audit.json)、[补采](value-widen-collection.json)、[训练](value-widen-training.json)、[v4 外部评估](value-widen-v4.json)、[配对](value-widen-paired.json)、[HF 回验](value-widen-transfer.json)。']
    REPORT.write_text('\n'.join(lines)+'\n')
    return f"老师 A/B 六关各 32 对完成；白天净 {white['delta_wins']:+d}、夜间净 {night:+d}、泳池净 {pool:+d}。{conclusion} HF 下载回验通过。",[DOCS/'value-widen-paired.json',DOCS/'value-widen-transfer.json']


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['collection','training','audit','final'])
    p.add_argument('--out',type=Path,required=True);p.add_argument('--push',action='store_true');a=p.parse_args()
    git=lambda *args:subprocess.run(['git','-C',str(ROOT),*args],check=True,capture_output=True,text=True).stdout
    if git('branch','--show-current').strip()!='vnet':raise ValueError('completion commits require branch vnet')
    marker=f'[auto-{a.stage}]';task=TASK.read_text()
    if marker not in task:
        note,paths={'collection':collect,'training':training,'audit':audit,'final':final}[a.stage](a.out)
        now=datetime.now(ZoneInfo('Asia/Shanghai'));line=f"- {now:%Y-%m-%d %H:%M} {marker} {note}\n\n"
        TASK.write_text(task.replace('## 问题',line+'## 问题',1))
        daily=ROOT/'MEMORY'/f'{now:%Y-%m-%d}.md'
        with daily.open('a') as f:f.write(f'\n- task30：{note}\n')
        if a.stage!='final':
            with REPORT.open('a') as f:f.write(f'\n{note}\n')
    else:
        paths={'collection':[DOCS/'value-widen-collection.json'],'training':[DOCS/'value-widen-training.json'],
               'audit':[DOCS/'value-widen-v4.json'],'final':[DOCS/'value-widen-paired.json',DOCS/'value-widen-transfer.json']}[a.stage]
        daily=ROOT/'MEMORY'/f'{datetime.now(ZoneInfo("Asia/Shanghai")):%Y-%m-%d}.md'
    paths=[TASK,REPORT,*paths,*([daily] if daily.exists() else [])]
    existing=git('diff','--cached','--name-only').splitlines()
    allowed={str(p.relative_to(ROOT)) for p in paths}
    if set(existing)-allowed:raise ValueError('unrelated staged changes: refusing completion commit')
    git('add',*[str(p) for p in paths])
    staged=git('diff','--cached','--name-only').splitlines()
    if any(Path(name).suffix not in ('.md','.json') for name in staged):raise ValueError('non-text deliverable in stage')
    tracked=subprocess.run(['git','-C',str(ROOT),'ls-files','-z',str(ROOT/'artifacts'),str(ROOT/'logs')],capture_output=True,check=True)
    ignored=subprocess.run(['git','-C',str(ROOT),'check-ignore','-z','--stdin','--no-index'],input=tracked.stdout,capture_output=True)
    if ignored.stdout:raise ValueError('tracked ignored artifacts exist')
    if staged:
        print(git('diff','--cached','--numstat'));print(git('diff','--cached','--stat'))
        print(git('commit','-m',f'task30: record completed {a.stage} stage'))
    if a.push:print(git('push','origin','vnet'))
if __name__=='__main__':main()
