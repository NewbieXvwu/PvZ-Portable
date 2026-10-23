"""Render the completed roof terminal audit without promoting pilot data to evidence."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'python')]
import terminal_candidate_audit as d


def percentage(value):
    return '—' if value is None else f'{100*value:.1f}%'


def render():
    data=d.load(d.OUT/'screen-report.json')
    repro=d.load(ROOT/'MEMORY/docs/task46-roof-terminal-repro.json')
    selection=d.load(d.OUT/'selection-active.json')
    if data['status']!='complete' or repro['status']!='complete':
        raise ValueError('full screen and independent reproduction must complete first')
    if data['sampling_sha256']!=d.sha(d.OUT/'selection-active.json'):
        raise ValueError('report sampling mismatch')
    overall=data['overall'];totals=data['attribution_totals']
    cpu=d.spent_cpu()
    if cpu+data['cpu_allowance_seconds']>72000:raise ValueError('CPU budget exceeded')
    lines=['# task46 屋顶第三阶段：完整候选终局审计','',
        f"正式面板共 {overall['roots']} 个决策根、{overall['base_seed_units']} 个关卡×reset seed 单元、"
        f"{overall['candidates']} 个合法候选、{overall['labels']} 条冻结搜索老师真实终局标签。"
        f"每根 {selection['choice']['samples']} 个共同随机未来；原版倍率与 r7 卡组保持冻结。",'',
        '取点先尝试原定时间间隔密度，实测 CPU 门禁拒绝后，在正式标签采集前登记稀疏后备。'
        '六个非 L45 旧点计入其 reset seed 覆盖，其余单元只用候选数和实测 CPU 选择一个 0–10000 tick 学生决策点。'
        '六个旧点保留原老师动作前缀，其中三个超出 10000 tick，作为明确的命名点例外；L45 按追加 B 末句留后续。'
        '因此这些数据审计选中点，不能估计整个开局时间窗的发生率或关卡胜率。','',
        '每条终局分支在执行候选后立即交给冻结搜索老师；2500 tick 学生续走只用于独立计算 v4 短叶，'
        '没有插入终局老师轨迹。根物理状态、学生历史、阳光收入上下文、合法候选集合、输入哈希、共同未来和完整老师动作链均逐条校验。','',
        '## 四项审计','',
        '“改善”是相对于该根学生动作的共同未来终局均值差：胜率至少 1/3、末波至少 1、终 tick 至少 600。'
        '本次只有两个未来，非零胜率差实际至少为 1/2。以下是观察到的均值改善，并不等于统计上确认的真改善。','',
        '| 终局指标 | 改善植物动作 | branch1 保留 | 生产可表达改善 | 候选生成遗漏 | halving 裁掉 | 叶排序未居首 | 0.02 初始门槛挡住 | 独立核验拒绝 | 执行 | 仅引擎合法铲子 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for metric,label in [('won','胜率'),('wave','末波'),('tick','终 tick')]:
        c=totals[metric]
        fields=['improving_plant_actions','static_retained','production_representable_improvers',
            'candidate_generation_omission','halving_pruned','leaf_ranking_not_top',
            'initial_margin_blocked','verification_rejected','executed','outside_production_action_space']
        lines.append('| '+label+' | '+' | '.join(str(c.get(k,0)) for k in fields)+' |')
    lines+=['','branch1 保留数以改善植物动作作分母，属于重叠覆盖指标；后续各原因按生产搜索经过的第一个阻断阶段互斥归因。'
        '静态漏掉但由学生候选补回的动作不算最终候选生成遗漏。铲子是全引擎合法空间的一部分，生产搜索不表达它们，单列而不归罪于静态或 halving。','',
        'v4 的排序一致率以同根候选对计算，终局真值相等的候选对不计，v4 相等计半分。'
        '“有差距”采用上表改善阈值；“严格”计全部非平局真值。置信区间按关卡×reset seed 聚类自助法计算，'
        '不是把成千候选对当独立对局。少于两个有可比较候选对的有效聚类时不报告区间。','',
        '| 终局指标 | v4 有差距一致率 / 候选对 | 聚类 95% 区间 | v4 严格一致率 | 短叶均未终局时一致率 | 老师路径辅助叶一致率 |',
        '|---|---:|---|---:|---:|---:|']
    for metric,label in [('won','胜率'),('wave','末波'),('tick','终 tick')]:
        m=overall['metrics'][metric];v=m['v4'];interval=v['cluster_bootstrap_95']
        ci='—' if interval is None else '–'.join(percentage(x) for x in interval)
        lines.append(f"| {label} | {percentage(v['accuracy'])} / {v['pairs']} | {ci} | "
            f"{percentage(m['v4_strict']['accuracy'])} | {percentage(m['v4_nonterminal']['accuracy'])} | "
            f"{percentage(m['teacher_path_v4']['accuracy'])} |")
    lines+=['','老师路径辅助叶与正式 2500 tick 学生叶是不同对象，分别报告；短叶已终局者用真实胜负评分。'
        '末波和终 tick 不替代胜负：更晚死亡不自动等于更高获胜率。','',
        '## 分关卡与选中点','',
        '| 关卡 | 根数 | 完整终局标签 | 标签中的胜局 | 可分胜率根 | 可分末波根 | v4 末波有差距一致率 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for level,a in data['by_level'].items():
        lines.append(f"| L{level} | {a['roots']} | {a['labels']} | {a['terminal_wins']} | "
            f"{a['separable_roots']['won']} | {a['separable_roots']['wave']} | "
            f"{percentage(a['metrics']['wave']['v4']['accuracy'])} |")
    lines+=['','表中胜局来自不同候选分支，不能当作老师或学生独立开局胜率。','',
        '| 关卡 / reset seed | 前缀 | 决策（1起） | tick | 合法候选 | 学生动作终局均值（胜率 / 波 / tick / 丢车） |',
        '|---|---|---:|---:|---:|---|']
    for g in data['groups']:
        c=g['cell'];p=g['point'];m=g['candidates'][g['student_position']]['means']
        lines.append(f"| L{c['level']} / {c['seed']} | {'原老师旧点' if c.get('archive') else '学生'} | "
            f"{p['decision']+1} | {p['tick']} | {len(g['candidates'])} | "
            f"{percentage(m['won'])} / {m['wave']:.1f} / {m['tick']:.1f} / {m['mowers_used']:.1f} |")
    lines+=['','## 证据与边界','',
        f"独立目录同未来复现 {len(repro['cases'])} 条固定分支；根、叶值、终局状态及完整解压老师动作链精确一致。"
        f"计量 CPU {cpu/3600:.3f} 小时，另计 {data['cpu_allowance_seconds']} 秒测试/报告等余量；合计 "
        f"{(cpu+data['cpu_allowance_seconds'])/3600:.3f} 小时。原始标签和动作链留在 ignored artifacts，不提交 git。",'',
        '生产参数、0.02 门槛、卡组、qc 均未改动。两 reset seed 与两个共同未来只足以描述本批点；'
        '不能据此确认全关卡优胜动作、决定新课程准入，或把小均值差解释成显著收益。','',
        '- [冻结计划](task46-roof-terminal-plan.json)、[取点与 CPU 门禁](task46-roof-terminal-sampling.json)',
        '- [完整结论层](task46-roof-terminal-results.json)、[独立复现](task46-roof-terminal-repro.json)',
        '- 通用采集器适配 fog E 的根恢复/共同未来/全候选终局逻辑；终局续走直接复用 v5 bb2ec969 冻结源码。'
        '来源与 SHA 见计划，fog/v5 分支只读。','']
    destination=ROOT/'MEMORY/docs/task46-roof-terminal-diagnosis.md'
    destination.write_text('\n'.join(lines))
    return destination


if __name__=='__main__':print(render())
