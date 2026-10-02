from pathlib import Path
import hashlib
import json

root = Path('/home/newbiexvwu/PvZ-Portable')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
audit_path = root/'artifacts/t5/perf/reward_v2_corrected_curve_delivery_v1.json'
audit = json.loads(audit_path.read_text())
state_path = root/'artifacts/research/reward_v2_packet_cost_reevaluation_v2/state.json'
state = json.loads(state_path.read_text())
assert state['status'] == 'complete' and len(state['results']) == 60
series = [r for r in audit['descriptive_learning_progress']['series'] if r['scope'] == 'ordinary_tasks']
observed = []
for reward in ('R0', 'R1', 'R2', 'R3'):
    for mode in ('greedy', 'sampled'):
        arms = sorted((r for r in series if r['reward'] == reward and r['mode'] == mode), key=lambda r: r['initialization_seed'])
        assert [r['initialization_seed'] for r in arms] == [0, 1, 2]
        qualifying = []
        for index in range(1, 5):
            seeds = [r['initialization_seed'] for r in arms if sum(n <= index for n in r['nodes_with_both_cap3_and_cap5_above_own_baseline']) >= 2]
            if len(seeds) >= 2:
                qualifying.append({'node_index': index, 'initialization_seeds': seeds,
                    'maximum_actual_decisions_at_node': max(r['points'][index]['decisions'] for r in arms if r['initialization_seed'] in seeds)})
        observed.append({'reward': reward, 'mode': mode,
            'first_majority_with_multiple_above_baseline_cap3_cap5_nodes': qualifying[0] if qualifying else None,
            'qualifying_initializations_at_final_node': [r['initialization_seed'] for r in arms if r['multiple_frozen_nodes_above_baseline']]})
diags = []
for f in audit['curve_files']:
    path = root/f['path']
    assert sha(path) == f['sha256']
    curve = json.loads(path.read_text())
    name = curve['experiment_id']
    candidates = [root/'artifacts/t5/perf'/f'{name}_{stage}_diagnostics{suffix}.json'
                  for stage in ('complete', 'completed') for suffix in ('_compact', '_summary', '')]
    diagnostic = next(p for p in candidates if p.is_file())
    data = json.loads(diagnostic.read_text())
    assert data['state_sha256'] == curve['original_state_sha256']
    original = root/curve['original_state_path']
    assert sha(original) == curve['original_state_sha256']
    original_state = json.loads(original.read_text())
    last = original_state['update_history'][-1]
    summary = data['summary']
    fields = ('episodes', 'decisions', 'won', 'zero_tick_fraction', 'immediate_plant_shovels', 'immediate_shovel_fraction_of_plants',
              'mc_value_mse', 'mc_value_explained_variance', 'max_abs_shaping_telescoping_error')
    diags.append({'experiment_id': name, 'source_path': str(diagnostic.relative_to(root)), 'source_sha256': sha(diagnostic),
        'original_state_sha256': curve['original_state_sha256'], 'training_summary': {k: summary[k] for k in fields},
        'last_update': {'update': last['update'], 'counters': last['counters'], 'losses': last['losses']},
        'last_update_note': 'Last update may contain only one episode at the frozen decision boundary; not a full-run or learning-quality statistic.'})
report = {'schema_version': 1, 'status': 'complete_analysis_locally_not_delivered',
    'scope': 'All12 corrected old-encoder reward candidates; cap1-only training, ordinary cap3/cap5 validation transfer',
    'curve_delivery_audit': str(audit_path.relative_to(root)), 'curve_delivery_audit_sha256': sha(audit_path),
    'reevaluation_state_sha256': sha(state_path), 'completed_nodes': 60, 'evaluated_jobs': 268800,
    'allowed_conveyor_rows_changed': audit['conveyor_changed_rows'], 'win_labels_changed': audit['win_label_changes'],
    'non_conveyor_rows_changed': audit['non_conveyor_changed_rows'],
    'baseline_full_row_differences': sum(p['full_row_differences'] for p in audit['paired_actual_baselines']),
    'baseline_paired_rows_compared': sum(p['rows_compared'] for p in audit['paired_actual_baselines']),
    'observed_repeatability_budgets': observed,
    'budget_derivation': 'At each frozen post-training node, count initializations with at least two prior-or-current nodes at which BOTH ordinary cap3 and cap5 wins exceed their own actual baseline. Report the first node with at least two such initializations and their largest actual decision count. Descriptive goal-related accounting, not an added significance test.',
    'future_B_frozen': False,
    'future_B_reason': 'Common-input/reference256 runtime and actual recovery gate, common-pool dynamic-prefix checks and comparable real-wall protocol remain incomplete. Records measured old-model onset without freezing unmeasured new-model runtime or declaring reward winner.',
    'completed_node_reevaluation_wall_sum_seconds': sum(r['reevaluation_seconds'] for r in state['results']),
    'timing_scope': 'Sum of completed-node recorded wall; excludes unrecorded abandoned work before interruption, waiting and analysis; not original-plus-resumed end-to-end wall. Original training wall separate in each curve.',
    'max_process_tree_RSS_bytes': max(r['resources']['peak_process_tree_rss_bytes'] for r in state['results']),
    'min_system_available_bytes': min(r['resources']['min_system_available_bytes'] for r in state['results']),
    'diagnostic_sources': diags,
    'interpretations': [
        'R0, R1 and R3 have ordinary multicap validation wins at multiple frozen nodes in at least two initializations. This does not support the claim that this environment cannot learn.',
        'R0 seed0 cap5 greedy275->76 and sampled143->104 at375k->500k; R0 seed1 greedy182->1 but sampled100->165; R1 seed0 greedy186->1 but sampled225->224. Greedy and sampled decline are distinct observed phenomena, without causal proof of forgetting.',
        'R2 ordinary multicap progress only in initialization1; R3 initialization2 has no ordinary multicap validation wins despite two successful initializations. Removing discount or shaping does not guarantee repeatability.',
        'R3 final greedy wins favor night/fog with weak day/pool and roof. Averaging hides terrain failures; no universally best reward declared.',
        'Original value/advantage and planting/shoveling diagnostics retained. Phi telescoping near machine precision does not support spurious net episode reward from immediate loops; excessive loops can consume decisions and damage learning.',
        'Old-encoder transfer does not establish new-input/curriculum/event-wait benefit, reference256 recovery, architecture comparison or complete-level acceptance. No thresholds/tasks/seeds changed.'],
    'HF_dry_run': {'status': 'pass', 'files': 268984, 'bytes_note': '148.2 MB reported by hf_sync',
        'log': 'logs/t5_research/reward_v2_packet_cost_reevaluation_v2_hf_dry_run.log', 'uploaded': False,
        'scope': 'Closed corrected evaluation tree; original models/training shards/training logs need separate run uploads.'},
    'git_delivery_complete': False, 'T5_E_gate_passed': False, 'goal_complete': False,
    'analysis_generator_sha256': sha(Path(__file__))}
destination = root/'artifacts/t5/perf/reward_v2_corrected_reward_analysis_v1.json'
assert not destination.exists()
destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')

lines = ['### 完整修正复评与奖励结论（2026-10-02）', '',
    '同协议复评v2已完成60/60检查点、35原任务×64原种子×greedy/sampled×12候选×5节点，共268800逐局结果。逐项核验全部原检查点SHA、12个原训练状态SHA、全部原/新逐局文件SHA、任务/种子和Wilson；原训练与旧失败v1未修改。576行变化全部限于已有原生依据的7个传送带任务，其中105行胜负改变，实际非传送带变化0。三初始化各R0/R1、R0/R2、R0/R3实际未训练基线共40320配对行，完整返回字段差异0，补上此前旧二进制空槽费用造成的基线不确定性。', '',
    '12份修正曲线另存artifacts/t5/curves/*_packet_cost_v2.json，未覆盖原曲线。独立导出器research_corrected_curve_delivery.py从完整raw重新算Wilson及终局指标，另存普通关卡摘要与相邻节点同task/seed的保留/丢失/新增胜局。审计见[reward_v2_corrected_curve_delivery_v1.json](artifacts/t5/perf/reward_v2_corrected_curve_delivery_v1.json)，结论/诊断来源见[reward_v2_corrected_reward_analysis_v1.json](artifacts/t5/perf/reward_v2_corrected_reward_analysis_v1.json)。', '',
    '下表为普通关卡诊断；完整35任务仍在全部原始结果和曲线中，没有删掉传送带任务。三波每格640局，五波每格576局。向量依次为未训练/125k/250k/375k/500k，实际超出决策边界的计数逐节点保留。', '',
    '| 奖励/初始化 | 贪心三波胜局 | 贪心五波胜局 | 采样三波胜局 | 采样五波胜局 |', '|---|---|---|---|---|']
for reward in ('R0', 'R1', 'R2', 'R3'):
    for seed in range(3):
        arms = {r['mode']: r for r in series if r['reward'] == reward and r['initialization_seed'] == seed}
        cells = ['/'.join(str(p[f'cap{cap}']['won']) for p in arms[mode]['points']) for mode in ('greedy', 'sampled') for cap in (3, 5)]
        lines.append(f'| {reward}/{seed} | '+' | '.join(cells)+' |')
lines.extend(['', '500k普通五波终局及95% Wilson区间：', '',
    '| 奖励/初始化 | 贪心：胜局/576；95%区间 | 采样：胜局/576；95%区间 |', '|---|---|---|'])
for reward in ('R0', 'R1', 'R2', 'R3'):
    for seed in range(3):
        arms = {r['mode']: r for r in series if r['reward'] == reward and r['initialization_seed'] == seed}
        cells = []
        for mode in ('greedy', 'sampled'):
            value = arms[mode]['points'][-1]['cap5']
            lo, hi = value['wilson_95']
            cells.append(f"{value['won']}/576；{100*lo:.2f}%–{100*hi:.2f}%")
        lines.append(f'| {reward}/{seed} | '+' | '.join(cells)+' |')
lines.extend(['',
    '这些结果支持“旧编码器PPO在此环境能够学到可重复的多波迁移”，不能据原失败宣布整体设计致命。R0、R1、R3均有至少两个初始化在多个原定节点超过自身普通多波基线；R0/R1采样三初始化都重复成功，R2只有初始化1重复成功。训练只练五个cap1任务，三/五波属于验证迁移，尚非正式多波课程或完整关卡能力。', '',
    '实测进步预算现已记录：R0贪心和采样第一次有两个初始化各在至少两个预定节点同时超过自身三/五波起点，为375k节点，二者最大实际375107决策；R0采样三初始化满足该描述为500k节点。新输入/新课程的B与真实等墙钟仍须在256参考恢复及端到端成本完成后冻结，不以旧模型耗时外推，预算分析不把临时门禁改成通过。', '',
    '退化没有被空槽修复消除。R0/init0普通五波375k→500k：贪心275→76（丢225、增26），采样143→104（丢92、增53）；两种模式都下降。R0/init1贪心182→1（丢182、增1），采样100→165（丢39、增104）；R1/init0贪心186→1（丢186、增1），采样225→224（丢71、增70）。后两者不能描述为所有获胜行为全部消失；后续需用冻结历史/概率边际诊断argmax切换与策略分布退化，当前末段探针是非部署的只读证据，不能单独作因果归因。', '',
    'R3/init0与init1贪心普通五波均249/576，但优势主要在night/fog，roof全部0；初始化2普通三/五波两模式全部0。因此无折扣塑形有两个积极重复，仍未解决跨初始化失败及五地形学习；不能把平均值选为普遍最佳奖励。R2去塑形且无折扣也不是通用修复。下一项奖励/课程选择须保持同一共同输入、任务池和PPO，先比较均衡采样与进步优先，再评援助撤除；不同时改多个因素。', '',
    '原训练诊断保留：R3/init2训练18/6005胜、立即种铲占plant39.17%、MC价值MSE0.0841而EV−0.8565；小MSE不代表有胜利信号。R0/init0训练6316/11151胜、零tick29.36%、立即种铲6.78%，也有末段迁移退化。实际Phi望远镜抵消误差约1e−15，不支持“种铲无限创造整局净奖励”的解释；循环仍浪费决策预算。全部12诊断来源SHA、末update的loss与样本规模注释已放入结论JSON，末update有时仅1局，不将它当完整阶段损失。', '',
    f"完整节点记录的复评墙钟之和{report['completed_node_reevaluation_wall_sum_seconds']:.3f}秒；不包含中断前未记录工作、等待间隙或分析，不冒充重启前后总墙钟。八worker的进程树RSS峰值{report['max_process_tree_RSS_bytes']}字节、系统available最低{report['min_system_available_bytes']}字节；树RSS含共享页重复计数。HF完整树空跑268984文件/148.2MB通过，未上传；原训练run的模型/原分片/日志仍走独立HF同步。", '',
    '当前HEAD仍7f2b094；新曲线、分析、3840局池报告与两个正式释放协议均仅本地。GPU仍不可见，256参考两臂未启动；.git只读，GitHub写入被工具审批阻止，原文“MCP tool call requires approval, but approval policy is never”。未绕过写入限制，未把新实验先跑后补发布。事件等待继续按既定工程门禁→共同基线后独立三初始化对照→T6前合并评估的清单推进。整体T5-E/B/T6/T7与HF交付仍未标记完成。', ''])
fragment = Path('/tmp/pvz_corrected_reward_report_v1.md')
fragment.write_text('\n'.join(lines))
print('analysis', destination.name, destination.stat().st_size, 'report', fragment)
print('observed', observed)
