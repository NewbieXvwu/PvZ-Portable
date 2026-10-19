# 老师候选覆盖：默认关闭的学习式补充候选

验收进行中，尚无采用结论。生产默认仍是原来的候选集。

enable：在原 search_play.py 命令追加 `--candidate-coverage-model /home/newbiexvwu/pvz-h8/artifacts/cuda/task11/proposer/model.pt --candidate-coverage-budget 2`；不传 checkpoint 即关闭。4 是预登记次要预算，不根据600点挑选后重报。

方案和时间戳见 [预登记](teacher-candidate-coverage-plan.json)。独立23局，18训练/5验证，各24搜索局面，共552；与task9验收的361整局种子无交集。每个合法植物动作和等待都采3个共同未来，续走v29、叶值v3、静态branch1、2500tick、CPU mask61。小模型以状态、动作成本、学生/静态分数及卡内排名与分差、当前候选集合为输入，预测相对集合的实测增量覆盖值。空间互补由集合条件训练学习，不编码植物身份、列阈值、距离优先级或手调成本/分歧组合分数。

模型只追加合法、唯一的动作，原候选/策略锚点/剪枝/验证规则均保留。旧路径不加载模型、不新增推理，验收要求600历史标签及四位小数targets精确重现。检查点不进入git，冻结SHA随最终报告交付。

最终验收将填入原task9 A83/B236显著见证的进入候选/最终选择率、600点标签一致率、实际续走增幅、同老师换decision_seed的噪声底线及整局聚类区间。这里不声称胜率收益；主控512局反事实干预目前没有可测胜率/丢车差别，后续新标签训练与胜局验收由主控负责。

## task10

已完成 `RolloutCuda --summary-json PATH` 与 bench 文件解析；8/30道各3批的旧/新结果和leaf逐字节一致，同次JSON计时与旧console解析完全一致，见 [原生核验](cuda-summary-validation.json)。共享GPU期间只核验正确性，未比较速度；bench独占门禁保留。
