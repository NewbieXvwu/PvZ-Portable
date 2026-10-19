# 老师候选覆盖：默认关闭的学习式补充候选

主预算2对B类有部分覆盖收益：显著见证进入候选从12/236（5.08%）升至88/236（37.29%），最终选中29/236（12.29%），实际续走增加24.15%；A类仅48/83→49/83，最终选中仍为0，不能声称修复了贴脸雷。次要预算4的B覆盖165/236（69.92%），续走增加43.77%。

新老师换seed一致率也明显降低（预算2为68.50%，旧老师77.50%），因此建议维持默认关闭，供主控后续新标签试线评审，不直接推广。预算2/4均按预登记完整报告；未依据600点调模型或选预算，未做新胜率实验。

用法：在原 search_play.py 命令追加：

```bash
--candidate-coverage-model /home/newbiexvwu/pvz-h8/artifacts/cuda/task11/proposer/model.pt \
--candidate-coverage-budget 2
```

不传 checkpoint 即关闭；预算0也不追加。模型与原卡组、v29续走相配，换学生/任务需重验。检查点位于本机ignored artifacts，未入git，SHA见冻结文件。

**600/600旧标签及四位小数targets精确复现。** 原候选完整保留为有序前缀；新增合法、唯一、每点不超过预算；同老师换future seed候选不变，预算4前两个追加与预算2相同。所有断言逐点通过。

## 显著替代的覆盖与选择

沿用原task9的A83/B236个显著见证（原witness_action，未在本次重选）；新老师续走仍v29，其他正式参数不变。区间为Wilson95%。

| 类别 | 老师 | 见证进入候选 | 最终选中见证 |
|---|---|---:|---:|
| A | old | 48/83=57.83% [47.09%, 67.88%] | 0/83=0.00% [0.00%, 4.42%] |
| A | new2 | 49/83=59.04% [48.29%, 68.99%] | 0/83=0.00% [0.00%, 4.42%] |
| A | new4 | 49/83=59.04% [48.29%, 68.99%] | 0/83=0.00% [0.00%, 4.42%] |
| B | old | 12/236=5.08% [2.93%, 8.68%] | 0/236=0.00% [0.00%, 1.60%] |
| B | new2 | 88/236=37.29% [31.37%, 43.62%] | 29/236=12.29% [8.69%, 17.09%] |
| B | new4 | 165/236=69.92% [63.78%, 75.41%] | 80/236=33.90% [28.16%, 40.15%] |

这里衡量能否看见/选择原较优替代，不重做64未来显著性检验，也不证明该替代在v29下仍显著，更不证明胜率改善。主控512局反事实干预目前没有可测胜率/丢车差别；新标签训练和胜局验收由主控负责。

## 实际成本与标签变化

600点来自361整局，A/B各300。下表均值与区间按整局聚类bootstrap10000次；成本为实际C++续走调用计数，含分阶段幸存者和实际触发的两边V9验证。

| 老师 | 实际续走/点，95%CI | 相比旧老师增幅，95%CI | worker+模拟器CPU秒/点 |
|---|---:|---:|---:|
| old | 20.867 [20.096, 21.648] | — | 0.6075 |
| new2 | 25.907 [25.191, 26.615] | 24.15% [21.89%, 26.68%] | 0.7275 |
| new4 | 30.000 [29.311, 30.677] | 43.77% [40.65%, 47.13%] | 0.8380 |

CPU时间是六臂固定顺序、并行背景负载下的描述值，未做专门ABBA性能采用门禁；主要成本结论使用实际续走数。此样本只覆盖两类审计局面，不外推为生产整局平均成本。

| 对照 | 标签一致率与整局95%CI |
|---|---:|
| 旧/旧换seed噪声 | 77.50% [74.05%, 80.88%] |
| 旧/new2同seed | 80.67% [77.59%, 83.69%] |
| new2/new2换seed噪声 | 68.50% [64.62%, 72.19%] |
| 旧/new4同seed | 72.17% [68.77%, 75.49%] |
| new4/new4换seed噪声 | 67.17% [63.31%, 70.88%] |

追加空间覆盖（每个局面的集合差再求和，不重复计算同点同格）：

| 类别/预算 | 新动作 | 新格子 | 新行 | 新列 | 追加动作按阳光成本计数 |
|---|---:|---:|---:|---:|---|
| A/new2 | 600 | 459 | 199 | 217 | 25:152, 50:96, 100:350, 175:2 |
| A/new4 | 1199 | 949 | 344 | 343 | 25:317, 50:190, 100:690, 175:2 |
| B/new2 | 600 | 593 | 123 | 540 | 50:584, 100:16 |
| B/new4 | 1200 | 1191 | 151 | 1048 | 50:1168, 100:32 |

旧/新一致率减去旧老师噪声基线（百分点，配对整局95%CI）：

- new2: +3.17pp [-1.47, +7.85]pp。
- new4: -5.33pp [-10.15, -0.49]pp。

新增候选老师自身换seed一致率相对旧老师的变化（百分点，配对整局95%CI；同一600点，追加派生比较）：

- new2: -9.00pp [-11.92, -6.10]pp。
- new4: -10.33pp [-13.55, -7.24]pp。

覆盖扩大同时带来更低的标签自一致性；本次保持原K2/V9/3futures等参数不变，未通过加验证样本掩盖这项代价。
这次有意更改候选覆盖，不把task8的3pp近似不变门槛当成采用门槛；噪声对照用于识别标签变化幅度。分类结果、六臂总续走、追加成本/卡槽/空间分布见完整验证JSON。

## 训练、冻结与限制

模型输入为既有公共状态编码和任务context，加动作成本、学生/静态分数、各自在同卡内的排名与最佳分差、通用格子/卡槽one-hot。当前候选经学习的动作嵌入mean/max pooling编码，预测追加动作的增量覆盖值；每追加一个候选重新预测。没有植物类别分支、列阈值或手调成本/分歧组合权重。空间互补由集合条件目标学习，未硬写距离优先级。

训练目标是3个共同未来上，动作叶值超过现有集合最高叶值的正差均值。三个上下文为旧候选、旧候选+训练oracle、旧候选+随机动作。训练和验证来自未入600点的独立23局，18训练/5验证，各24局面，共552点、85374实际续走，源档案SHA全部未变。固定80epoch、Adam0.001、weight_decay0.0001、batch8contexts，按验证MSE最低冻结；不使用验收点拟合。

验证MSE=7.0159142e-05，零增益基线=7.151981e-05，改善1.90%，学习信号较弱。模型1,553,179字节，冻结于2026-10-20T04:32:49.273407+08:00，SHA `b3fe42d9bbd7990f2aad908ebbe16ea6a38ea5b5baed69bcfc2db187f9aaddc2`。不得把小验证集上的MSE改善写成泛化/胜率结论。

证据：[预登记](teacher-candidate-coverage-plan.json)、[训练与模型冻结](teacher-candidate-coverage-freeze.json)、[完整600点验证](teacher-candidate-coverage-validation.json)。原始点记录、数据、命令和资源日志位于 `artifacts/cuda/task11/`。

## task10

`RolloutCuda --summary-json PATH`与bench文件解析已完成。旧/新程序8/30道各3批，结果和leaf逐字节一致，输出计数/维度/长度/错误位均通过；同次JSON计时与旧console解析完全一致。GPU共享期间只核验正确性，不比较速度，bench独占门禁保留。见 [原生核验](cuda-summary-validation.json)。

## 复现入口

在 pvz-h8 工作树，以单线程数学库环境运行：

```bash
export PYTHONPATH=python:scripts
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
~/.venvs/ml/bin/python3 scripts/teacher_candidate_coverage.py collect \
  --selection artifacts/cuda/task11/training_selection.json \
  --out artifacts/cuda/task11/dense_train --workers 4
~/.venvs/ml/bin/python3 scripts/train_candidate_coverage.py \
  --data artifacts/cuda/task11/dense_train \
  --selection artifacts/cuda/task11/training_selection.json \
  --out artifacts/cuda/task11/proposer
~/.venvs/ml/bin/python3 scripts/teacher_candidate_coverage.py audit \
  --selection artifacts/cuda/task9/native31_selection.json \
  --model artifacts/cuda/task11/proposer/model.pt \
  --out artifacts/cuda/task11/acceptance --workers 4
~/.venvs/ml/bin/python3 scripts/teacher_coverage_stats.py \
  --audit artifacts/cuda/task11/acceptance \
  --witnesses artifacts/cuda/task9/point_statistics.json \
  --out MEMORY/docs/teacher-candidate-coverage-validation.json
```

实际执行由 `artifacts/cuda/task11/pipeline.py` 和 `collect_driver.py` 包装内存监视，每秒检查自有子树；低于2GiB暂停、恢复至3GiB后继续，原始命令、RSS、swap、暂停记录及返回码同目录归档。不要直接把上述命令当成省略资源闸门的理由。检查点、npz、滚动输出不入git。

所有数值只针对task9抽样的两类局面；不外推为生产整局平均成本，也不证明新标签能提升胜率。模型目前只在原day_l7_z17七卡组/v29学生下验收，换学生或任务需要重验。



## 执行与资源

600点、361局严格重放226,902帧，六臂实际运行92,293次续走；未缓存冒充实测。30项定向回归无skip，CLI关闭/预算2/预算4各1次原生命令行搜索通过（max-actions1，只验接口）。冻结代码、模型和输入SHA验后仍全等，361验收源档案及23训练源档案未变。最多4自有游戏进程；被监视子树RSS峰3.57GiB、swap0，可用最低3.97GiB，暂停样本数0，所有实验已退出。见 [执行明细](teacher-candidate-coverage-execution.json)。
