# 价值网络跨区加宽（task30）

本轮观察到白天保持、夜间与泳池净胜局增加；保留候选，建议主控扩大样本验收。
默认值保持不变。每关 32 对无法把“不显著”解释为已经证明非劣；是否切换由主控决定。

## 数据与隔离

原始外部体检 1360 局；加上独立补采种子后，最终外部 A6 共 1384 局。每局每波取首个非终局帧，逐波计算 AUC；同一种子从拟合、停止及初始化历史中整体排除。单一结果类别的 AUC 保留为空。
现有 L7 z13/z15 分叉 354005 行，97 文件、352 种子；整局特征 84736 帧。按种子哈希留出停止集；补采哈希余数 1/2 的 20% 种子另留外部 A6。训练批次与停止指标均按 75% 分叉 / 25% 整局结果混合。
v3 元数据中的 Mac 全部 93 万行原始分叉在本机不全，本轮使用可用本地分叉并从 v3 热启动。trained_on 包含初始化历史与本轮停止种子。补采老师与原有学生局分布不同，分别报告外部指标。

| 补采组 | 策略 | 胜/局 |
|---|---|---:|
| task30_pool24 | dw4i_s1 | 1/32 |
| task30_pool21 | identity_dw3_s1 | 8/16 |
| task30_night20 | dw4i_s1 | 19/32 |
| task30_day10 | dw4i_s1 | 32/32 |

## 编码与兼容

v2 跳过负植物类型；原版传送带手牌按引擎免费标志编码，同类多包按任一可用聚合。追加夜间、泳池、雾、屋顶、原版传送带、公开坟头数量 6 个标量。空槽实际污染旧 scalars[4]（总波数）和 scalars[58]（类型 53 就绪位）。
白天原版体检帧中，53.6% 含空槽，89.8% 含免费但原价阳光不足的手牌；2-10 分别为 25.1% / 98.0%。旧 v1 逐位及缓存签名保留。C++ 只做学生前向，价值前向在 Python，CPU 叶子按检查点编码；CUDA 预编码叶子仍只支持 v1，v2 提前拒绝。静态 branch1 仍固定旧编码。
新检查点编码 `pvz_value_net.featurize v2`，最佳 epoch 10，外部种子交集 0。7 项无游戏兼容/种子隔离/热启动门禁与独立 Release 编译通过。

## 外部校准与逐波排序

下表 Brier 汇总全部波首帧，受存活波数影响；逐波样本数、AUC、预测/实际胜率与校准分箱见 JSON。整局间 AUC 不替代同局候选排序或老师实战。

| 区 | 胜/局 | v3 Brier | v4 Brier | branch1 Brier |
|---|---:|---:|---:|---:|
| day_d7 | 182/256 | 0.169 | 0.136 | 0.180 |
| day_d7p | 172/256 | 0.145 | 0.099 | 0.146 |
| day_l7_control | 201/256 | 0.162 | 0.127 | 0.145 |
| day_original_conveyor | 48/48 | 0.044 | 0.009 | 0.013 |
| day_original_teacher_transfer | 7/7 | 0.042 | 0.006 | 0.011 |
| night_n9_n10 | 38/256 | 0.187 | 0.092 | 0.245 |
| night_original_conveyor | 0/32 | 0.015 | 0.057 | 0.111 |
| night_original_teacher_transfer | 3/3 | 0.246 | 0.129 | 0.034 |
| pool_p9k | 0/256 | 0.515 | 0.000 | 0.587 |
| pool_teacher_transfer | 5/14 | 0.483 | 0.257 | 0.484 |

| 区 | 波 | v3 AUC | v4 AUC | branch1 AUC |
|---|---:|---:|---:|---:|
| day_d7 | 5 | 0.762 | 0.811 | 0.688 |
| day_d7 | 10 | 0.640 | 0.777 | 0.589 |
| day_d7 | 15 | 0.570 | 0.749 | 0.575 |
| day_d7p | 5 | 0.847 | 0.870 | 0.814 |
| day_d7p | 10 | 0.867 | 0.936 | 0.900 |
| day_d7p | 15 | 0.829 | 0.922 | 0.880 |
| day_l7_control | 5 | 0.812 | 0.790 | 0.740 |
| day_l7_control | 10 | 0.861 | 0.848 | 0.833 |
| day_l7_control | 15 | 0.857 | 0.840 | 0.823 |
| day_original_conveyor | 5 | — | — | — |
| day_original_conveyor | 10 | — | — | — |
| day_original_conveyor | 15 | — | — | — |
| day_original_teacher_transfer | 5 | — | — | — |
| day_original_teacher_transfer | 10 | — | — | — |
| day_original_teacher_transfer | 15 | — | — | — |
| night_n9_n10 | 5 | 0.795 | 0.933 | 0.886 |
| night_n9_n10 | 10 | 0.827 | 0.920 | 0.870 |
| night_n9_n10 | 15 | 0.690 | 0.897 | 0.781 |
| night_original_conveyor | 5 | — | — | — |
| night_original_conveyor | 10 | — | — | — |
| night_original_conveyor | 15 | — | — | — |
| night_original_teacher_transfer | 5 | — | — | — |
| night_original_teacher_transfer | 10 | — | — | — |
| night_original_teacher_transfer | 15 | — | — | — |
| pool_p9k | 5 | — | — | — |
| pool_p9k | 10 | — | — | — |
| pool_p9k | 15 | — | — | — |
| pool_teacher_transfer | 5 | 0.756 | 0.822 | 0.711 |
| pool_teacher_transfer | 10 | 0.500 | 0.875 | 0.825 |
| pool_teacher_transfer | 15 | 0.700 | 0.733 | 0.867 |

| 区 | 胜/局 | v3 平均逐波 AUC | v4 平均逐波 AUC | branch1 平均逐波 AUC | 可算 AUC 的波数 |
|---|---:|---:|---:|---:|---:|
| day_d7 | 182/256 | 0.585 | 0.720 | 0.560 | 30 |
| day_d7p | 172/256 | 0.739 | 0.869 | 0.800 | 30 |
| day_l7_control | 201/256 | 0.848 | 0.834 | 0.809 | 30 |
| day_original_conveyor | 48/48 | — | — | — | 0 |
| day_original_teacher_transfer | 7/7 | — | — | — | 0 |
| night_n9_n10 | 38/256 | 0.757 | 0.920 | 0.834 | 30 |
| night_original_conveyor | 0/32 | — | — | — | 0 |
| night_original_teacher_transfer | 3/3 | — | — | — | 0 |
| pool_p9k | 0/256 | — | — | — | 0 |
| pool_teacher_transfer | 5/14 | 0.684 | 0.857 | 0.830 | 20 |

v4 在这些全败区把胜率压到近零：pool_p9k（实际 0/256，v4 平均预测 0.005）。校准方向正确（v3 高估到 0.5–0.7），但叶值几乎不随局面变化，同局候选的排序只能靠实战 A/B 检验；泳池老师数据（有胜有负）仍给出可算的排序信息。

## 老师同种子 A/B

固定 dw4i_s1 推演、beta=1、branch1 静态估值、verify=9、student-candidates=2，其余 queue103 默认；只换叶子 v3→v4。泳池 L21/L24 在任何 A/B 开局前固定，覆盖 20/40 波。每关同种子 32 对，双侧精确 McNemar。

| 任务 | v3 胜 | v4 胜 | 净胜局 | v4 独胜:独负 | p |
|---|---:|---:|---:|---:|---:|
| night_l14_eval_n10 | 29/32 | 31/32 | +2 | 3:1 | 0.625 |
| night_l17_eval_n10 | 22/32 | 31/32 | +9 | 9:0 | 0.003906 |
| night_l20_eval_n10_orig | 20/32 | 29/32 | +9 | 12:3 | 0.03516 |
| pool_l21_eval_p9k | 10/32 | 19/32 | +9 | 13:4 | 0.04904 |
| pool_l24_eval_p9k | 3/32 | 3/32 | +0 | 3:3 | 1 |
| day_l7_z15_eval | 32/32 | 32/32 | +0 | 0:0 | 1 |

本轮观察到白天保持、夜间与泳池净胜局增加；保留候选，建议主控扩大样本验收。

本轮同时改变编码与训练覆盖，收益不能单独归因于其中一项。A6 是不同整局间的胜负预测，老师实战才检验同局动作选择；本轮观察信号仍需扩大配对样本确认。

## 交付

候选已上传 HF `realnewbiexvwu/pvz-agent-artifacts/transfer/value_v4.pt`，下载回验 SHA256 `14c372e19dcd73fd7ee0493600907efefa0f7edafa7b884e1fdecd875008cd8d` 一致。编码、数据与配对工具在 vnet 分支；模型、帧、缓存与滚动日志不入 git。
结论层：[原始体检](value-widen-audit.json)、[补采](value-widen-collection.json)、[训练](value-widen-training.json)、[v4 外部评估](value-widen-v4.json)、[配对](value-widen-paired.json)、[HF 回验](value-widen-transfer.json)。
