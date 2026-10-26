# task96：r11 重训复现审计

2026-10-26，repro 分支。生产代码、qc、门禁、其他工作树均未修改。本文的确定性诊断尚在等待 qc 释放 GPU，最终算子与成本结论待补。

## 已有证据支持什么

- **存在同 seed 训练分叉。** CodeBuddy 已完成的同数据、同配方、seed=1、1 epoch 双跑，21/21 个实际权重张量不同，val loss 为 3.7510763180 / 3.7643170536。不能用 `.pt` 文件 SHA 不同代替张量比较；本次两者都核验了。
- **现存输入没有发现标签漂移。** r11 与 A 的 348 标签逐项行数、训练/验证分配一致；CodeBuddy 已比较 348 个引用段 SHA256，全部相同。追加元数据审计显示段大小全同、不是硬链接，r11 的全部段 mtime 均早于训练。
- **“r11 使用 930d776d，训练器重写导致落差”是错误归因。** 该旧版本没有 r11 已使用的 recipe/identity 功能。部署日志定位当时部署版本为 **4f4d217b**，与 A 的 **bec2b9e6** 相比，训练器源码、配方解析、pack 指纹算法、value encoder 完全相同；网络类 AST 也相同。
- 仍不能把 heldout 的 **166→124 / 192** 全部归因于训练随机性，更不能从两次 1 epoch loss 推算胜率方差。r11 的逐次运行源码/环境回执缺失；task84 的多种子完整训练及配对评估才负责量化泛化波动。

## 数据与冻结证据

两轮总计 16,982,931 行，6,605,039 株，55,773 个 seed，验证 1,704,229 行。两份 `source_rows.json` 的文件 SHA256 都是：

`fd0f9c12901b4e4c7015cdf40515b3167c6ba39c791d86145e441d93f1500039`

名义 corpus index 为 `7d15b915a2192805.npz`。r11 侧 index 已被正常 prune，A 侧是 81KB 的段索引，不是损坏或缺数据。A 索引 SHA256：

`1351618d6c10459e9d30b5a2aa70a3a73ddbb05db239fc01bab75a5b6b02b4cb`

逐标签段名、archives/rows/training_rows/validation_rows、大小、双方纳秒 mtime、是否独立 inode 见 [task96-data-metadata.json](task96-data-metadata.json)。348 段内容 SHA 相等沿用 CodeBuddy 在 `9ffc611a` 已完成的 ~9.6GB 检查，本次没有重复全量哈希；该次未提交逐段摘要清单，这是证据保存限制。现存文件与较早 mtime 支持没有漂移，但不能补造训练启动时的不可变内容快照。

`archive_fingerprint` 实际哈希相对路径、文件大小及可选 `training_rows`；故意不含 mtime，也不含文件内容 SHA。pack 名再由 sources 与 labels/wait_keep/obs/cells/aux/packet_head/context/label_weight 等选项导出。**同名只能证明选项/路径/大小摘要相同，不能证明内容相同**；等长重标可能逃过它。

## 代码来源与差异

`qc_setup.log` 记录 10-24 19:47 部署 4f4d217b，21:35 stage4 恢复后重启；下一次 deploy12 是 10-25 10:48。r11 训练在 10-25 08:30–09:12，其后评估回执中的 `search_play.py`、`pvz_promotion.py`、`promotion_gate.py` 三个源码 SHA 都与 4f4d217b 一致。CQ 分支 HEAD 没有因按路径 checkout 而移动，所以只看 cq reflog 会误判。

| 训练链文件：4f4d217b → bec2b9e6 | 实际差异与本次相关性 |
| --- | --- |
| `scripts/train_rollout_policy.py` | 完全相同，SHA `5cc2f20803dc3f525640c76194d0ab6d70b51419d32352719e3dc09c93bc5b4c` |
| `python/pvz_recipe.py`、`pvz_training_pack.py`、`pvz_value_net.py` | 完全相同 |
| `scripts/pack_training_rows.py` | 普通 seed 解析改为 `archive_seed`，兼容 `seed_<demo>_f<future>`；本次直接读已生成且相同的段 |
| `python/pvz_frame_store.py` | 新增 archive_seed、v2size 注册、冻结 fog 签名；encode_frames/encode_student_frames/target_values AST 相同 |
| `python/pvz_rollout_policy.py` | 只改 v2size 原生导出版本号；RolloutNet/features AST 相同 |
| `python/pvz_student_observation.py` | 新增 v2size；raw zombie 三字段读取兼容追加字段；相同已编码段不重编码 |
| `python/pvz_training_rows.py` | SourceRowCounts 使用 archive_seed；完整 diff 见证据，无优化器改动 |

[task96-code-provenance.json](task96-code-provenance.json) 保存部署日志关键行、完整训练链 diff、源码 SHA 与评估交叉核验结果。部署记录比主检出 HEAD 更可靠，但没有训练启动时的逐文件 receipt，无法绝对排除未记录的临时覆盖。此处明确区分已记录部署版本与逐次实际运行源码。

`encoder_signature` 只覆盖 features/featurize_numpy/_col_of/target_values/encode_frames 以及选定数值常量；v2 又有冻结语义签名。它不是训练链签名，不覆盖 optimizer、训练循环、所有 merge/read 路径、网络实现或运行库。即使未来 (c) 发生，它也不保证发现。

## 已完成的全量双跑

复用 CodeBuddy 的两个产物，没有重跑 1698 万行。七个训练链源码文件逐字节匹配 bec2b9e6。两次除输出路径外的配方相同。

| 项 | run1 | run2 |
| --- | ---: | ---: |
| seed / epoch | 1 / 1 | 1 / 1 |
| train loss | 4.2911643558 | 4.3041004138 |
| val loss | 3.7510763180 | 3.7643170536 |
| epoch 秒 | 890.387 | 334.052 |
| 包含读取的总秒 | 1224.317 | 591.676 |

规范张量 SHA（按名称、dtype、shape、实际数值序列化）：

- run1 `5d52761abc8b3240506771a252beee316312ab7d066500ce266db5392bc5722e`
- run2 `148fa6bef2eef889a98a0ae62f3fbdf3d4e441c60d8f16db4750c4c0f8d46f6e`

逐张量差异与原始配方/metrics 见 [task96-existing-runs.json](task96-existing-runs.json)。第一轮 epoch 在 15 分钟内，但含加载共 20.4 分钟，超过任务要求的总时限；没有为了合规而重复实验。两轮未捕获 deterministic 标志及启动环境，资源竞争不同，**1224/592 或 890/334 不是确定性训练成本**。

训练没有 DataLoader。`precompute_workers=8` 指预计算，并非训练 DataLoader worker；读现成 pack 时不是训练洗牌来源。入口先 `torch.manual_seed(seed)`，训练每 epoch 用 CPU `torch.randperm`；验证由排好序的 seed 集与私有 `random.Random(validation.seed)` 划分。受限诊断会记录初始参数和实际 permutation SHA，检验“同 seed”是否真的给出同初始化与同顺序。

## 受限 GPU 算子诊断（排队中）

脚本 [task96_determinism_probe.py](../../scripts/task96_determinism_probe.py) 使用上述冻结训练器和一个真实标签段：`n20o1_fast_search`，2368 行、32 seeds，段 SHA `806fbdfce4ffa8003eb9a84433a6b911edd3c060ea6cd388864863de23527854`。输入复制并验证到 repro 自己的 ignored artifacts，加载器只替换成读这个段；其余调用实际 trainer.main，batch=128，1 epoch。该实验用来定位算子/种子，不能替代全量训练或泛化测量。

按顺序启动独立子进程：strict 无 workspace → default1 → strict1 → strict2 → default2。strict 设置 deterministic algorithms、cuDNN deterministic、benchmark=False；除首个报错探查外，在 import torch 前设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`。记录实际 traceback、每步 loss、初始/最终张量摘要、permutation 摘要、环境与耗时；ABBA 次序减少首轮启动偏差。

每子进程最多150秒、总训练主动时间上限750秒；同任务最多一个 GPU 训练，qc 正在 GPU 训练时排队，运行中检测到 qc 就退出让行。诊断不启动游戏。结果到齐后补齐报错位置、复现性及成本；含每步 qc 准入检查的短样本耗时只代表该诊断，不能外推全部语料的生产开销。

## 冻结与修法建议（生产未改）

1. 每轮先固定标签快照；对 frames/meta/显式训练行清单以及 pack index 和每个引用段保存内容 SHA256。内容清单写入 `round_*/inputs.json`，训练前后核验。不把路径、大小或 mtime 当内容身份。现有不可变段可按内容地址复用，避免重复复制整库；保留到评估回执已固定其依赖。
2. `round_*/runtime.json` 记录实际命令、resolved recipe SHA、Git revision、dirty diff SHA、训练链各源码 SHA（含导入依赖），以及 Python/numpy/PyTorch/CUDA/cuDNN/驱动/GPU、精度与确定性设置、CUBLAS workspace。配方来源 revision 与实际训练运行 revision 分开保存。
3. 显式独立的 CPU generator 管理模型初始化与训练 permutation，验证 split 使用独立 seed；存启动 generator 状态摘要、初始化张量 SHA、各 epoch permutation SHA。以后即使引入 DataLoader，再显式配置 generator/worker_init_fn 以及 Python/numpy RNG；当前不存在 worker 需要补种子的证据。
4. 验收比较实际张量与 loss，模型文件 SHA另作文件完整性校验；不能因 metadata/输出路径差异误判权重不同。相同库/设备下检查逐位复现，跨版本/跨设备另设数值容差与行为评估。
5. task92 可以处理可测的训练方差，但不能代替数据/代码/环境冻结。先把确定性诊断完成并测清作用；对于 42/192 的 heldout 落差，仍需多个完整训练及新种子配对评估，不能用单个 r11 称“幸运”就认定原因。

## 复核命令

所有新写入均在 repro 内；`/tmp/exp96` 只是 CodeBuddy 已留的冻结只读训练代码快照。执行前保持 qc GPU 优先。

```sh
/home/newbiexvwu/.venvs/ml/bin/python3 -B /home/newbiexvwu/PvZ-search/.claude/worktrees/repro/scripts/task96_existing_runs.py
/home/newbiexvwu/.venvs/ml/bin/python3 -B /home/newbiexvwu/PvZ-search/.claude/worktrees/repro/scripts/task96_provenance.py
/home/newbiexvwu/.venvs/ml/bin/python3 -B /home/newbiexvwu/PvZ-search/.claude/worktrees/repro/scripts/task96_determinism_probe.py prepare
/home/newbiexvwu/.venvs/ml/bin/python3 -B /home/newbiexvwu/PvZ-search/.claude/worktrees/repro/scripts/task96_determinism_probe.py run
```
