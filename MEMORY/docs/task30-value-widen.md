# task30（主控 2026-10-22 19:00 下发给台式机 Codex 第二个会话 codex2，与 task27 并行）：价值网络跟上加宽——体检、重训、老师 A/B

先读仓库根目录 `AGENTS.md`（优先级最高），再读 DESIGN §13（价值网络）、§14.4–§14.6，`MEMORY/docs/student-bottleneck-plan.md` 末尾 10-22 的几条
（尤其 18:20 的槽号 / 类型审计结论），以及 `scripts/train_value_net.py`、`scripts/train_value_branches.py`、`python/pvz_value_net.py`。

## 背景
老师 = 学生推演 + 搜索；搜索的叶子估值与静态估值用价值网络（`--value value_net/v3`、`--static-value value_branch1`，见 `~/pvz-logs/queue103.sh` 的 `run()`）。
这两个网络只在**白天 L7（z13–z15 倍率）**的分叉数据上训过（v3 共 93 万行），从没见过夜间（坟头、蘑菇、阳光菇、无天降阳光）、泳池（6 行、水路）、原版传送带开局。
现在夜间 / 泳池 / 传送带的全部标签都靠它们。另有已知输入缺陷：旧编码 `features()` 把空卡槽 -1 写进 scalars[4]（总波数那格）与冷却末格；
传送带关 10 个槽常有空的，价值网络在这些帧上输入有误（编码器按签名冻结，旧网络不能直接改）。

## 要做的
工作区（台式机）：`git -C ~/PvZ-search worktree add ~/PvZ-search/.claude/worktrees/vnet -b vnet origin/pvz-env`，只在这里改代码、单独编译。
**同时最多 4 个游戏进程**（q103 用 16 个、task27 最多 4 个、第二条线最多 2 个）；GPU 训练前 `pgrep -f "train_rollout_policy|train_value"`，task27 的 dw5i 在训时不要同时训。

1. **体检（不跑新局）**：用 `~/PvZ-search/artifacts/student/frames_v2` 里已有的对局（结果已知），按区（白天 d7 / 白天原版传送带 / d7p / 夜间 n9·n10 / 2-10 原版 / 泳池 p9k）
   给 value_net/v3 与 value_branch1 算 DESIGN §13 的 A6 口径（每局每波一个样本、逐波 AUC、按种子分割）与校准（预测胜率 vs 实际），和白天 L7 对照。写清“出了白天还剩多少”。
2. **新编码版本**：在 `pvz_value_net.py` 加一个新编码版本（空槽跳过——与 C++ `EnvironmentRollout.cpp:194` 一致；需要的话加区/地形与原版传送带的标量），
   旧版本逐位不变（签名测试照旧通过），检查点里记录编码版本，C++ 侧若有价值网络前向要同步并做数值对拍。
3. **重训**：数据 = 已有 L7 分叉数据 + 各区的对局 / 分叉数据（先用 frames_v2 里现成的整局结果；若体检显示某区明显差，再用 `generate_value_data.py` / 分叉采集按区补，≤4 进程），训 value v4（新编码），按区报 A6 指标与校准，和 v3 并列。
4. **老师 A/B**：老师（dw4i_s1 推演，beta=1，其余参数照 queue103 的 `run()`）只换价值网络 v3 → v4，在夜间 2-4、2-7、2-10 原版与泳池两关上同种子配对（每关 32 局起，≤4 进程），报告胜局差与配对检验。
   v4 不劣于 v3（白天 L7 也要测，不能掉）且夜间 / 泳池更好，才建议换；**不改默认值**，由主控决定。
报告写 `MEMORY/docs/value-widen.md`。

## 规则
- **不要动任何 tmux 队列（q103 等）与别的代理会话；不要改别的工作树与 `~/PvZ-search` 主工作区**（frames_v2 只读）。
- 分支 `vnet` 提交，可以 `git push origin vnet`；不推 pvz-env、不合并。不改默认值；不写手工规则或公式；不写字面密钥（`$HF_TOKEN` / `$PVZ_HF_REPO`）；
  不 `git add -f`；`.pt/.npz/.log` 不进 git，价值网络传 HF `transfer/value_v4.pt`。
- 每完成一小步就提交，并在本文末尾“结果”里记一行；疑问写“问题”，按最佳判断继续，不要停下来等。额度用完由 CodeBuddy 接手，进度一定要落在提交和“结果”里。

## 结果

- 2026-10-22：首批 A6 离线体检完成（1312 局，固定种子、排除旧训练种子）；泳池 0/256、v3 Brier 0.515；报告与逐波指标见 `value-widen.md` / `value-widen-audit.json`。白天原版传送带待补齐，继续第 2 步并准备补充输赢数据。

- 2026-10-22：新价值编码 v2 与检查点版本分派完成；空槽跳过、传送带免费与同类手牌聚合，追加公开场景/坟头标量。旧 v1 逐位及签名不变，5 项无游戏测试通过。CPU 搜索与排序诊断按模型编码；CUDA 叶子仍仅 v1，v2 提前拒绝。

- 2026-10-22：离线体检补齐白天原版传送带 48/48，现共 1360 局；独立 Release 编译成功。2560 局 / 81920 帧训练特征准备完成，种子隔离、分叉迁移与热启动共 7 项测试通过；已启动 4 worker 老师补采，并冻结 6 关各 32 对 A/B 方案，尚未训练/跑 A/B。

- 2026-10-22：量化编码缺陷：白天原版 53.6% 帧含空槽、89.8% 帧有免费但原价阳光不足的手牌；97 份可用分叉共 354005 行、352 种子，审计重叠 0。补采扩展为旧 identity 老师 L21 + dw4i 原版传送带；任何 A/B 前把泳池固定为 L21/L24，另留 20% 补采种子作外部 A6。L24 32 局仍在跑，训练/AB 尚未开始。

- 2026-10-22：L24 p9k / dw4i_s1 老师补采完整 32 局完成，1/32 胜（2036144–2036175），已接着补旧 identity 老师 L21；详细采样汇总见 `value-widen-collection.json`。HF 仓库访问已验证，上传/下载哈希门禁工具就绪；仍未开始训练或 A/B。

- 2026-10-22：L21 p9k / identity_dw3_s1 老师补采 16 局完成，8/16 胜（2006144–2006159）；2-10 原版 32 局已启动。v4 检查点 trained_on 纳入初始化 v3 的全部历史种子，热启动泄漏检查已补强，5 项编码/训练数据门禁通过。

- 2026-10-22：外部评估器增加严格种子交集拒绝、模型 SHA256 与清单指纹；1360 局原始外部清单复核通过。2-10 补采先完成 4 局，3 胜 1 负；继续完整批次，尚未训练 v4。

- 2026-10-22：本任务本地串行执行器已运行：`/home/newbiexvwu/PvZ-search/.claude/worktrees/vnet/artifacts/task30/continue_collect.py`（补采）与 `/home/newbiexvwu/PvZ-search/.claude/worktrees/vnet/artifacts/task30/finish_pipeline.py`（封存测试种子→附加特征→v3 基线；等学生训练退出→v4 训练/离线评估→v4 配对）。重入先查这两个进程，勿重复启动；总进展看同目录 `pipeline.log`、`continue_collect.log`，每阶段仍需汇总并提交，执行器已接上 HF 上传/下载 SHA256 回验及每阶段报告/提交，代理继续核查最终结论。

- 2026-10-22：收紧 n9/n10 的标签解析，排除未来扰动卡组误入；1360 局冻结区组核对完全不变。2-10 补采进展 6/32、5 胜 1 负；继续固定配置采集。

- 2026-10-22：长任务完成钩子就绪：补采、训练、A6、最终 A/B 各阶段完成后自动写结果/记忆并只提交 md/json，候选自动上传 HF 并下载回验，最终生成完整 value-widen.md。未完成批次发布门禁用当前数据验证通过；当前仍在补采，尚无 v4 结果。

- 2026-10-22：自动完成钩子经隔离合成端到端验收：四阶段各一次仅文本提交，重复 final 不重复提交，真实报告未变；7 项原有门禁仍通过。当前真实 2-10 补采继续运行，尚无 v4 训练或配对结果。

- 2026-10-22 22:18 [auto-collection] 补采全部完成：112 局；外部 A6 1384 局；训练/停止特征 84736 帧。

- 2026-10-22 22:58 [auto-training] v4 训练完成：编码 v2，最佳 epoch 10，停止损失 0.3992；外部种子交集 0。

- 2026-10-22 23:00 [auto-audit] v4 / v3 / branch1 同一外部种子的逐波 AUC 与校准完成，详见 value-widen-v4.json。

- 2026-10-22 23:05 [publish] v4 训练完成（编码 v2，最佳 epoch 10，停止损失 0.3992，外部种子交集 0）并已上传 HF `transfer/value_v4.pt`，下载回验 SHA256 一致；同一外部 A6 上 v4 的 Brier 在白天/夜间/泳池老师数据全部优于 v3，白天 L7 逐波 AUC 0.834（v3 0.848）、其余区 0.720–0.920；泳池 p9k 0 胜区 v4 预测饱和在 0.005（std 0.011），需实战 A/B 检验。

- 2026-10-23 01:28 [ab-v3] 老师基线臂（v3 叶子）六关各 32 局完成：白天 L7 32/32、夜间 L14 29/32 / L17 22/32 / L20 原版 20/32、泳池 L21 10/32 / L24 3/32（平均 106–111 秒/局）。白天 L7 已顶到 32/32，只能检验不掉；v4 臂已同种子启动。

- 2026-10-23 04:00 [auto-final] 老师 A/B 六关各 32 对完成；白天净 +0、夜间净 +20、泳池净 +9。本轮观察到白天保持、夜间与泳池净胜局增加；保留候选，建议主控扩大样本验收。 HF 下载回验通过。

- 2026-10-23 04:12 [resume] 重启后续跑：核对 continue_collect.py / finish_pipeline.py 均已退出、补采与特征落盘完整（1384 外部局、84736 帧），按剩余阶段续跑。v4 训练在 22:57–22:58 完成，当时 `pgrep -f train_rollout_policy` 为空；dw5i 学生训练约 23:01 才重新出现，二者无 GPU 重叠。A/B 全程 4 个本任务游戏进程。

- 2026-10-23 09:36 [step5-a] 白天压力对 v3 臂完成：`day_l7_train_d7_m150` 32/32、`day_l10_orig_train_m150` 31/32（204 / 407 秒每局）。两关在 m150 下都接近顶格，仍只能检验“不掉”；已另冻结 m250 版（`day_l7_train_d7_m250` / `day_l10_orig_train_m250`）作为真正的压力补充，v4 臂已同种子启动。

- 2026-10-23 11:10 [step5-a] 白天压力对 v4 臂完成：`day_l7_train_d7_m150` v3 32/32 vs v4 32/32（净 0），`day_l10_orig_train_m150` v3 31/32 vs v4 32/32（净 +1，p=1.0）。m150 下两关都近顶格，结论仍是"不掉"；真正的压力补充在 m250 版计划（已冻结 `ab_day_hard`）。

## 问题

- 本地缺少 v3 记录的 Mac 端全部 93 万行原始分叉，重训先用现有 L7 z13 与 z15d 分叉；明确记录实际数据量。泳池现有整局数据几乎全败，不能报告未定义的 AUC 为 0.5，已补采老师数据。
- 任务书背景的‘冷却末格’按实际索引应为类型 53 的就绪位 scalars[58]；空槽同时污染 scalars[4]。v2 已按真实索引修复，v1 逐位/签名保留。

## 第 5 步（主控 2026-10-23 08:35 追加）：v4 能不能换上——补两项验证
报告收到：夜间净 +20、泳池净 +9（全来自 L21）、白天 L7 两臂都 32/32 顶格。顶格只能证明“没掉”，换上之前补：
1. **白天有压力下不掉**：老师（dw4i_s1 推演，beta=1，其余照 queue103 的 `run()`）v3 vs v4 同种子配对，`day_l7_train_d7_m150`（倍率族见 DESIGN §14.13，清单 `~/PvZ-search/.claude/worktrees/cq/artifacts/stage4/multiplier_panel.json`）与 `day_l10_orig_train_m150` 各 32 局；判据：v4 胜局不少于 v3 − 2 且配对检验不显著变差。
2. **泳池输局区叶值压平**（你报告里写的 0.005）：在泳池 L24/L26 上取 200 个搜索决策，比较 v3 / v4 对候选的排序区分度（候选间叶值标准差、top1 与 top2 差），说明压平是否让搜索在泳池退化成随机；若是，给出原因（训练数据里泳池几乎全输？）与修法建议（不要自己改默认值）。
3. 结论写进 `MEMORY/docs/value-widen.md` 新一节：建议“换 / 不换 / 分区换”。同时最多 4 个游戏进程，课程队列 qc 不要动。

- 2026-10-23 12:20 [step5-b] 泳池叶值压平探针完成（L24/L26 各 4 局、每臂 480 搜索决策）：v4 平均叶值 0.0018、候选间 std 0.0007（v3 的 1/49）、100% 决策 top1−top2 落在搜索自身 0.02 推翻阈值内、8 局零推翻（v3 推翻 19.8%）——压平成立，搜索退化为无条件跟随学生。原因：泳池训练标签几乎全败，v4 把该区学成常数 0；代价有界（探针 8 局两臂皆 0 胜，可赢的 L21 上 v4 净 +9），伤的是硬而可赢泳池局的挖掘能力。修法建议：补泳池决策级标签重训 v5，不要分区混用公式。结论节已写入 value-widen.md：建议整臂换 v4（附 CUDA 叶子仅 v1 的限制），m250 白天压力补充两臂已启动（workers=4）。

- 2026-10-23 13:05 [restart-resume] 12:40 重启杀掉了 m250 两臂；核对 v3 L7 已落盘 16/32 局（JSON 逐行完整、0 胜，与 m250 压力预期一致），search_play 按 (arm, seed) 去重原生续跑，无需作废。磁盘充足（848G）。v3 臂已用 run_in_background 后台续跑（workers=4，qc 队列未动），随后 v4 臂同种子补齐后出 paired.json。
- 2026-10-23 18:20 [step5-c] m250 v3 臂两关 32/32 完成（断点续跑计入原有 16 局）：L7 0/32 胜（m250 压穿老师，两臂预计都顶不住，配对只能证等劣）、L10 32/32 全胜（m250 仍有余力）。v4 臂同种子已启动（workers=4）。
- 2026-10-23 23:30 [step5-d] m250 配对完成（两臂各 32/32，complete=true）：L7 v3 0/32 vs v4 2/32（净 +2，p=0.5，无 v4 受损局）、L10 两臂 32/32 顶格（净 0，p=1.0）。m150+m250 判据全部满足，白天压力验证通过，value-widen.md 结论节已补记并移除回退条款。第 5 步三项验证全部完成，建议不变：整臂换 v4。

- 2026-10-23 15:50 [step6] 整臂换 v4 完成。合并：`git merge --no-ff origin/pvz-env` → 合并提交 `335bfc48`（主线 203 个提交进 vnet）。冲突 4 处按主控规则解：两份 MEMORY 日志两侧条目全保留（10-23 的 vnet 12:20 条目按时间插到 task27:4 之后、task35 之前）；任务书以 vnet 侧为准（主线副本是旧快照）；`search_play.py` 以主线结构为准——CUDA 叶子块与 CUDA 服务启动随 `encoder_version != 1` 保护一起删、`--margin` 不恢复，只把 `encode_frame(f, net.encoder_version)` 与 `--leaf-probe/--leaf-probe-limit` 落回主线（语法校验通过，全文无 cuda/rollout_backend 残留）。默认值：`curriculum_queue.py DEFAULT_VALUE` 与 `unified_panel.py --value` → `value_net/v4`；DESIGN 新增 §14.17（依据链 value-widen.md 第 5 步）；`check_panel_type_counts.py` 等历史夹具与任务 27 溯源脚本不动。v4 已复制到主检出 `/home/newbiexvwu/PvZ-search/artifacts/value_net/v4/model.pt`，SHA256 `14c372e19dcd73fd7ee0493600907efefa0f7edafa7b884e1fdecd875008cd8d` 与 HF 回验值一致。测试：`pytest python -q`（资源目录 `PVZ_RESOURCE_DIR=/home/newbiexvwu/.cache/pvz-research-resources`）= **312 passed / 1 failed / 111 subtests passed**；唯一失败 `test_identity_training.py::test_cpu_training_checkpoint_is_loadable`（`train_rollout_policy.py:549` 空候选哨兵索引越界）在合并前主线 origin/pvz-env (deac6118) 临时 worktree 上同样失败——非本次合并引入。冒烟：合并重建引擎后用主检出 v4 跑 `day_l7_train_d7_m150` 1 局（1 个游戏进程），seed 5512288 胜、30 波、exit 0——C++ 推演路径 + 编码 v2 叶子可用。v5 立项节已写入 value-widen.md（只写不跑）。

## 第 6 步：采用 v4 并合进主线（主控 2026-10-23 15:40 追加）

**主控决定：整臂换 v4**（不分区）。依据是 value-widen.md 第 5 步：白天 m150/m250 不掉，夜间 96 对净 +20，泳池 L21 +9，L24 净 0。泳池全败区叶值压平是已知限制，修法另开 v5（见第 4 条），不挡这次换。

1. **合主线**：在本工作树 `git merge --no-ff origin/pvz-env`。冲突四处：`MEMORY/2026-10-22.md`、`MEMORY/2026-10-23.md`、`task30-value-widen.md`、`scripts/search_play.py`。规则：
   - 日志两侧条目全保留，按时间排；
   - `search_play.py` 以主线结构为准：主线已删 CUDA（`_leaf_values` 的 CUDA 路径、`--rollout-backend`）和判死开关（`--margin` 已成常量、`--candidate-generator` 等），这些位置不要加回来。只把 vnet 的功能落在主线结构上：`encode_frame(f, net.encoder_version)` 和 `--leaf-probe` 诊断。`--leaf-probe` 保留，v5 要用它验收；它只写诊断行，不改决策。
   - CUDA 的 `encoder_version != 1` 保护随 CUDA 一起删。
2. **v4 成为唯一默认**：
   - `scripts/curriculum_queue.py` 的 `DEFAULT_VALUE` 改指 `artifacts/value_net/v4/model.pt`。
   - 把 `artifacts/task30/v4/model.pt` 复制到主检出 `/home/newbiexvwu/PvZ-search/artifacts/value_net/v4/model.pt`，不进 git；复制后核对 SHA256 与 HF 回验值一致。
   - `git grep -n "value_net/v3"` 找出其余默认值（`search_play.py`、面板/评估脚本、DESIGN 的现行配置描述），凡是"现行默认"的都改 v4。历史报告里记录当时用 v3 的句子不改。
   - DESIGN.md 现行方案里价值网的描述改成 v4，并写一行依据（链 value-widen.md 第 5 步）。
3. **测试**：用本工作树自己的 `build/` 跑 `python3 -m pytest python -q -p no:cacheprovider`，与合并前主线比不得多出失败。另做一次冒烟：用 v4 跑 `search_play.py` 一局 day_l7（≤2 个游戏进程），确认 C++ 推演路径能用 v4（编码 v2）。
4. **v5 立项**：只写，不跑。在 value-widen.md 末尾新节"v5：泳池决策级标签"写一页计划，内容包括：
   - 标签来源：泳池可赢关的老师同局候选结果，用分叉采集通道；
   - 需要多少点；
   - 验收：泳池 L24/L26 用 `--leaf-probe` 量离散度，要回到 v3 量级，同时 L21 配对不掉。
   - 主控之后另派。
5. 提交，`git push origin vnet`，在本任务书"结果"写 `[step6]` 一行：合并提交号、冲突怎么解、测试数字、v4 SHA256。**不推 pvz-env**，主控审后快进并在 qc 两轮之间切换。

规则照旧：≤2 个游戏进程；不碰 tmux `qc`；只用绝对路径；禁止 `git add -f`；`.pt` 不进 git。
