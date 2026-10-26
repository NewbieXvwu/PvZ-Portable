# deploy14 课程审计（task100）

任务书：[task100-curriculum-audit.md](task100-curriculum-audit.md)。只读生产，未操作 tmux qc、修改 cq/state、运行游戏或训练。

审计快照：2026-10-26，state.updated **2026-10-26T11:51:46**，round=18、正在采 round19；收集学生仍为 r11。时间均为本机日志时间（Asia/Shanghai）。下面的“08:50 后组数”按 `curriculum.log` 完成时间 ≥08:50:00 统计，不把 dry-run 的 added 行当实际采集。

## 逐格对账

索引从 0 开始；“当前档”是 state 已调整、下组将采的档位；窗口列列出已经有数据的档位，当前档没有窗口时即尚未测。13 个示范格均为 2.0 倍原始难度，档位数字是示范决策号，不能读成僵尸倍率。表内示范格在 state.cells 中，均未退休；L25 单列为 retired。

| 格 | cells / 退休 | 08:50 后完成组 | 当前档（索引/档数） | 已测窗口胜/局 | teacher_rates 命中 | 当前种子块：已占/总数（余） |
|---|---|---:|---|---|---|---|
| `fog_user_v2_l31_snow` | 在 / 否 | 1 | 1.25（1/7） | 1.0: 48/48 | base 31/32 | 0/2048（2048） |
| `fog79b_l36_best` | 在 / 否 | 1 | 1.25（1/7） | 1.0: 22/24 | base 32/32 | 0/2048（2048） |
| `fog79b_l37_best` | 在 / 否 | 1 | 1（0/7） | 1.0: 5/24 | base 31/32 | 24/8192（8168） |
| `fog79b_l38_best` | 在 / 否 | 1 | 1.25（1/7） | 1.0: 22/24 | base 32/32 | 0/2048（2048） |
| `pool_l25_orig_train` | 不在 / 是：误判起点格 | 0 | 0.5（0/8） | — | base 13/16 | 0/2048（2048） |
| `demo_fog_31_snow_m200_hv2` | 在 / 否 | 0 | 706（1/10） | 777: 24/24 | base 8/8 | 24/256（232） |
| `demo_fog_32_snow_m200_hv2` | 在 / 否 | 1 | 1119（1/9） | 1231: 24/24 | base 8/8 | 24/256（232） |
| `demo_fog_33_cactus_snow_m200_hv2` | 在 / 否 | 1 | 762（1/6） | 839: 24/24 | base 8/8 | 24/256（232） |
| `demo_fog_33_cattail_snow_m200_hv2` | 在 / 否 | 1 | 788（1/8） | 867: 24/24 | base 8/8 | 24/256（232） |
| `demo_fog_34_cactus_snow_m200_hv2` | 在 / 否 | 1 | 1024（1/6） | 1126: 24/24 | base 8/8 | 24/256（232） |
| `demo_fog_39_cactus_snow_m200_hv2` | 在 / 否 | 1 | 1375（1/5） | 1513: 24/24 | base 8/8 | 24/256（232） |
| `demo_roof_42_melon_m200` | 在 / 否 | 1 | 1293（1/4） | 1422: 22/24 | base 8/8 | 24/256（232） |
| `demo_roof_43_melon_m200` | 在 / 否 | 1 | 965（1/6） | 1061: 24/24 | base 8/8 | 24/256（232） |
| `demo_roof_44_melon_m200` | 在 / 否 | 1 | 1229（1/7） | 1352: 24/24 | base 8/8 | 24/256（232） |
| `demo_roof_42_gloommelon_m200` | 在 / 否 | 1 | 1160（1/6） | 1276: 24/24 | base 8/8 | 24/256（232） |
| `demo_roof_48_doom_m200` | 在 / 否 | 1 | 1070（1/9） | 1100: 24/24 | base 8/8 | 24/256（232） |
| `demo_roof_49_doom_m200` | 在 / 否 | 1 | 1482（0/6） | 1482: 9/24 | base 7/8 | 24/256（232） |
| `demo_roof_46_doom_m200` | 在 / 否 | 1 | 835（1/7） | 918: 24/24 | base 8/8 | 24/256（232） |

L31 `_hv2` 首组实际在 08:39:35–08:43:19 完成 24/24；L32 首组在 08:43:19–08:50:29 完成 24/24，跨 deploy14 切换边界。二者均已采，L31 的本时间窗为零不属于漏采。其余示范首组为 08:55:44–09:30:15；fog79b 三组分别于 09:34:32、09:43:21、09:48:20 完成；fog_user_v2 于 09:56:40 完成 48/48。所有这些组 rc=0、beta=0，因此窗口是学生胜率，不是混合老师胜率。

门禁区域的任务与训练格须分别对账：

| v3 区域 / 任务 | state.cells | 退休/原因 | 08:50 后训练组 | 当前档 / 窗口 | teacher_rates |
|---|---|---|---:|---|---|
| fog / `fog_user_l31_snow` | 不在 | 在 retired：08:37:27 superseded by fog_user_v2 | 0 | 1.0（0/7）；142/216=65.74% | base 25/32 |
| roof / `roof_l41_train_r7` | 在 | 否 | 1：10:42:55，1/48 | 1.0（0/7）；最近15/256=5.86%，累计85/408 | level:41 13/16，base无条目 |

门禁任务不需要在训练 cells 内才可评估。v3 配置确实含 fog/roof，各 192 对、双方各跑一次；共 1920 对/3840 局。**但 fog 配置仍是 `fog_observation=human`，现课程是 `human_v2`**；旧雾格已被替换，不能把这次旧观测门禁当成新观测覆盖。roof 的实际门禁任务是 L41 r7，不涵盖七个 L42–49 示范起点。

## 异常与建议（交主控分派）

1. **L25 从未采，重启反复 added→retired。** state.retired_cells 中 last_round=0、windows空、mult_index=0、ladder=[0.5,1,1.25,1.5,1.75,2,2.5,3]，teacher base 13/16 已命中，不是缺底数。日志 08:44:02、08:50:46（同秒两套）、11:43:43 均有同样配对。旧 `reconcile_start_cells` 以 ladder 判断示范，误退有倍率阶梯的普通格。43458aa8 已将生产磁盘脚本 `scripts/curriculum_queue.py:464` 改以 archive 判断；11:43 启动进程仍打印误退，说明该进程未加载此后修复。建议下次受控重启核验 restored/不再退休/首组 rc=0，不能仅检查文件已改。修复不由本任务部署。
2. **L31/L32 已采的 48 局没有进入 r18 训练。** 标签 `c18demofog31snowm200hv2s777_fast_search`、`c18demofog32snowm200hv2s1231_fast_search` 各有24个原始 `frames.npz`（`/home/newbiexvwu/PvZ-search/artifacts/student/frames_v2/seed_20261031_f44000000/...` 等）；r18 `recipe.json.labels` 两项均缺，pipeline/frames 对应档案也都是0/24。state 的 last_round/offset/windows 保存了两组，但 state.log 的 r18.groups 仅23个重启后组。根因：`scripts/curriculum_queue.py:1259–1275` 在整轮结束时只写本次 plan；恢复时 allocate 因 last_round==18 跳过完成组，而 `pack_labels:663–669` 只汇总 state.log.groups，提前完成组失去训练标签。建议 task98 持久化“完成组→帧标签”账本，恢复时合并，不仅恢复偏移；补组边界停止/崩溃的训练配方回归验证。此问题会让“窗口显示采过”与“真正进训练”脱节。
3. **v3 fog 观测已过时。** `python/pvz_promotion.py:56–62` 的 PANEL_TASKS_V3 固定旧 fog_user；cq `artifacts/task73c/panel_v3_source.json` 同为human，而新课程 source 是 `artifacts/task70/fog_user_v2_panel.json` 的 human_v2。建议任务负责人明确门禁目标并登记新版本/冻结语义；不原地改历史证书。本次只报告差异。
4. **档位不动：L37 与示范L49属正常下限，尚无“卡档”证据。** L37=5/24、L49=9/24，均<0.5但索引已经为0；`adjust:526–541` 只在index>0降档。其余15个活跃新格已升一档且新档尚无窗口；不能因当前档未测而再次诊断卡档。全55个active cells 的索引均在合法范围内。L25 不动由误退休解释，不是升降阈值问题。建议 L37、L49 优先看老师可教性/采集数据，不增手写打法。
5. **本快照无 active 格种子不足3个48局组。** 对全部55个active cells按当前实际任务/倍率、state.offsets 做差，均≥144；13个起点格均256−24=232，可容4个48局组（或9个24局保底组），建议 task99 在到达144之前续领。旧 `demo_fog_32_snow_m200` 已退休且 offset216/256，余40不足一个48局组；新 `_hv2` 是另一个任务族、offset24，不能把旧偏移直接误认新族余量。任务书称旧L32曾耗尽，与plan_round offset+games>len 判据一致；本次不假称它依然active。`plan_round:1161–1168` 仍会退休耗尽起点，不会续领；`reconcile_start_cells:469–473` 重新新增退休格时也不清除退休副本，需task99验证续领恢复/不反复重退。
6. **18个新增格 teacher_rates 全部精确base命中，无缺底数。** roof L41只有level回退；v3旧fog也有base。全55个active的更广检查另发现26个格既无base也无level，`allocate:582` 会按未知默认1.0排名，这属于缺证据而非已知老师100%胜：

`day_l8_train_d7`, `day_l9_train_d7`, `day_l1_train_d7`, `day_l3_train_d7`, `day_l4_train_d7`, `day_l6_train_d7`, `day_l10_orig_train`, `night_l11_train_n10`, `night_l12_train_n10`, `night_l13_train_n10`, `night_l16_train_n10`, `night_l18_train_n10`, `night_l19_train_n10`, `pool_l26_train_p9k`, `pool_l27_train_p9k`, `pool_l28_train_p9k`, `pool_l29_train_p9k`, `day_l8_train_d7q`, `day_l9_train_d7q`, `day_l10_train_d7q`, `night_l11_train_n10q`, `night_l12_train_n10q`, `night_l13_train_n10q`, `night_l16_train_n10q`, `night_l18_train_n10q`, `night_l19_train_n10q`.

15:40 复核：按 `teacher.get(base, teacher.get(f'level:{level}', 1.0))` 的同一查找规则（`curriculum_queue.py:581`）在 15:05 快照上重算，缺底数仍是同样的 26 格，18 个新格与 13 个示范格全部 base 命中——不是靠“读过配置文件”下的结论，而是同一条查找路径算出来的。

后果可以量化：16:15 那份 16 组计划里有 **8 组（384 局，正好一半预算）** 落在既无 base 也无 level 底数的格上——`pool_l26/27/28/29`、`night_l19_train_n10` 与 `night_l19_train_n10q`、`day_l1_train_d7`、`day_l3_train_d7`。这些格按默认 1.0 参与排名，于是“学生 15.6%–57%”被读成“可教差距大”，自然优先排；但没有老师底数就无法区分“学生弱”和“这关对谁都难”。补底数的收益不是报表好看，而是把这半轮预算花在有可教差距证据的格上。

建议主控按采集优先级补老师底数。现有底数是base级而非档位级：fog v2 31/32及fog79b数据源为1.0，升到1.25后仍沿用；示范底数也是特定起点的小样本，不能声称每个新档都已测老师。

同秒重复 added 日志并不等于重复执行采样：08:50:46 两套初始化与 deploy14 dry-run/启动吻合；只有 L25 的误退在11:43仍复现，其余新增格的窗口/offset没有被清空。后续应按实际组标签与偏移核验，不能数added行当组数。

## r18 热补与 v3 首次运行（运行前观测）

- 11:25:23、11:30:35 两次训练启动均因 plane_of broadcast 失败；11:33:37 达到2次失败停qc。
- 11:43:42 主控重启后，11:46:17重走pack，11:48:57 pack done并启动训练。pack.log 说明 built=0、reused=475（不是坏段重建），总18,466,229决策；train.log已完整读入并打印packet/spread权重，跨过旧崩溃点。首次核验训练PID861864存活、未产model.pt，不能据此宣称训练完成。
- 对r18实际pack `1216b129a66d4b74.npz` 的475段逐一只读检查 NPY headers：唯一非空plane_of行数不符为 `ce3ce4006691a0e1.npz` board，offsets所示84行、plane_of72。现 `train_rollout_policy.py:609–615` 在长度不符时生成−1标记，降级为不使用planes；这是读兼容热补，task99根因修复仍须覆盖这段短尾示范。
- job18 已冻结 **[39344,41264)**，长1920，账本allocated中恰有一次；前块[37808,39344)紧邻，未重叠。仍从第一池[31024,65536)领，未跳第二池[100000000,101000000)。v3首轮不应沿用1536长度。按v3十任务顺序分片，fog应为 **[40880,41072)**、roof **[41072,41264)**；须待首次门禁实际manifest与结果再核对，不把已领号当已运行。
- 训练完成、首次v3门禁实际结果和证书仍待观察，task100未完成。

## r18 热补与 v3 首次门禁：完成核验（15:10 快照）

**训练确实跑完了。** pack 11:46:17→11:48:57（built=0、reused=475、18,466,229 决策），控制器计时训练11:48:57→12:43:02（3244.888s，含进程启动/退出；训练脚本自身打印3226s），峰值匿名内存3.75 GB，12个epoch，val_loss 3.8065→3.6007、cell_acc 0.4637→0.4918；产出 `student/model.pt`（730,798 字节，sha256 `4d3801dce5fbe36207027dc1b215447452bdad522502956678c7db6e56761ba0`）。`train.log` 开头两段 traceback 是 11:25/11:30 的旧失败，成功段在其后，日志是追加而非覆盖，不能因为首屏是 traceback 就判成又崩了。

**v3 首轮实际执行与预测一致。** manifest `panel_version=v3-task73c`、10 任务 × 192 种子 = 1920 对 / 3840 局；`seed_block=[39344,41264)`，与 ledger 中该区间的唯一一次分配吻合；fog 分片 `[40880,41072)`、roof 分片 `[41072,41264)` 与此前按十任务顺序推算的完全一致。manifest 中 fog 任务显式带 `fog_observation=human`，证实“门禁跑的是旧雾观测”不是配置文件层面的猜测，而是这一轮实际执行的参数。

**结果（decision.json v3，approved=false）：**

| 区域 | 对局 | candidate / incumbent | delta | CI95 | McNemar p |
|---|---:|---|---:|---|---:|
| overall | 1920 | 1462 / 1547 | −4.43pp | [−6.34, −2.52] | 3.5e−6 |
| day | 384 | 349 / 368 | −4.95pp | [−8.60, −1.43] | 0.0040 |
| heldout | 384 | 263 / 325 | −16.15pp | [−22.07, −10.08] | 1.8e−7 |
| night | 384 | 371 / 373 | −0.52pp | [−3.19, 2.12] | 0.42 |
| pool | 384 | 277 / 291 | −3.65pp | [−9.09, 1.82] | 0.11 |
| fog | 192 | 192 / 186 | +3.13pp | [0.54, 6.65] | 1.0 |
| roof | 192 | 10 / 4 | +3.13pp | [−0.87, 7.47] | 0.97 |

拒绝原因是 `overall_ci_lower_not_above_minus_3pp` + `significant_zone_regression:day` + `significant_zone_regression:heldout`；heldout 的 −16pp 是主因，不是噪声。

**两区存在地板/天花板与覆盖限制。** roof的192对里178对双方皆负，只有14对分出胜负，差值CI跨0；fog在旧观测下186/192对双方皆胜，另6对候选独胜，其差值CI下界为正，不能说完全没有区分能力。它们测到的是既定面板上r18与r11的差异，没有对训练新增格做消融归因。roof仍是L41 r7，不覆盖七个L42–49示范起点；fog仍是human观测，不能替代human_v2。

**r19 也成功完成训练与门禁。** 14:04:55 起训练（2502s、18,598,230 决策、无 plane_of 崩溃）→ 14:46 评估 → 15:01 完成，approved=false：overall −2.50pp（CI [−4.33, −0.67]）、heldout −13.80pp（p=7.9e−7）、day −1.30pp（p=0.26）、pool +3.13pp。这证明连续两轮能跑完，不能证明读兼容降级没有触发：后续复核r19的497段pack仍引用 `ce3ce4006691a0e1.npz`，该段board_offsets长85、plane_of长72，仍需热补处理。r19 门禁用了新块 `[41264,43184)`（fog `[42800,42992)`、roof `[42992,43184)`），没有复用 r18 种子。

**heldout 连续两轮显著退化，需要单独排查。** r18 −16.15pp、r19 −13.80pp，两轮都触发veto，且门禁按配对数据正常执行，不能当成“门禁进程失败”。本审计没有消融实验，无法排除训练组成、丢标签或plane读兼容处理对模型的影响。建议主控单独分派策略与数据归因，task98继续处理已经证实的账本/异常隔离问题。

**r18 的丢标签没有在 r19 复现，但 r18 那 48 局仍永久缺席。** r19 配方含 `c19demofog31snowm200hv2s706`、`c19demofog32snowm200hv2s1119`，source_rows 分别读入 48 档案 / 1706 训练行与 48 档案 / 2276 训练行。这证明“中途重启丢组”是条件触发，不是每轮必丢。但 r18 的 `c18` 两标签仍不在任何一轮配方里，除非把旧帧重新打包进配方，否则那 48 局不会进入训练。

**L25 修复尚未生效，时间线要说清。** 修复提交 `43458aa8` 的时间是 11:47:48，现 qc 进程启动于 11:43:42（`ps lstart`） —— 进程先于修复 4 分钟，所以它持的是旧代码；15:05 快照里 L25 仍在 `retired_cells`（mult_index=0、last_round=0、windows 空），08:44/08:50/11:43 三次 added→retired 配对之后再无变化。验收标准仍是下次受控重启后：L25 回到 cells、不再被退休、首组 rc=0。

## 15:05 状态刷新（round 19 完成、round 20 在采）

上面的逐格表是 11:51 快照，保留作初始证据。下表是同一批 18 格的 15:05 读数（state `updated=2026-10-26T15:05:10`、round=19；“08:50 后组”按日志完成时间统计，已把 `_m125` 这类变体后缀归回同一格）。

| 格 | 状态 | 当前档（索引/档数） | 08:50 后组·局/胜 | 窗口（档:胜/局） | 当前档 offset/容量 |
|---|---|---|---:|---|---|
| `fog_user_v2_l31_snow` | cells | 1.25（1/7） | 1·48/48 | 1.0: 48/48 | 0/2048 |
| `fog79b_l36_best` | cells | 1.25（1/7） | 2·72/53 | 1.0: 22/24；1.25: 31/48 | 48/2048 |
| `fog79b_l37_best` | cells | 1.0（0/7） | 1·24/5 | 1.0: 5/24 | 24/8192 |
| `fog79b_l38_best` | cells | 1.25（1/7） | 2·72/55 | 1.0: 22/24；1.25: 33/48 | 48/2048 |
| `pool_l25_orig_train` | retired | 0.5（0/8） | 0 | — | — |
| `demo_fog_31_snow_m200_hv2` | cells | 635（2/10） | 1·48/48 | 777: 24/24；706: 48/48 | 120/256（余 136） |
| `demo_fog_32_snow_m200_hv2` | cells | 1007（2/9） | 2·72/72 | 1231: 24/24；1119: 48/48 | 72/256 |
| `demo_fog_33_cactus_snow_m200_hv2` | cells | 686（2/6） | 2·72/72 | 839: 24/24；762: 48/48 | 72/256 |
| `demo_fog_33_cattail_snow_m200_hv2` | cells | 709（2/8） | 2·72/72 | 867: 24/24；788: 48/48 | 72/256 |
| `demo_fog_34_cactus_snow_m200_hv2` | cells | 922（2/6） | 2·72/72 | 1126: 24/24；1024: 48/48 | 72/256 |
| `demo_fog_39_cactus_snow_m200_hv2` | cells | 1238（2/5） | 2·72/72 | 1513: 24/24；1375: 48/48 | 72/256 |
| `demo_roof_42_melon_m200` | cells | 1270（2/4） | 2·72/65 | 1422: 22/24；1293: 43/48 | 72/256 |
| `demo_roof_43_melon_m200` | cells | 869（2/6） | 2·72/72 | 1061: 24/24；965: 48/48 | 72/256 |
| `demo_roof_44_melon_m200` | cells | 1229（1/7） | 2·72/72 | 1352: 24/24；1229: 48/48 | 72/256 |
| `demo_roof_42_gloommelon_m200` | cells | 1044（2/6） | 2·72/71 | 1276: 24/24；1160: 47/48 | 72/256 |
| `demo_roof_48_doom_m200` | cells | 1070（1/9） | 2·72/31 | 1100: 24/24；1070: 7/48 | 72/256 |
| `demo_roof_49_doom_m200` | cells | 1482（0/6） | 1·24/9 | 1482: 9/24 | 24/256 |
| `demo_roof_46_doom_m200` | cells | 835（1/7） | 2·72/72 | 918: 24/24；835: 48/48 | 72/256 |

刷新后的新发现，按重要性排：

1. **示范起点格的“升一档”不是小步，L48 出现了断崖。** `demo_roof_48_doom_m200` 的阶梯是 `[1100,1070,1041,946,852,757,663,568,474]`（数字是示范决策号，越大越靠后、越容易），index 0 的 1100 是 24/24，index 1 的 1070 只有 **7/48（14.6%）**。相邻两档差 30 个决策号，胜率掉 85 个百分点。后果是可预测的：`adjust` 会把它降回 index 0，下一轮又 100% 升到 index 1，在 0↔1 之间来回震荡，每轮烧掉一组。这不是“策略卡住”，是阶梯本身在 L48 顶部过粗（很可能 1070 正好落在人类放 doom 之前）。建议主控按“相邻档胜率落差”检查其余 12 个示范格的阶梯，而不是只看 index 有没有动。 这条预测在 38 分钟内被日志证实：15:37:47 的计划打印 `demo_roof_48_doom_m200: 1070 -> 1100 (win 0.15)`，16:15:54 又打印 `1100 -> 1070 (win 1.00)`——两次升降之间**没有跑过任何一组**，纯粹是记账来回，说明该格的 `adjust` 已经进入稳定振荡。
2. **`demo_fog_31_snow_m200_hv2` 是第一个真正要撞上种子枯竭的起点格：120/256，余 136，不足 3 个 48 局组。** 它本轮 706 档 48/48、下一档 635 还没采；再跑两组 48 局就只剩 40 —— 与已退休的旧 `demo_fog_32_snow_m200`（216/256、余 40 被退休）完全同型。task99 的续领应先给它，而不是给余 184 的其它 12 个。
3. **L31（`fog_user_v2`）升到 1.25 后还没采过**（last_round=18，窗口只有 1.0: 48/48）。这不是丢格，是 round 19/20 的调度把它排在后面；但它是 deploy14 的唯一新雾格，建议下轮优先排，否则 1.25 档长期没有数据。**16:39:15 已补上**：`c20foguserv2l31snowm125` 48 局 40 胜（83.33%），未超过严格的85%升档阈值，应留在1.25档；该组在16:15重启后的计划里，需等整轮写账再核验标签，不能提前声称已进log。
4. **fog79b 三格分叉：** L36、L38 升到 1.25 后拿到 31/48（0.65）与 33/48（0.69），位于0.5–0.85区间，应保持当前档；L37 仍停在 index 0、窗口 5/24，与 11:51 一致，是“地板 + 无进展”，与第 4 条异常（老师可教性）同一件事，不是卡档 bug。
5. **`day_l9_train_d7` 的 offset=3584 不是越界。** 它的 1.0 档来自 `artifacts/stage4/day_panel.json`，该任务有 8192 个种子，余 4608。这里特意写出来是因为按“面板变体 2048”去套会算出 −1536 的假越界；容量必须按当前档实际所属清单取（base 任务 8192 / 变体 2048 / 示范 256）。
6. **11 个 active 格已经 3 轮没被排过**（last_round ≤ 17）：`day_l7_train_d7`、`day_l9_train_d7`、`day_l3_train_d7`、`day_l10_orig_train`、`night_l13_train_n10`、`night_l17_train_n10`、`night_l18_train_n10`、`day_l9_train_d7q`、`night_l11_train_n10q`、`night_l13_train_n10q`、`night_l16_train_n10q`。每轮 768 局摊 55 格，必然有格轮空，这不必然是 bug；但如果它们里有本该优先的格，说明 `allocate` 的排序把它们压得太死（其中 9 个正是上文“无 base 也无 level 底数”的格，默认 1.0 参与排名）。
7. **55 个 active 格的 `mult_index` 刷新后仍全部落在合法区间**（示范格对 `len(ladder)`、普通格对 7 档）；本快照没有新的越界。

领号余量也一并算清（15:05快照）：第一池 `[31024,65536)` 已用到43184，余22352，按每轮1920还能完整分配11轮；第二池 `[100000000,101000000)` 尚未动，可分配520轮。`promotion_gate.allocate_seed_block:122–133` 会按登记顺序自动转到第二池；无需人工扩池或提前切换。第一池不足一个完整块的尾部不会跨池拼接，所有登记池都放不下完整块时才报耗尽，且不会回绕复用。冻结policy仍保留旧seed_pool字段以维持历史证书哈希，实际来源以ledger和seed_block为准。

## 组标签账本：丢失面比 r18 那一例更大（代码级复核）

r18 丢两组曾被解释成“提前完成组在重启后不被计入”。读 `curriculum_queue.py` 后，实际机制更宽，值得单列：

- `run_round` 每组完成后只落 `cells`（windows / last_round）与 `offsets`（`state['offsets'][task_id] = offset + games` 后立刻 `save_state`，:1189–1190），**组的 log 条目要等整轮 plan 全部跑完才 append**（:1265–1275，`groups=[... for item in plan]`，只含本次 plan）。
- `pack_labels` 只从 `state['log']` 的 groups 取标签（:663–666），没有任何“扫帧目录”的兜底。
- 结论：**任何轮内重启都会丢掉该轮已完成组的标签**，不只是“提前完成”的那几个。偏移已推进所以不会重采，帧也在磁盘上，丢的只是“这些帧该进训练”的记录。r18 只是碰巧丢得少（重启发生在两组之后）。

现场暴露（15:31 快照）：round 20 从 15:05 起已 rc=0 完成 **12 组 / 480 局**，而 `state['log']` 里**没有 round 20 条目**（`state.updated=15:26:54` 但 `round` 字段仍是 19）。此刻重启，这 12 组的标签会全部缺席下一份配方——与 r18 的 48 局同型，只是规模 10 倍。

加剧因素：生产 cq 的 `run_round` 在任一组 `rc!=0` 时直接 `raise SystemExit`（:1261）；task98 的按格/按组隔离只在 qcwatch 工作树里，未部署。所以“一次组失败 → 整条队列退出 → 重启 → 丢当轮已完成组标签”是一条真实可达的路径，不是理论风险。

补一条正面核验：deploy14 的 24 个组（round 18–20，18 个新格）在 `artifacts/student/frames_v2/` 下都能按 label 找到**恰好 games 个帧目录且无一为空**（0 处不符）。也就是说丢的从来不是采集数据，只是账本——这让修复有明确落点：

1. 组完成时就把该组 dict 追加进本轮 log 条目（没有条目就新建），而不是等整轮结束一次性写；
2. 或在 `pack_labels` 里回填“本轮 plan 中已完成、但 log 里没有”的组（组完成时 cells 已带 last_round，可与 plan 对账）；
3. 回归测试按组边界崩溃重启：断言已完成组的标签仍出现在下一份配方里，而不只是断言偏移没重采。

还有一个容易忽略的副作用：标签由 `label_for(round, base, multiplier)` 决定（:623–629），同一轮重启后若重排同一格会得到**同名标签**，两次尝试的帧会合并到同一标签下（偏移已推进，种子不同，不会覆盖）。所以“重排”能让被丢的标签被动复原，但也会把中断组的残帧一起带进来——它不能替代显式记账，反而说明标签命名需要带 attempt/offset 才不会把两次尝试混成一组。

## L25 修复已上线：受控重启 + 首组核验（15:47）

15:37:38 出现 `round 20: stop flag seen ...; stopping at group boundary`，15:37:46 起新进程（PID 906610），15:37:47 日志出现 `cell restored: pool_l25_orig_train`，15:38:15 起首组 `c20p25origm050`。这就是异常 1 要求的受控重启，三条验收标准逐条对上了：

| 验收项 | 结果 |
|---|---|
| 回到 cells | ✅ `pool_l25_orig_train` 在 `cells`，不在 `retired_cells`；`mult_index=0`、`ladder` 8 档 |
| 不再被退休 | ✅ 重启后日志只有 `cell restored`，没有新的 `start cell retired` |
| 首组 rc=0 | ✅ 15:40:23 `pool_l25_orig_train_m050 mult=0.5 offset=0 games=24 wins=1 rate=0.04 beta=0 rc=0 128s` |

首组数据本身是个强信号：**L25 在 0.5 倍僵尸数下学生只赢 1/24（4.2%）**，而老师底数是 13/16（81%）。这不是“格在却不采”那类基础设施问题，是这条课程线上最大的可教差距之一；比 L37 的 5/24（1.0 档）更低。它落在阶梯最低档，按 `adjust` 不会再降，后续会一直停在这里产出低胜率样本——建议主控在补老师底数与 task99 的续领里把它当“地板格”处理，并确认 0.5 档是否已经低于可教学区间（低于它就没有更低的档可退）。

L25 是传送带关（`conveyor_start=original`、`deck=[]`），所以先排除“传送带本身不会玩”这个解释：课程里另两条传送带格都不差——`day_l10_orig_train`（L10 白天）1.0 档 304/306、3.0 档仍 104/120；`night_l20_train_n10_orig`（L20 夜）1.0 档 206/296（70%）。4% 更像是**泳池×传送带**这个组合缺训练覆盖：全 55 格里只有 L25 这一条泳池传送带格，而且它此前从未被采过（正是本次审计的起因），所以这不是“学生退步”，是“这个组合第一次被测量”。建议主控把它当分布外探针看，别按普通地板格去加档或写规则。

再补一条：**这次重启没有重采**。round 18–20 里 48 个 task 的组局数总和都不超过各自 offset 跨度（0 处重复），说明 `state['offsets']` 先占后存的写法确实挡住了重复采样——丢的只是标签，不是算力。

另外，这次重启正好落在组边界上，且发生在 13 组完成之后（12 组 + 15:37:38 刚结束的 `demo_fog_33_cactus`）——它是“组标签账本丢失面”一节的现场样本，而且**已经用计划表证实**（不必等本轮收尾）：

- 15:37:47 那份计划 16 组/744 局，第一行就是 `pool_l25_orig_train 0.5 24 局 c20p25origm050`；
- 16:15:54 那份计划 16 组/768 局，`pool_l25_orig_train` **不在其中**，15:37 计划里的 `demo_fog_33_cattail`、`demo_fog_34`、`demo_fog_39` 也都不在。

计划由 `print_plan` 打到 `qc.out`，是重启时重新 `plan_round` 生成的；没被重排的已完成组既不在新计划里，也不会进 `state['log']`，所以 `c20p25origm050` 这 24 局（L25 有史以来第一组）的标签不会再出现在任何配方里。

需要说清边界：这 24 局丢的是**样本**，不是格——L25 胜率 4% 会在下一轮继续被优先排到（新标签 `c21…`），所以不会重现“格在却不采”。代价是 24 局算力与一条“首次测量”的记录没进训练。

还有一个连带代价：round20此前已完成528（第一次重启前）+168（两次重启之间）=696局；16:15后又排了768局。若新计划全部完成，合计会达到1464局，超过单次768局预算。这里后一个768是计划量，不能在轮未结束时写成“实际已采”。偏移不同的新增局并非重复种子，而是轮内预算与标签账本缺少累计。

## 11:59 补充核验

在隔离内存副本执行纯函数：deploy14提交0b8dcc26的 `reconcile_start_cells` 对原生L25产生误退，磁盘热补副本保留该格；当前 `pack_labels` 对实际state快照稳定漏出L31/L32两标签；复制的 `freeze_panel` 对job18块生成1920个互不重复种子，fog/roof分片与上文一致。未调用领号、保存状态或采集入口。

退休格余量补齐：`demo_fog_31_snow_m200`、`demo_fog_32_snow_m200`、`demo_fog_33_cactus_snow_m200`、`demo_fog_33_cattail_snow_m200` 都是216/256、余40，均不足3轮。它们已由 `_hv2` 替换；task99不应将这些历史族一概重新激活。前四个新旧雾族还共享各自同一未来种子区间（44000000、44300000、44100000、44200000起，各256），新族从偏移0重采；因此“新族余232”是任务语义下的偏移余量，并非232个从未在旧族用过的全局新种子。续领账本需要明确族/存档/观测语义，避免误算新鲜度。

11:59只读 `/proc/861864` 确认r18训练仍为同一活进程（父PID860060），train.log已记epoch1完成，seconds=274.316；尚无 `student/model.pt`、`complete.json` 或首次v3评估结果。继续观察该进程，不因等待超时启动另一份任务。

12:00后续轮询：同一训练PID仍存活，epoch2完成（213.524秒，val_loss=3.72647），job18仍queued且seed_block未变。当前“等待”有活进程和新增epoch日志支持；尚不能验收最终训练或门禁。

## 训练输入逐格复核

以r18学生目录的 `source_rows.json` 核验训练实际读取，而不只检查recipe声明。L25未采，无该格标签；17个已采新格中，L31/L32缺失，其余15个均被读取且training_rows>0：

| 格 | 读入档案数 | 保留决策行 | 训练行 | 验证行 |
|---|---:|---:|---:|---:|
| `demo_fog_31_snow_m200_hv2` | 缺失 | — | — | — |
| `demo_fog_32_snow_m200_hv2` | 缺失 | — | — | — |
| `demo_fog_33_cactus_snow_m200_hv2` | 24 | 575 | 509 | 66 |
| `demo_fog_33_cattail_snow_m200_hv2` | 24 | 310 | 310 | 0 |
| `demo_fog_34_cactus_snow_m200_hv2` | 24 | 632 | 550 | 82 |
| `demo_fog_39_cactus_snow_m200_hv2` | 24 | 738 | 648 | 90 |
| `demo_roof_42_gloommelon_m200` | 24 | 316 | 293 | 23 |
| `demo_roof_42_melon_m200` | 24 | 757 | 687 | 70 |
| `demo_roof_43_melon_m200` | 24 | 579 | 533 | 46 |
| `demo_roof_44_melon_m200` | 24 | 619 | 505 | 114 |
| `demo_roof_46_doom_m200` | 24 | 746 | 650 | 96 |
| `demo_roof_48_doom_m200` | 24 | 84 | 71 | 13 |
| `demo_roof_49_doom_m200` | 24 | 1455 | 1212 | 243 |
| `fog79b_l36_best` | 24 | 5684 | 4516 | 1168 |
| `fog79b_l37_best` | 24 | 8034 | 7030 | 1004 |
| `fog79b_l38_best` | 24 | 5368 | 4674 | 694 |
| `fog_user_v2_l31_snow` | 48 | 11397 | 10202 | 1195 |

15格合计 32390 训练行。示范L33香蒲的验证行是0，但训练行310；这不是未采/漏读。L48保留84行、训练71/验证13，与异常段行数一致；不能把读兼容解释为丢掉整个标签。

源文件SHA256：`5ba4063682461c4088915881fb74d1250da5423c8c08e1e6a6e2539f69de6e2b`（/home/newbiexvwu/pvz-logs/curriculum_state.pipeline/round_000018/student/source_rows.json）。

## 16:15 重新部署后的复核（版本边界）

16:15:42 又出现一次 `stop flag`，16:15:51 起新进程，`cq/scripts/curriculum_queue.py` 从 80,468 字节涨到 98,858 字节（sha256 `629ca49a1f152dbfd21d9bd745df35fd5d9e9dbdd2bbfce90f975d88892db17f`），cq HEAD 为 `c84ad7d1`（zones+qcpipe 上线）。这一版**没有** task98 的按格/组隔离（`isolate_cells` 出现 0 次），所以上文几条结论继续成立，只是行号变了：

| 结论 | 旧版（37d235b3） | 现版（629ca49a） |
|---|---|---|
| 组标签只在该轮 plan 全跑完后写入 | :1265 | :1547–1555 |
| `pack_labels` 只读 `state.log` | :663–666 | :678–681 |
| 偏移先占后存（不重采） | :1189–1190 | :1471–1472 |
| 任一组 rc≠0 直接退出 | :1261 | :1543 |
| 起点格以 archive 识别（L25 修复） | :464 | :479 |
| 老师底数查找 | :581 | :596 |

也就是说：**部署没有改掉账本行为**。第二次重启把 round 20 又切了一刀——第一批 13 组（528 局，含 15:37:38 刚结束的 `demo_fog_33_cactus`）与第二批 4 组（168 局，含 L25 的首组 `c20p25origm050` 和 16:15:42 的 `demo_fog_39_cactus`）都没有 log 条目。40 分钟内两次重启、17 组、696 局落在暴露窗口里，这已经不是边角情况。

还有一个新的操作风险：16:15:54 的启动探针只测到 **0.02 GiB/worker、2 秒**（15:38 那次是 0.88 GiB、28 秒）。真实一组一局约 100 秒，2 秒不可能覆盖一局，所以这个值只是 Python 壳的 RSS，不是“python+engine”。后果是 `workers=16/16` 按 20 MB/worker 算出来，而真实占用若仍是 0.85 GiB，16 个就是 13.6 GiB。实测这一组 16:15:54–16:20:48 跑完 48/48、294 秒、无 OOM，但同期 `free` 显示可用内存一度只剩 6 GiB（另一个工作树在跑 4.5 GB 的训练），余量并不宽。建议主控核对 `measure_worker_bytes` 在 demo 起点格上为何只跑 2 秒（`games=1` 的探针在 demo 路径下可能直接结束），别让偏小的探针值决定 worker 数。

## 证据指纹与复核位置

生产日志/模型/npz均未加入git。快照关键文件SHA256：

| 文件（绝对路径） | SHA256 |
|---|---|
| /home/newbiexvwu/pvz-logs/curriculum_state.json | `ab162a899171f5905d66a55fa3026b3ef5e989e4ff62f295e478c326feed819b` |
| /home/newbiexvwu/pvz-logs/curriculum.log | `ce4f3d9cedac1c85d10e44924da5d89146f456298846987b6576f83784c2dc78` |
| /home/newbiexvwu/pvz-logs/teacher_rates.json | `8763a6f6faa5ba3ddc67635e858de6587b1c88aadfbb74c1687195edf2c065ae` |
| /home/newbiexvwu/pvz-logs/gate_seed_ledger.json | `b99286d8ff806301a48b0778169cd0c12194ff9b3ed74fea9a981833a6971687` |
| /home/newbiexvwu/PvZ-search/.claude/worktrees/cq/configs/human_demo_tasks.json | `73061e547426087cb022244b5424dfe05df6d9b6e52a5ebc71174ce85dd21805` |
| /home/newbiexvwu/PvZ-search/.claude/worktrees/cq/artifacts/task73c/panel_v3_source.json | `bb0a9129b88023fe8c7722f267f1d184883ddea538860354421f6664d8e2f636` |
| /home/newbiexvwu/PvZ-search/.claude/worktrees/cq/scripts/curriculum_queue.py | `37d235b30b970bb1d2c3a065d9165fabce2a901f282072a4ec59c5f2385ca17d` |
| /home/newbiexvwu/PvZ-search/.claude/worktrees/cq/scripts/train_rollout_policy.py | `6bffc3e5cbf10c57d5dea7e84a3e465f6c78c31ff5934ac8ad9c9ace91d2ca59` |
| /home/newbiexvwu/PvZ-search/.claude/worktrees/cq/python/pvz_promotion.py | `6a28c360fac4acded269370995861284b00e97ad03de151c36f2cfc8157cdfd6` |

动态文件后续会增长，以上指纹对应本次快照。后续运行结果另记观测时间，不覆盖初始证据。

15:10 完成核验用的第二批指纹（state 停在 15:05:10，round=19）：

| 文件（绝对路径） | SHA256 |
|---|---|
| /home/newbiexvwu/pvz-logs/curriculum_state.json | `699fe315582d455238c2dd097aa400c67b28ef5823255dd5d9e7ba5cec3e404f` |
| /home/newbiexvwu/pvz-logs/gate_seed_ledger.json | `250ba7b9b07892a5639905cfeb2ee7e7ff5f304e87fbb6a6ba769c734778978f` |
| …/round_000018/complete.json | `598eaef49f1a24b464d06f35079a10aa6db19952e1701f42c4b1f26aa4eb0a0c` |
| …/round_000018/evaluations/5324917e2314ca2e713d/decision.json | `fb7cd67936efb0703cffd1e4401e97587a95bb0e626c820946860de84081504f` |
| …/round_000019/complete.json | `cb30b5b48dceac32e427b76a877c5e08c08c646d3b42908b5c72ab6c15cf1018` |

## 16:58 独立证书复算与探针根因

对r18实际评估目录 `/home/newbiexvwu/pvz-logs/curriculum_state.pipeline/round_000018/evaluations/5324917e2314ca2e713d` 再做只读复核：20份games.jsonl共3840行，各任务两边的192个种子与冻结manifest精确一致，无重复、无搜索动作（arm=fast、searched=0），模型/任务/β配置一致；20个对局文件SHA256全部匹配decision。重建1920条paired与证书完全相同，再用复制的纯策略模块重算全部overall/zones/CI/p值/拒绝理由，逐字段相等。registration=decision.contract、contract_key正确；模型双方、引擎、价值网和源面板的当前SHA256均与冻结值相同。r18训练完成、首次v3门禁正确执行且拒绝晋升这两项已得到直接证据。13:00:27前台记录 `student retained: round 18 rejected ...`，当前student_round仍为11。

16:15探针低估的原因已能从代码和清单直接定位，而不仅是2秒运行时长的猜测：`measure_worker_bytes:249–250` 调 `command_for` 时没有传item中的demo_archive/demo_decision/future_seeds；因此它走普通任务分支，使用示范未来流的offset=72作为清单种子偏移。该首格 `demo_roof_42_gloommelon_m200` 的manifest.seeds只有 `[20261042]`，`search_play.py:900` 的 `[72:73]` 是空列表，`jobs:922–924` 因而为0。程序并未校验至少完成一局，空jobs也可正常退出，所以0.02GiB测到的是空任务启动开销。未为复现启动游戏，也未调用生产写入。建议负责人给探针传递完整示范参数，并校验实际完成恰好一局，空任务不得被当作有效RSS证据。

报告审校同时纠正了四类表述：升档须严格>0.85（40/48、31/48、33/48均不升）；登记池自动按序切换第二池；r19成功不等于坏段已修复/未走降级；轮未结束时不得把计划局数写成实际完成量。16:52:57实际round20为22个完成组、936局，state.log尚无round20条目，r20配方尚不存在；1464是当前各计划若全部跑完的合计预测，后续仍需按实际结束证据核对。
