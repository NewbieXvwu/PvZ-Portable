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

建议主控按采集优先级补老师底数。现有底数是base级而非档位级：fog v2 31/32及fog79b数据源为1.0，升到1.25后仍沿用；示范底数也是特定起点的小样本，不能声称每个新档都已测老师。

同秒重复 added 日志并不等于重复执行采样：08:50:46 两套初始化与 deploy14 dry-run/启动吻合；只有 L25 的误退在11:43仍复现，其余新增格的窗口/offset没有被清空。后续应按实际组标签与偏移核验，不能数added行当组数。

## r18 热补与 v3 首次运行：待运行完成

- 11:25:23、11:30:35 两次训练启动均因 plane_of broadcast 失败；11:33:37 达到2次失败停qc。
- 11:43:42 主控重启后，11:46:17重走pack，11:48:57 pack done并启动训练。pack.log 说明 built=0、reused=475（不是坏段重建），总18,466,229决策；train.log已完整读入并打印packet/spread权重，跨过旧崩溃点。首次核验训练PID861864存活、未产model.pt，不能据此宣称训练完成。
- 对r18实际pack `1216b129a66d4b74.npz` 的475段逐一只读检查 NPY headers：唯一非空plane_of行数不符为 `ce3ce4006691a0e1.npz` board，offsets所示84行、plane_of72。现 `train_rollout_policy.py:609–615` 在长度不符时生成−1标记，降级为不使用planes；这是读兼容热补，task99根因修复仍须覆盖这段短尾示范。
- job18 已冻结 **[39344,41264)**，长1920，账本allocated中恰有一次；前块[37808,39344)紧邻，未重叠。仍从第一池[31024,65536)领，未跳第二池[100000000,101000000)。v3首轮不应沿用1536长度。按v3十任务顺序分片，fog应为 **[40880,41072)**、roof **[41072,41264)**；须待首次门禁实际manifest与结果再核对，不把已领号当已运行。
- 训练完成、首次v3门禁实际结果和证书仍待观察，task100未完成。

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
