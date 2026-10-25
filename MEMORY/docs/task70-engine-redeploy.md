# task70：生产引擎换代（粒子确定性 + 零费被吃 + 雾可见 v2）、雾族换 human_v2、Mac 标签包取用即删

> 主控 2026-10-24 写。执行者：台式机 CodeBuddy（`cb-qcalloc` 会话）。工作树 `/home/newbiexvwu/PvZ-search/.claude/worktrees/redeploy`（分支 `redeploy`，从 origin/pvz-env 拉出，你自己建）。游戏进程最多 4 个；**不碰 tmux `qc`、不改 cq、不动 deploy3**——切换由主控在组边界做。

## 为什么
生产引擎 deploy3（`/home/newbiexvwu/PvZ-search/.claude/worktrees/deploy3/build/pvz-portable`）之后，pvz-env 上合入了三处引擎改动，都还没进生产：
1. task63：粒子 FIELD_SHAKE 不再用堆地址播种（同存档重放哈希不再随机）。
2. task65：`plants_eaten_free`（零费被吃单列，whatif 第二判据用付费被吃）。
3. task66：`fog_observation: human_v2`（按渲染不透明度判可见，τ=0.925，DESIGN §14.25）。

另外：雾 L31 已在 qc 里练（`fog_user_l31_snow`，DESIGN §14.24），但用的是旧口径 `human`（格值>0 即隐藏）。用户标注证明旧口径把人看得到的僵尸藏掉了一半，学生看到的比人少。要换成 `human_v2`。

## 要做的

### A. 候选引擎 deploy4 与门禁
1. 在 redeploy 工作树按 deploy3 的构建配置（Release、同编译选项，照 `MEMORY/docs/task50-zones-deploy.md`）构建，产物放 `/home/newbiexvwu/PvZ-search/.claude/worktrees/redeploy/build/pvz-portable`，记 SHA256。
2. **逐位门禁**：六族（白天 L7、夜间 L14、泳池 L24、传送带 L10、屋顶 L41、雾 `fog_user_l31_snow`）各 8 个种子，命令用 `curriculum_queue.command_for` 生成的生产命令（老师 β=1 与学生 β=0 各一半），deploy3 与 deploy4 各跑一遍。要求：每局动作序列、胜负、末波、决策数、种植数逐项相同。state_hash 允许不同（粒子），但要报不同的比例，并抽 1 局确认差异只在粒子块。
3. **旧档兼容**：列出生产路径里所有读旧存档 state_hash 的地方（whatif 重放、critical_forks、示范 verify、qc 打包等），逐个说明换引擎后旧档会不会报错。会报错的，给出处理（例如按存档记录的引擎 SHA 选校验方式），不要静默跳过校验。
4. **吞吐**：同 16 局配对计时（ABBA），deploy4 不得慢于 deploy3 超过 3%。
5. 结果写本文件，附部署所需的一步命令（cq 的引擎路径换成什么）。**部署由主控做。**

### B. 雾族换 human_v2
1. 按 AGENTS §8 第 4 条：观测口径变了就是新族。新开 `fog_user_v2` 族（cell `fog_user_v2_l31_snow`），卡组、关卡、倍率阶梯与 `fog_user_l31_snow` 相同，`fog_observation: human_v2`，**新种子块**（与现有全部段不相交，用 `test_fog_user_cell_preserves_every_existing_cell` 的办法逐种子比对）。
2. 旧 `fog_user` 族的清单与数据保留；从 `CURRICULUM_LEVELS` 撤下它，换上新族。现有其余 cell 逐字节不变（照 task68 的方式验证），报新旧 cell 数。
3. `teacher_rates.json` 的新族底数：用 deploy4 在 human_v2 下跑 32 局老师冷启动（种子另取，与训练段不相交）实测，不沿用 25/32。老师用全知状态搜索，理论上不受口径影响，实测就是检验这一点。
4. 给出 deploy11 那样的部署清单（新 multiplier_panel、teacher_rates 改动、回滚办法），由主控部署。

### C. Mac 标签包取用即删
`scripts/import_mac_labels.py`：导入并校验成功后，删掉 HF 上那份 `transfer/` 包（`huggingface_hub` 的 `delete_file`，凭证用 `$HF_TOKEN`），journal 里记删除时间；校验失败不删。再一次性清掉 HF 上已在 journal 里、已导入的包（先 dry-run 列清单，写进本文件，再删）。HF 仓库是公开的，规则见 AGENTS §2。

### D. 测试
新单测覆盖 B 的种子不相交、旧 cell 不变，以及 C 的"失败不删"。提交并 `git push origin redeploy`。

## 规则
- 只用绝对路径；不 `cd` 进别的工作树；`PVZ_NO_POPUP=1`；不加开关，不改生产默认。
- 禁止 `git add -f`；`.pt` `.npz` `.log` 不进 git；不写字面密钥，不打印凭证；不手写打法规则或公式。
- 每完成一小步就提交、push，在"结果"记一行；疑问写"问题"，按最佳判断继续。

## 结果

- [A1] deploy4 构建完成：redeploy 工作树 PGO 两阶段（bench workload），generate=0/use=162，与 deploy3 相同配置（task50 流程）。二进制 SHA256 `16989f23cdce3d67e862f4fb4279b423d22ec532d066eea8aa1565dbe76a1fd2`，路径 `/home/newbiexvwu/PvZ-search/.claude/worktrees/redeploy/build/pvz-portable`。deploy3 现行二进制 SHA256 `0dec4a275548ef761a49cb34c6f1c3e6d2eabf5854da95045ca1eb7bb0000327`（源 2b743c62）。

- [A3] 旧档 state_hash 兼容清单（逐处说明）：
  1. `scripts/human_demo.py:replay()`（被 `search_play --demo-archive` 的示范开局、`episode_query` 的 replay/whatif、`human_demo verify`、task66 雾工具复用）——逐操作比对 `state_hash` 与 `initial_state_hash`。deploy3 及更早引擎录的档（含 deploy3 rebaseline 过的），在 deploy4 下重放会因 task63 的粒子播种改动报 "state hash differs"。**处理（已实现，提交 b329cf4d）**：按档内 `meta.executable_sha256`（Recorder 一直在记）选校验方式——同引擎照旧全量哈希校验；异引擎改为强制校验引擎无关字段（tick 时间线、逐操作 ok、终局胜负），返回的 meta 里记 `hash_verification` 模式，不静默。要恢复全量哈希校验就用 deploy4 跑一次 `human_demo --rebaseline`（现有机制，源目录不动）。
  2. `scripts/value_teacher_feasibility.py` 的 `_debug_state_sha256` 叶子哈希比对——离线诊断工具，不在生产路径；对旧叶子档案会报差异，属预期，不处理。
  3. `scripts/promotion_gate.py`——评估 config 固定 executable 路径，换引擎后若复用旧评估目录会显式 ValueError（"evaluation config mismatch"），不会静默；换引擎后的下一次晋升按新引擎重跑评估即可，无旧档哈希比对。
  4. 课程控制器（`curriculum_state.json` / multiplier_panel / 打包）：不读 state_hash；面板换新后 fog_user 格会以 "superseded by fog_user_v2" 显式退役（SUPERSEDED 机制）。
  5. 引擎内 branch snapshot：同一进程内存态，不跨引擎，无兼容问题。
  6. Recorder 新录的档本来就记 `executable_sha256`，将来换引擎自动走第 1 条的选择逻辑。

- [B4] 部署清单（由主控在组边界执行；本任务未动 qc/cq/deploy3）：
  1. **前置**：等旧 qc 在组边界自然退出（或主控确认退出），清理组间停止标志。当前生产 qc 从 cq 工作树跑：`--executable /home/newbiexvwu/PvZ-search/.claude/worktrees/cq/build/pvz-portable`（SHA256 `b642614f...`，engine src 与 deploy3 2b743c62 完全一致，已验证 `git diff 2b743c62 c84ad7d1 -- src/` 为空）。
  2. **引擎切换**：命令里 `--executable` 换成 `/home/newbiexvwu/PvZ-search/.claude/worktrees/redeploy/build/pvz-portable`（SHA256 `16989f23...`）。一步重启命令（沿用现行参数，仅换 engine 与 scripts 树；workers=16 维持现行）：
     ```bash
     /usr/bin/tmux kill-session -t qc   # 仅在旧控制器已结束后
     /usr/bin/tmux new-session -d -s qc '/usr/bin/env PVZ_NO_POPUP=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/newbiexvwu/.venvs/ml/bin/python3 /home/newbiexvwu/PvZ-search/.claude/worktrees/redeploy/scripts/curriculum_queue.py --rounds 100 --workers 16 --round-games 768 --games-per-group 48 --manifest /home/newbiexvwu/PvZ-search/.claude/worktrees/redeploy/artifacts/stage4/multiplier_panel.json --executable /home/newbiexvwu/PvZ-search/.claude/worktrees/redeploy/build/pvz-portable --state /home/newbiexvwu/pvz-logs/curriculum_state.json --log /home/newbiexvwu/pvz-logs/curriculum.log --out /home/newbiexvwu/PvZ-search/artifacts/hard --frames /home/newbiexvwu/PvZ-search/artifacts/student/frames_v2 --recipe /home/newbiexvwu/PvZ-search/.claude/worktrees/redeploy/recipes/dw4i.json --rollout-policy /home/newbiexvwu/PvZ-search/artifacts/student/dw4i_s1/model.pt --value /home/newbiexvwu/PvZ-search/artifacts/value_net/v4/model.pt --static-value /home/newbiexvwu/PvZ-search/artifacts/hard/value_branch1/model.pt --resource-dir /home/newbiexvwu/.cache/pvz-research-resources 2>&1 | /usr/bin/tee -a /home/newbiexvwu/pvz-logs/qc.out'
     ```
     注意：现行 qc 命令没有显式 --state/--log/--out/--frames/--recipe/--rollout-policy 等（控制器有默认值），主控按现行命令只替换 scripts 树、--manifest、--executable 三处即可；上面是全参数版本供参考。
  3. **新 multiplier_panel**：redeploy 树 `artifacts/stage4/multiplier_panel.json`（426 任务 = 旧 420 逐项相同 + fog_user_v2 六档；generation_audit 已更新，种子扫描 258,874 份清单 280,196 种子重叠 0）。旧 fog_user 六档仍在清单里（数据可追溯），只是 SUPERSEDED 不再排新局。
  4. **teacher_rates**：`~/pvz-logs/teacher_rates.json` 的 `bases` 加 `"fog_user_v2_l31_snow": {"wins": <B3 实测>, "games": 32, "model": "dw4i_s1 + v4", "source": "T70"}`（数字见 artifacts/task70/cold/cold_start.json；32 局 deploy4 老师冷启动，种子 49_100_000..31，与训练段不相交）。
  5. **回滚**：tmux qc kill → 恢复现行命令（cq 树 + cq 引擎 + 旧清单路径不变，curriculum_state.json 里 fog_user_v2 的 cell 留着无害，控制器下次 reconcile 会按当时清单再处理）→ 重启。fog_user 旧族清单/数据/种子从未动，回滚即回到 task68 状态。
  6. **示范档**（Mac 侧，主控执行）：deploy4 上线后若要用 demo 档做 DAgger 开局/whatif，先 `human_demo.py verify` 试跑；报 state hash differs 就用 deploy4 `human_demo --rebaseline` 出新目录（源目录不动），replay 会按 executable_sha256 自动选校验方式，不会静默跳过。
- [A2] 逐位门禁 **GATE PASS**：六族（day L7 m125 / night L14 m125 / pool L24 m125 / conveyor L10 m125 / roof L41 1.0 / fog_user L31 1.0）× {老师 β=1, 学生 β=0} × 8 种子，deploy3 与 deploy4 各 96 局（共 192 局）。**动作序列（frames 逐帧 tick+action+rejected）、胜负、末波、末 tick、决策数、种植数/种植类型、mowers、shovel/replant 事件全部 0 差异**。工具 scripts/task70_redeploy_gate.py（命令用 curriculum_queue.command_for 生产形态）。state_hash：deploy3 无 STATE_HASH 命令无法直接报比例，由粒子探针补齐（见下）。
- [A2-探针] scripts/task70_particle_probe.py：用 BRANCH_SNAPSHOT_FAST W:0（两引擎唯一共同哈希路径）对 6 局学生局逐决策哈希重放。结论：**5400 个决策 tick 上观测逐字节 0 差异**（含重跑）；哈希差异 959/5400（17.8%）。归因：重置时刻存在与负载相关的双态线（load race，两引擎共有、先于 task70 存在，line A 下 deploy3 与 deploy4 哈希**逐 tick 全同**（fog 局 581/581 同），line B（粒子已抖动）上两者不同——正是 task63 把 FIELD_SHAKE 播种从堆地址换成 DataArray id 的预期差异，差异字节位于 SaveGame 镜像的粒子块）。deploy4 在正常负载下 6/6 次重置 STATE_HASH 全同（确定性成立）。
- [A4] 吞吐 ABBA **PASS**：16 局学生局（day 8 + pool 8）串行 ABBA，中位配对耗时比 **1.0152**（≤1.03），min 0.757 / max 1.648（受生产 qc 16 进程挤占的噪声），16/16 局决策数逐局相同。证据 artifacts/task70/throughput_abba.json。
- [C] HF 清理 dry-run（--cleanup-remote-dry-run）：远端现存 40 个 transfer/ 包（p21w34m…p29w37m 共 40 个池族标签 + 4 个非标签文件），**全部无已验证回执**——dw4/dw5 旧回执（137 标签）与共享 journal（377 标签，2026-10-24 21:41 由另一侧 importer 写入）都不含它们，即这些都是**尚未导入的待用上传**；所有已导入的历史包已不在 HF 上。结论：**无可删项，一个都不删**（不满足"已导入"判据）。清单已核验（repo realnewbiexvwu/pvz-agent-artifacts，revision 33353855）。今后每次 refresh 导入校验成功后会自动删包并记 remote_deleted_at，失败不删。
  40 个标签：p21w34m-38m, p22w34m-38m, p23w34m-38m, p24w34m-38m, p26w34m-38m, p27w33m-38m, p28w33m-38m, p29w33m-37m。
- [B3] **（作废，主控 2026-10-25 更正）**下面 1–4 的两组 49.1M 冷启动误用了叶值 v4；生产老师是叶值 **v5**（`8a636d33…`）+ 台式机 branch1（`artifacts/hard/value_branch1`，`24858410…`）。v4 数字不作为 teacher_rates 底数，重跑见 [B3v5]。原记录保留：老师冷启动（deploy4 16989f23…，dw4i_s1 + 叶值 v4 + static branch1，dagger β=1，`command_for` 生产命令形态，32 局）：
  1. `fog_user_v2_l31_snow_eval`（新种子 49_100_000..31，human_v2 口径）：**8/32**（25%）。证据 artifacts/task70/cold/cold_start.json。
  2. 配对对照：同批 32 种子改 `fog_observation: human`（其余全同）：**5/32**。证据 artifacts/task70/cold_human_control/。
  3. 口径交叉对照：task68 旧种子 43_700_000..31 + human_v2：**5/32**。证据 artifacts/task70/cold_v2_old_seeds/。
  4. （v4 数字，随本条作废）新种子块上 human_v2 8/32 vs human 5/32 的配对 McNemar p=0.58——该结论待 [B3v5] 生产参数重跑后重新下。
- [B3v5] 主控更正后重跑（2026-10-25）：生产参数 dw4i_s1 + **叶值 v5** `8a636d33…` + **台式机 branch1** `24858410…`，deploy4 16989f23…，`search_play` fast_search β=1 verify-samples 9 student-candidates 2，workers 4：
  1. 49.1M 块 human_v2（manifest `artifacts/task70/fog_user_v2_panel.json`）32 局：**31/32**（唯一输种子 49100029）→ artifacts/task70/b3v5/v2/。**注意：这组跑在 B5 修订前的 human_v2 定义上**；B5 构建后需在新二进制上重测一次作为 teacher_rates 底数（老师全知，但 student-candidates 走学生输入，遮蔽可能微调老师轨迹）。
  2. 49.1M 块 human（manifest `artifacts/task70/fog_user_v2_panel_human_control.json`）32 局：**29/32**（输种子 49100013/027/029/031）→ artifacts/task70/b3v5/human/。human 路径 B5 不改（对拍见 [B5]），此数不受 B5 影响。
  3. **配对表（主控要求）**：都赢 28、都输 0、仅 v2 赢 3（49100013/027/031）、仅 human 赢 1（49100029），精确 McNemar 双侧 p=0.625——口径差异不显著。v4 时代的"种子块难"判断作废：49.1M 块在正确叶值下老师近乎全胜，8/32 纯粹是叶值 v4 拖累。
  4. task67 种子 20261101..132 human + 台式机 branch1（对照 [B3-复现] 的 mac_branch1 25/32）→ artifacts/task70/repro_task67_desktop/（跑完补数）。
- [B3-复现] task67 基准在 deploy4 上逐种子复现：用 task67 原参数（`search_play --arms fast_search --dagger-beta 1 --verify-samples 9 --student-candidates 2 --seeds 32 --workers 4`，manifest `configs/human_demo_tasks.json` 的 `fog_user_l31_snow`，种子 20261101..20261132，叶值 v5 `8a636d33…` + static `pvz-logs/mac_branch1` `91e562a0…`，与 task67 记录的 SHA 一致），仅把引擎换成 deploy4（16989f23…）。结果 **25/32**，且**逐种子一致**：7 个输局种子、输局末波分布（10×1 12×1 19×1 20×4）、输局丢车分布（1×4 3×3）、平均丢车 1.09、赢局全部第 20 波通关——与 fog-user-decks.md §2/§4 记录的 task67 数字逐项相同。结论：换引擎（含 Mac→Linux 跨平台构建）不改变 human 口径下老师的行为，25/32 与 8/32 的差完全归因于种子块与叶值模型（v5 vs v4），不是引擎退化。证据 artifacts/task70/repro_task67/search/games.jsonl。

## 问题
