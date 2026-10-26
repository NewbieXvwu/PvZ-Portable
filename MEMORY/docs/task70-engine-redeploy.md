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
  2. **引擎切换**：命令里 `--executable` 换成 `/home/newbiexvwu/PvZ-search/.claude/worktrees/redeploy/build/pvz-portable`（B5 修订后 SHA256 `6264aaf4ad4de7d16f1c46986662abe81d0b692b38595f805f5c9bd881c20117`；16989f23 是 B5 前版本，已留档 artifacts/task70/deploy4_pre_b5/）。一步重启命令（沿用现行参数，仅换 engine 与 scripts 树；workers=16 维持现行）：
     ```bash
     /usr/bin/tmux kill-session -t qc   # 仅在旧控制器已结束后
     /usr/bin/tmux new-session -d -s qc '/usr/bin/env PVZ_NO_POPUP=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/newbiexvwu/.venvs/ml/bin/python3 /home/newbiexvwu/PvZ-search/.claude/worktrees/redeploy/scripts/curriculum_queue.py --rounds 100 --workers 16 --round-games 768 --games-per-group 48 --manifest /home/newbiexvwu/PvZ-search/.claude/worktrees/redeploy/artifacts/stage4/multiplier_panel.json --executable /home/newbiexvwu/PvZ-search/.claude/worktrees/redeploy/build/pvz-portable --state /home/newbiexvwu/pvz-logs/curriculum_state.json --log /home/newbiexvwu/pvz-logs/curriculum.log --out /home/newbiexvwu/PvZ-search/artifacts/hard --frames /home/newbiexvwu/PvZ-search/artifacts/student/frames_v2 --recipe /home/newbiexvwu/PvZ-search/.claude/worktrees/redeploy/recipes/dw4i.json --rollout-policy /home/newbiexvwu/PvZ-search/artifacts/student/dw4i_s1/model.pt --value /home/newbiexvwu/PvZ-search/artifacts/value_net/v4/model.pt --static-value /home/newbiexvwu/PvZ-search/artifacts/hard/value_branch1/model.pt --resource-dir /home/newbiexvwu/.cache/pvz-research-resources 2>&1 | /usr/bin/tee -a /home/newbiexvwu/pvz-logs/qc.out'
     ```
     注意：现行 qc 命令没有显式 --state/--log/--out/--frames/--recipe/--rollout-policy 等（控制器有默认值），主控按现行命令只替换 scripts 树、--manifest、--executable 三处即可；上面是全参数版本供参考。
  3. **新 multiplier_panel**：redeploy 树 `artifacts/stage4/multiplier_panel.json`（426 任务 = 旧 420 逐项相同 + fog_user_v2 六档；generation_audit 已更新，种子扫描 258,874 份清单 280,196 种子重叠 0）。旧 fog_user 六档仍在清单里（数据可追溯），只是 SUPERSEDED 不再排新局。
  4. **teacher_rates**：`~/pvz-logs/teacher_rates.json` 的 `bases` 加 `"fog_user_v2_l31_snow": {"wins": 31, "games": 32, "model": "dw4i_s1 + v5 + 台式机 branch1", "source": "T70"}`（B5 修订后实测 31/32，见 artifacts/task70/b3v5/v2_post_b5/；32 局 deploy4 6264aaf4 老师冷启动，种子 49_100_000..31，与训练段不相交）。
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
  1. 49.1M 块 human_v2（manifest `artifacts/task70/fog_user_v2_panel.json`）32 局：pre-B5 定义 **31/32**（输 49100029，artifacts/task70/b3v5/v2/）；**B5 修订后（6264aaf4）重测 31/32（输 49100010）→ artifacts/task70/b3v5/v2_post_b5/，这就是 teacher_rates 底数**。B5 遮蔽经 student-candidates 翻动了 2 个种子（49100010 胜→负、49100029 负→胜），净效应为零，31/32 不变。
  2. 49.1M 块 human（manifest `artifacts/task70/fog_user_v2_panel_human_control.json`）32 局：**29/32**（输种子 49100013/027/029/031）→ artifacts/task70/b3v5/human/。human 路径 B5 不改（对拍见 [B5]），此数不受 B5 影响。
  3. **配对表（主控要求）**：都赢 28、都输 0、仅 v2 赢 3（49100013/027/031）、仅 human 赢 1（49100029），精确 McNemar 双侧 p=0.625——口径差异不显著。v4 时代的"种子块难"判断作废：49.1M 块在正确叶值下老师近乎全胜，8/32 纯粹是叶值 v4 拖累。
  4. task67 种子 20261101..132 human + **台式机 branch1**（生产静态网）：**29/32**，输种子 20261102/111/124 → artifacts/task70/repro_task67_desktop/。对照 [B3-复现] 的 mac_branch1 25/32（同一叶值 v5、同一 deploy4，只换静态网）：静态网选择本身移动 4 局（输局集合只共享 20261102），teacher_rates 与后续对照一律以台式机 branch1（生产）为准。
- [B3-复现] task67 基准在 deploy4 上逐种子复现：用 task67 原参数（`search_play --arms fast_search --dagger-beta 1 --verify-samples 9 --student-candidates 2 --seeds 32 --workers 4`，manifest `configs/human_demo_tasks.json` 的 `fog_user_l31_snow`，种子 20261101..20261132，叶值 v5 `8a636d33…` + static `pvz-logs/mac_branch1` `91e562a0…`，与 task67 记录的 SHA 一致），仅把引擎换成 deploy4（16989f23…）。结果 **25/32**，且**逐种子一致**：7 个输局种子、输局末波分布（10×1 12×1 19×1 20×4）、输局丢车分布（1×4 3×3）、平均丢车 1.09、赢局全部第 20 波通关——与 fog-user-decks.md §2/§4 记录的 task67 数字逐项相同。结论：换引擎（含 Mac→Linux 跨平台构建）不改变 human 口径下老师的行为，25/32 与 8/32 的差完全归因于种子块与叶值模型（v5 vs v4），不是引擎退化。证据 artifacts/task70/repro_task67/search/games.jsonl。
- [B5] 雾中己方植物遮蔽（主控追加，2026-10-25）：修订 `human_v2` 定义——植物格锚点渲染不透明度 `>= 0.925` 时保留行/列/种类，`mPlantHealth/mPlantMaxHealth/mState/mStateCountdown/mDoSpecialCountdown/mLaunchCounter` 六字段改固定 **-1**（沿用 `target_zombie_id` 未知约定；`launch_rate` 类型常数保持原值）。完整/lean/v2/C++ 学生输入四处走同一个 `EnvironmentPlantFogHidden`（LawnApp）；C++ 学生输入按 JSON 同值的 -1 字段计算（血量通道 -1.0，进度公式作用于 -1），与 python 解码器逐位一致（test_cpp_rollout 5+48 子测试全过）。特权态（老师视图）不遮蔽。特权诊断新增 `fog_plants`（每株植物锚点不透明度，仅 privileged_state）。对拍基线 artifacts/task70/deploy4_pre_b5/（16989f23）：human/xray/None 公开观察逐字节相同（`test_fog_plant_masking` 6 项全过）；human_v2 仅五字段差且差值全为 -1。新二进制 SHA256 `6264aaf4ad4de7d16f1c46986662abe81d0b692b38595f805f5c9bd881c20117`（增量重链，PGO 配置同 A1）。**注意**：`test_fog_family_only_changes_what_is_reported`（task66 的跨模式 state_hash 测试）在 qc 生产负载下抖动——同二进制同模式同种子连跑两次哈希即不同（A2 探针的重置粒子双态线，新旧二进制都抖），特权态逐项相等不受影响。
- [B6] B6 与 deploy5 准备（主控 2026-10-25 14:00 追加的执行结果）：
  1. **B6.1 示范起点格去透视**：`human_demo.reset_archive()/replay()` 支持 `fog_observation` 覆盖（只改呈现不改棋局——state_hash 与特权逻辑态不变，复用 task66 不变式）；`search_play` 示范一致性检查豁免 `fog_observation` 字段、允许 `task_id` 的 `_hv2` 前缀匹配、恢复时传入清单值。`configs/human_demo_tasks.json`：4 个旧示范起点格（L31 snow、L32 snow、L33 cactus/cattail）退休（仅摘除 demo_start 字段，任务与其种子保留），各新增一个 `_hv2` 版本（`fog_observation: human_v2`，archive/阶梯 index 0/未来种子块逐项沿用原格：44.0M/44.1M/44.2M/44.3M 各 ×256——与原格共用同一存档与种子块，只是换呈现口径，不存在相交问题）。测试 `python/test_demo_start_hv2.py`（合成存档，null vs human_v2 恢复：特权逻辑态相等；开阔格真值、雾内格血量/状态 = -1；不传覆盖 == 显式 null）；`python/test_curriculum_queue.py` 注册起点格集合期望改为 4 个 `_hv2` id（54 测试 + 4 子测试过）。示范档已从 cq 树复制到 redeploy 树 `artifacts/human_demo/raw/seed_2026103{1,2,3}/`（git 忽略，不入库），保证本树清单可解析。
  2. **B6.2 fog_user_v2 入课程**：恢复 `scripts/make_multiplier_tasks.py` 的 `CURRICULUM_LEVELS['fog_user_v2']=(31,)` 与 `SUPERSEDED['fog_user']='fog_user_v2'`（1613ecdc 暂撤的两处复原），`test_multiplier_tasks.py`/`test_curriculum_queue.py` 期望随之改回。
  3. **B6.3 引擎 deploy5 门禁**：B5 修订后二进制 SHA256 `6264aaf4ad4de7d16f1c46986662abe81d0b692b38595f805f5c9bd881c20117`（PGO 流程同 A1，redeploy 树 build/pvz-portable）。**逐位门禁 GATE PASS**（照 A2）：六族（day L7 / night L14 / pool L24 / conveyor L10 / roof L41 / fog_user L31）× {老师 β=1, 学生 β=0} × 8 种子，deploy4-pre-B5（16989f23，留档 artifacts/task70/deploy4_pre_b5/）vs deploy5 各 96 局：**动作序列（逐帧 tick+action+rejected）/胜负/末波/决策数/种植数/mowers 0 差异**；state_hash 差 65/96（A2 探针已归因的重置粒子双态线，两引擎共有，特权逻辑态不受影响）。工具 scripts/task70_redeploy_gate.py（新增整组 resume-skip：games.jsonl 已满种子的组跳过，断点续跑不重算）。**吞吐 ABBA PASS**（照 A4）：16 局学生局串行 ABBA，中位配对比 **1.0152**（≤1.03），min 0.757 / max 1.648（生产 qc 挤占噪声），16/16 局决策数逐局相同。证据 artifacts/task70/b6_gate/、artifacts/task70/b6_abba.json。
  4. **B6.4 teacher_rates 增量**（scripts/task70_b6_teacher_rates.py；生产模型 dw4i_s1 + 叶值 v5 8a636d33 + 台式机 branch1 24858410，引擎 6264aaf4，β=1；示范格每格 8 局：demo_start 阶梯 index 0 决策号起手、种子取该格未来块前 8 个）：
     - `fog_user_v2_l31_snow`：**31/32**（[B3v5]/B5 后重测，见 artifacts/task70/b3v5/v2_post_b5/）。
     - `demo_fog_31_snow_m200_hv2` **8/8**（决策 777）；`demo_fog_32_snow_m200_hv2` **8/8**（1231）；`demo_fog_33_cactus_snow_m200_hv2` **8/8**（839）；`demo_fog_33_cattail_snow_m200_hv2` **8/8**（867）。证据 artifacts/task70/b6_teacher_rates/teacher_rates.json。
  5. **部署清单（主控执行；本任务未动 qc/cq/生产状态）**——B4 清单仍然有效，此处只列增量：
     - 代码 rev：redeploy 分支最新提交（含 281921b0 与本次）；引擎与一步重启命令照 B4 第 2 条（`--executable` = redeploy 树 build/pvz-portable，SHA 6264aaf4…）。
     - multiplier_panel：照 B4 第 3 条不变（redeploy 树 artifacts/stage4/multiplier_panel.json，426 任务；`_hv2` 示范格不入面板，走 configs/human_demo_tasks.json）。
     - teacher_rates（~/pvz-logs/teacher_rates.json 的 bases）：照 B4 第 4 条（fog_user_v2_l31_snow 31/32）**再加 4 条**：`demo_fog_31_snow_m200_hv2` / `demo_fog_32_snow_m200_hv2` / `demo_fog_33_cactus_snow_m200_hv2` / `demo_fog_33_cattail_snow_m200_hv2`，各 `{"wins": 8, "games": 8, "model": "dw4i_s1 + v5 + 台式机 branch1", "source": "T70"}`。
     - 回滚：fog_user_v2/fog_user 族 = revert B6.2 提交（LEVELS/SUPERSEDED 复原，fog_user 恢复排新局）；示范格 = revert B6.1 提交（281921b0：旧格 demo_start 恢复、_hv2 撤下；旧格数据从未删除）；引擎 = 换回 deploy4-pre-B5 16989f23（artifacts/task70/deploy4_pre_b5/，B5 的 C++ 修订随二进制一并回退）。

- [B7] 后台任务失败不让 qc 整体退出（主控 2026-10-25 17:00 追加，与 B6 一起进 deploy14）：
  1. **合入修复**：`git merge origin/pvz-env`（redeploy 提交 f02e7f26）带进 51d67c66——`pvz_frame_store.archive_seed()`（`seed_<n>` 与示范起点局 `seed_<示范>_f<未来>` 两种目录统一按未来种子解析）及 `pack_training_rows.py` / `backfill_observation_v2.py` / `relabel_verify.py` / `student_packet_recall.py` 四处调用点，加 test_compact_loader 用例。第 13 轮崩溃的那个解析点已随合并进树。
  2. **B7.1 失败不再杀 qc**（`scripts/curriculum_queue.py` TrainingPipeline）：`poll()` 捕获后台任务异常——job 标 `failed`、`error`/`attempts` 写入 state 持久化、curriculum.log 记原因，**采集继续**；在下一个组边界自动重试一次（status 回 `queued`、重置 failed 门闩重投）；**同一轮连续失败 2 次**（`MAX_BACKGROUND_ATTEMPTS=2`）才写明 "stopping qc" 并 raise 停 qc。attempts 随 state 持久：中途重启后，曾败 1 次的任务只剩 1 次机会。`close()` 收尾 drain 改为只等 `queued` 任务、`poll(retry=False)` 永不重试；收尾期单次失败只记日志（"failed during final drain"）不致命，不再死循环也不再让正常收尾变崩溃。
  3. **B7.2 全仓库 `seed_` 目录名解析扫描**：会碰到示范起点局目录的（默认/参数指向训练语料 frames_v2 的）5 处统一改 `archive_seed`：`teacher_counterfactual_audit.py`（--archives 默认 frames_v2）、`dw4_night_sun_candidate.py`、`dw4_night_card_logit.py`（--frames-root 默认 frames_v2）、`value_widen_audit.py`、`train_value_widen.py`（--data）。其余 `seed_` 解析点结构上只扫各自私有输出树（feasibility_report/value_teacher_sparse 的 teacher out、bench_qc_pack 自建 view、task70_particle_probe 的 gate 输出），碰不到示范目录，未改。
  4. **B7.3 测试**：`python/test_background_failure_retry.py` 3 项——首败仅记录（status/error/attempts 落 state）且组边界照常推进；重试恰好一次；第二次连败 raise（"2 times" + "stopping qc" 日志）；重试成功则整轮照常 done；close 不重试、不挂起。回归 test_curriculum_queue + test_multiplier_tasks + test_compact_loader **71 passed**。
  5. **deploy14 增量**：无引擎变化（仍 6264aaf4）；scripts 树换到含 B7 的 redeploy 最新提交即可，curriculum_queue.py 的失败语义随部署生效。回滚：revert 本提交即回到"后台失败即退"旧行为。

## 问题

- **示范起点格数量对不上（B6.1）**：主控追加里说"现有 5 个示范起点格（含 task71b 的 L34）"，但 redeploy 树的 `configs/human_demo_tasks.json` 在 B6.1 之前只有 **4** 个 demo_start 格（L31 snow、L32 snow、L33 cactus/cattail）。task71b 的 L34 格（`demo_fog_34_cactus_snow_m200`，提交 a6becca3）只在 `origin/demo` 分支上，未合入 redeploy/pvz-env，本次未处理。若要 L34 也去透视：先合 a6becca3，再照同样模式加 `_hv2`（archive 与决策阶梯见 MEMORY/docs/demo-start-states.md），teacher_rates 按同法补测 8 局。

## 主控追加（2026-10-25 14:00，B6 与 deploy14 准备）
B5 收到：human_v2 修订后老师 31/32，作为 fog_user_v2 的 teacher_rates 底数。下一步准备 deploy14，部署由主控做：
1. **发现的漏洞——示范起点格是透视的**：`configs/human_demo_tasks.json` 的示范任务没有 `fog_observation`，录制时为 null，所以 `demo_start` 格里学生看到的是 x-ray。`search_play` 恢复示范时又要求 `environment_task` 逐字段一致。
   - 改法：`fog_observation` 只改呈现、不改棋局（task66 的不变式测试已证明 state_hash 不变），恢复时允许它与录制时不同，使用清单里的值。
   - 按 AGENTS §8 第 4 条新开示范起点族：现有 5 个示范起点格（L31 snow、L32 snow、L33 cactus/cattail、L34 cactus，含 task71b 的 L32/L34）各一个 `_hv2` 版本，`fog_observation: human_v2`，阶梯与种子块沿用原格（与旧格不相交时另取，写清）。旧格退休。
   - 测试：同存档同决策号，human_v2 与 null 两种口径下 state_hash 一致，公开观察按口径遮挡。
2. **fog_user_v2 入课程**：恢复 `CURRICULUM_LEVELS['fog_user_v2']=(31,)` 与 `SUPERSEDED['fog_user']='fog_user_v2'`（主控在 1613ecdc 暂时撤下的那两处），测试随之改回。
3. **引擎**：B5 改了 C++，按 deploy4 的 PGO 流程构建 deploy5，记 SHA；照 A2 做逐位门禁（human/xray 族与 deploy4 逐位一致），照 A4 做吞吐 ABBA。
4. 部署清单写进本文件：新代码 rev、引擎路径、teacher_rates 增量（fog_user_v2_l31_snow 31/32、新示范起点格各 8/8）、回滚方式。push origin redeploy。

## 主控追加（2026-10-25 17:00，B7：后台任务失败不能让 qc 整体退出）
15:30 第 13 轮后台打包崩溃：示范起点局的目录名是 `seed_<示范>_f<未来>`，`pack_training_rows.py` 把它当 int 解析。`Pipeline.poll` 把异常往上抛，qc 整个退出，采集停了 80 分钟。修复在 pvz-env 51d67c66（`pvz_frame_store.archive_seed`），已热补到 cq，qc 16:54 重启并续上第 13 轮。B7 要做：
1. 后台任务（打包 / 训练 / 评估）失败时：记日志，job 标 `failed` 并记下错误，**采集继续**；到下一个组边界重试一次；同一轮连续失败 2 次才停 qc，并在 curriculum.log 写明原因。
2. 全仓库再扫一遍按目录名解析种子的地方（`seed_` + `split`/`removeprefix`），凡是会碰到示范起点局目录的，统一改用 `archive_seed`。
3. 测试：注入一个会失败的后台任务，断言采集继续、重试恰好一次、第二次失败才退出。
B7 和 B6 一起进 deploy14。

## 主控追加（2026-10-25 22:30，B8：deploy14 一次性集成）
主控已把要上线的分支全部合进 pvz-env **a8d00858**，Mac 全量 pytest 569 通过。包含：
- redeploy：B5 / B6 / B7；
- gatepower：task73c 门禁 v3；
- demo：13 个示范起点格，雾夜格按 B6 改成 `_hv2`，屋顶 L42–L49 7 格；
- conveyor2：task75 / 75b，原生 L25 0.5 倍进课程，另有三头豌豆段错误的 C++ 修复；
- fog36：task79 / 79b，L36–L38 厚雾族进课程。

示范存档已拷到 cq 的 `artifacts/human_demo/raw/`（新增 seed_20261039 / 42 / 43 / 44 / 46 / 48 / 49，校验和与 Mac 一致）。B8 要做：
1. 在 redeploy 工作树 `git merge origin/pvz-env`（目标 a8d00858），把示范存档同步到这个树。
2. **引擎**：a8d00858 的 src 相比 6264aaf4 多了 task75 的 C++ 改动（L25 体型字段、L40 storm 遮挡、三头豌豆修复）。按 A1 的 PGO 流程构建 deploy5b，记 SHA，然后：
   - 照 A2 对 6264aaf4 做逐位门禁：六族 × 老师 / 学生 × 8 种子；
   - 照 A4 做吞吐 ABBA；
   - 门禁不过就停下写原因，不要带病上线。
3. **清单**：重新生成 `artifacts/stage4/multiplier_panel.json`，旧任务逐项不变，只能新增。用 generation_audit 机械核对种子不相交。
4. **teacher_rates 增量**：列出新进课程的格和它们的底数来源：
   - fog_user_v2 L31：31/32；
   - fog79b L36 / L37 / L38：task79 的 r11 数；
   - native_l25 0.5 倍：13/16；
   - 13 个示范起点格：B6.4 的做法。
   给出写入 `~/pvz-logs/teacher_rates.json` 的 JSON 片段。
5. **dry-run**：用新清单 + 新引擎 + 新 scripts 跑 `curriculum_queue.py --dry-run`，确认能读现行 `curriculum_state.json`。门禁 v3 首次运行时要能正确领种子，并在新区（fog / roof）上评估现役。在本树 `--state` 的拷贝上试，不碰正式 state。
6. 写一份 `~/pvz-logs/qc_deploy14.sh`，结构照 `qc_deploy13.sh`：
   - 组边界等待，备份 state；
   - checkout a8d00858（或 B8 合并后的提交）的 scripts / python / recipes / configs 到 cq；
   - 拷引擎到 `cq/build5/`；
   - 先 dry-run，失败就回滚；
   - 重启，并写 curriculum.log。
   **只写脚本、不执行**，由主控执行。
7. 结果写本文件，`git push origin redeploy`。每个任务最多 2 个游戏进程。

## B9：task95 / deploy15 引擎（2026-10-26，build2已验收，v2boss修复版build4验证中）

本节对应[task95任务书](task95-deploy15-engine.md)。只准备引擎与部署脚本；不执行部署，不操作生产qc/cq。本任务游戏进程上限2；吞吐阶段串行运行1个游戏进程，其他任务不作调整。

22:40续做：build2虽通过现有策略门禁，但不能加载Python导出的v2boss模型。三处通道修复acc3b431已用于PGO归档build4（de952714651c6544f4dba16f8ae832fe1ad1cb75bfc3e1d5495964d3d9a83845），可加载并前向；build4六族96对动作/逻辑0差异、L50四种子对拍均PASS，正在串行ABBA，随后补候选哈希。下表暂保留build2历史结果，最终候选与脚本待build4整体验收后统一更新。

| 项目 | 证据与结论 |
|---|---|
| 源码差异 | a8d00858至a630fbd842f6e8a2493f09de3d433608bd71a254，src共7文件、+137/-11；task82初始化、task85增量Boss观测、task81特权增量字段及demo测试缝。具体逐项影响见task95结果；预期不改现有策略的游戏逻辑。 |
| PGO留档 | 首建`artifacts/task95/engine/pvz-portable`，SHA256 `07703a5781e70d2c3b1b3ad7b0c473ddfaf2a313ac2ce4f5724035ba44e184be`，141份profile，Release/O3/LTO、ffp-contract=off。**该构建吞吐不过（见下行），不作候选**：`/usr/local/lib/libopenmpt.a`（10-25 17:02 新装）让`find_library`选中静态库，二进制68.7 MB、`.text` 4.55 MB，deploy14是10.4 MB / 2.48 MB。按deploy14的共享库链接重建`artifacts/task95/engine/pvz-portable-build2`，SHA256 `83aed3bed2b74392b20367915d84908faf9384a986a115893baecdccf65f2029`（10.65 MB、`.text` 2.70 MB、`NEEDED`含`libopenmpt.so.0`）。原构建保留不覆盖。 |
| 旧引擎 | 本任务留档`artifacts/task95/engine/deploy14-pvz-portable`，SHA256 `76ff36dc026f67641aefccd5f30f6cdae65eee14ee197dc24497413123a7874a`。 |
| 六族门禁 | **候选83aed3be PASS**：6族×老师/学生×8种子、每引擎96局。候选臂17:03–18:44跑完，96对完整动作流（非截断前缀）与全部逻辑字段0差异；旧臂36个`games.jsonl`与192个帧档复用原门禁副本，`diff -rq`逐字节一致。state_hash新旧差96/96属task82初始化字段（见下行）。原07703a57同样96对0差异，结论保留。门禁命令由`command_for`生成，该函数在部署树与当前 cq 代码中逐字节相同；`search_play`/`pvz_tasks` 的差异只在`deck_random`分支，四个生产清单无此字段，故门禁命令与牌组同生产一致。证据`artifacts/task95/gate_b2/gate_evidence.json`。 |
| 同引擎确定性 | L7 seed30000探针：07703a57与83aed3be各8个独立进程的reset/60/660tick哈希线全部相同，同进程4次reset后60tick也全同；旧引擎对照分别3种哈希线、4种哈希。**六族正式任务复核（候选83aed3be）**：按A2六族清单首种子、地形匹配deck，每族8独立进程+同进程4次reset、三个时间点（0/60/660tick），候选在6/6族、两种模式下均只有1条哈希线，且两模式相互一致（PASS）；旧引擎同法6/6族不稳定（多进程2–4种、同进程1–2种），task82的效果在全部六族上成立。证据`artifacts/task95/state_hash_suite_build2.json`。 |
| 吞吐ABBA | **83aed3be完整16对PASS**：逐对old/new中位数0.9934474352≥0.98，决策数16/16相同。续跑保留已落盘9对及第10对new臂，运行产物与归档SHA相同，未重测已完成记录。07703a57两次完整实测0.9477023114与0.9687719632均失败，全部保留。静态/动态链接偏差已确认，函数尺寸不能单独排除源码性能影响；最终验收使用完整ABBA。[逐对数字与输入指纹](task95-throughput.json)。 |
| L50 | **候选83aed3be PASS**：`boss_l50_orig_eval` 种子85000000..3，旧/新引擎各4局，同用A2 `dw4i_s1`、beta=0、`conveyor_start=original` 生产形态命令。4/4 逐帧动作流一致，全部逻辑字段（终局、final_tick 5895–7720、决策数、mowers 等）一致；候选Boss块存在且字段齐全（active/stage/head/action/target_row/target_col/projectile，107–144帧/局，动作含head/idle/summon），旧引擎无该块。四局均到终局（均输），结论不依赖是否读取Boss字段。证据`artifacts/task95/boss_l50.json`。**边界**：本行只证明 JSON `boss` 块与终局一致；`observation=6` 向量路径有通道数不一致（C++ 按 106 加载/分配、`Encode` 只传 101，Python 为 105），当前无 v2boss 模型，且 Python 导出的 v2boss 模型已被 build2 以`invalid rollout network weights`拒绝（已复现），故不宣称 v2boss 学生可用；修复分支`task95-bossfix`已在非 PGO 临时构建上验证可加载并前向，生产候选仍是 build2；详见 task95 问题。 |

模型固定为A2夹具：`dw4i_s1`、`value_net/v4`、`value_branch1`，不把旧夹具标成现役v5。二进制/模型指纹、逐族统计及新旧探针数值见[task95-engine-gate.json](task95-engine-gate.json)。吞吐原始记录留在本地`artifacts/task95/`，未提交二进制、帧归档或日志。

部署脚本版本见[scripts/qc_deploy15.sh](../../scripts/qc_deploy15.sh)，与`/home/newbiexvwu/pvz-logs/qc_deploy15.sh`内容一致，两份`bash -n`均通过，**只写、未执行**。以**当前生产代码**为基线：deploy14c 已于 16:15 上线 deploy14fix `0b8dcc26` + `fd1a0b61` 的五个文件（`curriculum_queue` / `pvz_tasks` / `search_play` / `select_student` / `train_rollout_policy`），面板、teacher_rates、state 未动。deploy15 **不 checkout 代码**（deploy14 有这一步；本任务只换引擎，回退代码会撤掉 deploy14c），改为只读断言：`0b8dcc26` 之后 `scripts python recipes configs` 的改动集必须恰好是这五个文件，且五个 blob 与 `fd1a0b61` 逐一同哈希，否则拒绝部署。不更换面板/teacher_rates/配方，只复制归档候选至`cq/build6/`并切换引擎路径。执行前验证六族、吞吐、Boss与哈希证据；未完成或失败即拒绝部署。回滚仅换回`cq/build5/`旧引擎。**最终候选为83aed3be（build2）**：`ENGINE_SHA`、归档路径与四条证据路径（`gate_b2/gate_evidence.json`、`abba3/throughput_abba.json`、`boss_l50.json`、`task95-engine-gate.json` 的 `state_hash_probe_build2` 与 `six_family_hash_suite`）已全部指向build2；07703a57因吞吐不过被排除，记录保留。前置检查的Python块已按脚本参数单独运行通过（只读断言，未执行部署），两份脚本`bash -n`通过且内容一致。`~/pvz-logs/qc_deploy15.sh` 仍**只写、未执行**，等主控执行。
