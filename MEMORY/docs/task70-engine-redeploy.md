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

## 问题
