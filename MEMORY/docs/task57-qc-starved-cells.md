# task57：qc 课程里有 13 个格从没被采过——找出原因，改分配，让老师会赢而学生不会的格拿到数据

> 主控 2026-10-24 06:40（台式机时间）写。执行者：台式机 Codex（Business，codex3）。工作树 `/home/newbiexvwu/PvZ-search/.claude/worktrees/qcalloc`（分支 `qcalloc`，从 pvz-env 拉出，主控已建好）。最多 2 个游戏进程；**不碰 tmux `qc`**。

## 现象（主控 06:35 读 `~/pvz-logs/curriculum_state.json`）
- 以下格的 `last_round` 都是 0，即只有初始化窗口、**从没采过一组**：
  - `night_l19_train_n10`、`night_l19_train_n10q`、`night_l20_train_n10_orig`；
  - `pool_l21`…`pool_l29_train_p9k`，共 8 格；
  - `roof_l41_train_r7`（04:11 新加）。
- task56（分支 `l20gap`）实证：L20 老师 29/32、学生 0/128，**dw4 与 dw5i_A 的训练集里 L20 来源都是 0**。学生不会，是因为这关的数据从来没进过训练。
- task55（`MEMORY/docs/adventure-gap.md`）也写了：最近完整一轮 768 局只覆盖白天 / 夜间。
- 读 `scripts/curriculum_queue.py` 的 `allocate`：
  - 先按 `last_round` 最久的轮转"防遗忘"组（预算 `budget * guard_ratio`）；
  - 余下的给胜率在边缘区间、离 0.675 最近的格。学生胜率 0 的格（L20、泳池）离 0.675 最远，几乎排不上。
  - 但 last_round=0 的格按理应排在防遗忘轮转的最前面，实际上没被选到，说明还有别的原因。
  - **先查清实际原因**：guard_ratio 的值、重启 / 续轮时计划怎么生成、`last_round` 何时写回等。

## 要做的
1. **复盘**：用 state 与 `curriculum.log` 重放最近几轮的 `allocate`，解释每一轮为什么选了那些格、为什么没选这 13 格。结论写成表。
2. **改分配**（唯一实现，不加开关）。原则：
   - 采集的价值在于**老师能纠正、学生还不会**的地方。学生胜率 0 但老师会赢的格（L20：老师 29/32），正是最该采的；
   - 学生胜率只是信号之一。用已知的老师胜率（面板 / 门禁记录）或少量老师探针，估"可教差距"，参与排序；
   - 每个格都有最低采集频率保底，不能饿死；
   - **全败、老师也赢不了**的格（雾区那种）不在这里解决，保持现状，不进 qc；
   - 学生在这些格里常常开局就崩，beta0（学生执行）采到的状态只有开局。要不要对这类格提高 beta（老师执行一部分），用 DAgger 的口径讨论利弊，写出建议。**这个改动由主控决定**，你先只做分析和可选实现。
3. **验证**：
   - 用本工作树独立的 state 副本做 dry run，展示新分配下接下来 5 轮每轮选哪些格、每格多少局；
   - 现有测试全过，新分配补单测，覆盖"饿死格被保底""老师会赢学生全败的格优先"。
4. **配方**：task56 说 qc 采到的新组会按普通流程进入后台训练（流水线的 `pack_labels`）。核实这一点；同时把已有的 32 局 L20 初始化档案（`n20o1_fast_search`）补进训练来源。若需要改配方，按 AGENTS 规则新开配方文件，不覆盖 `recipes/dw4i.json` 的历史语义。
5. 在"结果"写部署说明。**主控在 qc 组边界切换**。

## 规则
- 只用绝对路径；不 `cd` 进别的工作树；不手写打法规则（分配规则属于课程控制，可以写，但要以测量数据驱动）。
- 禁止 `git add -f`；`.pt` `.npz` `.log` 不进 git；不写字面密钥。
- 每完成一小步就提交、`git push origin qcalloc`，在"结果"记一行；疑问写"问题"，按最佳判断继续。

## 结果

- [x] 复盘（2026-10-24 07:4x，state round=2 + curriculum.log 全史重放 + 真实 allocate 重演）：
  - 饿死是三个机制叠加，不是数据缺失：
    1. **guard 预算算术**：guard_budget = max(per_group, int(budget×0.12)) = max(48, 92) = 92 局，
       而一组 48 局、两组 96 > 92 → **每轮防遗忘只轮转 1 个格**（38 格要 38 轮才轮一圈）；
    2. **并列按字母序**：guard 候选按 (last_round, base) 排，last_round=0 的 13 格里
       day/night 前段字母先轮走，night_l19/pool/roof 永远垫底；
    3. **边缘带垄断剩余预算**：adjust() 把 >0.85 升档、<0.5 降档，中难度格永远停在
       [0.5,0.85] 边缘带内；每轮 ≥15 个边缘格把剩余 720 局吃光。全败格在 others 里按
       |win−0.675| 排序，win=0 距离最大 → **恰好排最后**，与教学价值倒序；无数据格
       距离 0.0 理应排最前，但预算到不了。
  - 三轮 2304 局全部流向 day/night 中难度格（部分格三轮被采 3 次），13 格 last_round=0。
  - 附带发现：allocate 在 plan 时就写 last_round=round_index，控制器若在组间崩溃重启，
    本轮未跑的组不会被重排（靠 gap 排序的 age 兜底）——未修，记录在案。
- [x] 改分配（唯一实现，无开关）：scripts/curriculum_queue.py 的 allocate 重写为
  **保底（FLOOR_ROUNDS=4 轮内必采一次，每格 per_group//2 局，预算 ceil(格数/4) 组/轮）**
  + **可教差距排序（老师胜率 − 学生胜率，降序；老师无证据按可教处理，同分最久未采先）**。
  老师证据读 --teacher-rates（默认 ~/pvz-logs/teacher_rates.json，缺文件 = 全部未知）。
  证据文件已生成：level→{wins,games,model,source}，来自 task55 冻结表现役老师测量
  （T52/T50/TV4/P，只收 games≥8）：L7=16/16, L14=32/32, L17=31/32, L20=29/32,
  L21=17/32, L22=5/16(旧), L23=4/16(旧), L24=3/32, L41=13/16。
- [x] 验证：python/test_curriculum_queue.py 38 过（+4 subtests），test_curriculum_pipeline、
  test_roof_panel 全过。新单测：饿死格保底、连续 12 轮无格间隔超 FLOOR_ROUNDS+1、
  老师会赢学生全败优先（L20>L17中难度>L24沉底）、load_teacher_rates 缺文件/两种格式。
  独立 state 副本 dry run 5 轮（预算 768/组 48）：round 3-4 内 12 个饿死格全部拿到数据
  （每格 1-4 组），此后每格按 ≤4 轮周期轮转；pool_l26/27/29（学生 0 且老师未知）持续
  高频，L21-24 按老师证据降权，L24（老师 3/32）只走保底不再吸预算。
- [x] 配方核实：TrainingPipeline.enqueue() 把 state log 标签 ∪ recipe 标签写进快照
  recipe（round_000002 快照 = 32 个 c1/c2 标签 + 170 个 dw4i 标签 = 202），
  _prepare_corpus 缺标签直接报错、不静默缩水 → qc 新组确实按普通流程进后台训练。
  L20 初始化档案 n20o1_fast_search 在 frames_v2（seed_2290000..，32 局）但不在任何
  recipe 标签里 → 新开 recipes/dw4i2.json（= dw4i.json + n20o1_fast_search），
  不覆盖 dw4i.json 历史语义。

## 问题

- allocate 在 plan 时写 last_round，控制器组间崩溃重启后本轮未跑的组不重排（见上）。
  要修需把 last_round 写挪到 run_round 组完成时，牵涉防遗忘语义，留给主控决定。
- dagger-beta 提升建议（**只分析，未实现，主控决定**）：pool/roof/L20 这类学生开局即崩的
  格，beta0 下学生轨迹只覆盖开局 few 十步，老师标签也只落在这些状态上——DAgger 恰恰
  要求标签覆盖学生自己访问的状态分布，beta0 在这点上是对的；但"学生访问的分布"塌缩成
  开局，中后盘状态永远没有标签，学生永远学不会中后盘。Ross et al. 2011 的 β-mixing
  （部分轨迹由老师执行）是标准解法：对全败格把 dagger-beta 提到 ~0.5，既扩大状态覆盖
  又保持学生分布成分。search_play.py 已支持 --dagger-beta 参数，改动点只在
  command_for 按"学生胜率 < 0.1 的格用 beta 0.5"分流——一行条件，等主控批准。
- 老师证据里 L22/L23 用的是旧 identity 老师（10-22，5/16、4/16），可能低估；L25/27-29
  无任何老师测量（按可教处理，采了才知道）。新面板出来后应更新 teacher_rates.json。

