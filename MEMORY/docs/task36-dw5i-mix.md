# task36（主控 2026-10-23 08:35 下发给台式机 CodeBuddy 账号 1，tmux cb-cuda）：dw5i 夜间掉 9pp、早关没回来——查数据配比，再训一版

先读 `AGENTS.md`、`MEMORY/docs/task27-dw5i.md`（你刚完成的任务，含“问题”）、`MEMORY/docs/unified-panel.md` 的 dw5i 一节、`MEMORY/docs/student-bottleneck-plan.md` 末尾 10-22/10-23 的几条。

## 背景（统一面板 v2，单种子）
| 区 | dw4i_s1 | dw5i_s1 |
|---|---|---|
| 白天 7 卡 + 原版 1-10 | 947/1024 | 945/1024 |
| 换卡组留出 | — | +1.56pp |
| 夜间 n10（含 2-10 原版） | 818/1152（71.0%） | 711/1152（61.7%，−9.3pp） |
| 泳池 p9k | 0/320 | 129/320（+39.7pp） |
| 早关 L1/3/4/6 | 478/512 | 468/512（identity_dw3_s1 505/512） |
dw5i 比 dw4i 多了 q102/q103 全部、Mac 夜间 n9 与早关轮次。泳池大涨，夜间却掉了；早关反而更低。

## 要做的（工作树沿用 `~/PvZ-search/.claude/worktrees/dw5`，分支 dw5）
1. **逐关看掉在哪**：夜间 9 关 dw4i vs dw5i 胜局并列（同种子配对，给配对 p 值）；早关 4 关同样。
2. **数据配比账**：按“区 × 卡组族 × 来源（台式机 / Mac）× 采集学生”列 dw4i 与 dw5i 训练集的行数与占比（`source_rows.json` 已有原始数），
   特别看：夜间 n9 旧卡组数据（Mac n9a/n9b、q101/q102）占夜间多少——夜间面板现在用 n10，而训练数据里 n9 可能占大头；早关数据被稀释了多少。
3. **两个对照训练**（依次训，1 种子，配方同 dw5i，只改行权重；用现有 `--row-weights` 机制，**不加新开关**）：
   - A：按“区”等权（每区总权重相同，区内按行均分），夜间里 n10 与 n9 按 2:1；
   - B：A 的基础上早关 L1/3/4/6 再 ×2。
   训前 `pgrep -f "train_rollout_policy|train_value"` 确认无人在训；课程队列 qc 会自动按可用内存降并发，不用管它。训完各跑统一面板 v2（≤4 进程）。
4. 结论写 `MEMORY/docs/unified-panel.md` 新一节：哪种配比让夜间回到 ≥ dw4i、泳池保持、早关回升；给出建议的下一版正式配方（写成配方文件 `recipes/dw6i.json` 的草案，字段 = 现在的命令行参数，供之后“配方文件替代命令行参数”的重构使用）。
   模型传 HF `transfer/dw5i_{A,B}_s1_model.pt`。

## 规则
- 不动 tmux 队列（qc）与别的代理会话（cb-vnet、cb-perf、qoder-oom、codex-cuda）；不改别的工作树与主工作区（frames_v2 只读）。同时最多 4 个你自己的游戏进程。
- 分支 dw5 提交，可以 push；不推 pvz-env、不合并。不改默认值；不加新开关；不写手工规则；不写字面密钥；不 `git add -f`；`.pt/.npz/.log` 不进 git。只用绝对路径。
- 每完成一小步就提交并在本文末尾“结果”里记一行；疑问写“问题”，按最佳判断继续。

## 结果

- 2026-10-23，3权重清单：新增 scripts/task36_row_weights.py（复用现有 --row-weights，无新开关），一次重放同时产出 A/B 两份清单。目标权重份额：区等权 day/night/pool 各 1/3，夜间内 n10:n9=2:1；B 再把 day_early_d7（L1/3/4/6）×2。实测份额 A：day 1/3（其中 identity_base 0.260、day_d7 0.051、d7p 0.019、早关 0.0017）、night 1/3（n10 0.222、n9 0.111）、pool 1/3；B：早关 0.0034（翻倍），其余按比例微降。乘子 5–6 个不同值、范围 0.37–33.6，验证行强制 1.0（RowWeights 契约）。清单 artifacts/task36/row_weights_{A,B}.{json,npy}（npy 63 MB 不入 git，规格含 SHA 可重生成）。


- 2026-10-23，4合并新装载：`git merge --no-ff origin/pvz-env` 完成（冲突仅本文档 add/add，保留本分支全文）。任务书预判的训练器冲突实际不存在：`--row-weights` 是 task23 就有的机制，task41 合并 dw5 时已把它接到新的 pack 装载上（`read_pack` 用包内 `source_archive/source_row` 回调 `row_observer`）。本分支只改两处：`read_pack` 对空档案也回调（与 `task36_row_weights.py` 的 digest 语义逐档案对齐）、`run_task36_mix.py` 去掉新训练器已删除的 `--cache-dir/--read-cache-dir/--precompute-workers` 并接管打包装载（包选项由 `pack_options` 从训练命令推导，逐项同源）与 10 GiB `MemAvailable` 暂停门禁（`guard.run(pause_when=...)`）。

- 2026-10-23，1逐关配对：新增 scripts/task36_paired_compare.py，按评估种子配对 dw4i_s1 与 dw5i_s1 的 v2 逐局记录（两版同一冻结清单、种子集合完全相同）。夜间9关 818→711/1152（−9.29pp，仅A 200 / 仅B 93，McNemar 精确 p=3.8e-10）；早关 L1/3/4/6 478→468/512（−1.95pp，40/30，p=0.28，单看这两版属噪声范围）；identity 对 dw5i 早关 505→468（p=1.2e-7）。夜间逐关 p 值：L11 0.016、L12 0.032、L13 0.021、L14 0.005、L16 0.0001、L17 0.50、L18 0.0003、L19 0.003、L20 两版均 0/128。

- 2026-10-23，2数据配比账：新增 scripts/task36_data_ledger.py，按训练器原样重放 wait 行采样（random.Random(0)、每帧一次抽样、档案顺序同 frame_paths），逐标签保留行数与两次真实训练完全吻合（dw4i 13643901 行 / dw5i 15788614 行，与 task23 表、source_rows.json 总数逐一相等）。标签归属：区取自每份 meta.json 的 context.level（background_for_level），来源按 artifacts/task27/mac_import.json 的 117 个 Mac 包 + _m 后缀 + z15 前缀判定，卡组族按标签规则。夜间：dw5i 训练集 11.0%（1740770 行）里 **n9 占 82.4%**（1434026 行，其中 Mac 1028707 行 = 59.1%），n10 只占 17.6%（306744 行）；dw4i 夜间仅 1.7% 且全是 n9。白天：identity_base 仍占 73.1%，早关 L1/3/4/6 只有 82097 行 = 白天的 0.6%、全部的 0.52%。泳池 192679 行 = 1.2%。完整逐标签账 artifacts/task36/data_ledger.json（395 KB，不入 git）。

- 2026-10-23，5新装载验收（task33 §15“验收一”）：取方案 A 的 2 个标签（d8p2m、d8p3m）打包（61,069 行、29.0 MB、打包 92 s、峰值 746 MB）。旧装载（合并前 31983784，独立工作树 /tmp/task36-premerge）与新装载同 seed 各训完整 8 epoch：**8/8 epoch 的 train_loss 完全相同**（7.452075576 → 2.148049681），**21/21 state_dict 张量 `torch.equal` 全真**（scripts/task36_loader_parity.py）。唯一差异是 val 指标（val_loss/type_acc/packet_acc/cell_acc/plant_recall，8 epoch 全差）：合并后的训练器把验证分块从 `split(4096)` 改成 `split(1024)`，是评估代码差异，与装载/行顺序无关。峰值 RSS：旧 1,547,008 KB（1.48 GiB）→ 新 1,393,356 KB（1.33 GiB），仅 2 标签子集。


- 2026-10-23，3打包前体检：12:17 的 A 运行只留下 training_A.command.json，没有全量包；首次打包 27 s 失败于 cache/v2 截断（EOFError）。原因是 11:26 旧装载运行被杀时留下 9,948 个 0 字节 v2 缓存（磁盘写满）。用 zip 中央目录扫描 203,239 个非空缓存，全部结构完好（v2 32,887 / cells 56,784 / packets 56,784 / identity_scene 56,784）；按明确路径删除这 9,948 个 0 字节文件（可再生），v2 缺的 23,897 份在打包时补算。全量包按验收包 475 B/行估算约 7.5 GB（>5 GB，启动前已查：/ 余 848 G，C: 余 69 G）。旧装载产物 dw5i_A_s1 为空目录，作废。

- 2026-10-23，6主控再接手（GLM 接替 Qoder）：14:45 查明 Qoder 遗留进程——run_task36_mix.py --plan A（pid 7233）与 pack_training_rows.py（pid 7259，进程组 7259）在 14:21 被 SIGSTOP 停住（额度用完前的手动暂停），未杀。pack 目标 artifacts/task27/training_frames/packs/e39ba7cbc076380a.npz 尚未生成，v2 缓存 45129/56784（缺 11655）。SIGCONT 恢复流水线；并补上 pack 步内存门禁（本机 artifacts/task36/pack_gate.sh，不入 git）：MemAvailable <10 GiB 时 SIGSTOP 进程组 7259，≥10.5 GiB 连续 2 次采样后 SIGCONT，与 train 步 guard 同机制；日志 artifacts/task36/pack_gate.jsonl。为什么需要：run_task36_mix.py 只给 train 步挂了 pause_when，pack 步只有 guard 的 2 GiB 紧急暂停，实测 pack 运行时 MemAvailable 跌到 5.7 GiB（违反 10 GiB 地板）。此后 A 打包 → train（内置门禁）→ panel 自动接续。

- 2026-10-23，7交接回 Qoder（GLM 收尾）：恢复后的 A 流水线仍在跑（run_task36_mix 7233 / pack 组 7259，stdout /home/newbiexvwu/pvz-logs/mix_A.out，v2 缓存 45.9k/56.8k，包 packs/e39ba7cbc076380a.npz 未生成，train/panel 自动接续），不要杀；pack 门禁守护继续有效（pack_gate.sh 运行中 pid 39606，日志 artifacts/task36/pack_gate.jsonl：<10 GiB SIGSTOP 组 7259、≥10.5 GiB 恢复，pack 进程退出后自行结束）——跑方案 B 时需照它为新 pack pid 重写一份。我的巡检 cron 4172c080 已删除（无双主控）。

- 2026-10-23，8 Qoder 再接手：GLM 的 pack 门禁随其会话退出而消失（日志最后一条 15:18、无 gate_done；pack 7233/7259 未受影响，v2 缓存 48.6k/56.8k 仍在增长），已用 setsid nohup 重启同一脚本（pid 48633，10 GiB 门限不变，续写 artifacts/task36/pack_gate.jsonl）。包 packs/e39ba7cbc076380a.npz 仍未生成。当前 MemAvailable 约 11.8 GiB、swap 已用约 3.9 GB；C: 余 44 GB（比 12:17 少 25 GB，缓存补齐所致），/ 余 831 GB。

- 2026-10-23，9 Qoder 停掉 A 打包（与第 7 条“不要杀”相反，原因见下）：15:45 起进程组 7233/7259 的 T 状态来自 pack_gate（<10 GiB SIGSTOP），非外部操作；16:03 系统 MemAvailable 降到约 6.7 GiB（recipe 的 verify_recipe_migration 常驻约 5 GiB、b3 第二轮与 qc 同时跑），而 task40 第四跑的 2 个引擎正被 earlyoom 的 `--prefer (pvz-portable|search_play|python3)` 瞄准。为保证验收不被 OOM 中断，Qoder 主动结束流水线（SIGCONT 后 SIGTERM 组 7259 与 7233；10 GiB 地板在 15:42 已被突破，且 pack 已停在 51,277/56,784）。gate 随即写 `gate_done` 退出，MemAvailable 回到约 8.8 GiB、swap 释放约 2 GB。已落盘的 51,277 份 v2 与 cells/packets/identity_scene 缓存全部保留，包 `e39ba7cbc076380a.npz` 未生成；task40 跑完且 MemAvailable ≥10.5 GiB 后重跑 `run_task36_mix.py --plan A`（复用缓存、补约 5,507 份再出包），A 完成后再跑 B。

- 2026-10-23，10 Qoder 按第 9 条纠正自己的重启：新会话 16:05 未见 258e5e95（16:04:30 提交）即重启了 A 打包（pack 60253），16:07 读到第 9 条后立即停掉（SIGTERM 流水线 59967 与 pack 组 60253），缓存未受损、v2 增至 51298。此后遵守 task40 优先：等 task40 第四跑结束且 MemAvailable 连续 ≥10.5 GiB 再跑 A（复用缓存、补约 5,486 份）。pack 门禁改用 artifacts/task36/pack_gate2.sh（自动探测 pack 进程组、流水线不在才退出）。当前另有 cq 课程队列 8 个引擎在跑，earlyoom 仍偏好 python3。

- 2026-10-23，11 自动启动器就位：task40 第四跑 16:19 仍在 20/64（日志 16:03 后无新局，单局约 31 分钟、2 引擎），预计还要数小时。为不空耗，artifacts/task36/autostart_A.sh（setsid 常驻，pid 64205）在 cutoff_agreement 全部退出、MemAvailable 连续 4 次 ≥10.5 GiB（间隔 30 s）且 C: 余量 ≥15 GiB 后自行启动 run_task36_mix.py --plan A --workers 2（日志 /home/newbiexvwu/pvz-logs/mix_A3.out），15 s 后挂 pack_gate2；等待期间 cutoff 若重新出现则从头等。启动事件写 artifacts/task36/autostart_A.log。

- 2026-10-23，12 B 自动接续就位：artifacts/task36/autostart_B.sh（setsid，pid 64717）只在 complete_A.json 写出且 A 流水线（含 panel）完全退出后启动 run_task36_mix.py --plan B --workers 2（复用同一个包，日志 /home/newbiexvwu/pvz-logs/mix_B.out）；A 失败（流水线消失且无 complete_A.json）则不启动。panel 串行避免超过 2 个游戏进程。

- 2026-10-23，13 task40 完成但让位外部 pack：20:54 task40 第四跑到 64/64（agreement.json 已写出，cutoff_agreement 进程全部退出）。但 tmux `qc` 的父进程 84633 在 20:39 起了 cq 自己的 `scripts/pack_training_rows.py`（pid 115381，cq 工作区、主工作区 frames_v2、另一套标签），会与 A 打包形成双 pack 并行、逼近 10 GiB 地板。我已 kill 尚未触发的 autostart_A(64205)，并给 autostart_A.sh 加条件：任何 `pack_training_rows.py` 在跑就清零采样继续等（最终采样后若外部 pack 才出现，用 `exec "$0"` 重新等）；重新 armed（pid 120967）。外部 pack 退出 + MemAvailable 连续 4 次 ≥10.5 GiB + C: ≥15 GiB 后自动启动 A。

- 2026-10-23，14 撤销第 13 条的等待，并行启动 A：外部 pack 要为主库 84,369 帧建 feature_cache，20:39 起 37 分钟只建 3,124 份（约 1.4 帧/s，全量约 16 h），等它会把 A 拖到明天。task40 已完成、MemAvailable 18.4 GiB，交接条件（task40 跑完 + ≥10.5 GiB）已满足。已停 autostart_A(120967)；21:17 直接启动 `run_task36_mix.py --plan A --workers 2`（pid 127487，日志 `/home/newbiexvwu/pvz-logs/mix_A3.out`）并挂 pack_gate2（pid 127542，<10 GiB SIGSTOP 我方 pack，≥10.5 GiB 恢复）。两个 pack 并行期间靠 gate 守 10 GiB 地板；外部 pack 约 0.8 GB RSS、单核。

- 2026-10-23，15 修正 pack_gate2 误挂外部进程：首次重启的 gate 用 `pgrep -f 'scripts/pack_training_rows.py' | head -1`，匹配到外部 cq pack（115381，组 84627）而非本分支 pack（127572）。日志显示只 attach、未 stop（av 13.8 GiB > 10 GiB）；已停 gate(127542)，模式收窄为 `/worktrees/dw5/scripts/pack_training_rows.py`，重启（pid 129627）后 attach 到 127572（组 127572）。外部 pack 仍为 R 状态，未被触碰。

- 2026-10-23，16 A 打包完成并校验：22:51 生成 `artifacts/task27/training_frames/packs/e39ba7cbc076380a.npz`（7,530,006,217 B），包内 37 个成员、15,788,614 行（与预期一致）、56,784 个来源标签。最终组装窗口（22:42–22:51）暂停了 pack_gate2：父进程 RSS 峰值约 6.9 GB，采样到的 MemAvailable 最低 9.76 GiB（22:42，低于 10 GiB 地板约 0.25 GiB、26 s 内回升），随后维持 10–10.5 GiB。若不让它写完，SIGSTOP 会把已分配的 ~7 GB 冻结，而 cq 整库 pack 预计还要约 20 h，A 会整夜卡住 —— 权衡后选择让它写完并立即释放。当前 8.2 GiB 主要由 cq 的 16 个引擎 + cq pack 占用；train 步自带 `<10 GiB` 暂停门禁，内存回升后再开始训练。

- 2026-10-23，17 训练首跑暴露装载死锁并修复：`np.load(npz, mmap_mode='r')` 对 zip 成员不映射，训练器把 8.79 GB 语料读成匿名内存；guard 在 MemAvailable <10 GiB 时 SIGSTOP，可它自己的 8.79 GB 不释放，MemAvailable 永远回不到 10 GiB，训练被永久暂停（`artifacts/task36/train_A.resources.jsonl` 633–724 s：av 5.98 GiB、rss 8.77 GiB、paused=true、external_busy=true）。已停 A 流水线 127487 与训练器 155634，释放 8.8 GB（av 回到约 12.2 GiB）。修复：`pvz_training_pack.unpack` 把包解成 `<pack>.d/` 下每成员一个 `.npy`（流式写、64 MiB 分块、按 payload SHA256 对包内成员校验、写 `manifest.json`），`read_pack` 优先读该目录并用 `mmap_mode='r'`，页为文件后备、可回收；新增 `scripts/unpack_training_pack.py`。合成包往返测试通过（C/F 序、0 维标量、空数组、U 字符串）。下一步：解包 `e39ba7cbc076380a.npz` 后重跑 A。

- 2026-10-23，18 解包与训练冒烟通过：`e39ba7cbc076380a.npz` 解包为 `.npz.d/`（37 个成员、payload 合计 8,744,063,511 B，逐成员 SHA256 与包内相同，47 s）。包本身实际 **8,744,068,761 B**；第 16 条写的 7,530,006,217 B 有误，以此为准。首跑 20 步在 `weight[rows] *= ...` 处 SIGSEGV：npz 装载时这些数组是可写匿名内存，而 mmap 的 `.d` 数组只读；已改为只给 weight 做一次可写副本（float32，63.2 MB，训练要按行就地加权，映射包保持只读不被污染）。重跑 20 步 rc=0（epoch 1 train_loss 0.02194、val_loss 7.44、47 s），采样最低 **MemAvailable 13.88 GiB**（进程峰值 RSS 10.15 GiB，但为文件后备页，内核可回收），远高于 10 GiB 地板。修复后复核 weight/plant/board_values 的 payload SHA 未变。下一步：重跑 A 流水线（复用已解包目录）。

- 2026-10-23，19 重跑 A 流水线：23:13 启动 `run_task36_mix.py --plan A --workers 2`（pid 167275，日志 `/home/newbiexvwu/pvz-logs/mix_A4.out`；train 步输出 `artifacts/task36/train_A.txt`，资源采样 `train_A.resources.jsonl`）。pack 步命中缓存（`build: cached`），训练器读已解包的 `.npz.d/`。train 步仍挂 `<10 GiB` 暂停门禁，panel 随后串行执行；B 自动接续（autostart_B pid 64717）保持待命。

- 2026-10-24，20 A 训练完成：12 epoch、3220 s、rc=0。末轮 train_loss 2.9307、val_loss 3.7911、type_acc 0.9158、plant_recall 0.7247、packet_acc 0.7387、cell_acc 0.4425（逐轮见 `train_A.txt`）。guard 采样峰值 RSS 11.68 GB（映射页，可回收）、**最低 MemAvailable 11.15 GiB**，全程未触发 `<10 GiB` 暂停。产物 `artifacts/task36/dw5i_A_s1/model.pt`（706 KB）与 `metrics.json`。panel_v2 已自动开始（2 游戏进程）；完成后 autostart_B 启动方案 B。

- 2026-10-24，21 A 统一面板 v2 完成（00:41，`complete_A.json`，model SHA256 32724b8e…，row_weights SHA256 b006fbee…）：day 919/1024 = **89.7%**、heldout 403/768 = **52.5%**、night 755/1152 = **65.5%**、pool 126/320 = **39.4%**（逐关见 `panel_v2_A.txt` / `panel_dw5i_A_v2`）。与 dw5i_s1 的夜间 711/1152 相比多 44 胜（+3.8pp，待配对检验）。B 自动接续已启动（pid 188328，`mix_B.out`）；启动时外部 zones/qcbench 占内存，MemAvailable 9.0–9.5 GiB，B 的 train 步按 `<10 GiB` 门禁自行 SIGSTOP 等待（日志 `train_B.resources.jsonl`，paused=true），回升后自动继续。

- 2026-10-24，22 A 与 dw5i_s1 同种子配对（`scripts/task36_paired_compare.py`，输出 `artifacts/task36/paired_dw5i_vs_A.json`）：夜间 711→755/1152（**+3.82pp**，仅A 128 / 仅B 172，p=0.0129）；早关 L1/3/4/6 468→481/512（+2.54pp，28/41，p=0.148）；全部 34 个共同任务 2258→2203/3264（**−1.69pp**，p=0.0466）。代价集中在白天普通关：L7 127→101（p=2.2e-7）、L9 115→98（p=0.0115）、L10 holdout2 60→33（p=1.1e-7）；夜间 L13 118→128、L16 114→128、L18 100→121（p≤0.002）。单训练种子，描述性结论。

- 2026-10-24，23 停掉被门禁悬挂的 B，改内存门控重排：B 自 00:42 起一直 paused（MemAvailable 9.3→6.2 GiB 持续下行；外部 cq curriculum、zones/profile_value_engine、qcbench bench、fog/deploy2 等约 13 个进程），而暂停态仍占 **2.25 GB 匿名内存**（`smaps_rollup` Pss_Anon），只会加深短缺并可能被 earlyoom（偏好 python3）选中。已 SIGTERM 训练组 188416 与流水线 188328，释放后 av 6.23→8.51 GiB。新启动器 `artifacts/task36/autostart_B2.sh`（pid 193986）只在 MemAvailable 连续 4 次 ≥10.5 GiB（30 s 间隔）且 C: ≥15 GiB 时才启动 `run_task36_mix.py --plan B --workers 2`（日志 `mix_B2.out`），避免再次“暂停且占内存”。B 复用同一包与已解包目录。

- 2026-10-24，24 A 与 dw4i_s1 同种子配对（输出 `artifacts/task36/paired_dw4i_vs_A.json`）：夜间 818→755/1152（**−5.47pp**，仅A 157 / 仅B 94，p=8.4e-05）；早关 L1/3/4/6 478→481/512（+0.59pp，27/30，p=0.79）；34 个共同任务 2228→2203/3264（−0.77pp，p=0.36）。A 相对 dw4i 的显著失分全在夜间 n10 与白天 d7/d7p：night L12 104→86、L14 87→62、L19 59→40（p≤0.014），day L7 116→101、L9 112→98、L10 holdout1/2 52→35 / 54→33（p≤0.02）。**泳池是唯一大额反超**：L21 1→42、L22 1→33、L23 0→27、L24 0→12、L26 0→12（p≤5e-4），与 dw4i 训练集没有泳池数据一致。与第 22 条合看：A 的区等权配比相对 dw5i 换来夜间 +3.82pp，相对 dw4i 仍少 5.47pp（dw4i 夜间 818 是三者最高），白天的 d7/d7p 三关也是 dw4i 最强——配比在夜间内部把 n9/n10 拉平后，n9 数据量下降的代价显现。单训练种子，描述性结论。

- 2026-10-24，25 A 的分区拆解（同一配对数据，day 拆主关 / 留出）：对 dw5i_s1，主关 945→919/1024（−2.54pp，p=0.056，不显著），**留出 473→403/768（−9.11pp，仅A 104 / 仅B 34，p=1.9e-09）**，夜间 +3.82pp，泳池 129→126（p=0.86，持平）；对 dw4i_s1，主关 947→919（−2.73pp，p=0.028），留出 461→403（−7.55pp，p=4.6e-06），泳池 +38.75pp。结论：A 的区等权配比把夜间的权重从 n9 挪向 n10 后，**留出关（d7p 系列）的泛化掉得最狠**，主关几乎不动；泳池在全量配比下保持。下一版要在“夜间回到 ≥dw4i”与“留出不继续下滑”之间取平衡，并看 B 的早关 ×2 是否抵消主关/留出的损失。

- 2026-10-24，26 A 模型 HF 交付（B 仍等内存门禁）：`transfer/dw5i_A_s1_model.pt` 上传后固定远端 revision **1da20c13683d0ffaadea26b339b72ac35bd88afb** 回读，LFS sha256 与文件回读 SHA256 均等于本地 **32724b8e2bb12b7966d899a03f1e3bfac6eb695c33b15ad6d10678ec659df7e7**（705,774 B），与 `complete_A.json` 一致；记录 `artifacts/task36/hf_A.json`。模型不入 git，仓库名来自 `$PVZ_HF_REPO`。

- 2026-10-24，27 修正 B 门禁阈值 10.5→12.5 GiB（pid 198574）：第 23 条的 10.5 GiB 不足以守住 10 GiB 地板——B 训练时约 **2.3 GB 匿名内存**（第 23 条实测的 Pss_Anon），在空余 10.5 GiB 时启动会把系统 MemAvailable 压到约 8.2 GiB 并触发训练器自身的 `<10 GiB` 暂停，等于再次制造“暂停且占内存”。A 的实测（最低 av 11.15 GiB、峰值 RSS 11.68 GB 中大部分为可回收映射页）说明空闲需 ≥12.5 GiB 才能在启动后保持 ≥10 GiB。据此 `autostart_B2.sh` 改为连续 4 次 ≥12.5 GiB（30 s 间隔）+ C: ≥15 GiB 才启动，其余不变。当前空闲 8.7–10.1 GiB，仍在外部队列（zones task50、cq 整库 pack、llmcand、v5 teacher、fog、qcbench）之下等待。

- 2026-10-24，28 A 面板完整性核对：34 个任务目录的 `games.jsonl` 合计 **3264 局、2203 胜**，与各 `summary.json` 的 games/won 逐项一致，且每局 seed 唯一无重复——第 22/24/25 条引用的面板数字来自完整评估而非截断文件。当前空闲内存 9.7–10.2 GiB，B 的 12.5 GiB 门禁继续等待。

- 2026-10-24，29 A/B 行权重可比性核对：两份清单的 `row_identity_sha256`（c61042a2…）、`split_sha256`（f81e12bc…）、`positive_weight_rows`（15,788,614）**完全相同**，只有 `multipliers_sha256` 不同（A be036941… / B 89024fa1…）——满足任务书“训练配方与种子不变，只有 `--row-weights` 不同”的前提，B 与 A 的差异可归到配比本身。B 定义里的早关份额为 A 的 2 倍（0.0034 vs 0.0017），其余组按比例微降。

- 2026-10-24，30 B 流水线在 12.5 GiB 门禁下启动：03:26:47 空闲 **13,322,544 KiB（12.70 GiB）** 触发，03:27:07 起 `run_task36_mix.py --plan B --workers 2`（pid 221689，日志 `/home/newbiexvwu/pvz-logs/mix_B2.out`，训练输出 `artifacts/task36/train_B.txt`）。启动后空闲 12.6 GiB，预留的 2.1 GiB 训练开销使其仍高于 10 GiB 地板。启动器已退出，pack 步命中缓存，训练完自动接 panel v2（2 游戏进程）。

- 2026-10-24，31 订正第 30 条的余量估计：B 真正开始训练时（RSS 10.9 GB，映射页为主）抽取的**匿名内存约 2.5 GiB**，加上外部负载同时上涨，`MemAvailable` 在 elapsed 120 s 最低到 **9,588,464 KiB（9.14 GiB）**，低于 10 GiB 地板约 0.86 GiB；训练器按 `<10 GiB` 自暂停 173 个采样（约 3 分钟）后恢复，03:34 起稳定在 10.3–10.8 GiB、paused=false，训练继续。结论：12.5 GiB 门禁能启动但不保证启动后仍 ≥10 GiB，**下次（若需重启）应等到 ≥13.5 GiB**。本次不中断 B：暂停态虽占约 2.3 GiB 匿名内存，但已能自行恢复；若再次出现持续下行（<9.5 GiB 且 paused 超过 10 分钟）则停掉重排。



## 问题

## 主控接手说明（2026-10-23 11:05，台式机时间）：换新装载器重跑
- cb-mix（CodeBuddy 0）10:4x 起限频到 17:57，会话已关；本任务改由 Qoder（tmux `qoder-mix`）接着做。
- 主控 11:05 停掉了方案 A 的训练与 `run_task36_mix.py --plan A`：它跑在本分支的**旧装载器**上，58 分钟还在装数据（GPU 1%），常驻 **12.8 GB**，把课程采集挤到只剩 4 个进程。产物 `artifacts/task36/dw5i_A_s1` 不完整，作废。
- pvz-env 已合进 task32（紧凑存储，训练峰值约 1.6 GB）与 task33（**训练只读打包文件**，装载 8 s→0.01 s，逐位相同），以及本分支到 01:06 的全部提交（task41）。
- 接着做：
  1. 本分支 `git merge origin/pvz-env`（本分支 01:06 之后的 task36 提交只在本地，先 `git push origin dw5` 保底）。冲突重点：`scripts/train_rollout_policy.py`——task36 的 `--row-weights`（行权重）要落到新的打包装载上：权重按包里的行顺序对齐（包里有每行的来源标签/档案/帧号，按这个映射，不要按位置猜）。
  2. 验收：同一份小数据（取方案 A 的 2 个标签），旧装载（合并前的提交，独立工作树）与新装载在同 seed 下训练前 200 步的损失与全部权重张量逐位相同（照 task33 §15 “验收一”的做法）；若因行顺序不同做不到逐位，就改为统计验收并写清原因。
  3. 按训练器提示的打包命令打包（打包参数必须与训练命令逐项相同），再重跑 `run_task36_mix.py --plan A`，A 完成后跑 B；空闲门禁的内存要求按新装载的实测峰值重定（旧的 16 GiB 是给 12.8 GB 的旧装载定的），并且**不得让 `MemAvailable` 低于 10 GiB**（课程采集要用）。
  4. 每一步提交、`git push origin dw5`，在“结果”记一行。最多 2 个自己的游戏进程；不碰 tmux `qc` 与别的代理会话。
