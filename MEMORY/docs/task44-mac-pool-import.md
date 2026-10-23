# task44：把 Mac 采的泳池轮次接进台式机语料，并在课程控制器里登记 Mac 那半

> 主控 2026-10-23 18:00（台式机时间）写。执行者：台式机 Codex（codex3）。工作树 `/home/newbiexvwu/PvZ-search/.claude/worktrees/macpool`（分支 `macpool`，从 pvz-env 拉出，主控已建好）。
> 同时最多 2 个你自己的游戏进程（本任务基本不需要）。不碰 tmux `qc` 与别的代理会话。

## 为什么（做了没用上）
- Mac 在 10-22 到 10-23 用 dw5i_s1 当采集学生，在泳池 p9k 族上采了 32 个标签：`p{21,22,23,24,26,27,28,29}w{1..4}m`，每个 48 局，共 1536 局。这些都已经上传到 HF `transfer/<label>.tgz`。脚本是 `artifacts/hard_mac/mac_dagger_p5a.sh`，只在 Mac 上。
- 台式机一直没拉。`scripts/import_task27_mac.py` 的标签正则不认 `p\d+w\d+m`，所以它们不在任何训练集里。
- Mac 从 10-23 17:45 起接着采第 5–8 轮（`mac_dagger_p5b.sh`，`p..w5m`…`p..w8m`），以后还会有。
- 泳池是学生最弱的区（dw5i 面板泳池 129/320），这批数据正好补在这里。

## 步骤
1. **读现有导入链**：`scripts/import_dw4_transfer.py`（`hf_environment`、`import_archive`）、`scripts/import_task27_mac.py`，以及它们写的 `mac_import.json` / journal。弄清三件事：
   - 归档落到哪：主检出 `artifacts/student/frames_v2/<seed>/<label>_fast_search/`；
   - 落地格式：npz（`frames.npz`），不是旧 JSONL；
   - 怎么防重复、怎么校验 SHA256。
2. **做成一个通用导入，不再按任务各写一份**：把"Mac 标签正则 + 已导入记录 + 下载校验解包"收成一个入口（如 `scripts/import_mac_labels.py`）。
   - 标签模式写成一张表：旧的 z/d/n 系列 + 新的 `p\d+w\d+m`。
   - 已导入的不重下，以 HF 的 lfs sha256 对账。
   - task23/27 的旧脚本若只剩被新入口替代的逻辑就删掉；删之前 `git grep` 确认引用。
   - 只读 `$HF_TOKEN` / `$PVZ_HF_REPO`，不写字面值。
3. **导入**：把现有 32 个 `p..w{1..4}m` 导进主检出的 `frames_v2`。
   - 逐标签记：局数、帧档案数、行数（按 `pvz_training_pack` 的口径）、SHA256；
   - 结果写 `MEMORY/docs/task44-mac-import.json`（KB 级，进 git）；
   - 磁盘余量先看 `df -h /mnt/c`，低于 15 GB 就停下写"问题"。
4. **登记给训练**：
   - task36（qoder-mix，分支 `dw5`）在定 dw6i 的数据配比。在 `MEMORY/docs/task36-dw5i-mix.md` 的"问题"节里**追加一段**，写清新增标签、行数、所属区、采集学生（dw5i_s1），供它决定是否纳入。**只追加，不改别的。** 写进你自己的分支；主控合并时会转过去。
   - `recipes/` 里不要改现有配方，那是历史证据。
5. **持续导入**：Mac 每采完一个标签就上传一个，新入口要能反复跑、只补新的。
   - 在 `MEMORY/notes/desktop.md` 写一行用法；
   - 在本文"结果"写下一次该什么时候跑。主控会把它挂进巡检。
6. 测试：`PVZ_RESOURCE_DIR=/home/newbiexvwu/.cache/pvz-research-resources python -m pytest python -q` 全过；新入口要有单测，用假的 HF 列表加两个小 tgz。

## 规则
- 只用绝对路径；不 `cd` 进别的工作树；不跑采集、不训练。
- 禁止 `git add -f`；`.pt` `.npz` `.tgz` `.log` 不进 git；不写字面密钥；不手写规则。
- 每完成一小步就提交、`git push origin macpool`，在"结果"记一行；疑问写"问题"，按最佳判断继续。

## 结果

- 2026-10-23，步骤1：读完 task23/task27 导入器和两份历史报告及 journal；主检出以 `seed_*/<label>_fast_search/{frames.npz,meta.json}` 落地，HF LFS SHA256 对账后复用，独占写入且同名异内容拒绝覆盖；旧 z 轮次的冻结选择继续保留。`df -h /mnt/c` 可用约 28 GB（高于 15 GB）。下一步统一入口并迁移调用方，不动历史配方。

- 2026-10-23，步骤2：统一入口 `/home/newbiexvwu/PvZ-search/.claude/worktrees/macpool/scripts/import_mac_labels.py` 已实现 z/d/n/p 标签表、历史账本迁移、LFS SHA256 校验、原子独占写入、逐标签 journal、冻结 z 选择与 15 GiB 磁盘余量门禁；调用方已迁移，旧 task23/task27 导入脚本删除。假 HF + 小 tgz 单测 11 过，调用方编译检查通过。

- 2026-10-23，步骤6：`PVZ_RESOURCE_DIR=/home/newbiexvwu/.cache/pvz-research-resources /home/newbiexvwu/.venvs/ml/bin/python3 -m pytest /home/newbiexvwu/PvZ-search/.claude/worktrees/macpool/python -q` 全过：333 passed / 111 subtests passed，62.39 秒，无跳过。使用与本分支 `src/`、`CMakeLists.txt` 无差异的 `f2ae755a` deploy2 二进制副本（仅复制到本工作树 ignored build，未修改 deploy2）。

- 2026-10-23，步骤5：已在 `MEMORY/notes/desktop.md` 登记通用入口的绝对路径用法、环境变量、共享账本位置与磁盘门禁。下一次运行：Mac 第 5–8 轮每上传完一个标签后的下一次主控巡检即可运行同一入口；不新建定时任务、不改 qc。当前 HF 观察到前四轮外还有 p21/p22/p23/p24 的第 5 轮上传，前四轮导入完成后用同一入口补入并验证复用。

- 2026-10-23，步骤3：32 个 `p{21,22,23,24,26,27,28,29}w{1..4}m` 全部导入主检出，共 1536 局 / 1536 个 npz 档案 / 1,724,858 行（wait_keep=1，采样与训练/验证切分前）；新增 3072 个 frame/meta 文件。逐标签局数、档案数、行数及 HF 包 SHA256 写入 `MEMORY/docs/task44-mac-import.json`（11,247 字节）。独立重扫全部元数据、NPZ rows 和文件 SHA256，与原始帧计数及经 LFS 校验的包账本全一致，1536 局均为 dw5i_s1、对应泳池关。C 盘仍约 28 GB。

- 2026-10-23，步骤5实跑：同一入口第二次选取当前 HF 的 37 个泳池标签，前四轮 32 包全部 SHA 对账复用、零重下 / 零新增；仅补入 `p21w5m,p22w5m,p23w5m,p24w5m,p26w5m`，240 局 / 240 档案 / 258,196 行、新增 480 文件。新增包也逐文件、元数据、NPZ rows 独立核验，证据补进同一 JSON。累计 37 标签 / 1776 局 / 1,983,054 行。下一次主控巡检：关注 p27/p28/p29 的第 5 轮及后续第 6–8 轮，上传完就重跑笔记中的统一入口。

- 2026-10-23，步骤4：仅在本分支 `MEMORY/docs/task36-dw5i-mix.md` 的“问题”节追加 Mac 数据段（字节前缀核对原文完全不变），登记前四轮与已上传第五轮的标签、局数、行数、p9k 所属区、dw5i_s1 采集学生及证据位置，交主控转给 task36 决定配比；现有 recipes 全部未改。

## 问题

- task23/task27 的历史 study-plan JSON 仍记录原脚本名及其 SHA256，这是冻结证据，保留原样；可执行调用方全部迁移到统一入口，旧脚本已删。统一入口只读取导出的 `HF_TOKEN` / `PVZ_HF_REPO`，不再自行读取 shell 配置。
