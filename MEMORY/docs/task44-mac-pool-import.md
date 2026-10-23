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

## 问题
