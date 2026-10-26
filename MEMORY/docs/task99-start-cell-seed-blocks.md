# task99：示范起点格的未来种子块自动续期

> 主控 2026-10-26 写。执行者：台式机 Codex（plus 号，tmux `codex-v6`）。工作树 `/home/newbiexvwu/PvZ-search/.claude/worktrees/startseeds`（分支 `startseeds`，主控建）。每个任务最多 2 个游戏进程；**不碰 tmux `qc`、cq**。

## 为什么
起点格每格只登记了 256 个未来种子（`configs/human_demo_tasks.json` 的 `demo_start.future_seed_start/count`）。qc 每组用 24–48 个，几轮就用完了：L32 已经在 10-26 06:50 用完。deploy14 之后用完的格会被退休，不再让 qc 停机，但这些格本身是课程里最有价值的部分，不该因为种子用完就停采。现在已有 24 个起点格，GPT Pro 交回后还会更多。

## 要做的
1. **账本**：起点格的未来种子改为按需领块。每次 128 或 256 个，从一个专用池里领，记进加锁账本（可复用 task88 / task89 的账本机制）。
   - 新池与全部现有段不相交，用 `scripts/gate_seed_inventory.py` 机械核对；
   - `human_demo_tasks.json` 里原有的块作为第一段保留，旧数据的可比性不变。
2. **qc 侧**：`plan_round` 发现当前块不够时，先续领新块，领不到才退休。续领要写进 state 和日志。原有偏移语义不变，只是把种子表从一段扩成多段。
3. **测试**：
   - 续领之后种子与旧块不相交、与其他格不相交；
   - 重启后 state 能恢复多段种子表；
   - 已退休的格在续领后能重新激活（deploy14 已退休 L32）。
4. 写清上线步骤，与 task98 一起在组边界上线。结论写 `MEMORY/docs/start-cell-seed-blocks.md`，`git push origin startseeds`。

## 规则
- 只用绝对路径；不 `cd` 进别的工作树；`PVZ_NO_POPUP=1`。
- 禁止 `git add -f`；`.pt` `.npz` `.log` 不进 git；不写字面密钥；不手写打法规则或公式。
- 每完成一小步就提交、push，在“结果”里记一行；有疑问写进“问题”，按最佳判断继续。

## 结果

- 2026-10-26 打包根因已修复：首次平面出现在后续存档时，为此前无平面的行回填-1；平面值按存储dtype规范化，缓存/无缓存输入一致。新段逐行保存plane_of，复用前用npz头检查长度和精度，损坏旧段自动重建，不升PACK_VERSION、不改变抽样。实际ce3ce400标签24份存档在隔离目录重打包：84行、board/scalar plane_of均84，前12行为-1，缓存与无缓存payload逐字节相同；线上段与源存档SHA不变。证据见`start-cell-pack-check.json`。未来种子块续期与部署步骤仍在实施。

- 2026-10-26 第 1–3 步（账本 + qc 侧 + 测试）：起点格未来种子改为按需续领。专用池 `START_SEED_POOL=[200,000,000, 201,000,000)`（`pvz_promotion`；full scan 核对与全部门禁/示范/llmplay/120–140M 段不相交，字面量复核 31 命中全是 metrics 的 rss/io 字段），账本 `~/pvz-logs/start_cell_seed_ledger.json`（git 外、`fcntl` 锁、块只发一次、复用 task88 的 `allocate_seed_block`）；`plan_round` 在块不够时先领 256 个接到种子表尾部，领不到才退休，续领写进 state 的 `seed_blocks` 与 changes 日志；`reconcile_start_cells` 重启保留多段表（只有清单换块才重置）；`--restore-start-cell` 从 `retired_cells` 恢复已退休的起点格并保留偏移（与在跑格种子重叠时打 WARNING）。9 项新测试（含不相交、重启恢复、退休再激活、池耗尽退休、清单换块重置）+ 原有测试全过（`/home/newbiexvwu/.venvs/ml/bin/python3`）。第一版候选 [90M,91M) 因 task89 的 e2e 测试已用 90,000,000 起而废弃（见“问题”）。

- 2026-10-26 第 3 步（测试收尾）：8 个相关测试文件共 **123 passed, 2 skipped**（含续领 9 项）；`test_curriculum_queue.py` 单文件 63 passed。干跑（生产 state 副本、L32 `_hv2` 偏移 240）实测续领 `[200000000,200000256)`、24 局拿 `44300240..44300255 + 200000000..200000007`，预览账本在临时目录、生产账本未创建。
- 2026-10-26 第 4 步（上线步骤 + 结论）：结论写 `MEMORY/docs/start-cell-seed-blocks.md`，含池依据与两次盘点、多段种子表语义、重启/恢复规则、与 task98 一起在组边界上线的 6 步（无需新参数）、回滚注意，以及三个问题（90M 与 task89 e2e 冲突、8 位僵尸 id 误报、旧 L32 与 `_hv2` 共用块）。**未跑游戏进程、未动生产 state/账本/tmux qc/cq**。task99 四项全部完成。

- 2026-10-26 上线前核对（只读）：`merge-base(startseeds, cq)` = cq 的 HEAD `c84ad7d1`，startseeds 领先 692、落后 0，`merge-tree` rc=0 → 可 fast-forward。但发现 startseeds 缺 deploy14c 的 `391c8a64`（起点格以 `archive` 识别，否则原生阶梯格 `pool_l25_orig_train` 会被当起点格退休），已 cherry-pick 进 `08cc5c72`（124 passed）。另记两条部署注意：cq 工作树有约 130 个已暂存文件（ff 前须清理/备份），deploy15 的 `diff --quiet 0b8dcc26` pin 合并后必失败、需换 REV。全部写进 `start-cell-seed-blocks.md` 的“上线前核对”。

- 2026-10-26 测试补齐（任务书第 3 条的两个缺口）：新增“两格续领拿到的块也不相交”（跨格不相交）与“清单块落进续领池时启动即拒绝”（`load_demo_start_cells` 的 `ValueError` 硬闸，清单块不在账本里、撞了查不出来）。`test_curriculum_queue.py` 66 passed；八个相关文件合计 126 passed, 2 skipped。

- 2026-10-26 长跑测试：12 轮、每轮 48 局、每轮都重启（落盘 → `reconcile_start_cells`），种子表按需长到第三段、576 个种子零重复、全程不退休。单文件 67 passed，八个相关文件 127 passed。

- 2026-10-26 续领失败面收窄：`plan_round` 现在把 `SeedPoolExhausted` / `ValueError`（账本损坏、池不符）/ `OSError`（账本不可写）都当作“这一格领不到”→ 退休并记日志，不再让 qc 停在 plan 阶段（task98 的单点异常原则）。新增 `test_corrupt_ledger_retires_instead_of_crashing`；单文件 68 passed，八个相关文件 128 passed。

## 问题

## 主控追加（2026-10-26 11:55）：起点格打包丢 plane_of
第 18 轮训练失败两次，qc 因此停机（11:33）。原因是段 `ce3ce4006691a0e1`（标签 `c18demoroof48doomm200s1100_fast_search`，L48 起点在 1100，存档总共 1108 个决策）的 `board_plane_of` 只有 72 个，行数却是 84。训练读段时 broadcast 失败。
- 主控已热补 `train_rollout_policy.read_segments`（570f5e7d）：行数不符的段按“无平面”读。这只是止血。
- 请在 `pack_training_rows` 里找到起点格（尤其是离终局只差几步的起点）为什么会少记 plane id，从根上修好，并加测试：起点格的段，plane_of 长度必须等于行数。
- 修好后，在 ce3ce400 对应的标签上重新打包，验证长度一致。
