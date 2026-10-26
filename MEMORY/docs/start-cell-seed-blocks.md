# 示范起点格未来种子续期（task99）

当前阶段：实现与测试完成，**未上线**——等与 task98 一起在组边界部署。没有改 `configs/human_demo_tasks.json`、
没有动线上 `~/pvz-logs/curriculum_state.json`、生产账本、tmux `qc`/`cq`，也没有跑游戏进程。

## 为什么

起点格每格只登记 256 个未来种子（`demo_start.future_seed_start/count`），qc 每组 24–48 个，
6–10 组就用完；L32 在 10-26 06:50 用完（offset 216 + 48 > 256），旧规则只能把格退休。
这些起点格是课程里最贵的部分（示范存档 + 起点决策号阶梯），不该因为种子用完停采。

## 设计

1. **专用池** `pvz_promotion.START_SEED_POOL = [200,000,000, 201,000,000)`（1,000,000 个 = 3,906 段 × 256）。
   机械核对与全部已登记段不相交：门禁 `[31024,65536)` / `[100,000,000,101,000,000)`、
   示范 2026xxxx 与 44,000,000–47,000,000、llmplay 83,000,000+，以及扫描里出现过的
   `[120,000,000,120,006,000)`、`[126,000,000,126,020,000)`、`[130,000,000,140,000,000)`
   ——最高占用到 140,000,000，新池在其上方，9 位数（与门禁第二段同宽）。
   **第一版候选 `[90,000,000, 91,000,000)` 已废弃**：task89 的 llm-run-import e2e 测试
   已经在 `artifacts/e2e/scan_jobs.json` 声明、并在 `scan_out/*/games.jsonl` 实际跑过
   从 90,000,000 起的未来种子块。见“问题”。
2. **账本** `~/pvz-logs/start_cell_seed_ledger.json`（git 外）。复用 task88 的
   `promotion_gate.allocate_seed_block`：`fcntl` 文件锁、按池序发块、块只发一次、从不回绕；
   文件不存在时第一次领取才创建。与门禁账本分开，互不影响（池不同，混用会触发 ledger 的池校验）。
3. **多段种子表**：格子里 `seeds` 仍是扁平列表（偏移语义不变：offset 是从表头数的绝对下标），
   新增 `seed_blocks` 记录每一段的 `[low, high)`。清单里的块**永远是第一段**，旧数据的
   offset / label / 抽样契约都不变。
4. **qc 侧**：`plan_round` 发现 `offset + games > len(seeds)` 时先续领 256 个接到表尾，
   再排这一组；续领写进 state（`seed_blocks`）和 changes 日志（`seeds renewed +256 from [lo,hi) ...`）。
   只有领不到（池耗尽）才退休，并记 `retired_reason='seeds exhausted'`。
   清单块**必须落在续领池之外**：`load_demo_start_cells` 在启动时就检查，落进
   `[200,000,000, 201,000,000)` 直接 `ValueError`（清单块不在账本里，撞了也查不出来）。
5. **重启**：`reconcile_start_cells` 只在“状态里的表不再以清单块开头”（清单换了块）时重置，
   否则保留续领段。已退休的起点格可用 `--restore-start-cell <task_id>` 从 `retired_cells`
   原样取回（保留偏移、续领段、档位）；清单里已经没有的格标 `legacy_restored`，否则下一次重启
   会被当成“已删的格”再退休。恢复时若它的未来种子与在跑的格重叠，日志里打
   `WARNING future seeds overlap ...`（种子跨格复用会让证据不独立）。
6. **盘点**：`scripts/gate_seed_inventory.py` 把新池（常量）与它的账本都算进占用，
   `--ignore-start-pool` 用来审“新池本身”是否与旧段不相交。

## 证据

- 测试（`/home/newbiexvwu/.venvs/ml/bin/python3`）：
  `python/test_curriculum_queue.py` **66 passed, 1 skipped**（新增 11 项：续领后排 48 局且偏移不变、
  两段有序不相交、两格续领拿到的块也不相交、池在全部已登记段之外、重启保留多段表、
  清单换块才重置、退休格恢复后续领、恢复时重叠告警、legacy 恢复不被二次退休、池耗尽退休、
  清单块落进续领池时启动即拒绝）；
  八个相关测试文件合计 **126 passed, 2 skipped**。
- 池不相交（full scan，274,472 个 JSON/JSONL + 全部声明段）：`MEMORY/docs/gate-seed-inventory.json`
  的 `candidate` = `{low: 200000000, high: 201000000, overlaps: [[200000000, 201000000]],
  self_registered: true}` —— 唯一的“重叠”是池自己（它已作为占用段登记）；再跑
  `python3 scripts/gate_seed_inventory.py --quick --ignore-start-pool --candidate 200000000 201000000`
  → `disjoint: true`，`overlaps: []`。最高占用段是 `[130,000,000,140,000,000)`，池在其上方。
- 全机字面量复核（`--verify-lo 200000000 --verify-hi 201000000`）：**31 命中，全部是
  `~/pvz-logs/metrics/*.jsonl` 的 `rss_bytes` / `io.read_bytes` / `io.rchar`（8 个文件），
  零种子声明**（task88 也遇到过 rss/哈希这类误报）。
- 干跑（`--dry-run`，生产 state 的副本，只把 L32 的 `_hv2` 偏移改成 240）：
  续领一段 `[200000000,200000256)`（state 里 `segments 2`），该组 24 局拿到
  `44300240..44300255 + 200000000..200000007`，偏移仍是 240；预览在临时账本副本上领，
  生产账本没有被创建（`~/pvz-logs/start_cell_seed_ledger.json` 仍不存在）。

## 上线步骤（与 task98 一起，组边界）

1. 把 `startseeds` 合进生产 `cq` 工作树（merge，不手抄），确认 `git diff --stat` 只含
   `python/pvz_promotion.py`、`scripts/curriculum_queue.py`、`scripts/gate_seed_inventory.py` 与测试。
2. 在组边界停 qc：`touch ~/pvz-logs/stop_curr`，等当前组跑完（`stop` 只在组间检查）。
3. 用新代码启动，**不需要新参数**：`--start-seed-ledger` 默认就是
   `/home/newbiexvwu/pvz-logs/start_cell_seed_ledger.json`。第一次有格用完 256 个时账本自动创建，
   日志出现 `seeds renewed +256 from [200000000,200000256)`，state 里该格多出 `seed_blocks` 第二段。
4. 观察 2–3 轮：`grep 'seeds renewed' ~/pvz-logs/curriculum.log`；确认没有格因为
   `seeds exhausted` 被退休；确认 `~/pvz-logs/start_cell_seed_ledger.json` 的 `allocated` 每段只出现一次。
5. **不要**对 `demo_fog_32_snow_m200`（deploy14 退休的旧 L32）用 `--restore-start-cell`：
   它与 `_hv2` 共用块 `44300000..44300255`，且清单条目没有 `fog_observation: human_v2`
   （重放会回到 x-ray）。L32 由在跑的 `_hv2` 承载，续领后它不会再退休。恢复参数留给
   “将来确实因为种子用完退休、且块与观测契约都不冲突”的格；恢复时脚本会打重叠 WARNING。
6. 回滚注意：旧代码不认识 `seed_blocks`，重启时 `reconcile_start_cells` 会用清单块覆盖种子表，
   offset ≥ 256 的格会被重新判为用完并退休。回滚前先确认没有格的 offset 落在续领段里，
   或把 state 备份后再回滚。

## 问题（按最佳判断处理，记录备查）

**合并前发现**：startseeds 缺 `391c8a64`（archive 识别起点格），已 cherry-pick 进 `08cc5c72`。
不并的话，task99 上线会把 deploy14c 的修复覆盖掉，`pool_l25_orig_train` 又会被当起点格退休。
冲突解决：retire 判据同时保留 task99 的 `legacy_restored` 例外；恢复也按 `archive` 判定。
相关测试 124 passed。


1. **第一版候选 90,000,000 与 task89 的 e2e 测试种子冲突。** 发现路径：full scan 的 `occupied`
   里出现了 `[90,000,000, …)`，来源是 `llmimport` 工作树的 `artifacts/e2e/scan_jobs.json`
   （声明了 90,000,000 / 90,001,000 … 的 future_seed_start）和 `scan_out/*/games.jsonl`
   （实际用 90,000,000、90,000,001 跑过局）。虽然只是测试数据，但“块只发一次、永不复用”是硬规则，
   所以换池。**教训：`--quick` 只读 DESIGN + 账本 + 已登记清单，跳过 `scan_json`，审新池必须跑 full scan。**
2. **8 位窗口的字面量复核会被僵尸 id 污染。** `frames.jsonl` 的 `zombies[i][9]` 是引擎僵尸 id，
   实测在 `[90,000,000, 91,000,000)` 命中 30,990 次（按字段分类全是 id，不是种子）。9 位窗口没有这一类；
   但 9 位仍可能命中 `rss_bytes` 之类的十进制大数，命中必须按字段甄别（task88 也踩过 rss/hash）。
3. **deploy14 退休的旧 L32 `demo_fog_32_snow_m200` 不能直接恢复，原因有两条**：
   - 它与在跑的 `_hv2` 共用块 `44300000..44300255`（同存档、同阶梯），offset 216 之后会用
     hv2 将来也要用的种子；恢复时脚本会打 `WARNING future seeds overlap …`。
   - 它的清单条目（`configs/human_demo_tasks.json` 第 18 行）**没有 `fog_observation: human_v2`**，
     重放会回到 x-ray 观测——这正是 task70 B6 退役它、另建 `_hv2` 的原因。
   所以 L32 由 `_hv2` 承载（同存档、同阶梯，offset 72），续领后它不会再退休；`--restore-start-cell`
   留给“块不冲突、观测契约仍有效”的退休格。若以后确实要恢复某个旧 id，先确认这两条都过。

## 上线前核对（2026-10-26 实测，只读）

- **合并方向**：`git merge-base startseeds cq` = `c84ad7d1`（cq 的 HEAD），startseeds 领先 692 个
  提交、cq 领先 0 个；`git merge-tree --write-tree startseeds cq` rc=0，**可以 fast-forward**。
- **必须带上 `391c8a64`（origin/deploy14c）**：起点格改以 `archive` 识别，因为原生倍率阶梯格
  （`pool_l25_orig_train`）也带 `ladder`，用 `ladder` 判定会把它当起点格退休（deploy14 后一直没采）。
  startseeds 原来没有这个提交，已在 `08cc5c72` 并入（含 `test_native_ladder_cell_is_not_a_start_cell`）。
- **cq 的工作树不能直接 ff**：`git -C cq status --short` 有约 130 个已暂存文件（运行中的
  deploy14fix 副本），相对 `0b8dcc26` 唯一实质差异就是那 11 行 archive 补丁（现已并入 startseeds）。
  ff 前必须先备份并清理 cq 的索引/工作树（`git -C cq diff --cached --stat` 逐项确认没有别的未提交改动）。
- **deploy15 的 preflight 会拒绝**：`qc_deploy15.sh` 里有
  `git -C "$W" diff --quiet 0b8dcc26 -- scripts python recipes configs || abort`；合并 task99 后
  这条必然失败，task99 的部署脚本要么换 REV（= startseeds 合并提交），要么去掉这条 pin。
- **与生产代码的差异只剩 task99 本身**（2026-10-26 实测）：`diff cq/scripts/curriculum_queue.py
  startseeds/scripts/curriculum_queue.py` 只有续领相关的增删（imports、清单块入池闸、`seed_blocks`、
  `renew_start_seeds`、`plan_round` 的续领分支、两个 CLI 参数、dry-run 预览），archive 判据已一致；
  `pvz_promotion.py` 只多 `START_SEED_POOL`/`START_SEED_LEDGER_DEFAULT`/`START_SEED_BLOCK` 三行。
  **没有任何格用完 256 个之前，qc 行为与现在完全一致。**
- **本任务改的文件**：`scripts/curriculum_queue.py`、`python/pvz_promotion.py`、
  `scripts/gate_seed_inventory.py`、`python/test_curriculum_queue.py`（+ 文档/盘点 json）。

## 边界与未做

- 池 1,000,000 个 ≈ 3,906 段；24 个起点格每格每 6–10 组领一段，按当前节奏够用很久。
  真耗尽时行为与旧版一致：退休并记 `retired_reason`，不静默复用。
- 续领的块在“领到但本轮没跑”（例如组前看到 stop 标志）时会浪费 256 个种子：state 未保存，
  账本已记。块不会复用，所以只是浪费，不会污染数据。
- 没有改卡组、没有新增起点格；旧数据（`seeds` 前 256 个、offset、label）完全不变。

## 起点短局的 plane_of 修复（同日追加任务）

`CompactBoards.append` 此前在首次出现常量平面前不记映射。ce3ce400 标签的前 4 份存档各保留 3 行，
直到第 5 份存档才有平面，前 12 行因此漏记，出现 84 行 / 72 个 id。现首次出现平面时回填先前行的 -1，
后续行继续按平面编号追加。

平面值也转为 store 声明的 dtype，修正无缓存 float32 平面与缓存 float16 板面之间的精度差；
CSR 和常量平面还原的训练输入遵守同一精度。新段的 board/scalar plane_of 均逐行写入；老的
“无平面且空映射”格式仍可读取。复用已有段时只查 npz 头，长度或平面精度不符就重建。
采样、PACK_VERSION=2 和源指纹不变。

对实际 `c18demoroof48doomm200s1100_fast_search` 的 24 份存档，使用相同选行契约在本树 ignored
目录重打包，验证 84 行、两份映射均 84、前 12 个 board id 为 -1；缓存/无缓存全部 payload 逐字节一致。
源段与存档逐一 SHA 回验未变。详见 [start-cell-pack-check.json](start-cell-pack-check.json)。

组边界上线（打包部分）：与 task98 一并合入后，仅重打包该标签；旧段的错误形状会被新 packer 自动识别，
无需重建全语料。随后检查段 `ce3ce4006691a0e1.npz` 的 `rows`、`board_plane_of.shape[0]`、
`scalar_plane_of.shape[0]` 均为 84，再允许失败训练重试。读取端的止血补丁不能代替重新打包：
它避免崩溃，却无法从漏掉的映射恢复被剥离的平面值。

```bash
PVZ_NO_POPUP=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/newbiexvwu/.venvs/ml/bin/python3 /home/newbiexvwu/PvZ-search/.claude/worktrees/cq/scripts/pack_training_rows.py --data /home/newbiexvwu/pvz-logs/curriculum_state.pipeline/frames --labels c18demoroof48doomm200s1100_fast_search --wait-keep 0.1 --obs v2 --cells --packet-head identity --aux-horizon 0
```
