# task68：雾夜第一个冷启动族 `fog_user_l31_snow` 接进 qc 课程

> 主控 2026-10-24 写。执行者：Mac CodeBuddy（`cb-demo` 会话，接着 task67 做）。工作树新建 `/Users/newbiexvwu/PvZAgent/.claude/worktrees/fogq`（分支 `fogq`，从 origin/pvz-env 拉出，主控已建好）。最多 2 个游戏进程；不碰 p5d 与其他会话。**只改代码、清单与测试，部署由主控在台式机做。**

## 为什么
task67（`MEMORY/docs/fog-user-decks.md`，demo 分支）的结论：
- L31 用用户卡组（阳光菇/小喷菇/大喷菇/睡莲/高坚果/窝瓜/缠绕海草/樱桃/坚果/**寒冰射手**），老师冷启动 **25/32**（78%，配对 p<1e-6）。同一批种子下，原 qc 雾族卡组是 0/32。
- 唯一的差别是寒冰射手换掉了路灯花。老师会把路灯花反复种下（每局 4.3 株），把经济节奏带偏。
- 这是雾夜区第一个过 50% 的冷启动族。**主控决定接入 qc，起始倍率 1.0。**

## 要做的
1. **登记任务族**（AGENTS §8：新族名，旧族不动，DESIGN 登记）：
   - 族名 `fog_user_l31_snow`；关卡 31；卡组 `[9,8,10,16,23,17,19,2,3,5]`；`fog_observation: human`。任务族的其他字段，例如二周目、香蒲所有权，照 task67 面板的用法。
   - 训练种子段、评估种子段都要新开，与示范种子 20261031–34、task67 面板 20261101–20261132、现有 qc 各族都不重叠，写进 DESIGN。
2. **接入课程清单**：
   - 照 task50 / zones 接入 L41 屋顶格的做法：`scripts/make_multiplier_tasks.py` 的 `FAMILIES` / `CURRICULUM_LEVELS`，以及 `artifacts/stage4/multiplier_panel.json` 的生成方式。
   - 在 qc 清单里加一个 cell：`fog_user_l31_snow`，倍率阶梯从 1.0 起，与其他族同样的升降档规则。
   - 重新生成清单时，**现有 cell 的 task_id、种子、偏移一律不变**。用测试或比对证明，其余 cell 逐字节相同。
3. **老师证据**：`~/pvz-logs/teacher_rates.json` 的格式见 `curriculum_queue.py`（`--teacher-rates`）。准备一条 `fog_user_l31_snow: 0.78`（来源 task67），给出要追加的 JSON 片段，主控部署时写入台式机。
4. **学生能吃这个族的数据**：
   - 确认 qc 生产学生（v2 观测）在 `fog_observation: human` 的任务上能正常采集、打标签、打包、训练。task45 已验证过非雾存档的学生输入逐位一致，这里要验证雾族一局能完整走通。
   - 在 Mac 上用 `curriculum_queue.py` 的同一条 `search_play` 命令（`command_for`），对新 cell 跑 2 局冒烟，产出的帧能被 `pack_training_rows.py` 打包。
5. **promotion 面板不动**：晋升门禁的冻结面板不加雾族，避免改变门禁合同。在报告里写明这一点，并说明新族的学生进展怎么看：qc 自己的组胜率窗口。
6. 测试全过（`python/test_curriculum_*.py`、`make_multiplier_tasks` 相关），提交并 `git push origin fogq`，结果写本文件。

## 规则
- 只用绝对路径；不 `cd` 进别的工作树；`PVZ_NO_POPUP=1`。
- 禁止 `git add -f`；`.pt` `.npz` `.log` 不进 git；不写字面密钥；不手写打法规则。
- 每完成一小步就提交、push，在"结果"记一行；疑问写"问题"，按最佳判断继续。

## 结果

### 1. 新族登记（2026-10-24）

- 族名 `fog_user_l31_snow`，关卡 31，卡组 `[9,8,10,16,23,17,19,2,3,5]`（= task67 面板同一集合与顺序），`fog_observation: human`，二周目、倍率 1.0、不限波、阳光 50、无预置植物。卡组含睡莲 16 覆盖水路；不含升级植物（43 不在卡组），`profile_for_deck` 不需要 `owned_upgrade_plants`，槽位数 10 = 卡组长度。
- 生成器 `scripts/make_fog_user_panel.py`（复用 `zone_panel.task_document` / `check_decks`，无手写打法或卡牌排序），本机清单 `artifacts/task68/fog_user_panel.json`：`fog_user_l31_snow`（8192 训练种子）+ `fog_user_l31_snow_eval`（32 评估种子）。清单不入 git。
- 种子段（新开，DESIGN §14.24 已登记）：训练 `43,000,000–43,008,191`；评估 `43,700,000–43,700,031`；六档倍率 `43,100,000 + 档序号×100,000`（每任务 2048，最高 `43,602,047`）。与示范 20261031–20261034、task67 面板 20261101–20261132、雾区 39M/40M/41M/42M、屋顶 44M/45M/46M/48M、倍率 5M/700k、夜间 1.9M/2.2M、泳池 2.0M、n10p 2.4M、n10q 3.3M/3.4M 全部不相交（`test_fog_user_cell_preserves_every_existing_cell` 逐种子断言）。

### 2. qc 清单加 cell，现有 cell 逐字节不变

- `scripts/make_multiplier_tasks.py`：`FAMILIES` 追加 `('fog_user', 'fog_user', 'fog_user_l{level}_snow', (31,), 'fog')`；`CURRICULUM_LEVELS['fog_user'] = (31,)`；`source_paths()` 增加 `fog_user`（`artifacts/task68/fog_user_panel.json`）；新增 `--fog-user-manifest`。
- 种子块：十个旧族已占 `10×32×2048 = 655,360`，档宽 700,000，再插一个族会平移所有已发种子。所以新族用独立块（与 roof 同一规则），`INDEPENDENT_BLOCKS = {'roof': (48_000_000, 100_000, 41), 'fog_user': (43_100_000, 100_000, 31)}`，`seed_start()` 按块计算，旧族索引与公式一字未动。
- **逐字节证明**：`python/test_multiplier_tasks.py::test_fog_user_cell_preserves_every_existing_cell` 用同一批源任务先去掉新族重建、再带新族重建，逐任务 `json.dumps(sort_keys=True)` 比较：**现有 414 个 cell 全部逐字节相同**（task_id、种子、偏移、卡组、全部复制字段），新增 6 个 cell，总 420；新种子与旧 cell 种子、base 训练段、评估段、示范段、task67 段零交集。
- **真实生成器跑通**：本机没有 Linux 主目录里的源清单，用 9 份合成源清单（id/关卡/卡组与真实一致）跑 `scripts/make_multiplier_tasks.py`：`420 tasks, 860160 seeds, decks_checked=860160, seed_overlap=0`，`independent_blocks={fog_user:43100000, roof:48000000}`。
- **离线课程 dry run**（`/tmp/task68/dry_run.py`，走生产 `curriculum_queue` 代码）：训练集合 39 个 base，`fog_user_l31_snow` 在列，`fog_` 前缀只有它；阶梯 `[1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]`；cell `level 31, mult_index 0 → 1.0`；`task_id_for` 解析到 `fog_user_l31_snow`，清单指向新面板；`label_for` = `c1foguserl31snowm100`。把其余 cell 的老师胜率设 0 时，新族按“可教差距”排第一（`0.78125`）。

### 3. 老师证据片段（主控部署时写入台式机 `~/pvz-logs/teacher_rates.json`）

```json
{
  "bases": {
    "fog_user_l31_snow": {"wins": 25, "games": 32}
  }
}
```

`load_teacher_rates` 读出来是 `0.78125`（task67：L31 用户卡组老师冷启动 25/32；同种子对照 `fog_l31_train_f8` 0/32，配对 McNemar p<1e-6）。原有条目不用改，追加 `bases` 里这一条即可。

### 4. 学生能吃这个族的数据（2 局冒烟，Mac）

- 用 `curriculum_queue.command_for` 生成的生产采集命令（`--dagger-beta 0`、`--verify-samples 9`、`--student-candidates 2`、`--workers 2`）跑 `fog_user_l31_snow` 偏移 0 的 2 局：种子 `43000000` / `43000001`，**2 局都输**（`mean_final_wave 11.0`、`mean_seconds 9.8`、`overrides_per_game 14.5`、`searched_per_game 102.5`、`replans 4.0`）。这只是管线冒烟，不是胜率证据。
- 帧档案 `artifacts/task68/smoke_frames/seed_4300000{0,1}/s68smoke_fast_search/`，`meta.json` 记录 `fog_observation: human`、`deck [9,8,10,16,23,17,19,2,3,5]`、`level 31`、`task_id fog_user_l31_snow`、`label_source teacher`。
- 打包：`pack_training_rows.py` 用生产配方同款选项（`--obs v2 --packet-head identity --cells --wait-keep 0.1`）→ **302 行 / 1 段**，`packs/9f90a677f7d12540.npz`。
- 训练：同一份 v2 配方（`epochs 1`、CPU）读该 pack 成功：`302 decisions (84 plants), 2 seeds, 160 validation rows`，packet 权重覆盖 8/9/16/17/19 五种植物，`val_loss 7.15`，模型写出。**采集 → 打标签 → 打包 → 训练在 `fog_observation: human` 任务上完整走通。**
- 环境：`fogq` 本树新构建的 `build/pvz-portable`（`cmake -G Ninja -DCMAKE_BUILD_TYPE=Release`），学生 `dw4i_s1`、叶值 v5、静态 branch1（后两者只读引用 demo 工作树的文件）。

### 5. promotion 面板不动

晋升门禁的冻结面板（`pvz_promotion.freeze_panel`）**不加雾族**：改面板等于改门禁合同，历史证书会全部失效。新族的进展看 qc 自己的组胜率窗口 —— `~/pvz-logs/curriculum_state.json` 的 `cells.fog_user_l31_snow.windows["1.0"]`（每倍率最近 256 局），升降档按同一条 0.85 / 0.5 规则。

### 6. 测试与提交

- `python/test_curriculum_queue.py` / `test_curriculum_pipeline.py` / `test_multiplier_tasks.py`：**63 passed, 4 subtests passed**。
- 全套 `python/`：**525 passed, 49 skipped, 133 subtests passed, 2 failed**。两个失败都在 `python/test_human_demo.py`，**在本族改动之前就存在**（`27c52101` 上 `test_human_demo.py` 与本提交逐字节相同，`read_segments` 两处都只有 7 处 `source_weight`、都不在分段读取里）：`test_record_replay_export_and_restore`（分段包丢 `source_weight`，task61 回归）与 `test_teacher_starts_at_demo_and_scores_demo_candidate`（引擎内叶值后 `_W` 要 `value_files`）。这两处在 `demo` 的合并提交 `ac51b7c3` 里已经修好，本分支不动它们（避免同一补丁分叉），见问题 6。
- 改动的测试：`test_zones_admission_only_measured_roof_and_one_fog_family`（fog 前缀集合从“空”改为 `{fog_user_l31_snow}`，fog_f8/f8p 仍不进课程）；`test_roof_allocation_preserves_every_existing_seed_block` 改为按族名而不是下标比较（新族排在末尾会改变下标）；`test_seed_blocks_do_not_overlap` 的旧族块宽排除独立块族；新增两个 fog_user 测试。

## 问题

1. **本机跑不了真实的清单重生成。** `make_multiplier_tasks.source_paths()` 指向 `/home/newbiexvwu/PvZ-search/artifacts/stage4/*.json` 与本工作树的 `artifacts/task45|46/*`，这台 Mac 上一份都没有（`artifacts/` 是本地产物、不进 git）。所以“现有 cell 逐字节相同”用两条证据合成：真实生成器 + 合成源清单（420 任务 / 860,160 种子 / 0 重叠）+ 单测里新旧 FAMILIES 的前缀逐字节比对（414 个）。**部署顺序**：先 `python3 scripts/make_fog_user_panel.py`，再 `python3 scripts/make_multiplier_tasks.py`（后者缺新面板会直接 FileNotFoundError，不会静默少 cell）。
2. **跨清单种子穷举审计要在台式机重跑。** Mac 上 `scan_manifests()` 扫不到任何清单，生成审计里 `scanned_manifests=0 / scanned_seeds=0`，这一项在 Mac 上等于没做；面板内部的 860,160 个种子互不重复是验过的。DESIGN 登记的各段不相交由单测断言，不替代台式机的全仓穷举。
3. **2 局冒烟两局都输**（终局第 11–12 波、20 波总量），只能证明管线通。真正的起点胜率用 task67 的 25/32。
4. **小样本训练的验证集切分**：`validation.fraction=0.1` 在 2 个种子上算出 0 个验证种子，训练器会直接报 `recipe split must produce nonempty training and validation sets`。冒烟时改成 0.5（1 训 1 验）。这是样本量问题，不是雾族问题；正式采集的组局数远大于 2，不需要改配方。
5. 新族 1.0 档直接用 base 任务的 8192 种子，倍率变体每任务 2048 —— 与 roof/fog 既有做法一致，没改。
6. **本分支自带两个与本任务无关的既有失败**（`test_human_demo.py` 两项，见结果 6）。它们在 `origin/pvz-env` 上就有，`demo` 分支的 `ac51b7c3` 已修（分段包 `source_weight`、引擎内叶值的 `value_files` 打桩）。本分支没有重复修，避免同一补丁在两个分支分叉；**合并 fogq 回主线时以 demo 的那两处为准**，不要因为本分支的红灯另写一遍。
