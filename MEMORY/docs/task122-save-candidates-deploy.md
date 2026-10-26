# task122：老师 `--save-candidates` 接进生产课程（随 deploy14h）

主控 2026-10-27 派给 Mac cbp-task121（CodeBuddy，已熟悉 deploy14h）。分支 `rebalance2`（在 task121 基础上继续）。

## 背景

task120（分支 blindspot，`MEMORY/docs/task120-teacher-blind-spot.md`）：引擎合法动作只含当前买得起的卡，学生的消费习惯让贵卡对搜索永久不可见。`search_play --save-candidates` 让搜索评估“攒够再种”。生产卡组复测：气球 L33/34/39 老师 8/48 → 20/48（McNemar p=0.0018），L42 / L7 / 8 个蒸馏缺口关不退步；每局墙钟 +30–70%（L7 约 +5%）。

## 主控决定

- 先只在**雾夜区课程格（L31–L40，含示范起点格和 opening119 的 L32/L37 格）**打开 `--save-candidates`，其他区保持关闭。墙钟代价集中在雾夜区；下一步看雾夜区老师胜率与每小时局数再决定是否全开。
- 开关做成课程格 / 区级的可配置项（不硬编码关号判断在搜索里），记进每组日志，便于对照。

## 要做的

1. 把 blindspot 分支里 `search_play --save-candidates` 的实现合进 rebalance2（只取它需要的文件）。
2. 课程控制器按格把 `--save-candidates` 传给采集；雾夜区格打开。
3. 更新 `scripts/qc_deploy14h.sh`：CODE_FILES 加上新文件；**引擎若需要改动**说明清楚（若只是 Python 搜索侧则不用换引擎）；未激活 / 回滚分支保持 deploy14g 在役参数（主控已修过一次）。复制到台式机 `~/pvz-logs/qc_deploy14h.sh`，不执行。
4. 只跑相关测试（AGENTS §6.5）。推送，结果写在本文件末尾。

## 执行结果（2026-10-26）

分支 `rebalance2`，在 task121 之上。实现提交 9afe6536，部署包 0525539d。

### 1. 搜索侧：原样取 blindspot 的实现

`scripts/search_play.py` 与 `python/test_save_candidates.py` 从 `origin/blindspot` 整文件取来。
这两个文件在 blindspot 与 rebalance 的 merge-base 之间只差这一个功能（`git diff ad651176 origin/rebalance -- scripts/search_play.py` 为空），所以没有手工合并，也没有夹带 task120 的其他改动。

- 默认 `--save-candidates 0`（关）；探针把 `env.episode` 置空，等待与未来评估不进导出操作流。
- 只在"卡价 ≤ 2× 当前余额"时触发（`--save-share 0.5`），同卡 600 tick 冷却，最多 1800 tick。
- **不改引擎**：全部在 Python 搜索侧（`restore_fast` / `snapshot_fast` / `static_scores` 都是现有接口），deploy14h 不换 build6。

### 2. 课程控制器：区段默认 + 逐格覆盖

`scripts/curriculum_queue.py`：

- `--save-candidates-levels`（默认 `31-40`，空串 = 全关）→ `parse_levels()`。
- `save_candidates_for(cell, task, levels)`：清单任务里写了 `save_candidates` 就按它，否则按区段。判断在控制器，搜索里没有任何关号。
- `plan_round()` 把结果写进 plan item，`command_for()` 只在为 1 时才追加 `--save-candidates 1`（为 0 时命令行与之前逐字相同）。
- 每组日志行加 `save=0/1`，item 也带着这个字段进 `state['log']`，所以轮报告里能逐组对照。
- OOM 探针（`measure_worker_bytes`）与 dry-run 的 `collect:` 预览都传同一个值，预览里能直接看到哪些格带了这个 flag。

默认 31–40 正好覆盖雾夜区课程格（L32/L37 的 opening119 专项格、雾夜示范起点格、fog 训练格），其他区不受影响。**14h 的 `start_qc` 不需要加参数**：两个分支都用控制器默认值，回滚分支仍是 deploy14g 的在役命令行（主控 2026-10-26 修过的那一版），旧代码不认识这个 flag，所以不能显式传。

### 3. 部署包

`scripts/qc_deploy14h.sh`：REV 改为 `9afe653685c60997abe1f165dd0758d50dc846d7`；CODE_FILES 增加
`scripts/search_play.py` 与 `python/test_save_candidates.py`。preflight 基线用台式机 cq 工作树当前值：
`scripts/search_play.py` = `52d00f23…`（仍是 deploy14g 那份，已核对等于 7592889a 的内容），
`python/test_save_candidates.py` = null（不存在，回滚时删除）。其余七个文件的基线与 task121 相同。
已复制到台式机 `~/pvz-logs/qc_deploy14h.sh`（SHA256 `5c3f12007571b99474166b4d60ff0cf52b0f3acbc2c66d7262ebe97c62d5d9d1`），未执行。
另在 cq 工作树做了一次 `git fetch origin rebalance2`，让 preflight 的 `cat-file -e` 能找到 REV（不动工作区、不改分支）。

### 4. 测试（只跑相关的）

- `python/test_save_candidates.py`：6 passed（blindspot 原样）。
- `python/test_curriculum_queue.py`：78 passed / 1 skipped / 4 subtests，其中 3 项新增：
  区段解析（`31-40` / `31,32,37` / 空串 / 逆序报错）、逐格覆盖优先于区段、
  只有开的时候命令里才出现 `--save-candidates`，以及 plan_round 给雾格标 1、日格标 0。
- 引用 `search_play` 的其余纯 Python 测试（agreement stats / roof / critical forks / b3 / label noise / student observation）：66 passed / 1 skipped。
- `bash -n scripts/qc_deploy14h.sh`。

### 5. 上线时要盯的

- 墙钟：task120 生产卡组实测 +32%（L32 78.9→104.1 s）、+34%（L37 76.8→102.8 s）。雾夜区每轮 12 格 × 48 局，预算按这个比例算。
- L37 在 8 局泛化里出现过 1 局由胜转负（噪声内，但门禁要盯）。
- 只对"卡价 ≤ 2× 余额"的卡生效；屋顶 300/200 的卡仍然看不见（task120 结论：屋顶要从学生攒钱侧解）。
- 探针不跨决策绑定，长局里可能出现"等到一半又不种"的空等，属已知代价。

### 未做

没跑对局、没执行部署脚本、没动台式机 qc 与看门狗、没改引擎、没改旧配方。
