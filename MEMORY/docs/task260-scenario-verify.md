# task260：scenario 复验不一致修复

分支 `scenfix`，引擎源码基线 `ec539401`。task260b 已完成定位、146 份重新复验及目标库追加导入。根因是导入指定的旧 `imitater1` 引擎缺少 scenario 注入命令。使用本分支源码独立编译的静态引擎复验后，141 份原异常项全部通过并成为有效入库条目，另外 5 份保持真实失败。最终本包 scenario 为 **310 份成功、5 份真实失败、0 份复验异常**。

## 根因与双引擎证据

原导入引擎是 `~/PvZ-search/artifacts/production/imitater1/pvz-portable`，输出 BuildNum 4107、CommitDate 2026-11-03。其 SHA256 为 `63b4067ef53ab4122bbde18ee5ee46649f8e98735b5e7ac89300ac8064fa6d8c`。

分别使用该引擎与现存 `~/pvz-logs/llmplay-build/pvz-portable-238c` 重放日、雾、夜各一份失败样本。三份均出现同一个分叉：`RESEED` 被接受，紧接着原本成功的 `SET_SUN` 和 `SPAWN` 被旧引擎拒绝，返回 `ok=False`。阳光保留为旧值，指定僵尸未注入，后续种植也会因阳光不足而拒绝。静态引擎接受同样的注入，目标 ID、类型、行与 x 坐标匹配，最后判成功。

| 样本 | 注入操作号 | 旧引擎阳光 → 正确阳光 | 目标 ID | 两引擎最终 tick |
|---|---|---|---|---|
| scenario238_day_007 | 103 SET_SUN、104 SPAWN | 0 → 625 | 66453508 | 7020 |
| scenario238_fog_052 | 128 SET_SUN、129 SPAWN | 35 → 50 | 66453512 | 7811 |
| scenario238_night_024 | 140 SET_SUN、141 SPAWN | 0 → 125 | 66387974 | 9241 |

样本 SHA256：

```text
day_007    487672b0d4f8577e26a194f43713f26a96b3720bc27fb336fc17e73b08e0ba7f
fog_052    b5f8fa0eaccae076be65d96c9d2c4a5f55097eb487913092acf87f8f38c457b2
night_024  8aabf37e05bb1ed4c75b5adced93dcd3205cfc744302586048107e65fe80b3b3
```

[`Game._replay_to`](../../scripts/llmplay/game.py) 会执行操作日志，但没有核对每条操作记录的 `ok`，所以旧引擎拒绝注入后仍继续重放。最终判据看到的是另一个棋盘，表现为 `meta=True replay=False`。目标以对象 ID 识别，后续自然出生的僵尸还可能占用原定注入目标的 ID，旧引擎的最终存活目标因而不能用于判断 GPT 的实际操作效果。

task248 及 `0fea7ad4` / `e91e220b` 的起点、推车和终局修复均已存在，见 [11-05 记录](../2026-11-05.md)。本次 146 份正确引擎重放中，每条操作的接受结果和终点 tick 都与原日志一致。雾样本根 tick 6600、视界 1200、终点 7811，双方都在 7811 判成功。未发现本批由起点或等待时长错位造成的异常。

## 修复方式与生产边界

修复采用导入器已有的 `--executable` 参数选择支持注入命令的独立引擎。运行代码无须改动，判据及重放继续调用现有唯一实现。没有更新 `cq` 工作树、`imitater1` 引擎或 GPT 包。

为取得确定来源的复验引擎，从台式机仓库用 `git archive ec539401 CMakeLists.txt CMake src` 提取源码，在 `~/pvz-logs/task260/` 独立构建。复用已安装的无界面 SDL 静态库及现存 mimalloc 源码，使用 Release、`BUILD_STATIC=ON`、链接参数 `-static`、`PVZ_ROLLOUT_BLAS=OFF`，两个编译进程。引擎 SHA256 为 `fa2e81a2be1ee6dbcc3bfdaf5530c9984cdaa0b2c09fe3e9c1c4605bd7318504`，该摘要已写入新复验回执。

普通对局按每条操作后的 `STATE_HASH` 对照：L6、L16、L26、L36 四份整局一致，L44 屋顶整局首次对照从操作 437 出现哈希差异，双方均在 tick 69945 第 40 波获胜。随后将该局截到操作 437 定位时，未复现哈希差异，原因未定。**现有证据不足以批准整体替换采集引擎，因此上线采用独立导入引擎。**

另取覆盖五种地形的 8 份 drill，共 42,804 条操作，两引擎逐操作哈希及完整判据结果全部相同，包含成功题和失败题。判据代码未改，库内全部 2,691 条 `drill238` 索引记录保持原样，包含本包的 1,945 份 drill。

## 146 份重新复验与入库

原日志记录 scenario 共 315 份：169 份通过并成功，141 份复验异常，5 份通过但局部失败。任务书的日 62、雾 54、夜 25 合计 141，146 是全部 loss 数量。

从原 ZIP 仅提取这 146 份，逐份确认输入 SHA 与原回执一致。使用源码引擎调用部署导入器已有的 `verify_once → verify_short → Game.load_archive` 路径，未另写重放或判据。146 份共用约 53 秒，结果如下。

| 原分类 | 份数 | 正确引擎复验 | 处理 |
|---|---|---|---|
| 日间判据不一致 | 62 | 全部成功 | 追加有效入库 |
| 雾关判据不一致 | 54 | 全部成功 | 追加有效入库 |
| 夜间判据不一致 | 25 | 全部成功 | 追加有效入库 |
| 原通过但局部失败 | 5 | 全部仍失败，判据与原记录相符 | 保留原回执及条目 |

146 份共有 `RESEED`、`SET_SUN`、`SPAWN` 各 146 次，`SET_COOLDOWN` 28 次，未使用 `SET_WAVE`。所有操作的 `ok` 与 `end_tick` 均匹配原日志，注入目标的 ID、类型、行与 x 坐标全部匹配题面。

5 份真实失败均来自雾关，且没有放弃标记：

| 题号 | 未来种子 | 失败原因 |
|---|---|---|
| scenario238_fog_001 | 239000003 | 使用 1 辆推车 |
| scenario238_fog_001 | 239000002 | 目标仍在场 |
| scenario238_fog_013 | 239000027 | 使用 1 辆推车 |
| scenario238_fog_013 | 239000026 | 使用 1 辆推车 |
| scenario238_fog_024 | 239000049 | 目标仍在场 |

追加导入使用 `~/PvZ-search/.claude/worktrees/llmimport/scripts/import_llm_run.py`，实际软链解析为 `cq/scripts/import_llm_run.py`。先在导入锁内用此次复验生成的回执替换 141 份旧失败回执，再调用同一部署模块的 `run()`，复用这些回执，未再重放。单纯重跑命令行会继续复用旧失败回执，不能修复这批记录。

调用参数：

```text
input_path       ~/pvz-logs/task260/retry-input       # 仅 141 份原失败项
dest             ~/PvZ-search/.claude/worktrees/llmimport/artifacts/task105/import
manifest         cq/configs/human_demo_tasks.json
resources        ~/.cache/pvz-research-resources
executable       ~/pvz-logs/task260/build/pvz-portable.static
verify_workers   1
llm_demo_data    None
```

结果为新增有效条目 141，移除有效条目 0，原成功回执字节内容全部保持不变。本次仅追加任务指定的存档库，`llm_demo_data=None` 保证未写入 `~/pvz-logs/gptbc/data` 生产训练数据。原 169 份成功回执及 5 份真实失败回执保留。

## GPT 包下一版

本批重新复验与 GPT 的成功、失败反馈全部相符，未发现 scenario 判据向 GPT 返回错误成功的证据。下一版包的 scenario 判据无需修改，继续保留 task248 的持久起点修复、task237 注入命令及 `58be2506` 的 SPAWN x 范围限制。本任务未打包。

## 上线步骤

1. 主控从本分支源码构建支持注入的引擎，部署到独立路径，例如 `artifacts/production/scenario238/pvz-portable`。可复用 [`build_static_engine.sh`](../../scripts/llmplay/build_static_engine.sh)，构建目录与采集引擎隔离。
2. 后续 GPT scenario 导入固定传 `--executable ~/PvZ-search/artifacts/production/scenario238/pvz-portable`。采集仍使用原 `imitater1`。本批 141 项已经完成恢复，无须再次导入。
3. 未来遇到旧失败回执时，使用部署导入器的 `verify_once` 重新生成该失败项回执，再由原 `run` 发布索引。成功回执继续复用。若后续需要进入生产训练数据，由主控在现有导入流程中传入相应 `llm_demo_data`。

所有远端实验通过 `systemd-run --user` 后台运行，使用 `nice -n 19 ionice -c3`。各次启动前检查 C 盘，始终约剩余 30 GB，满足 25 GB 下限。实验文件均在 `~/pvz-logs/task260/`，完成后删除。原 ZIP、导入日志及既有构建产物保留。
