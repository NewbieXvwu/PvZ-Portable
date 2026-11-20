# task275：238n 屋顶与缺好卡组关派题

分支 `pkg238n`，基于 `pkg238` 的 `df9c4ab6`，功能提交 `0f503c5a`。未部署生产，未快进 `pkg238`。

排序集中在 `decks.dispatch_priority()`：回退子项 → 屋顶 L41–L49 resume（所有卡组，保留原倍率）→ 屋顶 1.0 倍 native / frontier → L22/L23/L27/L29/L44/L47/L49 decks → 其余原有层次。resume 在截取前 256 项前按同一优先级排序，进度展示与开局检查也沿用此顺序。

7 关使用现有 decks 层自选卡组，记录胜负和种子，保留与历史胜组互差至少 2 张的约束。carry 中泳池关已达到旧配额，因此将这 7 关目标各提高 3 套：L22/23/27/29 为 27→30，L44 为 23→26，L47 为 25→28，L49 为 26→29。任务族与历史定义保持原样。

## 定向结果

使用原 carry 调用 `next_task()` 连续认领 20 项，全部为屋顶 resume，关卡分布如下。carry 没有待续的回退子项。

| 层 | L41 | L42 | L43 | L47 | L49 | 合计 |
|---|---:|---:|---:|---:|---:|---:|
| resume | 7 | 4 | 1 | 3 | 5 | 20 |

将 resume 标为完成后，下一项为 L41 frontier ×1.0。再完成屋顶整局层后，下一项为 L44 decks ×1.0，7 个目标关均有待办且各差 3 套。定向 pytest：`-k 'task275 or task263_focus_resume_and_native_tail or resume_fresh_priority'`，3 项通过，用时 2.66 秒。

## 包与台式机验收

包目录：`/Users/newbiexvwu/Downloads/pvz-llmplay-238n/`。使用任务指定的 `build_package.py --reuse-package` 命令，沿用 238m 的静态引擎、carry 和题库资产。carry 16,613 条，截至 **2026-11-17 07:01:15 HKT**，resume 8,457 / drill 927 / scenario 300。

| 文件 | 字节 |
|---|---:|
| llmplay_pkg.7z.001 | 20,000,000 |
| llmplay_pkg.7z.002 | 482,138 |
| source.zip（`git archive HEAD scripts python configs src`） | 4,211,851 |

`llmplay_res.zip` 沿用 238m。接口两卷与源码经 HF `transfer/pkg238n/` 中转，台式机取用后已 `delete_folder` 删除。资源使用台式机本地缓存。

台式机在 `~/pvz-logs/task238n/` 解包，以 `nice -n 19 ionice -c3`、`MemoryMax=4G` 运行 `source/scripts/llmplay/accept_package.py`，**exit 0**。三进程分别认领 L41/L47/L49 屋顶 resume，互不重复。drill 真实判分与导入复验一致，scenario、L47 模仿者导出及回退链导入通过，连续两次导出进度保持 16,618 条。验收目录已删除。
