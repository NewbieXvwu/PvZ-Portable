# task173：deploy19 集成与一次性演练

分支 `deploy19`，起点 `origin/deploy18`（3ee03561），包含示范起点重放去状态摘要热修。只准备代码和部署脚本，不部署。新生产引擎由主控从固定源码构建。

## 集成范围

合入 `origin/fognative`、`origin/fogspeed`、`origin/purgelive` 和 `origin/purgepaths`。`origin/purgefix` 发布后再合入。冲突按简化优先，保留现行采集与门禁语义，沿用清理后的内存任务源，不恢复摘要或重复验证。

已合入 fognative `9607bbdd`、fogspeed `b75b37c9`、purgelive `56bc810a`、purgepaths `4d0daf5e`。日志保留两边追加记录，资源路径采用共享缓存及现有环境变量覆盖，已删除的整包摘要对拍与实机看门狗探测测试保持删除。

集成修复两处路径/配置问题：配方路径先展开 `~`，避免 `panel_v2.resource_dir` 拼到仓库下。读取旧冻结配方时丢弃已退役、值为空的 `row_weights`，使组边界切换后已排队的旧配方仍能读取，行倍权实现与新配方字段继续保持删除。

雾关纯学生整局使用 C++，搜索采集使用既有原生根前向及续走。fogspeed 的两项引擎优化为原生整局跳过绘制高度工作，以及纯超时等待只计算仍需使用的公开状态。门禁质量的 45 种子 A/B 由主控安排，本任务只做下述最小链路演练。

## 部署与回滚

仿 deploy18 的组边界暂停、备份和启动方式，准备 `scripts/qc_deploy19.sh` 与 `scripts/deploy19_at_boundary.sh`。新引擎使用稳定生产路径，不在切换期间编译。保留一条命令回滚到实际旧启动函数，沿用既有边界助手。源码固定到集成提交，数据与模型沿用现行目录。

当前阶段的安装源码固定为 `e08b7a4f`，比较基线为 `3ee03561`。脚本从主仓库 Git 对象提取安装文件，保存实际旧文件与旧启动函数后，在组边界切换。安装列表补齐本次改动的打包入口、看门狗和资源路径工具。原有备份与调度记录格式沿用 deploy18，没有增加清单文件。

生产引擎由主控构建后放到 `$HOME/PvZ-search/artifacts/production/deploy19/pvz-portable`。启动显式使用 `--native-games --obs-v3 --min-eval-units 4 --gate-version 6`，A/B、采集份额和数据位置沿用 deploy18。尚未激活19时，脚本复用日志目录内的实际 deploy18 启动函数。回滚恢复备份文件及同一个旧启动函数，保留原控制器状态与种子记录。

主控审定最终源码并备好引擎后，沿用部署18的安装方式：

```sh
R=$HOME/PvZ-search
git -C "$R" fetch origin deploy19
git -C "$R" show origin/deploy19:scripts/qc_deploy19.sh > "$HOME/pvz-logs/qc_deploy19.sh"
git -C "$R" show origin/deploy19:scripts/deploy19_at_boundary.sh > "$HOME/pvz-logs/deploy19_at_boundary.sh"
bash "$HOME/pvz-logs/deploy19_at_boundary.sh"
```

一条命令回滚：`bash "$HOME/pvz-logs/qc_deploy19.sh" --rollback`。备份目录记录于 `$HOME/pvz-logs/deploy19.latest`。这些命令仅作交付说明，未在生产执行。

## 一次性 Mac 演练

依次运行普通、示范起点与雾关采集，随后真实打包、训练一步，再运行最小挑选/门禁链路。采用已有任务定义与数据读法，串行执行，遵守 Mac 并发上限和 AGENTS §6.5 第 8 条。只演练一次集成后的链路，发现真实错误时修复受影响部分。完成后删除临时目录，不保留演练脚本或常驻检查，不新增校验、manifest 或 JSON 中间文件。

## 结果与交付

已用本分支独立构建的 Mac 引擎完成一次最小链路演练，串行执行，未触碰其他训练。采集使用现有任务卡组，普通 L7 与雾 L31 采用0.1倍率及原任务波数，示范起点使用真实 L25 原版传送带存档末段、保持存档原环境参数。普通/雾/示范分别在755/436/27个决策后得到完整终局，obs v3附件分别为67,825/55,685/10,494字节。

随后按现行v2timer行契约真实打包，保留295行（63个种植标签），其中217行进入验证。以现行A结构完成CPU单线程训练1个优化步，生成736,558字节模型。真实调用挑选入口，学生与候选在白天、夜间、泳池、雾和屋顶各跑1对；再经过门禁单元执行入口跑同样的5对，并调用现行v6判定，屋顶1对波数端点进入Wilcoxon路径。小样本返回拒绝，属于这次链路输出，不作为生产晋升或原生路径质量判定。

演练任务定义只驻留内存，没有写新任务清单或配方中间文件。产物均来自原有采集、打包、训练和挑选接口，演练脚本、模型、帧档、构建目录及全部临时产物在完成后删除，没有留下常驻检查。生产Linux引擎仍由主控构建，本任务未部署。

截至本阶段提交，`origin/purgefix` 尚未发布。其余四分支和deploy18示范热修已集成并推送。purgefix发布后再补合并、刷新安装源码固定值，最后交主控审定。

## purgefix2：已删生产者的残留消费者（2026-10-30）

基于 deploy19 `73e3c3ad`，对照 task169 清理前后的活代码差异，用 `git grep` 追踪被删字段、方法和旁路文件的引用，覆盖 search_play、human_demo、curriculum_queue、打包、训练、挑选、门禁、教师查询及 llmplay 导入。沿用 `3ee03561` 的热修方向，直接删除摘要依赖，不恢复生产者，不以 `.get()` 兜底比较，不增加运行时校验。

逐处处理：

- `human_demo.Recorder.__init__`：删除可执行文件摘要、初始状态摘要和 hash_schema 的残留写入，录制不再依赖已清理的摘要链。
- `human_demo.Recorder._step`：删除对引擎响应 `state_hash` 的取键及逐操作摘要记录。
- `human_demo.replay`：删除 executable_sha256 读取、整份引擎哈希计算和 hash_verification 记录。
- `human_demo.replay`：删除 initial_state_hash / ticks.state_hash 比对及 strict_hashes 开关，保留原有动作时间线、ok 和终局数据检查。
- `human_demo.verify`：删除 strict_hashes 参数与转发，使用同一条无摘要回放路径。
- `human_demo.rebaseline`：删除仅用于读取、比较并重写旧状态摘要的函数及 CLI 子命令。
- `import_llm_run.verify_copy`：删除 strict_hashes 参数转发，继续在入库副本上执行原有动作回放。
- `import_llm_run.run`：包内去重键删除 initial_state_hash 读取，保留原有任务、种子与操作数。
- `episode_query.capture`：删除对 search_play 已不再提供的 checkpoint_sha256 字段的读取与落盘。
- `episode_query.main`：删除检查点指纹读取和展示，只显示原检查点路径。
- `render_episode.collect`：删除另一条采集路径中仅供上述展示使用的检查点摘要生成。
- `replay_roof_wave_gate.replay`：删除 contract 的 candidate/incumbent SHA 取键、依赖它的重复记录检查和输出摘要，直接从现有 paired / 屋顶局记录重算。
- `search_play` 与相关测试：移除过期状态哈希注释、strict_hashes 调用和哈希夹具，原有动作时间线失败用例改为破坏 end_tick。

其余追踪结果：被删的环境资源/源码摘要、experiment_manifest/working_tree.patch、debug_replay/save_replay/replay_record、选择/训练/配方/门禁 SHA、demo_archive_sha256、row_weights 身份字段及 prepared.json，现行执行代码没有残留消费者。`selection_ready.json` 仍由 `_prepare_selection` 写入；token `ready.json` 仍由 `SparseTokenWriter.finish` 写入；pack/cache 身份、门禁 contract_key、HF MANIFEST/归档读取及 resume 指针仍有各自数据来源。`verified` 仍由示范 verify / 导入回放产生，C++ STATE_HASH 仍提供显式状态诊断接口，均未冒充已删生产者处理。

台式机只跑相关 `pytest -k`：11 项通过，另 5 个子用例通过；每次先看可用内存，限制 `ulimit -v 8000000`，不运行真实对局或整套重型测试。回归夹具覆盖不带摘要的示范录制、示范起点与反事实回放，以及无 SHA contract 的屋顶重算；现有导入、打包命令、续训、挑选和门禁用例通过。未操作 qc、看门狗、stop_curr 或部署流程。

## 暂缓原生整局（2026-10-30，主控本次指示）

修复提交 `1553e7ee` 已推送 `purgefix2`，合并提交 `e7543899` 已推送 `deploy19`；本轮接替尚未发布的 purgefix 消费者排查工作。

`qc_deploy19.sh` 启动参数由 `--native-games` 改为 `--no-native-games`。仅删除正向参数不会关闭此功能，因为 curriculum_queue 的默认值为 True；显式关闭后沿现有参数传递覆盖采集、挑选和门禁，等待主控完成 45 种子 A/B 后再开。其余启动参数不变，obs-v3、min-eval-units 4、gate-version 6、A/B 配方、采集份额及数据位置照旧；部署模式标记仍保留原名 `native`，不代表原生整局开关。

部署源码固定值推进为 `e7543899`，确保将来安装时带上本次消费者修复；激活日志同步注明原生整局关闭。脚本仅做 `bash -n` 语法检查，未执行安装、切换或回滚，未触碰生产 qc / 看门狗 / stop_curr，未运行 pkill。完成提交推送后停下，等待主控。
