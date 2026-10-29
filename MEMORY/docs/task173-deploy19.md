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
