# task170：deploy18 集成与生产调用链核对

只写代码和部署说明，不部署。工作分支 `deploy18`，起点为当时最新 `origin/deploy17` 的 `aa836760`，组装期间补入新推送的 `2f038dbe`。不新增校验、预检或manifest。

## 集成内容

按顺序合入 gateunits `87cf094b`、obsrec `8907187b`、roofwave `e7f9443e`、abqueue `6017cd67`、purgedead `6d147a62`、purgelive `c772b812`。追加日志冲突保留两边记录。门禁冲突采用清理后的单一v6路径，保留屋顶90对精确Wilcoxon、逐对波数和四区Holm，删除旧版本兼容及重复证书重算。评估继续使用gateunits的统一保底分配，未恢复挑选最多2单元的旧限制。

最新deploy17的原版面板与精确退役两个d10orig标签一并保留，已删旧实验遇到其路径修订时仍保持删除。这里沿用上游已登记的面板，没有生成新manifest。

## 误删与路径结论

只读查看生产 `qc_deploy17.sh`、`deploy17_at_boundary.sh`、`qc_watchdog.sh`，并核对课程控制器的子进程路径。控制器调用的 search_play、pack_training_rows、train_rollout_policy、select_student和promotion_gate均在，部署17的边界助手、随机卡组生成器和看门狗也在，未发现这些生产必需脚本被误删。

GPU服务实际从 `~/pvz-logs/gpu-window/code/323b867…/scripts/gpu_window_runner.py` 启动，队列的训练/评估与数据准备脚本指向独立冻结代码目录（另含cfa42c1…）。task105的supervise、pipeline、secondary、publish、takeover_scan和search_play实际运行在llmimport工作树，日志目录的task105_refresh只调用该树的task105_results。它们本来未进入deploy17基线，本次不会删除或替换这些独立服务目录。上述服务使用的共用脚本在集成树中仍存在，无需恢复已经退役的一次性脚本。

发现并修正的是路径回归：purgelive把共享数据根从主仓库改成代码工作树，导致cq下缺失日间任务表和价值网络。只读文件元信息确认原文件仍在主仓库。共享任务源按工作树归属推导主仓库，部署命令显式保留旧数据/模型位置。最新deploy17又将原版传送带面板改为版本化文件，因此该项直接使用其修复，不再依赖conveyor工作树。

## 部署与回滚

部署脚本沿用deploy17的组边界等待、备份与恢复机制，名称为 `qc_deploy18.sh` 和 `deploy18_at_boundary.sh`，共用已有 `deploy17_boundary.py`。源码从主仓库Git对象按固定提交提取，新引擎使用稳定的 `artifacts/production/deploy18/pvz-portable`，不在切换时编译。准备文件仅保存在本分支，没有放入生产日志目录。

待完成相关小测试后补充固定源码、部署命令与最终结论。
