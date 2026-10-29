# task170：deploy18 集成与生产调用链核对

只写代码和部署说明，不部署。工作分支 `deploy18`，起点为当时最新 `origin/deploy17` 的 `aa836760`，组装期间补入新推送的 `2f038dbe` 至 `80b118e1`。不新增校验、预检或manifest。

## 集成内容

按顺序合入 gateunits `87cf094b`、obsrec `8907187b`、roofwave `e7f9443e`、abqueue `6017cd67`、purgedead `6d147a62`、purgelive `c772b812`。追加日志冲突保留两边记录。门禁冲突采用清理后的单一v6路径，保留屋顶90对精确Wilcoxon、逐对波数和四区Holm，删除旧版本兼容及重复证书重算。评估继续使用gateunits的统一保底分配，未恢复挑选最多2单元的旧限制。

最新deploy17的原版面板与精确退役两个d10orig标签一并保留，已删旧实验遇到其路径修订时仍保持删除。这里沿用上游已登记的面板，没有生成新manifest。

## 误删与路径结论

只读查看生产 `qc_deploy17.sh`、`deploy17_at_boundary.sh`、`qc_watchdog.sh`，并核对课程控制器的子进程路径。控制器调用的 search_play、pack_training_rows、train_rollout_policy、select_student和promotion_gate均在，部署17的边界助手、随机卡组生成器和看门狗也在，未发现这些生产必需脚本被误删。

GPU服务实际从 `~/pvz-logs/gpu-window/code/323b867…/scripts/gpu_window_runner.py` 启动，队列的训练/评估与数据准备脚本指向独立冻结代码目录（另含cfa42c1…）。task105的supervise、pipeline、secondary、publish、takeover_scan和search_play实际运行在llmimport工作树，日志目录的task105_refresh只调用该树的task105_results。它们本来未进入deploy17基线，本次不会删除或替换这些独立服务目录。上述服务使用的共用脚本在集成树中仍存在，无需恢复已经退役的一次性脚本。

发现并修正的是路径回归：purgelive把共享数据根从主仓库改成代码工作树，导致cq下缺失日间任务表和价值网络。只读文件元信息确认原文件仍在主仓库。共享任务源按工作树归属推导主仓库，部署命令显式保留旧数据/模型位置。最新deploy17又将原版传送带面板改为版本化文件，因此该项直接使用其修复，不再依赖conveyor工作树。

## 部署与回滚

部署脚本沿用deploy17的组边界等待、备份与恢复机制，名称为 `qc_deploy18.sh` 和 `deploy18_at_boundary.sh`，共用已有 `deploy17_boundary.py`。源码从主仓库Git对象按固定提交提取，新引擎使用稳定的 `artifacts/production/deploy18/pvz-portable`，不在切换时编译。准备文件仅保存在本分支，没有放入生产日志目录。

应用源码固定为 `994e1345`，清理的比较基线为 `80b118e1`。部署脚本在暂停前从Git读取待装文件，并直接根据两个提交的差异列出已删源码。到达已持久化的组边界后，先把将替换和删除的文件都纳入原有备份，再停旧控制器、安装并清理源码，写active18后启动。回滚使用同一份差异和备份恢复删除项，没有另写清单文件。仅作用于cq内的代码与配置，不清理档案、附件或独立服务目录。

激活命令显式设置 `--min-eval-units 4 --gate-version 6 --obs-v3`。A/B仍为v2timer A与history10 B1，每配方2候选、训练并行1、示范与随机卡组份额各0.25。B2/B3已与最新A同步退役标签和provenance路径，保持各自只有一个实验字段。按task167说明，B1完成2个完整轮后才由主控改B槽位，部署18不自动推进实验阶段。

所有数据和模型入口在启动命令里显式保留原主仓库位置，避免清理后的默认代码根改变生产读写目录。未激活18时，启动函数复用当前日志目录里的部署17启动函数，不提前使用新引擎、附件或门禁。回滚同样恢复备份的实际旧启动函数。

主控审后，先按现有构建流程从 `994e1345` 生成包含富观测输出的新引擎，放到 `/home/newbiexvwu/PvZ-search/artifacts/production/deploy18/pvz-portable`。本任务没有编译或运行该引擎。然后安装、执行两个准备好的壳脚本：

```sh
R=/home/newbiexvwu/PvZ-search
git -C "$R" fetch origin deploy18
git -C "$R" show origin/deploy18:scripts/qc_deploy18.sh > ~/pvz-logs/qc_deploy18.sh
git -C "$R" show origin/deploy18:scripts/deploy18_at_boundary.sh > ~/pvz-logs/deploy18_at_boundary.sh
bash ~/pvz-logs/deploy18_at_boundary.sh
```

一条命令回滚：

```sh
bash ~/pvz-logs/qc_deploy18.sh --rollback
```

备份目录记录于 `~/pvz-logs/deploy18.latest`。回滚保留控制器状态、学生、已完成组与种子偏移/账本，恢复旧代码、配置和启动函数。原来的边界助手只处理记录下来的控制器进程树。此处命令均为交付说明，本任务未在生产执行。

## 小测试与最终结论

直接相关的 `pytest -k` 已通过：4单元预留和实际命令传递、obs附件开关及无损块编码、v6门禁/历史r29与r27、配对结果续跑、A/B共包与私有权重、退役标签、环境协议，以及部署18激活和含删除项的备份恢复。每次使用 `timeout 300`、`ulimit -f 2000000`，一次一个pytest，单次最长约1.6秒。没有整套运行三个重型测试文件，没有启动训练或游戏进程。

`curriculum_queue.py --help` 和两个壳脚本语法检查通过。没有新增校验器或manifest，没有恢复旧版门禁兼容和已删证书审计。此次核对未发现生产必需脚本误删，已修正合并带来的共享路径和过时测试摘要引用。源码、部署说明与回滚入口准备完成，提交推送后停下供主控审阅。
