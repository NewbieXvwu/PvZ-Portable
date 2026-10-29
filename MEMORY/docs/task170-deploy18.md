# task170：deploy18 集成与生产调用链核对

只写代码和部署说明，不部署。工作分支 `deploy18`，起点为当时最新 `origin/deploy17` 的 `aa836760`，现已合入deploy17最新提交 `ad9e751f`（合并提交 `27045c37`）。按主控决定清除后续提交带入的面板JSON与退役机制，不新增校验、manifest或JSON中间文件。

## 集成内容

按顺序合入 gateunits `87cf094b`、obsrec `8907187b`、roofwave `e7f9443e`、abqueue `6017cd67`、purgedead `6d147a62`、purgelive `c772b812`。追加日志冲突保留两边记录。门禁冲突采用清理后的单一v6路径，保留屋顶90对精确Wilcoxon、逐对波数和四区Holm，删除旧版本兼容及重复证书重算。评估继续使用gateunits的统一保底分配，未恢复挑选最多2单元的旧限制。

保留deploy17的day_orig内存生成方式。`day_identity`、`unified_v1`和`unified_v2`现由 `scripts/unified_panel.py` 在内存生成，删除三个约8.5万行的JSON。课程源、单局任务读取与倍率生成入口共用该读取函数，冻结状态里的旧面板路径仍能直接解析为对应任务。panel_v2配方改用 `builtin:unified_v2` 和 `builtin:day_identity`。

一次性将三个生成面板的79项任务与原JSON完整任务字典逐项比较，全部一致，覆盖id、关卡、卡组、完整种子列表及扰动参数。只读查看生产状态的380个格子，其中直接使用这些生成源的 `day_l10_orig_train` 仍为L10，定义继续沿用day_orig函数。没有留下核对脚本或常驻检查。

删除task165两个逐路径JSON附件、apply_deploy17_paths、stabilize_production_paths、qc_deploy17_paths、retire_task20和退役测试。移除pvz_corpus_policy及配方、帧目录、课程打包里的标签过滤。丢失的256局d10orig由用户在数据层处理，软链已移到 `~/pvz-logs/retired_d10orig_links/`，代码按普通目录与标签处理，不记录或跳过这批档案。

## 误删与路径结论

只读查看生产 `qc_deploy17.sh`、`deploy17_at_boundary.sh`、`qc_watchdog.sh`，并核对课程控制器的子进程路径。控制器调用的 search_play、pack_training_rows、train_rollout_policy、select_student和promotion_gate均在，部署17的边界助手、随机卡组生成器和看门狗也在，未发现这些生产必需脚本被误删。

GPU服务实际从 `~/pvz-logs/gpu-window/code/323b867…/scripts/gpu_window_runner.py` 启动，队列的训练/评估与数据准备脚本指向独立冻结代码目录（另含cfa42c1…）。task105的supervise、pipeline、secondary、publish、takeover_scan和search_play实际运行在llmimport工作树，日志目录的task105_refresh只调用该树的task105_results。它们本来未进入deploy17基线，本次不会删除或替换这些独立服务目录。上述服务使用的共用脚本在集成树中仍存在，无需恢复已经退役的一次性脚本。

发现并修正的是路径回归：purgelive把共享数据根从主仓库改成代码工作树，导致cq下缺失日间任务表和价值网络。只读文件元信息确认原文件仍在主仓库。共享任务源按工作树归属推导主仓库，部署命令显式保留旧数据/模型位置。原版传送带源使用day_orig生成函数，不依赖conveyor工作树。

## 部署与回滚

部署脚本沿用deploy17的组边界等待、备份与恢复机制，名称为 `qc_deploy18.sh` 和 `deploy18_at_boundary.sh`，共用已有 `deploy17_boundary.py`。源码从主仓库Git对象按固定提交提取，新引擎使用稳定的 `artifacts/production/deploy18/pvz-portable`，不在切换时编译。准备文件仅保存在本分支，没有放入生产日志目录。

应用源码固定为 `ea8b9dbf`，已包含 `ad9e751f`。清理比较基线继续使用 `80b118e1`，覆盖其后删除的day_orig JSON与本次删除项。安装列表包含day_orig、unified_panel和make_day_panel生成模块，移除已删JSON及退役模块。部署脚本在暂停前从Git读取待装文件，并直接根据两个提交的差异列出已删源码。到达已持久化的组边界后，先把将替换和删除的文件都纳入原有备份，再停旧控制器、安装并清理源码，写active18后启动。回滚使用同一份差异和备份恢复删除项，没有另写清单文件。仅作用于cq内的代码与配置，不清理档案、附件或独立服务目录。

激活命令显式设置 `--min-eval-units 4 --gate-version 6 --obs-v3`。A/B仍为v2timer A与history10 B1，每配方2候选、训练并行1、示范与随机卡组份额各0.25。现行配方已不索要丢失的d10orig标签，B2/B3保持各自只有一个实验字段。按task167说明，B1完成2个完整轮后才由主控改B槽位，部署18不自动推进实验阶段。

所有数据和模型入口在启动命令里显式保留原主仓库位置，避免清理后的默认代码根改变生产读写目录。未激活18时，启动函数复用当前日志目录里的部署17启动函数，不提前使用新引擎、附件或门禁。回滚同样恢复备份的实际旧启动函数。

主控审后，先按现有构建流程从 `ea8b9dbf` 生成包含富观测输出的新引擎，放到 `/home/newbiexvwu/PvZ-search/artifacts/production/deploy18/pvz-portable`。本任务没有编译或运行该引擎。然后安装、执行两个准备好的壳脚本：

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

前次集成的保底评估、obs附件、屋顶门禁、A/B共包和边界回滚相关小测试已通过。本次变更运行现有任务读取与课程入口/打包小项，共13项通过，另含5个配方子用例。修正现有测试夹具对已删旧配方和已撤销解释器回退的依赖，改用现行配方与指定解释器。每次使用 `timeout 300`、`ulimit -f 2000000`，一次一个pytest，没有整套运行三个重型测试文件。

`curriculum_queue.py --help` 和壳脚本语法检查通过。源码、部署说明与回滚入口准备完成。没有生产写入，没有启动训练或对局，提交推送deploy18后停下等主控。
