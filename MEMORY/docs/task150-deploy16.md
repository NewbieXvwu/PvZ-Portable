# task150：deploy16 集成准备

工作树 `/home/newbiexvwu/PvZ-search/.claude/worktrees/prodint16`，分支 `prodint16`。
基线是 fetch 后的 `origin/prodint=e2f8cc16`（包含生产包 4beab441 与提取兼容修复）。

## 合入与部署包

- `c4e89f61` 合入 `origin/obstimer=4f243a98`：obsup2 / build7 观测链、task145 v2timer、task147 原生输入 9 与缓存复用。
- `0a7f3957` 合入 `origin/sysfail=7285d1a2`：空轮不入队、连续三组零有效局暂停、每 180 秒单局探测恢复、后台系统故障不消耗有限重试预算。唯一冲突为 2026-10-28 日志 add/add，保留两边全部条目，代码无冲突。
- `4e479b04`：A/B 两套 v2timer 配方（init=null，2 个构建 worker），共享训练行选项，B 保留温和关卡再平衡；新增独立缓存预构建入口。
- `0be80647`：可回退部署脚本及定向启动测试。脚本固定安装 `4e479b04` 的 16 个运行文件，含观测/编码/渲染/训练链及 sysfail 控制器/选拔/门禁；新二进制单独使用，无生产二进制替换。

待主控在组边界执行的脚本：`/home/newbiexvwu/pvz-logs/qc_deploy16.sh`。
仓库副本为 `scripts/qc_deploy16.sh`。模式文件 `/home/newbiexvwu/pvz-logs/qc_deploy16.mode` 已写 `v2timer`。
模式是选择器，脚本安装完成后才创建 `qc_deploy16.active`：在准备阶段，若看门狗发现这个新脚本，提取后的 `start_qc` 仍使用原服务参数；安装后才选择 v2timer A/B + build16。没有修改或启动看门狗。
脚本保留 deckcells 的 `--extra-manifest`、gate v5、K=2 与训练串行；进程提前退出导致 kill 失败会忽略；备份完成才写标记，支持 `--rollback` 恢复旧文件与启动参数，保留控制器状态和种子账本。切换路径不做缓存构建、训练、重放或哈希检查。

## 独立构建与冒烟

构建输出 `/home/newbiexvwu/PvZ-search/.claude/worktrees/prodint16/build16/pvz-portable`，CMake Release / Ninja 两个编译任务，复用已有 mimalloc 源码而独立生成全部构建产物。构建日志 `/home/newbiexvwu/pvz-logs/task150-build.txt`。

集成树上的 width64 / meanmax / row_pool / identity 的未训练 v2timer 学生，原生 L7、seed150000 到 tick33549 终局。结果在工作树 `artifacts/task150/smoke.json`（不入 git），这是接口冒烟，不代表胜率。相关定向测试合计 24 passed，全部先看内存、`ulimit -v 8000000`、只选 `-k` 且 pytest 串行；没有整套测试或额外全量校验。

## 后台缓存预构建（持续运行）

入口 `scripts/precompute_timer_cache.py`，使用新 A 配方所指的共享训练数据和缓存根；覆盖所有现有档案，A/B 共用同一份特征缓存。原缓存只有 v2，因此先生成 v2sleep 视图，再只补四个 timer 通道；缺旧缓存的档案才走普通编码。不重建段、不训练、不修改控制器状态。

后台主 PID **16149**，nice **19**，**2 worker**；每进程 AS 8,000,000 KiB、DATA 2,000,000 KiB。较大的 AS 允许 libtorch 只读映射，DATA 单独限制数据分配。两 worker 加主进程 RSS 约 **1.25 GiB**，启动及观测时机器 MemAvailable 约 **14–15 GiB**。

- 数据：`/home/newbiexvwu/pvz-logs/curriculum_state.pipeline/frames`
- 缓存：`/home/newbiexvwu/pvz-logs/curriculum_state.pipeline/cache`（新增 v2sleep/v2timer 命名空间）
- 进度：`/home/newbiexvwu/pvz-logs/task150-precompute.progress.json`
- 日志：`/home/newbiexvwu/pvz-logs/task150-precompute.txt`
- PID 记录：`/home/newbiexvwu/pvz-logs/task150-precompute.pid`

2026-10-29 00:58 上海时间的进度为 **1,933 / 80,141**，该时点 ETA **03:13**。这是早期吞吐外推，随目录、档案大小和生产负载变化；实时文件每 30 秒更新。进程与两个 worker 已实际存活并产出缓存，尚未全量完成。任务书末尾另有交付进度；缓存完成后仍需补记最终结果。

本次只准备，没有执行部署，未操作 tmux qc、生产状态文件、看门狗或现役二进制。
