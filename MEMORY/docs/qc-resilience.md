# qc 异常隔离与看门狗（task98）

任务书：[task98-qc-resilience.md](task98-qc-resilience.md)。审计基线：e9515a40。
当前阶段：审计、控制器修复及看门狗交付完成，尚未上线；下表列的是原实现及分类。

## 主循环可达异常审计

| 入口 / 路径 | 显式 raise 或隐式异常 | 分类与处理方向 |
|---|---|---|
| argparse、load_recipe、绝对路径/worker 校验、freeze_panel | parser.error→SystemExit；文件、JSON、字段、配方/门禁版本错误 | 配置错误；启动失败，保留严格校验 |
| load_panel→source_paths/load_tasks | read_text、JSON、KeyError、TypeError、倍率 float | 必需清单/源清单错误；启动失败；IO 重试一次 |
| load_demo_start_cells | 阶梯顺序/重复、缺种子块、缺存档/未验证等五个 ValueError；元数据 IO/JSON/类型异常 | 配置错误；启动失败，不把未验证示范放进课程 |
| 主状态读取、TrainingPipeline.__init__ | 状态 JSON/顶层结构、旧学生缺失 RuntimeError；历史测量 int、历史 job 字段、目录 IO | 全局结构/学生错误启动失败；单格与单 job 隔离，IO 重试一次 |
| init_from_runs | 读 config/games、JSON 已捕获；后续 float、行类型、种子/字段类型未隔离 | 单个历史 run 跳过，错误不能污染窗口；不能再按目录名 int 解析种子 |
| reconcile_cells / reconcile_start_cells / make_state | cell.get、level/index/int、窗口/阶梯类型、排序字段等 | 单格错误隔离到 retired_cells，带原因保留；全局清单错误仍启动失败 |
| plan_round→adjust/allocate→cell_level/rate/student_rate | 坏 mult_index/窗口产生 IndexError/KeyError/TypeError/ValueError | 单格校验/退休后再分配，避免坏格影响所有排序 |
| plan_round→task_id_for | 缺倍率任务显式 KeyError；manifest/seeds 等字段错误 | 单格退休并记录，其他格继续 |
| plan_round 种子耗尽 | 基线已改为示范退休/普通跳组（不再 SystemExit） | 保留不回绕原则；补测试余量不足、恢复与其他格继续 |
| startup_worker_estimate→measure_worker_bytes | probe timeout / 非零退出 RuntimeError，Popen/临时目录 IO；已有回退 | 保留保守 RSS 回退；IO 重试一次，不终止课程 |
| run_round→command_for/subprocess.run | 单组命令字段、Popen OSError、子进程失败显式 SystemExit | 先持久化偏移；组错误记失败并继续，启动 IO 重试一次，不把失败计为胜负 |
| run_round→games.jsonl/窗口统计/metrics | 缺文件、JSON、坏行/数字、Metrics.update 等 | 该组记录错误、跳过；可选监控错误不阻断采集；已占偏移不退回 |
| log_line / save_state / 状态目录 | mkdir/open/write/replace 的 OSError | 重试一次；日志降级 stderr，状态仍无法持久化时等待恢复，禁止无持久偏移开下一组 |
| load_teacher_rates | 读/JSON 已回退；根类型/items、wins/games/float 未捕获 | 可选证据；单项跳过，IO 重试一次后降级为未知，不改胜率为伪造值 |
| pipeline.reservation / _plan_build / pending_builds / enqueue | recipe/文件指纹/stat/link IO、类型异常、缺标签；write_recipe；_seeded→SeedPoolExhausted/账本错误 | 仅失败该训练快照；保留原学生、禁止无门禁晋升，IO 重试一次 |
| _prepare_corpus | 缺标签、固定语料变更、目标不是符号链接三个 RuntimeError | 拒绝该快照，绝不缩小语料冒充成功 |
| _command / _admit / _run | 子进程非零、内存需求不可能满足、先前失败、冻结模型/配方改变、未产模型等 RuntimeError/ValueError；后台 BaseException 重抛 | 单后台任务最多两次尝试后保留失败，采集继续；不能串联毒化后续任务 |
| _evaluate / evaluation_contract / valid_certificate | contract 指纹/JSON/字段/IO；已有 try 只罩住命令和验证、罩不住前面的 contract/_admit | 整个门禁错误关闭候选晋升，不停止前台；不绕过证书验证 |
| poll / stale comparison / close | future.result 异常；原二次失败 RuntimeError；重新分配种子/证书/complete.json/保存错误 | 隔离失败 job、继续其他 job；关闭时不反复处理已失败项 |
| print_plan、主 for 循环、恢复 enqueue、finally close | 上述漏出的 Exception/SystemExit | 运行边界兜底记录并继续；KeyboardInterrupt/终止信号不吞；全局状态 IO 失败先等待后保存 |

操作系统强杀、解释器退出不属于可捕获单格异常，由看门狗处理。持续不可写的状态盘不能靠跳格解决，必须停止开新局并等待恢复。

### 基线显式退出逐条清单

行号固定指向e9515a40的`curriculum_queue.py`。该文件没有`sys.exit()`调用；全部显式raise/argparse退出如下。

| 行 | 路径与错误 | 分类 / 当前处理 |
|---|---|---|
| 245 | measure_worker_bytes：探针超时 | 单探针，回退保守RSS |
| 249 | measure_worker_bytes：探针子进程非零 | 单探针，回退保守RSS |
| 427 | load_demo_start_cells：阶梯非降序 | 配置，启动失败 |
| 429 | load_demo_start_cells：阶梯重复 | 配置，启动失败 |
| 435 | load_demo_start_cells：缺未来种子块 | 配置，启动失败 |
| 441 | load_demo_start_cells：缺示范存档 | 配置，启动失败 |
| 443 | load_demo_start_cells：示范未验证 | 配置，启动失败 |
| 619 | task_id_for：缺倍率任务 | 单格，退休并记录 |
| 729 | TrainingPipeline.__init__：已登记学生丢失 | 配置，启动失败 |
| 798 | _evaluate：门禁证书不匹配 | 单候选，关闭晋升 |
| 923 | _prepare_corpus：标签缺档案 | 单快照，拒绝缩小语料 |
| 930 | _prepare_corpus：固定语料来源变更 | 单快照，拒绝覆盖 |
| 932 | _prepare_corpus：目标不是预期链接 | 单快照，拒绝覆盖 |
| 954 | _command：后台子进程非零 | 单快照，两次尝试后隔离 |
| 963 | _admit：内存需求超过物理总量 | 单快照，两次尝试后隔离 |
| 983 | _run：此前后台失败导致本轮也失败 | 删除串联毒化，快照独立处理 |
| 1003 | _run：冻结的已训练模型/配方变化 | 单快照，拒绝复用 |
| 1037 | _run：训练未产出模型 | 单快照，两次尝试后隔离 |
| 1049 | _run：BaseException重抛给future | 保留，前台poll处理该future |
| 1075 | poll：后台累计二次失败抛停qc | 改为隔离快照，继续采集 |
| 1260 | run_round：采集非零触发SystemExit | 改为记录失败组并继续 |
| 1346 | main：训练配方校验失败 | 配置，parser.error保持 |
| 1348 | main：worker非正数 | 配置，parser.error保持 |
| 1354 | main：路径不是绝对路径 | 配置，parser.error保持 |
| 1356 | main：种子账本不是绝对路径 | 配置，parser.error保持 |
| 1363 | main：晋升面板不符合协议 | 配置，parser.error保持 |

## 看门狗交付约束

只交付仓库脚本与 `~/pvz-logs/qc_watchdog.sh`，不启动监控、不调用真实 tmux 或部署。
每两分钟检查、三十分钟无日志告警；只对死亡会话重启，活但无进度不强杀。
最近部署脚本按修改时间选择，只提取 `start_qc` 函数及其所需赋值，不执行部署主程序。
停止标志、部署进程、单实例锁及三十分钟两次重启熔断均已由隔离测试覆盖。
归档保存`qc.out`最后50行，名字为`qc_crash_<时间>_<随机>.out`；动作写`qc_setup.log`。
重启次数保存在`qc_watchdog.restarts`，失败熔断后写`qc_watchdog.halted`，不会因看门狗自身重启而清零。
主控修复原因后可移走halted文件；旧重启记录仍按30分钟窗口约束。预算损坏时拒绝重启，不冒充空账本。
`curriculum.log`按新行数跟踪；单纯touch不会假装有进度。部署/停止条件在提取启动命令后及真正启动前再次检查。

仓库与本机脚本均有执行权限。由主控在组边界上线控制器后，再启动`/home/newbiexvwu/pvz-logs/qc_watchdog.sh`。
本任务没有启动看门狗、没有执行任何真实tmux命令或部署脚本。

## 验证与交付

控制器实现：坏格在adjust/allocate之前隔离，组失败保留偏移并继续，失败组不纳入训练标签；历史run与可选老师证据逐项隔离。
IO重试一次；状态写仍失败时主循环等待120秒，恢复保存成功后才允许新采集。KeyboardInterrupt不吞。
后台快照最多两次尝试，失败关闭该候选晋升并保留原因，后续快照继续；重启可恢复只有一次失败的快照。
门禁池耗尽关闭该快照，不回绕种子；丢失future的已登记queued任务在poll恢复。

补齐坏退休格、坏历史job、门禁账本IO和孤立queued任务测试；示范身份按archive区分，带倍率阶梯的普通L25保持活跃并走普通任务种子。
采集即使返回0也要有完整、合法的逐局结果；缺结果的组记失败，避免空标签污染后续打包。
隔离看门狗测试执行真实shell逻辑，tmux/pgrep/date由私有目录里的假命令替代；最新真实部署脚本只提取并`bash -n`，从不执行。
最终99项通过、1项跳过、4个子断言通过；包含23个异常隔离用例和13个看门狗用例。输入指纹见[qc-resilience-evidence.json](qc-resilience-evidence.json)。跳过项是原课程测试的真实示范存档检查，本工作树没有这些数值制品。
测试未启动游戏、未操作qc或cq。部署由主控在组边界执行。
