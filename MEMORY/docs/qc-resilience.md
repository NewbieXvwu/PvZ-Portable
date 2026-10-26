# qc 异常隔离与看门狗（task98）

任务书：[task98-qc-resilience.md](task98-qc-resilience.md)。审计基线：e9515a40。
当前阶段：完成调用链审计，尚未上线；下表列的是原实现及处理方向。

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

## 看门狗交付约束

只交付仓库脚本与 `~/pvz-logs/qc_watchdog.sh`，不启动监控、不调用真实 tmux 或部署。
每两分钟检查、三十分钟无日志告警；只对死亡会话重启，活但无进度不强杀。
最近部署脚本按修改时间选择，只提取 `start_qc` 函数及其所需赋值，不执行部署主程序。
停止标志、部署进程、单实例锁及三十分钟两次重启熔断均须由隔离测试覆盖。

## 验证与交付

实现和故障注入测试进行中。部署仍由主控在组边界执行。
