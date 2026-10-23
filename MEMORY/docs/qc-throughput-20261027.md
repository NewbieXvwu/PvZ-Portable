# 正式课程采集吞吐：10-27 决策证据（task51）

状态：进行中；尚无受控比值或 GPU 决策结论。任务书见 [task51](task51-qc-throughput.md)。

## 口径

- 采集段吞吐 = 本次实际新增且完成的有效局数 / 子命令墙钟小时。包含 Python/worker/引擎启动、搜索、标签生成与正式帧归档，不能用 task35 的引擎 CPU 秒或逐局 `seconds` 之和替代。
- 含轮间吞吐 = 完整轮次新增有效局数 / 从该轮首组开始至下一轮首组开始的墙钟小时。阻塞的打包、训练和轮间等待均计入；并发训练不再重复加一次时长。
- 分任务族 × 倍率列出，保留 worker 数、worker-hours、每 worker 的局/小时和可取得的 MemAvailable。worker 归一化用于解释并发变化，不把它当成独占 CPU 测速。
- 整体先固定共同任务组合与权重，再用总局数 / 总墙钟计比值；不将不同任务混合占比直接归因于优化。轮间共享开销若分摊到族，必须标明分摊口径；总体使用完整墙钟。
- 未完成的打包/训练周期保留右删失标记，只能给已知时长或上界，不以零时长补齐。缺少 worker、时间戳或版本证据的字段标未知。

## 已确认的历史统计陷阱

`curriculum.log` 的 `games` 来自整个标签目录的 `games.jsonl` 行数。控制器重启并复用标签后，新增 48 局会报告为累计 96、144、192……；`config.json` 也被后一次启动覆盖，不能拿最终 config 的 worker 数解释所有历史段。

实例：`c2n14n10m100_48` 目前 336 行/336 个唯一种子，最终 config 为 offset 514、seeds 48、workers 16。历史多次运行不能分别按 96/144/192/… 局计吞吐。优先使用 `qc.out` 每次启动的 `N games on W workers`、完成进度、控制器完成记录及真实唯一 seed 交叉还原；失败和无对应启动记录的段单列，不静默推断成功。

数据源为只读的 `/home/newbiexvwu/pvz-logs/curriculum.log`、`qc.out`、`metrics/`、历史 queue 日志及状态/逐局记录。正式 qc 不修改、不重启。

## 版本与受控对照

旧版已锁定为 `1b89bfa92554602ff83a44bd969de7ac532894ba`：cq reflog 记录 10-22 19:45:33 快进到此提交，commit count 为 **2074**，与当天 qc 输出 BuildNum 2074 一致。现存 cq 二进制在 10-23 已重建（当前头文件为 2242），不能冒充 10-22 旧二进制。

旧源码/Python 已在本树 ignored 目录冻结，旧引擎只编译本树 `build-old/`，Release、历史默认编译配置；生成的版本头单独记录冻结提交。现行 cq Python 冻结为 `26abefe09c790541046cc20a4344a2f8a995a5c9`，配 deploy2。task50 真正完成后再冻结 zones + deploy3，不以“二进制已经存在”代替其完成门禁。

控制学生固定 dw4i_s1、静态网 branch1、beta=0、verify-samples=9、student-candidates=2、workers=2，正式帧归档写入本树隔离目录。旧时期默认叶网 v3、现行时期 v4；这是实际方案的时代对照，模型 SHA 和参数均冻结，不能把差异单独归因于引擎。beta=0 的执行轨迹仍由同一学生驱动，核对逐种子胜负/末波/决策数。

四族为白天 L7 d7、夜间 L14 n10、泳池 L24 p9k、原版传送带 L10。每族每臂 16 局，固定同倍率、同任务定义与同 16 种子，命令从各自快照的 `curriculum_queue.command_for` 生成，保留归档开销。采用两块各 8 局的 ABBA/BAAB 顺序，控制整个任务同时最多 2 个游戏进程；所有输出独立标签，拒绝把已有局数混入计时。具体清单、模型/代码/引擎 SHA 与命令将在运行前冻结。

清单已冻结：四族均 **1.5 倍率**，原 multiplier panel offset64 起16种子，8局×两块，每臂每族16局，共128局；工具 [bench_qc_collection.py](../../scripts/bench_qc_collection.py)。旧引擎 SHA256 `dd03838136240fc4c1af36205ce3a8a4a486fc54ca5e1ba82ed23b796e36e2cb`，deploy2 SHA256 `bf1472f7b858c1badc4f1e81ed9398d2f7633da408499e0c6b10dff35bea4aef`。命令、种子与全部输入 SHA 在 ignored `artifacts/task51/collection-plan.json`；6项证据门禁测试通过。每条子命令记录墙钟、CPU、MemAvailable/RSS/load，等待采用进程终止事件，不把2秒遥测轮询间隔加成尾部耗时。

## 结果

历史有效局数还原、受控采集及完整轮间比值待补齐。判据保持 ≥2.0 / 1.5–2.0 / <1.5，不因现有单项基准提前判档。
