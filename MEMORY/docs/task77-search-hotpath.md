# task77：老师搜索的热路径——每个决策的耗时花在哪，能不能逐位不变地再快一截

> 主控 2026-10-25 写。执行者：台式机 Codex（Plus 号，新开 tmux `codex-perf`，`CODEX_HOME=/home/newbiexvwu/.codex-plus`）。工作树自建 `/home/newbiexvwu/PvZ-search/.claude/worktrees/perf2`（分支 `perf2`，从 origin/pvz-env 拉出）。游戏进程上限原为4，主控因内存紧张明确改为同时最多 **2 个，qc 优先**；**不碰 tmux `qc`、cq、build4、流水线**。

## 为什么
- qc 的产出（每小时多少局老师标签）直接决定学生学得多快。
- task35 优化过引擎 CPU，task51 测得采集吞吐约 1.8 倍。但老师搜索（fast_search：每卡最优格 + halving + 9 个未来复核 + rollout 2500 tick，学生前向）仍是每局耗时的大头。
- 之后又加了引擎内叶值（task37）、v5 叶值和 build4，热点可能已经变了。

## 要做的
1. **剖析**：用生产命令形态（`curriculum_queue.command_for`，deploy4 引擎，v5，台式机 branch1，现役学生），在白天 L7、夜间 L14、泳池 L24、雾 L31、屋顶 L41 各跑 4 局。用 perf / 采样剖析加 Python cProfile，给出每决策耗时的分解：rollout 引擎步、学生前向（C++ 内）、叶值前向、Python 调度与 JSON 往返、快照分支。
2. **改进**：挑占比最大的 2–3 处做优化，要求老师输出**逐位不变**——同种子同局动作序列、标签、胜负完全一致，照 task70 A2 的门禁脚本。不改搜索算法本身（候选数、未来数、视界、门槛都不动）。
3. **验收**：六族 × 8 种子，改前改后逐位一致；ABBA 配对计时与 qc 同负载，报每局耗时比与区间。
4. 结论写 `MEMORY/docs/search-hotpath.md`，提交并 `git push origin perf2`。部署由主控做（重建 PGO 引擎）。

## 规则
- 只用绝对路径；不 `cd` 进别的工作树；`PVZ_NO_POPUP=1`。不加开关，不改生产默认。
- 禁止 `git add -f`；`.pt` `.npz` `.log` 不进 git；不写字面密钥；不手写打法规则或公式。
- 每完成一小步就提交、push，在"结果"记一行；疑问写"问题"，按最佳判断继续。

## 结果

- [0] 已冻结生产 deploy4（16989f23）、在役 round11、v5、台式机 branch1 与六族清单；添加独立剖析工具，启动五地形各 4 局。细节见 search-hotpath.md；原始数据仅留 ignored artifacts/task77，不操作 qc/cq/流水线。
- [门禁工具] 补齐六族×8种子的前后对照（每族前4老师 β=1、后4学生 β=0，均搜索打标签），比较完整帧、动作、标签、软目标、复核及 task70 逻辑统计；补16局 ABBA 配对计时与确定性 bootstrap 区间。5项证据回归测试通过；真实前后验收尚未开始。
- [优化1] 叶值/静态值卷积：首层核打包按通道集缓存（候选分支通常只改数值、通道集不变）+ 无填充 memcpy 快路径（先清零再按偏移整段拷贝，替代逐格越界判断）；`DataArray` 迭代器取址改走 `get()` 并强制内联；`Reanimation::Update` 把 `gEnvSimSkip` 判断前置，省掉 rollout 每 tick 的无谓浮点加载。逐位一致由离线核检查 `scripts/task77_value_kernel_check.py` 证明：value/static 各 480 个 logit（三遍合计，含换权重、全零、±0、稠密、稀疏通道），before/after SHA256 完全相同；候选引擎已按生产 flags 重建（build-candidate.log）。
- [剖析补采] 首批 day/night 剖析缺引擎计数器（旧版 driver），`profile` 阶段改为不覆盖证据、另存 `profile_complete/`，按同一任务与种子重采；`profile-summary` 汇总每决策毫秒分解。
- [分解] 五地形各 4 局剖析完成：每决策耗时里 rollout 引擎步 41.8–91.3 ms、C++ 学生前向 14.8–47.8 ms，二者占 76–86%；叶值+静态值前向 1.0–6.9 ms、显式快照往返 0.5–0.8 ms、恢复/IPC 与 Python 调度约 8–20 ms。BLAS 平面样本无法单独归给学生；样本格式审计与最终采样热点以 search-hotpath.md 为准。
- [计时口径] ABBA 的 A 臂改为显式传同 flags 的 `pvz-portable-release-before`：生产 deploy4 带本机 PGO（历史实测 18%），拿它当 A 会把编译差异算进补丁。`abba_baseline()` 拒绝静默回落到生产二进制，并加了一条回归测试（6 项通过）。
- [门禁通过] 六族×8种子（老师 4 + 学生 4，均搜索打标签）端到端逐位一致：`gate_evidence.json` pass=true，48/48 对局；比较含 task70 COMPARE_FIELDS、rollout/halving/student_candidate 计数与完整决策帧归档。旧臂老师 20 局复用剖析局，不重打。候选 `b4999d65…`，报告 §7.1。
- [计时进行中] 按主控新约束（同时最多 2 个游戏进程）以 `--workers 2` 跑 ABBA（A=同 flags 改前二进制 `51949eff…`，B=候选），24 个 block × 4 臂；结果补进报告 §7.2。
- [交接 2026-10-25 14:57] 按主控指令（Codex 额度恢复）停在这里。已完成：冻结、五地形剖析与分解（报告 §3–§4）、三处逐位不变优化（§5）、离线核检查、48/48 端到端门禁（§7.1）、报告主体。**未完成：ABBA 配对计时**——`--workers 2` 跑到 3 个 block（day 的 A、B、B 各差最后一臂）即中断，没有 `abba_evidence.json`；`abba()` 无断点续跑，下一轮按报告 §2 的命令整轮重跑（约 70–90 分钟），A 臂务必显式传 `--baseline …/pvz-portable-release-before`。已停止本任务的全部游戏进程。
- [ABBA 重跑] 按主控 2 进程上限给 `abba` 加 `--count`（每臂局数，也即并发游戏数），以 `--workers 2 --count 2` 六族重跑；结果与逐位门禁一起写进报告 §7。
- [接续审计] 重新核验门禁为48/48策略与完整帧一致，state_hash实为35同/13异，报告已纠正。旧计时两轮均未完成，本轮在全新 `artifacts/task77/abba_final` 独立跑六族×2种子×ABBA，拒绝拼接历史结果。补严格四臂配置、原始耗时和完整帧审计；采样导出冻结计数，读取拒绝错位数组；8项证据测试通过。坏泳池样本1份保留并排除，待补采。
- [核检查补强] 换模型前最后输入与换模后首输入均保持全部通道活动，实际覆盖缓存失效场景；value/static各480个logit仍逐字节一致（9762c586 / f4f8f4a0），证据另存 value_kernel_check_final，不覆盖旧核检查。
- [验收审计补齐] 新增 `recover-samples`（拒绝与本任务引擎并跑；补采后校验原局完整帧）及按族整体重采样的区间。修正 qc 负载监测只精确匹配 Python 控制器 argv，不再误计 tmux 整棵树；本轮另有只读负载监测。ABBA 当前完成白天，夜间在跑，尚不作提速结论。
- [基线复现] `prepare` 固定从原提交c0249d62提取源码，不再取优化后的当前树充当“改前”；本轮原冻结和新提取的4个引擎文件及搜索驱动均逐字节吻合原提交。
- [负载分层] day/night四臂的全部8个边界均有同一真实qc控制器，单独计算qc正常运行期4对ABBA；完整六族计时仍全部保留。新增回归测试排除tmux祖先误命中和中途停机窗口，10项证据测试通过。

## 问题

- redeploy/build 已由 task70 B5 重建，不能当原 deploy4；正式基线只读复制生产 cq/build4（SHA 与 task70 deploy4 一致）。系统无 perf，采用现有 SIGPROF 采样器并结合 cProfile。
- 本轮计时期间原qc控制器退出，已只读核实无课程控制器在跑并向主控询问负载安排；继续保留实际计时，不操作qc，也不把无qc窗口计作恒定qc负载。
