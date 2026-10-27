# task132：示范行加权的配方选项（只做代码，实验由主控安排）

主控 2026-10-27 派给台式机 codex-t125。分支 `demoweight`，基于 `origin/pvz-env`。

## 背景

- 课程里来自用户示范起点格的对局（自 r13 起 4440 / 30750 局，雾 / 屋顶为主）和普通课程局在训练里同权。示范本身的行（`demo_*` 任务的人类操作）也只是普通行。
- 想测“示范行加权”能否在不增加采集的前提下补上屋顶 / 雾的短板。门禁 v5（deploy14i）上线后由主控安排实验；本任务只把开关做出来。

## 要做的

1. 训练配方加一个可选项（例如 `demo_row_weight`，默认 1.0 = 现状逐位不变），对来源为用户示范的行（人类操作行）乘权；再加一个可选项对“从示范起点格开局”的课程行乘权（默认 1.0）。行来源靠打包时已有的元数据判定；没有就在打包里补一个来源标记（不改旧缓存签名的默认路径）。
2. 与现有的关卡平衡 / spread 权重的组合顺序写清楚（乘在哪一步、是否参与 cap）。
3. 小测试：默认值下打包张量与权重逐位不变；加权时只有目标行权重变化。
4. 给出一个实验配方草稿（`recipes/` 下新文件，不改旧配方），例如示范行 ×4、示范起点格行 ×2。

## 约束

- AGENTS §6.5 第 7 条：台式机测试只跑相关单测，`ulimit -v 8000000`，先看 `free -g`。不训练、不部署、不碰 qc / 状态文件 / 看门狗 / `~/pvz-logs/qc_deploy*.sh`。
- 结论写在本文件末尾，推送分支 `demoweight`。


## 执行结论（2026-10-28）


分支 `demoweight`，基于 origin/pvz-env 979b784b；工作树
`/home/newbiexvwu/PvZ-search/.claude/worktrees/demoweight`。

两个配方选项默认均1.0：

- `demo_row_weight`：meta.json 的 label_source=human，人类实际操作行。
- `demo_start_row_weight`：label_source=teacher（含历史缺省）且 demo_archive 非空，示范起点课程续打行。

不靠任务名或目录名判断；LLM示范、反事实行与没有示范起点的普通课程行不会误乘。
已有包保留source/source_archive/source_row，因此不需要新来源标记或重打包。
默认不增加元数据I/O或浮点运算，pack_options、特征缓存和共享合并签名均不改。
启用后按既有逐存档行回调读取小元数据，与关卡计数共用一次读取，没有全量帧解压或重放。

组合顺序为wait保留补偿/label_weight、source_weight、packet_balance、spread、row_weights、
关卡平衡，最后再乘本任务两个来源倍数。两个倍数不参与此前的factor cap或level share cap，
也不重新归一化。因此最终关卡份额可高于原cap；level_shares.json仍描述加权前的关卡平衡阶段。
验证行同样乘权，行顺序、seed切分和全部非权重张量不变。

实验草稿 `/home/newbiexvwu/PvZ-search/.claude/worktrees/demoweight/recipes/dw4i2_rebalanced_p05_demo4_start2.json`，
仅增加人类×4、示范起点×2并改输出与说明，旧配方原样保留。由主控安排数据和实验。

先看free -g，ulimit -v 8000000、单线程下3个直接相关小测试通过（1.91秒）：
单文件包与分段包均验证默认张量/权重逐位相同、加权只改变指定来源行，并覆盖两个开关独立使用、
历史teacher缺省、错误参数和容易误判的demo_*名称。没有启动训练或游戏，没有部署，
没有读写QC状态或看门狗，没有改qc_deploy脚本。
