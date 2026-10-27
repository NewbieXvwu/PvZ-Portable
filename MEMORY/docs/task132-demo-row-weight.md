# task132 交付（2026-10-28，仅代码）

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
