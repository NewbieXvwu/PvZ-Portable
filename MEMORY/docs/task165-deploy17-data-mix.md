# task165：deploy17 数据配比准备

## 基线与授权范围

生产现行 deploy16，cq 工作树安装源 REV `ba257927`；从该提交建立独立 `deploy17` 分支/工作树。
不复制 cq 的未提交改动，不从 cppnet 分支继承 token 内核或模型变更。
依据 `origin/t160memo:MEMORY/docs/task160-decision-memo.md` v1.2 §0、5、6、8，
以及 `origin/datamix:MEMORY/docs/task160-d-production-profile.md`。
用户本次明确将 B 定为 E1 的1.0权重对照，优先于备忘中后续 E13 卡组签名平衡方案。

只准备代码、配置、卡池与部署/回滚脚本；不执行部署，不操作正在运行的 qc、看门狗、stop_curr
或生产二进制，不 pkill。提交推送 deploy17 后停下等主控审。

## 实施步骤

1. 合入 `origin/datalevers`（task159）：默认关闭的确定性整局降采样，以及GPT Pro胜局/屋顶候选卡池。
   本次不启用 label_keep，保持 A 全部训练数据与参数不变。
2. 采集改为示范起点格局数上限25%，开局面板保留25%；腾出的预算分给随机卡组与倍率格，
   标准768局目标为四类各192局。按局数约束，不以不同大小的组数代替份额；格子不足时避免超示范上限。
   复用现有种子偏移、轮转与格子状态，报告实际生产状态下的规划前后分配。
3. 冻结扩充后的随机训练卡池，显式排除当前门禁、挑选、rd137评估与v6草案的留出多重集签名。
   只读小账本/meta与JSON面板，不重放存档、不加载全量帧；新卡组族登记DESIGN。
4. A沿用deploy16 `dw4i2_v2timer.json`；B从A复制，唯一实质差异为33个历史`z*`标签的
   label_weight 1.5→1.0，不附加关卡平衡/抽样/网络结构变化。
5. 参考 `~/pvz-logs/qc_deploy16.sh`、`deploy16_at_boundary.sh`，准备版本化的
   `qc_deploy17.sh`与边界入口：主控审后组边界切换，变更前备份代码/配置/启动参数，失败恢复；
   提供一条命令回滚，保留控制器状态、在任模型和种子账本。脚本只写，不调用生产服务。

## 验收与资源限制

只运行直接相关的单个 `pytest -k` 项，先 `free -g`，使用 nice10、`ulimit -v 8000000`与timeout；
不编译游戏、不训练、不全量打包、不回放存档。壳脚本仅语法检查/隔离夹具测试，不执行真实部署。
最终在本任务书追加文件清单、每轮各类局数前后对照、卡池/留出排除结果及准确的回滚命令。

## 当前进度

任务书先于实现建立；待完成后追加结论。

## 实现结论（2026-10-29，准备完成，未部署）

基线为`ba257927`，依赖快进合入`origin/datalevers`的`fda65c7d`；任务书先提交为`d1387474`，
数据与运行时代码提交为`ae739b1c`。部署脚本明确固定安装源`REV=ae739b1c`，继续使用
`prodint16/build16/pvz-portable`。未继承cppnet的token内核、未更换模型/引擎/门禁口径。

### 采集分配前后

生产状态只读快照：round47，updated=`2026-10-29T18:51:35`，342个活跃格子；
规划下一完整轮时只在内存副本去掉本轮completed_groups，没有落盘、续领种子或重放存档。
生产任务表、老师胜率证据与现有offset均用于规划，结果见
[task165-deploy17-allocation.json](task165-deploy17-allocation.json)。

| 类别 | deploy16局/轮 | deploy17局/轮 | 份额变化 | 新配置组数 |
|---|---:|---:|---|---:|
| 开局面板 | 192 | 192 | 25%→25% | 16×12（两种β） |
| 示范起点 | 576 | 192（上限） | 75%→25% | 8×24 |
| 随机卡组 | 0 | 192（预留/上限） | 0→25% | 8×24 |
| 倍率格（既有日/夜/泳池/雾/屋顶） | 0 | 192 | 0→25% | 8×24 |
| 合计 | 768 | 768 | 100% | 40组 |

新增`--demo-start-share`，默认None时完全沿用旧规划器；deploy17启动才显式设置它和
`--random-deck-share`均为0.25。先为随机卡组分预算，再限示范，剩余交给倍率格；
各类内部仍用既有last_round保底/可教差距排序。样本不足的随机或示范配额回流倍率格，
不为凑满轮预算突破示范上限；全体格子不足时可少采。

续跑按completed_groups扣掉已完成的各类局数和开局β组，不重复采已完成的格；
退休格的已完成局也计入上限。若组边界切换时旧配置已经采过192局示范，余下不再追加示范；
过渡轮的既成局数不会倒退，第一个完整新轮落实四类配比。既有offset与种子账本不回退。

### 卡池与留出

`configs/deploy17_random_deck_pool.json`冻结38套新增卡组，来源是本机task105真实合并账本
1578条记录里的已验证胜局meta，再合并task148候选。按植物多重集跨关去重后，38套均有GPT胜局
来源，其中2套也命中task148屋顶候选（`also_sources`保留双来源，代表任务按task159规则取合法较小关号）。
这里只读小账本/meta，没有读帧或重验胜局；源账本/候选JSON的SHA256保存在配置中。

| 地形 | 原rd137训练卡组 | 新增 | 合计 |
|---|---:|---:|---:|
| day | 4 | 6 | 10 |
| night | 4 | 1 | 5 |
| pool | 4 | 5 | 9 |
| fog | 4 | 12 | 16 |
| roof | 0 | 14 | 14 |
| 合计 | 16 | 38 | 54 |

派生训练面板为486个倍率任务（54×9档），约15.9MB，仅在忽略的`artifacts/task165/`，不入git。
原rd137的任务id、16套训练卡组/8套评估卡组和种子段不变；新增复用task159的rd159族与独立种子段，
冻结顺序避免部署时因账本增长重排种子槽。新族已在DESIGN登记。

`configs/deploy17_deck_holdouts.json`是当前生产v5门禁、挑选/挑选专用留出、rd137评估和v6草案
的**训练排除快照**，共26个不同多重集签名；不替换任何评估面板。
`--pool-registry`只读冻结池，生成时再与该排除快照、当前生产门禁/挑选文件做廉价集合比较。
54套训练与留出交集为空，v5地形/黑白名单/底座约束成立，升级所有权随任务profile生成。

### A/B配方与共包接入

A文件`recipes/dw4i2_v2timer.json`没有修改。新B为`recipes/dw4i2_v2timer_history10.json`，
与A的唯一字段差异是33个明确列出的z*历史标签权重从1.5变1.0；没有source_weight关卡平衡、
没有label_keep、没有初始化/训练步数/优化器/观测/网络/采样变更。保持原来的A种子1/2、B种子3/4，
同一轮冻结语料、同一12epoch预算与挑选/门禁。

原管线会拒绝A/B标签权重不同，而且A的标签权重已烘焙在不可变包中，不能只改JSON后仍加载旧权重。
现在允许只在label_weight/source_weight不同的候选共享包；B训练命令附`--pack-recipe <本轮A快照>`。
按包已有source/source_row顺序记录存档跨度，加载后在训练进程的**私有weight向量**上乘
`B_label_weight/A_pack_label_weight`，本次33标签为2/3，其余为1。
不改共享段/合并缓存，不分配全行倍率向量，不重新解压帧或重建特征包；计算仅为既有权重切片缩放。
`shared_pack_weights.json`记录包来源、SHA256、原/目标权重和实际倍率，检查点仍记真实B标签权重。
旧排队任务的冻结配方不改，新B只作用于部署后新入队的轮次。

### 文件清单

| 文件 | 作用 |
|---|---|
| `scripts/curriculum_queue.py` | 按类别预留/封顶、扣除已完成局；候选共包并向B传包配方 |
| `python/pvz_recipe.py`、`scripts/train_rollout_policy.py`、`python/pvz_training_rows.py` | 共包CLI与加载后私有标签权重调整 |
| `scripts/make_random_deck_cells.py` | task159基础上增加冻结池入口与留出集合排除 |
| `configs/deploy17_random_deck_pool.json`、`configs/deploy17_deck_holdouts.json` | 固定训练卡池与训练排除集合 |
| `recipes/dw4i2_v2timer_history10.json` | 单变量B配方 |
| `scripts/qc_deploy17.sh`、`scripts/deploy17_at_boundary.sh`、`scripts/deploy17_boundary.py` | 待审部署、边界屏障、备份与回滚 |
| `python/test_deploy17.py` | 小型规划/共包/CLI/脚本隔离夹具 |
| `DESIGN.md`、本任务书、分配JSON、日记/索引 | 任务族登记、分配结论与交付记录 |

task159随合入带来的`python/pvz_training_pack.py`、`scripts/pack_training_rows.py`与`scripts/deck_rules.py`
一并列入待安装依赖。确定性整局降采样代码可用，但本次两配方均未启用。

### 待主控审后的部署与一条命令回滚

脚本目前只在本分支`scripts/`，没有写入`~/pvz-logs`，因此不会被运行中的看门狗发现/切换。
主控审后才可按以下方式安装、执行（本任务没有执行这些命令）：

```bash
cp /home/newbiexvwu/PvZ-search/.claude/worktrees/deploy17/scripts/qc_deploy17.sh ~/pvz-logs/qc_deploy17.sh
cp /home/newbiexvwu/PvZ-search/.claude/worktrees/deploy17/scripts/deploy17_at_boundary.sh ~/pvz-logs/deploy17_at_boundary.sh
bash ~/pvz-logs/deploy17_at_boundary.sh
```

准备文件和生成卡池面板在暂停前完成，不重建特征语料。真正切换时短暂停住调度器，若前台采集子进程
仍在则立即恢复，继续等该组；仅在最后一组rc已记录且标签已进入持久状态后才保持屏障，防止下一组竞态。
只扫描日志末64KiB/新增行、当前状态里的标签字节串（本机约10MB），不做档案重放或模型自检；
检查仅用于防止丢失刚完成组的记录，后台训练/门禁不被当成“前台组未结束”而阻塞等待。

然后备份所有将改的代码/配置、生成面板、mode/active标记及旧`start_qc`完整启动函数，落
`~/pvz-logs/deploy17_<时间>_<pid>/backed_up`后才停止旧控制器、安装新文件、写active标记并启动。
只处理该调度器已记录的进程树，PID重用会跳过；不使用pkill，不替换生产二进制、不操作看门狗。
准备好的17脚本即使先被看门狗发现，active标记未写前仍按原deploy16参数启动。

一条命令回滚到最近一次完整备份（同样等组边界）：

```bash
bash ~/pvz-logs/qc_deploy17.sh --rollback
```

也可附备份目录选择具体一次切换。恢复原文件及“不存在”状态，再运行备份的旧启动函数；
成功回滚有完成标记，可重复调用。切换失败/被中断会恢复旧服务或解除未切换的暂停。
回滚的是代码/卡池入口/启动配置，**不回退控制器状态、已晋升学生、种子偏移/账本，也不删除已采的新标签**。
后台任务沿用deploy16的持久化恢复/重试语义，不把旧候选配方改成新B。

### 小测试与交付边界

相关`pytest -k`小项通过，均先看free、nice10、8,000,000KiB地址空间和timeout：覆盖实际25%分配/
部分轮续跑/退休格计数、留出隔离、B私有重标与小型实体B包的一致性、真实CLI在加载后截停（不创建
模型/优化器、不训练）、看门狗启动参数、临时文件备份/恢复与模拟PID边界屏障。脚本语法检查通过。
这些测试只用临时夹具；没有执行真实部署脚本，没有发信号给生产进程，没有改qc、看门狗、stop_curr
或生产二进制，也没有全量重打包/回放/训练。提交推送deploy17后停下，等待主控审。

## 主控批准后的切换观察（2026-10-29 23:29–23:37）

用户已审过并启动`deploy17-boundary`，授权观察，失败时执行既有`--rollback`。
23:29:52在round51的`c51opening119l28scoreboardm100_o696_b0`完成后切换，
该组12局、10胜、rc=0；脚本写入`qc_deploy17.active=ae739b1c`后返回0。
**这只说明安装脚本完成，不能作为新控制器已正常启动的证据。**

新控制器在`load_panel`退出，未开始新采集组。原因是
`scripts/make_multiplier_tasks.py:source_paths({})`的`day_orig`默认源仍指向
`/home/newbiexvwu/PvZ-search/.claude/worktrees/conveyor/artifacts/stage4/day_panel.json`，
该路径不存在；其余12项默认源文件存在。这是源清单注册的旧绝对路径依赖，并非新生成的
task165随机卡池里写入了该路径。错误记录在`qc.out`及看门狗23:29:59、23:31:59的crash尾档。

23:32:54按本次明确授权执行：

```bash
bash ~/pvz-logs/qc_deploy17.sh --rollback
```

回滚返回0，备份目录为`~/pvz-logs/deploy17_20261029T232921_268487`，有
`rollback.complete`，deploy17 active标记已撤销。恢复的deploy16同样在该缺失源清单处退出；
23:33:59看门狗记录`CRITICAL HALTED`（30分钟内已重启两次），当前采集没有恢复。
未再次部署、未改看门狗/stop_curr/生产二进制；已向主控报告并询问是否授权修复旧路径后恢复deploy16。

因此本次**没有deploy17第一个完整轮**；四类各192局、新卡池实际使用、B训练日志及
`shared_pack_weights.json`实产倍率均待成功切换后验收，不能用准备阶段的规划或小测试冒充。
当前round51为切换前已有的部分轮，恢复后的完整新轮仍须按实际运行配置认定。
