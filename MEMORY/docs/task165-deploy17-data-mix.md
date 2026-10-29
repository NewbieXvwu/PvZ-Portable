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

## 删除工作树后的紧急恢复（2026-10-29 23:57起，跨至10月30日）

用户确认停机来自清理磁盘时删除conveyor工作树，随后明确授权核对原任务、重建依赖、清除
看门狗HALTED、重新部署deploy17，并检查cq/prodint16/obsup/deploy17的代码、配方与生产状态JSON。

### 原版白天面板恢复

主控复制来的面板MD5为`54a4c6f8863274766d432fc6fe81e161`，26项，**缺少生产已有的
`day_l10_orig_train`**，因此没有直接沿用。生产状态有该格、level=10、原版段offset=306，
生产倍率清单保留了六档该任务的完整配置。恢复的conveyor `cc74fcc3`保留原生成器，
dw4冻结的`dataset_inventory.json`还保存了256份原版档案的种子和帧/meta SHA。

状态本身没有完整卡组/种子数组，不能单凭offset声称反推出原文件全部字节。本次用上述
生产证据与原生成器重建11项面板：

| 任务 | 关卡/卡组 | 开局 | 种子 |
|---|---|---|---|
| day_l10_orig_train | L10 / [] | conveyor_start=original | 1880000..1888191，共8192 |
| day_l10_orig_eval | L10 / [] | conveyor_start=original | 30000..31023，共1024 |

训练任务除task_id、倍率和种子字段外，与生产六档倍率任务逐字段相同；offset306保留，
继续对应1880306，不改名、不换卡组、不回退种子。重建面板MD5为
`8e5298b327d32c6130f15ef0664b4eed`。这是语义恢复，不宣称与已删文件逐字相同。
用户复制来的文件及恢复依据备份在`~/pvz-logs/task165_recovery_20261029T235917/`。

### 重部署和实际采集

23:59:18重新执行已批准的`qc_deploy17.sh --at-boundary`，写入active=ae739b1c；
实际控制器参数包含历史权重1.0的新B、task165扩充卡池、demo/random share各0.25、
gate_v5、A/B各两候选、训练并行1，仍使用`prodint16/build16/pvz-portable`。
HALTED与旧重启预算先备份再清除，看门狗进程1104保留；未改stop_curr或生产二进制。

启动时另发现`pending_builds`对约1600个标签逐一遍历全部档案目录，重复估算同步挡住采集。
局部修复`ad4fe41c`将目录一次按标签分组，保持采集覆盖优先级、排序、原文件大小、
段指纹与命中规则。直接相关的目录合并/段合同`pytest -k`测试1项通过（0.12秒），
先看free、nice10、8,000,000KiB与timeout60；没有全量打包或重放验收。

00:10:14实际恢复round51的`c51opening119l28scoreboardm100_o708_b1`，15个worker；
00:12:51该组12局完成，11胜、rc=0、157秒，确认采集实际恢复。
随后在该组完成且持久化的边界安装启动修复，控制器重启为PID274855，active改为
`ad4fe41c`，部署脚本固定REV同步改为该提交；只替换控制器Python，不部署gateunits。
这次局部更新的旧代码/脚本/active与PID边界记录在恢复目录的`startup_fix/`中，
临时重启保护HALTED已清除。正常一键回滚仍为`bash ~/pvz-logs/qc_deploy17.sh --rollback`，
使用23:59的新完整备份，保留状态与已完成局数。

### 其他删除路径：已恢复与仍缺原始数据

完整逐文件/行号命中、状态JSON字段位置、目标存在性和256份丢失原始档案的SHA清单见
[task165-deleted-paths.json](task165-deleted-paths.json)。四工作树代码/配方直接命中127行：
conveyor 28、dw5 84、deploy3 12、deploy15 3；扩展同一行匹配另有4条仅在forks报告文件名中
出现deploy3的相关行。读取280份顶层状态/备份状态和各轮冻结配方/完成报告/挑选输入JSON，
记录132条命中，包含旧已完成合同与Agent池的历史控制台文本；后者明确标为console_history，
没有把它们当生产依赖去恢复全部退役Agent。

| 路径/工作树 | 处理结果 |
|---|---|
| conveyor | 主控已恢复cc74fcc3；原版面板按上文重建 |
| dw5 | 恢复7974e324 |
| deploy15 | 恢复9f1db5d9；旧部署脚本的源码引用可解析 |
| deploy3 | 原来为detached而非分支；依据原部署报告恢复2b743c62源码 |
| v6 | Agent池配置仍引用其路径/任务书，恢复535677ff |
| dw5/task27/unified_v2.json | 重建34项，SHA256 a03243d6f7f55f30e4fdc500e06e36e9c44fb95357d976d82104edac532f862f，与生产历史合同完全一致 |
| dw5/task27/training_s1.command.json | 从冻结study-plan重建，SHA256 4ccc95bab86ea07cd0d5cc28f54c98ac30e7bcd4307d54698dd57d08f0847455，与配方provenance完全一致；study_plan.json亦恢复 |

**仍缺原始备份，不能声称全部恢复：**conveyor/task20/frames中的d10orig1/2各128份，
共256档案、512个frames.npz/meta.json文件。dw4/task23和生产训练视图虽然保留目录，
内部文件只是回指已删conveyor原文件的软链，不能当副本。曾检查的本机HF缓存/当前transfer
没有这批原帧；已向主控询问Mac或磁盘备份。生产后台打包已明确以缺少
`seed_1880000/d10orig1_fast_search/frames.npz`失败，采集可继续，**训练/B实产核对仍受阻**。
没有删掉这些训练标签、替换数据或用空文件蒙过指纹。

旧dw5实验还缺模型、完整训练视图、Mac导入账本、source_rows与task36完整结果；
旧deploy3 PGO二进制也随清理丢失，现存二进制均不匹配其历史SHA。
Git能恢复源码，无法据历史SHA反推这些二进制/对局数据；没有用现役模型/引擎冒充旧证据，
没有为修复历史审计路径重新训练或重跑旧实验。缺失项逐路径列在JSON附件中，待原备份恢复。
现行采集使用prodint16/build16和gate_v5，以上旧模型/旧门禁合同不在当前采集调用链；
丢失的256份原帧则属于当前训练语料，必须补回。

启动修复后的观察：00:12:51重启，00:14:31已结束恢复队列估算并进入单局RSS探针；
修复前对应区间为23:59:18至00:08:58。两次观察的缓存状态不同，这只是恢复过程的耗时记录，
不是交错性能基准。估算错误仍显式保留，未绕过缺失原帧的数据完整性要求。

最终恢复确认：00:17:11新控制器已完成RSS探针（单worker估计0.86GiB），进入
`round51 c51opening119l29scoreboardm100_o696_b0`，16/16 worker、MemAvailable约20GiB。
`qc_deploy17.active=ad4fe41c`、HALTED不存在；采集已恢复并推进，后台训练仍因上述原帧缺失失败。
恢复代码及完整清单已推送deploy17（ad4fe41c、a63d0f61），未将gateunits部署到生产。

## 主控决定放弃原档案、去除工作树依赖（2026-10-30，准备中，未部署）

用户明确：没有备份，停止找备份；放弃task20的256份d10orig档案，不追求恢复原状。
先确认采集正常，再准备排除语料、固定面板、清理非cq/obsup工作树路径；**先报告主控，再部署**。
观察到00:22:21、00:26:03、00:30:12连续采集组均12局且rc=0，生产仍推进。

实现采用精确退役标签规则，统一用于旧/新配方、档案目录和打包选项，不把任意文件丢失改成
静默跳过。边界更新时仅移除已知种子范围内256个断掉的目录软链；其他标签段保持原身份，
只更新小型语料索引，不改变PACK_VERSION、不强制重建段。实际训练视图只读目录链接盘点显示，
指向其他工作树的条目恰为这256个，其他语料无需搬家。

将原版面板放到受git跟踪的`configs/panels/day_orig_panel.json`，同时固定旧v1/v2与identity
面板，保留任务定义/种子不变。运行所需引擎准备复制到`artifacts/production/deploy16/`，
不再用生产包工作树作为运行时路径；旧实验/证据改用`artifacts/archive/<来源>/`，不伪造已丢失
的模型和结果。部署源使用主仓库Git对象，cq/obsup保留为允许的代码目录。

4项相关小测试通过（约1.42秒），先free、nice10、8GB地址空间与timeout90：覆盖精确/可回滚
的软链移除、旧配方与真实训练加载入口排除、已有段复用不重建、固定面板与路径规则。
测试只用临时夹具，未训练/打包真实语料。后续准备边界更新和回滚脚本，再报告待审结果。

边界更新准备已完成：`qc_deploy17_paths.sh`从主仓库Git对象提取固定版本，不依赖deploy17工作树；
复制当前原样引擎到稳定路径在暂停前完成。然后等组完成且持久化，备份现有文件/启动参数/链接，
仅更新cq的必要运行模块，并对cq、obsup及从当前引擎发现的生产包目录做路径文本迁移，
保留这些目录里的其他本地改动。版本化面板同时安装到主仓库和相应代码目录，兼容旧配方解析器。
未执行更新脚本；实际只读preview计划为134个文件、256个退役目录链接。

最终相关小项7项通过（1.51秒，nice10/8GB/timeout90，101项未选中），补充已有A/B配方合同、
待建段合同及备份/恢复/不可覆写不同引擎的夹具；壳脚本仅语法检查。
没有训练真实模型、全量重包或执行真实迁移。待提交固定源码与操作脚本后报告主控。

### 待审交付（源码2f038dbe，生产仍为ad4fe41c）

源码/配置/小测试已提交推送`2f038dbe`。操作脚本固定使用这份源码，动态导出的启动函数也固定
同一版本；未把生产控制器换成新代码。完整110条原非cq/obsup代码/配方引用及其固定路径映射、
四份面板SHA、134文件预览和256链接数见
[task165-stable-path-plan.json](task165-stable-path-plan.json)。这份附件`deployed=false`是本次交付状态。

当前生产状态再次核对：`day_l10_orig_train`仍为L10、offset306；固定面板的空卡组、原版传送带、
关卡/地形/阳光/预种植/playthrough等与生产六档倍率任务一致，原训练种子1880000..1888191。
day_orig面板只是把已确认的内容纳入版本控制；没有新开卡组或重领种子。

本次部署生效时将：

- 只移除两标签的256个已知断掉目录链接；新旧配方加载都明确排除两标签。其他缺失照常报错。
- 保持每标签段的配置身份和PACK_VERSION=2，只重建少了两标签的小索引，复用其他已有段。
- 将原样引擎放到`artifacts/production/deploy16/pvz-portable`，四份面板来自跟代码绑定的configs。
- 保留当前模型、课程状态、已采数据、种子偏移/账本、A/B数值配方和四类配额；不部署gateunits。
- 历史证书/冻结任务文件保留旧provenance，避免改写证据；其训练入口按新明确退役规则解析。
  旧研究脚本转向固定archive命名空间，并不以缺失的旧模型/结果为当前生产启动条件。

主控审后安装本分支的`scripts/qc_deploy17_paths.sh`到`~/pvz-logs/qc_deploy17_paths.sh`，执行：

```bash
bash ~/pvz-logs/qc_deploy17_paths.sh --at-boundary
```

脚本先形成review.json并准备稳定引擎，再等实际采集组完成、状态持久化后暂停旧控制器；
备份文件和256链接的目标字符串，切代码/路径/退役规则，重新启动。备份目录写入
`~/pvz-logs/deploy17_paths.latest`。主控如需回滚：

```bash
bash ~/pvz-logs/qc_deploy17_paths.sh --rollback
```

该回滚恢复更新前的代码、启动文件和链接，不回退模型、已采数据、课程状态或账本；
也会回到更新前因原档案缺失而无法训练的状态。稳定引擎是只读原样副本，回滚不删除它。
待主控审后才执行更新；本次只读preview未移除生产软链、改配置或重启任何进程。

操作脚本提交`c2576413`后补充看门狗启动桥接：若新updater是最新qc_deploy脚本，
看门狗仍能只提取启动函数，转到现行`qc_deploy17.sh`，不执行更新体；重复执行/回滚也可直接返回。
新增对应`-k`小项0.04秒通过，连同前述7项共8个相关用例通过；未执行真实服务操作。


### 2026-10-30：按主控补充简化 day_orig（只提交，不部署）

本节替代前节将 day_orig 面板纳入 git 的方案。删除 `configs/panels/day_orig_panel.json`，
新增 `scripts/day_orig_tasks.py:day_original_tasks()`，在内存中按关卡、`deck_for_level` 和固定种子段
生成原有 11 个任务。L1–10 跳过 L5，普通评估段 30000..31023；L10 原版传送带评估同段，
训练段 1880000..1888191。原版开局仍为空卡组、`conveyor_start=original`。
这里沿用前节已从生产状态确认的 task id、关卡、卡组和种子定义，没有改变任务族或偏移。

`make_multiplier_tasks.py`、采集共用的 `render_episode.load_training_task`、`b3_noninferiority.py`、
`build_adventure_gap.py`、`unified_panel.py` 接入同一函数。CLI/队列使用 `builtin:day_orig`
仅作来源标记，不落临时 JSON；状态中已有的旧 conveyor 清单路径同样直接调用函数。
`recipes/panel_v1.json`、`panel_v2.json`、`panel_pressure.json` 移除 `day_orig_manifest`；
`pvz_recipe.py` 接受旧冻结配方的遗留字段，但不再要求它。

准备中的边界更新脚本和安装助手同步移除 day_orig JSON 的复制项，安装生成函数，
对 cq / obsup / 当前原生代码包的相关调用做局部替换，保留其余代码差异。
此前 task20 两标签共 256 局的弃用处理不变，不重打全部训练段。
路径计划文件的旧 preview 保留为历史记录，已注明此函数方案取代 day_orig JSON 项。

本次按“只改 conveyor / task20”范围完成上述接线及必要的配方、更新脚本和文档调整；
未继续全仓库清理，留给 task169 的 purgelive / purgedead。
按用户要求未新增校验、未复跑；只调整原有相关用例以适配函数接口。
只读查看生产日志：00:50:41 随机卡组 L32 完成 24 局、rc=0，随后开始下一组；
现行 active 仍为 `ad4fe41c`。没有改生产配置、移除生产链接或触发组边界更新。
部署与回滚仍使用前节两条命令，等待主控审后再执行。

实现提交 `18313d3c`；准备中的 `qc_deploy17.sh` 与 `qc_deploy17_paths.sh` 均改为固定该源码版本，替代前节 `2f038dbe`，避免更新时重新安装已删除的 day_orig JSON。仍未部署。
