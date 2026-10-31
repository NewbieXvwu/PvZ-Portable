# task186：GPT Pro 扩卡组专用包

## 范围与交付

分支`llmplay-decks`，工作树`~/PvZ-search/.claude/worktrees/llmplay-decks`，
基于`origin/llmplay5`的`49f1667e`。复用上一包静态引擎、资源、接口、
`episode_archive_v1`存档、断点续跑、占位与自动out.zip。
台式机产物目录`~/pvz-logs/llmplay-decks/`：接口7z分卷及原样llmplay_res.zip。
不重编引擎，不碰生产qc、看门狗或stop_curr，不使用进程批量终止。

本版包含用户后续确认的三项补充：雾关human_v2、低优先级原生传送带支线、
每套卡组为一胜尝试过的不同种子数。首次交付提交为`a269ea41`，本版取代其接口分卷。

## 主线：胜利卡组广度

任务族`llmplay_decks_l{level}_m100`，主线40个普通选卡冒险关：L1–49非5的倍数。
范围由`pvz_tasks.player_deck_levels`统一复用源码解析的传送带判定，并排除L15/L35小游戏。
L50已从主线队列移除，改打原生支线；其旧自选卡族定义保留供旧存档恢复。
全为原版1.0倍、不截波、二周目普通植物全解锁，升级所有权由profile_for_deck声明。

每项=(关卡,种子)。模型自选6–10张卡，与同关全部已登记胜组至少差2张。
差异沿用原定义：两边独有卡数量的较大值，顺序不计，替换一张算一张。
一胜登记并换组；同组最多三个不同种子的真实败局，第三败的存档note必须写一句失败原因。
前两败可提前换组。放弃、中途导出和重复种子不增加真实败局数。
不要求稳健复验，不设打法层或加压。reason原样写deck_reason，说明经济/主力/防守/应急/地形卡。

基础数量：白天、泳池每关12组；夜间、雾、屋顶每关16组，共576组。
每关至少3组6–7张、3组8张，先补大小配额再自由变化。
每地形每张合法非黑名单卡至少进入2个不同胜组签名；同组跨关获胜不重复计覆盖。
本片本轮主线数量、大小和地形覆盖均满后，先让支线补8胜；支线完成后主线每轮继续增加4组。
主线当前项都被占用时也可领支线。任务无终点，按≥10小时持续设计，只有用户说停才停；
本开发任务在交付并推送后停止。

## 雾：公开human_v2观测

全部当前可领的雾夜泳池任务（主线L31/32/33/34/36/37/38/39，支线L40）
明确设置fog_observation=human_v2，并随TaskSpec及environment_task保存。
引擎沿用生产/门禁已有遮挡：隐藏僵尸不进入公开观测，遮挡植物保留格子和种类，
其血量及状态字段为未知。observe的植物代码标?，明细写“看不见”，不输出-1/-1或假零血量。
summary僵尸总数为None，visible_zombies只数可见部分，unseen_plants说明植物状态未知。
离开可见范围的僵尸记unseen事件，不把消失当成击杀。

PROMPT解释雾遮右侧若干列、路灯花照亮周围、三叶草暂时吹散；未知不能当空格或无敌人。
llmplay5的carry仍只读结论，不搬旧对局文件，不重放；没有明确记录human_v2的旧雾关胜局
仅作为历史去重依据，不抵扣当前遮挡目标。旧存档始终读自身记录的环境，不改写旧任务族。
当前合法且非留出签名的其他原版1.0倍胜组可计入基础数量及覆盖。
只含结论的carry记录带carried标记，out.zip续跑时不会因旧对局文件本来不在包内就丢掉登记。

## 支线：原生传送带与固定卡开局

新族`llmplay_decks_native_l{level}_m100`，使用既有conveyor_start=original、deck=[]。
源码解析得到L5/10/20/25/30/40/45/50，全部纳入，每关8个不同种子胜局，共64局基础目标。
L50用原生Boss卡流，不自选卡。L15打僵尸/L35砸罐子需要接口未提供的小游戏动作，仍跳过。
支线不套主线卡组差异、大小配额或三败换组规则；到货顺序随种子变化，按名字选卡，
不能记固定槽位号。导出支线opening为空，不用任务空deck去索引动态传送带卡槽。

严格主线优先；本片主线本轮目标已满或当前主线项被占用后才领native_play。
支线达到8个不同种子胜局后不再派发该关；之后主线继续增长，progress.complete永远False。
沿用原claims锁/租期/释放，主线候选每关只派当前一项，使全被占用时可转到支线。

## 一胜的尝试成本

每次主线存档在meta根字段记录deck_seed_attempts与deck_attempt_seeds：
同关同任务同卡组至本次导出已尝试的不同种子数量及有序列表，包含当前种子。
已记录的败局、放弃和中途尝试都算；相同种子重开、重写、读档或重复导出只算一次。
每个尝试须导出或abandon留下记录。元数据只读取既有progress结论，不另建文件或重算对局。
列表随save/load/load_archive保留，恢复后继续合并；不会把尝试过其他卡组的种子算进来。
不为历史记录虚构已无法得知的尝试成本。

## 配置、黑名单与留出隔离

配置复用configs/llmplay_seeds.json的mode=decks及decks段，旧循环任务定义保留供历史读取。
包括targets、min_card_changes、max_losses、大小配额、coverage_per_card、growth_step、
native_order/native_wins及分片种子参数。coverage_priority按地形/卡号补重点目标，
如{"fog":{"27":4}}，主控稍后按task185补充。没有新增校验、manifest或JSON中间文件。

configs/deploy17_deck_holdouts.json从origin/deploy17原样纳入，按排序后的完整卡组签名
跨关拒绝；卡序不能绕过。黑名单只排零作用卡，不引入手写动作策略或强度配方：

| 范围 | 植物编号 |
|---|---|
| 全局 | 38、45、47 |
| day | 11、16、19、24、25、27、33、37、43 |
| night | 16、19、24、25、27、33、35、37、43 |
| pool | 11、25、27、33、37 |
| fog | 11、33、35、37 |
| roof（含旧Boss卡组定义） | 11、16、19、21、24、25、27、43、46 |

卡池从源码植物表中接口可表达的0–47号减对应黑名单，不用戴夫推荐。
泳池/雾主线带睡莲16，屋顶主线带花盆33；支线由引擎发牌，不适用主线禁卡池。

## 分片与种子

LLMPLAY_SHARD/LLMPLAY_SHARDS，0起编号，默认0/3；也可next_task(shard=i,shards=N)
或configure_shard(i,N)。N=1–40，N不是3时主线和支线分别按各自列表轮转分配。
每个新进程保持原设置；共用文件系统时给各会话独立LLMPLAY_OUT。

| 默认片 | 主线 | 支线 |
|---|---|---|
| 0 | 白天+夜间 | L5/10/20 |
| 1 | 泳池+雾 | L25/30/40 |
| 2 | 屋顶 | L45/50 |

主线每关初始百万种子窗口起点186000000+(level-1)*1000000。
支线从236000000起，按八关列表每关一个百万窗口；两条线及各关互不重叠。
窗口用尽均加64000000顺延，不物化全量种子。
阶段只计算当前已保存结论；启动与续跑不重放旧存档，不生成旧C/E诊断文件。

## 构建与验收

```bash
python3 scripts/llmplay/build_package.py \
  --reuse-package ~/pvz-logs/llmplay5/llmplay_pkg \
  --resource-zip ~/pvz-logs/llmplay5/llmplay_res.zip \
  --out ~/pvz-logs/llmplay-decks
```

PROMPT与包README同步三项补充；PROMPT不含用户指定的三个禁词。
引擎和资源直接复用llmplay5；未做冷解包、启动自检、全量测试或逐档重放。
用户补充前已有10项相关测试通过；本次扩展后的16项相关测试6.38秒通过，
均使用pytest -k并加ulimit -v 8000000，先看free -g，不启动引擎。

本次最终分卷已重打：llmplay_pkg.7z.001为5,621,562字节；资源包仍为原样的
34,258,402字节。两者及对应llmplay_pkg/均位于~/pvz-logs/llmplay-decks/。
尝试次数最后改为只读已发布progress，避免重复派生整份进度；其相关单项测试0.66秒通过。

## 用户补充：选卡前出怪表与LLM先验知识

新增公开`llmplay.roster(task_id, seed=item["seed"])`，在Game之前可调用。
它直接复用既有引擎：独立临时棋盘reset一次，再读取一次引擎波表，只返回种类及首次波次。
不推进tick，不重放，不占位，不计卡组尝试次数，不保存棋盘或任何JSON中间文件。
同进程同任务/种子缓存；默认种子为任务首颗，PROMPT始终传队列种子。
单次工作只含引擎启动/reset和一次读取，预计正常环境不足1秒、慢沙箱数秒；
这是选卡请求本身的查询，不是额外验收，不扫描旧存档或整关卡集。

返回zombies列表（type/name/slug/first_wave/sources/note）。首次波次取本seed的真实预生成波表，
不是gZombieDefs的mFirstAllowedWave；从1起，开局为0，无固定波次为None。
允许表中本seed未排入波表的种类、舞王伴舞、巨人投小鬼及Boss动态召唤明确标作可能威胁。
Boss候选直接解析现有Zombie.cpp的gBossZombieList与BossBungeeSpawn，不重编引擎。
不返回逐波数量、行/精确时机、随机数状态或完整内部观测；雾中当前视野保持human_v2。

PROMPT要求每次选新卡组前查表，利用LLM的PvZ先验知识，逐类威胁比较可选克制方案，
不必穷举；例示仙人掌/三叶草对空、磁力菇拆铁器、地刺或一次性植物处理车辆类、
保护伞应对蹦极/投石车等，具体由模型结合来敌、地形、合法卡与局面选择。
优先轮换克制方案、经济来源、主力输出方式和核心思路，deck_reason写清针对哪些威胁、
用什么克制及相对已有组的变化；接口不加入手写选卡配方或关键词校验。

本次仅运行相关`pytest -k roster`，4项通过（0.89秒，8GB虚拟内存上限，未启动引擎）。
最终接口分卷更新为5,623,668字节，位于~/pvz-logs/llmplay-decks/llmplay_pkg.7z.001；
资源包与引擎继续原样沿用llmplay5，任务范围、雾遮挡、支线及一胜规则保持。

## task188：现有task105导入链兼容性（2026-10-30）

在origin/llmimport的5c1992fa上，以L38雾关、L30原生传送带两份真实1.0倍胜局轨迹改成新族，
组成临时out.zip调用原run_sha，真实重放导入2/2通过（相关pytest -k，8GB虚拟内存上限，4.08秒）。
两类新族均按SHA独立登记；human_v2、deck_seed_attempts及种子列表原样保留；无稳健记录的胜局
分为half仍进入扫描作业，老师参数保持存档同关、同卡组，native保留空deck与original开局。
导入侧无需修改，不需因兼容性重启task105四服务。四服务不监听新zip；交回后仍须显式增量导入、
合并队列，再由主控按现有流程让scan/secondary重载；publish/refresh无需动。未启动/重启服务，
未改生产队列、清单或账本；所有测试文件在临时目录，未新增运行时校验、manifest或中间文件。

## task203：交回续跑与交付瘦身（2026-10-31）

基于origin/llmplay-decks af98308a。来源~/llm_runs/llmplay_decks_20261031_0350/out的2786条进度记录已按既有入口合并carry，仅读取结论，没有搬旧对局或重放；新版接口分卷覆盖至~/pvz-logs/llmplay-decks/，继续原样沿用引擎与资源。

逐局发布只写progress/report/budget/decks结论，.delivery仅留当前小型快照，旧代、旧out.zip和临时压缩文件即用即清。Game.export_archive成功登记后删除out内对应运行目录，返回archives中的正式路径。显式llmplay.export_bundle()在out同级固定的<out名>_exports目录生成连续ZIP分卷，每卷上限99000000字节；直接流式写分卷，不另留整包副本，新包成功后删除上一份，失败保留旧包。只含正式存档、进度、卡组登记与.py策略，预算内嵌进度，日志/缓存/运行存盘/其他包不参与。压缩锁与逐局发布锁分离，压缩不挡对局。

按用户追加要求，帧文件直接用现有pvz_frame_store.write_frames写frames.npz，Game.load_archive用frame_path/read_frames兼容读两种格式；没有另写解析。写函数新增默认开启的cache参数，接口调用cache=False只存同款无损frames数组，避免在GPT包里引入Torch与无用的训练特征计算；训练读取端原本就能按需建缓存。分卷对已压缩npz用ZIP_STORED，避免再压一次。resume支持首卷或分卷目录，并保留策略、卡组登记和预算。相关-k测试均设8GB上限。

task105导入使用现行task193/deploy22后的import-only入口，nice19、1–2个worker、2–4GiB上限，已有SHA回执复用；仅更新merged_progress索引，不排扫描。此次交回2753个路径去重为1379份SHA，1377通过、2份frame_count不符拒绝；旧llmplay4_partial的1046个路径去重为538份，534通过、4份frame_count不符拒绝。合并索引累计3791份SHA；不修补元数据、不重复重放，不需重载scan/secondary或其他服务，生产下轮读取新索引。

旧半成品的538个SHA目录确认全部已在导入库后，~/llm_runs/llmplay4_partial已删除，清掉3316734453字节。库内转换复用task32原工具及其内置逐帧比对，nice19、最多2个转换worker、3GiB上限；先处理冷档，待两批导入完成后对剩余新档用min-age=0转换并删JSONL，没有另加重放或校验链。

最终导入库体积从7316485861降至903508199字节（6.81→0.84GiB，期间还补入了旧半成品538份SHA）；3791个frames.jsonl全部转为frames.npz并删除，JSONL余0、NPZ共3791个。另清掉1个154532字节的本次转换临时文件。加上旧半成品目录删除，合计净减约9.06GiB。最终接口分卷llmplay_pkg.7z.001为6031116字节，carry保留2786条记录，位于~/pvz-logs/llmplay-decks/；以后只取export_bundle()返回的out外99MB分卷。所做筛选测试均通过，未做其他全量验收。

task203收尾追加：PROMPT/README均无用户规定的三个禁词，已移除外部获取分卷及手工拼接的说明；文档只说明llmplay.export_bundle()与llmplay.bundle_dir()，输出为out同级的<out目录名>_exports/（默认/mnt/data/out_exports/），外部获取方式由主控起跑提示词规定。README重打命令直接复用现成包的carry，不再依赖原始交回目录。新版接口分卷已覆盖~/pvz-logs/llmplay-decks/llmplay_pkg.7z.001，6026663字节，carry不变。

导入与并包完成后，按用户授权删除/home/newbiexvwu/llm_runs/llmplay_decks_20261031_0350整个原始解包目录，释放5321845240字节（约4.96GiB）；连同前述导入库压缩与旧半成品删除，累计净减15051557355字节（约14.02GiB）。开始前free -g显示available 21GiB，相关单项-k测试在ulimit -v 8000000、nice19下通过；生产qc、看门狗、stop_curr均未触动，没有新增校验或JSON中间文件，无需重载服务。
