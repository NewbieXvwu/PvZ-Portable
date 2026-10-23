# task32（主控 2026-10-22 22:45 下发给 Mac 上的 CodeBuddy 会话 cb-mem）：压内存——训练、采集、价值网络

先读仓库根目录 `AGENTS.md`（优先级最高），再读 `MEMORY/notes/desktop.md`（WSL 只有 24 GB→28 GB、内存打满时 wsl.exe 连不上的事故）、
`MEMORY/docs/student-bottleneck-plan.md` 末尾 10-22 的几条、`MEMORY/docs/unified-panel.md` 第 52 行起的训练资源表。

## 背景
台式机 WSL（24 GB）今晚 21:18 起被内存挤满，`wsl.exe` 再也连不上（里面的任务还在跑，但无法管理，只能重启）。同时在跑的是：
学生训练 dw5i（`scripts/train_rollout_policy.py`，dw4i 时 1364 万行、子树 RSS 求和峰约 15 GB、最低可用约 4 GB；dw5i 行数更多）、
课程采集 8 个游戏进程（每个 = Python `search_play` 约 90 MB + 引擎 `pvz-portable` 约 220 MB，Mac 实测）、价值网络补采 / 训练（`scripts/train_value_net.py`、10-15 一次 16.6 GB 被 OOM 杀）。
数据只会越来越多（课程、夜间、泳池、换卡组），不压内存下一次合训就进不去。用户认为很多东西还能压。

## 要做的
工作区（Mac）：主控已建 `/Users/newbiexvwu/PvZAgent/.claude/worktrees/mem`（分支 `mem`，基于 pvz-env）。Mac 有 48 GB，正跑着 10 个采集进程（别停它们）；
数据可用 `/Users/newbiexvwu/PvZAgent/artifacts/hard_mac/frames`（只读）。Python 用系统 `python3`（有 torch）。不需要 GPU。

1. **量清楚钱花在哪**（先量后改）：
   - 学生训练：`train_rollout_policy.py` 的数据装载（`python/pvz_frame_store.py`、`python/pvz_training_rows.py`，含特征缓存、CSR 棋盘、标签、软目标、`packet_info`、验证集）——
     用 Mac 上一个子集（如 20–50 万行）测**每行每个组成部分的字节数**和峰值（`tracemalloc` / `resource.getrusage` / 逐数组 `nbytes`），外推到 1500 万行。
   - 采集：一个 `search_play` worker（Python + 引擎）的常驻构成：torch 导入、学生 / 价值网络权重、观测缓存、快照；引擎侧快照与推演缓冲。
   - 价值网络训练：`train_value_net.py` / `train_value_branches.py` 的装载方式。
   结果写成表（组成部分 → 字节 / 行 → 1500 万行时 GB）。
2. **压**（每一项都要证明**结果不变**）：例如更紧的 dtype（能装下就用 uint8 / int16 / float16 存、计算时再转）、把只读大数组放进 `np.memmap`（磁盘缓存文件，按需换页）而不是常驻、
   去掉重复副本（训练集 / 验证集切片时别复制、标签与特征别各存一份）、分块装载、采集 worker 不必要的 torch 线程池 / 缓存。
   **硬门槛**：同一数据同一种子，改前改后训练出的模型权重**逐位相同**（或若因浮点顺序不能逐位，前 200 步损失逐步一致到 1e-6，并说明原因）；采集同种子对局 games.jsonl 逐局相同。
   做不到逐位相同的改法放进“问题”，由主控决定。
3. 所有改动做成**默认关闭的开关**或证明逐位相同后直接生效（逐位相同的可以直接改默认，写清证据）；在 Mac 上给出改前 / 改后的峰值内存与耗时对比表。
4. 报告写 `MEMORY/docs/memory-budget.md`：现状账本、每项改动省多少、改后 1500 万行训练 + 16 采集 + 价值网络在 28 GB 里怎么排。

## 规则
- 不动 Mac 上正在跑的采集（`mac_dagger_*`、`search_play`）与别的会话；同时最多 2 个你自己的游戏进程；不改主工作区与别的工作树。
- 分支 `mem` 提交，可以 `git push origin mem`；不推 pvz-env、不合并。不写手工规则；不写字面密钥；不 `git add -f`；`.pt/.npz/.log`/缓存文件不进 git。
- 只用绝对路径。每完成一小步就提交并在本文末尾“结果”里记一行；疑问写“问题”，按最佳判断继续，不要停下来等。

## 结果

- 第 1 步（量）：`scripts/measure_training_memory.py` 在 Mac 上按生产配方（`--wait-keep .1 --obs v2 --packet-head identity --cell-soft-weight 3`）跑 200 档案 / 61 903 行，逐组件记账 → `MEMORY/docs/task32-baseline.json`。外推 1500 万行 **14.32 GiB**：棋盘 CSR 7.49（536 B/行，134 非零/行）、`packet_info` 2.24（160 B）、标量 CSR 1.68（120 B）、`cands` 1.01（72 B）、`values` 0.62（44 B）、keep 掩码 0.39（28 B）、索引张量 0.22、packet/cell 0.22、`seed_of` 0.12。

- 第 2 步（压·学生训练）：`train_rollout_policy.py` 加两个默认关闭的开关 —— `--compact-data`（keep 掩码位打包、plant uint8 / packet·cell int16、`PacketStore` 按列拆成 int8/uint8/float32/uint8、标量 CSR 索引按位宽降到 uint8、按档案把恒定平面（可通行车道那一层 50/134 个非零）提到平面表）与 `--data-memmap`（棋盘/标量 CSR、`values`、`cands`、`packets` 写文件后只读映射，页面可回收）。逐位门槛：200 档案 61 903 行、同种子、生产配方跑一遍，四种组合（默认 / compact / memmap / 两者）21 个权重张量 sha256 **全同**；`data_memory` 从 54.6 MB 降到 35.1 MB（棋盘 33.7→20.7、packet 9.9→4.3）。新测试 `python/test_compact_loader.py`（默认 / compact / memmap / 两者 的密集棋盘逐位相等，含 -0.0）。

- 第 3 步（压·价值网络）：`train_value_net.py` 的训练棋盘从稠密 `(样本, 92, 6, 10) float16`（11 040 B/样本）换成同一套 CSR 行存储（约 330 B/样本，21 倍），`train_value_branches.py` 的多份 dump 合并改成写文件再映射（不再 `torch.cat` 复制一份）。新测试 `python/test_value_net_rows.py` 用第二次 `featurize` 对照 CSR 取回的棋盘逐位相等；60 档案端到端（2 折 2 epoch）跑通。2000 档案基准（61.9 万行、60 步、同种子）四种组合 21 个权重张量 sha256 **全同**，`data_memory` 54.6→35.1 MB 的同一比例：棋盘 335→206 MB、packet 99→43 MB。

- 第 4 步（引擎·无界面只读尺寸）：`-env` 模式下 `SexyAppBase::Init` 打开 `ImageLib::gSizeOnly`，`ImageLib::GetImage` 只解析文件头（PNG IHDR / GIF 屏描述符 / JPEG SOF / TGA 头）取宽高，不再分配像素；回退顺序与完整加载一致（基图缺失时探 `_name` / `name_` 的 alpha 图 —— `IMAGE_BLANK` 实际只有 `images/blank_.gif`，这就是第一版启动即 `Failed to load image` 的原因）。`SexyAppBase::GetImage` 对无像素的图只填宽高；`MemoryImage::GetBits` 在无可恢复像素时清零而不是从空指针 `memcpy`（字体 `Palletize` 会走这里）；`DoLoadImage` 跳过 alpha 合成。验收：同一动作序列（第 7 关 80 决策、17/27/47 关各 60 决策，覆盖白天/夜间/泳池/屋顶）两版 `frames` 摘要**逐字节相同**；每个引擎进程峰值常驻 **231.5 → 123.3 MB（-47%）**，启动 **1.02 → 0.81 s（-20%）**。`scripts/engine_replay_compare.py` 每个引擎放到独立 spawn 子进程里跑，否则 `RUSAGE_CHILDREN` 的 `ru_maxrss` 会把第一个引擎的峰值报给第二个。
- 第 5 步（引擎·pak 映射 + 动画定义按需加载）：`PakInterface` 不再把 45 MB 的 `main.pak` 读成私有副本再整体 XOR 解码 —— POSIX 下 `mmap` 只读映射，XOR 改在 `ReadAt` 里按读取解（Windows 仍走 malloc 副本，行为不变）；无界面模式下 `ReanimatorLoadDefinitions` 不再把已编译的 ~150 个动画定义全部预读，改由 `ReanimatorEnsureDefinitionLoaded` 在第一次用到时加载（内容相同，只是时机）。验收：第 7 关 300 决策（到第 6 波，大量僵尸会触发按需加载）+ 第 17/27/47 关 150 决策，`frames` 摘要**逐字节相同**；每个引擎进程峰值常驻 **233.3 → 59.7 MB（-74%）**、物理占用 **91.9 → 21.4 MB**、启动 **1.05 → 0.33 s（-69%）**。堆栈归因：`DefMapReadFromCache` 25.5 MB、pak 私有副本 43.9 MB。
- 第 6 步（转正·删开关）：`train_rollout_policy.py` 的紧凑存储改成**唯一实现**，删掉 `--compact-data` / `--data-memmap` / `--dense-boards` 与稠密路径（`_board_store`、`CompactBoards` 的文件后备、`PacketStore` 的列后备）；`_spilled_array` 留给 `train_value_branches.py` 合并 dump。`--data-memmap` 不保留：15M 行时紧凑行约 8.3 GiB，28 GB 预算里放得下，而换页版本在 2000 档案实测只把峰值 2.65→2.64 GB、却要在临时盘占 8.3 GB。门槛：同一 200 档案 / 66 993 行 / 同种子 / 生产配方，改前（`--compact-data`）与改后 21 个权重张量 sha256 **全同**；峰值 2.457→2.423 GB，耗时 34 s 不变。`python/test_compact_loader.py` 重写成单实现的测试（CSR 往返逐位含 -0.0、恒定平面提出去再写回、packet 列还原、缓存与非缓存两条路径一致）；`test_pipeline_perf` / `test_rowpool_aux` / `test_student_observation` 去掉 `dense_boards=True` 后全过。
- 第 7 步（报告 + 整局门槛）：`MEMORY/docs/memory-budget.md` 写成（账本、每项省多少、28 GB 排布、复现命令、还剩什么）。`scripts/engine_replay_compare.py` 加 `--student`：用学生自己下整局（每决策一帧，到 terminal），同种子多局逐帧比摘要。`scripts/measure_collection_memory.py` 量 worker 常驻构成（解释器 23.0 / 导入后 24.4 / 装网络 205.0 MiB，引擎 57.2 MiB）。合并 `origin/pvz-env`（弹窗改 stderr 那次），重编后重放仍逐帧相同。
- 第 8 步（整局门槛）：同种子 64 局、每局 400 决策、学生（v25）自己下，两版引擎**每局逐帧摘要全部相同**（最深第 13 波，无一局在 400 决策内结束）；峰值常驻 222.9→60.6 MB，启动 1.048→0.343 s，64 局总耗时 89.8→44.2 s。数字已写进 `MEMORY/docs/memory-budget.md`。
- 第 9 步（收尾修正）：`PacketStore.__getitem__` 的堆叠轴改 `dim=-1` 并支持 `(row, slot, component)` 单格取值（原来的 `dim=2` 只在标量下标下成立，向量下标直接报错，之前没人走到）；`python/test_identity_training.py` 的 identity 对照改成按列还原。7 个相关测试文件 24 项全过。
- 第 10 步（cands 稀疏）：`CandidateStore` 只存每行排名靠前的候选（uint8 计数 + 6 B/候选），`__getitem__` 还原成 (12,3) 的 NaN 填充块，调用方不用改。实测 **72 → 25.9 B/行**（66 993 行 1.73 MB），1500 万行省 **0.64 GiB**；同 200 档案 30 步权重 21 张 sha256 **全同**。顺带发现 `data_memory` 之前没算 `cands`（账本漏项），已补 `cand_bytes`。
- 第 11 步（values 稀疏 + 验证分块）：ValueStore 只存非 NaN 的候选（uint8 槽位 + float32 值），__getitem__ 还原成 (11,) 的 NaN 填充行，spread() 直接给出比较过的行与极差 —— **44 → 17.6 B/行**（66 993 行 1.18 MB），1500 万行省 **0.36 GiB**。验证集分块 4096 → 1024（无梯度，分块只需装一次前向；格子头那个 (chunk, 320, 6, 10) 输入是整轮里最高的分配）：峰值 **2432 → 1621 MB（-33%）**，权重 21 张 sha256 **全同**。注意 val_loss 6.9200 → 6.9081：它一直是「各块均值按行数加权」，本身随分块大小变，这次一次性说明。数据合计 39.70 MB / 66 993 行 = **593 B/行 → 8.29 GiB**（1500 万行）。
- 第 12 步（分布与停止条件）：报告补「当前分布（前 15 项及占比）」——训练数据 593 B/行（棋盘 60.9% / 标量 18.2% / packet 11.8% / 格候选 4.4% / 目标 3.0% / 其余 1.7%），引擎 malloc 7.9 MB（动画定义 51.9%、定义树节点 12.4%、pak 记录表 6.1%…）。连续三项尝试都 <2%：格子头检查点 0%（峰值不在那）、标量降 float16 1%（只有 15% 非零可无损）、packet progress 降 float16 0%（62% 可无损但必须 100%）。前 15 项已无可证明的冗余，按主控的停止条件收工。
## 问题

## 追加（主控 2026-10-22 23:05，用户的两点）
1. **“结果不变”放宽**：此前实验已明确模型与搜索本身噪声很大（同一老师换随机未来只有约 75% 的决策相同），所以**不必逐位相同**。
   收益足够大、代价不大的改法可以接受：训练侧要求同数据同配方下验证损失与面板胜率在种子噪声内（给出对照数字）；采集侧要求同种子对局的胜率 / 到达波数在噪声内，
   且**游戏逻辑本身不能变**（同一动作序列重放，引擎的状态轨迹必须逐帧相同——这条仍是硬门槛，因为它决定标签的含义）。
2. **游戏程序里不必要的加载**：`pvz-portable -env` 是无界面模式，但 `LawnApp::LoadingThreadProc`（src/LawnApp.cpp:2538 起）仍然加载
   `LoadingImages`、`LoadingFonts` 资源组，初始化 `PoolEffect`、`ZenGarden`、`ReanimatorCache`、粒子 / 拖尾定义、`PreloadForUser`；Mac 实测每个引擎进程常驻约 220 MB，台式机 16 个就是 3.5 GB。
   查清无界面模式下哪些东西游戏逻辑真的要用（例如动画定义的时间轴会影响攻击 / 判定，不能丢；贴图像素、字体、水面特效、禅境花园大概率不用；
   若有代码在逻辑里读图片尺寸，改成只读尺寸不解码像素），按需加载或跳过。**验收**：同一动作序列重放逐帧状态相同（`frames` 逐字节或逐字段相同），同种子 64 局 games.jsonl 相同；
   报告每个引擎进程的常驻内存与启动时间改前 / 改后。这一项优先级高（每个采集 worker、每次搜索推演都受益），可以先做。

## 追加（主控 2026-10-22 23:55）：不再加开关
重构不加开关：新实现直接替换旧实现，验收通过后旧路径在同一系列提交里删掉，A/B 用上一个提交对比。你已加的 `--compact-data` / `--data-memmap`：既然权重逐位一致，就把紧凑存储改成唯一实现、删掉旧路径与这两个开关
（`--data-memmap` 若与常驻 RAM 是二选一的取舍，按 28 GB 预算给出结论后只留一种，或把它登记进 `MEMORY/docs/flags.md` 并写到期日）。价值网络 CSR 同理。引擎跳过不必要加载也不加开关。

## 追加（主控 2026-10-23 00:45，用户）：持续压榨，直到没有油水
用户的原话：“得继续给优化代理加压，让它持续优化直到真正完全炸不出油水为止。”本任务不再以“做完列出的几项”为完成：
每做完一项就重测、把当前的耗时 / 内存分布（前 15 项及占比）贴进报告，再挑下一个最大项继续；**停止条件**：连续 3 项尝试的收益都 < 2%，且分布前 15 项里没有可以证明是冗余的部分。没达到就继续，不要自己宣布完成。

## 下一阶段（主控 2026-10-23 08:30 追加）：存储瘦身（与 task33 分工：它管训练装载的打包，你管档案与日志的存量）
1. **旧 JSONL 档案转 npz**：`artifacts/value_net/episodes*` 等 6 866 个 `frames.jsonl`（7.54 GB，task33 实测转 npz 约 7%）。写正式转换工具（逐帧字段相同的校验内置），在 Mac 上全量转换，
   校验通过后删除原 jsonl（只删已逐帧校验过的；删前列清单进报告）；读取端保留 jsonl 读路径并在 `python/pvz_compat.py` 登记为 LEGACY（若 task33 尚未建该文件，先建最小版本，合并时以它为准）。
   台式机的同类档案给出同一命令，由主控择时执行（不要远程动台式机）。
2. **日志轮转**：`artifacts/**/*.log`（221 个、0.49 GB、无轮转）与 `artifacts/hard_mac/*.out`：写 `scripts/rotate_logs.py`（超过 N MB 的压缩成 .gz 留最后 1 MB 明文），给出 Mac 与台式机各自的调用方式。
3. **每局一个的小 json**（29 679 个 config/meta/summary）：评估能否在打包时并入包头（与 task33 第 2 / 5 项对齐，接口在“问题”里和主控确认后再动）。
4. 完成后把本分支与最新 `origin/pvz-env` 合并、全量测试、push。仍按停止条件持续压榨。
- 第 13 步（档案转 npz·工具）：`scripts/convert_frames_npz.py` —— 逐档转换 + 五道校验后才删原文件：①npz 里存的字节与 jsonl 原文逐字节相同；②每一帧解析出的 dict 全等；③行索引恰好等于有标签且未被拒的帧号；④`encoder` 签名是当前版本；⑤boards/scalars/plant/packet/cell/values 的行数与行索引一致。写盘走临时文件 + 原子 rename，默认跳过 60 分钟内改过的档案（不碰正在写的采集）。冒烟：教师诊断档 1.4 MB→9.5%、value_net 档 636 016→58 764 B（9.2%），577 帧逐帧全等，`cache_episode` 直接复用 npz 里的行缓存不再重算。`python/pvz_compat.py` 建最小版（只登记 `jsonl-archives`，合并 task33 时以它的完整版为准），`pvz_frame_store.py` 两处加 `LEGACY(jsonl-archives)` 标记（与 task33 措辞逐字一致，便于合并）；`scripts/pack_frames.py` 改走 `frame_paths`（jsonl/npz 都收）、`scripts/early_fix_check.py` 改走 `frame_path`，避免转换后读不到。
