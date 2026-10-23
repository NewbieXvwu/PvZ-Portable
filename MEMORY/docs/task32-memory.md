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

## 问题

**第 15 步（小 json 并包头）的接口提案，等主控确认再动。**

现状：30 521 个小 json、13.9 MiB 数据、119.6 MiB 块；其中 22 116 个是 `hard_mac/frames` 下 npz 档案旁的
`meta.json`（每局一个，平均 388 B）。真正的浪费是**每文件 4 KB 的块**，不是数据。

建议形态（与 task33 第 3 项“打包成顺序大文件”对齐，分两级）：

1. **档案级**：`frames.npz` 增加一个 `meta` 键（uint8 的 JSON 字节，与 `frames` 同一写法）。写入端
   `write_frames` / `write_frames_from_source` 接受可选 `meta: dict`；`meta.json` 仍写，直到转换工具
   确认过；读取端新增 `pvz_frame_store.frame_meta(archive) -> dict`：npz 里有 `meta` 就用它，否则读
   `meta.json`（按 `LEGACY(jsonl-archives)` 的方式登记）。收益：22 116 × 4 KB ≈ 86 MiB 的块浪费 +
   8 MiB 数据 → npz 内压缩后约 +2 MiB。
2. **包级**（task33 的 pack）：包头放一张 `meta` 表 —— `meta_offsets: int64[n_episodes+1]` +
   `meta_bytes: uint8[...]`（每局一个 JSON 对象，顺序与档案表一致），`pack_meta(pack, i)` 取第 i 局。
   包里的局不再需要任何旁挂文件；未打包的旧档案走第 1 级。
3. `config.json`（218 个）/ `summary.json`（215 个）是**每轮一个**，不是每局一个，并入包头只省
   0.3 MiB 数据 + 1.7 MiB 块，建议保持独立文件（人要看、要 grep），不做。

需要主控拍三件事：

- (a) 转换完成后是否**删除**已嵌入 npz 的 `meta.json`（Mac 上 22 116 + 其余约 8 000 个，删了省
  ~119 MiB 块）；台式机按同一条命令由主控择时执行。
- (b) 包头 `meta` 表由 task33 的 pack 工具实现，还是我这边先做档案级、包级留给 task33？两边都动
  `pvz_frame_store.py` 的写入端，需要定一个归属，避免冲突。
- (c) `meta` 键会不会破坏现有 npz 的 `encoder` 签名判定（`cache_episode` 比对签名后直接复用档案）；
  我的实现里签名不覆盖 `meta`，只覆盖编码器源码 —— 若 task33 要求签名覆盖包头字段，需要一起改。

未做任何改动；等确认。

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
- 第 14 步（日志轮转）：`scripts/rotate_logs.py` —— 超过 `--min-mb 2` 的日志把除最后 1 MB 以外的部分压进 `<name>.log.gz`（每次轮转追加一个 gzip member，长期日志最终是一个 .gz），最后 1 MB 留明文、切点对齐行首；120 分钟内改过的文件跳过（可能还在写），改写走同一 inode 的 seek/write/truncate 不 rename。回环测试：5.9 MB 日志轮转后 gz+明文尾巴**逐字节等于原文**；追加后再轮转仍等于完整历史。Mac 实测 104 个文件 463.3 MB → 明文 109.0 + gz 28.8 MB，**省 325.5 MB（70%）**，9 个正在写的跳过。命令与数字进 `MEMORY/docs/memory-budget.md` §7.2。
- 第 15 步（小 json 并包头·只评估未动）：`artifacts` 下 ≤64 KB 的小 json 共 30 521 个、13.9 MiB 数据，但占 **119.6 MiB 块**（4 KB 一块，平均每文件 460 B，块浪费 105.7 MiB）。分布：`hard_mac/frames` 22 116 个 meta.json（8.19 MiB，已是 npz 档案的旁挂文件）、`value_net/episodes` 2 048（2.92 MiB）、`value_net/episodes_search` 2 048（0.43）、`student/frames` 2 048（0.43）、`relabel/z15dag1{2,3}bv` 1 280（1.08）、`student/frames_deep_mac` 512（0.11），其余零散；`config.json` 218 个 / 0.23 MiB、`summary.json` 215 个 / 0.07 MiB 是**每轮一个**不是每局一个。键：每个 meta 都有 `task`+`outcome`，2 051 个有 `totals`/`policy_revision`/`frame_count`/`checkpoint`/`checkpoint_sha256`/`simulator_build`，1 280 个有 `relabel`。**按主控要求只做评估，接口写进「问题」等确认，未改任何读写路径。**
- 第 16 步（档案转 npz·全量）：6 658 个档案（+ 第 13 步冒烟已转的 1 个）**全部通过五道校验后删原文件，失败 0**：7 509 703 263 B JSONL（7.00 GiB）→ 563 481 079 B npz（537 MiB），**7.50%**，省 6.95 GB，196 s（`--jobs 4`）。Mac 上 `frames.jsonl` 剩 0 个，`artifacts` 11 GB → 6.2 GB（含日志轮转）。逐档清单 `MEMORY/docs/frames-npz-converted.txt.gz`（6 658 行、70 KB、sha256 见 report），数字进 `MEMORY/docs/memory-budget.md` §7.1。合并 `origin/pvz-env`（task33 打包装载转正 + CUDA 线退役 + task34 OOM）：冲突只有任务书与 `pvz_compat.py`（取 task33 完整版，`jsonl-archives` 的移除条件改成"所有机器与 HF 上都不再"）；`train_rollout_policy.py` 自动合并为 task33 的包装载 + 本任务的紧凑数组。全量测试：**253 passed / 1 failed / 23 skipped**，失败的是 task34 的 `test_dagger_experiment_runner.py::test_low_memory_pauses_and_resumes_only_own_parent_and_child`，在 `origin/pvz-env` 原样代码上同样失败（已用 pvz-env 版本的两文件单独复现），不是本次合并引入。
- 第 17 步（档案容器改 LZMA）：`pvz_frame_store._save` 从 `np.savez_compressed`（zlib）改成 `zipfile.ZIP_LZMA` —— 12 档样本上**只要 59.6% 的体积**（235 KB → 140 KB/局），因为 72% 的字节是 frames JSON，LZMA 能跨整局建模重复。代价：写 0.055→0.63 s/局、读 4.4→12 ms/局（一局游戏几十秒到几分钟，均可忽略）。`np.load` 无需改动（zipfile 解码），实测 LZMA 与 deflate 档案都能被 `read_frames`/`cache_episode` 读，且 `cache_episode` 仍直接复用档案（签名命中）。**训练包必须保持 ZIP_STORED**（`pack_training_rows.py` 用 `np.savez`，`read_pack` 走 `mmap_mode`），`python/test_frame_store_container.py` 三项测试钉住：LZMA 容器、0 维字符串、包不被 LZMA 污染。转换工具加 `--recompress`：逐档把所有数组成员读出来重写，**逐个数组 dtype/形状/值全等**才原子替换，ZIP_STORED 的包与 120 分钟内改过的文件跳过。
- 第 18 步（LZMA 存量重压）：30 120 个 npz 里 22 205 个重压成功、失败 0，**5 195 921 622 B → 3 105 601 019 B（59.8%），省 2.09 GB**，8 并发 1 970 s；7 915 个跳过（训练包 ZIP_STORED、已重压、120 分钟内改过）。`artifacts` **11 GB → 4.3 GB**（转换 6.95 GB + 日志 0.33 GB + LZMA 2.09 GB）。数字与命令进 `MEMORY/docs/memory-budget.md` §7.3。
