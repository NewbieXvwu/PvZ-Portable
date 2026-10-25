# task81：给 ChatGPT GPT Pro（离线沙箱，单次 3 小时、3 个子代理）准备“LLM 打全程”包

> 主控 2026-10-25 写。执行者：台式机 Codex（plus 号，tmux `codex-llmplay`）。工作树 `/home/newbiexvwu/PvZ-search/.claude/worktrees/llmplay`（分支 `llmplay`，从 origin/pvz-env 拉出）。最多 2 个游戏进程；**不碰 tmux `qc`、cq、别的工作树**。

## 为什么
- 老师在屋顶 L42–L49 冷启动全部 0/16（task74）。人类示范已经有 L42–L44，L46–L49 还没有。
- 用户的 Business 订阅可以在 ChatGPT 网页端用 GPT Pro。它有一个**不联网**的 Linux 沙箱（x86_64），单次请求能连续干约 3 小时，还能并行开 3 个子代理。它按请求计费：中途补充信息不另算，一旦停下来，下一次就是新的一单。
- 用法：把引擎 + 最小接口打成压缩包上传，让它从开局打到通关，交回操作序列。它在时间上（任意 tick 读档重来）和空间上（任意合法格）都有完全的自由。
- 交回的序列和人类示范同一用途：在我们的引擎上重放校验，然后做中途接手扫描（task71 流程），老师接得住的点才进课程。**不当模仿标签。**
- 用户已确认：游戏资源文件随包上传，责任由用户承担（只上传到这一个私有会话，不进 HF、不进 git）。

## 要做的
1. **沙箱接口 `llmplay/`**（新目录，放在仓库 `scripts/llmplay/` 下，测试进 `python/`）：
   - 纯 Python，只依赖标准库，加上我们的引擎二进制。不能依赖 torch、numpy 以外的包；numpy 能不用就不用，沙箱里未必有。
   - **状态持久化靠操作日志**：一局 = (任务, 操作序列)。读档到 tick N = 从 reset 重放到 N。为了快，进程内可以用 `snapshot_fast` / `restore_fast` 缓存。工具调用之间进程可能被回收，所以磁盘上的日志才是唯一的真相。
   - **一次调用多步**：模型写 Python 脚本调用 API，一次跑完几十到几百步，最后只打印压缩摘要。最少需要这些 API：
     - `Game(task_id)` / `Game.load(path)` / `save(path)`；
     - `plant(card, row, col)`、`shovel(row, col)`、`wait(ticks)`、`wait_until(tick | wave | 条件回调)`；
     - `rewind(tick)`、`branch()`（复制一份独立的局）；
     - `observe(level='brief'|'full')`：文字局面，复用 `episode_query._fmt_scene` 的渲染；
     - `legal()`：当前合法动作；
     - `summary()`：当前波、最远存活波、剩余小推车、阳光、各行威胁；
     - `export_archive(dir)`：导出 **episode_archive_v1** 存档，必须能被 `scripts/human_demo.py verify` 直接校验通过，`label_source` 写 `llm_gpt_pro`。
   - 动作时间轴完全自由：任意 tick 都能执行合法动作，不限于老师的决策点。
   - 收集阳光与人类示范一致（`sun_collection=environment_auto`）。
2. **引擎二进制**：
   - 沙箱的 glibc 版本可能比台式机的 2.43 老。要在 Docker 的 `ubuntu:22.04` 容器里编出无窗口版本（静态链接更好），引擎版本和 deploy4 相同（SHA 16989f23 的源码）。
   - 用这个二进制在台式机和 Mac 上分别重放现有 4 局人类示范，必须逐位一致（`verify` 通过）。不一致就写进“问题”并停下来，不要往下做。
   - 资源目录取自 `/home/newbiexvwu/.cache/pvz-research-resources`。
3. **离线自测**：
   - 在 `docker run --network none ubuntu:22.04`（另外再测一次 `python:3.11-slim`）里，只用上传包里的东西：解压 → 跑自检 → 用 API 脚本打 L41 的 200 步 → 导出存档。
   - 然后回到台式机，用正式引擎 `verify` 这份存档，要求逐位一致。
   - 记下包的大小。ChatGPT 单文件上限约 512 MB。
4. **安全扫描**：
   - 包里不能有 `.git`、`artifacts/`、日志、模型权重、`~/.pvz-teacher-env.sh` 之类的文件，也不能有任何密钥。
   - 用 `hf_`、`sk-`、`AIza`、`ghp_`、`token`、`password` 这些模式扫描包内文件，结果写进“结果”。**不要打印命中的值。**
   - 包和资源文件都只放在本机 `~/pvz-logs/llmplay_pkg/`，不进 git，不上传 HF。
5. **任务书 `scripts/llmplay/PROMPT.md`**（给 GPT Pro 看，用户直接粘贴）。主控定的要求，必须原样体现：
   - **目标**：从开局打到通关，交回存档。关卡顺序：L49 → L48 → L47 → L46 → L45 → L50，全部 2.0 倍，卡组见 `configs/human_demo_tasks.json` 里对应的 `demo_roof_*` 族。L46 / L47 / L45 / L50 的族由主控另行登记，未登记前跳过。3 个子代理各拿一关并行。通关一关后，用不同种子再打（种子由主控在包里给出），或者换卡组变体。任务单要做到 3 小时内干不完。
   - **允许写搜索程序，但有硬止损**。“效果不佳”的定义：
     - 每关先用搜索，最多 **25 分钟**；
     - 每 **10 分钟** 记一次“最远存活波”。连续两个检查点没有提升 ≥1 波，或者 25 分钟到了还没有一局通关，**立即停止全局搜索，转手动**；
     - 手动 = 按波推进：每次读当前局面，写一段覆盖接下来 1–3 波的脚本执行，看摘要，不满意就 `rewind` 重写这一段；
     - 手动模式里允许**局部**小搜索（单段 ≤20 个变体、≤5 分钟），用来挑这一段的最好写法；
     - 同一段 rewind 超过 8 次仍过不去，就退回更早一段重写；
     - 一关累计超过 **70 分钟**仍未通关，先保存最远进度的存档，换下一关，有余力再回来。
   - **不改引擎，不碰内存与状态**。所有影响只能通过 API 动作发生；交回后我们会在原版引擎上逐位重放，改过引擎的结果一律作废。
   - **持续交付**：每通关一关、每打出新的最远纪录，就立刻把存档追加进 `/mnt/data/out/` 并重新打包成 `out.zip`。到 2 小时 45 分时停止开新局，只做打包与说明。
   - **交付物**：每关的存档、`report.md`（每关结果、用了搜索还是手动、在哪一波卡住、怎么解决的）、写过的搜索或脚本代码（我们只用于分析，不进生产）。
   - **把 3 小时用满**：没到 2 小时 45 分不要主动结束；全部做完了就按任务单继续打更多种子。
   - 中途用户可能会粘贴补充信息，照着调整。
6. **主控验收**：完成后在“结果”写清：包路径、大小、SHA256、离线自测记录、逐位一致记录、PROMPT.md 路径；`git push origin llmplay`（只推代码和文档）。上传由用户手动做。

## 规则
- 只用绝对路径；不 `cd` 进别的工作树；`PVZ_NO_POPUP=1`。不改生产默认、不加开关。
- 禁止 `git add -f`；`.pt` `.npz` `.log`、资源文件、二进制、压缩包都不进 git。不写字面密钥，不打印凭证。不手写打法规则或公式：接口里不放任何打法提示，任务书只写目标、工具与止损规则。
- 每完成一小步就提交、push，在“结果”里记一行；有疑问写进“问题”，按最佳判断继续。

## 结果

- 2026-10-26 01:11 **新包 `139f3ee5…` 断网容器重跑到一半，按主控指令停下，Codex 接手**（§3 口径，包变了就要重跑）。已完成：隔离 daemon 复用（仍在本目录 `docker-test/docker.sock` 跑着，pid 见 `docker.pid`，sock 属 root，客户端要 `sudo -n <docker> --host unix://…`）；**python:3.11-slim 容器全过**：`--network none` 解压（引擎 644 → 接口自补可执行位）→ `SELFCHECK OK` → play200 **181 操作 / tick 9441 / wave 3 / result 2**（与宿主逐字一致），存档在 `docker-test/out/retest2/slim/l41/`；ubuntu:22.04 无 python3，已 `--network host` 装好（3.10.12）并 commit 成 **`lp22py:retest2`**（装完容器已删）。**未做**：lp22py:retest2 的断网自测、两份存档回宿主独立引擎 verify + ticks 逐条比对、清理（rmi 三个镜像 / builder prune / 停 daemon / umount / 删 data·exec·docker·tgz，前后留档）。交接材料：stage 在 `/tmp/lp_retest2/pkg/`（zip 副本 + `play200.py` + `run_in_pkg.sh`）；**注意包内 zip 有顶层 `llmplay_pkg/` 目录，解压后要 cd 进去**，旧 `out/play200.py` 的任务名 `smoke_roof_41` 已不存在（现为 `demo_roof_41_doom_m200`）；镜像前后留档 `docker-test/out/retest2/images-before.txt`、容器日志 `slim.log` / `2204-install.log`。
- 2026-10-26 00:48 **E 层族名对齐主控并重打正式包（取代 `7be31b84…`）**：`demo_conv_30/40_orig_m100` 改名 `demo_pool_30_orig_m100` / `demo_storm_40_orig_m100`（种子槽 81031000 / 81032000 不变），同步改了 PROMPT §1 表格、`python/test_llmplay.py`、DESIGN §14.31；生成器加 `LEGACY_TASK_IDS` 清掉旧条目。`check_tasks.py` 改用一次性预算会话（`PVZ_LLMPLAY_SESSION` 指临时文件）——宿主 `~/.llmplay_session/budget.json` 里 Codex 测试留下的墙钟已过 2:45，会把所有新开局拒掉（冷包自检同理，宿主侧跑要带这个变量；沙箱里 HOME 是新的，不受影响）。验收：`make_llmplay_tasks.py` 幂等 0 新增、`python/test_llmplay.py` **40 条全过**、`check_tasks.py` **33/33 OK**（含改名后两个 E 层族起局）。**事故与恢复**：重打时 `build_package.py` 先清空 out 目录，把 stage 里的引擎和旧 zip（`7be31b84…`）一起删了；引擎 `a84f10cf…` 从 `mac_bundle/pkg/build/pvz-portable` 按 SHA 核对后恢复，引擎字节未变。新包 `~/pvz-logs/llmplay_pkg/llmplay_pkg.zip` = **44,649,577 字节 / 28 文件 / SHA256 `139f3ee55746741d3eb49e3662aa9d6890569a246233fc41962d5882a92af26b`**，凭据 0 命中、包内无 `demo_conv` 残留、`PROMPT.md` / `configs/llmplay_seeds.json` 与工作树逐字一致；冷解压（zipfile，644 权限）`unshare -rn` L41 200 步 **181 操作 / tick 9441 / wave 3 / result 2** + `SELFCHECK OK`；导出存档回**独立引擎**（hostbuild2 `e849cf8f…`）`verify` 通过 `{"verified": true, "operations": 181, "tick": 9441, "wave": 3, "result": 2}`。Mac 包 `7e7ff496…` 里的引擎同为 `a84f10cf`，不受本次改名影响，未重打。
- 2026-10-26 03:10 **新包补跑两个断网容器（task81 §3 口径，之前记的缺口已关闭）**：自建隔离 daemon（官方静态 docker 29.9.2，data / exec / socket 全在 `docker-test/`，`--bridge=none --iptables=false --ip6tables=false`）。**坑**：本机留的那份 `docker-29.9.2.tgz` 是截断的（76,574,720 / 应为 86,374,273 字节，`tar` 报 `unexpected EOF`），已重新下载。两个 `--network none` 容器里都只用包里东西：**python:3.11-slim**（Python 3.11.17）与 **ubuntu:22.04 + python3**（3.10.12，装好后 `docker commit` 成 `lp22py:final`），都用 Python 的 `zipfile` 解压（引擎权限 **644**，正好验接口自补可执行位）→ `SELFCHECK OK`（含稳健性两次结果一致）→ L41 200 步 **181 操作 / tick 9441 / wave 3 / result 2**。两份存档回台式机用**独立引擎**（hostbuild2 `e849cf8f…`，不是包里那个）`verify` 都通过；两份 `ticks.jsonl` 与宿主机逐条比 `action/tick/end_tick/ok` **0 处不同**，只有 `state_hash` 181/181 不同（已知跨进程不可复现）。清理：容器（全 `--rm`）无残留，`docker rmi` 三个镜像、`docker builder prune -af`（回收 0B），删除前后 `docker images` / `docker ps -a` 落在 `docker-test/out/retest/`；停 daemon 后先 `umount` 残留的 `exec/netns/default`，再删 data / exec / docker 二进制与压缩包，目录 **311 MB → 2.9 MB**（只剩日志、证据与重跑脚本）。证据：`docker-test/out/retest/{slim,2204}/l41/`。
- 2026-10-26 02:50 **修掉稳健性试打污染止损判据（重打最终包）**：复核时发现 `robustness()` 的内部试打局会调用 `session.record_wave()` —— 换过未来的试打如果打到更高波，会把 `budget.json` 的 `last_progress_wave` 顶上去并清零 `segment_rewrites`，等于骗过 §A 的「连续两个检查点无提升」判据。现在内部局（`_internal`）不记波次、不写最远进度、不进自动 observe；补 1 条单测（试打后 `last_progress_wave` / `max_wave` 不变），`python/test_llmplay.py` **40 条全过**。重打：`llmplay_pkg.zip` = **44,649,552 字节 / 28 文件 / SHA256 `7be31b84864b189cb2d584226a2ddc2cad028898bb542e782b29c9cc441e900a`**；冷解压 `unshare -rn` `SELFCHECK OK`；导出存档回独立引擎 `verify` 通过（181 操作 / tick 9441 / wave 3 / result 2）；`robustness_check.py` 三条仍全 OK；28 成员 CRC 通过、凭据 0 命中、禁入路径 0。`mac_bundle.tar.gz` 同步重打：**48,771,577 字节 / SHA256 `7e7ff496b5176c1681538d53d7abfc323e86dd294994c901616313fe041b36cc`**，包内引擎 4/4 PASS / `MACCHECK OK`。
- 2026-10-26 02:35 **task81c / task81d 验收矩阵（一屏看全）**：任务书 = `scripts/llmplay/PROMPT.md`（237 行）；交付包 = `~/pvz-logs/llmplay_pkg/llmplay_pkg.zip`，44,649,552 字节，SHA256 `7be31b84…`。

  | 任务书条目 | 状态 | 证据 |
  |---|---|---|
  | 81c-1 每任务 40 种子、新独立块、机械核对、按 AGENTS §8 登记 | 完成 | 33 族 × 40 = 1,320 种子（81,000,000–81,032,039）；全仓 1990 JSON / 4,376,583 条外来命中 0；DESIGN §14.31 |
  | 81c-2 A–E 五层写进 PROMPT，逐层推进（每任务 5 胜、其中 3 个达稳健） | 完成 | PROMPT §1 表格 + 推进条件 |
  | 81c-3 C 层 3.0/4.0 倍新族、D 层 fog79 2.0 倍新族，DESIGN 同步、无头 reset 通过 | 完成 | 8 + 8 族，`check_tasks.py` 33/33 OK |
  | 81c-4 E 层原生传送带 | **部分** | L30/L40 已可跑并登记（reset + 120 tick OK）；L50 待 boss50 合入 |
  | 81c-5 重打正式包、断网自检、记录 SHA、push | 完成 | `7be31b84…`；`unshare -rn` `SELFCHECK OK`；已 push |
  | 81c-6 PROMPT 两条硬话 + 多样性 | 完成 | PROMPT §1 末两条、§6 末条 |
  | 81d-1 `robustness()`（检查点 0/10/20/30、k=8、RESEED、闭环、计预算、独立种子块） | 完成 | `robustness_check.py`：可复现 / RESEED 只改未来 / 预算 +1,200 全 OK |
  | 81d-2 export_archive 记录 `robustness` + `policy_sha256` + `manual_patch` | 完成 | 导出存档 `verify` 通过；rewind 局实测 `manual_patch: true` |
  | 81d-3 PROMPT 四句原样 + 分层推进条件 | 完成 | PROMPT 开头、§4、§1 |
  | 81d-4 战损/丢车/被吃只进 report | 完成 | PROMPT §4 末段 |
  | 81d-5 测试（可复现 / RESEED 语义 / 预算）+ 重打包 + push | 完成 | `python/test_llmplay.py` **39 条** + `robustness_check.py` |

  剩余只有 §2 的 **Mac 上重放 4 局**（环境阻断，台式机两个引擎各 4/4 通过、Docker 分支 4/4 通过，自包含包 `7e7ff496…` 已备好）。
- 2026-10-26 02:20 **最终包（排除三个开发脚本）**（已被 02:50 的 `7be31b84…` 取代：修掉试打污染止损判据）：`~/pvz-logs/llmplay_pkg/llmplay_pkg.zip`，**44,649,473 字节**，**28 文件**，SHA256 **`5b882359ae3063f97e03010e957005fd79d1a490b337e64479b4d494afe41178`**（取代 01:30 的 `ddfe0f9f…`）。原因是 `build_package.py` 的忽略列表漏了 `make_llmplay_tasks.py` / `check_tasks.py`（会随 `llmplay/` 进包，前者还会改写清单），一并把 `robustness_check.py` 也排除；包里现在只有 `__init__/budget/game/scene/selfcheck` 五个模块。验收：冷解压 `unshare -rn python3 selfcheck.py` → **`SELFCHECK OK`**（33 任务、181 操作、稳健性两次一致）；导出存档回台式机用独立引擎（hostbuild2）`verify` 通过 `{"verified": true, "operations": 181, "tick": 9441, "wave": 3, "result": 2}`；28 成员 CRC 通过、凭据模式 0 命中、禁入路径 0。**Mac 自检包同步重打**：`mac_bundle.tar.gz` = **48,772,851 字节**，SHA256 **`f31cc5290d4c93b885f874ec5b96a2731eb487b94db3bae7eaebfe25e8375131`**，46 文件 / 68 项、无 `__pycache__`、凭据 0 命中，包内引擎本地模式 **4/4 PASS / `MACCHECK OK`**（39,590 / 57,596 / 41,107 / 41,831）。两个包都只在本机，不进 git、不上 HF。
- 2026-10-26 02:00 **稳健性三条实测脚本**：新增 `scripts/llmplay/robustness_check.py`（要引擎、单进程）。在 L41 doom 族上实测：**可复现 OK**（同打法同种子两次 `robustness()` 结果逐字相同）、**预算 OK**（搜索 tick +1,200 ≥ 2×2×300）、**RESEED 只改未来 OK**（换种子前后 tick / 阳光 / 植物数不变，2,400 tick 后的僵尸类型不同），末行 `ROBUSTNESS CHECK OK`。
- 2026-10-26 01:45 **Mac 自检包随新正式包重打**（已被 02:20 的 `f31cc529…` 取代）：`mac_bundle.sh` 用新 stage 重打，`~/pvz-logs/llmplay_pkg/mac_bundle.tar.gz` = **48,776,677 字节**，SHA256 **`40c79c8b39a550843b91ad74890fa192ae4513f4e9d2e12cf4742608aea7162c`**，解出 **48 文件 / 70 项**、无 `__pycache__`；包内引擎本地模式重放 4 局 **4/4 PASS**（39,590 / 57,596 / 41,107 / 41,831，tick 39,479 / 57,363 / 40,969 / 41,682，四局 `result=1`，末行 `MACCHECK OK`，4 分 51 秒）。凭据模式 0 命中、禁入路径 0；`run.sh` 里的 Apple Silicon `--platform linux/amd64` 分支仍在。**Mac 一项仍需在 Mac 上执行**，`问题` 节里的路径与 SHA 已同步。
- 2026-10-26 01:30 **重打正式包（task81c + task81d）**（已被 02:20 的 `5b882359…` 取代：`llmplay/` 里误带了两个开发脚本）：`/home/newbiexvwu/pvz-logs/llmplay_pkg/llmplay_pkg.zip`，**44,654,631 字节**，**30 文件**，SHA256 **`ddfe0f9f331d4f661835f0bdb67a78f4739163161b66024f54535fccb574b836`**。包内 33 个任务（A4 / B11 / C8 / D8 / E2），每任务 **40 种子**（`extra_seeds.json` 与 `llmplay_seeds.json` 同块），稳健性配置 4 检查点 × 8 未来；引擎未动（`a84f10cf…`，与 21:20 4/4 PASS 的是同一份字节）。冷解压（Python `zipfile`，引擎权限 644 → 接口自补可执行位）后 `unshare -rn python3 selfcheck.py` → **`SELFCHECK OK`**（33 任务、181 操作、稳健性两次结果一致、meta 含 `robustness`/`manual_patch`）；同一份冷解压导出的存档回台式机用**独立引擎**（hostbuild2 `e849cf8f…`）`human_demo.py verify` 通过：`{"verified": true, "operations": 181, "tick": 9441, "wave": 3, "result": 2}`。整包扫描：**30 成员 CRC 全通过、凭据模式（`hf_`/`sk-`/`AIza`/`ghp_`/`password=`）0 命中、禁入路径 0**；包内 `llmplay/game.py`、`PROMPT.md`、`configs/llmplay_seeds.json`、`configs/llmplay_robustness.json` 与 HEAD **逐字一致**。包仍只在本机 `~/pvz-logs/llmplay_pkg/`，不进 git、不上 HF。
- 2026-10-26 01:20 **task81d `robustness()` 与导出 meta**（commit 92593c6b）：`Game.robustness(policy, k=8, checkpoints=[0,10,20,30])` —— 对每个**已打到**的检查点（波开始时）读档 → `RESEED` 成 k 个新未来（种子取 `configs/llmplay_robustness.json` 的独立块 **83,000,000–83,000,307**）→ 用模型自己的**闭环**函数打到终局，返回每个检查点的胜局数。一次调用不推进局面直接报错（固定动作序列在新未来里没有意义）；调用计入搜索预算（实测 2 个未来 × 300 tick → `search_ticks` +600）；内部试打局不写「最远进度」、不进自动 observe、不能用 rewind/save/export（RESEED 后日志不可重放）；`rewind` 自动把局标 `manual_patch`。导出 meta 增加 `robustness`（含 `policy_sha256`、`passed` = 第 10 波 8 个新未来 ≥5 胜）与 `manual_patch`。实测：同打法同种子两次结果逐字相同；RESEED 前后 tick/阳光/植物数不变、出怪种类改变；`manual_patch` 真值路径实测为 True。单测 **39 条全过**。
- 2026-10-26 00:45 **task81c 任务分层与种子块登记**（commit 081e1c05）：新增 `scripts/llmplay/make_llmplay_tasks.py`（可重复执行）与 `configs/llmplay_seeds.json`。33 个任务分五层：**A** 屋顶 L46–L49 doom m200（4）；**B** 屋顶 L42–L44 全部卡组变体 + 新登记 `demo_roof_41_doom_m200`（11）；**C** 新族 L46–L49 doom `_m300`/`_m400`（8）；**D** 新族 fog79 L36–L39 `_m200`（8）；**E** 原生传送带 `demo_conv_30_orig_m100`/`demo_conv_40_orig_m100`（2，`deck: []` + `conveyor_start: "original"`，命名避开 task75 的 `*_orig_*`）。种子新块 **81,000,000–81,032,039**（每任务 40、步长 1000）：全仓 1990 份 JSON / 4,376,583 个种子条目，排除本次新文件后落在 [81.0M, 81.1M) 的**外来种子 0 个**；稳健性块 **83,000,000–83,000,307**（4 检查点 × 100 + 8）同样 0 冲突。`scripts/llmplay/check_tasks.py` 对 33 族各 reset + 120 tick **全部起局成功**（3.0/4.0 倍、human_v2、传送带均正常，L30 在 120 tick 已进第 1 波）。DESIGN §14.31 已登记；L50 待 boss50 分支合入。
- 2026-10-25 23:15 **Mac Docker 分支补 Apple Silicon 平台**（接手 Codex 未提交的改动）：`mac_bundle.sh` 生成的 `run.sh` 在 `uname -m` 为 `arm64` / `aarch64` 时给 `docker run` 加 `--platform linux/amd64` —— 包里是 x86_64 静态 ELF，Apple Silicon 默认拉 arm64 镜像会 `exec format error`；x86_64 上变量为空，命令与 22:40 实测那条逐字相同。`bash -n` 通过，并用假 `uname` 注入 arm64 验证分支命中。重打 `mac_bundle.tar.gz`（**48,754,632 字节**，SHA256 `abe4001b…`），解出 **44 文件 / 66 项**、无 `__pycache__`；用包内引擎本地模式重放 4 局 **4/4 PASS**（39,590 / 57,596 / 41,107 / 41,831，末行 `MACCHECK OK`）。凭据模式（`hf_` / `sk-` / `AIza` / `ghp_` / `password`）**0 命中**，`token` 45 处全是 `branch_action_token` 这类代码标识符；禁入路径 0。包仍在 `~/pvz-logs/llmplay_pkg/`，不进 git、不上 HF。
- 2026-10-25 22:55 **Mac 包安全扫描 + 单测复跑**：`mac_bundle.tar.gz` 解出 **44 个文件**（含目录 65 项），凭据模式（`hf_` / `sk-` / `AIza` / `ghp_` / `token` / `password`）**0 命中**，禁入路径（`.git` / `artifacts` / `logs` / `.pt` / `.npz` / `.log` / `teacher-env`）**0**；含 `main.pak` 与引擎，只在本机，不进 git、不上 HF。`python/test_llmplay.py` **33 条全过**（改打包器 ignore 列表后复跑）。
- 2026-10-25 22:50 **包可复现性检查**：`build_package.py` 的 ignore 列表补上 `mac_bundle.sh`（新脚本不该进包；当前交付包是在它出现之前打的，不受影响）。用交付包里的引擎重建到 `/tmp/pkgrebuild`：**26 文件、44,636,271 字节**，与 `~/pvz-logs/llmplay_pkg/llmplay_pkg.zip` **逐成员 CRC 全同**、文件数/大小一致，只有 zip 内时间戳不同导致 SHA 不同（重建 `8dc52edd…` vs 交付的 `1000afb7…`）；密钥扫描 0 命中。重建产物已删，交付包未改动。
- 2026-10-25 22:40 **Mac 自检包的 Docker 分支实测 + 清理**：用本目录的隔离 daemon（静态 docker 29.9.2，data/exec/socket 全在 `docker-test/`，`--bridge=none --iptables=false`）拉 `python:3.11-slim`（189 MB / 内容 47.8 MB），把 bundle 挂到 `/w` 跑 `pip install numpy` + `bash /w/run.sh local /w/pkg/build/pvz-portable`：**4/4 PASS、`MACCHECK OK`**，操作数 39,590 / 57,596 / 41,107 / 41,831，与宿主机结果逐条相同。**注意**：`--bridge=none` 下容器没有网络，`pip install numpy` 会失败（实测报 `Temporary failure in name resolution`），所以本轮用 `--network host` 只给 pip 联网；Mac 的 Docker Desktop 默认有网，`./run.sh` 不用改。清理：`docker ps -a` 空、`rmi python:3.11-slim`、`builder prune -af`（回收 0B），删除前后 `docker images` 存 `docker-test/out/macbundle/{images-before,images-after,ps-before}.txt`（前 1 个镜像 189 MB，后空）；停 daemon 后 `umount` 掉 `exec/netns/default`，再删 `data/exec/docker/tgz`，`docker-test` **310 MB → 2.0 MB**；解压副本 `mac_bundle/`（112 MB）删掉，只留 `mac_bundle.tar.gz`（47 MB）。
- 2026-10-25 22:20 **Mac 自检做成自包含包（修掉"干净 Mac 跑不了"的缺陷）**：`scripts/llmplay/mac_check.sh` 要求 4 局示范在 `artifacts/human_demo/raw/`，但那些存档**不入 git**（AGENTS §1），所以在一台新 clone 的 Mac 上必然 `MISSING`。新增 `scripts/llmplay/mac_bundle.sh <stage> <示范根> <输出目录>`：把正式包内容 + 4 局示范 + verify 依赖模块（`human_demo.py`、`pvz_event_env.py`、`pvz_wait_events.py`、`pvz_frame_store.py`）+ 一个 `run.sh` 装成一个目录并打成 tar.gz。产物 `~/pvz-logs/llmplay_pkg/mac_bundle.tar.gz`，**48,755,133 字节**，SHA256 `27e12adf48f7de97ffadab483211a9282822dc32010fed595531eb7390d314e9`，解压 112 MB / 65 项；含 `main.pak`，只在本机，不进 git、不上 HF。`run.sh` 两种模式：默认 `./run.sh`（Docker：`python:3.11-slim` + `pip install numpy`，跑包里那个 Linux ELF）、`./run.sh local <Mach-O 引擎>`（Mac 本地编的引擎；检测到 ELF 会提示改用 Docker）。实测（Linux 本机、包内引擎）：4/4 `PASS`，操作数 39,590 / 57,596 / 41,107 / 41,831，tick 39,479 / 57,363 / 40,969 / 41,682，四局 `result=1`，末行 `MACCHECK OK`，4 分 25 秒。修了两个实测撞到的坑：① `pvz_env` 以 `cwd=引擎所在目录` 起进程，相对路径的 `--executable` 直接 `ENOENT`，`run.sh` 现在先转绝对路径；② 引擎把 INFO 日志混在 stdout，JSON 解析改成取最后一行 `{"verified"...}`。Docker 分支另测（见 22:40 条）：`python:3.11-slim` 容器里 4/4 PASS。**Mac 那一项仍需在 Mac 上执行。**
- 2026-10-25 21:50 **新包 `1000afb7…` 重跑 §3 的两个断网容器验收**（旧包 21:00 那轮是 `c912da70…`，包变了就要重跑）。按 21:00 的配方自建隔离 daemon（官方静态 `docker-29.9.2.tgz`，data-root/exec-root/socket 全在 `~/pvz-logs/llmplay_pkg/docker-test/`，`--bridge=none --iptables=false --ip6tables=false`，不碰宿主网络）：
  ① **ubuntu:22.04**（镜像里没 python3，先 `--network host` 装 python3 3.10.12 再 `docker commit` 成 `lp22py:final`）→ `--network none` 容器内用 **Python 的 zipfile** 解压新包：`build/pvz-portable` 权限是 **644**（正好验证新加的自补可执行位），`python3 selfcheck.py` → `SELFCHECK OK`，L41 200 步 → **189 操作**、导出 `episode_archive_v1`（`label_source=llm_gpt_pro`）。
  ② **python:3.11-slim**（Python 3.11.17 / glibc 2.41）→ 同样 `--network none`、同样 zipfile 解压、`SELFCHECK OK`、189 操作。
  ③ 两份存档回台式机用**独立引擎**（`~/llmplay-build/hostbuild2/pvz-portable`，SHA `e849cf8f…`，不是包里那个）`human_demo.py verify`：**都通过** `{"verified": true, "operations": 189, "tick": 9955, "wave": 4, "result": 2}`；两份 `ticks.jsonl` 逐条比 `action/tick/end_tick/ok` **0 处不同**，只有 `state_hash` 189/189 不同（已知跨进程不可复现）。
  ④ **清理**：容器（全是 `--rm`）无残留；`docker rmi lp22py:final python:3.11-slim ubuntu:22.04`、`docker builder prune -af`（回收 0B）；删除前后 `docker images` 落在 `docker-test/out/docker-images-{before,after}-final.txt`（前 3 个镜像，后空）。停 daemon 后 `mount` 无残留挂载点，删掉 data/exec/docker 二进制与 86 MB 压缩包，目录 **310 MB → 2.0 MB**，只剩日志、两份存档、重跑脚本。
  证据：`docker-test/out/final/{2204.log,slim.log}` 与 `docker-test/out/final/{2204,slim}/l41/`。
- 2026-10-25 21:45 **重打正式包（取代 `c912da70…`）**：`/home/newbiexvwu/pvz-logs/llmplay_pkg/llmplay_pkg.zip`，**44,636,271 字节**，26 文件，SHA256 **`1000afb7363f2ec93cd66651b89832f68e7f18b7b9e6e68799d6968473868a83`**，密钥扫描 0 命中。
  改了三处，都是复核时实测撞到的：
  ① **引擎可执行位由接口自己补**（`llmplay/game.py` 新增 `_ensure_executable`，`_start()` 前调用）。Python 的 `zipfile` 解压**不保留权限位**，冷解压后 `build/pvz-portable` 是 0644，模型自写脚本直接 Popen 会报 `PermissionError`（以前只有 `selfcheck.py` 会 chmod，跳过自检就中招）。实测把包内二进制改回 0644 后跑整套 API 脚本，不再需要手工 chmod。
  ② **PROMPT 补两条**：`Game.load()` 是惰性的（第一次 `observe()`/`summary()`/`wait()` 前 `game.obs` 是 `None`）；可执行位不用自己补。
  ③ **`mac_check.sh` 不再进包**（它依赖仓库的 `human_demo.py` 与示范存档，沙箱里没有）；上一版是 26 文件，中途误打成 27 文件后已修掉。
  验收：`python/test_llmplay.py` **33 条全过**（新增 2 条可执行位测试）；冷解压到 `/tmp/pkg3` 后 `unshare -rn python3 selfcheck.py` → `SELFCHECK OK`（189 操作）；同一份冷解压跑整套 API 脚本（list_tasks / observe / legend / summary / timeline / legal / plant / wait / wait_until 字符串与回调两种 / rewind / branch / save / load / export_archive）全通，导出 37 操作存档；L41 200 步导出 **189 操作 / tick 9955 / wave 4 / result 2**。两份存档用**独立引擎**（`~/llmplay-build/hostbuild2/pvz-portable`，SHA `e849cf8f…`，不是包里那个）跑 `human_demo.py verify` 都通过：`{"verified": true, "operations": 37, "tick": 600, "wave": 0, "result": 0}` 与 `{"verified": true, "operations": 189, "tick": 9955, "wave": 4, "result": 2}`。包内引擎仍是 `a84f10cf…`（与 21:20 跑过 4/4 PASS 的同一份），引擎未动。
  整包重扫（含二进制与资源）：**26 成员、CRC 全通过、凭据模式 0 命中、禁入路径 0**（无 `.git` / `artifacts` / `logs` / `.pt` / `.npz` / `.log`）；包仍只在 `~/pvz-logs/llmplay_pkg/`，不进 git、不上 HF。
- 2026-10-25 21:27 **接手复核（Codex 停下的地方）**：工作树干净、HEAD `7428da79` 已推送；`python/test_llmplay.py` **31 条全过**；正式包 SHA256 仍是 `c912da70…`（44,635,895 字节），冷解压后包内 `llmplay/{budget,game,scene,selfcheck}.py`、`PROMPT.md`、`scripts/{episode_query,pvz_tasks,pvz_constants}.py` 与 HEAD **逐字一致**（没有包比代码旧的问题）；包内 `build/pvz-portable` 与 21:20 跑过 4/4 PASS 的 `stage/build/pvz-portable` **同 SHA256**（`a84f10cf…`），所以那轮 PASS 验的就是交付的这个字节序列。Mac 通路再确认一次仍不存在（无 sshd、`/etc/hosts` 无内网主机、`ip neigh` 只有 WSL 网关）。**唯一未完成项仍是 §2 的 Mac 重放**，已在“问题”补上 Mac + Docker 的一条命令，等主控决定是自己在 Mac 上跑还是改掉该子句。
- 2026-10-25 21:25 **验收矩阵（一屏看全）**：

  | 任务书条目 | 状态 | 证据 |
  |---|---|---|
  | §1 接口（Game/plant/shovel/wait/wait_until/rewind/branch/observe/legal/summary/export_archive） | 完成 | `python/test_llmplay.py` **33 条**全过；两个容器里 200 步实跑 |
  | §2 引擎二进制：静态、无窗口、glibc 兼容 | 完成 | `ldd: not a dynamic executable`；ubuntu 22.04（glibc 2.35）与 python:3.11-slim（2.41）均能 `RESET_V2` 起 L41 |
  | §2 台式机重放 4 局示范 | 完成 | `mac_check.sh` 对 deploy4 与包内二进制各 4/4 PASS（HEAD `8e98e01b` 前后） |
  | §2 **Mac 上重放 4 局** | **未完成（环境阻断）** | 本机无 Mac 通路：无 sshd、无 mDNS、无内网主机、`known_hosts` 只有 GitHub。**已备好自包含包**（22:20 条）：`mac_bundle.tar.gz` 拷到 Mac 上 `./run.sh` 一条命令，不用 clone 仓库 |
  | §3 两镜像断网自测 + 回台式机 verify | 完成 | 21:00 条（旧包）与 **21:50 条（当前包 `1000afb7…` 重跑）**：两镜像各 189 操作 / tick 9955 / wave 4 / result 2，独立引擎 `verified: true` |
  | §4 安全扫描 + 包只在本机 | 完成 | zip 全成员凭据模式 0 命中、禁入路径 0；包在 `~/pvz-logs/llmplay_pkg/`，不在 git、不在 HF |
  | §5 `scripts/llmplay/PROMPT.md` | 完成 | 主控 18:30 已审读，§G 三处意见已改并重打包 |
  | §6 主控验收记录 | 完成 | 包路径 / 44,636,271 字节 / SHA256 `1000afb7…` / 离线自测 / 逐位一致 / PROMPT 路径均在"结果" |

  只剩 Mac 一行。要关闭它只有两条路：① 在 Mac 上跑 `scripts/llmplay/mac_check.sh <资源目录>`（台式机已两次 4/4 PASS，脚本可用）；② 主控认为包里发的是 Linux ELF、Mac 那一条本就不适用于本交付物，在任务书里改掉 §2 的那句。**我没有自行改动 §2**，因为那是主控定的验收条件。
- 2026-10-25 21:20 **同一脚本对"包里那个二进制"再跑一遍**：把 4 份示范拷到 `~/pvz-logs/llmplay_pkg/maccheck_archives/`（不写别的工作树），用 `scripts/llmplay/mac_check.sh` 配**包内静态二进制**（`stage/build/pvz-portable`，SHA `a84f10cf20bd`）重放：4/4 `PASS`、`MACCHECK OK`，四局操作数 39,590 / 57,596 / 41,107 / 41,831，tick 39,479 / 57,363 / 40,969 / 41,682，`result=1` 全通关。加上 deploy4 那一轮，台式机上已有两个互不相同的引擎各自 4/4 通过。
- 2026-10-25 21:20 **§2 的 Mac 子句需要澄清**：包里发的是 **Linux 静态 ELF**（`ldd` 报 `not a dynamic executable`，`file` 报 `ELF 64-bit LSB executable`），macOS 内核不能直接执行它，所以"用这个二进制在 Mac 上重放"只有两种可能：① Mac 上装 Docker/虚拟机跑这个 ELF；② Mac 用它自己从同一份源码编出来的 Mach-O 引擎。无论哪种都要在 Mac 上操作，本机无法代办。这一点写进"问题"，请主控在下一轮明确要哪一种。
- 2026-10-25 21:15 **Mac 检查做成一条命令**：新增 `scripts/llmplay/mac_check.sh`（参数：资源目录、引擎；默认读 `$PVZ_RESOURCE_DIR` 与 `build/pvz-portable`，可用 `PVZ_DEMO_RAW` 指向存档根）。它按 seed 找到 4 份示范、逐份 `human_demo.py verify`，核对 `operations` 是否等于 39,590 / 57,596 / 41,107 / 41,831，每局打一行 `PASS`，末行 `MACCHECK OK` / `MACCHECK FAILED`。**在台式机上用 deploy4（16989f23cdce）实测 4/4 PASS、`MACCHECK OK`**，四局 `verified:true` 且 tick/wave/result 与记录一致，说明脚本本身可用；Mac 上只需跑 `scripts/llmplay/mac_check.sh <资源目录>` 一条命令。仍未解决的是**在 Mac 上执行**这一步：本机 `sshd` 未安装、WSL 与 Windows 的 interop 已坏（`cmd.exe` 也报 `UtilAcceptVsock: accept4 failed 110`）、`/etc/hosts` 无内网主机、`who` 无外部登录，没有任何到 Mac 的通路。
- 2026-10-25 21:00 **补齐 Docker 两镜像断网验收**（前一条"环境证据缺失"已解决）。不用 Docker Desktop（WSL 集成未开、`docker.exe` 走 vsock 报 110），改用官方静态包 `docker-29.9.2.tgz` 自建**隔离 daemon**：data-root / exec-root / socket 全在 `~/pvz-logs/llmplay_pkg/docker-test/`，启动参数 `--bridge=none --iptables=false --ip6tables=false`，不碰宿主网络、不装系统包、不建 `/var/run/docker.sock`。
  ① **ubuntu:22.04**：基础镜像里**没有 python3**（实测 `python3: command not found`），先验证静态引擎本身——`ldd` 报 `not a dynamic executable`，直接喂 `RESET_V2 41 20261041 2 7 0 0 0 0 0 - - 0,1,2,3,4,5,33 1 0 -` 拿到 `ok:true` 与 `level:41 / terrain:4 / roof:true / wave_count:20`，说明 glibc 2.35 上可跑；随后在同一镜像装 python3（3.10.12）并 `docker commit`，再以 `--network none` 离线跑包：`SELFCHECK OK`（189 操作），L41 200 步导出 `episode_archive_v1`（189 操作 / tick 9955 / wave 4 / result 2）。
  ② **python:3.11-slim**（Python 3.11.17 / glibc 2.41）：全程 `--network none`，容器内用 zipfile 解压 zip → `selfcheck.py` `SELFCHECK OK`（189 操作）→ 200 步导出（同样是 189 操作 / tick 9955 / wave 4 / result 2）。两份存档回台式机用正式引擎（deploy4，SHA `16989f23cdce`）`human_demo.py verify` **都通过**：`{"verified": true, "operations": 189, "tick": 9955, "wave": 4, "result": 2}`；两份 `ticks.jsonl` 逐字段比对只有 `state_hash` 不同（已知的跨进程不可复现问题），action / tick / ok 189/189 完全一致。
  ③ **清理**（§E）：`docker rm` 容器、`docker rmi` 三个镜像（lp22py:local / python:3.11-slim / ubuntu:22.04）、`docker builder prune -af`（回收 0B）；删除前后的 `docker images` 分别落在 `docker-test/out/docker-images-before.txt`（3 个镜像）与 `docker-images-after.txt`（空）；停 daemon 后先 `umount` 残留的 `exec/netns/default`（nsfs）再删 data/exec 与 225 MB 二进制、83 MB 压缩包，目录从约 390 MB 降到 82 MB；确认证据完整后再删掉解压副本 `extract/`，最终只剩 1.1 MB（日志、两份存档、重跑脚本）。
  证据：`docker-test/out/{selfcheck-2204.log,play200-2204.log,verify-2204.json,selfcheck-slim.log,play200-slim.log,verify-slim.json,docker-images-before.txt,docker-images-after.txt}`；重跑脚本 `docker-test/start_dockerd.sh` 与 `docker-test/out/play200.py` 留在本机，不进 git。**Mac 侧仍未验**（见"问题"），其余 §2 / §3 / §E 项至此都有实测记录。
- 2026-10-25 **Codex 接手验收修复**（代码提交 `57eec964`）：2:45 门禁的旧表达式 `2 * 60 * 45` 实际是 90 分钟，现改为 165 分钟；状态行改为时:分。搜索关闭时切手动，连续两个无进展检查点可提前止损，重写达到上限后允许退到更早 tick。最远操作日志按关卡跨脚本/分支保留，耗尽时从纪录导出；真实引擎验证当前 tick 120、自动导出 tick 300，正式引擎 `verify` 通过（5 操作）。预算首次创建与图例标记改用同一 flock 事务，日志原子写入；读档/rewind/快照恢复重建逐波历史，observe 附历史，植物类型 0 不再误判为空槽。`python/test_llmplay.py` **31 条全过**；人类示范日志实测两次读档历史一致、快照 rewind 到 tick 3000 后历史一致且无未来行。
- 2026-10-25 **上一版正式包**（已被 21:45 的 `1000afb7…` 取代）：`/home/newbiexvwu/pvz-logs/llmplay_pkg/llmplay_pkg.zip`，**44,635,895 字节**，26 文件，SHA256 **`c912da70a36012b465f088c360e8744229e5e08c019369b925018a34a8f4a32c`**。冷解压后 `unshare -rn` 断网 `selfcheck.py` 通过；L41 200 步入口生成 189 操作，tick 9955 / wave 4 / result 2，存档回正式引擎 `human_demo.py verify` 通过。对 zip 所有成员（包含二进制与资源）扫描凭据模式 0 命中，禁入路径 0，CRC 检查通过。预编引擎指纹保持 `a84f10cf20bd6bb6ba445da389dd282b95e60a2105acee8cf00ee1142cd6f0eb`；断网重放四局示范再次 ALL PASS：39,590 / 57,596 / 41,107 / 41,831 操作，826 / 1,309 / 892 / 922 帧观测全部一致，四局均通关。`state_hash` 仍仅作诊断，Docker / Mac 未测的限制见“问题”。探路包保持原路径与 SHA256；任务书为 `scripts/llmplay/PROMPT.md`。本次游戏进程最多 2 个，未操作其它任务的运行进程。

- 2026-10-25 17:05 接口骨架 + 打包器 + 单测（commit 58656c0a，已推 llmplay）：`scripts/llmplay/` 的 Game/observe/legal/summary/rewind/branch/save/load/export_archive，任务书 PROMPT.md，build_package.py，`python/test_llmplay.py` 8 条全过。合法动作直接取引擎 lean 观测里的 `legal_actions`（Board::CanPlantAt），不在 Python 里重写种植规则。
- 2026-10-25 17:05 构建环境：Docker 不可用（Docker Desktop 未对本 WSL 发行版开集成，`docker.exe` 走 vsock 报错）。改用 ubuntu 22.04 rootfs（ubuntu-base-22.04.5）chroot；jammy 的 GCC 11 编不过引擎（`#include <format>` 是 C++20，libstdc++ 11 没有），于是改为宿主 GCC 15 + 全静态链接（`-static`）——静态二进制不依赖目标机 glibc，比"在 22.04 里编"更稳。静态 SDL2 另编（只留 dummy 视频/音频后端，见下一条）。
- 2026-10-25 18:00 **探路包已出**（§F）：`~/pvz-logs/llmplay_pkg/probe/llmplay_probe.zip`，52,078,406 字节，SHA256 `380a0111e62bc091c91e9ed87030e2da0b8ab9d7e2212bd606e3f13c3636b66f`，825 个文件，源码 rev `8bf65b19`。断网（`unshare -n`）实测：`build.sh` 用 GCC 15.2 + cmake 4.2.3 编译 72 秒成功；`selfcheck.py` 对新编引擎与预编静态二进制各跑一遍，4 局人类示范全部 PASS（时间线 39,590–57,596 个操作逐条一致、每个决策点的阳光/波次/植物数/僵尸数一致、终局 tick/波数/结果一致）。判据按 §F 改成"结果一致"，state_hash 只作诊断。
- 2026-10-25 18:05 接口 §A/§B/§D 重写（commit 5d0d14a7）：observe 改为状态行 + 局面头 + 5×9 网格 + 逐行明细 + 卡片 + 事件 + timeline；预算/检查点/rewind 上限/2:45 门禁全部落在 `llmplay_session/budget.json`，开场实测吞吐（约 650–3000 tick/s）换算 25 分钟搜索预算；脚本退出时自动打印一次完整 observe。测试 13 条全过。
- 2026-10-25 20:15 主控通知 Codex 额度恢复：当前进度已全部提交并推送到 `origin/llmplay`（HEAD `3f3493a2`），工作树干净；最终正式包 `~/pvz-logs/llmplay_pkg/llmplay_pkg.zip`，44,635,183 字节，SHA256 `39fe7f11dc3ef31efa296ddec8c09141a313f8399b3e1e138f20f076b8f579c5`，冷解压 `selfcheck.py` 通过。task81 按主控指示在此停止，后续交给 Codex。
- 2026-10-25 20:10 **修掉分支持数跨脚本累积**：`branches` 原来写进 `budget.json`，但分支引擎随进程退出就没了；上一条脚本开过 8 个分支后，下一条脚本一开局就永远被 `branch()` 拒绝（即使一个分支都不存在）。改成**进程内计数**，进程退出自然归零；`close()` 仍然减一。单测补 2 条（分支计数不落盘、关掉一个后可再开），23 条全过。**最后一次重打正式包**（取代 20:00 的 `fbff2e60…`，主控 20:05 指示这是 task81 的收尾版）：44,635,183 字节，SHA256 `39fe7f11dc3ef31efa296ddec8c09141a313f8399b3e1e138f20f076b8f579c5`，26 文件，密钥扫描 0 命中；冷解压后断网 `selfcheck.py` 通过（L41、189 操作、tick 9955、result 2）。task81 到此完成，不再继续打磨。
- 2026-10-25 20:00 PROMPT 里的预算文件路径同步成 `~/.llmplay_session/budget.json`（原来还写着旧的相对路径 `llmplay_session/budget.json`）。**重打正式包**（取代 19:50 的 `1ac479aa…`）：44,635,078 字节，SHA256 `fbff2e6012b3af5bf155709f93e41dd9583363f8ff8edf121702c8cf77883e87`，26 文件，密钥扫描 0 命中；冷解压后断网 `selfcheck.py` 通过。
- 2026-10-25 19:50 **修掉 3 个子代理并发写预算的丢更新**：3 个子代理是 3 个进程，原实现每个进程只在启动时读一次 `budget.json`，之后每步把整份内存状态写回去 —— 后写的会覆盖先写的，预算/墙钟/rewind 计数都会少记，等于架空 §A。改成：所有读改写走 `flock` 事务（加锁 → 重读磁盘 → 改 → 原子替换写回），读入口（状态行/门禁）先 `refresh()`；`save()` 改为临时文件 + `os.replace`。单测补 1 条并发测试（4 进程 × 50 tick，必须正好 200），21 条全过；包内同样实测 4 进程 × 50 = 200。**重打正式包**（取代 19:35 的 `8f62690b…`）：44,635,054 字节，SHA256 `1ac479aa1695afad2d6357a7d2dcf61ef0575572d5d11207ad7fe4634f16a3fa`，26 文件，密钥扫描 0 命中；冷解压后断网 `selfcheck.py` 仍为 `SELFCHECK OK`。
- 2026-10-25 19:35 **修掉"换 cwd 就重置预算"的漏洞**：会话文件原来固定在 `当前目录/llmplay_session/budget.json`，模型只要从另一个目录跑脚本（沙箱里很常见：包在 `/mnt/data/llmplay_pkg`、输出在 `/mnt/data/out`），墙钟、搜索预算、rewind 计数就全部清零 —— 直接架空 §A 的止损。改成固定写 `~/.llmplay_session/budget.json`（仍可用 `PVZ_LLMPLAY_SESSION` 覆盖；没有 HOME 才退回当前目录）。单测补 1 条（20 条全过）；跨进程实测：进程 A 在 `/tmp/cwd_a` 记 1234 tick，进程 B 在 `/tmp/cwd_b` 读到同一个文件、同一个 1234。**重打正式包**（取代 19:10 的 `231976a0…`）：44,634,576 字节，SHA256 `8f62690b80e3f0b269ce4d2900b78c5ffd83af920e4fbf474fd9fae08bafcbec`，26 文件，密钥扫描 0 命中；冷解压后断网 `selfcheck.py` 仍为 `SELFCHECK OK`。
- 2026-10-25 19:20 **跨进程持久化实测**（设计前提是"工具调用之间进程可能被回收"）：进程 A 在 L46 种下、铲掉（植物 16→15）、走 300 tick 存 `run.json`（7 操作）；进程 B 重新加载后 `wait_until(360)`（整数形式）、再走 60 tick、导出存档（12 操作），回台式机 `verify` 通过（tick 420 / result 0）。`shovel()` 与 `wait_until(tick)` 也一并验过。
- 2026-10-25 19:10 **修掉退出自动 observe 的刷屏 + 全 API 实测**：搜索脚本开出多个 `branch()` 时，原实现会在退出时给每个推进过的局各打印一份完整 observe（实测 3 个局就淹掉脚本自己的输出）。改成只对**最后推进过**的那一局打印完整 observe，其余局打一行 `（本次脚本还推进了 N 个其它局：task@tick…）`；单测补 1 条（19 条全过）。用包里的引擎实测 `wait_until(回调)`、`rewind`、`branch`、`save`/`load`、`export_archive`：L46 走 300 tick 后 rewind 到 270、branch 推进到 450 且父局不动、load 后继续到 330，导出存档回台式机 `verify` 通过（7 操作 / tick 330 / result 0）。**重打正式包**（取代 18:45 的 `2842de59…`）：44,634,447 字节，SHA256 `231976a0fb229298f00e98cb912b7ca298522e719e0f1972927d13a7cf9b712f`，26 文件，密钥扫描 0 命中；冷解压后断网 `selfcheck.py` 仍为 `SELFCHECK OK`（L41、189 操作、tick 9955、result 2）。
- 2026-10-25 18:58 单测补 1 条 `PackageTest` 断言：任务清单必须含 `demo_roof_46_doom_m200` / `demo_roof_47_doom_m200`（守住 §G.1 的登记），`python/test_llmplay.py` 18 条全过。测试文件不进包，包无需重打。
- 2026-10-25 18:55 **正式包冷解压复测**：把 `llmplay_pkg.zip` 解到全新目录 `/tmp/pkgtest/`（不是打包器的 `stage/`），断网 `unshare -rn python3 selfcheck.py` 通过（L41、189 操作、tick 9955、result 2）；再从这份解压副本导出存档，回台式机 `human_demo.py verify` 通过（`{"verified": true, "operations": 189, "tick": 9955, "wave": 4, "result": 2}`）。确认 zip 自包含、目录层级与 `PROMPT.md` 的 `llmplay_pkg/` 路径一致。
- 2026-10-25 18:45 **按 §G 重打正式包**（取代 18:10 的 `e6c48503…`）：`~/pvz-logs/llmplay_pkg/llmplay_pkg.zip`，44,634,175 字节，SHA256 `2842de59cad4a885cd47c73271cd2ead93517b70d6a90fb117210cb78c3a1111`，26 个文件，密钥扫描 0 命中。改动：① 合入 `origin/pvz-env`（`ca5efd7e`），任务清单含 `demo_roof_46_doom_m200` / `demo_roof_47_doom_m200`，PROMPT 关卡表改为 L49 → L48 → L47 → L46；② 网格格内僵尸改成类型短代码+只数（`Gar1 Imp2`，放不下截断标 `+n`），图例补 33 个短代码；③ 截图按主控意见放弃，PROMPT 保留"本会话没有截图"。断网（`unshare -rn`）`selfcheck.py` 通过：L41 走 200 步、189 个操作、tick 9955、result 2；导出的存档回台式机用正式引擎 `human_demo.py verify` 通过（`{"verified": true, "operations": 189, "tick": 9955, "wave": 4, "result": 2}`）。单测 17 条全过（新增 4 条网格短代码/截断测试）。
- 2026-10-25 18:10 **正式包已出**：`~/pvz-logs/llmplay_pkg/llmplay_pkg.zip`，44,633,163 字节，SHA256 `e6c48503d69b63b19f172040583ca6a13bd9a2987e25df398f596dfbd1759fc8`，26 个文件（接口 + 引擎 + 资源 + 5 个常量来源 C++ 文件 + 任务清单 + PROMPT.md + 示例）。密钥扫描 0 命中（`hf_` / `sk-` / `AIza` / `ghp_` / `token` / `password`；命中值是函数调用的一律过滤，例如 `token = branch_action_token(action)`）。断网 `selfcheck.py` 通过（L41 走 200 步、189 个操作、导出存档），这份存档回台式机用正式引擎 `human_demo.py verify` 通过（tick 9955 / wave 4 / result 2，与导出时一致）。任务书 `scripts/llmplay/PROMPT.md`。


### observe() 样例（§B 要求主控审读；第 15 波的满场局面）

主控要求贴"L49 第 15 波"。L49 是 40 波 ×2.0 倍，打到第 15 波需要真策略，而本任务
不许手写打法，所以这里用**现成的人类示范**（L32 雾夜 30 波，同一套接口、同一份观测
代码）重放到第 15 波取的一帧。格式与 L49 完全一致，差别只在关卡、卡组与地形。
复现方式：把示范存档的 ticks.jsonl 读成 ops 传给 `Game(task_id, ops=..., tick=32051)`
（tick 32051 就是第 15 波那一帧，重放会停在那里），再 `observe()`。

```
[已用 0:29 / 3:00 | 本关 L32 模式=搜索 | 搜索预算 0% | 最远 15 波 | 本段重写 0/8]
L32 第 15/30 波 tick 32051 距下一波约 3091 tick 阳光 150 小推车 r0:待命 r1:待命 r2:待命 r3:待命 r4:待命 r5:待命 本关出怪 普通僵尸、路障僵尸、橄榄球僵尸、小丑僵尸 进行中

图例：网格每格 = 植物短代码+血量百分比（Mel85 = 西瓜剩 85% 血）与僵尸短代码+只数。
Gar1 = 1 只巨人，Gar1 Imp2 = 1 只巨人 + 2 只小鬼；格内按从左到右的威胁顺序，
放不下时截断并标 +n（n = 还有几只没显示）。
.. = 空地，~~ = 水路；r0–r5 = 行，c0–c8 = 列（c0 最左，僵尸从右往左走）。
血量都是当前/上限，位置列号保留一位小数。
僵尸短代码：
  Zom=普通 Flg=旗帜 Con=路障 Pol=撑杆 Buc=铁桶 Nws=报纸 Scr=铁门
  Ftb=橄榄球 Dan=舞王 Bkp=伴舞 Duk=鸭子救生圈 Div=潜水 Zam=冰车 Sld=雪橇
  Dol=海豚骑士 Cln=小丑 Bal=气球 Min=矿工 Pog=跳跳 Yet=雪人 Bun=蹦极
  Lad=梯子 Cat=投石车 Gar=巨人 Imp=小鬼 Bss=僵王博士 Pea=豌豆头 Wal=坚果头
  Jal=辣椒头 Gat=机枪头 Squ=窝瓜头 Tln=高坚果头 Red=红眼巨人

空间网格（植物 + 该格僵尸）：
        c0               c1               c2               c3               c4               c5               c6               c7               c8               
  r0    ShS100           ShS100           Sno100           Fum100           Puf100           Puf100           Puf99            ..     Con1 Zom1 ..               
  r1    ShS100           ShS100           Sno100           Fum100           Fum100           Puf60            Puf100           Puf100 Zom1      ..     Zom1      
  r2    Lil100           Lil100           Lil100           Lil100           Lil100           ~~               Lil100           ~~     Con1 Zom1 ~~               
  r3    Lil100           Lil100           Lil100           Lil100           Lil100           ~~               ~~               ~~     Zom3      ~~               
  r4    ShS100           ShS100           Sno100           Fum100           Fum100           Puf100           Puf52            ..               ..     Zom2      
  r5    ShS100           ShS100           Sno100           Fum100           Fum100           Puf57            Puf100           Puf81  Zom1      ..     Zom1      

逐行明细：
  r0 植物: c0 阳光菇 300/300 阳光菇成年 · c1 阳光菇 300/300 阳光菇成年 · c2 寒冰射手 300/300 正常 · c3 大喷菇 300/300 正常 · c4 小喷菇 300/300 正常 · c5 小喷菇 300/300 正常 · c6 小喷菇 296/300 正常
     僵尸: 路障僵尸 c7.5 本体32/270 护具0/370 减速 无头 | 普通僵尸 c7.5 本体270/270 | 路障僵尸 c9.5 本体270/270 护具370/370 | 普通僵尸 c9.5 本体270/270
     最前: 路障僵尸 约 234 tick 到第 6 列，约 2307 tick 到小推车
  r1 植物: c0 阳光菇 300/300 阳光菇成年 · c1 阳光菇 300/300 阳光菇成年 · c2 寒冰射手 300/300 正常 · c3 大喷菇 300/300 正常 · c4 大喷菇 300/300 正常 · c5 小喷菇 180/300 正常 · c6 小喷菇 300/300 正常 · c7 小喷菇 300/300 正常
     僵尸: 普通僵尸 c7.5 本体46/270 减速 无头 | 普通僵尸 c9.5 本体270/270
     最前: 普通僵尸 约 312 tick 到第 6 列，约 1765 tick 到小推车
  r2 植物: c0 睡莲 300/300 正常 · c0 阳光菇 300/300 阳光菇成年 · c1 睡莲 300/300 正常 · c1 阳光菇 300/300 阳光菇成年 · c2 睡莲 300/300 正常 · c2 寒冰射手 300/300 正常 · c3 睡莲 300/300 正常 · c3 大喷菇 300/300 正常 · c4 睡莲 300/300 正常 · c4 大喷菇 300/300 正常 · c6 睡莲 300/300 正常 · c6 高坚果 8000/8000 正常
     僵尸: 路障僵尸 c7.5 本体140/270 护具0/370 减速 | 普通僵尸 c7.5 本体170/270 | 路障僵尸 c9.5 本体270/270 护具370/370
     最前: 路障僵尸 约 181 tick 到第 6 列，约 2028 tick 到小推车
  r3 植物: c0 睡莲 300/300 正常 · c0 阳光菇 300/300 阳光菇成年 · c1 睡莲 300/300 正常 · c1 阳光菇 300/300 阳光菇成年 · c2 睡莲 300/300 正常 · c2 寒冰射手 300/300 正常 · c3 睡莲 300/300 正常 · c3 大喷菇 300/300 正常 · c4 睡莲 300/300 正常 · c4 大喷菇 300/300 正常
     僵尸: 普通僵尸 c7.5 本体150/270 减速 | 普通僵尸 c7.5 本体170/270 减速 | 普通僵尸 c7.5 本体130/270 减速 | 普通僵尸 c9.5 本体270/270
     最前: 普通僵尸 约 884 tick 到第 4 列，约 2105 tick 到小推车
  r4 植物: c0 阳光菇 300/300 阳光菇成年 · c1 阳光菇 300/300 阳光菇成年 · c2 寒冰射手 300/300 正常 · c3 大喷菇 300/300 正常 · c4 大喷菇 300/300 正常 · c5 小喷菇 300/300 正常 · c6 小喷菇 156/300 正常
     僵尸: 普通僵尸 c8.5 本体190/270 减速 | 路障僵尸 c9.5 本体270/270 护具370/370
     最前: 普通僵尸 约 524 tick 到第 6 列，约 2452 tick 到小推车
  r5 植物: c0 阳光菇 300/300 阳光菇成年 · c1 阳光菇 300/300 阳光菇成年 · c2 寒冰射手 300/300 正常 · c3 大喷菇 300/300 正常 · c4 大喷菇 300/300 正常 · c5 小喷菇 172/300 正常 · c6 小喷菇 300/300 正常 · c7 小喷菇 244/300 正常
     僵尸: 普通僵尸 c7.5 本体33/270 减速 无头 | 普通僵尸 c8.5 本体110/270 减速 | 路障僵尸 c9.5 本体270/270 护具370/370 | 普通僵尸 c9.5 本体270/270
     最前: 普通僵尸 约 359 tick 到第 6 列，约 2011 tick 到小推车

卡片：
  [0] 阳光菇 25☀ 就绪 买得起 可种 6 格（例 r0c7、r0c8、r1c8、r4c7）
  [1] 小喷菇 0☀ 就绪 买得起 可种 6 格（例 r0c7、r0c8、r1c8、r4c7）
  [2] 大喷菇 75☀ 冷却剩 60 tick 买得起 此刻没有可种的格
  [3] 睡莲 25☀ 就绪 买得起 可种 7 格（例 r2c5、r2c7、r2c8、r3c5）
  [4] 高坚果 125☀ 冷却剩 1059 tick 买得起 此刻没有可种的格
  [5] 窝瓜 50☀ 就绪 买得起 可种 6 格（例 r0c7、r0c8、r1c8、r4c7）
  [6] 缠绕水草 25☀ 就绪 买得起 可种 7 格（例 r2c5、r2c7、r2c8、r3c5）
  [7] 樱桃炸弹 150☀ 就绪 买得起 可种 6 格（例 r0c7、r0c8、r1c8、r4c7）
  [8] 寒冰射手 175☀ 就绪 阳光不够（差 25） 此刻没有可种的格
  [9] 坚果墙 50☀ 就绪 买得起 可种 6 格（例 r0c7、r0c8、r1c8、r4c7）

事件（自上次 observe 以来）：
  （自上次 observe 以来没有值得记的事件）
```

## 问题

- ~~**接手验收仍缺环境证据**~~（2026-10-25 21:00 已解决）：当时只试了 Docker Desktop 一条路（WSL 集成未开、`docker.exe` 走 vsock 报 110）。改用官方静态 Docker 自建隔离 daemon 后，`ubuntu:22.04` 与 `python:3.11-slim` 两个 `--network none` 测试、对应镜像清理记录都已补上，详见"结果"21:00 条。**唯一仍未验的是 Mac 侧**（下条）。
- **Docker Desktop 不可用，但 Docker 本身可用**（2026-10-25 21:00 更正）：`docker` 命令是 Docker Desktop 的 WSL 占位脚本，提示本发行版未启用集成，`/var/run/docker.sock` 与 rootless 套接字都不存在。绕开方式：下载官方静态包 `docker-29.9.2.tgz`，用 `sudo` 起一个**只在本目录**的 daemon（`--data-root/--exec-root/--host` 全指向 `~/pvz-logs/llmplay_pkg/docker-test/`，`--bridge=none --iptables=false`），不需要 Docker Desktop、不改系统网络、不装 apt 包。两个镜像的断网测试因此可以真做（见"结果"）。另：jammy 的 GCC 11 确实编不过引擎（`#include <format>` 要 GCC 13+），所以 ubuntu:22.04 里只验证了预编静态二进制；"从源码编"仍只能在目标沙箱里测。
- **截图（§C）跑不通，已按主控意见放弃**：`LawnApp::EnvironmentScreenshot` 要求非 headless（有窗口）并从 GL framebuffer 读回。实测 `SDL_VIDEODRIVER=dummy` 下引擎能起来、RESET_V2 正常，但 `SCREENSHOT` 返回 `ok:false`（静态 SDL2 只编了 dummy 视频后端，没有 GL）。要做成需要一套 EGL/Mesa 软件 GL + 带该后端的 SDL 构建，成本高。主控 18:30 决定放弃（§G.3），PROMPT 里保留"本会话没有截图"，只用文字观测。
- **清理（§E）**：截至 20:40 本机没有可用的 Docker，没有容器/镜像可删（21:00 自建 daemon 后的清理记录见"结果"：3 个镜像 + 1 个容器 + builder 缓存已删，删除前后 `docker images` 有留档）；已删 `~/llmplay-build/hostbuild`（旧构建树 81 MB）与 `/tmp` 下全部自测目录。`~/llmplay-build/rootfs`（2.4 GB，早先 22.04 chroot 用，文件属 root）已由主控于 18:30 代删。**教训**：chroot 用完必须先 `umount` `/proc`、`/sys`、`/dev`、`src` 这些绑定挂载，否则递归删除会穿过 `src` 删到整块数据盘；主控这次是先确认没有挂载点残留、再用 `rm --one-file-system` 删的。
- **Mac 侧没验（唯一未完成项）**：**2026-10-25 22:20 更新 —— 用自包含包，不要用下面的 `mac_check.sh`**：把 `~/pvz-logs/llmplay_pkg/mac_bundle.tar.gz`（**2026-10-26 02:50 最终版：48,771,577 字节，SHA256 `7e7ff496…`**；更早的 48,772,851 / `f31cc529…`、48,776,677 / `40c79c8b…`、48,755,133 / `27e12adf…` 已被取代）拷到 Mac，`tar xzf` 后 `cd mac_bundle && ./run.sh`（Docker 模式）或 `./run.sh local <Mach-O 引擎>`；期望四行 PASS + `MACCHECK OK`。下面的 `mac_check.sh` 写法只在"已经有仓库 + 示范存档"的机器上成立，示范存档不入 git，新 clone 的 Mac 会 `MISSING`。这台机器不是 Mac；`~/.ssh/config` 只有 GitHub 入口，`/etc/hosts` 无内网主机，本机没有 sshd 在监听、`who` 里也没有外部登录，因此没有可用的 Mac 验证入口。§2 的“Mac 上重放 4 局”仍需用户在自己的 Mac 上跑一次。**已备好一条命令**（在 Mac 的仓库根目录执行）：`scripts/llmplay/mac_check.sh <资源目录>`——它逐份 verify 四局、核对操作数并打 `MACCHECK OK`（台式机已用两个不同引擎各跑过一遍，均 4/4 PASS）。等价的手工命令如下（四份存档在 `artifacts/human_demo/raw/seed_2026103{1,2,3}/`）：
  ```bash
  R="$PVZ_RESOURCE_DIR"   # Mac 上的游戏资源目录
  python3 scripts/human_demo.py --resource-dir "$R" verify --archive artifacts/human_demo/raw/seed_20261031/demo_fog_31_snow_m200
  python3 scripts/human_demo.py --resource-dir "$R" verify --archive artifacts/human_demo/raw/seed_20261032/demo_fog_32_snow_m200
  python3 scripts/human_demo.py --resource-dir "$R" verify --archive artifacts/human_demo/raw/seed_20261033/demo_fog_33_cactus_snow_m200
  python3 scripts/human_demo.py --resource-dir "$R" verify --archive artifacts/human_demo/raw/seed_20261033/demo_fog_33_cattail_snow_m200
  ```
  期望每局输出 `{"verified": true, ...}`，且 `operations` / `tick` / `wave` 与台式机记录一致（39,590 / 57,596 / 41,107 / 41,831 个操作，四局均 `result=1` 通关）。把这段输出贴回“结果”即可关闭本项。注意包里发的是 Linux 静态 ELF，macOS 不能直接执行；Mac 上要么用自己编的 Mach-O 引擎（默认 `build/pvz-portable`，脚本会自动用它），要么在 Mac 的 Docker 里跑这个 ELF。请主控明确要哪一种。反过来，**Linux 侧已经能逐 tick 复现 Mac 录的四局**（`final_audit/four-demos.txt`：826 / 1309 / 892 / 922 帧观测全一致），所以剩下的只是 Mac 端的自证。
  Mac 上若只装了 Docker、没有本地构建的引擎，用下面这一条即可跑同一个 ELF —— 与台式机已验过的
  `python:3.11-slim` 断网路径是同一个镜像、同一个二进制：

  ```bash
  PKG="$HOME/pvz-logs/llmplay_pkg/stage"          # 或解压后的 llmplay_pkg 目录
  docker run --rm -v "$PWD":/w -v "$PKG":/pkg:ro -v "$PVZ_RESOURCE_DIR":/res:ro -w /w \
    python:3.11-slim bash -c 'scripts/llmplay/mac_check.sh /res /pkg/build/pvz-portable'
  ```

  期望末行 `MACCHECK OK`。包里那个 ELF 与台式机 21:20 跑过的 `stage/build/pvz-portable`
  是同一份（SHA256 `a84f10cf…`），所以这条命令换到 Mac 上跑，验的是同一串字节。
- **`state_hash` 跨进程不可复现**：同一个二进制、同一个任务，两次 reset 得到的开局哈希就不同（实测 `d3ec36d5…` / `83328bcc…` / `cefd2775…`）；`STATE_HASH` 命令与 reset 响应里的哈希也对不上。所以它不能当门槛，导出的存档也不能假装哈希一致（见结果里 export_archive 的处理：`executable_sha256` 写成自解释标记，让官方 verify 走 tick/ok/终局的路径）。**推测**是 SaveGame V4 快照里混进了地址或未初始化字段；主控 18:30 已记录、另派人查，本任务不修。

- ~~**task81c / task81d 的验收边界**~~（2026-10-26 03:10 已解决：最终包 `7be31b84…` 已在 `python:3.11-slim` 与 `ubuntu:22.04 + python3 3.10.12` 两个 `--network none` 容器里跑过 `SELFCHECK OK` + 200 步导出，两份存档回独立引擎 `verify` 通过，见“结果”03:10 条。下面保留当时的理由）：新包 `ddfe0f9f…` 的断网自测用 `unshare -rn`，**没有重跑 ubuntu:22.04 / python:3.11-slim 两个容器**。理由：引擎字节未变（`a84f10cf…`，与 21:50 那轮两个容器验的是同一份），改动只有纯 Python 与配置；新代码对 Python 3.10 兼容（`from __future__ import annotations`，没有 3.11+ 专有语法）。要按 task81 §3 的口径补跑，隔离 daemon 需重新下载静态 docker（约 86 MB）与两个镜像（约 250 MB）；主控需要时再跑。
- **L50 仍不在包里（E 层只有 L30 / L40）**：`boss_l50_orig_*` 要等 boss50 分支合入 pvz-env 才有；合入后在 `scripts/llmplay/make_llmplay_tasks.py` 的 `LAYERS['E']` 追加、重跑生成器与 `check_tasks.py`，再重打包即可。
- **E 层族名已对齐主控**：L30 用 `demo_pool_30_orig_m100`（与 task85c 在 boss50 上登记的同一个名字），L40 用 `demo_storm_40_orig_m100`；两者都是 `deck: []` + `conveyor_start: "original"`。原来的 `demo_conv_30/40_orig_m100` 已改名（种子槽不变），避免合并后出现两个同义族。L50 的 `demo_boss_50_orig_m100` 等 boss50 合入。
- **稳健性种子块**：`configs/llmplay_robustness.json` 的 83,000,000–83,000,307 在 DESIGN §14.31 登记；它与任务块 81,000,000–81,032,039 不相交（全仓 1990 份 JSON 机械核对，外来命中 0）。

## 主控追加（2026-10-25 17:30，用户审阅后修订；与上文冲突时以本节为准）
用户指出两个问题：① LLM 没有时间观念，用分钟约束它等于没约束；② “只打印简短摘要”信息不够，它上下文很长，应该给足信息，而且要让它看清某一帧场上的空间分布。

### A. 止损由工具执行，不靠模型自觉
- 所有预算由接口自己计数、写在磁盘上（会话目录里的 `budget.json`），模型无法绕过，也不需要记住：
  - **搜索预算**按“模拟的引擎 tick 总数”计（不是墙钟）。在 22.04 容器里单核实测吞吐，把“约 25 分钟的计算量”换算成 tick 数写进配置；每关一份。
  - **检查点**：每用掉 40% 预算检查一次“最远存活波”（由接口在每局结束时自动记录）。连续两个检查点没有提升 ≥1 波就判定搜索不佳。
  - 判定不佳或预算用尽后，搜索类入口（批量开局、`branch()` 超过 N 份等）对这一关**直接抛异常**，异常消息就是下一步怎么做：“本关搜索已停止，改用按波手动推进：observe → 写 1–3 波脚本 → 看结果 → 不满意就 rewind 重写”。
  - **手动段的重写次数**也由接口计数：同一段 rewind 第 9 次时拒绝执行，提示退回更早一段。
  - **单关总预算**（含手动）用尽时，接口自动保存最远进度存档，并提示换下一关。
- **墙钟由接口报**：会话第一次启动时把时间写入磁盘。之后每次输出的第一行都是状态行：`[已用 1:12 / 3:00 | 本关 L49 模式=手动 | 搜索预算 0% | 最远 23/40 波 | 本段重写 3/8]`。到 2:45 状态行改成“立即停止开新局，打包 out.zip 并写 report.md”，同时新开局的入口开始拒绝执行。
- 任务书里不要写分钟数让模型自己掐表，只写“以状态行和接口异常为准”。

### B. 观测要给足，并且按 LLM 好读的方式排版（主控定格式）
`observe()` 默认就是完整版，不另设简短版。每次输出依次包含：
1. **状态行**（见 A）。
2. **局面头**：关卡、第几波 / 总波数、当前 tick、距下一波约多少 tick、阳光、每行小推车是否还在、本关出怪类型。
3. **空间网格**：5 行 × 9 列，每格固定宽度，用短代码表示植物和血量（例如 `Mel85` = 西瓜，剩 85% 血）。僵尸按实际位置画在格子上，一格有多只就写个数，超出右边界的另画一栏“场外来袭”。图例只在会话的第一次输出和 `legend()` 里给，不必每次重复。
4. **逐行明细表**：每行列出植物（列、类型、血量 / 上限、状态：睡着 / 已醒 / 待装填 / 出土中等）和僵尸（类型、位置保留 1 位小数的列号、本体血量 + 护具血量、减速 / 冰冻、特殊状态：梯子已搭、蹦极目标格、投石车剩余弹药、巨人是否已扔小鬼）。每行最后一列给出“最前面的僵尸约多少 tick 到达第 k 列 / 小推车”。
5. **卡片**：费用、冷却剩余 tick、此刻买不买得起、能种在哪些格（数量 + 示例）。
6. **事件**：自上次 observe 以来发生了什么——种下、被吃、被偷（蹦极）、被砸（巨人 / 投石车）、击杀（按类型计数）、新波开始、小推车触发。被偷 / 被砸要写清格子和植物。
7. **`timeline()`**：每波一行的历史，包括波开始时的阳光、植物数、损失、击杀，便于回头决定 rewind 到哪一段。
- 字段名和代码用英文缩写，说明文字用中文。数值都换算成游戏概念（第几列、多少 tick 后），不要直接给像素 x 和内部计数器。
- 信息量以“人类玩家此刻在屏幕上看得到的东西”为上限：血量、位置、状态都能看到；**不给**未来的出怪表和随机数状态。
- 自测时把 L49 第 15 波的一次 `observe()` 输出贴进“结果”，主控审读后才算定稿。

### C. 截图（多模态）
- 试着让引擎在无窗口沙箱里离屏渲染：`SDL_VIDEODRIVER=offscreen` 或 `dummy`，加软件渲染；现有 `PvzEnv.screenshot()`（task66）目前只支持可见运行。
- 能跑通就在接口里加 `screenshot()`：把当前帧存成 PNG 并显示出来（ChatGPT 沙箱里的模型能看到 Python 显示的图片）。任务书里说明截图和文字互补，截图用来看整体布局，精确数值以文字为准。
- 跑不通就在“问题”里写清卡在哪，只用文字观测，不要为它拖住整个任务。
- 截图里有游戏画面，只留在用户的沙箱会话里，不进 git、不上传 HF。

### D. 一次调用多步
仍然用 Python 脚本，接口不变。补两点：
- 脚本里可以写“等到条件成立再行动”，例如 `wait_until(lambda s: s.wave >= 12)`。
- 每段脚本结束时**自动打印一次完整 observe()**，不用模型自己记得调用。

### E. 构建方式改为“源码优先、预编二进制兜底”（用户 17:35）
- 用户说 GPT Pro 沙箱里的工具挺全，所以**主路径是发源码，让它在沙箱里自己编译**。包里附上构建脚本 `build.sh`，先检查缺哪些依赖，缺了就切到无窗口最小构建。
- **逐位一致自检**：包里带上 4 局人类示范的数值存档（只有 frames / ticks / meta，不含画面）。沙箱编完后先跑 `selfcheck.py`，用新编出来的引擎重放这 4 局，必须全部 `verify` 通过才能开打。不通过就改用包里的预编二进制。
- 预编二进制仍按上文第 2 步在 `ubuntu:22.04` 容器里编，只作兜底。离线自测要把两条路径都走一遍：断网容器里从源码编译 + selfcheck，以及只用预编二进制 + selfcheck。
- **用完就删**：自测结束后删掉相关的容器、镜像和构建缓存（`docker rm`、`docker rmi ubuntu:22.04 python:3.11-slim`、`docker builder prune`），并在“结果”里贴出删除前后 `docker images` 的输出。C 盘只剩 39 GB。

### F. 先出探路包；逐位一致从门槛降为诊断（用户 17:45）
- **最优先**：先做一个“探路包”交给主控：源码 + 资源 + `build.sh` + `selfcheck.py` + 4 局示范数值存档，不含接口和任务书。用户会先用不计费的廉价模型（同一种沙箱）跑，看缺哪些库、沙箱编出来的引擎是否逐位一致。`selfcheck.py` 要把以下信息打印成一段可以直接复制回来的文本：编译器版本、glibc、CPU 型号、构建是否成功、每局 verify 结果、第一个不一致的操作序号。探路包放在 `~/pvz-logs/llmplay_pkg/probe/`，做完在“结果”写路径和 SHA256。
- **验收标准改为结果一致**：GPT Pro 交回的序列拿到台式机正式引擎上重放，所有动作都被接受且仍然通关，就算有效，然后用 `human_demo.py rebaseline` 以我们的引擎重新生成哈希。逐位一致只作诊断，不再是门槛。
- 原因：PvZ 的随机数是一条链，模拟只要差一点点，后面的出怪就会全变，序列可能整局作废。所以 selfcheck 不通过时，仍然先换预编二进制。

### G. 主控审读意见（2026-10-25 18:30）
observe 格式基本可以，探路包和正式包都收到。还要改三处，改完重新打正式包：
1. **合入 pvz-env**（`git merge origin/pvz-env`）：L46 `demo_roof_46_doom_m200`、L47 `demo_roof_47_doom_m200` 已经登记。PROMPT.md 的关卡表改成 L49 → L48 → L47 → L46 四关都可打；L45、L50 仍写“未登记，跳过”。
2. **网格里标出僵尸类型**：现在格子里只写 `z2`，看不出是什么僵尸，空间判断还得回头翻明细。改成类型短代码加只数，例如 `Gar1 Imp2`、`Bun1`。一格放不下就截断并标 `+n`。图例里补上短代码表。
3. **截图放弃**（软件 GL 成本高），PROMPT 里保留“本会话没有截图”。
- 另记一个问题，不必在本任务里修：同一二进制两次 reset 的 `state_hash` 不同，说明快照里可能混进了地址或未初始化的字段。写进“问题”，写清你观察到的现象，主控另派人查。
- 已代为删除 `~/llmplay-build/rootfs`：先确认没有挂载点残留，再用 `rm --one-file-system` 删除。**以后 chroot 用完必须先 umount `/proc`、`/sys`、`/dev`、`src` 绑定挂载**，否则递归删除会穿过 `src` 删到整块数据盘。

## 主控追加（2026-10-25 23:00，探路结果，用户用免费模型跑）
- 沙箱环境：Debian 13（trixie），内核 6.18，glibc 2.41，GCC 14.2，cmake 3.31，Python 3.13.5，CPU 是 Xeon Platinum 8573C，**5 核**，**离线**（apt 120 秒超时）。
- 源码构建缺 SDL2 头文件与库、openmpt 头文件，装不上，**从源码编译走不通**。
- **预编静态二进制（a84f10cf，与正式包 `build/pvz-portable` 是同一个文件）重放 4 局人类示范：ALL PASS**。被拒操作全部复现（6/6、88/88、33/33、38/38），逐 tick 观测核对 826 + 1309 + 892 + 922 次全部一致，终局 tick 与结果一致。开局 state_hash 2 局与存档相同、2 局不同（只作诊断，见 task82）。
- 结论：正式包直接用预编二进制即可，可以交给 GPT Pro。以后如果还要走源码路径，需要把 SDL2 / openmpt 源码放进 vendor；目前不需要。

## 主控追加（2026-10-26 00:00，task81c：工作量远远不够，重打正式包）
用户先用免费模型跑了正式包，**20 分钟就报告完成**：
- L46–L49 每关 3 个种子全部通关，共 12 份存档；
- 主控在 Mac 引擎上抽验 L49 一份，`verify` 通过；
- 它的做法是写一个带约 12 个参数的打法脚本，再做爬山搜索，个别局用 rewind 手动补救。
问题在于 `extra_seeds.json` 每个任务只有 3 个种子，L45 / L50 又未登记，任务单一下就打完了。GPT Pro 比免费模型强得多，必须保证 3 小时内绝对做不完。改法：
1. **种子**：每个任务给 **40 个**种子，取一个新的独立块（例如 81,000,000 起，每任务步长 1000）。用机械方式核对与现有全部段不相交，块的登记照 AGENTS §8 的做法。
2. **任务分层**，写进 PROMPT，按顺序推进，前一层每个任务至少拿到 5 个种子通关后才进下一层：
   - **A 层**：屋顶 L46–L49 的 doom 族（就是这次打通的）；
   - **B 层**：屋顶 L42–L44 已登记的全部卡组变体（melon / gloom / gloommelon / kernel），以及 L41；
   - **C 层**：更高倍率。在本仓库登记 L46–L49 doom 族的 **3.0 倍**和 **4.0 倍**新族（照现有 demo_roof 族的格式，族名带 `_m300` / `_m400`，DESIGN 同步登记，无头 reset 要通过）；
   - **D 层**：雾夜 L36–L39 已登记的 fog79 族，提高到 2.0 倍（同样新开 `_m200` 族）；
   - **E 层**：如果 llmplay 能跑原生传送带任务（L30 `pool_l30_orig_*`、L40 `storm_l40_orig_*`、L50 `boss_l50_orig_*`，boss50 分支合入后），就加进来。老师在这三关 0–6/16，最缺示范。如果接口不支持，写清缺什么，不阻塞前面几层。
3. **PROMPT 加两条硬话**：
   - “本会话的任务单设计为 3 小时内不可能做完。状态行显示 2:45 之前，**不允许写‘任务完成’**。”
   - “一层做完立刻进入下一层；所有层都做完，就把每个任务的种子数再翻倍。”
4. **多样性**：同一关的多个种子，**不要只用同一组参数一路通吃**。每通关一个种子就记一份参数；鼓励换打法（不同的主力植物、不同的布阵），在 report 里分开统计。
5. 重打正式包，断网自检，记录新的 SHA，拷贝的事由主控来做。`git push origin llmplay`。每个任务最多 2 个游戏进程。

## 主控追加（2026-10-26 00:20，task81d：不只要通关，还要“换个未来也能赢”）
用户担心：模型只求通关，会靠读档得来的“先知”在最后一刻救场（例如看准时机放樱桃），打出来的局面很脆。人类不知道未来，会提前堆火力、留余量。主控定的衡量办法**不是**手写的战损公式，而是胜率本身：**换一个同样合理的未来，这套打法还能不能赢**。引擎已有 `RESEED`（`PvzEnv.reseed`）：从当前状态重抽所有还没开始的波次与局内随机。
1. **接口加 `robustness()`**：在给定检查点（开局、第 10 / 20 / 30 波开始时，取存在的）分别读档、`RESEED` 成 k=8 个新未来，用模型自己的**闭环打法函数**跑到终局，返回每个检查点的胜局数。
   - 只接受闭环函数，也就是每一步按当前局面决定动作；固定的动作序列在新未来里没有意义。
   - 调用计入搜索预算。
   - 种子取包内独立的稳健性种子块，不与任务种子重叠。
2. **export_archive 记录稳健性**：meta 写入 `robustness`，内容是每个检查点的 k 局胜数，外加打法函数源码的哈希。用 rewind 手工补救过的局照样可以导出，但必须标 `manual_patch: true`。手工补救的部分没有闭环函数，稳健性按补救前的打法计算，并注明。
3. **PROMPT 改写目标**（原样写入）：
   - “这些对局会被当成教学用的中盘局面，交给另一个玩家接着打。我们要的是一个强玩家会觉得‘稳’的局面：火力和经济留有余量，不靠恰好知道下一波来什么。”
   - “一局算真正完成，需要：通关，并且 `robustness()` 在第 10 波检查点上 8 个新未来至少赢 5 个。只通关、不够稳的局仍然要交，但只算半个。”
   - “通关之后，用剩余预算提高稳健性，再进入下一个种子。尽量用不知道未来也会这样做的打法，少用 rewind 补救；补救局要在 report 里写明。”
   - 分层推进条件改为：每个任务至少 5 个种子通关，其中至少 3 个达到稳健标准。
4. 战损、丢车、被吃株数这些数，只在 report 里描述，**不进入任何目标或排序**。
5. 测试：`robustness()` 对同一打法、同一种子结果可复现；`RESEED` 只改未来波次（照 `PvzEnv.reseed` 的语义）；预算计数正确。然后重打包、自检，`git push origin llmplay`。
