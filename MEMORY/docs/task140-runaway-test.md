# task140：修掉 `test_persistent_state_error_*` 的无限写日志 / 内存暴涨

主控 2026-10-28 定。分支 `fixtest`，以生产控制器所在分支 `origin/deckcells` 为底。

## 事故

- 10-27 12:09，台式机：`pytest python/test_qc_resilience.py` 涨到 17 GB，生产 qc 被 OOM。
- 10-28 08:40 前后，Mac：两个 pytest 进程同时整套跑 `test_curriculum_queue.py`、`test_curriculum_pipeline.py`、`test_qc_resilience.py`。`pytest-150/151/152/test_persistent_state_error_bl0/curriculum.log` 分别写到 9.3、8.3、2.0 GB，Mac 磁盘写满、内存耗尽，系统冻结所有应用。

## 要做的

1. 找出 `test_persistent_state_error_*` 为什么不停写日志：测试本身的模拟状态错误是否导致控制器在重试循环里每轮写一行、永不退出？这条路径会不会在**生产**中出现（状态文件持久化失败时日志无限增长）？
2. 修复：
   - 测试必须有界，加迭代 / 时间上限，失败要快速失败；
   - 若生产代码在持久化错误时会无限重试写日志，改成退避 + 限次，最后明确报错退出，交给看门狗。
3. 在 `python/conftest.py`（没有就新建）给这三个文件加一个会话级保护：单测试超时（例如 120 秒）和 tmp 目录写入上限。实现要轻，不能拖慢正常测试（AGENTS §6.5）。
4. 只用 `-k` 跑这一个用例来验证，加 `timeout 300` 和 `ulimit -f 2000000`。遵守 AGENTS §6.5 第 8 条。
5. 生产代码若有改动，写 `scripts/qc_deploy14n.sh`（只替换相关文件，start_qc 照抄台式机最新部署脚本），不执行。结论写在本文件末尾，推送分支。

---

## 结论（2026-10-28，分支 `fixtest`，基于 `origin/deckcells` 6fa61c45）

### 1. 为什么不停写日志（已复现）

链路只有三环，缺一不可：

1. `python/test_qc_resilience.py` 的 `args()` 是手搭的 `SimpleNamespace`，**缺 `save_candidates_levels` 和 `random_deck_share`**
   —— 这两个参数是 task120 / task137 之后才进 `collect_rounds` → `plan_round` 的，fixture 没跟上。
2. `collect_rounds`（`scripts/curriculum_queue.py:2039`）把参数访问放在 `try` 里，
   `except (Exception, SystemExit)` **一律当成可重试**，写一行日志 + `time.sleep(120)` 后从头再来。
3. 测试里 `patch.object(q.time, 'sleep')` 把退避抹成 0 → 循环变成毫秒级死转，
   每转一圈往 `tmp_path/curriculum.log` 写一行
   `ERROR round boundary paused; retry in 120s: AttributeError: ...`。

实测（本机单进程，`-k test_persistent_state_error`）：**几十秒写出 977 MB**，被 `ulimit -f 2000000` 与 `timeout 300` 拦下。
事故里三个 pytest 并发，于是 9.3 / 8.3 / 2.0 GB。

最阴的一点：用例最后那句 `assert events == [...]` **永远执行不到**，所以它不报失败也不退出，
外面看是"测试还在跑"，实际在灌盘。内存暴涨是伴随现象（日志页缓存 + 每轮新建字符串 + pytest 捕获缓冲），主因是磁盘。

### 2. 生产里会不会出现

会，但形态不同：

- 生产 `time.sleep(120)` 是真的，速率被限住（约 720 行/天），**不会灌盘**；
- 但它是**无限重试、永不退出**：状态写失败（`save_state` → `StateWriteError`，`retry_io` 只重试一次就抛）、
  清单损坏、代码/参数不匹配，都会让控制器每 120 s 重来一次，永远不放弃。
  进程还活着 → 看门狗不会重启 → 采集实际停摆，日志里只剩一行行 `ERROR round boundary paused`。
- 结论：**测试炸盘的直接诱因是 sleep 被 patch，但根因（无限重试、把代码错误当临时故障）在生产里同样存在。**

### 3. 改动

| 提交 | 内容 |
|---|---|
| `84721586` | `scripts/curriculum_queue.py`：轮边界重试改为有界 + 指数退避（120/240/480/960/1800 s），连续失败 5 次后写 FATAL 并 `raise` → `main()` 退出非零 → 交给看门狗；`ROUND_FATAL_ERRORS`（AttributeError / TypeError / NameError / KeyError / AssertionError）**立即抛出、不再重试**；每完成一轮重置计数 |
| `760d0808` | `python/test_qc_resilience.py`：`args()` 补齐 `save_candidates_levels` / `random_deck_share`；新增两个用例把契约钉死——重试 5 次后抛错且日志只有 6 行、代码错误不重试且 `sleep` 一次都不调 |
| `f12cb00b` | `python/conftest.py`（新建）：只对 `test_curriculum_queue.py` / `test_curriculum_pipeline.py` / `test_qc_resilience.py` 生效的会话级护栏——单测试 120 s 超时（`PVZ_TEST_TIMEOUT` 可覆盖，SIGALRM 抛 `TimeoutError`）、单文件写入 256 MB 上限（`RLIMIT_FSIZE`，子进程继承）、测试后 tmp 目录 512 MB 检查。正常用例的代价是每测试两次系统调用 |
| `90464a80` | `scripts/qc_deploy14n.sh`：只替换 `scripts/curriculum_queue.py`，`start_qc` 与 deploy14l **逐字一致**（`diff` 已核），保留 deploy14l 打开的 `--extra-manifest` 随机卡组面板，带 `--rollback`；**未执行** |

设计取舍：只把"重试也不会好"的错误列为致命。`OSError` / `StateWriteError` 这类可能自己恢复的仍然重试，
只是有了上限和退避。放弃后进程退出，状态文件里保留着上一轮，重启走既有恢复路径。

### 4. 验证（遵守 AGENTS §6.5 第 8 条：只 `-k` 单跑、`timeout 300`、`ulimit -f 2000000`、同时只开一个 pytest）

- 修复前：`-k test_persistent_state_error` → 死转，写出 977 MB 后被限位杀掉（复现事故）。
- 修复后同一命令：`4 passed, 31 deselected in 0.03s`，tmp 12 KB
  （`-k 'persistent_state_error or round_boundary or keyboard_interrupt'`，覆盖这次改的 `collect_rounds` 三条路径）。
- 护栏本身单独验过：临时注入 `sleep(30)` 探针 + 读 `RLIMIT_FSIZE` 探针，`PVZ_TEST_TIMEOUT=3` 时 3.05 s 抛
  `TimeoutError`，rlimit 为 256 MB 且测试后恢复。探针已 `git checkout` 删除。
- **没有**整套跑这三个文件（§6.5 第 8 条），**没有**执行 deploy14n。

### 5. 还没做的

- deploy14n 需要控制器主人在台式机执行（作者不执行）。
- 同样的"无限重试 + 只写日志"写法若还散在别处，应逐个看；本次只处理轮边界这一条。
- `test_qc_resilience.py` 里其他手搭 `args()` 没有逐个复核——本次只补了 `collect_rounds` 用到的字段。
  以后再加参数，这条 fixture 还会失配；真正的护栏是新的 FATAL 分类（立即失败，而不是死转）。
