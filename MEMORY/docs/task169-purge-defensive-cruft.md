# task169：彻查并删除防御性冗余（manifest、自校验、写死路径、死代码）

用户 10-30 指示：仓库里堆了大量莫名其妙的 manifest、依赖地狱、“防空气”的校验和写死的依赖，用 DeepSeek 彻查整个仓库，全部删掉。

直接起因：10-29 主控清理磁盘时删了旧 worktree `conveyor`。生产代码里有多处写死 `.claude/worktrees/conveyor/artifacts/stage4/day_panel.json`（一个没进 git 的生成文件）和 `conveyor/artifacts/task20/frames`，deploy17 和回滚都起不来，停采约半小时。

政策依据：AGENTS.md §6.5（禁止挡在关键路径上的形式主义验证）。

## 要删的四类东西

1. **manifest / 指纹 / 自校验。** 包括：
   - SHA256 / md5 记录与比对；
   - `ready.json`、`prepared.json`、`archive_manifest.json`、`*_manifest.json` 一类旁路文件；
   - 加载前的哈希核对；
   - 启动预检、重放复验、逐位对拍；
   - “以防万一”的断言和防御分支。
2. **写死路径。** 指向 `.claude/worktrees/<名字>/` 的路径，以及某台机器的绝对路径（`/home/newbiexvwu/...`、`/Users/...`）。
   - 改成仓库内相对路径或命令行 / 配方里的配置项。
   - 能在运行时现算的就现算（例如任务表由关卡号 + `deck_for_level` + 种子段生成），不要再落盘成 JSON。
3. **死代码。**
   - 一次性任务脚本、旧实验配置、旧配方；
   - 没有任何活代码引用的模块，以及只测这些东西的测试。
4. **依赖地狱。**
   - 兼容旧格式的多套分支（例如已没有检查点在用的旧导出格式）；
   - 版本兼容垫片；
   - 只为旧检查点 / 旧存档保留的路径；
   - 从来没打开过的可选开关。

**目标是简化，默认删除。** 只有能说出它防住过的具体损失（坏数据进训练、生产状态被破坏）的检查才留下，在交付里写一行理由即可。

不要新增任何校验、预检、manifest 或 JSON 中间文件，也不要为这次清理本身设计流程。只测被删检查的测试一起删。

## 什么是“活代码”

生产入口及其闭包（import 关系，加上 `subprocess` 调起的脚本）：

- `scripts/curriculum_queue.py`：采集、打包、训练、挑选、门禁全链路；
- `src/` C++ 引擎与构建（`CMakeLists.txt`）；
- 部署与运维：`scripts/qc_deploy17.sh`、`scripts/deploy17_at_boundary.sh`、`scripts/deploy17_boundary.py`，以及看门狗；
- GPU 空档执行器（task162 的脚本）；
- task105 扫描服务（`origin/llmimport` 上的脚本）；
- `scripts/hf_sync.py`；
- 以上各项的测试。

`MEMORY/` 不在删除范围（那是历史记录），但其中指向被删文件的链接要改掉或标注“已删除”。`README.md`、`DESIGN.md` 里的引用同样要改。

## 分工（两个会话并行，文件不交叉）

基线都是 `origin/deploy17`（生产现行代码）。每完成一批就 `git fetch && git rebase origin/deploy17`。台式机 codexb 正在 deploy17 上修 conveyor 那几处写死路径，那几处留给它。

- **会话 1：分支 `purgelive`，只改活代码文件的内部。**
  - 删检查、manifest 的读写、写死路径、兼容分支。
  - 每删一类先用 `git grep` 找全引用，改完同批提交。
- **会话 2：分支 `purgedead`，只整文件删除非活文件。**
  - 先自己算出活文件闭包（不用落文件），然后删闭包之外的 `scripts/`、`python/`、`configs/`、`recipes/`、`experiments/`、测试等。
  - 删前按 AGENTS §4 查引用链，删后扫断链。

## 验收（Mac，遵守 AGENTS §6.5 第 8 条）

- 每批改动后：
  - 活模块逐个 `python3 -c "import ..."`；
  - `python3 scripts/curriculum_queue.py --help`；
  - 跑与改动相关的 `pytest -k`，加 `timeout 300`、`ulimit -f 2000000`，一次一个。
- **禁止整套运行**：`test_curriculum_queue.py`、`test_curriculum_pipeline.py`、`test_qc_resilience.py`。
- 不部署，不连台式机生产。主控审过后并入下一次部署（deploy18），在组边界热更新。

## 交付

结论追加在本文件末尾：
- 删了什么（按四类，大致文件数和行数）；
- 改了哪些写死路径；
- 留下了哪些检查（一行理由）。

每批提交推送各自分支。
