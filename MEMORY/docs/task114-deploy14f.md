# task114：deploy14f = deploy14e + task99（起点格种子续期）+ task95 引擎 build4

## 现状

- **生产**：deploy14e，`origin/deploy14e` 5500e8df / 47e31d23，23:18 上线，门禁 v4，看门狗在跑。
- **task99**（startseeds 分支，78aa9156）：起点格种子块续期，加 plane_of 打包根因修复。见 startseeds 工作树里的 `MEMORY/docs/start-cell-seed-blocks.md`。
  - 未部署，有一个待主控决定的问题：7 个（核对时是 10 个）新增起点格与原格共用同一个未来种子块。
  - **主控决定**：共用块的格各自分配新块，不共用。理由是同一未来种子被两个格采到会让数据重复、窗口相关。
- **task95**（deploy15 分支，9f1db5d9）：引擎 build4（de952714，共享库链接）。
  - 六族逐位门禁、L50、状态哈希、ABBA 1.022 全部通过；含 v2boss 观测修复与 task82 哈希确定性。
  - 它的 `~/pvz-logs/qc_deploy15.sh` 按旧代码写死，不能直接用。

## 要做的（分支 deploy14f，从 origin/deploy14e 开）

1. 合入 task99 的代码改动，冲突以 deploy14e 的现行行为为准：门禁 v4、单格隔离、completed_groups、多种子。按上面的决定处理共用块。
2. 合入 task95 中与引擎相匹配的源码改动（v2boss 等），确保 Python 侧能加载 build4 引擎。
3. 写 `scripts/qc_deploy14f.sh`（并复制到 `~/pvz-logs/`）：
   - 代码覆盖：同 14e 的写法，CODE_FILES 与基线 SHA 按 cq 现状重算；
   - **同时换引擎**：把 deploy15 的 build4 拷进 cq 的新目录（例如 `build6/`），核对 SHA 后，start_qc 指向它；回滚时退回 build5；
   - 不需要排空：只要求没有排队中的门禁任务，与 14e 相同；
   - **不要执行**，由主控执行。
4. 只跑相关测试。不要重跑 task95 已经做过的引擎门禁，它的证据直接引用（AGENTS.md §6.5）。
5. 推送 deploy14f，结果写进本文件末尾，并同步到 pvz-env。

最多 2 个游戏进程。
