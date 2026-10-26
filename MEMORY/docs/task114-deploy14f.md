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


## 结果（2026-10-27，task114）

- 在指定工作树从 origin/deploy14e/47e31d23 新建 deploy14f。核心提交 88cb4310，按 task99/78aa9156 的最终实现移入续期、检查工具与 plane_of 修复，没有合入 startseeds 的旧任务清单或无关任务；冲突保留 14e 的 v4、单格/单组隔离、completed_groups、K=2 多种子及无效示范起点退休。CLI 文案同步主控“只要求无已排队/运行门禁，不要求整轮排空”的决定。
- 续领池 [200000000,201000000)，独立加锁账本 ~/pvz-logs/start_cell_seed_ledger.json，每次256、不回绕。旧块仍为首段，offset不变；重启保留续领段，offset正好到清单末端的活跃格仍能续领。池耗尽/坏账本/IO失败只退休当前格；退休格须显式恢复。dry-run只在临时账本副本领取，真实账本与state不写。
- 共享块问题按当前状态处理：清单和生产state均有29个起点格，29×256初始种子无交叉；task99旧报告中的10个共享条目是3原格+7错误继承的派生格，后7格已经被cbd96daf删除demo_start，task113也修复生成器。没有复活无效格、没有把旧清单覆盖回来；现行LLM新增格使用46000000–47000000的各自独立块，续领也逐格独占。加入清单零共享回归，原格块/偏移不动。
- plane_of根因修复已移入：首次常量平面出现时补齐此前行的-1映射，统一平面存储dtype；旧坏段只检查npz头后按标签自动重建。引用task99既有实际84行打包证据，不重打包全量语料。
- 并入 task95/9f1db5d9 的匹配源码：本树src/与已验收build4对应deploy15源码一致，本次差异仅EnvironmentRollout.cpp三处v2boss通道修复；Python pvz_env.py原已相同。build4归档SHA已确认 de952714651c6544f4dba16f8ae832fe1ad1cb75bfc3e1d5495964d3d9a83845。直接引用 MEMORY/docs/task95-engine-gate.json 与 task95-throughput.json：六族96对、L50四对、状态哈希、16对ABBA中位1.021813全部通过；本次未重跑引擎门禁、未重编引擎。
- scripts/qc_deploy14f.sh固定代码REV=88cb4310，8个CODE_FILES及基线SHA按cq当前文件生成。仅准备：把deploy15已归档build4复制到cq/build6/pvz-portable并核对SHA，start_qc切换它；build5保留。无v3续采轮、无整轮排空、无回填；已排队/运行门禁时拒绝切换。
- 脚本加入持久引擎选择 ~/pvz-logs/qc_deploy14f.engine：未激活的草稿即使被现行看门狗选为最新脚本，也仍启动build5；激活后启动build6；回滚记录build5，后续看门狗不会拉回新引擎。新采集前失败可恢复代码；新采集已开始后的引擎回滚保留续期兼容代码，避免旧代码截断已续领种子表。state、offset、任何账本都不回滚。
- 仅跑相关测试，覆盖续领/重启/坏账本/独立分块、plane_of与增量打包、v4门禁与证书、多种子/pipeline、看门狗及模拟部署/回滚；修复合并产生的测试接口差异后，失败用例定向复跑通过。两项既有跳过分别依赖未入git用户示范档和本机原始对局档。部署脚本bash -n和看门狗启动函数提取通过；脚本已复制到 ~/pvz-logs/qc_deploy14f.sh，未执行。
- 本任务没有启动游戏或训练，没有修改生产state/offset/账本/teacher_rates，没有操作qc或看门狗会话。代码、部署包与本结果推送deploy14f，并合并同步pvz-env（保留其记分板等独立改动）。结果同时保存在 MEMORY/docs/task114-deploy14f.md。
