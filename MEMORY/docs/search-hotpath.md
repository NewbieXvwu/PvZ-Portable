# task77 搜索热路径

状态：执行中，尚未完成剖析或验收；不据此部署。

## 冻结输入与复现

实验工作树 `/home/newbiexvwu/PvZ-search/.claude/worktrees/perf2`，基线代码 `c0249d62`。
运行工具为 `scripts/task77_search_hotpath.py`，命令经 `curriculum_queue.command_for`
生成，不改候选数、未来数、视界或门槛。仅本任务启动的游戏进程最多 4 个。

| 输入 | SHA256 |
|---|---|
| 生产 deploy4，引擎只读复制自 cq/build4 | `16989f23cdce3d67e862f4fb4279b423d22ec532d066eea8aa1565dbe76a1fd2` |
| 在役 round 11 学生 | `1cf82f5e6cf7994ca961d91e878d4ac65a4563eea2be50a1e96a9787c22dfd4c` |
| v5 叶值 | `8a636d337c03dcf3037f42ce41f79abc5e0b898eb5a2b39d9a00ad3b9d88c3b4` |
| 台式机 branch1 | `2485841036cf3a8738174affb0dadc0201a925ca3ea0ebe6e3eb02e74621e314` |
| 六族冻结清单 | `428760ee5be9afe289bd59c9794eae0ece7111bd57bef0c9dd95227bde8ad111` |

白天/夜间/泳池/传送带沿 task70 A2 的 m125 任务；屋顶用 L41 r7 原版；
雾用在役 `fog_user_l31_snow`（human）。human_v2 尚待主控部署，不在本任务换族。
模型、引擎副本、完整基线 Python/C++ 源码与每族前 8 个既有种子冻结在
`artifacts/task77/`，全部保持 gitignore。预检中发现 redeploy/build 已被 task70 B5
重建为另一引擎，故正式实验取生产 cq/build4 的 deploy4 副本；未修改 cq 或 build4。

系统无 perf，使用仓库 SIGPROF 采样器及 worker 内 cProfile，另记录每种协议命令
的往返耗时、引擎既有 simulation_seconds/network_seconds。采样和 cProfile
有开销，其整局耗时不作为 ABBA 性能结论。

```bash
PVZ_NO_POPUP=1 /home/newbiexvwu/.venvs/ml/bin/python3 /home/newbiexvwu/PvZ-search/.claude/worktrees/perf2/scripts/task77_search_hotpath.py prepare
PVZ_NO_POPUP=1 /home/newbiexvwu/.venvs/ml/bin/python3 /home/newbiexvwu/PvZ-search/.claude/worktrees/perf2/scripts/task77_search_hotpath.py profile
```

`prepare` 拒绝覆盖既有冻结；第二条可续跑已落盘的局。完整命令和原始剖析在
`artifacts/task77/profile/`，后续在本报告补剖析、优化与验收结果。
