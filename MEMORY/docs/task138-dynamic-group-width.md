# task138：采集组开工后还能加 worker（组宽不再在开工时定死）

主控 2026-10-28 派给 Mac CodeBuddy（cbp-task129）。分支 `groupwidth`，以 `origin/deckcells`（task137，9aa61709）为底：生产控制器在 deploy14l 换上后就等于这个版本。

## 现象

- 一组的 worker 数在开工那一刻按 MemAvailable 和后台余量算出来，之后整组不变。开工时内存临时紧张（打包、门禁评估、换候选的空档），整组就只有 1–5 个 worker；48 局的组能拖 1 小时以上。10-27 21:48 有一组 2 worker、57 分钟只打了 20 局；10-28 05:14 有一组 1 worker，主控只好杀 qc 重开。
- 余量的口径问题 task134 已修，但“开工时定死”这一点没变。

## 要做的

1. `search_play` 的 worker 池改为可在组内扩容：组开工后每隔约 30 秒重新评估可用内存，有余量就加 worker，上限是 `--workers`。不缩容，正在跑的局不杀。
2. 结果、种子分配、断点续跑合同不变：同一组的种子集合、输出目录与 `games.jsonl` 语义不变。
3. 小测试，用模拟的内存读数：开工 1 worker，内存回升后扩到 N。
4. `scripts/qc_deploy14m.sh`：只替换相关文件。未激活和回滚分支照抄台式机最新 `~/pvz-logs/qc_deploy14l.sh` 的 `start_qc`（主控改过：EXTRA_ARG 由模式文件决定；stop_qc 不排空，直接杀 qc）。部署前检查要便宜。
5. 测试只在 Mac 上跑，最多 2 个游戏进程。结论写在本文件末尾，推送分支。不碰台式机。
