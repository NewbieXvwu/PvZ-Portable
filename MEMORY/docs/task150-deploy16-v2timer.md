# task150：deploy16 准备：prodint + v2timer + 系统性失败暂停

主控 2026-10-29 定。在台式机做。

## 合入

新建集成分支 `prodint16`，以 `origin/prodint`（生产 deploy14p = `4beab441`）为底，依次合入：

- `origin/obstimer`：obsup2（v2sleep + build7）+ task145 v2timer + task147 C++ 第 9 号输入版本与缓存提速；
- `origin/sysfail`：task149，游戏进程全线崩溃时暂停采集。

有冲突就按两边语义解决，在结论里写清楚。

## 部署脚本

1. 写 `~/pvz-logs/qc_deploy16.sh`，参照 `qc_deploy14p.sh` 的结构：
   - `start_qc` 与看门狗提取兼容，不要 `local` 之类会破坏提取的写法；
   - `kill` 后接 `|| true`；
   - 新增模式文件 `~/pvz-logs/qc_deploy16.mode` = `v2timer`，选择 v2timer 配方与新二进制。
   - 配方用 task145 的 `recipes/dw4i2_rebalanced_p05_v2timer.json` 思路，A/B 两套都要 v2timer 版本。
   - 现行 `--extra-manifest`（deckcells）照旧保留。
2. 新二进制用独立构建目录，不覆盖现役二进制。
3. **缓存预构建**：在新二进制与新配方下，把现有训练数据的 v2timer 缓存用复用分支预先建好，用 `nice -n 19`、2 个 worker，内存不抢生产（先看 `free -g`，留够 qc 的余量）。这一步可以慢，但必须后台跑，不阻塞别的事。
4. 冒烟：在集成工作树里，用新二进制跑 1 局 v2timer 学生对局，能走完即可。只跑与改动相关的 `-k` 测试，限内存。

## 约束

- **不执行部署**，不碰 tmux qc、状态文件、看门狗、现役二进制。部署由主控在组边界执行。
- 不加自校验或哈希检查。
- 结论写在本文件末尾：合入了哪些提交、部署脚本路径、缓存预构建的进度和预计完成时间。推送 `prodint16`。
