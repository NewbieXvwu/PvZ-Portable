# task122：老师 `--save-candidates` 接进生产课程（随 deploy14h）

主控 2026-10-27 派给 Mac cbp-task121（CodeBuddy，已熟悉 deploy14h）。分支 `rebalance2`（在 task121 基础上继续）。

## 背景

task120（分支 blindspot，`MEMORY/docs/task120-teacher-blind-spot.md`）：引擎合法动作只含当前买得起的卡，学生的消费习惯让贵卡对搜索永久不可见。`search_play --save-candidates` 让搜索评估“攒够再种”。生产卡组复测：气球 L33/34/39 老师 8/48 → 20/48（McNemar p=0.0018），L42 / L7 / 8 个蒸馏缺口关不退步；每局墙钟 +30–70%（L7 约 +5%）。

## 主控决定

- 先只在**雾夜区课程格（L31–L40，含示范起点格和 opening119 的 L32/L37 格）**打开 `--save-candidates`，其他区保持关闭。墙钟代价集中在雾夜区；下一步看雾夜区老师胜率与每小时局数再决定是否全开。
- 开关做成课程格 / 区级的可配置项（不硬编码关号判断在搜索里），记进每组日志，便于对照。

## 要做的

1. 把 blindspot 分支里 `search_play --save-candidates` 的实现合进 rebalance2（只取它需要的文件）。
2. 课程控制器按格把 `--save-candidates` 传给采集；雾夜区格打开。
3. 更新 `scripts/qc_deploy14h.sh`：CODE_FILES 加上新文件；**引擎若需要改动**说明清楚（若只是 Python 搜索侧则不用换引擎）；未激活 / 回滚分支保持 deploy14g 在役参数（主控已修过一次）。复制到台式机 `~/pvz-logs/qc_deploy14h.sh`，不执行。
4. 只跑相关测试（AGENTS §6.5）。推送，结果写在本文件末尾。
