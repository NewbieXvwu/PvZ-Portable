# 原生传送带 L25 / L30 / L40（task75）

## 任务登记

运行 `scripts/make_native_conveyor_panel.py` 生成 `artifacts/task75/native_conveyor_panel.json`。L25/L30 的任务为 `pool_l25_orig_eval/train`、`pool_l30_orig_eval/train`，L40 为 `storm_l40_orig_eval/train`。三关全部使用十空槽原生开局，后续发牌由游戏决定。旧泳池 p9k 初始牌任务与所有旧面板保留。

训练种子分别为 60000000–60008191、60100000–60108191、60200000–60208191，评估分别为 61000000–61000031、61100000–61100031。2026-10-25 机械扫描主目录与全部工作树的 artifacts/configs 清单，共 53889 个 JSON、748 项含种子任务、968772 个现有唯一种子，新增 24672 个种子与它们交集为零。包含现有倍率清单。扫描明细在本机 `artifacts/task75/seed_audit.json`。

## 观测审查（实现中）

L25 引擎把僵尸缩放设为 0.5，并把各部分血量除以四。完整观测有当前与最大血量，精简帧丢弃最大血量，v2 和 C++ 学生仅有当前血量，受伤的普通僵尸与小僵尸缺乏明确体型区分，需补充体型字段。

L30 原生开局是空的十槽传送带，p9k 变体由 reset 写入九张初始牌并把容量改成九槽。两者之后仍按游戏发牌，免费种植，容量差异会影响满带与补卡节奏。

L40 是 StormyNight，`StageHasFog=false`。既有 human/human_v2 均未处理暴风雨黑幕，新任务另用 `fog_observation: storm`。观测实现、测试与96局老师冷启动结果待续。
