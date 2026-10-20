# 运维值守记录（规则见 ops-handbook.md；只追加）

- 台式机 13:23（Mac 01:23）｜首轮值守。queue86 已 DONE；d35balemas1 z15–z20: 953 953 906 787 686 514（均 800）；d35balemas2: 941 907 837 681 551 416（均 722）
- 台式机 13:23（Mac 01:23）｜queue87：z17dag40 (v32, k=2, beta 0): 335；queue88：control d35bals1 z20 rowpool 523（教师 523）；queue89：built d88818f
- 台式机 13:23（Mac 01:23）｜GPU 45% / 3113 MiB；C 盘剩 28 GB；q87/q88/q89 均在跑
- 台式机 13:23（Mac 01:23）｜Mac z17_dagger40b 完成：569/640 胜；z17_dagger41b 进行中 310/640（2856s），search_play 进程 1 个
- 台式机 13:23（Mac 01:23）｜codex-cuda 停在空输入框，状态行 Goal hit usage limits（额度用完，非异常）；cb-cuda 于 13:16 新建
- 台式机 13:54（Mac 01:54）｜无新结果行。q87/q88/q89 会话均在，z17_dagger41.log 13:54 仍在写（q87 存活）；GPU 44% / 3112 MiB；C 盘剩 28 GB
- 台式机 13:54（Mac 01:54）｜Mac z17_dagger41b 进度 510/640（4680s），search_play 进程 1 个；无异常
- 台式机 15:17（Mac 03:17）｜第三轮。上轮遇 429 中断，本轮比计划晚约 23 分钟（Mac 02:24 → 03:17），已恢复正常节奏
- 台式机 15:17（Mac 03:17）｜queue87 已 DONE（14:57），新增 z17dag39b / z17dag40b extracted: 640 episodes in frames_v2；q87 会话退出属正常完成
- 台式机 15:17（Mac 03:17）｜需关注（未达手册异常条件）：q88 日志停在 12:46、q89 停在 13:13，两会话仍在但 2 小时无新行、窗格空白；当前仅 1 个 train_rollout_policy_dilution 进程（96% CPU，已跑 8797s）
- 台式机 15:17（Mac 03:17）｜GPU 53% / 3113 MiB；C 盘剩 29 GB；MemAvailable 11.0 GB；codex-cuda 仍为额度用完（非异常）
- 台式机 15:17（Mac 03:17）｜Mac z17_dagger41b 完成：580/640 胜（40b 为 569/640）；z17_dagger42b 进行中 330/640（3253s）；mac_dagger 进程 2 个，未空闲
