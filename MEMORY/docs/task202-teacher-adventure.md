# task202：生产老师的冒险模式通关范围

正在后台评测，尚未形成胜率结论。分支 `teacheradv`，采集源码为 deploy25 `97994745`，使用 Mac clang PGO 引擎、10路任务并行，每路1个游戏worker。

冒险1–50，原难度1.0倍，沿用北极星 `playthrough=2` 开局。标准卡组来自 `deck_for_level`，生产/门禁卡组直接调用北极星 `control_deck_for_level`；传送带关只跑一套原生卡池。每关两列使用相同8个种子：`932000000 + (关号-1)*16 + [0,8)`，共92组任务、736局。

雾/屋顶/Boss由GPT行为克隆模型续走与提供先验，其余地形由r52；叶值v5、屋顶/Boss叶值v5roof，静态网branch1。`--verify-samples 9 --student-candidates 2 --dagger-beta 1`，沿用生产31–40关的 `--save-candidates 1` 及生产雾观测（human_v2；40关storm），其余搜索参数和3000决策上限保持 deploy25 默认。无预种、示范起点或波数截断。

模型从台式机指定路径经SSH取用，运行缓存集中在 `~/pvz-task202/`，不导出训练帧。全部完成后在此写入逐关胜率表，标出8局全未通关的卡组及两列均全败的关，提交推送后清理缓存并发系统通知。
