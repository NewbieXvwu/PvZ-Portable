# L50 僵王博士接入

2026-10-25 完成 Boss 可见状态接入。完整观测、lean、v2 块和 C++ 学生输入只传递屏幕可见事实：是否露头、当前动作及目标、外观血量阶段，以及冰球/火球的颜色、行和位置。Boss 的内部计时器没有序列化。体力外观阶段来自 `src/Lawn/Zombie.cpp:5813-5833` 的 `<80%` 与 `<50%` 阈值；出球颜色和行来自 `src/Lawn/Zombie.cpp:10021-10068`，球的位置更新来自 `src/Lawn/Zombie.cpp:10140-10182`，阶段机来自 `src/Lawn/Zombie.cpp:10246-10480`。

旧输入版本保持原维度，新增 `v2boss`，Python 与 C++ 使用 4 个 Boss 空间通道和 2 个状态标量。`scripts/make_boss_panel.py` 登记 `boss_l50_orig_eval/train`，保留 L50 原生传送带牌表，评估种子 85,000,000–85,000,031，训练种子 85,100,000–85,108,191。

源码构建通过：`cmake --build /tmp/pvz-boss50-build -j2`。本机没有 `main.pak`、r11/branch1 模型和 `numpy`，未执行老师 8 局/32 局冷启动及完整 Boss 回归。
