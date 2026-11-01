# task218：人类示范进入干净训练

分支 humanrows，基于生产 arestart `448394fa`。clean_a / clean_b 的 `source_weight` 均增加 `human=1.0`，未知来源仍为 0，两份配方均已从工作树绝对路径调用 `load_recipe` 读取通过。

`human_demo.py export-batch` 只读取已 verified 且来源为 human 的档案，逐份调用原有 `export`，写入 `frames_v2/human/<任务标签>/seed_<种子>/<任务标签>/frames.npz`，已存在的任务和种子跳过。现有 `batch_sources` 同时扫描 mac197 和 human，复用控制器已有的来源合并、标签入包与待打包分片计算，控制器无需另写人类接入路径。不重放，不启动游戏，不新增清单。

相关小测试通过：真实导出小样本后调用控制器 enqueue 生成 A/B recipe，标签均包含人类任务，来源权重均为 1，未知老师保持 0，并能软链进入训练视图。两份生产配方读取通过，未训练或接触台式机生产。

本机实际找到 20 份 verified 人类存档：主仓库早期 demo_* 15 份，userpick 5 份（包含所述 L28/L29 三份，以及 L22/L49）。数值原档 23,401,759 字节，已上传 `realnewbiexvwu/pvz-agent-artifacts` 的 `transfer/task218/human_raw.tar.gz`，Mac 临时包已删。台式机取用时复用现有 HF 接收器，下载完成立即删除 HF 包，再由原有 export 写入生产帧目录。

主控在组边界执行以下命令。只安装两份配方和两个脚本，不改课程状态，之后由主控正常重启控制器一次载入，新的训练作业会纳入人类标签与权重。已生成的历史 recipe 不修改。

```bash
R=$HOME/PvZ-search
W=$R/.claude/worktrees/cq
PY=$HOME/.venvs/ml/bin/python
git -C "$R" fetch origin humanrows
for f in recipes/dw4i2_v2timer_clean_a.json recipes/dw4i2_v2timer_clean_b.json scripts/human_demo.py scripts/mac_collect_import.py; do
  git -C "$R" show "origin/humanrows:$f" > "$W/$f.task218"
  mv "$W/$f.task218" "$W/$f"
done
"$PY" "$W/scripts/mac_collect_import.py" transfer/task218/human_raw.tar.gz --repo realnewbiexvwu/pvz-agent-artifacts --frames "$HOME/pvz-logs/task218"
"$PY" "$W/scripts/human_demo.py" export-batch --archives "$HOME/pvz-logs/task218/mac197/human_raw" "$W/artifacts/human_demo/raw" "$R/.claude/worktrees/llmimport/artifacts/human_demo/raw" --out "$R/artifacts/student/frames_v2"
rm -rf "$HOME/pvz-logs/task218/mac197/human_raw"
```

后续新的人类录制继续用最后一条 export-batch 命令导入相应原档根目录即可，同标签沿用现有批次机制，新建训练作业会自动纳入。导出读取已验证的数值帧，不需要再 verify 或重放。

## task218b：训练种子 int32 范围

humanrows 原基线未包含 userpick 的随机录制入口，本次合入已有 task209 实现，`record --level` 随机种子改用 31 位。`export` 仅在训练副本里将 ≥2^31 的种子映射为 `seed & 0x7fffffff`，目录与 `task.seed` 一致，并记录 `task.original_seed`。批量导出用同一映射寻找现有副本，能跳过主控已经修正的副本。原档及其重放种子不改。相关 5 项小测试通过，覆盖 2^31、3174639343 和最大 uint32，未启动游戏或窗口，台式机生产未动。
